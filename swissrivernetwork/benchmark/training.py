import sys
import tempfile

import torch
import torch.nn as nn
import wandb
from ray.air import session
from ray.tune import Checkpoint, report
from torchinfo import summary
from tqdm import tqdm

from swissrivernetwork.benchmark.util import save

ISSUE_TAG = "\033[91m[issue]\033[0m "  # Red
INFO_TAG = "\033[94m[info]\033[0m "  # Blue
SUCCESS_TAG = "\033[92m[success]\033[0m "  # Green


def training_loop(
        config, dataloader_train, dataloader_valid, model, n_valid, use_embedding, edges=None,
        wandb_project: str | None = 'swissrivernetwork',
        verbose: int = 2
):
    if verbose >= 2:
        # Print data info:
        print(f'{INFO_TAG}Training sample size: {len(dataloader_train.dataset)}.')
        for ds in dataloader_train.dataset.datasets:
            print(f'  - Station {ds.embedding_idx}: {len(ds)} samples, sequence lengths: {ds.sequence_lengths}')
        print(f'{INFO_TAG}Validation samples size: {len(dataloader_valid.dataset)}.')
        for ds in dataloader_valid.dataset.datasets:
            print(f'  - Station {ds.embedding_idx}: {len(ds)} samples, sequence lengths: {ds.sequence_lengths}')

        # Print model summary:
        print(f'{INFO_TAG}Model Summary:')
        summary(model)
        print(f'{INFO_TAG}GPU available: {torch.cuda.is_available()}.')
        print(f'{INFO_TAG}Using device: {next(model.parameters()).device}.\n')

    # Login via command line: `wandb login <your_api_key>`
    if wandb_project:
        name = (
            session.get_trial_id() if session.get_trial_id() else f'{config["graph_name"]}_{model.__class__.__name__}')
        wandb.init(
            project=wandb_project,
            name=name,
            config=config,  # save hyperparameters
            mode='disabled' if config.get('dev_run', False) else None,
            # finish_previous=True  # each Ray Tune trial should create a separate wandb run automatically
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model.to(device)
        # Run the Training loop on the Model
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.MSELoss()
        validation_criterion = nn.MSELoss(reduction='sum')  # weight all samples equally

        for epoch in range(config['epochs']):
            model.train()
            losses = []

            # halo = EpochHalo(
            #     total_steps=len(dataloader_train), epoch=epoch + 1, total_epochs=config['epochs'],
            #     spinner_style='dots', prefix='Training',
            #     start_color=(255, 165, 0), end_color=(0, 0, 255)
            # )

            if verbose >= 2:
                iterator = tqdm(
                    dataloader_train, desc=f'Epoch {epoch + 1}/{config["epochs"]}', file=sys.stdout, colour='green'
                )
            else:
                iterator = dataloader_train

            for step, (_, e, x, y) in enumerate(iterator):
                e, x, y = e.to(device), x.to(device), y.to(device)
                optimizer.zero_grad()
                if edges is not None:
                    out = model(x, edges)
                elif use_embedding:
                    out = model(e, x)
                else:
                    out = model(x)
                mask = ~torch.isnan(y)  # mask NaNs
                loss = criterion(out[mask], y[mask])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if verbose >= 2:
                    iterator.set_postfix(loss=loss.item())
                # halo.update(step, loss.item())

            train_loss = sum(losses) / len(losses)

            # halo.succeed(f'Epoch {epoch + 1}: Avg Loss = {sum(losses) / len(losses):.4f}.')
            print(f'Epoch {epoch + 1}: Avg Loss = {train_loss:.4f}.')

            model.eval()
            validation_mse = 0
            with torch.no_grad():
                for _, e, x, y in dataloader_valid:
                    e, x, y = e.to(device), x.to(device), y.to(device)
                    if edges is not None:
                        out = model(x, edges)
                    elif use_embedding:
                        out = model(e, x)
                    else:
                        out = model(x)
                    mask = ~torch.isnan(y)  # mask NaNs
                    loss = validation_criterion(out[mask], y[mask])
                    validation_mse += loss.item()
            validation_mse /= n_valid  # normalize by dataset length

            # Register Ray Checkpoint
            checkpoint_dir = tempfile.mkdtemp()
            save(model.state_dict(), checkpoint_dir, f'lstm_epoch_{epoch + 1}.pth')
            # save(normalizer_at, checkpoint_dir, 'normalizer_at.pth')
            # save(normalizer_wt, checkpoint_dir, 'normalizer_wt.pth')
            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            # report epoch loss
            report({"validation_mse": validation_mse}, checkpoint=checkpoint)
            print(f'End of Epoch {epoch + 1}: {validation_mse:.5f}')

            wandb.log({'epoch': epoch + 1, 'train_loss': train_loss})
            wandb.log({'epoch': epoch + 1, 'valid_loss': validation_mse})
            wandb.log({'epoch': epoch + 1, 'validation_mse': validation_mse})

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            report(done=True, status="OOM")
        else:
            raise

    if wandb_project:
        wandb.finish()
