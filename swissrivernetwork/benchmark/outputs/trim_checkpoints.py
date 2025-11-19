"""
tester



@Author: linlin
@Date: Nov 06 2025
"""
from pathlib import Path

CUR_ABS_DIR = Path(__file__).resolve().parent
PROJ_DIR = (CUR_ABS_DIR / '../../../').resolve()
OUTPUT_DIR = (PROJ_DIR / 'swissrivernetwork/benchmark/outputs/ray_results/').resolve()


def trim_all_experiments():
    from swissrivernetwork.benchmark.util import trim_checkpoints

    for path_name in OUTPUT_DIR.iterdir():
        if not path_name.is_dir() or path_name.name == 'done':
            continue

        print('\n***********************')
        print(f'Trimming checkpoints in {path_name}...')
        trim_checkpoints(
            OUTPUT_DIR / path_name.name, keep_best_n=5, anchor_metric='validation_mse', mode='min', if_trim_best_n=True,
            keep_best_for_trimmed_trials=True, keep_last_for_trimmed_trials=False, remove_seperated_marker_files=True,
            verbose=True
        )
        print('Done.')
        print('***********************')


def trim_checkpoint():
    from swissrivernetwork.benchmark.util import trim_checkpoints

    # path_name = 'stgnn-zurich-wl90-none-2025-10-22_16-22-45'  # the case with only one experiment
    # path_name = 'lstm-zurich-wl90-none-2025-10-31_18-55-45'  # the case with multiple experiments
    path_name = 'lstm-zurich-fs7-wl90-none-2025-11-14_23-31-33'
    trim_checkpoints(
        OUTPUT_DIR / path_name, keep_best_n=10, anchor_metric='validation_mse', mode='min', if_trim_best_n=True,
        keep_best_for_trimmed_trials=True, keep_last_for_trimmed_trials=False, remove_seperated_marker_files=True,
        verbose=True
    )


if __name__ == '__main__':
    # trim_checkpoint()
    trim_all_experiments()
