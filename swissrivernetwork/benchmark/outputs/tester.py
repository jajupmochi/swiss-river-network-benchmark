"""
tester



@Author: linlin
@Date: Nov 06 2025
"""
from pathlib import Path

CUR_ABS_DIR = Path(__file__).resolve().parent
PROJ_DIR = (CUR_ABS_DIR / '../../../').resolve()
OUTPUT_DIR = (PROJ_DIR / 'swissrivernetwork/benchmark/outputs/ray_results/').resolve()


def trim_checkpoint_tester():
    from swissrivernetwork.benchmark.util import trim_checkpoints

    # path_name = 'stgnn-zurich-wl90-none-2025-10-22_16-22-45'  # the case with only one experiment
    path_name = 'lstm-zurich-wl90-none-2025-10-31_18-55-45'  # the case with multiple experiments
    trim_checkpoints(
        OUTPUT_DIR / path_name, keep_best_n=10, anchor_metric='validation_mse', mode='min', if_trim_best_n=True,
        keep_best_for_trimmed_trials=True, keep_last_for_trimmed_trials=False, verbose=True
    )


if __name__ == '__main__':
    trim_checkpoint_tester()
