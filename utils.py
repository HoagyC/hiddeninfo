import pickle
from datetime import datetime
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np

from classes import Loss
from classes import TrainResult

DATETIME_FMT = "%Y%m%d-%H%M%S"
RESULTS_DIR = Path("out/results")


def _save_train_result(train_result: TrainResult):
    now_str = datetime.now().strftime(DATETIME_FMT)
    train_result_path = RESULTS_DIR / train_result.tag / now_str / "train-result.pickle"
    if not train_result_path.parent.is_dir():
        train_result_path.parent.mkdir(parents=True)
    with train_result_path.open("wb") as f:
        print(train_result)
        pickle.dump(train_result, f)


def _load_train_result(tag: str) -> TrainResult:
    train_result_path = _get_train_result_path(tag)
    assert train_result_path is not None, f"No train results for {tag}"
    with train_result_path.open("rb") as f:
        return pickle.load(f)


def _get_train_result_path(tag: str) -> Optional[Path]:
    results_dir = RESULTS_DIR / tag
    if not results_dir.is_dir():
        return None
    time_strs = [time_dir.name for time_dir in results_dir.iterdir()]
    times = [datetime.strptime(time_str, DATETIME_FMT) for time_str in time_strs]
    latest_time_str = max(times).strftime(DATETIME_FMT)
    return RESULTS_DIR / tag / latest_time_str / "train-result.pickle"


def _get_average_loss(losses: List[Loss]) -> Loss:
    if len(losses) == 1:
        return losses[0]
    else:
        return Loss(
            float(np.mean([l.total_loss for l in losses])),
            float(np.mean([l.reconstruction_loss for l in losses])),
            float(np.mean([l.representation_loss for l in losses])),
            float(np.mean([l.reconstruction_loss_p1 for l in losses])),
            float(np.mean([l.reconstruction_loss_p2 for l in losses])),
        )
