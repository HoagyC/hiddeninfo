import dataclasses
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import train
from classes import Experiment
from classes import TrainResult
from utils import get_train_result_path
from utils import load_train_result
from utils import save_train_result

# With binary data and zero info, ideal prediction is always 0.5
ZERO_INFO_LOSS = 0.5**2


def display_experiments(
    *experiments_iterable: Experiment,
) -> Optional[List[TrainResult]]:
    experiments: List[Experiment] = list(experiments_iterable)
    tags = [experiment.tag for experiment in experiments]
    assert len(tags) == len(set(tags)), f"Found duplicate tags: {tags}"
    if not st.checkbox(f"Run?", value=False, key=str((tags, "run"))):
        return None
    force_retrain_models = st.checkbox(
        "Force retrain models?", value=False, key=str((tags, "retrain"))
    )

    train_results = _run_experiments(experiments, force_retrain_models)
    _plot_results(train_results)
    return train_results


def _plot_results(train_results: List[TrainResult]) -> None:
    df = pd.DataFrame(
        dataclasses.asdict(result)
        for train_result in train_results
        for result in train_result.step_results
    )

    losses = [
        "representation_loss",
        "reconstruction_loss_p1",
        "reconstruction_loss_p2",
    ]
    fig, axs = plt.subplots(1, len(losses), figsize=(5 * len(losses), 5))
    for loss_name, ax in zip(losses, axs):
        sns.lineplot(data=df, x="step", y=loss_name, hue="tag", ax=ax)
        if "reconstruction" in loss_name:
            sns.lineplot(
                x=[0, max(df.step)],
                y=[ZERO_INFO_LOSS, ZERO_INFO_LOSS],
                label="zero info loss",
                linestyle="dashed",
                ax=ax,
            )
        ax.set_title(loss_name)
        ax.set_yscale("linear")
        ax.set_ylim(([0, 0.4]))
    fig.tight_layout()
    st.pyplot(fig)


def _run_experiments(
    experiments: List[Experiment], force_retrain_models: bool
) -> List[TrainResult]:
    st.write(pd.DataFrame(dataclasses.asdict(experiment) for experiment in experiments))
    bar = st.progress(0.0)
    train_results: List[TrainResult] = []
    for i, experiment in enumerate(experiments):
        if force_retrain_models or get_train_result_path(experiment.tag) is None:
            train_result = train.train(experiment=experiment)
            save_train_result(train_result)
        else:
            train_result = load_train_result(experiment.tag)
        train_results.append(train_result)
        bar.progress((i + 1) / len(experiments))
    return train_results
