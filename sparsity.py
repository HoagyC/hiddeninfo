import dataclasses
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from classes import Experiment
from classes import TrainResult
from main import _run_experiments

base = Experiment(
    tag="base",
    n_models=8,
    activation_fn="sigmoid",
    use_class=False,
    num_batches=2_000,
    latent_size=10,
    hidden_size=80,
    n_hidden_layers=1,
    # Ideally, we want the representation loss to be as high as possible, conditioned on the
    # autoencoders still exhibiting the "hidden info" behaviour.
    # TODO: Experiment with making this number larger.
    representation_loss=1,
)


def _get_output_vals(train_result: TrainResult) -> Dict:
    last_step = max(step_result.step for step_result in train_result.step_results)
    reconstruction_loss_p2 = np.mean(
        [
            step_result.reconstruction_loss_p2
            for step_result in train_result.step_results
            if step_result.step >= last_step * 0.9
        ]
    )
    reconstruction_loss_p1 = np.mean(
        [
            step_result.reconstruction_loss_p1
            for step_result in train_result.step_results
            if step_result.step >= last_step * 0.9
        ]
    )
    output = dict(
        tag=train_result.tag,
        reconstruction_loss_p1=reconstruction_loss_p1,
        reconstruction_loss_p2=reconstruction_loss_p2,
    )

    return output


def run_independent_sparse() -> None:
    results = []
    out_loc = Path("out/multi_seq_results.pkl")
    for quadrant_sparsity in range(1, 6):
        with open(out_loc, "rb") as f:
            results = pickle.load(f)
        st.header(f"running sparsity {quadrant_sparsity}")
        ind_sparse_decoder = dataclasses.replace(
            base,
            tag="seq_sparse_dec" + str(quadrant_sparsity),
            loss_quadrants="bin_sum",
            quadrant_threshold=quadrant_sparsity,
            give_full_info=True,
            num_batches=10000,
        )

        ind_sparse_encoder = dataclasses.replace(
            base,
            tag="seq_sparse_enc" + str(quadrant_sparsity),
            loss_quadrants="bin_sum",
            quadrant_threshold=quadrant_sparsity,
            reconstruction_loss_scale=0,
            num_batches=10000,
        )
        ind_sparse_test = dataclasses.replace(
            base,
            tag="sequential_test" + str(quadrant_sparsity),
            load_decoders_from_tag=seq_sparse_decoder.tag,
            load_encoders_from_tag=seq_sparse_encoder.tag,
            num_batches=2000,
        )
        results += _run_experiments(
            seq_sparse_decoder, seq_sparse_encoder, seq_sparse_test
        )
        with open(out_loc, "wb") as f:
            pickle.dump(results, f)


def run_distill_sparse() -> None:
    results = []
    out_loc = Path("out/multi_distill_results.pkl")
    for quadrant_sparsity in range(1, 6):
        sparse_general = dataclasses.replace(
            base,
            tag="sparse_general",
            loss_quadrants="bin_sum",
            quadrant_threshold=quadrant_sparsity,
            num_batches=30_000,
        )
        sparse_general_enc = dataclasses.replace(
            sparse_general,
            tag="sparse_general_enc",
            load_decoders_from_tag=sparse_general.tag,
            shuffle_decoders=True,
        )
        sparse_general_dec = dataclasses.replace(
            sparse_general,
            tag="sparse_general_dec",
            load_encoders_from_tag=sparse_general.tag,
            shuffle_decoders=True,
        )


if __name__ == "__main__":
    output_dir = Path("out")
    with open(output_dir / "multi_seq_results.pkl", "rb") as f:
        independent_results = pickle.load(f)
    independent_vals = [
        _get_output_vals(tr) for tr in independent_results if "test" in tr.tag
    ]

    with open(output_dir / "multiresult.pkl", "rb") as f:
        distill_results = pickle.load(f)
    distill_enc_vals = [
        _get_output_vals(tr) for tr in distill_results if "enc" in tr.tag
    ]
    distill_dec_vals = [
        _get_output_vals(tr) for tr in distill_results if "dec" in tr.tag
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    sparsity_thresholds = list(range(1, 6))

    result_tags = ["independent", "retrain_encs", "retrain_decs"]
    result_lists = [independent_vals, distill_enc_vals, distill_dec_vals]

    for name, result_set in zip(result_tags, result_lists):
        p1s = [v["reconstruction_loss_p1"] for v in result_set]
        p2s = [v["reconstruction_loss_p2"] for v in result_set]
        print(p1s)
        ax1.plot(sparsity_thresholds, p1s, label=name)
        ax2.plot(sparsity_thresholds, p2s, label=name)

    ax1.legend()
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Sparsity threshold")
    ax1.set_ylabel("p1 reconstruction loss")

    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.set_xlabel("Sparsity threshold")
    ax2.set_ylabel("p2 reconstruction loss")

    plt.show()
