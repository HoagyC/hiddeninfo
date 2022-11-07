import dataclasses
from pathlib import Path
import pickle
import streamlit as st

from ..main import _run_experiments

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

results = []
out_loc = Path("out/multi_seq_results.pkl")
for quadrant_sparsity in range(1, 6):
    st.header(f"running sparsity {quadrant_sparsity}")
    seq_sparse_decoder = dataclasses.replace(
        base,
        tag="seq_sparse_dec" + str(quadrant_sparsity),
        loss_quadrants="bin_sum",
        quadrant_threshold=quadrant_sparsity,
        give_full_info=True,
        num_batches=10000,
    )

    seq_sparse_encoder = dataclasses.replace(
        base,
        tag="seq_sparse_enc" + str(quadrant_sparsity),
        loss_quadrants="bin_sum",
        quadrant_threshold=quadrant_sparsity,
        reconstruction_loss_scale=0,
        num_batches=10000,
    )
    seq_sparse_test = dataclasses.replace(
        base,
        tag="sequential_test" + str(quadrant_sparsity),
        load_decoders_from_tag=seq_sparse_decoder.tag,
        load_encoders_from_tag=seq_sparse_encoder.tag,
        num_batches=2000,
    )
    results += _run_experiments(seq_sparse_decoder, seq_sparse_encoder, seq_sparse_test)
    with open(out_loc, "wb") as f:
        pickle.dump(results, f)