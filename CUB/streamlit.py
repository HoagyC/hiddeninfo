import pickle
import os
import sys
from typing import Optional, Iterable

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.cub_classes import TTI_Output
from CUB.process_results import get_most_recent, get_all_runs
from CUB.cub_utils import download_from_aws, list_aws_files
from CUB.tti import graph_tti_output

def _load_train_results(folder) -> Optional[TTI_Output]:
    tti_file = folder + "tti_results.pkl"
    if tti_file in list_aws_files(folder, get_folders=False):
        config_file = folder + "config.pkl"
        download_from_aws([tti_file, config_file])
        with open(tti_file, "rb") as f:
            train_results = pickle.load(f)
        return train_results
    else:
        return None

def _display_experiments(*experiments_iterable: Iterable[str], load_all=False) -> None:
    run_folders = []
    for experiment_tag in experiments_iterable:
        if load_all:
            run_folders.extend(get_all_runs("out/" + experiment_tag))
        else:
            run_folders.append(get_most_recent("out/" + experiment_tag))
    
    for folder in run_folders:
        train_results = _load_train_results(folder)
        if train_results is None:
            st.write(f"Could not find results for {folder}")
            continue
        else:
            fig = graph_tti_output(train_results, return_fig=True, show=False)
            st.pyplot(fig)



def main() -> None:
    st.title("CUB Results")
    
    st.header("Joint")
    _display_experiments("joint")

    st.header("Independent")
    _display_experiments("ind_CtoY")

    # st.header("Sequential")
    # _display_experiments("seq_CtoY")

    # st.header("Multimodel")
    # _display_experiments("multimodel")

    # st.header("Sparse Multimodel")
    # _display_experiments("sparse_multimodel0.1-3", "sparse_multimodel1-3", "sparse_multimodel10-3")


if __name__ == "__main__":
    main()
