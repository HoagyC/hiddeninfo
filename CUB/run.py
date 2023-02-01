# Run this file to replicate all experiments

import dataclasses
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.train_CUB import train_multimodel
from CUB.cub_classes import Experiment, TTI_Config
from CUB.tti import run_tti


basic_tti_args = TTI_Config(
    use_attr=True,
    bottleneck=True,
    n_trials=5,
    use_invisible=True,
    class_level=True,
    data_dir2="CUB_processed",
    use_sigmoid=True,
    multimodel=False,
)

def main():
    # # Run original experiments
    # orig_run_fn(ind_XtoC_cfg)
    # orig_run_fn(ind_CtoY_cfg)

    # # Run new experiments
    # train_multimodel()


    # Run interventions on them (TTI)
    ind_tti_args = dataclasses.replace(basic_tti_args,
        model_dir="out/ind_XtoC/20221130-150657/final_model.pth",
        model_dir2="out/ind_CtoY/20221130-194327/final_model.pth",
        log_dir="tti/ind_XtoCtoY",
    )

    multimodel_tti_args = dataclasses.replace(basic_tti_args,
        model_dir="out/basic/20221201-041522/final_model.pth",
        log_dir="tti/multimodel",
        multimodel=True,
    )

    ind_tti_results = run_tti(ind_tti_args)
    multimodel_tti_results = run_tti(multimodel_tti_args)

    if not os.path.exists("tti"):
        os.mkdir("tti")
    with open("tti/ind_tti_results.pkl", "wb") as f:
        pickle.dump(ind_tti_results, f)
    with open("tti/multimodel_tti_results.pkl", "wb") as f:
        pickle.dump(multimodel_tti_results, f)




if __name__ == "__main__":
    main()
