# Run this file to replicate all experiments

import dataclasses
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.train_CUB import train_multimodel
from CUB.cub_classes import Experiment, TTI_Config
from CUB.tti import run_tti

from CUB.original_experiments import ind_XtoC_cfg, ind_CtoY_cfg, orig_run_fn



basic_tti_args = TTI_Config(
    use_attr=True,
    bottleneck=True,
    n_trials=5,
    use_invisible=True,
    class_level=True,
    data_dir2="CUB_processed",
    use_sigmoid=True,
)

def main():
    # Run original experiments
    orig_run_fn(ind_XtoC_cfg)
    orig_run_fn(ind_CtoY_cfg)

    # Run new experiments
    train_multimodel()


    # Run interventions on them (TTI)
    ind_tti_args = dataclasses.replace(basic_tti_args,
        model_dirs=["out/ind_XtoC/20221130-150657/final_model.pth"],
        model_dirs2=["out/ind_CtoY/20221130-194327/final_model.pth"],
        log_dir="out/ind_XtoCtoY",
    )

    multimodel_tti_args = dataclasses.replace(basic_tti_args,
        model_dirs=["out/multimodel/20221130-150657/final_model.pth"],
        log_dir="out/multimodel_XtoCtoY",
        multimodel=True,
    )

    ind_tti_results = run_tti(ind_tti_args)
    multimodel_tti_results = run_tti (multimodel_tti_args)



if __name__ == "__main__":
    main()
