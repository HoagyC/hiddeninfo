# Run this file to replicate all experiments

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.train_CUB import train_multimodel
from CUB.classes import Experiment

from CUB.original_experiments import ind_XtoC_cfg, ind_CtoY_cfg, orig_run_fn


def main():
    # Run original experiments
    orig_run_fn(ind_XtoC_cfg)
    orig_run_fn(ind_CtoY_cfg)

    # Run new experiments
    train_multimodel()


if __name__ == "__main__":
    main()
