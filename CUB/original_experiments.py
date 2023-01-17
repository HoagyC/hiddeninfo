import argparse
import dataclasses
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.cub_classes import Experiment
from CUB.config import N_ATTRIBUTES

from CUB.train_CUB import (
    train_X_to_C,
    train_oracle_C_to_y_and_test_on_Chat,
    train_Chat_to_y_and_test_on_Chat,
    train_X_to_C_to_y,
    train_X_to_y,
    train_X_to_Cy,
)

# Argument parser from github.com/yewsiang/ConceptBottleneck
# Used to check hyperparameter defaults for replicating experiments
def parse_arguments(experiment):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description="CUB Training")
    parser.add_argument("dataset", type=str, help="Name of the dataset.")
    parser.add_argument(
        "exp",
        type=str,
        choices=[
            "Concept_XtoC",
            "Independent_CtoY",
            "Sequential_CtoY",
            "Standard",
            "Multitask",
            "Joint",
            "Probe",
            "TTI",
            "Robustness",
            "HyperparameterSearch",
        ],
        help="Name of experiment to run.",
    )
    parser.add_argument("--seed", required=True, type=int, help="Numpy and torch seed.")
    parser.add_argument(
        "-log_dir", default=None, help="where the trained model is saved"
    )
    parser.add_argument("-batch_size", "-b", type=int, help="mini-batch size")
    parser.add_argument("-epochs", "-e", type=int, help="epochs for training process")
    parser.add_argument(
        "-save_step", default=1000, type=int, help="number of epochs to save model"
    )
    parser.add_argument("-lr", type=float, help="learning rate")
    parser.add_argument(
        "-weight_decay", type=float, default=5e-5, help="weight decay for optimizer"
    )
    parser.add_argument(
        "-pretrained",
        "-p",
        action="store_true",
        help="whether to load pretrained model & just fine-tune",
    )
    parser.add_argument(
        "-freeze",
        action="store_true",
        help="whether to freeze the bottom part of inception network",
    )
    parser.add_argument(
        "-use_aux", action="store_true", help="whether to use aux logits"
    )
    parser.add_argument(
        "-use_attr",
        action="store_true",
        help="whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)",
    )
    parser.add_argument(
        "-attr_loss_weight",
        default=1.0,
        type=float,
        help="weight for loss by predicting attributes",
    )
    parser.add_argument(
        "-no_img",
        action="store_true",
        help="if included, only use attributes (and not raw imgs) for class prediction",
    )
    parser.add_argument(
        "-bottleneck",
        help="whether to predict attributes before class labels",
        action="store_true",
    )
    parser.add_argument(
        "-weighted_loss",
        default="",  # note: may need to reduce lr
        help="Whether to use weighted loss for single attribute or multiple ones",
    )
    parser.add_argument(
        "-uncertain_labels",
        action="store_true",
        help="whether to use (normalized) attribute certainties as labels",
    )
    parser.add_argument(
        "-n_attributes",
        type=int,
        default=N_ATTRIBUTES,
        help="whether to apply bottlenecks to only a few attributes",
    )
    parser.add_argument(
        "-expand_dim",
        type=int,
        default=0,
        help="dimension of hidden layer (if we want to increase model capacity) - for bottleneck only",
    )
    parser.add_argument(
        "-n_class_attr",
        type=int,
        default=2,
        help="whether attr prediction is a binary or triary classification",
    )
    parser.add_argument(
        "-data_dir",
        default="official_datasets",
        help="directory to the training data",
    )
    parser.add_argument(
        "-image_dir", default="images", help="test image folder to run inference on"
    )
    parser.add_argument(
        "-resampling", help="Whether to use resampling", action="store_true"
    )
    parser.add_argument(
        "-end2end",
        action="store_true",
        help="Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg",
    )
    parser.add_argument(
        "-optimizer",
        default="SGD",
        help="Type of optimizer to use, options incl SGD, RMSProp, Adam",
    )
    parser.add_argument(
        "-ckpt", default="", help="For retraining on both train + val set"
    )
    parser.add_argument(
        "-scheduler_step",
        type=int,
        default=1000,
        help="Number of steps before decaying current learning rate by half",
    )
    parser.add_argument(
        "-normalize_loss",
        action="store_true",
        help="Whether to normalize loss by taking attr_loss_weight into account",
    )
    parser.add_argument(
        "-use_relu",
        action="store_true",
        help="Whether to include relu activation before using attributes to predict Y. "
        "For end2end & bottleneck model",
    )
    parser.add_argument(
        "-use_sigmoid",
        action="store_true",
        help="Whether to include sigmoid activation before using attributes to predict Y. "
        "For end2end & bottleneck model",
    )
    parser.add_argument(
        "-connect_CY",
        action="store_true",
        help="Whether to use concepts as auxiliary features (in multitasking) to predict Y",
    )

    parser.add_argument("--multimodel", default=False)
    args = parser.parse_args()
    args.three_class = args.n_class_attr == 3
    return (args,)


"""
Experiment for independent models


Training an XtoC Model

python3 src/experiments.py cub Concept_XtoC --seed 1 -ckpt 1 -log_dir ConceptModel__Seed1/outputs/ -e 1000 \
 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 \
-n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck

Getting predicted C logits

python3 src/CUB/generate_new_data.py ExtractConcepts --model_path ConceptModel__Seed1/outputs/best_model_1.pth \
--data_dir CUB_processed/class_attr_data_10 --out_dir ConceptModel1__PredConcepts

Training C to Y

python3 src/experiments.py cub Independent_CtoY --seed 1 -log_dir IndependentModel_WithVal___Seed1/outputs/ -e 500 \
-optimizer sgd -use_attr -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -no_img -b 64 \
-weight_decay 0.00005 -lr 0.001 -scheduler_step 1000 
"""

ind_XtoC_cfg = Experiment(
    tag="independent",
    seed=1,
    exp="Concept_XtoC",
    epochs=1000,
    optimizer="SGD",
    pretrained=True,
    use_attr=True,
    n_attributes=109,
    batch_size=64,
    weight_decay=4e-5,
    lr=0.01,
    scheduler_step=1000,
    use_aux=True,
    bottleneck=True,
)

ind_CtoY_cfg = Experiment(
    tag="independent",
    exp="Independent_CtoY",
    seed=1,
    epochs=500,
    optimizer="SGD",
    use_attr=True,
    n_attributes=109,
    no_img=True,
    batch_size=64,
    weight_decay=5e-5,
    lr=0.001,
    scheduler_step=1000,
)


"""
Sequential models

python3 src/experiments.py cub Sequential_CtoY --seed 1 -log_dir SequentialModel_WithVal__Seed1/outputs/ \
-e 1000 -optimizer sgd -pretrained -use_aux -use_attr -data_dir ConceptModel1__PredConcepts -n_attributes 112 \
-no_img -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 1000 


"""

seq_CtoY_cfg = Experiment(
    tag="seq_CtoY",
    exp="Sequential_CtoY",
    seed=1,
    epochs=1000,
    optimizer="SGD",
    pretrained=True,
    use_aux=True,
    use_attr=True,
    n_attributes=109,
    no_img=True,
    batch_size=64,
    weight_decay=4e-5,
    lr=0.001,
    scheduler_step=1000,
)

"""
Joint training

python3 src/experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0.01Model__Seed1/outputs/ \
-e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple \
-data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 \
-normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end

"""

joint_cfg = Experiment(
    tag="joint",
    exp="Joint",
    seed=1,
    epochs=1000,
    optimizer="SGD",
    pretrained=True,
    use_aux=True,
    use_attr=True,
    weighted_loss="multiple",
    n_attributes=109,
    attr_loss_weight=0.01,
    normalize_loss=True,
    batch_size=64,
    weight_decay=4e-4,
    lr=0.001,
    scheduler_step=1000,
    end2end=True,
)

"""
'Standard' experiment

python3 src/experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0Model_Seed1/outputs/ \
-e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple \
-data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0 \
-normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end

"""

standard_orig_cfg = Experiment(
    tag="standard_orig",
    exp="Joint",
    seed=1,
    epochs=1000,
    optimizer="SGD",
    pretrained=True,
    use_aux=True,
    use_attr=True,
    weighted_loss="multiple",
    n_attributes=109,
    attr_loss_weight=0.0,
    normalize_loss=True,
    batch_size=64,
    weight_decay=4e-4,
    lr=0.01,
    scheduler_step=20,
)


"""
Standard no bottleneck

python3 src/experiments.py cub Standard --seed 1 -ckpt 1 -log_dir StandardNoBNModel_Seed1/outputs/ \
-e 1000 -optimizer sgd -pretrained -use_aux -data_dir CUB_processed/class_attr_data_10 -b 64 \
-weight_decay 0.0004 -lr 0.01 -scheduler_step 20

"""

standard_nobottle_cfg = Experiment(
    tag="standard_nobottle",
    exp="Standard",
    seed=1,
    epochs=1000,
    optimizer="SGD",
    pretrained=True,
    use_aux=True,
    batch_size=64,
    weight_decay=4e-4,
    lr=0.01,
    scheduler_step=20,
)


"""
Multitask

python3 src/experiments.py cub Multitask --seed 1 -ckpt 1 -log_dir MultitaskModel_Seed1/outputs/ \
-e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple \
-data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 \
-normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20

"""

multitask_cfg = Experiment(
    tag="multitask",
    exp="Multitask",
    seed=1,
    epochs=1000,
    pretrained=True,
    use_aux=True,
    use_attr=True,
    weighted_loss="multiple",
    n_attributes=109,
    attr_loss_weight=0.01,
    normalize_loss=True,
    batch_size=64,
    weight_decay=4e-5,
    lr=0.01,
    scheduler_step=20,
)


def orig_run_fn(args: Experiment) -> None:
    experiment = args.exp
    if experiment == "Concept_XtoC":
        train_X_to_C(args)

    elif experiment == "Independent_CtoY":
        train_oracle_C_to_y_and_test_on_Chat(args)

    elif experiment == "Sequential_CtoY":
        train_Chat_to_y_and_test_on_Chat(args)

    elif experiment == "Joint":
        train_X_to_C_to_y(args)

    elif experiment == "Standard":
        train_X_to_y(args)

    elif experiment == "Multitask":
        train_X_to_Cy(args)


if __name__ == "__main__":

    original_cfgs = [
        ind_XtoC_cfg,
        ind_CtoY_cfg,
        seq_CtoY_cfg,
        joint_cfg,
        standard_orig_cfg,
        standard_nobottle_cfg,
        multitask_cfg,
    ]

    for cfg in original_cfgs[5:]:
        run_cfg = dataclasses.replace(cfg, epochs=1, quick=True)

        orig_run_fn(run_cfg)
