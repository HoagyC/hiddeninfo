"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import argparse
import copy
import dataclasses
from datetime import datetime
import math
import os
from pathlib import Path
import pickle
import random
import sys

from typing import Iterable, List, Optional, Tuple, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import torch
import wandb

from CUB.analysis import accuracy, binary_accuracy
from CUB.dataset import load_data, find_class_imbalance
from CUB.models import (
    IndependentModel,
    JointModel,
    Multimodel,
    SequentialModel,
    ThinMultimodel,
    CUB_Model,
    CUB_Multimodel
)
from CUB.cub_classes import TTI_Config, Meters, Experiment, RunRecord, N_CLASSES, N_ATTRIBUTES
import CUB.configs as cfgs

from CUB.cub_utils import upload_to_aws, get_secrets
from CUB.tti import run_tti

DATETIME_FMT = "%Y%m%d-%H%M%S"
RESULTS_DIR ="out"

def run_epoch(
    model: CUB_Model,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    meters: Meters,
    is_training: bool,
) -> None:
    if is_training:
        model.train()
    else:
        model.eval()

    for data in loader:
        inputs, class_labels, attr_labels, batch_attr_mask = data
        if sum(batch_attr_mask) < 2 and hasattr(model, "train_mode") and model.train_mode in ["XtoC", "CtoY"]:
            print("Skipping batch, consider increasing batch size or decreasing sparsity")
            continue

        attr_labels = [i.float() for i in attr_labels]
        attr_labels = torch.stack(attr_labels, dim=1)

        attr_labels = attr_labels.cuda() if torch.cuda.is_available() else attr_labels
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        class_labels = class_labels.cuda() if torch.cuda.is_available() else class_labels
        batch_attr_mask = batch_attr_mask.cuda() if torch.cuda.is_available() else batch_attr_mask
        attr_preds, aux_attr_preds, class_preds, aux_class_preds = model.generate_predictions(inputs, attr_labels, batch_attr_mask)

        loss = model.generate_loss(
            attr_preds=attr_preds, 
            aux_attr_preds=aux_attr_preds, 
            class_preds=class_preds,
            aux_class_preds=aux_class_preds, 
            attr_labels=attr_labels, 
            class_labels=class_labels,
            mask=batch_attr_mask
        )
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        meters.loss.update(loss.item(), inputs.size(0))

        if attr_preds is not None:
            # Note that the first dimension in all outputs is the number of models, then batch size, then the rest of the dimensions
            attr_labels = attr_labels.repeat(args.n_models, 1, 1)
            if attr_preds.shape[1] != attr_labels.shape[1]:
                attr_labels=attr_labels[batch_attr_mask]

            attr_pred_sigmoids = torch.nn.Sigmoid()(attr_preds)
            attr_acc = binary_accuracy(attr_pred_sigmoids, attr_labels[:, :, :args.n_attributes])
            meters.concept_acc.update(attr_acc.data.cpu().numpy(), inputs.size(0))
        
        if class_preds is not None:
            class_labels = class_labels.repeat(args.n_models, 1)
            
            if class_preds.shape[1] != class_labels.shape[1]:
                class_labels=class_labels[batch_attr_mask]
            # Collecting the n_model and batch size dimensions into one for the accuracy function
            class_acc = accuracy(class_preds.reshape(-1, N_CLASSES), class_labels.reshape(-1), topk=(1,))
            meters.label_acc.update(class_acc[0], inputs.size(0))


def train(
    model: CUB_Model,
    args: Experiment,
    n_epochs: int,
    init_epoch: int = 0,
):
    print(f"Running {args.tag}")
    now_str = datetime.now().strftime(DATETIME_FMT)
    run_save_path = Path(args.log_dir) / args.tag / now_str

    if not run_save_path.is_dir():
        run_save_path.mkdir(parents=True)

    wandb.init(project="distill_CUB", config=args.__dict__)

    if args.multimodel:
        assert type(model) == CUB_Multimodel
        if args.reset_pre_models:
            model.reset_pre_models()
        if args.reset_post_models:
            model.reset_post_models()
        
    model.train()
    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = make_optimizer(params, args)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=0.1
    )

    stop_epoch = (
        int(math.log(args.min_lr / args.lr) / math.log(args.lr_decay_size)) * args.scheduler_step
    )
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(args.base_dir, args.data_dir, "train.pkl")
    val_data_path = train_data_path.replace("train.pkl", "val.pkl")

    train_loader = load_data([train_data_path], args)
    val_loader = load_data([val_data_path], args)

    val_records = RunRecord()

    if args.use_test:
        test_data_path = train_data_path.replace("train.pkl", "test.pkl")
        test_loader = load_data([test_data_path], args)

    for epoch_ndx in range(init_epoch, init_epoch + n_epochs):
        train_meters = Meters()
        val_meters = Meters()

        run_epoch(
            model,
            optimizer,
            train_loader,
            train_meters,
            is_training=True
        )

        run_epoch(
            model,
            optimizer,
            val_loader,
            val_meters,
            is_training=False
        )

        if args.use_test:
            test_meters = Meters()
            run_epoch(
                model,
                optimizer,
                test_loader,
                test_meters,
                is_training=False
            )

        else:
            test_meters = None

        write_metrics(
            epoch_ndx,
            model,
            args,
            train_meters,
            val_meters,
            val_records,
            run_save_path,
            test_meters=test_meters,
        )
        if not (args.exp in ["Independent", "Sequential", "JustXtoC"] and model.train_mode == "XtoC") and \
            args.tti_int > 0 and epoch_ndx % args.tti_int == 0 and args.n_attributes == N_ATTRIBUTES:
            model_save_path = run_save_path / f"{epoch_ndx}_model.pth"
            torch.save(model, model_save_path)
            upload_to_aws(s3_file_name=str(run_save_path / "latest_model.pth"), local_file_name=model_save_path)

            tti_cfg = TTI_Config(
                log_dir=str(run_save_path),
                model_dir=str(model_save_path),
                data_dir=args.data_dir,
                multimodel=args.multimodel,
                sigmoid=False,
                model_sigmoid=False,
                seed=args.seed,
            )

            tti_results = run_tti(tti_cfg)
            wandb.log(
                    dict(
                    epoch = epoch_ndx,
                    tti0 = tti_results[0][1],
                    tti10 = tti_results[10][1],
                    ttilast = tti_results[-1][1],
                )
            )



        if epoch_ndx <= stop_epoch:
            scheduler.step()


        if epoch_ndx % args.save_step == 0:
            now_str = datetime.now().strftime(DATETIME_FMT)
            torch.save(model, run_save_path / f"{epoch_ndx}_model.pth")

        # if epoch_ndx >= 100 and val_meters.label_acc.avg and  < 3:
        #     print("Early stopping because of low accuracy")
        #     break

        if epoch_ndx - val_records.epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break

    final_save(model, run_save_path, args)
    return epoch_ndx


def final_save(model: torch.nn.Module, run_path: Path, args: Experiment):
    model_save_path = run_path / "final_model.pth"
    with open(run_path / "config.pkl", "wb") as f:
        pickle.dump(args, f)
    torch.save(model, model_save_path)
    upload_files = ["final_model.pth", "config.pkl", "log.txt"]
    for filename in upload_files:
        upload_to_aws(run_path / filename)


def write_metrics(
    epoch: int,
    model: torch.nn.Module,
    args: Experiment,
    train_meters: Meters,
    val_meters: Meters,
    val_records: RunRecord,
    save_path: Path,
    test_meters: Optional[Meters] = None,
) -> None:
    # If training independent models, use concept accuracy as key metric since we don't have labels
    if args.exp in ["Independent", "Sequential", "JustXtoC"] and model.train_mode=="XtoC":
        is_record = val_meters.concept_acc.avg > val_records.acc
        if is_record:
            val_records.acc = val_meters.concept_acc.avg
    else:
        is_record = val_meters.label_acc.avg > val_records.acc
        if is_record:
            val_records.acc = val_meters.label_acc.avg

    if is_record:
        val_records.epoch = epoch
        torch.save(model, save_path / f"best_model_{args.seed}.pth")
        # if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
        #    break

    
    train_loss_avg = train_meters.loss.avg
    val_loss_avg = val_meters.loss.avg

    test_loss = test_meters.loss.avg if test_meters else 0
    test_acc = test_meters.label_acc.avg if test_meters else 0

    metrics_dict = {
        "epoch": epoch,
        "train_loss": train_loss_avg,
        "train_acc": train_meters.label_acc.avg,
        "val_loss": val_loss_avg,
        "val_acc": val_meters.label_acc.avg,
        "best_val_epoch": val_records.epoch,
        "concept_train_acc": train_meters.concept_acc.avg,
        "concept_val_acc": val_meters.concept_acc.avg,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    wandb.log(metrics_dict)


def make_optimizer(params: Iterable, args: Experiment) -> torch.optim.Optimizer:
    optimizer: torch.optim.Optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer type {args.optimizer} not recognized")

    return optimizer


def train_multi(args: Experiment) -> None:
    model: CUB_Multimodel
    if args.load is not None:
        model = torch.load(args.load)
        if args.thin:
            assert type(model) == ThinMultimodel
        else:
            assert type(model) == Multimodel
    else:
        if args.thin:
            model = ThinMultimodel(args)
        else:
            model = Multimodel(args)

    if args.do_sep_train:
        model.train_mode = "separate"
        elapsed_epochs = train(model, args, n_epochs=args.epochs[0])
    else:
        elapsed_epochs = 0

    if args.reset_post_models:
        model.reset_post_models()
    if args.reset_pre_models:
        model.reset_pre_models()
    
    # model.train_mode = "shuffle"

    if args.freeze_pre_models:
        assert type(model.pre_models) == torch.nn.ModuleList
        for param in model.pre_models.parameters():
            param.requires_grad = False
    
    if args.freeze_post_models:
        assert type(model.post_models) == torch.nn.ModuleList
        for param in model.post_models.parameters():
            param.requires_grad = False

    # Train again with shuffling and freezing
    train(model, args, init_epoch=elapsed_epochs, n_epochs=args.epochs[1])

def train_switch(args):
    if args.exp == "Independent":
        train_independent(args)
    elif args.exp == "Joint":
        if args.load:
            model = torch.load(args.load)
            assert type(model) == JointModel
        else:
            model = JointModel(args)
        train(model, args, n_epochs=args.epochs[0])
    elif args.exp == "Multimodel":
        train_multi(args)
    elif args.exp == "Sequential":
        train_sequential(args)
    elif args.exp == "Alternating":
        train_alternating(args)
    elif args.exp == "JustXtoC":
        train_just_XtoC(args)
    else:
        raise ValueError(f"Experiment type {args.exp} not recognized")

def train_independent(args):
    if args.load:
        model = torch.load(args.load)
        assert type(model) == IndependentModel
    else:   
        model = IndependentModel(args, train_mode="XtoC")
    train(model, args, n_epochs=args.epochs[0])
    model.train_mode = "CtoY"
    train(model, args, n_epochs=args.epochs[1])

def train_just_XtoC(args):
    if args.load:
        model = torch.load(args.load)
        assert type(model) in [SequentialModel, IndependentModel]
    else:
        model = SequentialModel(args, train_mode="XtoC")
    
    train(model, args, n_epochs=args.epochs[0])


def train_sequential(args):
    if args.load:
        model = torch.load(args.load)
        assert type(model) == SequentialModel
    else:
        model = SequentialModel(args, train_mode="XtoC")
    train(model, args, n_epochs=args.epochs[0])
    model.train_mode = "CtoY"
    train(model, args, n_epochs=args.epochs[1])

def train_alternating(args):
    if args.load:
        model = torch.load(args.load)
        if args.thin:
            assert type(model) == ThinMultimodel
        else:
            assert type(model) == Multimodel
    else:
        if args.thin:
            model = ThinMultimodel(args)
        else:
            model = Multimodel(args)
    
    elapsed_epochs = 0
    assert len(args.epochs) == args.n_alternating * 2, "Number of alternating epochs must match number of alternating models"
    for ndx in range(args.n_alternating):
        if args.freeze_first == "pre":
            model, elapsed_epochs = run_frozen_pre(args, model, args.epochs[ndx * 2], elapsed_epochs=elapsed_epochs)
            model, elapsed_epochs = run_frozen_post(args, model, args.epochs[(ndx * 2) + 1], elapsed_epochs=elapsed_epochs)
        elif args.freeze_first == "post":
            model, elapsed_epochs = run_frozen_post(args, model, args.epochs[ndx * 2], elapsed_epochs=elapsed_epochs)
            model, elapsed_epochs = run_frozen_pre(args, model, args.epochs[(ndx * 2) + 1], elapsed_epochs=elapsed_epochs)
        else:
            raise ValueError(f"Freeze first {args.freeze_first} not recognized")

def run_frozen_pre(args, model, epochs, elapsed_epochs=0):
    model.train_mode = "shuffle"
    for param in model.pre_models.parameters():
        param.requires_grad = True
    for param in model.post_models.parameters():
        param.requires_grad = False

    if args.alternating_reset:
        model.reset_pre_models()

    # Train again with shuffling and freezing
    elapsed_epochs = train(model, args, init_epoch=elapsed_epochs, n_epochs=epochs)

    return model, elapsed_epochs

def run_frozen_post(args, model, epochs, elapsed_epochs=0):
    model.train_mode = "shuffle"
    for param in model.pre_models.parameters():
        param.requires_grad = False
    for param in model.post_models.parameters():
        param.requires_grad = True

    if args.alternating_reset:
        model.reset_post_models()

    # Train again with shuffling and freezing
    elapsed_epochs = train(model, args, init_epoch=elapsed_epochs, n_epochs=epochs)

    return model, elapsed_epochs


def _save_CUB_result(train_result):
    now_str = datetime.now().strftime(DATETIME_FMT)
    train_result_path = RESULTS_DIR / train_result.tag / now_str / "train-result.pickle"
    if not train_result_path.parent.is_dir():
        train_result_path.parent.mkdir(parents=True)
    with train_result_path.open("wb") as f:
        print(train_result)
        pickle.dump(train_result, f)


def make_configs_list() -> List[Experiment]:
    configs = [
        cfgs.multi_inst_post_cfg,
        copy.deepcopy(cfgs.multi_inst_post_cfg),
    ]
    for cfg in configs:
        cfg.tti_int = 10
        cfg.reset_post_models = False
    
    configs[1].seed = 2

    return configs

    # # Make all output folders specific to this run
    # for cfg in configs:
    #     cfg.log_dir = "big_run"
    #     cfg.tag += "_sparse"
    #     cfg.attr_sparsity = 5

    #     # If attrs are sparse, can increase batch if not training end-to-end
    #     if cfg.exp in ["Sequential", "Independent"]:
    #         cfg.batch_size *= cfg.attr_sparsity

    # joint_configs = [
    #     copy.deepcopy(cfgs.joint_inst_cfg) for _ in range(5)
    # ]
    # for i, cfg in enumerate(joint_configs):
    #     cfg.log_dir = "big_run"
    #     cfg.tag += f"_{i}"
    
    # seq_configs = [copy.deepcopy(cfgs.just_xtoc_cfg) for _ in range(5)]
    # for i, cfg in enumerate(seq_configs):
    #     cfg.log_dir = "big_run"
    #     cfg.tag += f"_{i}"

    # return joint_configs + seq_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-index', type=int, default=0, help='Index of config to run')
    args = parser.parse_args()
    configs = make_configs_list()
    args = configs[args.cfg_index]

    secrets = get_secrets()
    wandb.login(key=secrets["wandb_key"])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    train_switch(args)
    
    wandb.finish()