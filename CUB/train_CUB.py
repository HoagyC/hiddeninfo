"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import argparse
import dataclasses
from datetime import datetime
import os
from pathlib import Path
import pickle
import random
import sys

from typing import Iterable, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

import wandb

from CUB.analysis import accuracy, binary_accuracy
from CUB.dataset import load_data, find_class_imbalance
from CUB.models import (
    IndependentModel,
    JointModel,
    Multimodel,
)
from CUB.cub_classes import Experiment, Meters, RunRecord
from CUB.cub_classes import ind_XtoC_cfg, ind_CtoY_cfg, joint_cfg, multiple_cfg

from CUB.config import MIN_LR, BASE_DIR, LR_DECAY_SIZE
from CUB.cub_utils import upload_to_aws, get_secrets

DATETIME_FMT = "%Y%m%d-%H%M%S"
RESULTS_DIR ="out"

def run_epoch(
    model: torch.nn.Module,
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
        inputs, class_labels, attr_labels, attr_mask = data
        attr_labels = [i.float() for i in attr_labels]
        attr_labels = torch.stack(attr_labels, dim=1)

        attr_labels = attr_labels.cuda() if torch.cuda.is_available() else attr_labels
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        class_labels = class_labels.cuda() if torch.cuda.is_available() else class_labels
        attr_mask = attr_mask.cuda() if torch.cuda.is_available() else attr_mask
        attr_preds, aux_attr_preds, class_preds, aux_class_preds = model.generate_predictions(inputs, attr_labels, attr_mask)

        if is_training:
            loss = model.generate_loss(
                attr_preds=attr_preds, 
                aux_attr_preds=aux_attr_preds, 
                class_preds=class_preds,
                aux_class_preds=aux_class_preds, 
                attr_labels=attr_labels, 
                class_labels=class_labels,
                mask=attr_mask
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meters.loss.update(loss.item(), inputs.size(0))

        if attr_preds is not None:
            # Multimodel preds are a list of tensors, one for each model
            # so we concatenate them as if they were a larger batch
            if args.multimodel:
                attr_preds_t = torch.cat([torch.cat(a, dim=1) for a in attr_preds], dim=0)
                attr_labels = attr_labels.repeat(args.n_models, 1)
            else:
                attr_preds_t = torch.cat(attr_preds, dim=1)
            
            attr_pred_sigmoids = torch.nn.Sigmoid()(attr_preds_t)
            attr_acc = binary_accuracy(attr_pred_sigmoids, attr_labels)
            meters.concept_acc.update(attr_acc.data.cpu().numpy(), inputs.size(0))
        
        if class_preds is not None:
            if args.multimodel:
                class_preds = torch.cat(class_preds, dim=0)
                class_labels = class_labels.repeat(args.n_models)
            
            class_acc = accuracy(class_preds, class_labels, topk=(1,))
            meters.label_acc.update(class_acc[0], inputs.size(0))


def train(
    model: torch.nn.Module,
    args: Experiment,
    init_epoch: int = 0,
):
    print(f"Running {args.tag}")
    now_str = datetime.now().strftime(DATETIME_FMT)
    run_save_path = Path(args.log_dir) / args.tag / now_str
    if not run_save_path.is_dir():
        run_save_path.mkdir(parents=True)
    wandb.init(project="distill_CUB", config=args.__dict__)

    if args.multimodel:
        if args.reset_pre_models:
            model.reset_pre_models()
        if args.reset_post_models:
            model.reset_post_models()

    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = True

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = make_optimizer(params, args)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=0.1
    )

    stop_epoch = (
        int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    )
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, args.data_dir, "train.pkl")
    val_data_path = train_data_path.replace("train.pkl", "val.pkl")

    train_loader = load_data([train_data_path], args)
    val_loader = load_data([val_data_path], args)

    train_meters = Meters()
    val_meters = Meters()
    val_records = RunRecord()

    for epoch_ndx in range(init_epoch, args.epochs + init_epoch):
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

        write_metrics(
            epoch_ndx,
            model,
            args,
            train_meters,
            val_meters,
            val_records,
            run_save_path,
        )

        if epoch_ndx <= stop_epoch:
            scheduler.step()

        if epoch_ndx % 10 == 0:
            print("Current lr:", scheduler.get_lr())

        if epoch_ndx % args.save_step == 0:
            now_str = datetime.now().strftime(DATETIME_FMT)
            torch.save(model, run_save_path / f"{epoch_ndx}_model.pth")

        if epoch_ndx >= 100 and val_meters.label_acc.avg < 3:
            print("Early stopping because of low accuracy")
            break
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


def make_criteria(
    args: Experiment,
) -> Tuple[torch.nn.Module, Optional[List[torch.nn.Module]]]:
    criterion = torch.nn.CrossEntropyLoss()

    attr_criterion: Optional[List[torch.nn.Module]]

    if args.use_attr and not args.no_img:
        attr_criterion = []  # separate criterion (loss function) for each attribute
        if args.weighted_loss:
            train_data_path = os.path.join(BASE_DIR, args.data_dir, "train.pkl")
            imbalance = find_class_imbalance(train_data_path, True)
            for ratio in imbalance:
                attr_criterion.append(
                    torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda())
                )
        else:
            for _ in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None

    return criterion, attr_criterion


def write_metrics(
    epoch: int,
    model: torch.nn.Module,
    args: Experiment,
    train_meters: Meters,
    val_meters: Meters,
    val_records: RunRecord,
    save_path: Path,
) -> None:
    # If training independent models, use concept accuracy as key metric since we don't have labels
    if args.exp == "Independent_XtoC":
        is_record = val_meters.concept_acc.avg > val_records.concept_acc
        if is_record:
            val_records.concept_acc = val_meters.concept_acc.avg
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

    metrics_dict = {
        "epoch": epoch,
        "train_loss": train_loss_avg,
        "train_acc": train_meters.label_acc.avg,
        "val_loss": val_loss_avg,
        "val_acc": val_meters.label_acc.avg,
        "best_val_epoch": val_records.epoch,
        "concept_train_acc": train_meters.concept_acc.avg,
        "concept_val_acc": val_meters.concept_acc.avg,
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
    model = Multimodel(args)
    elapsed_epochs = train(model, args)

    if args.reset_post_models:
        model.reset_post_models()
    if args.reset_pre_models:
        model.reset_pre_models()
    
    args.shuffle_models = False

    if args.freeze_pre_models:
        for param in model.pre_models.parameters():
            param.requires_grad = False
    
    if args.freeze_post_models:
        for param in model.post_models.parameters():
            param.requires_grad = False

    # Train again with shuffling and freezing
    train(model, args, init_epoch=elapsed_epochs)

def train_single(args):
    if args.exp == "Concept_XtoC":
        model = IndependentModel(args, train_mode="XtoC")
    elif args.exp == "Independent_CtoY":
        model = IndependentModel(args, train_mode="CtoY")
    elif args.exp == "Joint":
        model = JointModel(args)
    train(model, args)


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
        ind_XtoC_cfg,
        ind_CtoY_cfg,
        joint_cfg,
        multiple_cfg
    ]
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-index', type=int, default=0, help='Index of config to run')
    args = parser.parse_args()
    configs = make_configs_list()
    args = configs[args.cfg_index]

    secrets = get_secrets()
    wandb.login(key=secrets["wandb_key"])

    if args.multimodel:
        train_multi(args)
    else:
        train_single(args)
    
    wandb.finish()