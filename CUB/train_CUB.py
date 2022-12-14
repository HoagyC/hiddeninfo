"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import argparse
import dataclasses
from datetime import datetime
import logging
import os
from pathlib import Path
import pickle
import random
import sys

from typing import Dict, Iterable, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from torch.utils.data import DataLoader

import numpy as np
import wandb

from CUB.analysis import accuracy, binary_accuracy
from CUB.dataset import load_data, find_class_imbalance
from CUB.models import (
    ModelXtoCY,
    ModelXtoChat_ChatToY,
    ModelXtoY,
    ModelXtoC,
    ModelOracleCtoY,
    ModelXtoCtoY,
    Multimodel,
)
from CUB.cub_classes import AverageMeter, Experiment, Meters, RunRecord
from CUB.config import MIN_LR, BASE_DIR, LR_DECAY_SIZE, AUX_LOSS_RATIO
from CUB.cub_utils import upload_to_aws, get_secrets

DATETIME_FMT = "%Y%m%d-%H%M%S"
RESULTS_DIR ="out"

def run_epoch_simple(model, optimizer, loader, meters, criterion, args, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels, attr_mask = data
        if isinstance(inputs, list):
            # inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        labels = labels.cuda() if torch.cuda.is_available() else labels
        attr_mask = attr_mask.cuda() if torch.cuda.is_available else attr_mask

        masked_inputs = inputs[attr_mask]
        masked_labels = labels[attr_mask]

        masked_outputs = model(masked_inputs)
        loss = criterion(masked_outputs, masked_labels)
        acc = accuracy(masked_outputs, labels, topk=(1,))
        meters.loss.update(loss.item(), masked_inputs.size(0))
        meters.label_acc.update(acc[0], masked_inputs.size(0))

        if is_training:
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()
            optimizer.step()  # optimizer step to update parameters

        if args.quick:
            break
    return meters


def run_twopart_epoch(
    model,
    optimizer,
    loader,
    meters,
    criterion,
    attr_criterion,
    args,
    is_training,
):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels = None
            attr_mask = torch.ones(inputs.shape[0]).to(torch.bool)
        else:
            dat_tup = data
            inputs, labels, attr_labels, attr_mask = data
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()  # .float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels = attr_labels.float()
            attr_labels = (
                attr_labels.cuda() if torch.cuda.is_available() else attr_labels
            )

        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        labels = labels.cuda() if torch.cuda.is_available() else labels
        if attr_labels is not None:
            attr_mask = attr_mask.cuda() if torch.cuda.is_available() else attr_mask
            masked_attr_labels = attr_labels[attr_mask]

        masked_inputs = inputs[attr_mask]
        masked_labels = labels[attr_mask]

        losses = []
        out_start = 0

        if is_training and args.use_aux:
            masked_outputs, masked_aux_outputs = model(masked_inputs)

            if not args.bottleneck:
                # loss main is for the main task label (always the first output)
                loss_main = criterion(
                    masked_outputs[0], masked_labels
                ) + AUX_LOSS_RATIO * criterion(masked_aux_outputs[0], masked_labels)
                losses.append(loss_main)
                out_start = 1
            if (
                attr_criterion is not None and args.attr_loss_weight > 0
            ):  # X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    masked_attr_output = (
                        masked_outputs[i + out_start]
                        .squeeze()
                        .type(torch.cuda.FloatTensor)
                    )
                    masked_attr_aux_output = (
                        masked_aux_outputs[i + out_start]
                        .squeeze()
                        .type(torch.cuda.FloatTensor)
                    )

                    masked_attr_target = masked_attr_labels[:, i]
                    main_loss = attr_criterion[i](
                        masked_attr_output, masked_attr_target
                    )

                    aux_loss = attr_criterion[i](
                        masked_attr_aux_output, masked_attr_target
                    )
                    losses.append(
                        args.attr_loss_weight * (main_loss + AUX_LOSS_RATIO * aux_loss)
                    )

        else:
            masked_outputs = model(masked_inputs)

            if not args.bottleneck:
                loss_main = criterion(masked_outputs[0], masked_labels)
                losses.append(loss_main)
                out_start = 1
            if (
                attr_criterion is not None and args.attr_loss_weight > 0
            ):  # X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    masked_attr_value = (
                        masked_outputs[i + out_start]
                        .squeeze()
                        .type(torch.cuda.FloatTensor)
                    )
                    masked_attr_target = masked_attr_labels[:, i]
                    attr_loss = attr_criterion[i](masked_attr_value, masked_attr_target)
                    losses.append(args.attr_loss_weight * attr_loss)

        if args.bottleneck:  # attribute accuracy
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(masked_outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            meters.concept_acc.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(
                masked_outputs[0], labels, topk=(1,)
            )  # only care about class prediction accuracy
            meters.label_acc.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses) / args.n_attributes
            else:  # cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (
                        1 + args.attr_loss_weight * args.n_attributes
                    )
        else:  # finetune
            total_loss = sum(losses)
        meters.loss.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if args.quick:
            break
    return meters


def run_multimodel_epoch(
    model,
    optimizer,
    loader,
    meters,
    criterion,
    attr_criterion,
    args,
    is_training,
):

    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        pre_model_ndx = random.randint(0, len(model.pre_models) - 1)
        if args.shuffle_models:
            post_model_ndx = random.randint(0, len(model.pre_models) - 1)
        else:
            post_model_ndx = pre_model_ndx

        pre_model = model.pre_models[pre_model_ndx]
        post_model = model.post_models[post_model_ndx]

        if attr_criterion is None:
            inputs, labels = data
            attr_labels = None
        else:
            inputs, labels, attr_labels, attr_mask_bin = data
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()  # .float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels = attr_labels.float()
            attr_labels = (
                attr_labels.cuda() if torch.cuda.is_available() else attr_labels
            )

        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        labels = labels.cuda() if torch.cuda.is_available() else labels
        attr_labels = attr_labels.cuda() if torch.cuda.is_available() else attr_labels
        attr_mask_bin = (
            attr_mask_bin.cuda() if torch.cuda.is_available() else attr_mask_bin
        )

        # args.use_aux adds an additional concept prediction layer to the pre_model, predicting
        # concepts from non-final layers to pass concept info to lower layers.
        if args.use_aux and is_training:
            concepts, aux_concepts = pre_model(inputs)
            concepts_t = torch.cat(concepts, dim=1)

        else:
            concepts = pre_model(inputs)
            concepts_t = torch.cat(concepts, dim=1)

        output_labels = post_model(concepts_t)

        losses = []
        # Loss main is for the main task label (always the first output)
        loss_main = 1.0 * criterion(output_labels, target=labels)
        losses.append(loss_main)

        # Adding losses separately for the different classes (if there are any non-masked attributes)
        if attr_mask_bin is not None and int(sum(attr_mask_bin)) != 0:
            for i in range(len(attr_criterion)):
                value = torch.masked_select(
                    concepts[i].type(torch.cuda.FloatTensor).squeeze(), attr_mask_bin
                )
                target = torch.masked_select(attr_labels[:, i], attr_mask_bin)
                attr_loss = attr_criterion[i](value, target) * args.attr_sparsity

                if args.use_aux and is_training:
                    aux_value = torch.masked_select(
                        aux_concepts[i].type(torch.cuda.FloatTensor).squeeze(),
                        attr_mask_bin,
                    )
                    aux_attr_loss = AUX_LOSS_RATIO * attr_criterion[i](aux_value, target) * args.attr_sparsity

                    attr_loss += aux_attr_loss

                losses.append(args.attr_loss_weight * attr_loss / args.n_attributes)
        else:
            for i in range(len(attr_criterion)):
                losses.append(0)

        # Calculating attribute accuracy
        sigmoid_outputs = torch.nn.Sigmoid()(concepts_t)
        concept_acc = binary_accuracy(sigmoid_outputs, attr_labels)
        meters.concept_acc.update(concept_acc.data.cpu().numpy(), inputs.size(0))

        label_acc = accuracy(
            output_labels, labels, topk=(1,)
        )  # only care about class prediction accuracy
        meters.label_acc.update(label_acc[0], inputs.size(0))

        total_loss = sum(losses)
        meters.loss.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if args.quick:
            break
    return meters


def train(
    model: torch.nn.Module,
    args: Experiment,
    split_models: bool = False,
    init_epoch: int = 0,
):
    print(f"Running {args.tag}")
    if args.quick:
        print("Speed running!")
    now_str = datetime.now().strftime(DATETIME_FMT)
    run_save_path = Path(args.log_dir) / args.tag / now_str
    if not run_save_path.is_dir():
        run_save_path.mkdir(parents=True)
    wandb.init(project="distill_CUB", config=args.__dict__)

    logging.basicConfig(filename=run_save_path / "log.txt", level=logging.INFO)
    logging.info(str(args) + "\n")

    if args.multimodel:
        if args.reset_pre_models:
            model.reset_pre_models()
        if args.reset_post_models:
            model.reset_post_models()

    model = model.cuda()

    criterion, attr_criterion = make_criteria(args)

    for param in model.parameters():
        param.requires_grad = True

    if args.freeze_pre_models:
        for param in model.pre_models.parameters():
            param.requires_grad = False
    if args.freeze_post_models:
        for param in model.post_models.parameters():
            param.requires_grad = False

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
    logging.info("train data path: %s\n" % train_data_path)

    train_loader = load_data([train_data_path], args, resampling=args.resampling)
    val_loader = load_data([val_data_path], args)

    val_records = RunRecord()

    for epoch_ndx in range(init_epoch, args.epochs + init_epoch):
        train_meters, val_meters = run_epoch(
            model,
            args,
            optimizer,
            criterion,
            attr_criterion,
            train_loader,
            val_loader,
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
    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, "train.pkl")
        if args.weighted_loss == "multiple":
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)

    attr_criterion: Optional[List[torch.nn.Module]]
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_attr and not args.no_img:
        attr_criterion = []  # separate criterion (loss function) for each attribute
        if args.weighted_loss:
            assert imbalance is not None
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
    if val_records.acc < val_meters.label_acc.avg:
        val_records.epoch = epoch
        val_records.acc = val_meters.label_acc.avg
        logging.info("New model best model at epoch %d\n" % epoch)
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
    logging.info(
        "Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t"
        "Val loss: %.4f\tVal acc: %.4f\t"
        "Best val epoch: %d\n"
        % (
            epoch,
            train_loss_avg,
            train_meters.label_acc.avg,
            val_loss_avg,
            val_meters.label_acc.avg,
            val_records.epoch,
        )
    )


def run_epoch(
    model: torch.nn.Module,
    args: Experiment,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    attr_criterion: Optional[List[torch.nn.Module]],
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[Meters, Meters]:

    train_meters = Meters()
    val_meters = Meters()

    if args.multimodel:
        run_multimodel_epoch(
            model,
            optimizer,
            train_loader,
            train_meters,
            criterion,
            attr_criterion,
            args,
            is_training=True,
        )

    elif args.no_img:
        run_epoch_simple(
            model,
            optimizer,
            train_loader,
            train_meters,
            criterion,
            args,
            is_training=True,
        )
    else:
        run_twopart_epoch(
            model,
            optimizer,
            train_loader,
            train_meters,
            criterion,
            attr_criterion,
            args,
            is_training=True,
        )

    with torch.no_grad():
        if args.multimodel:
            run_multimodel_epoch(
                model,
                optimizer,
                val_loader,
                val_meters,
                criterion,
                attr_criterion,
                args,
                is_training=False,
            )
        elif args.no_img:
            run_epoch_simple(
                model,
                optimizer,
                val_loader,
                val_meters,
                criterion,
                args,
                is_training=False,
            )
        else:
            run_twopart_epoch(
                model,
                optimizer,
                val_loader,
                val_meters,
                criterion,
                attr_criterion,
                args,
                is_training=False,
            )

    return train_meters, val_meters


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


def train_multimodel(args) -> None:
    secrets = get_secrets()
    wandb.login(key=secrets["wandb_key"])
    multiple_cfg = dataclasses.replace(
        args,
        multimodel=True,
        n_models=1,
        epochs=1000,
        use_aux=True,
        use_attr=True,
        bottleneck=True,
        normalize_loss=True,
        pretrained=True,
    )
    retrain_dec_cfg = dataclasses.replace(
        multiple_cfg,
        shuffle_models=True,
        freeze_post_models=True,
        reset_post_models=True,
    )
    retrain_enc_cfg = dataclasses.replace(
        multiple_cfg,
        shuffle_models=True,
        freeze_pre_models=True,
        reset_pre_models=True,
    )
    model = Multimodel(multiple_cfg)
    elapsed_epochs = train(model, multiple_cfg)
    train(model, retrain_dec_cfg, init_epoch=elapsed_epochs)
    wandb.finish()


def train_X_to_C(args):
    model = ModelXtoC(
        pretrained=args.pretrained,
        freeze=args.freeze,
        use_aux=args.use_aux,
        num_classes=args.num_classes,
        n_attributes=args.n_attributes,
        expand_dim=args.expand_dim,
        three_class=args.three_class,
    )
    train(model, args)


def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(
        n_class_attr=args.n_class_attr,
        n_attributes=args.n_attributes,
        num_classes=args.num_classes,
        expand_dim=args.expand_dim,
    )
    train(model, args)


def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(
        n_class_attr=args.n_class_attr,
        n_attributes=args.n_attributes,
        num_classes=args.num_classes,
        expand_dim=args.expand_dim,
    )
    train(model, args)


def train_X_to_C_to_y(args):
    model = ModelXtoCtoY(
        n_class_attr=args.n_class_attr,
        pretrained=args.pretrained,
        use_aux=args.use_aux,
        freeze=args.freeze,
        num_classes=args.num_classes,
        n_attributes=args.n_attributes,
        expand_dim=args.expand_dim,
        use_relu=args.use_relu,
        use_sigmoid=args.use_sigmoid,
    )
    train(model, args)


def train_X_to_y(args):
    model = ModelXtoY(
        pretrained=args.pretrained,
        freeze=args.freeze,
        use_aux=args.use_aux,
        num_classes=args.num_classes,
    )
    train(model, args)


def train_X_to_Cy(args):
    model = ModelXtoCY(
        pretrained=args.pretrained,
        freeze=args.freeze,
        num_classes=args.num_classes,
        use_aux=args.use_aux,
        n_attributes=args.n_attributes,
        three_class=args.three_class,
        connect_CY=args.connect_CY,
    )
    train(model, args)


def _save_CUB_result(train_result):
    now_str = datetime.now().strftime(DATETIME_FMT)
    train_result_path = RESULTS_DIR / train_result.tag / now_str / "train-result.pickle"
    if not train_result_path.parent.is_dir():
        train_result_path.parent.mkdir(parents=True)
    with train_result_path.open("wb") as f:
        print(train_result)
        pickle.dump(train_result, f)


if __name__ == "__main__":
    default_args = Experiment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr-sparsity', type=int, default=default_args.attr_sparsity,
                        help='Only use attrs if ndx % sparsity == 0')
    parser.add_argument('--attr-loss-weight', type=float, default=default_args.attr_loss_weight,
                        help='Relative weight of attribute loss')
    new_args = parser.parse_args()
    args = dataclasses.replace(
        default_args,
        attr_loss_weight=new_args.attr_loss_weight,
        attr_sparsity=new_args.attr_sparsity,
        tag="sparsemultimodel" + str(new_args.attr_loss_weight) + "-" + str(new_args.attr_sparsity),
    )
    train_multimodel(args)

    # train_X_to_C(args)
