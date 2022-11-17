"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import argparse
import dataclasses
import pdb
import os
import random
import sys

from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np
import wandb

from CUB.analysis import Logger, AverageMeter, accuracy, binary_accuracy
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


N_ATTRIBUTES = 312
N_CLASSES = 200
MIN_LR = 1e-04
BASE_DIR = "/root/hiddeninfo"
LR_DECAY_SIZE = 0.1


def run_epoch_simple(
    model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training
):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            # inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).cuda()
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda()
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()
            optimizer.step()  # optimizer step to update parameters
    return loss_meter, acc_meter


def run_epoch(
    model,
    optimizer,
    loader,
    loss_meter,
    acc_meter,
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
        else:
            inputs, labels, attr_labels = data
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

        outputs = model(inputs_var)
        losses = []
        out_start = 0
        if not args.bottleneck:
            loss_main = criterion(outputs[0], labels_var)
            losses.append(loss_main)
            out_start = 1
        if (
            attr_criterion is not None and args.attr_loss_weight > 0
        ):  # X -> A, cotraining, end2end
            for i in range(len(attr_criterion)):
                value = outputs[i + out_start].squeeze().type(torch.cuda.FloatTensor)
                target = attr_labels_var[:, i]
                attr_loss = attr_criterion[i](value, target)
                losses.append(args.attr_loss_weight * attr_loss)

        if args.bottleneck:  # attribute accuracy
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(
                outputs[0], labels, topk=(1,)
            )  # only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

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
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return loss_meter, acc_meter


def run_multimodel_epoch(
    model,
    optimizer,
    loader,
    loss_meter,
    concept_acc_meter,
    label_acc_meter,
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
        if args.shuffle_post_models:
            post_model_ndx = random.randint(0, len(model.pre_models) - 1)
        else:
            post_model_ndx = pre_model_ndx

        pre_model = model.pre_models[pre_model_ndx]
        post_model = model.post_models[post_model_ndx]

        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
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

        concepts = pre_model(inputs)
        concepts_t = torch.cat(concepts, dim=1)
        output_labels = post_model(concepts_t)

        losses = []
        # Loss main is for the main task label (always the first output)
        loss_main = 1.0 * criterion(output_labels, target=labels)
        losses.append(loss_main)

        # Adding losses separately for the different classes
        for i in range(len(attr_criterion)):
            value = torch.masked_select(
                concepts[i].type(torch.cuda.FloatTensor).squeeze(), attr_mask_bin
            )
            target = torch.masked_select(attr_labels[:, i], attr_mask_bin)

            attr_loss = attr_criterion[i](value, target)
            losses.append(args.attr_loss_weight * attr_loss / args.n_attributes)

        # Calculating attribute accuracy
        sigmoid_outputs = torch.nn.Sigmoid()(concepts_t)
        concept_acc = binary_accuracy(sigmoid_outputs, attr_labels)
        concept_acc_meter.update(concept_acc.data.cpu().numpy(), inputs.size(0))

        label_acc = accuracy(
            output_labels, labels, topk=(1,)
        )  # only care about class prediction accuracy
        label_acc_meter.update(label_acc[0], inputs.size(0))

        total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return loss_meter, concept_acc_meter, label_acc_meter


def train(model, args, split_models=False):
    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, "train.pkl")
        if args.weighted_loss == "multiple":
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)

    if os.path.exists(args.log_dir):  # job restarted by cluster
        for f in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f))
    else:
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, "log.txt"))
    logger.write(str(args) + "\n")
    logger.write(str(imbalance) + "\n")
    logger.flush()

    model = model.cuda()
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
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=0.1
    )
    stop_epoch = (
        int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    )
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, args.data_dir, "train.pkl")
    val_data_path = train_data_path.replace("train.pkl", "val.pkl")
    logger.write("train data path: %s\n" % train_data_path)

    if args.ckpt:  # retraining
        train_loader = load_data(
            [train_data_path, val_data_path],
            args.use_attr,
            args.no_img,
            args.batch_size,
            args.uncertain_labels,
            image_dir=args.image_dir,
            n_class_attr=args.n_class_attr,
            resampling=args.resampling,
            attr_sparsity=args.attr_sparsity,
        )
        val_loader = None
    else:
        train_loader = load_data(
            [train_data_path],
            args.use_attr,
            args.no_img,
            args.batch_size,
            args.uncertain_labels,
            image_dir=args.image_dir,
            n_class_attr=args.n_class_attr,
            resampling=args.resampling,
            attr_sparsity=args.attr_sparsity,
        )
        val_loader = load_data(
            [val_data_path],
            args.use_attr,
            args.no_img,
            args.batch_size,
            image_dir=args.image_dir,
            n_class_attr=args.n_class_attr,
            attr_sparsity=args.attr_sparsity,
        )

    best_val_epoch = -1
    best_val_loss = float("inf")
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_conc_acc_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        if args.multimodel:
            (
                train_loss_meter,
                train_conc_acc_meter,
                train_acc_meter,
            ) = run_multimodel_epoch(
                model,
                optimizer,
                train_loader,
                train_loss_meter,
                train_conc_acc_meter,
                train_acc_meter,
                criterion,
                attr_criterion,
                args,
                is_training=True,
            )

        elif args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(
                model,
                optimizer,
                train_loader,
                train_loss_meter,
                train_acc_meter,
                criterion,
                args,
                is_training=True,
            )
        else:
            train_loss_meter, train_acc_meter = run_epoch(
                model,
                optimizer,
                train_loader,
                train_loss_meter,
                train_acc_meter,
                criterion,
                attr_criterion,
                args,
                is_training=True,
            )

        if not args.ckpt:  # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
            val_conc_acc_meter = AverageMeter()

            with torch.no_grad():
                if args.multimodel:
                    (
                        val_loss_meter,
                        val_conc_acc_meter,
                        val_acc_meter,
                    ) = run_multimodel_epoch(
                        model,
                        optimizer,
                        val_loader,
                        val_loss_meter,
                        val_conc_acc_meter,
                        val_acc_meter,
                        criterion,
                        attr_criterion,
                        args,
                        is_training=False,
                    )
                elif args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(
                        model,
                        optimizer,
                        val_loader,
                        val_loss_meter,
                        val_acc_meter,
                        criterion,
                        args,
                        is_training=False,
                    )
                else:
                    val_loss_meter, val_acc_meter = run_epoch(
                        model,
                        optimizer,
                        val_loader,
                        val_loss_meter,
                        val_acc_meter,
                        criterion,
                        attr_criterion,
                        args,
                        is_training=False,
                    )

        else:  # retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write("New model best model at epoch %d\n" % epoch)
            torch.save(
                model, os.path.join(args.log_dir, "best_model_%d.pth" % args.seed)
            )
            # if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            #    break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg

        metrics_dict = {
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "train_acc": train_acc_meter.avg,
            "val_loss": val_loss_avg,
            "val_acc": val_acc_meter.avg,
            "best_val_epoch": best_val_epoch,
            "concept_train_acc": train_conc_acc_meter.avg,
            "concept_val_acc": val_conc_acc_meter.avg,
        }

        wandb.log(metrics_dict)
        logger.write(
            "Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t"
            "Val loss: %.4f\tVal acc: %.4f\t"
            "Best val epoch: %d\n"
            % (
                epoch,
                train_loss_avg,
                train_acc_meter.avg,
                val_loss_avg,
                val_acc_meter.avg,
                best_val_epoch,
            )
        )

        logger.flush()

        if epoch <= stop_epoch:
            scheduler.step(epoch)  # scheduler step to update lr at the end of epoch
        # inspect lr
        if epoch % 10 == 0:
            print("Current lr:", scheduler.get_lr())

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break


def train_multimodel(args):
    model = Multimodel(args)
    train(model, args)


def train_X_to_C(args):
    model = ModelXtoC(
        pretrained=args.pretrained,
        freeze=args.freeze,
        num_classes=N_CLASSES,
        n_attributes=args.n_attributes,
        expand_dim=args.expand_dim,
        three_class=args.three_class,
    )
    train(model, args)


def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(
        n_class_attr=args.n_class_attr,
        n_attributes=args.n_attributes,
        num_classes=N_CLASSES,
        expand_dim=args.expand_dim,
    )
    train(model, args)


def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(
        n_class_attr=args.n_class_attr,
        n_attributes=args.n_attributes,
        num_classes=N_CLASSES,
        expand_dim=args.expand_dim,
    )
    train(model, args)


def train_X_to_C_to_y(args):
    model = ModelXtoCtoY(
        n_class_attr=args.n_class_attr,
        pretrained=args.pretrained,
        freeze=args.freeze,
        num_classes=N_CLASSES,
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
        num_classes=N_CLASSES,
    )
    train(model, args)


def train_X_to_Cy(args):
    model = ModelXtoCY(
        pretrained=args.pretrained,
        freeze=args.freeze,
        num_classes=N_CLASSES,
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


@dataclasses.dataclass
class Experiment:
    tag: str = "basic"
    dataset: str = "CUB"
    exp: str = "multimodel"
    multimodel: bool = True
    seed: int = 0
    log_dir: str = "out"
    data_dir: str = "CUB_processed"
    image_dir: str = "images"
    end2end: bool = True
    optimizer: str = "SGD"
    ckpt: bool = False
    scheduler_step: int = 1000
    normalize_loss: bool = True
    use_relu: bool = True
    use_sigmoid: bool = False
    connect_CY: bool = False
    resampling: bool = False
    batch_size: int = 32
    epochs: int = 100
    save_step: int = 10
    lr: float = 1e-03
    weight_decay: float = 5e-5
    pretrained: bool = True
    freeze: bool = False
    use_attr: bool = True
    attr_loss_weight: float = 1.0
    no_img: bool = False
    bottleneck: bool = True
    weighted_loss: bool = False
    uncertain_labels: bool = True
    shuffle_post_models: bool = False
    n_models: int = 1
    n_attributes: int = N_ATTRIBUTES
    num_classes: int = N_CLASSES
    expand_dim: int = 500
    n_class_attr: int = 2
    attr_sparsity: int = 4
    three_class: int = (
        n_class_attr == 3
    )  # predict notvisible as a third class instead of binary


if __name__ == "__main__":
    args = Experiment()

    wandb.init(project="distill_CUB", config=args.__dict__)
    train_multimodel(args)

    # args = parse_arguments(None)[0]
    # print(args)
    # train_X_to_C(args)
