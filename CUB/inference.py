"""
Evaluate trained models on the official CUB test set
"""
import dataclasses
import os
import sys
import torch
from typing import List, Tuple, Union
import numpy as np
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.dataset import load_data
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES
from CUB.cub_classes import TTI_Config, Meters
from analysis import AverageMeter, multiclass_metric, accuracy, binary_accuracy

K = [1, 3, 5]  # top k class accuracies to compute

@dataclasses.dataclass
class Eval_Meter:
    class_accs: List[AverageMeter]


@dataclasses.dataclass
class Eval_Meter_Acc(Eval_Meter):
    attr_acc_tot: AverageMeter = dataclasses.field(default_factory=AverageMeter)
    attr_accs: List[AverageMeter] = dataclasses.field(default_factory=lambda: [])


@dataclasses.dataclass
class Eval_Output:
    class_labels: np.ndarray
    topk_classes: np.ndarray
    class_logits: np.ndarray
    attr_true_labels: np.ndarray
    attr_pred_outputs: np.ndarray
    attr_pred_sigmoids: np.ndarray
    wrong_idx: np.ndarray


def eval(args: TTI_Config) -> Tuple[Union[Eval_Meter, Eval_Meter_Acc], Eval_Output]:
    """
    Run inference using model (and model2 if bottleneck)
    Returns: (for notebook analysis)
    all_class_labels: flattened list of class labels for each image.
    topk_class_outputs: array of top k class ids predicted for each image. Shape = N_TEST * max(K)
    all_class_outputs: array of all logit outputs for class prediction. shape = N_TEST * N_CLASS
    all_attr_labels: flattened list of ground-truth labels for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs: flatted list of attribute logits (after ReLU/ Sigmoid respectively) predicted for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs_sigmoid: flatted list of attribute logits predicted (after Sigmoid) for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    wrong_idx: image ids where the model got the wrong class prediction (to compare with other models)
    """

    # Load models
    print(args.model_dir)
    model = torch.load(args.model_dir)
    model.eval()

    # Add meters for the overall attr_acc and (optional) for each attr
    meters: Union[Eval_Meter, Eval_Meter_Acc]
    meters = Eval_Meter_Acc(class_accs=[AverageMeter() for _ in range(len(K))])
  
    meters.attr_acc_tot = AverageMeter()

    # Compute acc for each feature individually in addition to the overall accuracy
    if args.feature_group_results:
        meters.attr_accs = [AverageMeter() for _ in range(args.n_attributes)]

    data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + ".pkl")
    loader = load_data([data_dir], args)

    dataset_len = len(loader.dataset)
    n_models = model.args.n_models if args.multimodel else 1
    total_size = dataset_len * n_models

    # Initialize arrays to store outputs
    all_class_labels = np.zeros(total_size, dtype=np.int32)
    all_class_logits = np.zeros((total_size, N_CLASSES), dtype=np.float32) # MUST BE REVERTED!!!!! TESTING ONLY

    top_class_preds = np.zeros(total_size, dtype=np.int32)
    topk_class_labels = np.zeros((total_size, max(K)), dtype=np.int32)
    topk_class_outputs = np.zeros((total_size, max(K)), dtype=np.int32)

    all_attr_labels = np.zeros((total_size, args.n_attributes), dtype=np.int32)
    all_attr_preds = np.zeros((total_size, args.n_attributes), dtype=np.float32)
    all_attr_preds_sigmoid = np.zeros((total_size, args.n_attributes), dtype=np.float32)

    # Run a normal epoch, get outputs and top K class outputs
    n_seen = 0
    for data in loader:
        inputs, class_labels, attr_labels, attr_mask = data

        attr_labels = [i.long() for i in attr_labels]
        attr_labels = torch.stack(attr_labels, dim=1)

        attr_labels = attr_labels.cuda() if torch.cuda.is_available() else attr_labels
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        class_labels = class_labels.cuda() if torch.cuda.is_available() else class_labels
        attr_mask = attr_mask.cuda() if torch.cuda.is_available() else attr_mask

        attr_preds, _, class_preds, _ = model.generate_predictions(inputs, attr_labels, attr_mask, straight_through=True)

        if args.multimodel:
            class_preds = torch.cat(class_preds, dim=0)
            class_labels = class_labels.repeat(model.args.n_models) # shape (batch_size) to (batch_size * n_models)

        class_acc = accuracy(class_preds, class_labels, topk=K)

        for m in range(len(meters.class_accs)):
            meters.class_accs[m].update(class_acc[m], inputs.size(0))
    
        if args.multimodel:
            attr_preds_t = torch.cat([torch.cat(a, dim=1) for a in attr_preds], dim=0)
            attr_labels = attr_labels.repeat(model.args.n_models, 1) # shape (batch_size, N_ATTRIBUTES) to (batch_size * n_models, N_ATTRIBUTES)
        else:
            attr_preds_t = torch.cat(attr_preds, dim=1)
        
        attr_preds_sigmoid = torch.nn.Sigmoid()(attr_preds_t)

        try:
            for i in range(args.n_attributes):
                acc = binary_accuracy(
                    attr_preds_sigmoid[:, i].squeeze(), attr_labels[:, i]
                )
                acc = acc.data.cpu().numpy()
                # acc = accuracy(attr_outputs_sigmoid[i], attr_labels[:, i], topk=(1,))
                meters.attr_acc_tot.update(acc, inputs.size(0))
                # keep track of accuracy of individual attributes
                if args.feature_group_results:
                    meters.attr_accs[i].update(acc, inputs.size(0))
        except:
            import pdb; pdb.set_trace()

        # Store outputs
        n_examples = inputs.size(0)
        if args.multimodel:
            n_examples = n_examples * model.args.n_models

        try:
            all_attr_preds[n_seen:n_seen + n_examples] = attr_preds_t.data.cpu().numpy()
        except ValueError:
            import pdb; pdb.set_trace()
        all_attr_preds_sigmoid[n_seen:n_seen + n_examples] = attr_preds_sigmoid.data.cpu().numpy()
        all_attr_labels[n_seen:n_seen + n_examples] = attr_labels.data.cpu().numpy()

        all_class_labels[n_seen:n_seen + n_examples] = class_labels.data.cpu().numpy()
        all_class_logits[n_seen:n_seen + n_examples] = class_preds.detach().cpu().numpy()

        # Get and store top K class predictions
        _, topk_preds = class_preds.topk(max(K), 1, True, True)
        _, preds = class_preds.topk(1, 1, True, True)

        top_class_preds[n_seen:n_seen + n_examples] = preds.detach().cpu().numpy().squeeze()
        topk_class_outputs[n_seen:n_seen + n_examples] = topk_preds.detach().cpu().numpy()
        topk_class_labels[n_seen:n_seen + n_examples] = class_labels.view(-1, 1).expand_as(preds).cpu().numpy()

        # np.set_printoptions(threshold=sys.maxsize)
        n_seen += n_examples

    wrong_idx = np.where(np.sum(topk_class_outputs == topk_class_labels, axis=1) == 0)[0]

    # Print top K accuracies
    for j in range(len(K)):
        print("Average top %d class accuracy: %.5f" % (K[j], meters.class_accs[j].avg))

    # print some metrics for attribute prediction performance
    assert type(meters) == Eval_Meter_Acc
    print("Average attribute accuracy: %.5f" % meters.attr_acc_tot.avg)
    all_attr_preds_int = all_attr_preds_sigmoid >= 0.5

    # Ignoring for now as not using feature_group_results
    if args.feature_group_results:
        n = len(all_attr_labels)
        all_attr_acc, all_attr_f1 = [], []
        for i in range(args.n_attributes):
            acc_meter = meters.attr_accs[i]
            attr_acc = float(acc_meter.avg)
            attr_preds = [
                all_attr_preds_int[j]
                for j in range(n)
                if j % args.n_attributes == i
            ]
            attr_labels = [
                all_attr_labels[j] for j in range(n) if j % args.n_attributes == i
            ]
            attr_f1 = f1_score(attr_labels, attr_preds)
            all_attr_acc.append(attr_acc)
            all_attr_f1.append(attr_f1)

        bins = np.arange(0, 1.01, 0.1)
        acc_bin_ids = np.digitize(np.array(all_attr_acc) / 100.0, bins)
        acc_counts_per_bin = [
            np.sum(acc_bin_ids == (i + 1)) for i in range(len(bins))
        ]
        f1_bin_ids = np.digitize(np.array(all_attr_f1), bins)
        f1_counts_per_bin = [
            np.sum(f1_bin_ids == (i + 1)) for i in range(len(bins))
        ]
        print("Accuracy bins:")
        print(acc_counts_per_bin)
        print("F1 bins:")
        print(f1_counts_per_bin)
        np.savetxt(os.path.join(args.log_dir, "concepts.txt"), f1_counts_per_bin)

    balanced_acc, report = multiclass_metric(all_attr_preds_int.flatten(), all_attr_labels.flatten())
    f1 = f1_score(all_attr_labels.flatten(), all_attr_preds_int.flatten())
    print(
        "Total 1's predicted:",
        sum(np.array(all_attr_preds_sigmoid) >= 0.5)
        / len(all_attr_preds_sigmoid),
    )
    print("Avg attribute balanced acc: %.5f" % (balanced_acc))
    print("Avg attribute F1 score: %.5f" % f1)
    print(report + "\n")

    output = Eval_Output(
        class_labels=all_class_labels,
        topk_classes=topk_class_outputs,
        class_logits=all_class_logits,
        attr_true_labels=all_attr_labels,
        attr_pred_outputs=all_attr_preds,
        attr_pred_sigmoids=all_attr_preds_sigmoid,
        wrong_idx=wrong_idx,
    )

    return output