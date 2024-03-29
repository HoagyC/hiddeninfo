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

from CUB.dataset import load_data, CUBDataset
from CUB.cub_classes import Meters, N_CLASSES, N_ATTRIBUTES, Experiment
from CUB.analysis import AverageMeter, multiclass_metric, accuracy, binary_accuracy

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


def run_eval(args: Experiment) -> Eval_Output:
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
    print(args.tti_model_dir)
    model = torch.load(args.tti_model_dir)
    model.eval()

    # Add meters for the overall attr_acc and (optional) for each attr
    meters: Union[Eval_Meter, Eval_Meter_Acc]
    meters = Eval_Meter_Acc(class_accs=[AverageMeter() for _ in range(len(K))])
  
    meters.attr_acc_tot = AverageMeter()

    # Compute acc for each feature individually in addition to the overall accuracy
    if args.feature_group_results:
        meters.attr_accs = [AverageMeter() for _ in range(args.n_attributes)]

    data_dir = os.path.join(args.base_dir, args.data_dir, args.tti_eval_data + ".pkl")
    loader = load_data([data_dir], args)

    dataset: CUBDataset = loader.dataset # type: ignore 
    dataset_len = len(dataset) 
    n_models = model.args.n_models if args.multimodel else 1
    total_size = dataset_len * n_models

    # Initialize arrays to store outputs
    all_class_labels = np.zeros((n_models, dataset_len), dtype=np.int32)
    all_class_logits = np.zeros((n_models, dataset_len, N_CLASSES), dtype=np.float32) # MUST BE REVERTED!!!!! TESTING ONLY

    top_class_preds = np.zeros((n_models, dataset_len), dtype=np.int32)
    topk_class_labels = np.zeros((n_models, dataset_len, max(K)), dtype=np.int32)
    topk_class_outputs = np.zeros((n_models, dataset_len, max(K)), dtype=np.int32)

    all_attr_labels = np.zeros((n_models, dataset_len, N_ATTRIBUTES), dtype=np.int32)
    all_attr_preds = np.zeros((n_models, dataset_len, args.n_attributes), dtype=np.float32)
    all_attr_preds_sigmoid = np.zeros((n_models, dataset_len, args.n_attributes), dtype=np.float32)

    # Run a normal epoch, get outputs and top K class outputs
    n_seen = 0
    for data in loader:
        inputs, class_labels, attr_labels, attr_mask = data

        attr_labels = [i.long() for i in attr_labels]
        attr_labels = torch.stack(attr_labels, dim=1)

        attr_labels = attr_labels.cuda() if torch.cuda.is_available() else attr_labels
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        class_labels = class_labels.cuda() if torch.cuda.is_available() else class_labels
        attr_mask = torch.ones_like(attr_mask).cuda() if torch.cuda.is_available() else torch.ones_like(attr_mask)

        attr_preds, _, class_preds, _ = model.generate_predictions(inputs, attr_labels, attr_mask, straight_through=True)
        
        class_labels = class_labels.repeat(model.args.n_models, 1) # shape batch_size to n_models * batch_size
        attr_labels = attr_labels.repeat(model.args.n_models, 1, 1) # shape batch_size * n_attrs to n_models * batch_size * n_attrs

        class_acc = accuracy(class_preds.reshape(-1, N_CLASSES), class_labels.reshape(-1), topk=K)

        for m in range(len(meters.class_accs)):
            meters.class_accs[m].update(class_acc[m], inputs.size(0))
    
        assert attr_preds.shape == (model.args.n_models, inputs.shape[0], args.n_attributes)    
        attr_preds_sigmoid = torch.nn.Sigmoid()(attr_preds)

        for i in range(N_ATTRIBUTES):
            acc = binary_accuracy(
                attr_preds_sigmoid[:, :, i].reshape(-1), attr_labels[:, :, i].reshape(-1)
            )
            acc = acc.data.cpu().numpy()
            # acc = accuracy(attr_outputs_sigmoid[i], attr_labels[:, i], topk=(1,))
            meters.attr_acc_tot.update(acc, inputs.size(0))
            # keep track of accuracy of individual attributes
            if args.feature_group_results:
                meters.attr_accs[i].update(acc, inputs.size(0))
            
        # Store outputs
        n_examples = inputs.size(0)

        all_attr_preds[:, n_seen:n_seen + n_examples] = attr_preds.data.cpu().numpy()
        all_attr_preds_sigmoid[:, n_seen:n_seen + n_examples] = attr_preds_sigmoid.data.cpu().numpy()
        all_attr_labels[:, n_seen:n_seen + n_examples] = attr_labels.data.cpu().numpy()

        all_class_labels[:, n_seen:n_seen + n_examples] = class_labels.data.cpu().numpy()
        all_class_logits[:, n_seen:n_seen + n_examples] = class_preds.detach().cpu().numpy()

        # Get and store top K class predictions
        topk_preds = class_preds.reshape(-1, N_CLASSES).topk(max(K), 1, True, True)[1].reshape(-1, inputs.size(0), max(K))
        preds = class_preds.reshape(-1, N_CLASSES).topk(1, 1, True, True)[1].reshape(-1, inputs.size(0))

        top_class_preds[:, n_seen:n_seen + n_examples] = preds.detach().cpu().numpy()
        topk_class_outputs[:, n_seen:n_seen + n_examples] = topk_preds.detach().cpu().numpy()
        topk_class_labels[:,n_seen:n_seen + n_examples] = class_labels.unsqueeze(2).expand_as(topk_preds).cpu().numpy()

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

    balanced_acc, report = multiclass_metric(all_attr_preds_int[:, :, :N_ATTRIBUTES].flatten(), all_attr_labels.flatten())
    f1 = f1_score(all_attr_labels.flatten(), all_attr_preds_int[:, :, :N_ATTRIBUTES].flatten())
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