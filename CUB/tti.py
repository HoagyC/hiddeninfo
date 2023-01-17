#!/usr/bin/env python
# coding: utf-8
import argparse
import dataclasses
import os
import sys
import torch
import pickle
import random

from collections import defaultdict

from typing import List, Tuple, Dict, Optional

import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# from CUB.inference import *
from CUB.config import N_CLASSES, N_ATTRIBUTES
from CUB.cub_utils import make_attr_id_to_name_dict, make_class_id_to_folder_dict
from CUB.cub_classes import TTI_Config, TTI_Output
from CUB.inference import eval

replace_cached: List = []

def get_stage2_pred(a_hat, model):
    """Takes intermediate representation and predict class outputs, to see how they change when intervening."""
    stage2_inputs = torch.from_numpy(np.array(a_hat)).cuda().float()
    class_outputs = model(stage2_inputs)
    class_outputs = torch.nn.Softmax()(class_outputs)
    return class_outputs.data.cpu().numpy()


def build_mask(data: List[Dict], min_count: int = 10) -> np.ndarray:
    # Count the number of times each attribute is known to be true or false for each class
    class_attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES, 2))
    for d in data:
        class_label = d["class_label"]
        certainties = d["attribute_certainty"]
        for attr_idx, a in enumerate(d["attribute_label"]):
            if a == 0 and certainties[attr_idx] == 1:  # not visible
                continue
            class_attr_count[class_label][attr_idx][a] += 1

    # Get those class/attribute pairs where more common to be true/false, treating equal as true
    class_attr_min_label = np.argmin(class_attr_count, axis=2)
    class_attr_max_label = np.argmax(class_attr_count, axis=2)
    # If no. T/F are equal, argmax == argmin == 0
    equal_count = np.where(class_attr_min_label == class_attr_max_label)
    class_attr_max_label[equal_count] = 1

    # Get number of classes where the attribute is more common than not, select for at least min_class_count
    attr_class_count = np.sum(class_attr_max_label, axis=0)
    mask = np.where(attr_class_count >= min_count)[0]
    return mask



def build_dict_from_mask(attr_mask: np.ndarray) -> Dict: 
    """
        Generate a dict which contains the groups of attributes and the attr_ids which they contain
        attributes.txt has a attribute label on each line eg "198 has_belly_color::blue"
        eg has_bill_shape has 9 sub attributes, and attr_group_dict[0] = list(range(1, 10))
    """
    attr_group_dict = dict()
    curr_group_idx = 0
    attribute_loc = "CUB_200_2011/attributes/attributes.txt"

    with open(attribute_loc, "r") as f:
        all_lines = f.readlines()
        line0 = all_lines[0]
        prefix = line0.split()[1][:10] # first 10 chars of the attribute label
        attr_group_dict[curr_group_idx] = [0]
        for i, line in enumerate(all_lines[1:]):
            curr = line.split()[1][:10]
            if curr != prefix:
                curr_group_idx += 1
                prefix = curr
                attr_group_dict[curr_group_idx] = [i + 1]
            else:
                attr_group_dict[curr_group_idx].append(i + 1)

    # Removing attrs that are screened off by the mask
    for group_id, attr_ids in attr_group_dict.items():
        new_attr_ids = []
        for attr_id in attr_ids:
            if attr_id in attr_mask:
                new_attr_ids.append(attr_id)
        attr_group_dict[group_id] = new_attr_ids

    # Switching to using ids only amongst attributes that are actually used (not masked)
    total_so_far = 0
    for group_id, attr_ids in attr_group_dict.items():
        class_attr_ids = list(range(total_so_far, total_so_far + len(attr_ids)))
        total_so_far += len(attr_ids)
        attr_group_dict[group_id] = class_attr_ids

    import pdb; pdb.set_trace()
    
    return attr_group_dict
    

def run_tti(args) -> List[Tuple[int, float]]:
    """
    Returns pairs of (n, score) where n is a number of attribute
    and score is the accuracy when n attributes are intervened on
    """
    train_data = pickle.load(open(os.path.join(args.data_dir2, "train.pkl"), "rb"))

    # Make a filter for attributes that are not common enough, as list of IDs to keep
    attr_mask = build_mask(train_data, min_count=10) 

    # Build 2D lists of attributes and certainties
    test_instance_attr_labels, test_uncertainty_attr_labels = [], []
    test_data = pickle.load(open(os.path.join(args.data_dir2, "test.pkl"), "rb"))
    for d in test_data:
        test_instance_attr_labels.extend(list(np.array(d["attribute_label"])[attr_mask]))
        test_uncertainty_attr_labels.extend(list(np.array(d["attribute_certainty"])[attr_mask]))
    
    # Build a dict which contains the groups of attributes and the attr_ids which they contain after masking
    attr_group_dict = build_dict_from_mask(attr_mask=attr_mask)

    # Run one epoch, get lots of detail about performance
    _, eval_output = eval(args)
    class_outputs = eval_output.topk_classes[:, 0]
    attr_binary_outputs = np.rint(eval_output.attr_pred_sigmoids).astype(int)  # RoundINT

    # Get 5, 95 percentiles for how much each attribute was predicted to be true [0,1]
    preds_by_attr = defaultdict(list)
    ptl_5, ptl_95 = dict(), dict()
    for i, val in enumerate(eval_output.attr_pred_outputs):
        attr_idx = i % args.n_attributes
        preds_by_attr[attr_idx].append(val)

    for attr_idx in range(args.n_attributes):
        preds = preds_by_attr[attr_idx]
        ptl_5[attr_idx] = np.percentile(preds, 5)
        ptl_95[attr_idx] = np.percentile(preds, 95)

    # Get main model and attr -> label model
    model = torch.load(args.model_dir)
    if args.model_dir2:
        model2 = torch.load(args.model_dir2)
    elif args.multimodel:
        model2 = model.post_models
    else:  # end2end, split model into 2
        all_mods = list(model.modules())
        # model = ListModule(all_mods[:-1])
        model2 = all_mods[-1]  # last fully connected layer

    results = []
    for n_replace in list(range(args.n_groups + 1)):
        # Check that number of attributes matches between the 'raw' data and the class aggregated data
        assert len(test_instance_attr_labels) == len(
            eval_output.attr_labels
        ), "len(instance_attr_labels): %d, len(eval_output.attr_labels): %d" % (
            len(test_instance_attr_labels),
            len(eval_output.attr_labels),
        )
        assert len(test_uncertainty_attr_labels) == len(
            eval_output.attr_labels
        ), "len(uncertainty_attr_labels): %d, len(eval_output.attr_labels): %d" % (
            len(test_uncertainty_attr_labels),
            len(eval_output.attr_labels),
        )

        if args.class_level:
            replace_val = "class_level"
        else:
            replace_val = "instance_level"

        all_class_acc = []

        if args.multimodel:
            n_trials = args.n_trials * len(model2)
        else:
            n_trials = args.n_trials

        for ndx in range(n_trials):
            b_attr_new = np.array(eval_output.attr_pred_outputs[:])

            def replace_random(attr_preds):
                replace_idx = []
                group_replace_idx = list(
                    random.sample(list(range(args.n_groups)), n_replace)
                )
                for i in group_replace_idx:
                    replace_idx.extend(attr_group_dict[i])
                return replace_idx

            # Following paper, will only use random intervention on CUB for now
            replace_fn = lambda attr_preds: replace_random(attr_preds)

            # List of attr_ids that have been changed in terms of the big 1D list
            attr_replace_idxs: List = []
            all_attr_ids = []  # list of where attrs have been replaced

            # Intervene on 1, then 2, etc, so caching which to intervene on
            # stores the attrs that have been changed, by the attr_id, not the longID
            global replace_cached
            if n_replace == 1:
                replace_cached = []

            for img_id in range(len(eval_output.class_labels)):
                # Get just the attr outputs for the current img
                attr_preds = eval_output.attr_pred_outputs[
                    img_id * args.n_attributes : (img_id + 1) * args.n_attributes
                ]
                attr_preds_sigmoid = eval_output.attr_pred_sigmoids[
                    img_id * args.n_attributes : (img_id + 1) * args.n_attributes
                ]
                # Get a list of all attrs (in the flattened list) that we will intervene on
                replace_idx = replace_fn(attr_preds)
                all_attr_ids.extend(replace_idx)
                attr_replace_idxs.extend(np.array(replace_idx) + img_id * args.n_attributes)

            replace_cached = all_attr_ids
            pred_vals = attr_binary_outputs[attr_replace_idxs]
            true_vals = np.array(eval_output.attr_labels)[attr_replace_idxs]

            # instance has the original attrs whereas b_attr has attrs averaged at the class level
            if replace_val == "class_level":
                b_attr_new[attr_replace_idxs] = np.array(eval_output.attr_labels)[
                    attr_replace_idxs
                ]
            else:
                b_attr_new[attr_replace_idxs] = np.array(test_instance_attr_labels)[
                    attr_replace_idxs
                ]

            # Zeroing out invisible attrs in the new attr array
            if args.use_invisible:
                # [0] because np.where returns a tuple with one element for each dimension in the array
                not_visible_idx = np.where(np.array(test_uncertainty_attr_labels) == 1)[0]
                for idx in attr_replace_idxs:
                    if idx in not_visible_idx:
                        b_attr_new[idx] = 0

            if args.use_relu or not args.use_sigmoid:  # replace with percentile values
                binary_vals = b_attr_new[attr_replace_idxs]
                for j, replace_idx in enumerate(attr_replace_idxs):
                    attr_idx = replace_idx % args.n_attributes
                    b_attr_new[replace_idx] = (1 - binary_vals[j]) * ptl_5[
                        attr_idx
                    ] + binary_vals[j] * ptl_95[attr_idx]

            # Evaluate the model on the new attributes

            if args.multimodel:
                model_use = model2[ndx % n_trials]
            else:
                model_use = model2
            model_use.eval()

            b_attr_new = b_attr_new.reshape(-1, args.n_attributes)
            stage2_inputs = torch.from_numpy(np.array(b_attr_new)).cuda()

            class_outputs = model_use(stage2_inputs)
            _, predictions = class_outputs.topk(k=1, dim=1) # returns top vals and indices
            predictions = predictions.data.cpu().numpy().squeeze()
            class_acc = np.mean(
                np.array(predictions) == np.array(eval_output.class_labels)
            )
            all_class_acc.append(class_acc * 100)

        # changing from max to sum - not sure why max would be appropriate
        acc = sum(all_class_acc) / len(all_class_acc)

        print(n_replace, acc)
        results.append((n_replace, acc))
    return results


ind_tti_args = TTI_Config(
    model_dirs=["out/ind_XtoC/20221130-150657/final_model.pth"],
    model_dirs2=["out/ind_CtoY/20221130-194327/final_model.pth"],
    use_attr=True,
    bottleneck=True,
    n_trials=5,
    use_invisible=True,
    class_level=True,
    data_dir2="CUB_processed",
    use_sigmoid=True,
    log_dir="TTI_ind",
)

def graph_tti_output(tti_output: List[Tuple[int, float]], save_dir: Optional[str] = None, show: bool = True, label: Optional[str]=None) -> None:
    """Graph the output of a TTI run"""
    n_replace, acc = zip(*tti_output)
    plt.plot(n_replace, acc, label=label)
    plt.xlabel("Number of groups replaced")
    plt.ylabel("Accuracy")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "tti_results.png"))
    if show:
        plt.show()

def graph_multi_tti_output(tti_outputs: List[TTI_Output], save_dir: Optional[str] = None):
    """Graph the output of multiple TTI runs"""
    for tti_o in tti_outputs:
        graph_tti_output(
            tti_output=tti_o.result,
            show=False, 
            label=f"Coef: {tti_o.coef}, Sparsity: {tti_o.sparsity}, Run: {tti_o.model_name}"
        )

    plt.legend()
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig("images/tti_results.png")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = ind_tti_args  # Set config for how to run TTI

    all_values = []
    values: List
    for i, model_dir in enumerate(args.model_dirs):
        print("----------")
        args.model_dir = model_dir
        args.model_dir2 = args.model_dirs2[i] if args.model_dirs2 else None
        all_values.append(run_tti(args))

    output_string = ""
    no_intervention_groups = np.array(all_values[0])[:, 0]
    values = [sum(np.array(values)[:, 1]) / len(all_values) for value in all_values]
    for no_intervention_group, value in zip(no_intervention_groups, values):
        output_string += "%.4f %.4f\n" % (no_intervention_group, value)
    print(output_string)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    output = open(os.path.join(args.log_dir, "results.txt"), "w")
    output.write(output_string)

