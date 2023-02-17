#!/usr/bin/env python
# coding: utf-8
import os
import sys
import torch
import pickle
import random

from collections import defaultdict as ddict

from typing import List, Tuple, Dict, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.config import N_CLASSES, N_ATTRIBUTES
from CUB.cub_classes import TTI_Config, TTI_Output
from CUB.inference import eval

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
        Takes the attribute mask, and attributes.txt which has an attribute 
        label on each line eg "198 has_belly_color::blue\n".

        Returns a dict which contains the groups of attributes and the attr_ids which they contain.
        
        For example, the has_bill_shape has 9 sub attributes but after masking only 4 attrs remain, 
        so attr_group_dict[0] = [0, 1, 2, 3]
    """
    attr_group_dict = dict()
    curr_group_idx = 0
    attribute_loc = "CUB_200_2011/attributes/attributes.txt"

    with open(attribute_loc, "r") as f:
        all_lines = f.readlines()
        line0 = all_lines[0]
        prefix = line0.split()[1][:10] # first 10 chars of the attribute group name
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
    
    return attr_group_dict
    

def run_tti(args) -> List[Tuple[int, float]]:
    """
    Returns pairs of (n, score) where n is a number of attributes
    and score is the accuracy when n attributes are intervened on
    """

    train_data = pickle.load(open(os.path.join(args.data_dir_raw, "train.pkl"), "rb"))

    # Make a filter for attributes that are not common enough, as list of IDs to keep
    attr_mask = build_mask(train_data, min_count=10) 

    test_data = pickle.load(open(os.path.join(args.data_dir_raw, "test.pkl"), "rb"))

    # Build numpy arrays of the labels for the attributes we are using
    raw_attr_labels = np.zeros((len(test_data), len(attr_mask)))
    raw_attr_uncertanties = np.zeros((len(test_data), len(attr_mask)))

    for ndx, d in enumerate(test_data):
        raw_attr_labels[ndx] = np.array(d["attribute_label"])[attr_mask]
        raw_attr_uncertanties[ndx] = np.array(d["attribute_certainty"])[attr_mask]
    
    # Build a dict which contains the groups of attributes and the attr_ids which they contain after masking
    attr_group_dict = build_dict_from_mask(attr_mask=attr_mask)

    # Run one epoch, get lots of detail about performance
    eval_output = eval(args)
    class_outputs = eval_output.topk_classes[:, 0]
    attr_binary_outputs = np.rint(eval_output.attr_pred_sigmoids).astype(int)  # RoundINT
    
    # Get 5, 95 percentiles for how much each attribute was predicted to be true [0,1]
    ptl_5, ptl_95 = dict(), dict()
    
    if args.flat_intervene:
        ptl_5 = ddict(lambda: args.intervene_vals[0])
        ptl_95 = ddict(lambda: args.intervene_vals[1])

    else:
        for attr_idx in range(args.n_attributes):
            ptl_5[attr_idx] = np.percentile(eval_output.attr_pred_outputs[:, attr_idx], 5)
            ptl_95[attr_idx] = np.percentile(eval_output.attr_pred_outputs[:, attr_idx], 95)
        
    
    # Creating the correct output array, where 'correct' is the 5th percentile of the attribute if false, and 95th percentile if true
    correct_attr_outputs = np.zeros_like(eval_output.attr_pred_outputs)
    if args.replace_class:
        for attr_idx in range(args.n_attributes):
            correct_attr_outputs[:, attr_idx] = np.where(
                eval_output.attr_true_labels[:, attr_idx] == 0, ptl_5[attr_idx], ptl_95[attr_idx]
            )
    else:
        for attr_idx in range(args.n_attributes):
            correct_attr_outputs[:, attr_idx] = np.where(
                raw_attr_labels[:, attr_idx] == 0, ptl_5[attr_idx], ptl_95[attr_idx]
            )

    # Get main model and attr -> label model
    model = torch.load(args.model_dir)
    if args.multimodel:
         # NOTE: these are numpy arrays and np.repeat is different to torch.repeat()
        raw_attr_labels = raw_attr_labels.repeat(model.args.n_models, axis=0)
        raw_attr_uncertanties = raw_attr_uncertanties.repeat(model.args.n_models, axis=0)

    # Check that number of attributes matches between the 'raw' data and the class aggregated data
    assert len(raw_attr_labels) == len(
        eval_output.attr_true_labels
    ), "len(instance_attr_labels): %d, len(eval_output.attr_labels): %d" % (
        len(raw_attr_labels),
        len(eval_output.attr_true_labels),
    )
    assert len(raw_attr_uncertanties) == len(
        eval_output.attr_true_labels
    ), "len(uncertainty_attr_labels): %d, len(eval_output.attr_labels): %d" % (
        len(raw_attr_uncertanties),
        len(eval_output.attr_true_labels),
    )

    # Print percentiles for the raw and corrected attr outputs
    # to check that they are from similar distributions
    print("Raw attr outputs:")
    print([np.percentile(eval_output.attr_pred_outputs, x) for x in range(0, 101, 10)])
    print("Corrected attr outputs:")
    print([np.percentile(correct_attr_outputs, x) for x in range(0, 101, 10)])



    results = []
    for n_replace in range(args.n_groups + 1):
        all_class_acc = []
        if args.multimodel:
            n_trials = args.n_trials * model.args.n_models
        else:
            n_trials = args.n_trials

        for ndx in range(n_trials):
            # Array of attr predictions, will be modified towards ground truth
            updated_attrs = np.array(eval_output.attr_pred_outputs[:])
            if n_replace > 0:
                for img_id in range(len(eval_output.class_labels)):
                    # Get a list of all attrs (in the flattened list) that we will intervene on for this img
                    replace_idxs = []
                    group_replace_idx = list(
                        random.sample(list(range(args.n_groups)), n_replace)
                    )
                    for i in group_replace_idx:
                        replace_idxs.extend(attr_group_dict[i])
                    
                    updated_attrs[img_id, replace_idxs] = correct_attr_outputs[img_id, replace_idxs]
            

            stage2_inputs = torch.from_numpy(updated_attrs).cuda()

            if args.multimodel: # if multimodel, we need to reshape the inputs to be (n_models, n_imgs, n_attrs)
                stage2_inputs = stage2_inputs.reshape(model.args.n_models, -1, args.n_attributes)
                class_outputs = []
                for i, post_model in enumerate(model.post_models):
                    post_model.eval()
                    class_outputs.append(post_model(stage2_inputs[i]))
                
                class_outputs = torch.cat(class_outputs, dim=0)
            else:
                model_use = model.second_model
                if model.args.exp == "Independent":
                    model_use = torch.nn.Sequential(torch.nn.Sigmoid(), model_use)
                model_use.eval()
                class_outputs = model_use(stage2_inputs)

            _, predictions = class_outputs.topk(k=1, dim=1) # topk returns a tuple of (values, indices)
            predictions = predictions.data.cpu().numpy().squeeze()
            class_acc = np.mean(
                np.array(predictions) == np.array(eval_output.class_labels)
            )
            all_class_acc.append(class_acc * 100) # convert to percent

        # changed from max to sum - not sure why max would be appropriate
        acc = sum(all_class_acc) / len(all_class_acc)

        print(n_replace, acc)
        results.append((n_replace, acc))
    return results


def graph_tti_output(
    tti_output: List[Tuple[int, float]], 
    save_dir: Optional[str] = None, 
    show: bool = True, 
    label: Optional[str]=None, 
    return_fig: bool = False
    ) -> Optional[Figure]:
    """Graph the output of a TTI run"""
    try:
        n_replace, acc = zip(*tti_output)
    except:
        breakpoint()
    plt.plot(n_replace, acc, label=label)
    plt.xlabel("Number of groups replaced")
    plt.ylabel("Accuracy")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "tti_results.png"))
    if show:
        plt.show()
    if return_fig:
        return plt.gcf()

def graph_multi_tti_output(tti_outputs: List[TTI_Output], save_dir: Optional[str] = None):
    """Graph the output of multiple TTI runs"""
    for i, tti_output in enumerate(tti_outputs):
        return_fig = i == len(tti_outputs) - 1
        graph_tti_output(
            tti_output=tti_output,
            return_fig=return_fig, 
        )

    plt.legend()
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig("images/tti_results.png")
