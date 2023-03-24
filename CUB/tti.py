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

from CUB.cub_classes import TTI_Config, TTI_Output, N_CLASSES, N_ATTRIBUTES_RAW
from CUB.inference import run_eval

def get_stage2_pred(a_hat, model):
    """Takes intermediate representation and predict class outputs, to see how they change when intervening."""
    stage2_inputs = torch.from_numpy(np.array(a_hat)).cuda().float()
    class_outputs = model(stage2_inputs)
    class_outputs = torch.nn.Softmax()(class_outputs)
    return class_outputs.data.cpu().numpy()


def build_mask(data: List[Dict], min_count: int = 10) -> np.ndarray:
    # Count the number of times each attribute is known to be true or false for each class
    class_attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES_RAW, 2))
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


def make_basic_tti_objs(args) -> None:
    train_data = pickle.load(open(os.path.join(args.data_dir_raw, "train.pkl"), "rb"))

    # Make a filter for attributes that are not common enough, as list of IDs to keep
    attr_mask = build_mask(train_data, min_count=10) 
    assert len(attr_mask) == args.n_attributes

    test_data = pickle.load(open(os.path.join(args.data_dir_raw, "test.pkl"), "rb"))

    # Build numpy arrays of the labels for the attributes we are using
    raw_masked_attr_test_labels = np.zeros((len(test_data), len(attr_mask)))

    for ndx, d in enumerate(test_data):
        raw_masked_attr_test_labels[ndx] = np.array(d["attribute_label"])[attr_mask]
    
    # Save them
    with open(os.path.join(args.data_dir_raw, "raw_masked_attr_test_labels.pkl"), "wb") as f:
        pickle.dump(raw_masked_attr_test_labels, f)
    
    # Build a dict which contains the groups of attributes and the attr_ids which they contain after masking
    attr_group_dict = build_dict_from_mask(attr_mask=attr_mask)
    with open(os.path.join(args.data_dir_raw, "attr_group_dict.pkl"), "wb") as f:
        pickle.dump(attr_group_dict, f)
    

def run_tti(args) -> List[Tuple[int, float, float]]:
    """
    Returns pairs of (n, score) where n is a number of attributes
    and score is the accuracy when n attributes are intervened on
    """
    # Load the data
    if os.path.exists(os.path.join(args.data_dir_raw, "raw_masked_attr_test_labels.pkl")) and os.path.exists(os.path.join(args.data_dir_raw, "attr_group_dict.pkl")):
        raw_masked_attr_test_labels = pickle.load(open(os.path.join(args.data_dir_raw, "raw_masked_attr_test_labels.pkl"), "rb"))
        attr_group_dict = pickle.load(open(os.path.join(args.data_dir_raw, "attr_group_dict.pkl"), "rb"))
    else:
        make_basic_tti_objs(args)
    

    # Run one epoch, and get the class and concept vector for each image
    eval_output = run_eval(args) # Returns Eval_Output object, bascially namedtuple of np arrays
    
    # Get 5, 95 percentiles for how much each attribute was predicted to be true [0,1]
    ptl_5: Dict[int, float] = dict()
    ptl_95: Dict[int, float] = dict()
    
    if args.flat_intervene:
        ptl_5 = ddict(lambda: args.intervene_vals[0])
        ptl_95 = ddict(lambda: args.intervene_vals[1])

    else:
        for attr_idx in range(args.n_attributes):
            ptl_5[attr_idx] = np.percentile(eval_output.attr_pred_outputs[:, :, attr_idx], 5)
            ptl_95[attr_idx] = np.percentile(eval_output.attr_pred_outputs[:, :, attr_idx], 95)
        
    # Creating the correct output array, where 'correct' is the 5th percentile of the attribute if false, and 95th percentile if true
    correct_attr_outputs = np.zeros_like(eval_output.attr_pred_outputs)
    # Whether to replace using the class data ie using the majority voting transform or the raw data
    if args.replace_class: # shapes here are (n_models, n_examples, n_attributes)
        for attr_idx in range(args.n_attributes):
            correct_attr_outputs[:, :, attr_idx] = np.where(
                eval_output.attr_true_labels[:, :, attr_idx] == 0, ptl_5[attr_idx], ptl_95[attr_idx]
            )
    else:
        for attr_idx in range(args.n_attributes):
            correct_attr_outputs[:, :, attr_idx] = np.where(
                raw_masked_attr_test_labels[:, :, attr_idx] == 0, ptl_5[attr_idx], ptl_95[attr_idx]
            )

    # Get main model and attr -> label model
    model = torch.load(args.model_dir)
    
    # Check that number of attributes matches between the 'raw' data and the class aggregated data
    assert len(raw_masked_attr_test_labels) == eval_output.attr_true_labels.shape[1], "len(instance_attr_labels): %d, len(eval_output.attr_labels): %d" % (
        len(raw_masked_attr_test_labels),
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
        all_class_acc_top1 = []
        all_class_acc_top5 = []
        all_mix_class_acc = []
        if args.multimodel and args.multimodel_type == "ensemble":
            n_trials = args.n_trials * model.args.n_models
        else:
            n_trials = args.n_trials
        for ndx in range(n_trials):
            # Array of attr predictions, will be modified towards ground truth
            updated_attrs = np.array(eval_output.attr_pred_outputs[:])
            if n_replace > 0:
                for img_id in range(eval_output.class_labels.shape[1]):
                    # Get a list of all attrs (in the flattened list) that we will intervene on for this img
                    replace_idxs = []
                    group_replace_idx = list(
                        random.sample(list(range(args.n_groups)), n_replace)
                    )
                    for i in group_replace_idx:
                        replace_idxs.extend(attr_group_dict[i])
                    
                    updated_attrs[:, img_id, replace_idxs] = correct_attr_outputs[:, img_id, replace_idxs]
            

            stage2_inputs = torch.from_numpy(updated_attrs).cuda()

            if args.multimodel: # if multimodel, we need to reshape the inputs to be (n_models, n_imgs, n_attrs)
                class_outputs_l = []
                class_outputs_mix = torch.zeros(stage2_inputs.shape[1], N_CLASSES).cuda()
                top1s = []
                for i, post_model in enumerate(model.post_models):
                    post_model.eval()
                    model_class_outputs = post_model(stage2_inputs[i])
                    class_outputs_l.append(model_class_outputs)
                    class_outputs_mix += post_model(stage2_inputs[i])
                    top1s.append(model_class_outputs.topk(k=1, dim=1)[1].data.cpu().numpy().squeeze())
                
                class_outputs = torch.stack(class_outputs_l, dim=0)
                
                class_outputs_mix /= model.args.n_models

            else:
                model_use = model.second_model
                if model.args.exp == "Independent":
                    model_use = torch.nn.Sequential(torch.nn.Sigmoid(), model_use)
                model_use.eval()
                class_outputs = model_use(stage2_inputs[0]).unsqueeze(0)

            # class_outputs shape is (n_models, n_imgs, n_classes)
            _, predictions_top1 = class_outputs.topk(k=1, dim=2) # topk returns a tuple of (values, indices)
            _, predictions_top5 = class_outputs.topk(k=5, dim=2)
            predictions_top1_np = predictions_top1.data.cpu().numpy().squeeze()
            predictions_top5_np = predictions_top5.data.cpu().numpy()

            correct_top1_t = np.array(predictions_top1_np) == np.array(eval_output.class_labels)

            # Get a tensor which is true if the class label is in the top 5 predictions
            # predictions_top5_np shape is (n_models, n_imgs, 5)
            correct_top5_t = np.zeros_like(predictions_top5_np, dtype=bool)
            for i in range(predictions_top5_np.shape[0]):
                for j in range(predictions_top5_np.shape[1]):
                    correct_top5_t[i, j] = eval_output.class_labels[i, j] in predictions_top5_np[i, j]

            class_acc_top1 = np.mean(correct_top1_t)
            all_class_acc_top1.append(class_acc_top1 * 100) # convert to percent

            class_acc_top5 = np.mean(correct_top5_t)
            all_class_acc_top5.append(class_acc_top5 * 100) # convert to percent

            if args.multimodel:
                _, mix_predictions = class_outputs_mix.topk(k=1, dim=1)
                mix_predictions = mix_predictions.data.cpu().numpy().squeeze()
                if ndx == 0:
                    print(
                        "top1 similarities: 0,1 0,mix, 1,mix", 
                        prop_equal(predictions_top1_np[0], predictions_top1_np[1]), 
                        prop_equal(predictions_top1_np[0], mix_predictions), 
                        prop_equal(predictions_top1_np[1], mix_predictions)
                    )
                correct_mix = np.array(mix_predictions) == np.array(eval_output.class_labels)
                mix_class_acc = np.mean(correct_mix)

                if ndx == 0:
                    print("cross accs", np.logical_and(correct_mix, correct_top1_t[0]).mean(), np.logical_or(correct_mix, correct_top1_t[1]).mean())

                all_mix_class_acc.append(mix_class_acc * 100)


        # changed from max to sum - not sure why max would be appropriate
        top1_acc = sum(all_class_acc_top1) / len(all_class_acc_top5)
        top5_acc = sum(all_class_acc_top5) / len(all_class_acc_top5)

        # TODO: Decide if we want to use mix acc somehow (I think not)
        print(n_replace, top1_acc, top5_acc)
        results.append((n_replace, top1_acc, top5_acc))

    return results


def prop_equal(t1: np.ndarray, t2: np.ndarray) -> float:
    return np.sum(t1 == t2) / len(t1)

def graph_tti_output(
    tti_output: List[Tuple[int, float, float]], 
    save_dir: Optional[str] = None, 
    show: bool = True, 
    label: Optional[str]=None, 
    return_fig: bool = False
    ) -> Optional[Figure]:
    """Graph the output of a TTI run"""
    try:
        n_replace, top1_acc, top5_acc = zip(*tti_output)
    except:
        breakpoint()
    plt.plot(n_replace, top1_acc, label=label)
    plt.xlabel("Number of groups replaced")
    plt.ylabel("Accuracy")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "tti_results.png"))
    if show:
        plt.show()
    if return_fig:
        return plt.gcf()
    return None
    
def graph_tti_simple(
    tti_output: List[Tuple[int, float, float]],
    save_dir: Optional[str] = None,
    show: bool = True,
    label: Optional[str] = None,
    return_fig: bool = False,
    fig: Optional[Figure] = None,
):
    """ Add a single point to a 2D graph showing the non-intervention (tti0) and full-intervention (ttilast) results"""
    if not fig:
        fig = plt.figure()

    n_replace, top1_acc, top5_acc = zip(*tti_output)
    plt.plot(top1_acc[0], top1_acc[-1], "o", label=label)
    plt.xlabel("Accuracy with no intervention")
    plt.ylabel("Accuracy with full intervention")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "tti_results.png"))
    if show:
        plt.show()
    return fig

def graph_multi_tti_output(tti_outputs: List[List[Tuple[int, float, float]]], save_dir: Optional[str] = None):
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
