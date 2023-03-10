import os
import sys

import pickle

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.configs import multi_inst_cfg
from CUB.dataset import load_data
from CUB.cub_utils import download_from_aws

from CUB.tti import build_mask, build_dict_from_mask


def main():
    st.title("CUB")
    BASE_PATH = "/root/hoagy-hiddeninfo-sync"

    model_with_two_joints_path = "big_run/multi_inst_joint/20230302-141428/final_model.pth"
    seq_model_path = "big_run/seq_inst/20230224-124623/final_model.pth"
    seq_sparse_model_path = "big_run/seq_inst_sparse/20230227-183548/final_model.pth"
    multi_path = "big_run/multimodel_inst/20230224-172742/final_model.pth"
    ind_path = "big_run/ind_inst/20230224-135153/final_model.pth"

    model_paths = [ind_path, seq_model_path, seq_sparse_model_path, model_with_two_joints_path, multi_path]

    download_from_aws(model_paths)
    st.write("Downloaded models")

    train_data_path = "CUB_instance_masked/train.pkl"
    val_data_path = "CUB_instance_masked/val.pkl"
    test_data_path = "CUB_instance_masked/test.pkl" 

    args = multi_inst_cfg

    # Looking at each model's activations
    train_loader = load_data([train_data_path], args)
    val_loader = load_data([val_data_path], args)
    test_loader = load_data([test_data_path], args)

    if os.path.exists("big_run_attrs.pkl"):
        big_run_attrs = pickle.load(open("big_run_attrs.pkl", "rb"))
        seq_attrs = big_run_attrs["seq_inst"]
        seq_sparse_attrs = big_run_attrs["seq_inst_sparse"]
        joint_attrs = big_run_attrs["multi_inst_joint"]
        multi_attrs = big_run_attrs["multimodel_inst"]
        ind_attrs = big_run_attrs["ind_inst"]

    else:
        is_downloaded = download_from_aws(["big_run_attrs.pkl"])
        if not is_downloaded:
            with torch.no_grad():
                for model_path in model_paths:
                    attr_container = []
                    full_path = os.path.join(BASE_PATH, model_path)
                    st.write((f"Loading model from {full_path}"))
                    model = torch.load(os.path.join(BASE_PATH, model_path))
                    model.eval()
                    for batch in train_loader:
                        inputs, class_labels, attr_labels, attr_mask = batch

                        attr_labels = [i.float() for i in attr_labels]
                        attr_labels = torch.stack(attr_labels, dim=1)

                        attr_labels = attr_labels.cuda() if torch.cuda.is_available() else attr_labels
                        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                        class_labels = class_labels.cuda() if torch.cuda.is_available() else class_labels
                        attr_mask = attr_mask.cuda() if torch.cuda.is_available() else attr_mask

                        output = model.generate_predictions(inputs, attr_labels, attr_mask, straight_through=True) # Need straight through for ind model
                        attr_container.append(output[0])

                    del(model) # Clear memory
                    model_name = model_path.split('/')[-3] # Should be eg "ind_inst"
                    big_run_attrs[model_name] = torch.cat(attr_container, dim=1)
                    print(f"Done with model {model_path.split('/')[-3]}")

            pickle.dump(big_run_attrs, open("big_run_attrs.pkl", "wb"))
            big_run_attrs = pickle.load(open("big_run_attrs.pkl", "rb"))
            seq_attrs = big_run_attrs["seq_inst"]
            seq_sparse_attrs = big_run_attrs["seq_inst_sparse"]
            joint_attrs = big_run_attrs["multi_inst_joint"]
            multi_attrs = big_run_attrs["multimodel_inst"]
            ind_attrs = big_run_attrs["ind_inst"]

    # Looking at the difference between the two joint models
    seq_attrs = seq_attrs[0]
    ind_attrs = ind_attrs[0]
    seq_sparse_attrs = seq_sparse_attrs[0]
    joint_attrs1 = joint_attrs[0]
    joint_attrs2 = joint_attrs[1]
    multi_attrs1 = multi_attrs[0]
    multi_attrs2 = multi_attrs[1]

    diff1 = joint_attrs1 - seq_attrs
    diff2 = joint_attrs2 - seq_attrs

    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    # Looking at the cosine similarities and abs differences between the joint models
    st.write("Cosine similarity between joint1 and seq")
    st.write(cos_sim(joint_attrs1, seq_attrs).mean(), torch.abs(joint_attrs1 - seq_attrs).mean())
    st.write("Cosine similarity between joint2 and seq")
    st.write(cos_sim(joint_attrs2, seq_attrs).mean(), torch.abs(joint_attrs2 - seq_attrs).mean())
    st.write("Cosime similiary of the deviations from seq")
    st.write(cos_sim(diff1, diff2).mean(), torch.abs(diff1 - diff2).mean())
    st.write("Cosine similarity of seq and seq_sparse")
    st.write(cos_sim(seq_attrs, ind_attrs).mean(), torch.abs(seq_attrs - ind_attrs).mean())

    # Looking at the cosine similarities with a sigmoid applied to the attr vectors, and the abs differences
    st.write("Cosine similarity between joint1 and seq with sigmoid")
    st.write(cos_sim(torch.sigmoid(joint_attrs1), torch.sigmoid(seq_attrs)).mean(), torch.abs(torch.sigmoid(joint_attrs1) - torch.sigmoid(seq_attrs)).mean())
    st.write("Cosine similarity between joint2 and seq with sigmoid")
    st.write(cos_sim(torch.sigmoid(joint_attrs2), torch.sigmoid(seq_attrs)).mean(), torch.abs(torch.sigmoid(joint_attrs2) - torch.sigmoid(seq_attrs)).mean())
    st.write("Cosime similiary of the deviations from seq with sigmoid")
    st.write(cos_sim(torch.sigmoid(diff1), torch.sigmoid(diff2)).mean(), torch.abs(torch.sigmoid(diff1) - torch.sigmoid(diff2)).mean())
    st.write("Cosine similarity of seq and seq_sparse with sigmoid")
    st.write(cos_sim(torch.sigmoid(seq_attrs), torch.sigmoid(ind_attrs)).mean(), torch.abs(torch.sigmoid(seq_attrs) - torch.sigmoid(ind_attrs)).mean())

    attrs_list = [seq_attrs, ind_attrs, joint_attrs1, joint_attrs2, multi_attrs1, multi_attrs2, seq_sparse_attrs, joint_attrs2 - joint_attrs1, multi_attrs2 - multi_attrs1]
    attr_names = ["seq", "ind", "joint1", "joint2", "multi1", "multi2", "seq_sparse", "jointdiff", "multidiff"]
    
    for i, attr in enumerate(attrs_list):
        assert attr.shape == attrs_list[0].shape, f"Attr {i} has shape {attr.shape} but should be {attrs_list[0].shape}"
    # Make a dataframe of the pairwise cosine similarities
    pw_cos_df = pd.DataFrame(columns=attr_names, index=attr_names)
    for i, attr1 in enumerate(attrs_list):
        for j, attr2 in enumerate(attrs_list):
            pw_cos_df.iloc[i, j] = cos_sim(attr1, attr2).mean().item()

    # Make a dataframe of the pairwise abs differences
    pw_abs_df = pd.DataFrame(columns=attr_names, index=attr_names)
    for i, attr1 in enumerate(attrs_list):
        for j, attr2 in enumerate(attrs_list):
            pw_abs_df.iloc[i, j] = torch.abs(attr1 - attr2).mean().item()
    
    # Make a dataframe of the pairwise cosine similarities with sigmoid
    pw_cos_df_sig = pd.DataFrame(columns=attr_names, index=attr_names)
    for i, attr1 in enumerate(attrs_list):
        for j, attr2 in enumerate(attrs_list):
            pw_cos_df_sig.iloc[i, j] = cos_sim(torch.sigmoid(attr1), torch.sigmoid(attr2)).mean().item()
        
    # Make a dataframe of the pairwise abs differences with sigmoid
    pw_abs_df_sig = pd.DataFrame(columns=attr_names, index=attr_names)
    for i, attr1 in enumerate(attrs_list):
        for j, attr2 in enumerate(attrs_list):
            pw_abs_df_sig.iloc[i, j] = torch.abs(torch.sigmoid(attr1) - torch.sigmoid(attr2)).mean().item()

    # Same again but with correlation instead of cosine similarity
    pw_corr_df = pd.DataFrame(columns=attr_names, index=attr_names)
    for i, attr1 in enumerate(attrs_list):
        for j, attr2 in enumerate(attrs_list):
            pw_corr_df.iloc[i, j] = torch.corrcoef(torch.stack((attr1, attr2)).reshape(2, -1))[0, 1].item()

    pw_corr_df_sig = pd.DataFrame(columns=attr_names, index=attr_names)

    for i, attr1 in enumerate(attrs_list):  
        for j, attr2 in enumerate(attrs_list):
            pw_corr_df_sig.iloc[i, j] = torch.corrcoef(torch.stack((torch.sigmoid(attr1), torch.sigmoid(attr2))).reshape(2, -1))[0, 1].item()
    
    
    # Display the dataframes
    st.write("Cosine similarities")
    st.write(pw_cos_df)
    st.write("Abs differences")
    st.write(pw_abs_df)
    st.write("Cosine similarities with sigmoid")
    st.write(pw_cos_df_sig)
    st.write("Abs differences with sigmoid")
    st.write(pw_abs_df_sig)
    st.write("Correlations")
    st.write(pw_corr_df)
    st.write("Correlations with sigmoid")
    st.write(pw_corr_df_sig)

    st.write("Predictions for this data were that seq and inst would be the most similar, because they had the same incentive, and a Schelling point for how to achieve it.")
    st.write("The joint models have the same incentive, but different Schelling points, so they should be more similar to each other than to seq or ind, but less similar to each other than to seq and ind.")

        
    st.write(pw_corr_df["ind"]["seq"])
    st.write(pw_corr_df_sig["ind"]["seq"])

    train_data = pickle.load(open(os.path.join("CUB_processed", "train.pkl"), "rb"))
    # Make a filter for attributes that are not common enough, as list of IDs to keep
    attr_mask = build_mask(train_data, min_count=10)
    attr_group_dict = build_dict_from_mask(attr_mask=attr_mask)
    ind_group_bool = attr_vec_to_onehot_groups(ind_attrs, attr_group_dict)
    seq_group_bool = attr_vec_to_onehot_groups(seq_attrs, attr_group_dict)
    st.write(torch.sum(ind_group_bool, dim=0), torch.sum(seq_group_bool, dim=0))
    st.write(len(attr_group_dict))

def attr_vec_to_onehot_groups(attr_vec, attr_group_dict):
    """
    Converts a vector of attribute IDs to a one-hot vector of attribute groups
    :param attr_vec: A tensor of attribute logits with shape (batch_size, num_attrs)
    :param attr_group_dict: A dictionary mapping attribute group IDs to lists of attribute IDs
    :return: A binary tensor of attribute group IDs with shape (batch_size, num_attrs)
    """

    new_attr_vec = torch.zeros_like(attr_vec)
    for group_id in attr_group_dict:
        group_start = attr_group_dict[group_id][0]
        max_in_attr_group = torch.argmax(attr_vec[:, attr_group_dict[group_id]], dim=1)
        print(group_start, max_in_attr_group)
        new_attr_vec[:, group_start + max_in_attr_group] = 1
    
    return new_attr_vec.to(torch.bool)



if __name__ == "__main__":
    main()