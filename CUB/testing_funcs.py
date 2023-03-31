from datetime import datetime
import functools
import itertools
import os
import pickle
import shutil
import sys
from typing import List

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.models import Multimodel, CUB_Multimodel
from CUB.cub_classes import Experiment
from CUB.inference import run_eval
from CUB.configs import multi_inst_cfg
from CUB.dataset import load_data, DataLoader
from CUB.cub_utils import download_from_aws, download_folder_from_aws, upload_to_aws

def download_model_locs(model_locs):
    for path in model_locs:
         # Check if each path exists, if not, try to download from aws
        if not os.path.exists(path):
            has_downloaded = download_from_aws([path])
            if not has_downloaded:
                raise FileNotFoundError(f"Could not find {path} and could not download from aws")

def compose_multimodels(models_list):
    download_model_locs(models_list)

    sep_models = [torch.load(model_path) for model_path in models_list]
    multimodel = Multimodel(multi_inst_cfg) 

    # Compose the different model.pre_model objects, which are nn.Modulelist objects into the new multimodel
    multimodel.pre_models = functools.reduce(lambda x, y: x + y, [m.pre_models for m in sep_models], nn.ModuleList())
    multimodel.post_models = functools.reduce(lambda x, y: x + y, [m.post_models for m in sep_models], nn.ModuleList())

    DATETIME_FMT = "%Y%m%d-%H%M%S"
    now_str = datetime.now().strftime(DATETIME_FMT)

    save_dir = "out/multi_attr_weight/" + now_str
    os.makedirs(save_dir, exist_ok=True)

    torch.save(multimodel, os.path.join(save_dir, "final_model.pth"))


def get_concept_entropy(loader: DataLoader):
    # Get the per-concept entropy of the dataset
    device = torch.device("cpu")

    all_labels = torch.zeros((len(loader.dataset), 109), dtype=torch.float32) # type: ignore 

    n_examples = 0
    for batch in loader:
        _, _, attr_labels, _ = batch
        attr_labels = [i.float() for i in attr_labels]
        attr_labels = torch.stack(attr_labels, dim=1)
        attr_labels = attr_labels.to(device)
        all_labels[n_examples:n_examples + attr_labels.shape[0]] = attr_labels
    
        n_examples += attr_labels.shape[0]
    
    # Now we have all the labels, we can get the entropy
    # Get the number of examples per concept
    concept_counts = all_labels.sum(dim=0)
    concept_probs = concept_counts / len(loader.dataset) # type: ignore

    # Add/subtract a small number if prob = 0 or prob = 1 to avoid log(0)
    concept_probs[concept_probs == 0] = 1e-6
    concept_probs[concept_probs == 1] = 1 - 1e-6
    concept_true_entropy = -concept_probs * torch.log2(concept_probs)
    concept_false_entropy = -(1 - concept_probs) * torch.log2(1 - concept_probs)

    concept_entropy = concept_true_entropy + concept_false_entropy
    return concept_entropy

def entropy_test_fn(prob: np.ndarray):
    # Testing function for the entropy calcs
    # Should be near 0 at 0 or 1, =1 at 0.5, and convex in between
    concept_true_entropy = -prob * np.log2(prob)
    concept_false_entropy = -(1 - prob) * np.log2(1 - prob)

    concept_entropy = concept_true_entropy + concept_false_entropy

    return concept_entropy
    
def compose_multi(models_list: List[str], tag: str = "multi_attr_weight"): # List of paths to models
    # Make multimodel from the two joint models
    download_model_locs(models_list)
    
    sep_models = [torch.load(model_path) for model_path in models_list]

    multimodel = Multimodel(multi_inst_cfg)
    multimodel.pre_models = nn.ModuleList([model.first_model for model in sep_models])
    multimodel.post_models = nn.ModuleList([model.second_model for model in sep_models])

    DATETIME_FMT = "%Y%m%d-%H%M%S"
    now_str = datetime.now().strftime(DATETIME_FMT)

    save_dir = os.path.join("out", tag, now_str)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(multimodel, os.path.join(save_dir, "final_model.pth"))

def get_diffs_by_concept(dataloader: DataLoader, seq_model: Multimodel, joint_model: Multimodel):
    # Check where the largest difference between the two models is, in terms of their concept vectors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_examples = 0
    diffs = torch.zeros(joint_model.args.n_attributes).to(device)

    with torch.no_grad():
        for batch in dataloader:
            inputs, _, _, _ = batch # class_labels, attr_labels, mask

            inputs = inputs.to(device)

            seq_concepts, _ = seq_model.pre_models[0](inputs) # Blank is the aux preds
            seq_concepts = torch.cat(seq_concepts, dim=1)

            joint_concepts, _ = joint_model.pre_models[0](inputs) # Blank is the aux preds
            joint_concepts = torch.cat(joint_concepts, dim=1)

            diff = (seq_concepts - joint_concepts).abs().mean(dim=0) # Getting the average absolute difference for each concept
            diffs += diff
            n_examples += inputs.shape[0]
    
    diffs /= n_examples
    return diffs.to("cpu").numpy()
    


def look_at_activations():
    model_with_two_joints_path = "big_run/multi_inst_joint/20230302-141428/final_model.pth"
    seq_model_path = "big_run/seq_inst/20230224-124623/final_model.pth"
    seq_sparse_model_path = "big_run/seq_inst_sparse/20230227-183548/final_model.pth"
    multi_path = "big_run/multimodel_inst/20230224-172742/final_model.pth"

    train_data_path = "CUB_instance_masked/train.pkl"
    val_data_path = "CUB_instance_masked/val.pkl"
    test_data_path = "CUB_instance_masked/test.pkl" 

    args = multi_inst_cfg

    # Looking at each model's activations
    train_loader = load_data([train_data_path], args)
    val_loader = load_data([val_data_path], args)
    test_loader = load_data([test_data_path], args)

    seq_attrs = []
    seq_sparse_attrs = []
    joint_attrs = []
    multi_attrs = []

    model_paths = [seq_model_path, seq_sparse_model_path, model_with_two_joints_path, multi_path]
    output_lists = [seq_attrs, seq_sparse_attrs, joint_attrs, multi_attrs]
    with torch.no_grad():
        
            for model_path, olist in zip(model_paths, output_lists):
                model = torch.load(model_path)
                model.eval()
                for batch in train_loader:
                    inputs, class_labels, attr_labels, attr_mask = batch

                    attr_labels = [i.float() for i in attr_labels]
                    attr_labels = torch.stack(attr_labels, dim=1)

                    attr_labels = attr_labels.cuda() if torch.cuda.is_available() else attr_labels
                    inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                    class_labels = class_labels.cuda() if torch.cuda.is_available() else class_labels
                    attr_mask = attr_mask.cuda() if torch.cuda.is_available() else attr_mask

                    output = model.generate_predictions(inputs, attr_labels, attr_mask)
                    olist.append(output[0])

            del(model) # Clear memory
            print(f"Done with model {model_path.split('/')[-3]}")


    seq_attrs = torch.cat(seq_attrs, dim=1)
    seq_sparse_attrs = torch.cat(seq_sparse_attrs, dim=1)
    joint_attrs = torch.cat(joint_attrs, dim=1)
    multi_attrs = torch.cat(multi_attrs, dim=1)

    # Looking at the difference between the two joint models
    seq_attrs = seq_attrs[0]
    seq_sparse_attrs = seq_sparse_attrs[0]
    joint_attrs1 = joint_attrs[0]
    joint_attrs2 = joint_attrs[1]
    multi_attrs1 = multi_attrs[0]
    multi_attrs2 = multi_attrs[1]

    diff1 = joint_attrs1 - seq_attrs
    diff2 = joint_attrs2 - seq_attrs

    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    print("Cosine similarity between joint1 and seq")
    print(cos_sim(joint_attrs1, seq_attrs).mean())
    print("Cosine similarity between joint2 and seq")
    print(cos_sim(joint_attrs2, seq_attrs).mean())
    print("Cosime similiary of the deviations from seq")
    print(cos_sim(diff1, diff2).mean())
    print("Cosine similarity of seq and seq_sparse")
    print(cos_sim(seq_attrs, seq_sparse_attrs).mean())


    breakpoint()

def concat_wandb_runs():
    # Concatenate the runs from wandb after training in separate stages

    import wandb
    api = wandb.Api()
    project_str = "/distillrepr/distill_CUB/runs/"
    run_ids = ["k1iluazx", "3qwc211z"]
    paths = [project_str + run_id for run_id in run_ids]

    # Get the run histories
    dfs = []
    for path in paths:
        run = api.run(path)
        data = run.history() # Returns a pandas dataframe
        dfs.append(data)

    # Concatenate them so that the epochs of the second run are after the first, so that the second run starts from epoch (last epoch of first run + 1)

    # Add the total epochs of the first run to the second run's epochs
    dfs[1]["epoch"] = dfs[1]["epoch"] + dfs[0]["epoch"].max() + 1
    
    # Concatenate the two dataframes
    df = pd.concat(dfs)
    

    # Graph using matplotlib with the x axis being the epoch and the y axis having "train_acc", "val_acc", "train_cross_acc0"

    # First check that the epochs are in order
    assert df["epoch"].is_monotonic_increasing
    # then check that the columns exist as expected
    assert "train_acc" in df.columns, f"train_acc not in {df.columns}"
    assert "val_acc" in df.columns, f"val_acc not in {df.columns}"
    assert "train_cross_acc0" in df.columns, f"train_cross_acc0 not in {df.columns}"

    # train_cross_acc0 is written on different lines to train_acc, so we need to collapse so there's one line for each value of epoch, making sure to ignore NaNs
    df = df.groupby("epoch").mean()

    # Plot the graph, with the x axis being the epoch and the y axis having "train_acc", "val_acc", "train_cross_acc0"
    # Note that epoch is the index of the dataframe, so we don't need to specify it
    df[150:].plot(y=["train_acc", "val_acc", "train_cross_acc0", "val_cross_acc0"])
    print(df["train_cross_acc0"])
    
    # Save figure
    plt.savefig("images/wandb_concat.png")


def save_all_premodels(runs_s3_loc: List[str]):
    '''
    Downloading the various premodels that are saved in AWS S3,
    checking their attr_loss_weight and whether they have use_dropout=True,
    then saving them all in a clearly labelled folder in AWS S3
    '''
    new_save_folder = "attr_dropout_base"
    os.makedirs(new_save_folder, exist_ok=True)

    for run in runs_s3_loc:
        download_folder_from_aws(run) # Downloads to the same folder as in s3
        model_path = run + "latest_model.pth"
        config_pkl = run + "config.pkl"
        model: CUB_Multimodel
        model = torch.load(model_path)
        attr_loss_weight = model.args.attr_loss_weight
        uses_dropout = model.args.use_pre_dropout
        
        print(f"Model {model_path} has attr_loss_weight {attr_loss_weight} and uses_dropout {uses_dropout}")

        if not isinstance(attr_loss_weight, list):
            attr_loss_weight = [attr_loss_weight] * 2 # Models are all n_model=2 but earlier models only had one attr_loss_weight
        
        # Save the premodel
        for ndx, premodel in enumerate(model.pre_models):
            weight_str = str(attr_loss_weight[ndx]).replace(".", "_")
            if model.args.exp == "MultiSequential":
                weight_str = "sep"
            model_folder = f"{new_save_folder}/premodel_attr_loss_weight_{weight_str}_drop_{uses_dropout}"
            if os.path.exists(model_folder):
                print(f"File {model_folder} already exists, skipping")
                continue

            os.makedirs(model_folder, exist_ok=True)
            config_path = os.path.join(model_folder, "config.pkl")
            shutil.copyfile(config_pkl, config_path)

            model_path = os.path.join(model_folder, "base_model.pth")
            args = model.args
            args.n_models = 1
            args.attr_loss_weight = attr_loss_weight[ndx]
            args.use_pre_dropout = uses_dropout

            new_model = Multimodel(args)
            new_model.pre_models = nn.ModuleList([premodel])
            torch.save(model, model_path)
    

def test_separation(model_loc, args: Experiment):
    """
    This function takes a Multimodel model with two premodels, and a data loader.
    It creates a classifier of the same size as the model's classifier, but with only 2 possible outputs
    instead of the model's classifier's 200 outputs. The classification task is to predict whether the
    concept vector has been generated by the first premodel or the second premodel.
    """
    print("Testing separation with model", model_loc)
    download_from_aws([model_loc])
    model = torch.load(model_loc)
    print(model.pre_models[0].use_dropout)

    train_data_path = os.path.join(args.base_dir, args.data_dir, "train.pkl")
    val_data_path = train_data_path.replace("train.pkl", "val.pkl")

    train_loader = load_data([train_data_path], args)
    val_loader = load_data([val_data_path], args)

    concept_vec_dim = model.args.n_attributes
    hidden_dim = model.args.expand_dim
    n_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a new classifier with len(model.pre_models) outputs
    classifier = nn.Sequential(
        nn.Softmax(dim=1),
        nn.Linear(concept_vec_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, len(model.pre_models))
    )

    classifier.to(device)
    classifier.train()

    # Create a new optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Create a new loss function
    loss_fn = nn.CrossEntropyLoss()
    running_av_loss = 0
    loss_horizon = 20
    loss_frac = 1/loss_horizon

    # Train the classifier
    for epoch in range(n_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_image, _, _, _ = batch # Blanks are class label, attribute labels, mask
            input_image = input_image.to(device)
            batch_size = input_image.shape[0]

            # Get the concept vectors from the premodels
            concept_vectors = []

            for premodel in model.pre_models:
                concept_vector, _ = premodel(input_image)
                concept_vector_t = torch.cat(concept_vector, dim=1)
                concept_vectors.append(concept_vector_t)

            # Concatenate the concept vectors into a tensor with first dim with length 2*batch_size
            concept_vector = torch.cat(concept_vectors, dim=0)

            # Get the predictions
            preds = classifier(concept_vector)

            # There are batch_size concept vectors from each premodel, in that order along the batch dimension
            target = torch.zeros(len(concept_vector), dtype=torch.long).to(device)
            for ndx in range(1, len(model.pre_models)):
                target[batch_size * ndx:batch_size * (ndx + 1)] = ndx

            # Calculate the loss
            loss = loss_fn(preds, target)
            loss.backward()
            optimizer.step()

            running_av_loss = (1 - loss_frac) * running_av_loss + loss_frac * loss.item()

        print(f"Epoch {epoch} finished, running average loss is {running_av_loss}")

    # Test the classifier
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_image, _, _, _ = batch
            input_image = input_image.to(device)
            batch_size = input_image.shape[0]

            # Get the concept vectors from the premodels 
            concept_vectors = []
            for premodel in model.pre_models:
                concept_vector, _ = premodel(input_image)
                concept_vector_t = torch.cat(concept_vector, dim=1)
                concept_vectors.append(concept_vector_t)

            # Concatenate the concept vectors into a tensor with first dim with length 2*batch_size
            concept_vector = torch.cat(concept_vectors, dim=0)

            # Get the predictions
            preds = classifier(concept_vector)

            # There are batch_size concept vectors from each premodel, in that order along the batch dimension
            target = torch.zeros(len(concept_vector), dtype=torch.long).to(device)
            for ndx in range(1, len(model.pre_models)):
                target[batch_size * ndx:batch_size * (ndx + 1)] = ndx

            # Get the predicted class
            _, predicted = torch.max(preds.data, 1)

            # Update the accuracy
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Accuracy: {correct / total}")

def make_loader(loader_loc: str):
    args = multi_inst_cfg
    return load_data([loader_loc], args)


if __name__ == "__main__":
    if sys.argv[1] == "compose":
        model_locs = [
            "attr_dropout_base/premodel_attr_loss_weight_0_2_drop_False/base_model.pth",
            "attr_dropout_base/premodel_attr_loss_weight_1_0_drop_False/base_model.pth"
        ]
        compose_multimodels(model_locs)
    elif sys.argv[1] == "test_diffs":
        # Test the differences between the models

        # Load the seq model
        seq_model_loc = "attr_dropout_base/premodel_attr_loss_weight_sep_drop_False/base_model.pth"

        train_loader = make_loader("CUB_instance_masked/train.pkl")
        val_loader = make_loader("CUB_instance_masked/val.pkl")  
                           
        download_model_locs([seq_model_loc])
        seq_model = torch.load(seq_model_loc)

        entropies = get_concept_entropy(train_loader)

        dropouts = ["False", "True"]
        weights = ["0_1","0_2", "0_3", "0_5", "1_0", "2", "3", "5", "10"]
        results = []

        for (weight, dropout) in itertools.product(weights, dropouts):
            # Load the solo model
            solo_model_loc = f"attr_dropout_base/premodel_attr_loss_weight_{weight}_drop_{dropout}/base_model.pth"
            download_model_locs([solo_model_loc])
            solo_model = torch.load(solo_model_loc)

            # Test the difference in the premodels
            train_diff = get_diffs_by_concept(train_loader, seq_model, solo_model)
            val_diff = get_diffs_by_concept(val_loader, seq_model, solo_model)

            # Get the correlation coefficients between the differences and the concept entropies
            train_coef = np.corrcoef(entropies, train_diff)[0, 1]
            val_coef = np.corrcoef(entropies, val_diff)[0, 1]

            results.append((weight, dropout, train_coef, val_coef))

            del solo_model

            print(f"Done with weight {weight}, dropout {dropout}")

        print(results)
        os.makedirs("out", exist_ok=True)
        pickle.dump(results, open("out/attr_loss_weight_diffs.pkl", "wb"))
        upload_to_aws("out/attr_loss_weight_diffs.pkl")
    
    elif sys.argv[1] == "plot_ent_graph":
        results = pickle.load(open("out/attr_loss_weight_diffs.pkl", "rb"))

        # Plotting a bar graph with the results for train and val correlations
        weights = ["0_1","0_2", "0_3", "0_5", "1_0", "2", "3", "5", "10"]
        dropouts = ["False", "True"]

        labels = [f"{x[0]}_{x[1]}" for x in results]
        train_corrs = [x[2] for x in results]
        val_corrs = [x[3] for x in results]

        plt.figure(figsize=(9, 8))
        plt.bar(range(len(results)), train_corrs, width=0.2, label="Train")

        # Put labels on xaxis of bar graph
        plt.xticks(range(len(results)), labels, rotation=45)

        plt.bar([x + 0.2 for x in range(len(results))], val_corrs, width=0.2, label="Val")
        plt.legend()

        plt.title("Correlation between concept entropy and difference in premodel predictions") 
        plt.xlabel("Premodel parameters")
        plt.ylabel("Correlation coefficient")

        plt.savefig("out/attr_loss_weight_diffs.png")
        upload_to_aws("out/attr_loss_weight_diffs.png")

    

### Old scripts graveyard
# timestamps = [
#     "20230322-174145/",
#     "20230322-183638/",
# ]

# folder = "out/multimodel_post_inst/"
# paths = [folder + timestamp + "latest_model.pth" for timestamp in timestamps]

# compose_multi(paths)

# # Making multiple pairs of multimodels
# joint_2paths1 = [folder + timestamp + "latest_model.pth" for timestamp in joint_timestamps[:2]]
# joint_2paths2 = [folder + timestamp + "latest_model.pth" for timestamp in joint_timestamps[2:]]
# compose_multi(joint_2paths1)


# paths = [
#     "out/multiseq_p1.csv",
#     "out/multiseq_p2.csv",
# ]
# concat_wandb_runs()

# run_s3_locs = [
#     "out/multi_attr_loss_weight_0.1,10_drop/20230324-183105/",
#     "out/multi_attr_loss_weight_0.1,10/20230324-183145/",
#     "out/multi_attr_loss_weight_0.2,5_drop/20230324-182851/",
#     "out/multi_attr_loss_weight_0.2,5/20230324-183119/",
#     "out/multi_attr_loss_weight_0.3,3_drop/20230324-180429/",
#     "out/multi_attr_loss_weight_0.5,2_drop/20230324-182823/",
#     "out/multimodel_post_inst/20230323-165608/", # [0.5, 2] no drop
#     "out/multimodel_post_inst/20230323-165611/", # [0.3, 3] no drop
#     "out/multimodel_seq/20230320-185823/", # seq 1 and 1 drop
#     "out/multi_seq_no_dropout/20230325-174548/", # seq 1 and 1 no drop
#     "out/multimodel_post_inst/20230320-124841/", # multi 1 and 1 drop
#     "out/multimodel_post_inst/20230320-130430/", # multi 1 and 1 no drop
# ]

# save_all_premodels(run_s3_locs)
# upload_to_aws("attr_dropout_base")


# model_loc = "out/multimodel_post_inst/20230320-130430/latest_model.pth"
# args = multi_inst_cfg

# test_separation(model_loc, args)