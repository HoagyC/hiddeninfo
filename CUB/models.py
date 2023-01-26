"""
Taken from yewsiang/ConceptBottlenecks
"""
import os
from typing import Optional, List, Tuple

import torch
from torch import nn

from CUB.analysis import accuracy
from CUB.template_model import MLP, inception_v3, End2EndModel
from CUB.cub_classes import Experiment
from CUB.dataset import find_class_imbalance
from CUB.config import BASE_DIR, AUX_LOSS_RATIO

# Basic model for predicting attributes from images
def ModelXtoC(
    pretrained: bool,
    num_classes: int,
    n_attributes: int,
    expand_dim: int,
) -> nn.Module:
    return inception_v3(
        pretrained=pretrained,
        freeze=False,
        aux_logits=True,
        num_classes=num_classes,
        n_attributes=n_attributes,
        bottleneck=True,
        expand_dim=expand_dim,
    )


# Basic model for predicting classes from attributes
def ModelCtoY(
    n_attributes: int, num_classes: int, expand_dim: int
) -> nn.Module:
    model = MLP(
        input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim
    )
    return model


class Multimodel(nn.Module):
    def __init__(self, args: Experiment):
        super().__init__()
        self.args = args
        self.pre_models: nn.ModuleList
        self.post_models: nn.ModuleList
        self.reset_pre_models()
        self.reset_post_models()
        self.train_mode = "separate"

    def reset_pre_models(self, pretrained: Optional[bool] = None) -> None:
        if pretrained is None:
            use_pretrained = self.args.pretrained
        else:
            use_pretrained = pretrained

        pre_models_list = [
            ModelXtoC(
                pretrained=use_pretrained,
                num_classes=self.args.num_classes,
                n_attributes=self.args.n_attributes,
                expand_dim=self.args.expand_dim,
                three_class=self.args.three_class,
            )
            for _ in range(self.args.n_models)
        ]
        self.pre_models = nn.ModuleList(pre_models_list)

    def reset_post_models(self) -> None:
        post_models_list = [
            ModelCtoY(
                n_attributes=self.args.n_attributes,
                num_classes=self.args.num_classes,
                expand_dim=self.args.expand_dim,
            )
            for _ in range(self.args.n_models)
        ]

        self.post_models = nn.ModuleList(post_models_list)

    def generate_predictions(self, inputs, attr_labels):
        attr_preds = []
        aux_attr_preds = []
        class_preds = []
        aux_class_preds = []

        # Train each pre model with its own post model
        if self.train_mode == "separate":
            for i in range(self.args.n_models):
                attr_pred, aux_attr_pred = self.pre_models[i](inputs)
                class_pred = self.post_models[i](attr_pred)
                aux_class_pred = self.post_models[i](aux_attr_pred)
                attr_preds.append(attr_pred)
                aux_attr_preds.append(aux_attr_pred)
                class_preds.append(class_pred)
                aux_class_preds.append(aux_class_pred)
        
        # Randomly shuffle which post model is used for each pre model
        elif self.train_mode == "shuffle":
            post_model_indices = torch.randperm(self.args.n_models)
            for i, j in enumerate(post_model_indices):
                attr_pred, aux_attr_pred = self.pre_models[i](inputs)
                class_pred = self.post_models[j](attr_pred)
                aux_class_pred = self.post_models[j](aux_attr_pred)
                attr_preds.append(attr_pred)
                aux_attr_preds.append(aux_attr_pred)
                class_preds.append(class_pred)
                aux_class_preds.append(aux_class_pred)

        return attr_preds, aux_attr_preds, class_preds, aux_class_preds
    
    def generate_loss(self, attr_preds, attr_labels, class_preds, class_labels, aux_class_preds, mask):
        class_loss = self.criterion(class_preds, class_labels)
        aux_class_loss = self.criterion(aux_class_preds, class_labels)
        class_loss += aux_class_loss * AUX_LOSS_RATIO
        
        attr_loss = 0
        for i in range(len(self.attr_criterion)):
            attr_loss += self.attr_criterion[i](
                attr_preds[i].squeeze(), attr_labels[mask, i] # Masking attr losses
            )
        
        attr_loss /= len(self.attr_criterion)
        loss = (attr_loss * self.attr_loss_ratio) + class_loss
        return loss
    


def make_weighted_criteria(args):
    attr_criterion = []
    train_data_path = os.path.join(BASE_DIR, args.data_dir, "train.pkl")
    imbalance = find_class_imbalance(train_data_path, True) # assume args.weighted loss is always "multiple" if not ""
    for ratio in imbalance:
        attr_criterion.append(
            torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda())
        )
    return attr_criterion

class JointModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.first_model = ModelXtoC()
        self.second_model = ModelCtoY()
        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]

        self.attr_loss_ratio = args.attr_loss_ratio

    def generate_predictions(self, inputs, attr_labels):
        attr_preds, aux_attr_preds = self.model1(inputs)
        class_preds = self.model2(attr_preds)
        aux_class_preds = self.model2(aux_attr_preds)

        return attr_preds, aux_attr_preds, class_preds, aux_class_preds

    
    def generate_loss(self, attr_preds, attr_labels, class_preds, class_labels, aux_class_preds, mask):
        class_loss = self.criterion(class_preds, class_labels)
        aux_class_loss = self.criterion(aux_class_preds, class_labels)
        class_loss += aux_class_loss * AUX_LOSS_RATIO
        
        attr_loss = 0
        for i in range(len(self.attr_criterion)):
            attr_loss += self.attr_criterion[i](
                attr_preds[i].squeeze(), attr_labels[mask, i] # Masking attr losses
            )
        
        attr_loss /= len(self.attr_criterion)
        loss = (attr_loss * self.attr_loss_ratio) + class_loss
        return loss

class IndependentModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.first_model = ModelXtoC()
        self.second_model = ModelCtoY()
        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]

        if args.exp == "Concept_XtoC":
            self.train_mode = "XtoC"
        elif args.exp == "Independent_CtoY":
            self.train_mode = "CtoY"
        else:
            raise ValueError(f"Invalid experiment name {args.exp} for IndependentModel")

        self.attr_loss_ratio = args.attr_loss_ratio

    
    def generate_predictions(self, inputs, attr_labels, mask):
        if self.train_mode == "XtoC":
            attr_preds, aux_attr_preds = self.first_model(inputs[mask])
        else:
            attr_preds, aux_attr_preds = None, None

        if self.train_mode == "CtoY":
            class_preds = self.second_model(attr_labels[mask])
        else:
            class_preds = None
        
        aux_class_preds = None

        return attr_preds, aux_attr_preds, class_preds, aux_class_preds

    
    def generate_loss(self, attr_preds, attr_labels, class_preds, class_labels, aux_class_preds):
        attr_loss = 0
        if self.train_mode == "XtoC":
            for i in range(len(self.attr_criterion)):
                attr_loss += self.attr_criterion[i](
                    attr_preds[i].squeeze(), attr_labels[:, i]
                )
        
            attr_loss /= len(self.attr_criterion)
        
        if self.train_mode == "CtoY":
            class_loss = self.criterion(class_preds, class_labels)
            aux_class_loss = self.criterion(aux_class_preds, class_labels)
            class_loss += aux_class_loss * AUX_LOSS_RATIO
        else:
            class_loss = 0

        loss = (attr_loss * self.attr_loss_ratio) + class_loss
        return loss

