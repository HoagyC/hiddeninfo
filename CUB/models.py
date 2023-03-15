"""
Taken from yewsiang/ConceptBottlenecks
"""
import os
from typing import Optional, List, Tuple, Iterable
from abc import ABC, abstractmethod

import torch
from torch import nn

from CUB.model_templates import MLP, inception_v3, FC
from CUB.cub_classes import Experiment
from CUB.dataset import find_class_imbalance

# Create loss criteria for each attribute, upweighting the less common ones
def make_weighted_criteria(args):
    attr_criterion = []
    train_data_path = os.path.join(args.base_dir, args.data_dir, "train.pkl")
    imbalance = find_class_imbalance(train_data_path, True) # assume args.weighted loss is always "multiple" if not ""
    for ratio in imbalance[:args.n_attributes]:
        attr_criterion.append(
            torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda())
        )
    return attr_criterion


def unsqueeze(x: Optional[torch.Tensor], dim: int = 0):
    if x is None:
        return None
    return x.unsqueeze(dim)

def squeeze(x: Optional[torch.Tensor], dim: int = 0):
    if x is None:
        return None
    return x.squeeze(dim)

# Basic model for predicting attributes from images
def ModelXtoC(args: Experiment) -> nn.Module:
    """
    Model for predicting attributes from images.
    Takes in an image and outputs a list of outputs for each attribute,
    where the output is a vector of size (batch_size, 1).
    """
    return inception_v3(
        pretrained=args.pretrained,
        freeze=False,
        aux_logits=True,
        num_classes=args.num_classes,
        n_attributes=args.n_attributes,
        expand_dim=args.expand_dim,
        thin_models=args.n_models if args.thin else 0
    )

# Basic model for predicting classes from attributes
def ModelCtoY(args: Experiment) -> nn.Module:
    model = MLP(
        input_dim=args.n_attributes, num_classes=args.num_classes, expand_dim=args.expand_dim, post_model_dropout=args.post_model_dropout
    )
    return model


# Base class that all models inherit from
class CUB_Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.train_mode: str

    @abstractmethod
    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through=None) -> \
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None, None, None

    @abstractmethod
    def generate_loss(self, 
        attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        attr_labels: Optional[torch.Tensor], # batch_size x n_attributes
        class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        class_labels: Optional[torch.Tensor], # batch_size
        aux_class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        aux_attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        mask: torch.Tensor,):
        pass

class CUB_Multimodel(CUB_Model):
    def __init__(self):
        super().__init__()
        self.pre_models: nn.ModuleList
        self.post_models: nn.ModuleList

        
    @abstractmethod
    def reset_pre_models(self) -> None:
        pass

    @abstractmethod
    def reset_post_models(self):
        pass


class Multimodel(CUB_Multimodel):
    def __init__(self, args: Experiment):
        super().__init__()
        self.args = args
        self.pre_models: nn.ModuleList
        self.post_models: nn.ModuleList
        self.reset_pre_models()
        self.reset_post_models()
        self.train_mode: str = "separate"
        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]
        self.av_diff_same: List[torch.Tensor] = []
        self.av_diff_switch: List[torch.Tensor] = []

    def reset_pre_models(self) -> None:
        pre_models_list = [
            ModelXtoC(self.args)
            for _ in range(self.args.n_models)
        ]
        self.pre_models = nn.ModuleList(pre_models_list)

    def reset_post_models(self) -> None:
        post_models_list = [
            nn.Sequential(nn.Sigmoid(), ModelCtoY(self.args))
            for _ in range(self.args.n_models)
        ]

        self.post_models = nn.ModuleList(post_models_list)

    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through=None):
        attr_preds = []
        aux_attr_preds = []
        class_preds = []
        aux_class_preds = []
        post_model_indices: Iterable
        # Train each pre model with its own post model
        if self.train_mode == "separate":
            post_model_indices = range(self.args.n_models)
        
        # Randomly shuffle which post model is used for each pre model
        elif self.train_mode == "shuffle":
            post_model_indices = torch.randperm(self.args.n_models)
            
        else:
            raise ValueError(f"Invalid train mode {self.train_mode}")
        
        if not hasattr(self, "av_diff_same"):
            self.av_diff_same = []
            self.av_diff_switch = []
        
        for i, j in enumerate(post_model_indices):
            attr_pred, aux_attr_pred = self.pre_models[i](inputs)
            other_attr_pred, other_aux_attr_pred = self.pre_models[j](inputs)

            attr_pred_input = torch.cat(attr_pred, dim=1)
            aux_attr_pred_input = torch.cat(aux_attr_pred, dim=1)

            other_attr_pred_input = torch.cat(other_attr_pred, dim=1)
            other_aux_attr_pred_input = torch.cat(other_aux_attr_pred, dim=1)

            diff = torch.sum(torch.abs(attr_pred_input - other_attr_pred_input))
            if self.training:
                if i == j:
                    self.av_diff_same.append(diff.detach())
                else:
                    self.av_diff_switch.append(diff.detach())
            
            if len(self.av_diff_switch) % 100 == 0 and self.training and len(self.av_diff_switch) > 0:
                print(f"Average diff same: {torch.mean(torch.stack(self.av_diff_same))}")
                print(f"Average diff switch: {torch.mean(torch.stack(self.av_diff_switch))}")

            class_pred = self.post_models[j](attr_pred_input)
            aux_class_pred = self.post_models[j](aux_attr_pred_input)
    
            attr_preds.append(attr_pred_input)
            aux_attr_preds.append(aux_attr_pred_input)
            class_preds.append(class_pred)
            aux_class_preds.append(aux_class_pred)

        return torch.stack(attr_preds), torch.stack(aux_attr_preds), torch.stack(class_preds), torch.stack(aux_class_preds)
    
    def generate_loss(
        self, 
        attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        attr_labels: Optional[torch.Tensor], # batch_size x n_attributes
        class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        class_labels: Optional[torch.Tensor], # batch_size
        aux_class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        aux_attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        mask: torch.Tensor,
    ):
        total_class_loss = 0.
        total_attr_loss = 0.
        
        assert attr_preds is not None and attr_labels is not None and aux_attr_preds is not None
        assert class_preds is not None and class_labels is not None and aux_class_preds is not None

        assert len(attr_preds) == len(aux_attr_preds) == len(class_preds) == len(aux_class_preds) == len(self.pre_models)
        for ndx in range(len(self.pre_models)):
            class_loss = self.criterion(class_preds[ndx], class_labels)
            aux_class_loss = self.criterion(aux_class_preds[ndx], class_labels)
            class_loss += aux_class_loss * self.args.aux_loss_ratio
            total_class_loss += class_loss
            
            attr_loss = 0.
            for i in range(len(self.attr_criterion)):
                attr_loss += self.attr_criterion[i](
                    attr_preds[ndx, mask, i], attr_labels[mask, i] # Masking attr losses
                )

                aux_attr_loss = self.attr_criterion[i](
                    aux_attr_preds[ndx, mask, i], attr_labels[mask, i] # Masking attr losses
                )
                attr_loss += aux_attr_loss * self.args.aux_loss_ratio
            
            attr_loss /= len(self.attr_criterion)
            total_attr_loss += attr_loss
    
        loss = (total_attr_loss * self.args.attr_loss_weight) + total_class_loss
        return loss
    

class SequentialModel(CUB_Model):
    def __init__(self, args: Experiment, train_mode: str) -> None:
        super().__init__()
        self.args = args
        self.first_model = ModelXtoC(self.args)
        self.second_model = nn.Sequential(nn.Sigmoid(), ModelCtoY(self.args))
        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]

        self.attr_loss_weight = args.attr_loss_weight
        self.train_mode = train_mode

    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through=None):
        attr_preds, aux_attr_preds = self.first_model(inputs[mask])

        # Attr preds are list of tensors, need to concat them with batch as d0
        # Detach to prevent gradients from flowing back to first model
        attr_preds_input = torch.cat(attr_preds, dim=1)
        aux_attr_preds_input = torch.cat(aux_attr_preds, dim=1)

        if self.train_mode == "CtoY":
            class_preds = self.second_model(attr_preds_input.detach())
            aux_class_preds = self.second_model(aux_attr_preds_input.detach())
            
        else:
            class_preds, aux_class_preds = None, None
        
        return unsqueeze(attr_preds_input), unsqueeze(aux_attr_preds_input), unsqueeze(class_preds), unsqueeze(aux_class_preds)

    def generate_loss(
        self,
        attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        attr_labels: Optional[torch.Tensor], # batch_size x n_attributes
        class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        class_labels: Optional[torch.Tensor], # batch_size
        aux_class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        aux_attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        mask: torch.Tensor, # batch_size
    ):
        attr_preds = squeeze(attr_preds) # Removing the n_models dimension
        aux_attr_preds = squeeze(aux_attr_preds)
        class_preds = squeeze(class_preds)
        aux_class_preds = squeeze(aux_class_preds)

        attr_loss = 0.
        if self.train_mode == "XtoC":
            assert attr_preds is not None and attr_labels is not None and aux_attr_preds is not None
            
            for i in range(len(self.attr_criterion)):
                attr_loss += self.attr_criterion[i](
                    attr_preds[mask, i], attr_labels[mask, i]
                )
        
            attr_loss /= len(self.attr_criterion)

        if self.train_mode == "CtoY":
            assert class_preds is not None and class_labels is not None and aux_class_preds is not None

            class_loss = self.criterion(class_preds, class_labels[mask])
        else:
            class_loss = 0

        loss = (attr_loss * self.attr_loss_weight) + class_loss
        return loss

class JointModel(CUB_Model):
    def __init__(self, args: Experiment) -> None:
        super().__init__()
        self.args = args
        self.first_model = ModelXtoC(self.args)

        self.second_model: nn.Module
        if self.args.model_sigmoid:
            self.second_model = nn.Sequential(nn.Sigmoid(), ModelCtoY(self.args))
        else:
            self.second_model = ModelCtoY(self.args)

        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]

        self.attr_loss_weight = args.attr_loss_weight
        self.train_mode = "joint"
        print(self.second_model)

    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through=None):
        attr_preds, aux_attr_preds = self.first_model(inputs)

        # Attr preds are list of tensors, need to concat them with batch as d0
        attr_preds_input = torch.cat(attr_preds, dim=1)
        aux_attr_preds_input = torch.cat(aux_attr_preds, dim=1)

        if self.args.gen_pred_sigmoid:
            # Apply a sigmoid to the input tensors
            attr_preds_input = torch.sigmoid(attr_preds_input)
            aux_attr_preds_input = torch.sigmoid(aux_attr_preds_input)
    
        class_preds = self.second_model(attr_preds_input)
        aux_class_preds = self.second_model(aux_attr_preds_input)

        # Add extra dimension for consistency with multimodels
        return unsqueeze(torch.cat(attr_preds, dim=1)), unsqueeze(torch.cat(aux_attr_preds, dim=1)), unsqueeze(class_preds), unsqueeze(aux_class_preds)

    def generate_loss(
        self,
        attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        attr_labels: Optional[torch.Tensor], # batch_size x n_attributes
        class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        class_labels: Optional[torch.Tensor], # batch_size
        aux_class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        aux_attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        mask: torch.Tensor, # batch_size
    ):
        attr_preds = squeeze(attr_preds) # Removing the n_models dimension
        aux_attr_preds = squeeze(aux_attr_preds)
        class_preds = squeeze(class_preds)
        aux_class_preds = squeeze(aux_class_preds)

        assert attr_preds is not None and attr_labels is not None and aux_attr_preds is not None
        assert class_preds is not None and class_labels is not None and aux_class_preds is not None

        class_loss = self.criterion(class_preds, class_labels)
        aux_class_loss = self.criterion(aux_class_preds, class_labels)
        class_loss += aux_class_loss * self.args.aux_loss_ratio
        
        attr_loss = 0.
        for i in range(len(self.attr_criterion)):
            attr_loss += self.attr_criterion[i](
                attr_preds[mask, i], attr_labels[mask, i] # Masking attr losses
            )

            aux_attr_loss = self.attr_criterion[i](
                aux_attr_preds[mask, i], attr_labels[mask, i] # Masking attr losses
            )
            attr_loss += aux_attr_loss * self.args.aux_loss_ratio

        
        attr_loss /= len(self.attr_criterion)
        loss = (attr_loss * self.attr_loss_weight) + class_loss
        return loss


class IndependentModel(CUB_Model):
    def __init__(self, args: Experiment, train_mode: str) -> None:
        super().__init__()
        self.args = args
        self.first_model = ModelXtoC(self.args)
        self.second_model = ModelCtoY(self.args)
        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]

        self.attr_loss_weight = args.attr_loss_weight
        self.train_mode = train_mode
        self.train()

    
    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through: bool = False):
        if straight_through:
            attr_preds, aux_attr_preds = self.first_model(inputs)

            # Attr preds are list of tensors, need to concat them with batch as d0
            attr_preds = torch.cat(attr_preds, dim=1)
            aux_attr_preds = torch.cat(aux_attr_preds, dim=1)

            # Apply a sigmoid to the input tensors
            attr_preds_input = torch.sigmoid(attr_preds)
            aux_attr_preds_input = torch.sigmoid(aux_attr_preds)

            class_preds = self.second_model(attr_preds_input)
            aux_class_preds = self.second_model(aux_attr_preds_input)


        else:
            if self.train_mode == "XtoC":
                try:
                    attr_preds, aux_attr_preds = self.first_model(inputs[mask])
                    attr_preds = torch.cat(attr_preds, dim=1)
                    aux_attr_preds = torch.cat(aux_attr_preds, dim=1)
                except:
                    import pdb; pdb.set_trace()
            else:
                attr_preds, aux_attr_preds = None, None

            if self.train_mode == "CtoY":
                class_preds = self.second_model(attr_labels[mask, :self.args.n_attributes])
            else:
                class_preds = None
            
            aux_class_preds = None

        return unsqueeze(attr_preds), unsqueeze(aux_attr_preds), unsqueeze(class_preds), unsqueeze(aux_class_preds)
    
    def generate_loss(
        self,
        attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        attr_labels: Optional[torch.Tensor], # batch_size x n_attributes
        class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        class_labels: Optional[torch.Tensor], # batch_size
        aux_class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        aux_attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        mask: torch.Tensor, # batch_size
    ):
        attr_preds = squeeze(attr_preds) # Removing the n_models dimension
        aux_attr_preds = squeeze(aux_attr_preds)
        class_preds = squeeze(class_preds)
        aux_class_preds = squeeze(aux_class_preds)

        assert attr_preds is not None and attr_labels is not None and aux_attr_preds is not None
        assert class_preds is not None and class_labels is not None and aux_class_preds is not None

        attr_loss = 0.
        if self.train_mode == "XtoC":
            for i in range(len(self.attr_criterion)):
                attr_loss += self.attr_criterion[i](
                    attr_preds[mask, i], attr_labels[mask, i]
                )
        
            attr_loss /= len(self.attr_criterion)
        
        if self.train_mode == "CtoY":
            class_loss = self.criterion(class_preds, class_labels[mask])
        else:
            class_loss = 0

        loss = (attr_loss * self.attr_loss_weight) + class_loss
        return loss

class ThinMultimodel(CUB_Multimodel):
    def __init__(self, args: Experiment):
        super().__init__()
        self.args = args
        self.pre_models = nn.ModuleList([ModelXtoC(self.args)]) # This creates a model which is the same except for the last layer, which there is multiple copies of
        self.reset_post_models() # Creates post models
        self.train_mode = "separate"
        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]


    def reset_pre_models(self) -> None:
        n_fcs = self.args.n_attributes * self.args.n_models
        new_fcs = nn.ModuleList()
        for _ in range(n_fcs):
            new_fcs.append(FC(2048, 1, self.args.expand_dim))
        self.pre_models.all_fc = new_fcs


    def reset_post_models(self) -> None:
        post_models_list = [
            nn.Sequential(
                nn.Sigmoid(),
                ModelCtoY(self.args)
            )
            for _ in range(self.args.n_models)
        ]

        self.post_models = nn.ModuleList(post_models_list)

    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through: bool = False):
        attr_preds = []
        aux_attr_preds = []
        class_preds = []
        aux_class_preds = []
        post_model_indices: Iterable

        attr_preds, aux_attr_preds = self.pre_models(inputs)

        # Train each pre model with its own post model
        if self.train_mode == "separate":
            post_model_indices = list(range(self.args.n_models))
        
        # Randomly shuffle which post model is used for each pre model
        elif self.train_mode == "shuffle":
            post_model_indices = torch.randperm(self.args.n_models)
        else:
            raise ValueError(f"Invalid train mode {self.train_mode}")

        for i, j in enumerate(post_model_indices):
            attr_pred_input = torch.cat(attr_preds[i], dim=1)
            aux_attr_pred_input = torch.cat(aux_attr_preds[i], dim=1)

            class_pred = self.post_models[j](attr_pred_input)
            aux_class_pred = self.post_models[j](aux_attr_pred_input)
    
            class_preds.append(class_pred)
            aux_class_preds.append(aux_class_pred)
        
        return torch.stack(attr_preds), torch.stack(aux_attr_preds), torch.stack(class_preds), torch.stack(aux_class_preds)
    
    def generate_loss(
        self, 
        attr_preds: Optional[torch.Tensor],
        attr_labels: Optional[torch.Tensor],
        class_preds: Optional[torch.Tensor], 
        class_labels: Optional[torch.Tensor], 
        aux_class_preds: Optional[torch.Tensor], 
        aux_attr_preds: Optional[torch.Tensor],
        mask: torch.Tensor,
    ):
        total_class_loss = 0.
        total_attr_loss = 0.

        assert attr_preds is not None and attr_labels is not None and aux_attr_preds is not None
        assert class_preds is not None and class_labels is not None and aux_class_preds is not None

        assert len(attr_preds) == len(aux_attr_preds) == len(class_preds) == len(aux_class_preds) == self.args.n_models

        for ndx in range(self.args.n_models):
            class_loss = self.criterion(class_preds[ndx], class_labels)
            aux_class_loss = self.criterion(aux_class_preds[ndx], class_labels)
            class_loss += aux_class_loss * self.args.aux_loss_ratio
            total_class_loss += class_loss
            
            attr_loss = 0.
            for i in range(len(self.attr_criterion)):
                attr_loss += self.attr_criterion[i](
                    attr_preds[ndx, i].squeeze(), attr_labels[mask, i] # Masking attr losses
                )

                aux_attr_loss = self.attr_criterion[i](
                    aux_attr_preds[ndx, i].squeeze(), attr_labels[mask, i] # Masking attr losses
                )
                attr_loss += aux_attr_loss * self.args.aux_loss_ratio
            
            attr_loss /= len(self.attr_criterion)
            total_attr_loss += attr_loss
    
        loss = (total_attr_loss * self.args.attr_loss_weight) + total_class_loss
        return loss