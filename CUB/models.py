"""
Taken from yewsiang/ConceptBottlenecks
"""
import os
from typing import Optional, List

import torch
from torch import nn

from CUB.analysis import accuracy
from CUB.template_model import MLP, inception_v3, End2EndModel
from CUB.cub_classes import Experiment
from CUB.dataset import find_class_imbalance
from CUB.config import BASE_DIR

class Multimodel(nn.Module):
    def __init__(self, args: Experiment):
        super().__init__()
        self.args = args
        self.pre_models: nn.ModuleList
        self.post_models: nn.ModuleList
        self.reset_pre_models()
        self.reset_post_models()

    def reset_pre_models(self, pretrained: Optional[bool] = None) -> None:
        if pretrained is None:
            use_pretrained = self.args.pretrained
        else:
            use_pretrained = pretrained

        pre_models_list = [
            ModelXtoC(
                pretrained=use_pretrained,
                freeze=self.args.freeze,
                use_aux=self.args.use_aux,
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
            ModelOracleCtoY(
                n_class_attr=self.args.n_class_attr,
                n_attributes=self.args.n_attributes,
                num_classes=self.args.num_classes,
                expand_dim=self.args.expand_dim,
            )
            for _ in range(self.args.n_models)
        ]

        self.post_models = nn.ModuleList(post_models_list)


# class SharedMultimodel(nn.Module):
#     def __init__(self, args: Experiment):
#         super().__init__()
#         self.args = args
#         self.pre_models: nn.ModuleList
#         self.post_models: nn.ModuleList
#         self.reset_pre_models()
#         self.reset_post_models()

#     def reset_pre_models(self, pretrained: Optional[bool] = None) -> None:
#         if pretrained is None:
#             use_pretrained = self.args.pretrained
#         else:
#             use_pretrained = pretrained

#         pre_models_list = [
#             ModelXtoC(
#                 pretrained=use_pretrained,
#                 freeze=self.args.freeze,
#                 use_aux=self.args.use_aux,
#                 num_classes=self.args.num_classes,
#                 n_attributes=self.args.n_attributes,
#                 expand_dim=self.args.expand_dim,
#                 three_class=self.args.three_class,
#             )
#             for _ in range(self.args.n_models)
#         ]
#         self.pre_models = nn.ModuleList(pre_models_list)

#     def reset_post_models(self) -> None:
#         post_models_list = [
#             ModelOracleCtoY(
#                 n_class_attr=self.args.n_class_attr,
#                 n_attributes=self.args.n_attributes,
#                 num_classes=self.args.num_classes,
#                 expand_dim=self.args.expand_dim,
#             )
#             for _ in range(self.args.n_models)
#         ]

#         self.post_models = nn.ModuleList(post_models_list)


# Independent & Sequential Model
def ModelXtoC(
    pretrained: bool,
    freeze: bool,
    use_aux: bool,
    num_classes: int,
    n_attributes: int,
    expand_dim: int,
    three_class: bool,
) -> nn.Module:
    return inception_v3(
        pretrained=pretrained,
        freeze=freeze,
        aux_logits=use_aux,
        num_classes=num_classes,
        n_attributes=n_attributes,
        bottleneck=True,
        expand_dim=expand_dim,
        three_class=three_class,
    )


# Independent Model
def ModelOracleCtoY(
    n_class_attr: int, n_attributes: int, num_classes: int, expand_dim: int
) -> nn.Module:
    # X -> C part is separate, this is only the C -> Y part
    if n_class_attr == 3:
        model = MLP(
            input_dim=n_attributes * n_class_attr,
            num_classes=num_classes,
            expand_dim=expand_dim,
        )
    else:
        model = MLP(
            input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim
        )
    return model


# Sequential Model
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)


# Joint Model
def ModelXtoCtoY(
    n_class_attr,
    pretrained,
    freeze,
    use_aux,
    num_classes,
    n_attributes,
    expand_dim,
    use_relu,
    use_sigmoid,
):
    model1 = inception_v3(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        n_attributes=n_attributes,
        aux_logits=use_aux,
        bottleneck=True, # BASTARDSS!!!! this is a different bottleneck flag than the one in the other model
        expand_dim=expand_dim,
        three_class=(n_class_attr == 3),
    )
    if n_class_attr == 3:
        model2 = MLP(
            input_dim=n_attributes * n_class_attr,
            num_classes=num_classes,
            expand_dim=expand_dim,
        )
    else:
        model2 = MLP(
            input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim
        )
    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)


# Standard Model
def ModelXtoY(pretrained, freeze, num_classes, use_aux):
    return inception_v3(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        aux_logits=use_aux,
    )


# Multitask Model
def ModelXtoCY(
    pretrained, use_aux, freeze, num_classes, n_attributes, three_class, connect_CY
):
    return inception_v3(
        pretrained=pretrained,
        aux_logits=use_aux,
        freeze=freeze,
        num_classes=num_classes,
        n_attributes=n_attributes,
        bottleneck=False,
        three_class=three_class,
        connect_CY=connect_CY,
    )


def make_weighted_criteria(args):
    attr_criterion = []
    train_data_path = os.path.join(BASE_DIR, args.data_dir, "train.pkl")
    imbalance = find_class_imbalance(train_data_path, True) # assume args.weighted loss is always "multiple" if not ""
    for ratio in imbalance:
        attr_criterion.append(
            torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda())
        )


class JointModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.first_model = ModelXtoC()
        self.second_model = MLP(
            input_dim=args.n_attributes, num_classes=args.num_classes, expand_dim=args.expand_dim
        )


class IndependentModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.first_model = ModelXtoC()
        self.second_model = ModelOracleCtoY()
        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]

        self.train_mode: str = "XtoC"
    
    def to_first_stage_mode(self):
        self.train_mode = "XtoC"
    
    def to_second_stage_mode(self):
        self.train_mode = "CtoY"

    
    def run_batch(self, batch):
        inputs, labels, attr_labels = batch
        if self.train_mode == "XtoC":
            outputs = self.first_model(inputs)
            loss = self.criterion(outputs, attr_labels)
            _, preds = torch.max(outputs, 1)
            return loss, preds

        if self.train_mode == "CtoY":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            acc = accuracy(outputs, labels, topk=(1,))
            if self.is_training:
                self.optimizer.zero_grad()  # zero the parameter gradients
                loss.backward()
                self.optimizer.step()  # optimizer step to update parameters

            return loss, preds

