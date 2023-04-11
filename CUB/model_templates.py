"""
InceptionV3 Network modified from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
New changes: add softmax layer + option for freezing lower layers except fc
"""
import os
import sys
import time

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.base_models.inception import Inception3
from CUB.base_models.resnet import resnet50, ResNet50_Weights
from CUB.cub_classes import Experiment

__all__ = ["MLP", "Inception3", "inception_v3", "End2EndModel"]

model_urls = {
    # Downloaded inception model (optional)
    "downloaded": "pretrained/inception_v3_google-1a9a5a14.pth",
    # Inception v3 ported from TensorFlow
    "inception_v3_google": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth",
}

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim, post_model_dropout=None):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, expand_dim)
        self.activation = torch.nn.ReLU()
        self.linear2 = nn.Linear(
            expand_dim, num_classes
        )  # softmax is handled by loss function
        self.expand_dim = expand_dim
        self.post_model_dropout=post_model_dropout

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        if self.post_model_dropout:
            x = F.dropout(x, p=self.post_model_dropout, training=self.training)
        x = self.linear2(x)
        return x

def resnet50_model(args: Experiment, weight_n: int = 1):
    ## List of kwargs that are used in the inception model
    # n_attributes, # args.n_attributes
    # expand_dim, # args.expand_dim
    # aux_logits=True, # True
    # transform_input=False, # None
    # thin_models=0, # args.thin_models
    # use_dropout=True, # use_dropout
    assert args.use_aux == False
        
    if weight_n == 1:
        weights = ResNet50_Weights.IMAGENET1K_V1
    elif weight_n == 2:
        weights = ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None
    
    model = resnet50(
        num_classes=args.n_attributes,
        weights=weights,
    )
    return model

def inception_v3(pretrained, freeze, **kwargs):
    """Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        
        model = Inception3(**kwargs)
    
        if os.path.exists(model_urls.get("downloaded")):
            model.load_partial_state_dict(torch.load(model_urls["downloaded"]))
        else:
            model.load_partial_state_dict(
                model_zoo.load_url(model_urls["inception_v3_google"])
            )
        if freeze:  # only finetune fc layer
            for name, param in model.named_parameters():
                if "fc" not in name:  # and 'Mixed_7c' not in name:
                    param.requires_grad = False
        return model
    else:
        return Inception3(**kwargs)


class WideMLPOut(nn.Module):
    def __init__(self, input_dim, expand_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, expand_dim)
        self.activation = torch.nn.ReLU()
        self.linear2 = nn.Linear(expand_dim, output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)
        return [y for y in x.t()] # Returning a list of tensors to match the output of Inceptionv3