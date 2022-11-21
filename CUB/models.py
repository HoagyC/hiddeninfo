"""
Taken from yewsiang/ConceptBottlenecks
"""
from typing import Optional

from torch import nn

from CUB.template_model import MLP, inception_v3, End2EndModel


class Multimodel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pre_models: List[torch.nn.Module]
        self.post_models: List[torch.nn.Module]
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


# Independent & Sequential Model
def ModelXtoC(
    pretrained: bool,
    freeze: bool,
    num_classes: int,
    n_attributes: int,
    expand_dim: int,
    three_class: bool,
) -> nn.Module:
    return inception_v3(
        pretrained=pretrained,
        freeze=freeze,
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
        bottleneck=True,
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
def ModelXtoY(pretrained, freeze, num_classes):
    return inception_v3(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
    )


# Multitask Model
def ModelXtoCY(pretrained, freeze, num_classes, n_attributes, three_class, connect_CY):
    return inception_v3(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        n_attributes=n_attributes,
        bottleneck=False,
        three_class=three_class,
        connect_CY=connect_CY,
    )
