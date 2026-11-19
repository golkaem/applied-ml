"""Model and optimizer factories for the Caltech-256 baseline."""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
from torchvision import models


def build_model(model_name: str, num_classes: int) -> tuple[nn.Module, str]:
    """Create an ImageNet-pretrained model with a new classifier."""
    # Load ImageNet weights and replace the original head
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes,
        )
        classifier_prefix = "classifier."
    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(
            weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        classifier_prefix = "fc."
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Every layer can de trained
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model, classifier_prefix


def build_optimizer(
    model: nn.Module,
    classifier_prefix: str,
    backbone_lr: float,
    classifier_lr: float,
    weight_decay: float,
) -> optim.AdamW:
    """Create AdamW with separate learning rates for backbone and head."""
    backbone_parameters = []
    classifier_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith(classifier_prefix):
            classifier_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    parameter_groups = []

    # During iterative unfreezing the backbone may still be fully frozen
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
                "name": "backbone",
            }
        )

    if classifier_parameters:
        parameter_groups.append(
            {
                "params": classifier_parameters,
                "lr": classifier_lr,
                "name": "head",
            }
        )

    if not parameter_groups:
        raise ValueError("The model has no trainable parameters")

    return optim.AdamW(parameter_groups, weight_decay=weight_decay)
