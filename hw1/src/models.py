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

    # Every layer can be trained
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


# Parts are listed from the head towards the input, in unfreezing order.
# Prefixes end with "." so that "features.1." does not match "features.10."
UNFREEZE_GROUPS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "mobilenet_v3_small": [
        ("head", ("classifier.",)),
        ("late", tuple(f"features.{i}." for i in range(9, 13))),
        ("mid", tuple(f"features.{i}." for i in range(4, 9))),
        ("early", tuple(f"features.{i}." for i in range(0, 4))),
    ],
    "shufflenet_v2_x1_0": [
        ("head", ("fc.",)),
        ("late", ("stage4.", "conv5.")),
        ("mid", ("stage3.",)),
        ("early", ("conv1.", "stage2.")),
    ],
}


def get_unfreeze_groups(
    model_name: str,
) -> list[tuple[str, tuple[str, ...]]]:
    """Return (part name, parameter name prefixes), head first."""
    if model_name not in UNFREEZE_GROUPS:
        raise ValueError(f"No unfreezing plan for model: {model_name}")
    return UNFREEZE_GROUPS[model_name]


def assert_groups_cover_model(model: nn.Module, model_name: str) -> None:
    """Check that every parameter belongs to exactly one part."""
    prefixes = [
        prefix
        for _, group_prefixes in get_unfreeze_groups(model_name)
        for prefix in group_prefixes
    ]
    for name, _ in model.named_parameters():
        matches = sum(name.startswith(prefix) for prefix in prefixes)
        if matches != 1:
            raise ValueError(
                f"Parameter {name} matches {matches} unfreezing groups"
            )


def freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_group(
    model: nn.Module,
    prefixes: tuple[str, ...],
) -> list[nn.Parameter]:
    """Make parameters with the given name prefixes trainable."""
    parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True
            parameters.append(parameter)

    if not parameters:
        raise ValueError(f"No parameters found for prefixes {prefixes}")

    return parameters
