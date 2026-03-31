from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    efficientnet_b2,
    mobilenet_v3_large,
    resnet18,
    resnet50,
)


TrainStage = Literal["head_only", "partial_finetune", "full_finetune"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    builder: Callable[..., nn.Module]
    default_weights: Any
    head_kind: Literal["fc", "classifier"]
    partial_tail_modules: int = 1


SUPPORTED_MODELS: Dict[str, ModelSpec] = {
    "resnet18_pretrained": ModelSpec(
        name="resnet18_pretrained",
        family="resnet18",
        builder=resnet18,
        default_weights=ResNet18_Weights.DEFAULT,
        head_kind="fc",
        partial_tail_modules=1,
    ),
    "mobilenet_v3_large_pretrained": ModelSpec(
        name="mobilenet_v3_large_pretrained",
        family="mobilenet_v3_large",
        builder=mobilenet_v3_large,
        default_weights=MobileNet_V3_Large_Weights.DEFAULT,
        head_kind="classifier",
        partial_tail_modules=2,
    ),
    "efficientnet_b0_pretrained": ModelSpec(
        name="efficientnet_b0_pretrained",
        family="efficientnet_b0",
        builder=efficientnet_b0,
        default_weights=EfficientNet_B0_Weights.DEFAULT,
        head_kind="classifier",
        partial_tail_modules=2,
    ),
    "resnet50_pretrained": ModelSpec(
        name="resnet50_pretrained",
        family="resnet50",
        builder=resnet50,
        default_weights=ResNet50_Weights.DEFAULT,
        head_kind="fc",
        partial_tail_modules=1,
    ),
    "efficientnet_b2_pretrained": ModelSpec(
        name="efficientnet_b2_pretrained",
        family="efficientnet_b2",
        builder=efficientnet_b2,
        default_weights=EfficientNet_B2_Weights.DEFAULT,
        head_kind="classifier",
        partial_tail_modules=2,
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    name = model_name.strip().lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported pretrained model: {model_name}")
    return SUPPORTED_MODELS[name]


def list_available_models() -> Dict[str, str]:
    return {
        "resnet18_pretrained": "ResNet18 transfer-learning baseline",
        "mobilenet_v3_large_pretrained": "MobileNetV3-Large transfer-learning baseline",
        "efficientnet_b0_pretrained": "EfficientNet-B0 transfer-learning baseline",
        "resnet50_pretrained": "ResNet50 transfer-learning baseline",
        "efficientnet_b2_pretrained": "EfficientNet-B2 transfer-learning baseline",
    }


def resolve_weights(model_name: str, pretrained: bool = True) -> Any:
    spec = get_model_spec(model_name)
    return spec.default_weights if pretrained else None


def get_weights_name(model_name: str, pretrained: bool = True) -> str:
    weights = resolve_weights(model_name=model_name, pretrained=pretrained)
    if weights is None:
        return "None"
    return str(getattr(weights, "name", "DEFAULT"))


def _find_last_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, nn.Sequential):
        for child in reversed(list(module.children())):
            if isinstance(child, nn.Linear):
                return child
    raise ValueError(f"Could not find a final nn.Linear inside module type: {type(module).__name__}")


def get_head_module(model: nn.Module, model_name: str) -> nn.Module:
    spec = get_model_spec(model_name)
    if spec.head_kind == "fc":
        return model.fc
    if spec.head_kind == "classifier":
        return model.classifier
    raise ValueError(f"Unsupported head kind: {spec.head_kind}")


def replace_classifier_head(model: nn.Module, model_name: str, num_classes: int = 3, dropout_p: float = 0.3) -> nn.Module:
    spec = get_model_spec(model_name)

    if spec.head_kind == "fc":
        in_features = int(model.fc.in_features)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    if spec.head_kind == "classifier":
        old_head = get_head_module(model, model_name)
        in_features = int(_find_last_linear(old_head).in_features)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported head kind: {spec.head_kind}")


def build_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    dropout_p: float = 0.3,
) -> nn.Module:
    spec = get_model_spec(model_name)
    weights = resolve_weights(model_name=model_name, pretrained=pretrained)
    model = spec.builder(weights=weights)
    model = replace_classifier_head(model=model, model_name=model_name, num_classes=num_classes, dropout_p=dropout_p)
    return model


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _get_partial_backbone_modules(model: nn.Module, model_name: str) -> List[nn.Module]:
    spec = get_model_spec(model_name)

    if spec.head_kind == "fc":
        return [model.layer4]

    if not hasattr(model, "features"):
        raise ValueError(f"Model {model_name} does not expose a .features module for partial fine-tuning.")

    features = list(model.features.children())
    if not features:
        raise ValueError(f"Model {model_name} has an empty .features container.")

    tail = max(1, min(spec.partial_tail_modules, len(features)))
    return features[-tail:]


def configure_trainable_stage(model: nn.Module, model_name: str, stage: TrainStage) -> nn.Module:
    _freeze_all(model)
    _unfreeze_module(get_head_module(model, model_name))

    if stage == "head_only":
        return model

    if stage == "partial_finetune":
        for module in _get_partial_backbone_modules(model, model_name):
            _unfreeze_module(module)
        return model

    if stage == "full_finetune":
        for param in model.parameters():
            param.requires_grad = True
        return model

    raise ValueError(f"Unsupported training stage: {stage}")


def get_trainable_parameter_groups(
    model: nn.Module,
    model_name: str,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    head_ids = {id(p) for p in get_head_module(model, model_name).parameters() if p.requires_grad}
    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in head_ids:
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups: List[Dict[str, Any]] = []
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": float(backbone_lr),
                "weight_decay": float(weight_decay),
                "name": "backbone",
            }
        )
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": float(head_lr),
                "weight_decay": float(weight_decay),
                "name": "head",
            }
        )
    return groups


def iter_trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for param in model.parameters():
        if param.requires_grad:
            yield param

