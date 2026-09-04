"""Utilities for injecting LoRA layers into Transformer models."""

from typing import List

import torch
import torch.nn as nn

from .lora import LoRALinear


def convert_linear_to_lora(
    linear_layer: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
) -> LoRALinear:
    """
    Convert a standard nn.Linear layer into a LoRALinear layer.

    The pretrained weights are copied into the frozen base layer.
    """

    lora_layer = LoRALinear(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        rank=rank,
        alpha=alpha,
        bias=linear_layer.bias is not None,
    )

    # Match the original layer's device and dtype.
    lora_layer = lora_layer.to(
        device=linear_layer.weight.device,
        dtype=linear_layer.weight.dtype,
    )

    with torch.no_grad():
        lora_layer.linear.weight.copy_(
            linear_layer.weight
        )

        if linear_layer.bias is not None:
            lora_layer.linear.bias.copy_(
                linear_layer.bias
            )

    return lora_layer


def inject_lora_into_distilbert(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: tuple[str, ...] = (
        "q_lin",
        "v_lin",
    ),
) -> List[str]:
    """
    Inject LoRA into selected DistilBERT attention projections.

    Default targets:
        - q_lin
        - v_lin

    Returns the names of all replaced modules.
    """

    replaced_layers = []

    transformer_layers = (
        model.distilbert
        .transformer
        .layer
    )

    for layer_index, layer in enumerate(
        transformer_layers
    ):
        attention = layer.attention

        for target_name in target_modules:
            if not hasattr(
                attention,
                target_name,
            ):
                raise AttributeError(
                    f"Attention module has no "
                    f"'{target_name}' layer."
                )

            original_layer = getattr(
                attention,
                target_name,
            )

            if not isinstance(
                original_layer,
                nn.Linear,
            ):
                raise TypeError(
                    f"{target_name} must be nn.Linear "
                    f"before LoRA injection."
                )

            lora_layer = convert_linear_to_lora(
                original_layer,
                rank=rank,
                alpha=alpha,
            )

            setattr(
                attention,
                target_name,
                lora_layer,
            )

            replaced_layers.append(
                f"layer.{layer_index}."
                f"attention.{target_name}"
            )

    return replaced_layers


def freeze_base_model(
    model: nn.Module,
) -> None:
    """
    Freeze every parameter in the model.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False


def enable_lora_parameters(
    model: nn.Module,
) -> None:
    """
    Enable gradients only for LoRA matrices.
    """

    for name, parameter in model.named_parameters():
        if (
            "lora_A" in name
            or "lora_B" in name
        ):
            parameter.requires_grad = True


def enable_classification_head(
    model: nn.Module,
) -> None:
    """
    Enable DistilBERT's task-specific classification head.
    """

    for parameter in (
        model.pre_classifier.parameters()
    ):
        parameter.requires_grad = True

    for parameter in (
        model.classifier.parameters()
    ):
        parameter.requires_grad = True


def prepare_distilbert_for_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: tuple[str, ...] = (
        "q_lin",
        "v_lin",
    ),
) -> List[str]:
    """
    Complete LoRA setup for DistilBERT classification.

    1. Freeze all pretrained weights.
    2. Inject custom LoRA layers.
    3. Enable LoRA parameters.
    4. Enable the classification head.
    """

    freeze_base_model(model)

    replaced_layers = (
        inject_lora_into_distilbert(
            model,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
        )
    )

    enable_lora_parameters(model)
    enable_classification_head(model)

    return replaced_layers