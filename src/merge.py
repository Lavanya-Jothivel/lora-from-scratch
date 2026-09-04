"""Utilities for merging LoRA adapters into base linear layers."""

import copy

import torch
import torch.nn as nn

from .lora import LoRALinear


def merge_lora_layer(
    lora_layer: LoRALinear,
) -> nn.Linear:
    """
    Merge a LoRALinear layer into a standard nn.Linear layer.

    W_merged = W0 + (alpha / rank) * (B @ A)
    """

    if not isinstance(
        lora_layer,
        LoRALinear,
    ):
        raise TypeError(
            "Expected a LoRALinear layer."
        )

    merged_layer = nn.Linear(
        lora_layer.in_features,
        lora_layer.out_features,
        bias=lora_layer.linear.bias is not None,
    )

    merged_layer = merged_layer.to(
        device=lora_layer.linear.weight.device,
        dtype=lora_layer.linear.weight.dtype,
    )

    with torch.no_grad():

        merged_weight = (
            lora_layer.linear.weight
            + lora_layer.delta_weight()
        )

        merged_layer.weight.copy_(
            merged_weight
        )

        if lora_layer.linear.bias is not None:
            merged_layer.bias.copy_(
                lora_layer.linear.bias
            )

    return merged_layer


def merge_distilbert_lora(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    Merge all LoRA query/value projections in DistilBERT.

    By default, a copy of the model is created so the original
    LoRA model remains unchanged.
    """

    if inplace:
        merged_model = model
    else:
        merged_model = copy.deepcopy(model)

    for layer in (
        merged_model
        .distilbert
        .transformer
        .layer
    ):
        attention = layer.attention

        for target_name in (
            "q_lin",
            "v_lin",
        ):
            module = getattr(
                attention,
                target_name,
            )

            if isinstance(
                module,
                LoRALinear,
            ):
                setattr(
                    attention,
                    target_name,
                    merge_lora_layer(module),
                )

    return merged_model


def count_lora_layers(
    model: nn.Module,
) -> int:
    """
    Count remaining LoRALinear modules.
    """

    return sum(
        1
        for module in model.modules()
        if isinstance(
            module,
            LoRALinear,
        )
    )


def max_output_difference(
    output_a: torch.Tensor,
    output_b: torch.Tensor,
) -> float:
    """
    Compute maximum absolute difference between two outputs.
    """

    return (
        output_a
        - output_b
    ).abs().max().item()