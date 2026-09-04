"""Minimal end-to-end demonstration of the custom LoRA implementation."""

import torch
import torch.nn as nn

from src.lora import LoRALinear
from src.merge import merge_lora_layer


def main():
    torch.manual_seed(42)

    print("=== LoRA From Scratch Demo ===")

    # Original dense layer.
    original = nn.Linear(
        in_features=16,
        out_features=8,
    )

    # Create custom LoRA layer.
    lora = LoRALinear(
        in_features=16,
        out_features=8,
        rank=4,
        alpha=8,
    )

    # Copy original pretrained weights
    # into the frozen base layer.
    with torch.no_grad():
        lora.linear.weight.copy_(
            original.weight
        )

        lora.linear.bias.copy_(
            original.bias
        )

    x = torch.randn(
        4,
        16,
    )

    # Because LoRA B starts at zero,
    # LoRA initially matches the base model.
    original_output = original(x)
    initial_lora_output = lora(x)

    print(
        "Initial output match:",
        torch.allclose(
            original_output,
            initial_lora_output,
            atol=1e-6,
        ),
    )

    # Simulate a learned LoRA update.
    with torch.no_grad():
        lora.lora_B.normal_(
            mean=0.0,
            std=0.02,
        )

    adapted_output = lora(x)

    print(
        "Output changed after LoRA update:",
        not torch.allclose(
            original_output,
            adapted_output,
            atol=1e-6,
        ),
    )

    # Merge LoRA update into base weight.
    merged = merge_lora_layer(
        lora
    )

    merged_output = merged(x)

    print(
        "Merged output matches LoRA:",
        torch.allclose(
            adapted_output,
            merged_output,
            atol=1e-5,
        ),
    )

    print(
        "Maximum merge difference:",
        (
            adapted_output
            - merged_output
        ).abs().max().item(),
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in lora.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in lora.parameters()
        if parameter.requires_grad
    )

    print(
        "Total parameters:",
        total_parameters,
    )

    print(
        "Trainable LoRA parameters:",
        trainable_parameters,
    )

    print(
        "Trainable percentage:",
        round(
            trainable_parameters
            / total_parameters
            * 100,
            2,
        ),
        "%",
    )


if __name__ == "__main__":
    main()