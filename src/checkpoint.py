"""Adapter-only checkpoint utilities for LoRA models."""

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


def get_adapter_state_dict(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Return only LoRA parameters and the task-specific
    classification head parameters.

    This keeps the saved checkpoint much smaller than
    storing the full pretrained model.
    """

    adapter_state = {}

    for name, parameter in model.state_dict().items():

        is_lora = (
            "lora_A" in name
            or "lora_B" in name
        )

        is_classifier = (
            name.startswith("pre_classifier")
            or name.startswith("classifier")
        )

        if is_lora or is_classifier:
            adapter_state[name] = (
                parameter
                .detach()
                .cpu()
                .clone()
            )

    return adapter_state


def save_adapter_checkpoint(
    model: nn.Module,
    path: str,
) -> Path:
    """
    Save only LoRA adapters and the classification head.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter_state = (
        get_adapter_state_dict(model)
    )

    torch.save(
        adapter_state,
        path,
    )

    return path


def load_adapter_checkpoint(
    model: nn.Module,
    path: str,
    map_location: str = "cpu",
) -> nn.Module:
    """
    Load an adapter-only checkpoint into an already
    prepared LoRA model.

    The model must already contain LoRALinear modules
    with the same rank and architecture.
    """

    adapter_state = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            adapter_state,
            strict=False,
        )
    )

    unexpected_keys = list(
        unexpected_keys
    )

    if unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys: "
            f"{unexpected_keys}"
        )

    return model


def checkpoint_size_mb(
    path: str,
) -> float:
    """
    Return checkpoint file size in megabytes.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return (
        path.stat().st_size
        / (1024 ** 2)
    )