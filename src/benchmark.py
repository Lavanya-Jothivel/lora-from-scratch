"""Inference benchmarking utilities for LoRA experiments."""

import time
from typing import List

import torch
import torch.nn as nn


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    warmup_runs: int = 10,
    measured_runs: int = 50,
) -> float:
    """
    Measure average inference latency in milliseconds per batch.
    """

    model.eval()

    device = next(
        model.parameters()
    ).device

    input_ids = input_ids.to(
        device
    )

    attention_mask = (
        attention_mask.to(
            device
        )
    )

    # Warmup runs
    for _ in range(
        warmup_runs
    ):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(
        measured_runs
    ):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    average_latency_ms = (
        (end_time - start_time)
        / measured_runs
        * 1000
    )

    return average_latency_ms


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> List[int]:
    """
    Generate classification predictions for a dataloader.
    """

    model.eval()

    predictions = []

    for batch in dataloader:

        input_ids = (
            batch["input_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        batch_predictions = (
            outputs.logits
            .argmax(dim=-1)
            .cpu()
            .tolist()
        )

        predictions.extend(
            batch_predictions
        )

    return predictions


def prediction_agreement(
    predictions_a: List[int],
    predictions_b: List[int],
) -> float:
    """
    Compute prediction agreement between two models.
    """

    if len(predictions_a) != len(
        predictions_b
    ):
        raise ValueError(
            "Prediction lists must have the same length."
        )

    if len(predictions_a) == 0:
        raise ValueError(
            "Prediction lists cannot be empty."
        )

    matching = sum(
        prediction_a
        == prediction_b
        for prediction_a, prediction_b
        in zip(
            predictions_a,
            predictions_b,
        )
    )

    return (
        matching
        / len(predictions_a)
    )


def latency_improvement(
    original_latency: float,
    merged_latency: float,
) -> float:
    """
    Return percentage latency improvement after merging.
    """

    if original_latency <= 0:
        raise ValueError(
            "Original latency must be greater than zero."
        )

    return (
        (
            original_latency
            - merged_latency
        )
        / original_latency
        * 100
    )