"""Training and evaluation utilities for LoRA experiments."""

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def count_parameters(
    model: nn.Module,
) -> Dict[str, int]:
    """
    Count total and trainable parameters.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total_parameters,
        "trainable": trainable_parameters,
    }


def trainable_percentage(
    model: nn.Module,
) -> float:
    """
    Return percentage of trainable parameters.
    """

    counts = count_parameters(model)

    return (
        counts["trainable"]
        / counts["total"]
        * 100
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Returns average training loss.
    """

    model.train()

    total_loss = 0.0

    for batch in dataloader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(
            **batch
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

    average_loss = (
        total_loss
        / len(dataloader)
    )

    return average_loss


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate classification loss and accuracy.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(
            **batch
        )

        loss = outputs.loss
        logits = outputs.logits

        predictions = (
            logits.argmax(
                dim=-1
            )
        )

        labels = batch["labels"]

        total_loss += (
            loss.item()
        )

        total_correct += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )

        total_examples += (
            labels.size(0)
        )

    average_loss = (
        total_loss
        / len(dataloader)
    )

    accuracy = (
        total_correct
        / total_examples
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 3,
) -> List[Dict[str, float]]:
    """
    Train and evaluate a model for multiple epochs.

    Returns experiment history.
    """

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = (
            evaluate_model(
                model=model,
                dataloader=validation_loader,
                device=device,
            )
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_accuracy": (
                validation_metrics["accuracy"]
            ),
        }

        history.append(
            epoch_result
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: "
            f"{validation_metrics['loss']:.4f} | "
            f"Val Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

    return history