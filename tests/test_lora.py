"""Unit tests for the custom LoRA implementation."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
)

from src.lora import LoRALinear
from src.injection import (
    convert_linear_to_lora,
    prepare_distilbert_for_lora,
)
from src.merge import (
    merge_lora_layer,
    merge_distilbert_lora,
    count_lora_layers,
)
from src.checkpoint import (
    save_adapter_checkpoint,
    load_adapter_checkpoint,
)


def create_tiny_distilbert():
    """
    Create a tiny DistilBERT model locally.

    No internet download is required.
    """

    config = DistilBertConfig(
        vocab_size=100,
        max_position_embeddings=64,
        n_layers=2,
        n_heads=4,
        dim=32,
        hidden_dim=64,
        num_labels=2,
    )

    return DistilBertForSequenceClassification(
        config
    )


def test_lora_output_shape():
    layer = LoRALinear(
        in_features=32,
        out_features=16,
        rank=4,
        alpha=8,
    )

    x = torch.randn(
        2,
        5,
        32,
    )

    y = layer(x)

    assert y.shape == (
        2,
        5,
        16,
    )


def test_base_weights_are_frozen():
    layer = LoRALinear(
        32,
        16,
        rank=4,
        alpha=8,
    )

    assert (
        layer.linear.weight.requires_grad
        is False
    )

    assert (
        layer.lora_A.requires_grad
        is True
    )

    assert (
        layer.lora_B.requires_grad
        is True
    )


def test_zero_initial_lora_matches_original_linear():
    torch.manual_seed(42)

    original = nn.Linear(
        32,
        16,
    )

    lora = convert_linear_to_lora(
        original,
        rank=4,
        alpha=8,
    )

    x = torch.randn(
        4,
        32,
    )

    original_output = original(x)
    lora_output = lora(x)

    assert torch.allclose(
        original_output,
        lora_output,
        atol=1e-6,
    )


def test_merge_preserves_output():
    torch.manual_seed(42)

    lora = LoRALinear(
        32,
        16,
        rank=4,
        alpha=8,
    )

    with torch.no_grad():
        lora.lora_B.normal_(
            mean=0.0,
            std=0.02,
        )

    x = torch.randn(
        4,
        32,
    )

    before_merge = lora(x)

    merged = merge_lora_layer(
        lora
    )

    after_merge = merged(x)

    assert torch.allclose(
        before_merge,
        after_merge,
        atol=1e-5,
    )


def test_distilbert_injection():
    model = create_tiny_distilbert()

    replaced = (
        prepare_distilbert_for_lora(
            model,
            rank=4,
            alpha=8,
        )
    )

    # 2 Transformer layers x q/v = 4 LoRA layers.
    assert len(replaced) == 4

    assert (
        count_lora_layers(model)
        == 4
    )


def test_distilbert_merge_preserves_output():
    torch.manual_seed(42)

    model = create_tiny_distilbert()

    prepare_distilbert_for_lora(
        model,
        rank=4,
        alpha=8,
    )

    # Give B non-zero values so that
    # the LoRA update is actually active.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(
                module,
                LoRALinear,
            ):
                module.lora_B.normal_(
                    mean=0.0,
                    std=0.02,
                )

    input_ids = torch.randint(
        0,
        100,
        (
            2,
            16,
        ),
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    model.eval()

    with torch.no_grad():
        original_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    merged_model = (
        merge_distilbert_lora(
            model
        )
    )

    merged_model.eval()

    with torch.no_grad():
        merged_logits = merged_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    assert (
        count_lora_layers(
            merged_model
        )
        == 0
    )

    assert torch.allclose(
        original_logits,
        merged_logits,
        atol=1e-5,
    )


def test_adapter_checkpoint_round_trip():
    torch.manual_seed(42)

    model_a = create_tiny_distilbert()

    prepare_distilbert_for_lora(
        model_a,
        rank=4,
        alpha=8,
    )

    with torch.no_grad():
        for module in model_a.modules():
            if isinstance(
                module,
                LoRALinear,
            ):
                module.lora_B.normal_(
                    mean=0.0,
                    std=0.02,
                )

    model_b = create_tiny_distilbert()

    prepare_distilbert_for_lora(
        model_b,
        rank=4,
        alpha=8,
    )

    # Make frozen base-model weights identical.
    base_state = model_a.state_dict()

    for name, parameter in model_b.state_dict().items():
        if (
            "lora_A" not in name
            and "lora_B" not in name
            and not name.startswith(
                "pre_classifier"
            )
            and not name.startswith(
                "classifier"
            )
        ):
            parameter.copy_(
                base_state[name]
            )

    # Windows-safe temporary checkpoint directory.
    with tempfile.TemporaryDirectory() as temp_dir:

        checkpoint_path = (
            Path(temp_dir)
            / "adapter.pt"
        )

        save_adapter_checkpoint(
            model_a,
            str(checkpoint_path),
        )

        load_adapter_checkpoint(
            model_b,
            str(checkpoint_path),
        )

        input_ids = torch.randint(
            0,
            100,
            (
                2,
                16,
            ),
        )

        attention_mask = torch.ones_like(
            input_ids
        )

        model_a.eval()
        model_b.eval()

        with torch.no_grad():
            logits_a = model_a(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            logits_b = model_b(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        assert torch.allclose(
            logits_a,
            logits_b,
            atol=1e-5,
        )