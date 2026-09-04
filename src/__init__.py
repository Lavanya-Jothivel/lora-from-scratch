"""LoRA from-scratch package."""

from .lora import LoRALinear

from .injection import (
    convert_linear_to_lora,
    inject_lora_into_distilbert,
    freeze_base_model,
    enable_lora_parameters,
    enable_classification_head,
    prepare_distilbert_for_lora,
)

from .merge import (
    merge_lora_layer,
    merge_distilbert_lora,
    count_lora_layers,
    max_output_difference,
)

from .checkpoint import (
    get_adapter_state_dict,
    save_adapter_checkpoint,
    load_adapter_checkpoint,
    checkpoint_size_mb,
)

from .train import (
    count_parameters,
    trainable_percentage,
    train_one_epoch,
    evaluate_model,
    fit,
)

from .benchmark import (
    benchmark_inference,
    predict,
    prediction_agreement,
    latency_improvement,
)

__all__ = [
    "LoRALinear",
    "convert_linear_to_lora",
    "inject_lora_into_distilbert",
    "freeze_base_model",
    "enable_lora_parameters",
    "enable_classification_head",
    "prepare_distilbert_for_lora",
]