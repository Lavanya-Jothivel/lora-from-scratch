"""Core LoRA layers implemented from scratch in PyTorch."""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Linear layer augmented with Low-Rank Adaptation (LoRA).

    Instead of updating the pretrained weight W0, LoRA learns:

        delta_W = B @ A

    and computes:

        y = W0(x) + (alpha / rank) * B(A(x))

    The original linear layer remains frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        bias: bool = True,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be greater than 0.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base transformation W0.
        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias,
        )

        for parameter in self.linear.parameters():
            parameter.requires_grad = False

        # A: [rank, in_features]
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                in_features,
            )
        )

        # B: [out_features, rank]
        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                rank,
            )
        )

        # Practical initialization used in this reproduction:
        # A is randomly initialized and B starts at zero.
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the frozen base layer plus the LoRA update."""

        base_output = self.linear(x)

        lora_output = (
            (x @ self.lora_A.T)
            @ self.lora_B.T
        )

        return (
            base_output
            + self.scaling * lora_output
        )

    def delta_weight(self) -> torch.Tensor:
        """Return the learned LoRA weight update."""

        return (
            self.scaling
            * (self.lora_B @ self.lora_A)
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}"
        )