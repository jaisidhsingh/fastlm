import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Literal, Optional


class OLMoCausalConv1d(nn.Conv1d):
    """
    CausalConv1d (aka short convolution) layer for efficient causal convolution operations.
    This implements a depthwise separable 1D convolution with causal padding.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        init_device: str = "cpu",
        activation: Literal["silu", "swish"] | None = "silu",
    ):
        """
        :param hidden_size: Number of input/output channels (must be equal for depthwise conv).
        :param kernel_size: Size of the convolution kernel.
        :param bias: Whether to include learnable bias.
        :param backend: Backend implementation ('triton' or 'cuda').
        :param dtype: The data type of the convolution weights and bias.
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param activation: Activation function ('silu' or 'swish').
        """
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=init_device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.activation = activation

    def forward( 
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            ``batch_size`` must be 1 if ``cu_seqlens`` is provided.
            When CP is enabled, input should be channel-parallel: ``(batch_size, seq_len, hidden_size/CP)``.
        :param cu_seqlens: Cumulative sequence lengths for variable-length sequences.
            Shape: ``(num_seqs + 1,)``.
        :returns: Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
            When CP is enabled, output is channel-parallel: ``(batch_size, seq_len, hidden_size/CP)``.
        """
        weight = self.weight
        bias = self.bias

        # (B, T, D) -> (B, D, T)
        if cu_seqlens is None:
            x = x.transpose(1, 2)

            # standard causal conv
            out = F.conv1d(
                x,
                weight,
                bias=bias,
                stride=1,
                padding=self.kernel_size[0] - 1,
                groups=self.groups,
            )

            # remove future positions
            if self.kernel_size[0] > 1:
                out = out[..., :-(self.kernel_size[0] - 1)]

            # (B, D, T) -> (B, T, D)
            out = out.transpose(1, 2)

        else:
            # packed sequences: process each segment independently
            # x shape: (total_tokens, D), batch_size must be 1
            outputs = []
            for i in range(len(cu_seqlens) - 1):
                start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                xi = x[start:end]                  # (Ti, D)
                xi = xi.unsqueeze(0).transpose(1, 2)  # (1, D, Ti)

                yi = F.conv1d(
                    xi,
                    weight,
                    bias=bias,
                    stride=1,
                    padding=self.kernel_size[0] - 1,
                    groups=self.groups,
                )

                if self.kernel_size[0] > 1:
                    yi = yi[..., :-(self.kernel_size[0] - 1)]

                yi = yi.transpose(1, 2).squeeze(0)  # (Ti, D)
                outputs.append(yi)

            out = torch.cat(outputs, dim=0)

        # activation (matches Mamba/OLMo style: after conv)
        if self.activation in ("silu", "swish"):
            out = F.silu(out)

        return out


class CausalConv1d(nn.Module): # Claude 4.6 Sonnet
    """
    Depthwise causal conv1d as used in GatedDeltaNet / Mamba-style SSMs.

    References:
      - Mamba (Gu & Dao, 2023): https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L126
      - GatedDeltaNet: https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py#L186
      - RWKV-7 / FLA ops: https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/ops/utils/conv.py
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        # Depthwise: groups = d_model, so each channel is convolved independently
        # padding = kernel_size - 1 on the left ensures causality (no future leakage)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,        # depthwise
            padding=kernel_size - 1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)  — standard seq-first layout used in FLA / GDN
        Returns:
            out: (B, T, D)

        Note: Mamba uses (B, D, L) internally and calls this inside its block
        with x.transpose(1,2). Here we handle the transpose ourselves so the
        caller always sees (B, T, D).
        """
        # (B, T, D) -> (B, D, T) for Conv1d
        x = rearrange(x, 'b t d -> b d t')

        # Conv1d with left-padding=kernel_size-1 produces length T + kernel_size - 1
        # We slice off the right padding to restore T  (causal mask)
        x = self.conv(x)
        x = x[..., :-(self.kernel_size - 1)]   # remove future-looking positions

        # Back to (B, T, D)
        x = rearrange(x, 'b d t -> b t d')
        return x


# ── How GatedDeltaNet uses it ────────────────────────────────────────────────
#
#   self.q_conv  = CausalConv1d(self.key_dim,   kernel_size)
#   self.k_conv  = CausalConv1d(self.key_dim,   kernel_size)
#   self.v_conv  = CausalConv1d(self.value_dim, kernel_size)
#   # β (input gate for delta rule) also gets a conv in some variants
#
#   # In the forward pass, after linear projections:
#   q = self.q_conv(q)           # (B, T, key_dim)
#   k = self.k_conv(k)           # (B, T, key_dim)
#   v = self.v_conv(v)           # (B, T, value_dim)
#   q, k = F.silu(q), F.silu(k) # activation applied after conv, not before
#
# ── Mamba does the same thing (slightly different layout) ────────────────────
#
#   # mamba_simple.py ~L126:
#   x = self.act(self.conv1d(x)[..., :seqlen])
#   # where conv1d is nn.Conv1d(..., padding=d_conv-1) and d_conv=4 by default
