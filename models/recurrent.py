import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from .convolution import CausalConv1d
from .components import RMSNorm

# from olmo_core.config import DType
# from olmo_core.distributed.parallel.context_parallel import (
#     all_to_all_cp2hp,
#     all_to_all_single_cp2hp,
#     all_to_all_single_hp2cp,
# )
# from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
# from olmo_core.nn.attention.flash_linear_attn_api import (
#     dispatch_chunk_gated_delta_rule,
#     has_fla,
# )
# from olmo_core.nn.attention.ring import (
#     RingContextParallelStyle,
#     UlyssesContextParallelStyle,
# )
# from olmo_core.nn.buffer_cache import BufferCache
# from olmo_core.nn.convolution import CausalConv1d
# from olmo_core.nn.feed_forward import ActivationFunction

# if TYPE_CHECKING:
#     from olmo_core.nn.transformer.init import InitMethod


class GatedDeltaNet(nn.Module):
    """
    The layer implementation for `Gated Delta Networks <https://arxiv.org/abs/2412.06464>`_.

    Modified from: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/layers/gated_deltanet.py#L34

    This is a linear attention variant that uses a gated delta rule for recurrent
    state updates, providing efficient O(n) sequence modeling.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param n_v_heads: The number of value heads. If ``None``, defaults to ``n_heads``.
        GVA is applied if ``n_v_heads`` > ``n_heads``.
    :param head_dim: The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    :param expand_v: The expansion ratio for the value dim. Default: 2.0.
    :param allow_neg_eigval: Allow negative eigenvalues. Default: ``True``. If set to ``True``, the beta
        will be multiplied by 2. See reference: `Unlocking State-Tracking in Linear RNNs Through Negative
        Eigenvalues <https://arxiv.org/abs/2411.12537>`_.
    :param conv_size: The kernel size of the short convolution. Default: 4.
    :param conv_bias: Whether to use bias in the short convolution. Default: ``False``.
    :param norm_eps: The epsilon value for the normalization layer. Default: 1e-5.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.n_heads * self.head_k_dim)
        self.value_dim = int(self.n_v_heads * self.head_v_dim)

        # Consistency checks: ensure expand_v produces integer dimensions
        assert math.isclose(self.n_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5)
        assert math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5)
        assert self.n_v_heads >= self.n_heads and self.n_v_heads % self.n_heads == 0

        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_v = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)
        self.w_a = nn.Linear(d_model, self.n_v_heads, bias=False, dtype=dtype, device=init_device)
        self.w_b = nn.Linear(d_model, self.n_v_heads, bias=False, dtype=dtype, device=init_device)

        self.A_log = nn.Parameter(torch.empty(self.n_v_heads, dtype=dtype, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(self.n_v_heads, dtype=dtype, device=init_device))

        self.q_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.k_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.v_conv1d = CausalConv1d(
            hidden_size=self.value_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.w_g = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)
        self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, device=init_device)  # type: ignore
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype, device=init_device)

        self.cp_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply gated delta network sequence mixing to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).

        :returns: The output with shape ``(batch_size, seq_len, d_model)``.
        """
        del kwargs  # Ignore any extra kwargs passed from attention interface
        B, T_og, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_v_heads * head_v_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        beta = self.w_b(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        g = -self.A_log.float().exp() * F.softplus(self.w_a(x).float() + self.dt_bias)

        q = self.q_conv1d(x=q, cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(x=k, cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(x=v, cu_seqlens=cu_doc_lens)

        T = q.size(1)
        q = q.view(B, T, -1, self.head_k_dim)
        k = k.view(B, T, -1, self.head_k_dim)
        v = v.view(B, T, -1, self.head_v_dim)

        if self.n_v_heads > self.n_heads:
            repeat_factor = self.n_v_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)

        o, _ = dispatch_chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_doc_lens, use_qk_l2norm_in_kernel=True
        )

        g = self.w_g(x).view(B, T, -1, self.head_v_dim)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(self.o_norm(o, g).view(B, T_og, -1))


    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError(
                f"init method '{init_method}' is not supported for GatedDeltaNet"
            )

        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for w in (self.w_q, self.w_k, self.w_v, self.w_a, self.w_b, self.w_g):
            init_linear(w, std=std, generator=generator)
        for w in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            init_linear(w, std=std, generator=generator)

        self.A_log.copy_(nn.init.uniform_(self.A_log, a=0, b=16, generator=generator).log())
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            nn.init.uniform_(self.dt_bias, generator=generator)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.copy_(inv_dt)

        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Compute FLOPs per token for Gated Delta Net.

        This accounts for:
        - Linear projections (w_q, w_k, w_v, w_a, w_b, w_g, w_out)
        - Short convolutions (q, k, v)
        - Gated delta rule recurrent computation
        - Gated RMS normalization
        """
        del seq_len
        # Linear projection FLOPs (2 ops per multiply-add)
        linear_flops = 2 * sum(
            m.weight.numel()
            for m in (self.w_q, self.w_k, self.w_v, self.w_a, self.w_b, self.w_g, self.w_out)
        )

        # Short convolution FLOPs (2 ops per multiply-add, kernel_size taps per output)
        conv_flops = (
            2
            * self.conv_size
            * (self.key_dim + self.key_dim + self.value_dim)  # q_conv1d  # k_conv1d  # v_conv1d
        )

        # Gated delta rule recurrent computation per token:
        # - Outer product k ⊗ v: n_v_heads * head_k_dim * head_v_dim
        # - State decay: n_v_heads * head_k_dim * head_v_dim
        # - Beta scaling: n_v_heads * head_k_dim * head_v_dim
        # - Query-state matmul: n_v_heads * head_k_dim * head_v_dim
        # Each is 2 FLOPs per element (multiply-add or similar)
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 2 * 4 * state_size

        return int(linear_flops + conv_flops + recurrent_flops)
