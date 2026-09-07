# Backend translation report

Source: `translation_test.log`, produced by `testing/translated_module_tests.py`.
Modules covered: `embeddings`, `rms_norm`, `lm_head`, `ffn`, `gated_attention`.
All modules run in bfloat16 on CUDA.

## Context that decides the reading

`src/models/translate_backend.py` does no arithmetic. Every translation is
either an identity pass-through (`translate_embeddings`, `translate_rms_norm`,
`translate_lm_head`) or a `split` view (`translate_ffn` splits `fc1` into
`gate_proj`/`up_proj`; `translate_gated_attention` splits `w_qkv` into
`q_proj`/`k_proj`/`v_proj`). No permutation is enabled in the test, because
`_forward` uses `permute_rope_qk=True` on the FLA side instead.

The two modules therefore hold bit-identical weights in every case. Any
non-zero difference in the log comes from the kernels, not from the
translation.

## Translated perfectly

- `embeddings` - zero forward, zero backward. Both backends are `nn.Embedding`.
- `lm_head` - zero forward, zero backward. Both backends are `nn.Linear`.

## Not translated perfectly

### rms_norm

Forward is bit-exact (0.0). Backward is not (Frobenius norm 0.131).

The forward match proves the weight mapping is right. The backward differs
because the legacy `RMSNorm.forward` upcasts to float32
(`self._norm(x.float()).type_as(x)`) and lets autograd differentiate the eager
operations, while FLA `RMSNorm` uses a fused Triton backward with its own
accumulation order. Different order gives different bfloat16 rounding.

### gated_attention and ffn

Both differ in forward and backward, by similar amounts.

Forward, with translated weights:

| Case | max abs error | In bfloat16 terms |
| --- | --- | --- |
| `gated_attention` | 1.953125e-3 = 2^-9 | one ulp for values in [1, 2) |
| `ffn` | 1.953125e-3 = 2^-9 | one ulp for values in [1, 2) |
| `gated_attention`, init weights | 1.220703e-4 = 2^-13 | one ulp for values in [0.031, 0.0625) |
| `ffn`, init weights | 1.220703e-4 = 2^-13 | one ulp for values in [0.031, 0.0625) |

Every value is an exact power of two equal to one rounding step at the output
scale. The two init rows are smaller only because a 0.02 init standard
deviation produces smaller activations, so one ulp is smaller in absolute
terms. The Frobenius norms agree: 0.0165 over 2 * 16 * 64 = 2048 elements is an
RMS of 3.6e-4, so most elements match exactly and a few are off by one step.

Cause: the legacy path runs `F.scaled_dot_product_attention` under
`SDPBackend.FLASH_ATTENTION` and a fused `w_qkv` matmul; FLA runs its own
attention path and three separate projections. Same math, different reduction
order.

## Tolerance

**Forward: workable.** The errors sit exactly at the bfloat16 rounding floor.
This cannot be improved without changing dtype. Nothing is left to fix in the
forward translation.

**Backward: the log does not say.** Every backward line prints only the
absolute Frobenius norm of the difference. Without the norm of the reference
gradient there is no scale to compare against, so 0.131 and 0.55 cannot be
interpreted. Answering this needs the relative error,
`||fla_grad - legacy_grad|| / ||legacy_grad||`. The largest raw differences land
on `o_proj` and `down_proj`, which carry the largest gradients, so the relative
error is expected to be small, but that is an expectation and not a
measurement.

## Two problems with the test itself

### 1. allclose is not calibrated for bfloat16

`torch.allclose` defaults to `rtol=1e-5, atol=1e-8`. bfloat16 resolution is
2^-8, about 3.9e-3. The check can only pass when the result is bit-exact, so
`allclose: False` carries no information for anything that is not exact. A
bfloat16-appropriate threshold is roughly `rtol=1e-2, atol=1e-3`.

### 2. The seeded initialization test shows the _initialize docstring premise is false

That test does no translation. It seeds both backends and checks whether they
draw the same weights. The result:

- `gated_attention`: `q_proj` 0.0, `k_proj` 0.102, `v_proj` 0.114, `o_proj` 0.071
- `ffn`: `gate_proj` 0.0, `up_proj` 0.103, `down_proj` 0.077
- `embeddings`, `rms_norm`, `lm_head`: 0.0

The pattern is exact. Where the legacy side has a fused weight (`w_qkv` at
(192, 64), `fc1` at (2 * hidden, 64)), only the first slice matches and the rest
diverge. Where the weight is unfused, everything matches. The errors on the
diverging slices are 5 sigma against the 0.02 init standard deviation, so those
are independent draws and not rounding.

The `_initialize` docstring claims that `normal_` fills element by element, so
one fused draw and the separate draws it splits into consume the same random
numbers in the same order. That does not hold: `normal_` on CUDA generates in a
vectorized, size-dependent layout, so the stream depends on the tensor shape.
`o_proj` and `down_proj` then differ for a second reason too, because
`_scale_residual_branches` redraws them from an RNG state that has already
diverged.

Practical consequence: seeding both backends does not give them equal weights.
Translation is required. The forward results say translation works.
