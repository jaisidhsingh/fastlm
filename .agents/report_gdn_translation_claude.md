# GDN backend translation report

Source: `gdn_translation_test.log`, produced by `testing/translated_module_tests.py`.
Modules covered: `gated_attention`, `ffn`, `embeddings`, `rms_norm`, `lm_head`, `gdn`.
Whole models covered: `attn`, `gdn`, `gdn+attn_3-1`, `gdn+attn_1-3`.
All forwards run in bfloat16 on CUDA. GDN runs in training mode at `seq_len=128`, so both its
forward and its backward use the `chunk` kernel, which is the kernel pretraining uses.

## What the GDN numbers mean

`fla/layers/gated_deltanet.py` and `fla/layers/legacy_gated_deltanet.py` differ only in the class
name. `translate_gated_deltanet` is therefore an identity map, and the two modules hold bit-identical
weights. A zero difference does not, by itself, say the translation is right — it would be zero even
if the translation were untested.

What the test does measure is the config path. `_fla_gdn` reads **only** `fla_config`, the output of
`builder.config_builder`; `_legacy_gdn` reads the legacy fields the way `Block.__init__` does. So a
zero difference means `config_builder` reproduces every argument legacy `Block` passes:

| argument | how a mismatch would show |
|---|---|
| `head_dim` | wrong projection shapes, `load_state_dict` fails |
| `allow_neg_eigval` | beta scaled by 2 inside the kernel, non-zero difference |
| `conv_size` | wrong conv weight shape, `load_state_dict` fails |
| `use_gate` | `g_proj`/`o_norm` mismatch, `load_state_dict` fails |
| `intra_doc`, `norm_eps` | same shapes, non-zero difference |

Before this change `head_dim` stayed at the config default 256 and `allow_neg_eigval` stayed `False`
against `gdn_neg_eigval: True`. Both are now mapped, and the zero difference confirms it.

## Module results

### GDN — exact everywhere

Every GDN number in the log is `0.0`:

- forward: `max absolute error 0.0`
- backward: all 13 parameters `equal: True`, including `A_log` and `dt_bias`
- initialization forward (translated): `0.0`
- initialization forward (seeded): every parameter `equal: True`, forward `0.0`

The seeded result is worth noting. For `gated_attention` and `ffn` the seeded test fails on every
weight after the first slice, because legacy holds fused weights (`w_qkv`, `fc1`) and CUDA `normal_`
generates in a shape-dependent layout. GDN has no fused weight, so seeding both backends does give
them equal weights. This is the one module where seeding is a substitute for translation.

The backward being exact also confirms the mode fix. `fused_recurrent_gated_delta_rule` raises
`NotImplementedError` in its backward (`fla/ops/gated_delta_rule/fused_recurrent.py:299`), and
`GatedDeltaNet.forward` selects that kernel whenever `q_len <= 64 and not self.training`. Running GDN
in training mode at `seq_len=128` pins `chunk` for both directions.

### The other modules — unchanged from the previous report

`embeddings` and `lm_head` are exact in forward and backward. `rms_norm` is exact in forward and
differs in backward by a relative Frobenius norm of `0.0029`, from the legacy eager float32 upcast
against FLA's fused Triton backward. `gated_attention` and `ffn` differ in forward by
`1.953125e-3 = 2^-9`, one bfloat16 ulp for values in `[1, 2)`, and in backward by relative Frobenius
norms of `0.0015` to `0.0037`. These are the numbers `.agents/report_translation_claude.md` already
covered, now with the reference norms that make them readable. Every backward relative error is under
0.4 percent.

## Whole-model results

All four architectures loaded with `strict=True` and produced logits. This is the check the
per-module tests structurally cannot do: layer-type routing, the `token_mixer_norm` to `attn_norm`
rename, the RoPE weight permutation, and weight tying.

| arch_id | attention layers | relative token rms | max absolute error | relative max absolute error |
|---|---|---|---|---|
| `attn` | `[0, 1, 2, 3]` | 0.00412 | 0.0078125 | 0.00518 |
| `gdn` | `[]` | 0.00741 | 0.0078125 | 0.00541 |
| `gdn+attn_3-1` | `[3]` | 0.00637 | 0.0078125 | 0.00546 |
| `gdn+attn_1-3` | `[1, 2, 3]` | 0.00389 | 0.0078125 | 0.00515 |

Config: 4 layers, `dim=64`, `n_heads=4`, `vocab_size=256`, `seq_len=128`, `tie_embeddings=True`,
`attn_gate=True`, `attn_qk_norm=True`, `gdn_neg_eigval=True`.

The attention layer indices match `Transformer._prepare_layers`. They did not before: for a reversed
ratio `build_hybrid_layers` used `(i + 1) % (r + 1) != 0` where legacy uses `i % (r + 1) != 0`, which
put every layer of a `gdn+attn_1-x` model on the wrong branch. That is fixed.

`max absolute error 0.0078125 = 2^-7` is one bfloat16 ulp for values in `[1, 2)`, identical across all
four architectures. bfloat16 has 8 mantissa bits, so its relative resolution is `2^-8 = 0.0039`. Every
relative token rms in the table is one to two ulps after four layers. This is rounding, not a
translation error.

`allclose: False` on all four rows is not informative. The threshold is
`atol + rtol * |reference| = 1e-3 + 1e-2 * |reference|`, and logits near zero fail it at one ulp while
carrying no information. Read the relative token rms instead.

### One observation without a verified cause

The pure `gdn` model has the **largest** relative token rms (0.00741) even though its GDN layers are
bit-exact at the module level, while the pure `attn` model has a smaller one (0.00412) even though its
attention layers differ by one ulp per layer. Error falls monotonically as attention replaces GDN:
0.00741 with 0 attention layers, 0.00637 with 1, 0.00389 with 3.

The GDN layers are not the source of the divergence — they are exact given exact inputs. They are
amplifying divergence introduced upstream by the RMSNorm, MLP, and residual paths. A plausible
mechanism is that the delta-rule recurrence compounds a perturbation along all 128 positions, where
attention does not. This is a hypothesis, not a measurement. It matters only if it grows with depth
and sequence length, which the current run does not test.

## Are we in a position to start learning rate decay from translated checkpoints?

**The translation itself: yes.** Structure and hyper-parameters are correct. Every architecture loads
strictly, the layer routing matches, and the numerical disagreement is at the bfloat16 rounding floor.

**The procedure around it: not yet.** Four things stand between this result and a safe anneal, and the
log speaks to none of them.

1. **No real checkpoint has been translated.** Every model in the log was freshly constructed at
   `dim=64`, 4 layers, `vocab_size=256`. Production runs are `d_model >= 256`, `n_layers >= 8`,
   `vocab_size=50304`, and real checkpoints nest under `state['state_dict']` with a `_orig_mod.`
   prefix from `torch.compile`. Load one, translate it, and compare validation loss against the legacy
   backend on the same batches. Loss is the quantity the anneal is judged on, and it has not been
   measured once.

2. **Error growth with depth is unmeasured.** Four layers give one to two ulps. Whether that stays at
   one to two ulps at 8, 16, or 24 layers is an open question, and the GDN amplification observed
   above is the reason to check rather than assume. Repeat the whole-model comparison at the depths
   and widths actually used.

3. **Optimizer state is not translated.** `translate_model` maps model weights only. Resuming a decay
   schedule with Adam moments reset to zero is a different run from resuming with them intact, and the
   difference is largest exactly where the decay starts. Decide explicitly whether the anneal restarts
   the optimizer, and if it must not, the moment tensors need the same key mapping.

4. **`intra_doc` is untested.** `src/constants.py` sets `intra_doc_masking: True`, but the harness runs
   `intra_doc=False`, passes no `cu_seqlens` and no `linear_mask`, and so never enters that branch in
   either backend. The config now carries the flag through to the layer, which is what was missing, but
   carrying it through is not the same as having compared it.

One smaller item: `A_log._no_weight_decay` and `dt_bias._no_weight_decay` are Python attributes on the
Parameter objects, set in `__init__`. A model that is constructed and then loaded keeps them, so
translation does not lose them. Confirm that the new backend's parameter grouping reads that attribute
the way `get_param_groups` does on the legacy side, or those two tensors will pick up weight decay
they did not have during pretraining.

**Recommendation.** Do item 1 next, on one real `gdn+attn_3-1` checkpoint. A validation loss that
matches the legacy backend to within a few thousandths of a nat settles items 1 and 2 together and is
cheap. Items 3 and 4 are decisions to make before the first anneal launches, not measurements.
