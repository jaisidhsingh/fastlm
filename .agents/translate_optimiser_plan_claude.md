# Translate AdamW optimizer state between backends


## Context

`src/models/translate_backend.py` translates model weights from the legacy `Transformer` to the FLA
backend, and `.agents/report_gdn_translation_claude.md` reports that translation as correct at the
bfloat16 rounding floor for every `arch_id`.

It translates weights only. `src/engine/engine.py:206` resumes with
`self.optimizer.load_state_dict(ckpt['optimizer'])`, so a translated checkpoint would start its
learning rate decay with Adam moments reset to zero. That is a different run from the one the decay
schedule assumes, and the difference is largest at the point the decay begins. This is the gap named
as item 3 in the GDN translation report.

Intended outcome: a legacy AdamW `optimizer.state_dict()` can be translated so that the FLA backend
resumes with the same first- and second-moment estimates on the same weights, and a cooldown started
from a translated checkpoint behaves as if the legacy backend had continued.

Scope, per the answers given: **AdamW only**. `nadamw`, `sgd`, `signSGD`, `sfo_adamw` and `muon`
raise rather than translate.

## Findings that decide the design

**The optimizer state dict is index-based, not name-based.** `torch.optim.Optimizer.state_dict()`
returns `{'state': {int: {...}}, 'param_groups': [{'params': [int, ...], ...}]}`. The integer is the
parameter's position in the flattened `param_groups` list. Nothing in it records a parameter name, so
the mapping must be reconstructed from both models.

**The index order comes from `get_param_groups`** (`src/models/legacy/construct.py:88`): the decay
group first, then the no-decay group, each in `named_parameters()` order. The no-decay group is
everything carrying `_no_weight_decay` (that is `A_log` and `dt_bias`), plus every name containing
`bias` or `norm`. `src/engine/engine.py:201` calls it on the **uncompiled** model, so no `_orig_mod.`
prefix reaches the indices.

**Neither backend registers a buffer.** `Transformer.freqs_cis` is a plain attribute, not a buffer.
So `named_parameters()` order is exactly the parameter order, with no interleaving to account for.

**Weight tying removes a parameter but not a state dict key.** `named_parameters()` deduplicates, so
a tied model has one entry, `embed_tokens.weight` on the legacy side and `model.embeddings.weight` on
the FLA side. `state_dict()` does **not** deduplicate and emits `lm_head.weight` as well. The
optimizer therefore has one index where `translate_model` has two keys. This has to be handled in both
directions.

**One FLA parameter has no legacy source.** FLA's `Attention` builds its gate as
`nn.Linear(hidden_size, hidden_size)` with bias (`fla/layers/attn.py:80`); the legacy `GatedAttention`
gate is bias-free. `translate_gated_attention` already synthesizes a zero `gate.bias`. Its optimizer
state must be synthesized too.

**Adam is elementwise, so every transform in `translate_model` is valid on moments.** Splitting
`w_qkv` into `q_proj`/`k_proj`/`v_proj`, splitting `fc1` into `gate_proj`/`up_proj`, and the RoPE row
permutation are all reorderings of independent coordinates. `exp_avg` and `exp_avg_sq` transform under
exactly the same operations as the weights, which is what makes reuse the right approach.

## Design — `src/optim/translate_backend.py`

New file. Reuses `translate_model` rather than reimplementing any mapping.

```python
def translate_adamw(
  optimizer_state_dict: Mapping[str, Any],
  *,
  legacy_model: nn.Module,
  fla_model: nn.Module,
  weight_decay: float,
  attn_layers: Collection[int],
  n_layers: int,
  num_heads: int,
) -> dict[str, Any]:
```

Import `translate_model` from `src.models.translate_backend` **directly**, not through
`src.models`. The package `__init__` imports `legacy.construct`, which imports `fla`; the module
itself imports nothing at runtime, so the direct import keeps this file cheap to load and to test.

### Step 1 — ordered parameter names on both sides

A private helper, used for both models:

```python
def _ordered_param_names(model: nn.Module, weight_decay: float) -> list[str]:
```

Call `get_param_groups(model, weight_decay)` (`src/models/legacy/construct.py:88`) and resolve each
returned Parameter back to its name through `{id(p): n for n, p in model.named_parameters()}`. This is
identity matching against the function the engine actually uses, so no ordering rule is duplicated and
none can drift. Raise if any Parameter fails to resolve.

Validate that the number of names equals the number of indices across
`optimizer_state_dict['param_groups']`, and that the per-group counts match. A mismatch means the
checkpoint was written by a differently configured model, and it must fail loudly rather than
misalign every moment by one position.

### Step 2 — recover the legacy-to-FLA name map from `translate_model` itself

Do not re-derive which FLA names each legacy name feeds. Observe it:

- build `probe = {name: torch.full(param.shape, float(i + 1)) for i, name in enumerate(legacy_names)}`
- when the legacy model is tied, add `lm_head.weight` pointing at the same tensor as
  `embed_tokens.weight`, because `translate_model` requires the key
- call `translate_model(probe, attn_layers=..., n_layers=..., num_heads=...)`
- every transform in `translate_model` is a split, reshape, transpose or `new_zeros`, so each output
  tensor is still constant-valued; read `int(t.flatten()[0]) - 1` to recover the source index

`gate.bias` comes back all-zero, which the `i + 1` offset makes unambiguous: value `0` means
synthesized, not "source index 0". Assert every output tensor is genuinely constant — that check is
what makes this trick safe, and it fails immediately if `translate_model` ever gains a transform that
mixes coordinates.

Drop the tied `lm_head.weight` from the resulting map, since it is not a separate parameter.

### Step 3 — translate the moments

Reject anything that is not AdamW state before touching it: every per-parameter entry must have keys
within `{'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}` and must contain `exp_avg` and
`exp_avg_sq`. Raise naming the offending key otherwise.

For each of `exp_avg`, `exp_avg_sq`, and `max_exp_avg_sq` when present, build
`{legacy_name: tensor}` (adding the tied `lm_head.weight` entry as above) and pass it through
`translate_model` with the same arguments. This gives correctly split and permuted moments under FLA
names, and gives zero moments for `gate.bias` for free.

`step` is a scalar, not parameter-shaped. Copy the source parameter's `step` to every FLA parameter it
feeds. For `gate.bias`, inherit the `step` of its source `w_gate.weight`, per the decision recorded
above: zero moments with the inherited step make its first update approximately zero and let it
accumulate normally, where `step = 0` would take a full bias-corrected step from a single gradient and
show up as a jump at the start of the decay.

Leave dtype and device alone. `AdamW.load_state_dict` casts each state tensor to its parameter's
device and dtype, and decides where `step` lives based on the target's `fused` and `capturable`
settings. Doing it here would only conflict.

### Step 4 — rebuild `param_groups`

Copy each legacy group's hyperparameters verbatim (`lr`, `betas`, `eps`, `weight_decay`, `amsgrad`,
`fused`, and the rest) and replace `params` with the FLA index lists implied by
`_ordered_param_names(fla_model, weight_decay)`.

Then check group membership: for each FLA parameter, is it in the same group index as its legacy
source? One case genuinely changes group. `w_gate.weight` is in the decay group; `gate.bias` contains
`bias`, so it lands in the no-decay group. That is correct behaviour, not a bug. Every other
disagreement is a bug. Report the `gate.bias` case as an expected exception and raise on any other.

## Wiring

`src/engine/engine.py:205-206` is the integration point:

```python
    if cfg.resume:
      self.optimizer.load_state_dict(ckpt['optimizer'])
```

A translated resume needs `ckpt['optimizer']` passed through `translate_adamw` first, gated on a
config flag. `src/constants.py` `DEFAULT_CONFIG` has no such flag today; adding one is a decision to
confirm before implementing, since the flag name shows up in every launch config. The same flag would
gate the `translate_model` call for `ckpt['state_dict']` at `src/engine/engine.py:173`, so introduce
one flag covering both, not two.

`ckpt['scheduler']` and `ckpt['scaler']` need no translation. `WSD`, `LinearCooldown` and the others in
`src/optim/lr_schedule.py` hold step counters and learning rates, and `GradScaler` holds a scale and a
growth counter. None of them reference parameters.

## Files

| File | Change |
|---|---|
| `src/optim/translate_backend.py` | new: `translate_adamw`, `_ordered_param_names`, and the probe-based name map |
| `testing/translated_optimiser_tests.py` | new: the checks below |
| `src/engine/engine.py` | the gated call at the resume path, once the flag is agreed |

Reuse, do not reimplement: `translate_model` (`src/models/translate_backend.py`), `get_param_groups`
(`src/models/legacy/construct.py:88`), `builder.parse_arch_id` and `builder.build_hybrid_layers` for
`attn_layers`, and the `_orig_mod.` stripping already in `src/utils/checkpoint_utils.py:121`.

## Verification

Needs the CUDA box. Run with `python -m testing.translated_optimiser_tests`, not
`python testing/...`, which puts `testing/` on `sys.path` instead of the repo root.

1. **Index recovery.** For each of `attn`, `gdn`, `gdn+attn_3-1`, `gdn+attn_1-3`, assert
   `_ordered_param_names(model, wd)` resolves to exactly the Parameter objects
   `get_param_groups(model, wd)` returns, in the same order, compared by `is`. Do this for both
   backends. This is the check that the whole design rests on.

2. **The probe is constant.** Assert every tensor `translate_model` returns from the probe dict has
   `t.min() == t.max()`. If a future transform mixes coordinates, this fails here rather than
   producing a silently wrong moment map.

3. **Two-step equivalence — the real test.** Build a legacy model, take two AdamW steps on fixed
   batches so the moments are non-trivial and `step == 2`. Translate weights and optimizer state into
   a fresh FLA model and optimizer. Take one more step on both backends with the same batch. Compare
   the resulting parameters through `translate_model`, using `_tensor_metrics` from
   `testing/translated_module_tests.py`. A wrong index map puts a moment on the wrong parameter and
   this diverges by far more than a rounding step. Expect agreement at the bfloat16 rounding floor,
   which for the whole-model forward was a relative token rms of 0.004 to 0.007.

4. **Identity for GDN.** For `arch_id='gdn'` the weight translation is the identity, so the translated
   moments must equal the source moments tensor for tensor, with only the indices changed. Assert
   `torch.equal`, not `allclose`.

5. **`gate.bias`.** Assert its `exp_avg` and `exp_avg_sq` are exactly zero, its `step` equals the
   `step` of `q_proj.weight` in the same layer, and that it sits in the no-decay group while its
   source sits in the decay group.

6. **Rejection.** Build a `sgd` and a `sfo_adamw` optimizer, take a step, and assert
   `translate_adamw` raises with the unexpected state key named. Assert a truncated `param_groups`
   raises rather than misaligning.

7. **A real checkpoint.** Load one `gdn+attn_3-1` checkpoint, translate weights and optimizer, and
   compare validation loss and the first few decay steps against the legacy backend on the same
   batches. This is the same run named as the next step in
   `.agents/report_gdn_translation_claude.md`; doing both translations in it at once costs nothing
   extra and is what actually clears the anneal to launch.
