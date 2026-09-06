# Translate legacy Gated DeltaNet into the FLA backend

> Status: plan only, nothing implemented. Written 2026-09-06 against commit `ad64878`.
> Every claim below was verified by reading the files named, or by the check described in
> "How the findings were verified". Re-verify before relying on any of it — the FLA vendored
> tree changes.

## Orientation

Two model backends exist in this repo and must produce identical results:

- **legacy** — `src/models/legacy/transformer.py` (`Transformer`, `Block`, `ModelConfig`).
  Its GDN layer is `fla/layers/legacy_gated_deltanet.py` (`LegacyGatedDeltaNet`), imported by
  `transformer.py` under the alias `GatedDeltaNet`.
- **FLA** — the vendored `fla/` tree, assembled from a config built by `src/models/builder.py`
  and instantiated with `AutoModelForCausalLM.from_config(...)`. Its GDN layer is
  `fla/layers/gated_deltanet.py` (`GatedDeltaNet`); its model is
  `fla/models/gated_deltanet/modeling_gated_deltanet.py`.

The bridge between them is `src/models/translate_backend.py`, which converts a **layer-local**
legacy state dict into FLA keys. It currently covers `gated_attention`, `ffn`, `embeddings`,
`rms_norm`, and `lm_head` — one `translate_*` function each. The harness that exercises them is
`testing/translated_module_tests.py`.

Naming trap worth knowing up front: `fla/layers/gated_deltanet.py` and
`fla/layers/legacy_gated_deltanet.py` both define a class reachable as `GatedDeltaNet`. Import
them explicitly and do not rely on the bare name.

## Context

`testing/translated_module_tests.py` verifies that the FLA backend reproduces the legacy backend for
`gated_attention`, `ffn`, `embeddings`, `rms_norm`, and `lm_head`. GDN is missing — the last commit is
literally `complete translation tests (except gdn)`. GDN is the layer the thesis is about, so the
untested module is the important one.

The goal: translate a legacy GDN state dict into the FLA backend and verify both produce matching
outputs.

Investigation found the translation itself is nearly trivial, and the real work is elsewhere. The
`__init__` bodies of `fla/layers/legacy_gated_deltanet.py` and `fla/layers/gated_deltanet.py` are
byte-identical in every parameter registration. Diffing the two class bodies yields only:

- legacy has an extra `intra_doc: bool = False` ctor kwarg and `self.intra_doc` (registers no parameter)
- all remaining differences are in `forward` (arg names, unpad/repad helpers, a runtime conv fusion)

So `translate_gated_deltanet` is an **identity map on all 13 keys**. What actually blocks the objective
is `src/models/builder.py`, which cannot currently build a GDN or hybrid model that matches legacy.

## GDN parameter inventory

Identical for both `LegacyGatedDeltaNet` and the current `GatedDeltaNet`. Legacy `Block` passes
`hidden_size=cfg.dim, num_heads=cfg.n_heads, head_dim=cfg.dim // cfg.n_heads` and does **not** pass
`expand_v`, `num_v_heads`, `use_short_conv`, `conv_bias`, `mode`, or `norm_eps`, so those take layer
defaults. With `D = hidden_size`, `expand_v = 2.0` (default), `num_v_heads = num_heads = H`:

| dim | value |
|---|---|
| `head_k_dim` | `head_dim` = `D / H` |
| `head_v_dim` | `int(head_dim * expand_v)` = `2D / H` |
| `key_dim` | `H * head_k_dim` = `D` |
| `value_dim` | `H * head_v_dim` = `2D` |

| key | shape | present when |
|---|---|---|
| `q_proj.weight` | `[key_dim, D]` | always |
| `k_proj.weight` | `[key_dim, D]` | always |
| `v_proj.weight` | `[value_dim, D]` | always |
| `a_proj.weight` | `[num_v_heads, D]` | always |
| `b_proj.weight` | `[num_v_heads, D]` | always |
| `A_log` | `[num_v_heads]`, float32 | always |
| `dt_bias` | `[num_v_heads]`, float32 | always |
| `q_conv1d.weight` | `[key_dim, 1, conv_size]` | `use_short_conv` (default True) |
| `k_conv1d.weight` | `[key_dim, 1, conv_size]` | `use_short_conv` |
| `v_conv1d.weight` | `[value_dim, 1, conv_size]` | `use_short_conv` |
| `g_proj.weight` | `[value_dim, D]` | `use_gate` |
| `o_norm.weight` | `[head_v_dim]` | always |
| `o_proj.weight` | `[D, value_dim]` | always |

`*_conv1d.bias` (`[channels]`) appears only with `conv_bias=True`, which is False on every path
through `builder.py` (`conv_bias` is not a `GatedDeltaNetConfig` field). `o_norm.bias` is registered
as `None` in both the `FusedRMSNormGated` (gated) and `RMSNorm` (ungated) branches and never appears.
`allow_neg_eigval` and `conv_size` register no parameters — the former is a kernel flag, the latter
only sets conv width.

`ShortConvolution` subclasses `nn.Conv1d` with `groups=channels`, hence the depthwise `[C, 1, k]`
weight. `a_proj`/`b_proj` are sized `num_v_heads`, **not** `value_dim`.

There are **no** `q_norm`/`k_norm` modules in GDN; q/k L2-norm happens inside the kernel via
`use_qk_l2norm_in_kernel=True`. Likewise beta sigmoid and the `A_log`/`dt_bias` gating are computed
in-kernel, so `a_proj`/`b_proj` outputs are raw pre-activations.

Fully qualified in a legacy `Transformer`, these sit under `layers.{i}.token_mixer.*`. In the FLA
model they sit under `model.layers.{i}.attn.*` — FLA names the token mixer `self.attn` for **both**
the attention and GDN branches, so the key name alone never tells you a layer's type.

## How the findings were verified

Anyone extending this plan should know which claims were checked and how.

- **Identical `__init__`** — `diff` of the two class bodies, sliced from `def __init__` to the next
  `def`. Output was only the `intra_doc` kwarg, `self.intra_doc`, and forward-path differences.
  Re-run: `diff <(sed -n '/  def __init__(/,/^  def forward\|^  def _use_fused/p' fla/layers/legacy_gated_deltanet.py) <(sed -n '/  def __init__(/,/^  def forward\|^  def _use_fused/p' fla/layers/gated_deltanet.py)`
- **Identical registered names** — grep for `self.<name> = ` over both files, sorted and compared.
- **S3 `KeyError`** — simulated the config's `attn` validation block in plain Python (no `fla`
  import) against the dict `builder.py` assigns. All three of `qkv_bias`, `window_size`,
  `rope_theta` were absent. Not observed as a live traceback — `fla` will not import here.
- **S1, S2, S4, S5** — read directly from `configuration_gated_deltanet.py` (defaults) and
  `modeling_gated_deltanet.py:50-59` (which kwargs reach `Attention`).
- **Not executed at all**: no GDN forward, no output comparison, no load. See Verification.

## Subproblems

### S1 — `builder.py` never passes `head_dim` (blocking, wrong shapes)

`get_pure_model_config` / `get_hybrid_model_config` pass only `hidden_size, num_heads,
num_hidden_layers, intermediate_size, max_position_embeddings, vocab_size` plus `expand_v`.
`GatedDeltaNetConfig.head_dim` therefore stays at its default **256**, while legacy `Block` uses
`head_dim = cfg.dim // cfg.n_heads`. For `dim=64, n_heads=4` that is 16, not 256 — every GDN
projection comes out the wrong shape and `load_state_dict` fails before any output comparison.

### S2 — `builder.py` drops the `gdn_*` flags (blocking, wrong algorithm)

`src/constants.py` `DEFAULT_CONFIG` carries `gdn_conv_size: 4`, `gdn_gate: True`,
`gdn_neg_eigval: True`, `intra_doc_masking: True`. None reach `GatedDeltaNetConfig`. The defaults it
falls back to are `conv_size=4` (matches by luck), `use_gate=True` (matches by luck), and
**`allow_neg_eigval=False`** (does *not* match `gdn_neg_eigval: True`). `allow_neg_eigval` changes the
kernel's beta scaling, so outputs diverge even with correctly-shaped weights.

### S3 — hybrid configs raise `KeyError` before a model is built (blocking)

`get_hybrid_model_config` constructs `GatedDeltaNetConfig(...)` **without** `attn`, then assigns
`config.attn = attn_config_to_insert` afterwards. The config's `attn` validation block — which injects
`qkv_bias`, `window_size`, `rope_theta` defaults — runs only inside `__init__`, so it never executes.
`GatedDeltaNetBlock.__init__` then reads `config.attn['qkv_bias']` and raises `KeyError`.
Verified by simulating the validation logic: the assigned dict has only
`{hidden_size, layers, num_heads, num_kv_heads, qk_norm, use_gate}`, so all three reads fail.

### S4 — hybrid attention layers ignore `qk_norm` and `use_gate` (blocking for hybrids)

`GatedDeltaNetBlock.__init__` (`fla/models/gated_deltanet/modeling_gated_deltanet.py:50-59`) forwards
only `hidden_size, num_heads, num_kv_heads, qkv_bias, window_size, rope_theta,
max_position_embeddings, layer_idx` to `Attention`. The `qk_norm` and `use_gate` entries that
`builder.py` puts in the attn dict are silently dropped, so hybrid attention layers are built with
`use_gate=False, qk_norm=False`. Legacy defaults both to `True`, so `w_gate`/`q_norm`/`k_norm` have no
destination and `strict=True` loading fails.

### S5 — `rope_theta` defaults to 10000, legacy uses 500000

The config's attn validation defaults `rope_theta` to `10000.`; legacy `Transformer` uses `500000`.
The existing per-module test hides this because `_fla_gated_attention` hardcodes `rope_theta=500000`;
the builder path would not. Silent numerical divergence, not a load error.

### S6 — `_forward` in the test harness has no GDN branch

`_forward` special-cases `gated_attention` and otherwise calls `module(inputs)`. Both GDN forwards
return a 3-tuple, so the bare call returns a tuple and the comparison breaks. GDN needs `[0]` but,
unlike attention, needs no `freqs_cis` and no RoPE permutation.

### S7 — `_normalize_config` does not default the GDN fields

`gdn_conv_size`, `gdn_gate`, `gdn_neg_eigval`, `intra_doc`, `expand_v` are never `setdefault`-ed, so
the GDN factories would `AttributeError` on a minimal config.

### S8 — `seq_len=16` silently routes GDN to a different kernel

Both GDN forwards do `mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode`.
The harness calls `.eval()` and uses `seq_len=16`, so GDN runs `fused_recurrent`, never the `chunk`
kernel used in training. Chunk size is 64. A passing test at `seq_len=16` says nothing about the
training path.

### S9 — the backward test's strict translator will trip on dropped gradients

`test_translated_module_bwd_pass` calls `translate_module(_gradients(legacy_module))`, and `_gradients`
silently drops any parameter whose `.grad is None`. A strict `translate_gated_deltanet` that requires
all 13 keys will raise a `missing_keys` `ValueError` on the gradient dict rather than report a
gradient mismatch.

### S10 — `_no_weight_decay` does not survive a state dict round trip

`A_log._no_weight_decay = True` and `dt_bias._no_weight_decay = True` are Python attributes on the
Parameter objects, not buffers. They are set in both `__init__`s, so a translated model that is
*constructed* keeps them. Worth noting only so the translation is not made to rebuild Parameters.

## Plan

### Step 1 — fix `src/models/builder.py` (S1, S2, S3, S4, S5)

`build_kwargs` currently only sets `expand_v`. Extend it to carry the full GDN mapping:

```python
def build_kwargs(cfg: SimpleNamespace, arch: str):
  kwargs = {}
  if 'gdn' in arch:
    kwargs['expand_v'] = vars(cfg).get('expand_v', 2)
    kwargs['head_dim'] = cfg.d_model // cfg.n_heads
    kwargs['conv_size'] = cfg.gdn_conv_size
    kwargs['use_gate'] = cfg.gdn_gate
    kwargs['allow_neg_eigval'] = cfg.gdn_neg_eigval
  return kwargs
```

`head_dim = cfg.d_model // cfg.n_heads` is the value legacy `Block` passes; it must be explicit
because `GatedDeltaNetConfig.head_dim` defaults to 256.

Note `build_kwargs` is called from both `get_pure_model_config` and `get_hybrid_model_config`, but its
`'gdn' in arch` guard uses `arch`, which for a hybrid `arch_id` like `gdn+attn_3-1` is the string
`gdn+attn` — so the guard already passes for hybrids. Verify this when implementing.

In `get_hybrid_model_config`, pass `attn` **into the constructor** rather than assigning it after, so
the config's validation runs and injects `qkv_bias`/`window_size`/`rope_theta`. Also set `rope_theta`
explicitly to match legacy:

```python
  attn_config_to_insert = dict(
    layers=build_hybrid_layers(cfg.n_layers, ratio),
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_kv_heads=cfg.n_heads,
    qk_norm=cfg.attn_qk_norm,
    use_gate=cfg.attn_gate,
    rope_theta=500000,
  )
  config = GatedDeltaNetConfig(
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_hidden_layers=cfg.n_layers,
    intermediate_size=int(cfg.d_model * float(Fraction(cfg.expand))),
    max_position_embeddings=cfg.seq_len,
    vocab_size=cfg.vocab_size,
    attn=attn_config_to_insert,
    **kwargs,
  )
  return config
```

S4 (`qk_norm`/`use_gate` dropped by `GatedDeltaNetBlock`) is in `fla/`, not `src/`. Two options —
decide before implementing:

- **Preferred**: patch `fla/models/gated_deltanet/modeling_gated_deltanet.py` to forward
  `qk_norm=config.attn.get('qk_norm', False)` and `use_gate=config.attn.get('use_gate', False)` to
  `Attention`. Small, local, and `fla.layers.attn.Attention` already accepts both.
- Alternative: leave `fla/` untouched and restrict hybrid verification to `attn_gate=False,
  attn_qk_norm=False`. This dodges the bug rather than fixing it and diverges from
  `DEFAULT_CONFIG`, which sets both `True`.

S4 only blocks **hybrid** models. Pure `gdn` needs S1–S3 only, so pure-GDN verification can proceed
independently of the S4 decision.

### Step 2 — add `translate_gated_deltanet` to `src/models/translate_backend.py`

Identity map with full validation, matching the file's existing conventions (keyword-only args after
`*`, `ValueError` never `assert`, `tuple(x.shape)` in messages, the
`missing keys=... unexpected keys=...` template).

Signature:

```python
def translate_gated_deltanet(
  state_dict: Mapping[str, Tensor],
  *,
  num_heads: int | None = None,
) -> dict[str, Tensor]:
```

Key sets:

```python
  required_keys = {
    'q_proj.weight', 'k_proj.weight', 'v_proj.weight',
    'a_proj.weight', 'b_proj.weight',
    'A_log', 'dt_bias',
    'o_norm.weight', 'o_proj.weight',
  }
  conv_keys = {'q_conv1d.weight', 'k_conv1d.weight', 'v_conv1d.weight'}
  conv_bias_keys = {'q_conv1d.bias', 'k_conv1d.bias', 'v_conv1d.bias'}
  gate_keys = {'g_proj.weight'}
  allowed_keys = required_keys | conv_keys | conv_bias_keys | gate_keys
```

Validation to implement:

- required/unexpected key check using the existing message template
- co-occurrence: the three `*_conv1d.weight` keys are all-or-none (mirrors the existing
  `q_norm`/`k_norm` check); same for the three conv biases
- shape consistency derived from the tensors themselves, since no config is passed:
  - `hidden_size = q_proj.weight.shape[1]`; `key_dim = q_proj.weight.shape[0]`
  - `k_proj.weight.shape == q_proj.weight.shape`
  - `value_dim = v_proj.weight.shape[0]`, with `v_proj.weight.shape[1] == hidden_size`
  - `num_v_heads = a_proj.weight.shape[0]`, `b_proj.weight.shape == a_proj.weight.shape`,
    `a_proj.weight.shape[1] == hidden_size`
  - `A_log.ndim == 1` and `A_log.shape == (num_v_heads,)`; same for `dt_bias`
  - `o_proj.weight.shape == (hidden_size, value_dim)`
  - `value_dim % num_v_heads == 0`; `head_v_dim = value_dim // num_v_heads`;
    `o_norm.weight.shape == (head_v_dim,)`
  - conv weights are 3-D and depthwise: `q/k_conv1d.weight.shape == (key_dim, 1, conv_size)` and
    `v_conv1d.weight.shape == (value_dim, 1, conv_size)`, all sharing one `conv_size`
  - `g_proj.weight.shape == (value_dim, hidden_size)` when present
  - when `num_heads` is given, check `key_dim % num_heads == 0` (validate it the way
    `translate_gated_attention` does, including the explicit `isinstance(num_heads, bool)` exclusion)

Return `dict(state_dict)` — a new dict, same tensor objects, sharing storage.

Docstring must follow the file's four-part convention and state explicitly that: the mapping is the
identity because both layers register identical parameters; `expand_v`, `allow_neg_eigval`,
`conv_size`, `use_gate`, `norm_eps`, and `intra_doc` are **not** encoded in the state dict and must be
matched on the target separately; `allow_neg_eigval` in particular changes the kernel result without
changing any shape.

### Step 3 — add the `gdn` spec to `testing/translated_module_tests.py` (S6, S7)

Factories, following the existing `(config, fla_config)` signature:

```python
def _legacy_gdn(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return LegacyGatedDeltaNet(
    hidden_size=config.dim,
    num_heads=config.n_heads,
    head_dim=config.dim // config.n_heads,
    allow_neg_eigval=config.gdn_neg_eigval,
    use_gate=config.gdn_gate,
    conv_size=config.gdn_conv_size,
    intra_doc=config.intra_doc,
  )


def _fla_gdn(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return GatedDeltaNet(
    hidden_size=fla_config.hidden_size,
    num_heads=config.n_heads,
    head_dim=config.dim // config.n_heads,
    expand_v=fla_config.expand_v,
    allow_neg_eigval=config.gdn_neg_eigval,
    use_gate=config.gdn_gate,
    conv_size=config.gdn_conv_size,
    norm_eps=fla_config.norm_eps,
  )
```

The legacy factory mirrors exactly what `Block.__init__` passes. Register both in the three spec maps
under `'gdn'`, and add `translate_gated_deltanet` to `TRANSLATE_MODULE_SPEC_MAP`.

`_normalize_config`: add `setdefault`s for `gdn_conv_size` (4), `gdn_gate` (True), `gdn_neg_eigval`
(True), `intra_doc` (False). Use `False` for `intra_doc` in the harness — the module tests pass no
`cu_seqlens` or `linear_mask`, and `intra_doc=True` reshapes on a `linear_mask` that is `None`.

`_forward`: widen the guard so GDN takes `[0]` without the attention-specific arguments:

```python
  if module_spec == 'gdn':
    return module(inputs)[0]

  if module_spec != 'gated_attention':
    return module(inputs)
```

Note `norm_eps`: legacy `Block` does not pass it, so the GDN layer uses its own default `1e-5`, while
`GatedDeltaNetConfig.norm_eps` defaults to `1e-6`. Pass the same value to both factories or the
`o_norm` epsilons differ. Reading it from `fla_config.norm_eps` on the FLA side but taking the layer
default on the legacy side is exactly the kind of silent mismatch to avoid — pin both to `1e-5`, or
pass `norm_eps=config.rmsnorm_eps` to both.

### Step 4 — make the GDN test exercise the training kernel (S8)

Add a `seq_len` override so GDN runs at a length above the 64-token chunk boundary. `main()` currently
builds one config for all specs; give GDN its own config with `seq_len=128` (and lift the
`min(config.seq_len, 16)` cap in `_make_inputs`, or make the cap config-driven). Without this, the GDN
test only ever exercises `fused_recurrent`, never the `chunk` kernel used in training.

Run the GDN comparison in both regimes and report both: `seq_len=16` (`fused_recurrent`) and
`seq_len=128` (`chunk`, with the modules in `.train()` so the mode switch does not fire).

### Step 5 — handle the backward-test gradient case (S9)

Either make `translate_gated_deltanet` tolerant when called on a gradient dict, or special-case `gdn`
in `test_translated_module_bwd_pass` to compare gradients by name without re-running strict
validation. Prefer the latter — it keeps the translator strict, which is the point of Step 2. Decide
during implementation once it is visible which GDN parameters actually receive gradients from a
`sum()` loss.

## Suggested order

Steps 1–3 are independent enough to do in any order, but this sequence keeps each one checkable:

1. **Step 1 (builder)** first — until S1/S2 are fixed nothing downstream can be shape-correct.
   Checkable locally via Verification step 1 (config assertions, no GPU, no `fla` import if you
   assert on the config object alone — but note `builder.py` imports `fla.models`, so this needs
   the CUDA box or a triton-capable env).
2. **Step 2 (translator)** next — pure Python, no `fla` import at runtime
   (`translate_backend.py` guards the `Tensor` import under `TYPE_CHECKING`), so its validation
   logic is unit-testable locally against hand-built dicts of the right shapes.
3. **Step 3 (harness)** to wire the two together.
4. **Steps 4–5** once the basic GDN comparison runs green.

Do not skip Verification step 4 (flag sensitivity). Steps 2 and 3 can both "pass" while testing
nothing, because an identity translator over identically-constructed modules trivially agrees —
step 4 is what proves the comparison has teeth.

## Files to modify

| File | Change |
|---|---|
| `src/models/builder.py` | `build_kwargs` GDN mapping (S1, S2); `get_hybrid_model_config` passes `attn` into ctor + `rope_theta` (S3, S5) |
| `src/models/translate_backend.py` | add `translate_gated_deltanet` (Step 2) |
| `testing/translated_module_tests.py` | GDN factories, spec-map entries, `_normalize_config` defaults, `_forward` branch, seq_len handling (S6, S7, S8, S9) |
| `fla/models/gated_deltanet/modeling_gated_deltanet.py` | forward `qk_norm`/`use_gate` to `Attention` (S4) — only if the preferred option is taken |

Reuse rather than rewrite: `builder.parse_arch_id` and `builder.build_hybrid_layers` already implement
the per-layer attn-vs-gdn decision with the same modulo rules as
`Transformer._prepare_layers`. Any later full-model translator should call them, not reimplement.

## Verification

**All verification below must run on the CUDA box.** On the darwin/arm64 dev machine `torch` is
absent from the repo environment, and even with a torch env on `PYTHONPATH=.` any `import fla`
fails at `fla/ops/abc/chunk.py` with `ModuleNotFoundError: No module named 'triton'`. `triton` does
not install on darwin/arm64. Consequences for whoever implements this:

- You can read, edit, and lint (`uvx ruff check`, `uvx ruff format`) locally.
- You can unit-test pure-Python logic by loading a module file directly with
  `importlib.util.spec_from_file_location`, bypassing `src/models/__init__.py` (which imports `fla`
  transitively and will fail). This is how the S3 check and the earlier init-alignment checks were run.
- You **cannot** run `testing/translated_module_tests.py`, instantiate any GDN layer, or compare any
  output locally. Do not report those as passing without a CUDA run.

1. **Config check (fast, catches S1–S3 without a GPU forward).** Build a config via
   `builder.config_builder` for `arch_id='gdn'` and for `arch_id='gdn+attn_3-1'`; assert
   `config.head_dim == d_model // n_heads`, `config.allow_neg_eigval == cfg.gdn_neg_eigval`,
   `config.conv_size == cfg.gdn_conv_size`, and that `config.attn` contains `qkv_bias`,
   `window_size`, `rope_theta` with `rope_theta == 500000`.

2. **Module-level GDN, forward.** `test_translated_module_fwd_pass('gdn', ...)` and
   `test_initialization_fwd_pass('gdn', ...)` at `model_dtype='float32'`. Expect
   `max absolute error` at float32 round-off. Run at `seq_len=16` and `seq_len=128` per Step 4.

3. **Module-level GDN, backward.** `test_translated_module_bwd_pass('gdn', ...)`; confirm gradients
   match for every parameter that receives one, and that `A_log`/`dt_bias` are among them.

4. **Flag sensitivity (guards against a vacuous pass).** Re-run step 2 with `gdn_neg_eigval` flipped
   on one side only. The test **must fail**. If it passes, the flag is not reaching the layer and the
   comparison is not testing what it claims.

5. **Full-model check.** Build a legacy `Transformer` with `token_mixer='gdn'`, translate the whole
   state dict, load into the FLA model, compare logits. Then repeat for one hybrid ratio
   (`gdn+attn_3-1`) to exercise the per-layer type decision. This is the check that catches
   composition bugs the per-module tests structurally cannot — note that FLA names the token mixer
   `self.attn` for **both** branches, so layer type must come from
   `build_hybrid_layers`, not the key name. A full-model translator does not exist yet; scope it as
   follow-up work if Step 5 grows past this plan.

6. **bf16 on CUDA.** Repeat step 2 at `model_dtype='bfloat16'` and record the error magnitude
   separately — float32 agreement does not imply bf16 agreement, and bf16 is the training regime.

Prefix note for step 5: real checkpoints nest under `state['state_dict']` and carry a `_orig_mod.`
prefix from `torch.compile`. `src/utils/checkpoint_utils.py:121` and
`src/models/legacy/to_hf.py:189` already strip it — reuse that handling.
