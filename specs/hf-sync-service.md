# HF Sync Service — Specification

## Purpose

After a training run completes on a cluster, this service:

1. Takes inventory of all artifacts (checkpoints + metrics) on the cluster
2. Compares against the canonical HF inventory stored in the GitHub repo
3. Uploads any new or modified artifacts to HuggingFace dataset repositories
4. Updates the canonical inventory and pushes it back to GitHub

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      GitHub Repo (fastlm)                    │
│  services/inventories/hf_inventory.json   ← canonical state  │
│  services/inventories/local_{cluster}.json ← audit snapshots │
└──────────────────────────┬───────────────────────────────────┘
                           │ git pull / git push
┌──────────────────────────▼───────────────────────────────────┐
│                    Cluster (mpi / capella / alpha)           │
│  SCALING_RESULTS_FOLDER[cluster_id]/                         │
│    {arch_id}/{n}/gbs_wise_results/gbs_{gbs}/                │
│      checkpoints/lr_{lr}/                                    │
│        ckpt_decayed_to_{d}.pt                                │
│        metrics_decayed_to_{d}.json                           │
└──────────────────────────┬───────────────────────────────────┘
                           │ HfApi().upload_file()
┌──────────────────────────▼───────────────────────────────────┐
│                  HuggingFace Hub (datasets)                   │
│  jaisidh/OpenThesis_{arch_id}/                               │
│    {n}/gbs_{gbs}/lr_{lr}/                                    │
│      ckpt_decayed_to_{d}.pt    ← state_dict only             │
│      metrics_decayed_to_{d}.json                             │
└──────────────────────────────────────────────────────────────┘
```

## Data Model

### `ArtifactState` (already defined in `services/utils.py`)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `arch_id` | `str` | Model architecture | `"attn"`, `"gdn"`, `"gdn+attn_3-1"` |
| `n` | `str` | Parameter scale ID | `"20M"`, `"300M"` |
| `d` | `str` | Token budget | `"0.5B"`, `"15.0B"` |
| `gbs` | `int` | Global batch size | `16`, `512` |
| `lr` | `float` | Learning rate | `0.008`, `0.00025` |
| `checkpoint_filename` | `str` | Checkpoint file name | `"ckpt_decayed_to_0p5B.pt"` |
| `metrics_filename` | `str` | Metrics file name | `"metrics_decayed_to_0p5B.json"` |
| `cluster_location` | `str` | Origin cluster | `"mpi"` or `"tud"` |
| `mtime_spec` | `str` | Modification timestamps | `"checkpoint_2025-...__metrics_2025-..."` |

A tuple of `(arch_id, n, d, gbs, lr, checkpoint_filename, metrics_filename, cluster_location, mtime_spec)` uniquely identifies an artifact. Two artifacts with different `mtime_spec` but otherwise identical fields are treated as **different** (i.e., a modified artifact triggers re-upload).

### `Inventory` (already defined in `services/utils.py`)

Columnar store backed by `defaultdict(list)`. Supports `push`, `pop`, `save` (JSON/CSV), `load` (JSON/CSV).

---

## Sync Flow (end-to-end)

### Step 0 — `git pull`

Before the sync runs, pull the latest `services/inventories/` from GitHub to get the most recent `hf_inventory.json`.

### Step 1 — `take_inventory(cluster_id)`

Already implemented in `services/utils.py`. Walks `SCALING_RESULTS_FOLDER[cluster_id]`, finds all `metrics_*.json` files, parses their paths into `ArtifactState` entries, returns an `Inventory`.

### Step 2 — `load_hf_state()`

Reads `services/inventories/hf_inventory.json` into an `Inventory` object. Returns an empty `Inventory` if the file does not exist (first run).

### Step 3 — `difference(local_inventory, hf_inventory)`

Already implemented in `services/utils.py`. Returns a `pd.DataFrame` of rows present in `local_inventory` but absent from `hf_inventory`. Returns `None` if there are no differences (nothing to upload).

### Step 4 — Upload loop

For each row in the changes DataFrame:

#### 4a. Ensure HF dataset repo exists

Check if `jaisidh/OpenThesis_{arch_id}` exists on HF Hub (type=`dataset`). If not, create it via `HfApi().create_repo()`.

#### 4b. Extract state_dict to temp file

Load the full checkpoint (contains `state_dict`, `optimizer`, `scheduler`, `scaler`, `config`), extract only `state_dict`, and save to a temp file under `TMP_FOLDER_FOR_UPLOAD[cluster_id]`. This reduces upload size and strips unnecessary training state.

#### 4c. Upload checkpoint state_dict

Upload the stripped `.pt` file to `jaisidh/OpenThesis_{arch_id}` at path `{n}/gbs_{gbs}/lr_{lr}/ckpt_decayed_to_{d}.pt`.

#### 4d. Upload metrics

Upload the metrics `.json` file to the same repo at path `{n}/gbs_{gbs}/lr_{lr}/metrics_decayed_to_{d}.json`.

#### 4e. Clean up temp file

Delete the temp state_dict file after successful upload.

### Step 5 — Update HF inventory

After **all** uploads succeed (not per-file), append the uploaded `ArtifactState` entries to `hf_inventory.json` and save it back to `services/inventories/hf_inventory.json`.

> **Important:** If any upload fails, the inventory is NOT updated. The failed artifact will be retried on the next sync run.

### Step 6 — Save local snapshot

Save the full local inventory to `services/inventories/local_inventory_{cluster_id}.json` for audit purposes.

### Step 7 — `git commit && git push`

Commit the updated `services/inventories/` files and push to GitHub.

---

## Files to Create / Modify

### `fastlm/services/sync.py` — Complete implementation

Replace all stubs with working functions:

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `load_hf_state()` | — | `Inventory` | Load `services/inventories/hf_inventory.json` |
| `ensure_hf_repo_exists(api, arch_id)` | `HfApi`, `str` | — | Create HF dataset repo if missing |
| `extract_model_state_dict_to_tmpfile(ckpt_path, cluster_id)` | `str`, `str` | `str` (tmp path) | Extract `state_dict`, save to tmp |
| `upload_artifact_to_hf(api, arch_id, src_path, dest_path, repo_type)` | `HfApi`, `str`, `str`, `str`, `str` | — | Upload a single file to HF |
| `update_hf_inventory(new_entries_df)` | `pd.DataFrame` | — | Append entries to `hf_inventory.json` |
| `main(cfg)` | `SyncServiceConfig` | — | Orchestrates the full sync flow |

### `fastlm/services/utils.py` — Minor fixes

| Change | Reason |
|--------|--------|
| Remove `beartype` dependency | Not in `pyproject.toml`, not needed for correctness |
| Fix `Inventory.load` CSV path | Currently expects `row['key']`/`row['value']` which doesn't match the columnar save format. Either fix or deprecate CSV support in favor of JSON-only. |

### `fastlm/pyproject.toml` — Add dependencies

Add `pandas`, `huggingface_hub`, `tyro` to the `[project] dependencies` list.

### `fastlm/services/inventories/` — Will be populated at runtime

- `hf_inventory.json` — canonical inventory (committed to git)
- `local_inventory_{cluster_id}.json` — per-cluster snapshots (committed to git)

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| No differences found | Print message, exit 0 |
| HF repo doesn't exist | Create it, then upload |
| Upload fails | Print error, skip inventory update, exit 1 |
| `hf_inventory.json` doesn't exist | Treat as empty (first run) |
| Temp file cleanup fails | Print warning, continue (non-fatal) |
| `TMP_FOLDER_FOR_UPLOAD` doesn't exist | Create it (`os.makedirs`) |

---

## CLI Interface

```bash
python -m services.sync --cluster-id mpi
python -m services.sync --cluster-id capella
python -m services.sync --cluster-id alpha
```

`SyncServiceConfig`:
```python
@dataclass
class SyncServiceConfig:
  cluster_id: str  # one of: mpi, capella, alpha
```

---

## Assumptions & Constraints

- User is authenticated with HuggingFace (`huggingface-cli login` or `HF_TOKEN` env var)
- `SCALING_RESULTS_FOLDER` and `TMP_FOLDER_FOR_UPLOAD` paths exist on the cluster
- `services/inventories/` is tracked in the main `fastlm` git repo
- One central `hf_inventory.json` tracks all `arch_id`s together
- `git pull` / `git push` are performed manually or by a wrapper script around this service (the service itself does NOT run git commands)
- Checkpoint files contain `state_dict` key; metrics files are valid JSON
