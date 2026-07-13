import os
from dataclasses import dataclass

import pandas as pd
import torch
import tyro
from huggingface_hub import HfApi

from services.utils import (
  ArtifactState,
  Inventory,
  difference,
  get_checkpoints_from_changes,
  take_inventory,
)
from src.constants import TMP_FOLDER_FOR_UPLOAD

_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_INVENTORIES_DIR = os.path.join(_SERVICES_DIR, 'inventories')
_HF_INVENTORY_PATH = os.path.join(_INVENTORIES_DIR, 'hf_inventory.json')
_HF_USERNAME = 'jaisidh'


@dataclass
class SyncServiceConfig:
  cluster_id: str  # one of: mpi, capella, alpha


# ---------------------------------------------------------------------------
# Step 2: load canonical HF inventory from GitHub-tracked JSON
# ---------------------------------------------------------------------------


def load_hf_state() -> Inventory:
  """Load the canonical HF inventory from the locally-tracked JSON file.

  Returns an empty Inventory if the file doesn't exist (first run).
  """
  hf_inventory = Inventory()
  if os.path.exists(_HF_INVENTORY_PATH):
    hf_inventory.load(_HF_INVENTORY_PATH)
    print(f'Loaded HF inventory with {len(hf_inventory)} entries from {_HF_INVENTORY_PATH}')
  else:
    print(f'No existing HF inventory found at {_HF_INVENTORY_PATH} — starting fresh.')
  return hf_inventory


# ---------------------------------------------------------------------------
# Step 4a: ensure the HF dataset repo exists
# ---------------------------------------------------------------------------


def ensure_hf_repo_exists(api: HfApi, arch_id: str) -> str:
  """Make sure the HF dataset repo for `arch_id` exists; create it if not.

  Returns the repo_id.
  """
  repo_id = f'{_HF_USERNAME}/OpenThesis_{arch_id}'
  print(f'Ensuring HF repo exists: {repo_id} ...')
  api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)
  return repo_id


# ---------------------------------------------------------------------------
# Step 4b: extract state_dict from a full checkpoint into a temp file
# ---------------------------------------------------------------------------


def extract_model_state_dict_to_tmpfile(checkpoint_path: str, cluster_id: str) -> str:
  """Load a full training checkpoint, extract only `state_dict`, save to tmp.

  Returns the path to the temporary file.
  """
  tmp_dir = TMP_FOLDER_FOR_UPLOAD[cluster_id]
  os.makedirs(tmp_dir, exist_ok=True)

  print(f'  Extracting state_dict from {checkpoint_path} ...')
  ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
  state_dict = ckpt['state_dict']

  fname = os.path.basename(checkpoint_path)
  tmpfile_path = os.path.join(tmp_dir, fname)
  torch.save(state_dict, tmpfile_path)
  print(f'  Saved state_dict to {tmpfile_path}')
  return tmpfile_path


# ---------------------------------------------------------------------------
# Step 4c/4d: upload a single file to HF
# ---------------------------------------------------------------------------


def upload_file_to_hf(api: HfApi, repo_id: str, src_path: str, dest_path: str) -> None:
  """Upload a single file to a HuggingFace dataset repo."""
  print(f'  Uploading {src_path}  ->  {repo_id}/{dest_path}')
  api.upload_file(
    path_or_fileobj=src_path,
    path_in_repo=dest_path,
    repo_id=repo_id,
    repo_type='dataset',
  )


# ---------------------------------------------------------------------------
# Step 5: append uploaded entries to the canonical HF inventory
# ---------------------------------------------------------------------------


def update_hf_inventory(uploaded_entries: pd.DataFrame) -> None:
  """Append successfully-uploaded artifact entries to hf_inventory.json."""
  hf_inventory = Inventory()
  if os.path.exists(_HF_INVENTORY_PATH):
    hf_inventory.load(_HF_INVENTORY_PATH)

  for _, row in uploaded_entries.iterrows():
    state = ArtifactState(
      arch_id=str(row['arch_id']),
      n=str(row['n']),
      d=str(row['d']),
      gbs=int(str(row['gbs'])),
      lr=float(str(row['lr'])),
      checkpoint_filename=str(row['checkpoint_filename']),
      metrics_filename=str(row['metrics_filename']),
      cluster_location=str(row['cluster_location']),
      mtime_spec=str(row['mtime_spec']),
    )
    hf_inventory.push(state)

  os.makedirs(_INVENTORIES_DIR, exist_ok=True)
  hf_inventory.save(_HF_INVENTORY_PATH, format='json')
  print(f'Updated HF inventory: {len(hf_inventory)} total entries at {_HF_INVENTORY_PATH}')


# ---------------------------------------------------------------------------
# Step 6: save a local snapshot for audit
# ---------------------------------------------------------------------------


def save_local_snapshot(local_inventory: Inventory, cluster_id: str) -> None:
  """Save the full local inventory to a per-cluster snapshot file."""
  snapshot_path = os.path.join(_INVENTORIES_DIR, f'local_inventory_{cluster_id}.json')
  os.makedirs(_INVENTORIES_DIR, exist_ok=True)
  local_inventory.save(snapshot_path, format='json')
  print(f'Saved local snapshot ({len(local_inventory)} entries) to {snapshot_path}')


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(cfg: SyncServiceConfig) -> None:
  cluster_id = cfg.cluster_id

  # Step 1: take inventory of the cluster
  print(f'\n=== Step 1: Taking inventory of cluster "{cluster_id}" ===')
  local_inventory = take_inventory(cluster_id)
  print(f'Found {len(local_inventory)} artifacts on cluster "{cluster_id}".')

  # Step 2: load canonical HF state
  print('\n=== Step 2: Loading HF state ===')
  hf_inventory = load_hf_state()

  # Step 3: diff
  print('\n=== Step 3: Comparing inventories ===')
  changes = difference(local_inventory, hf_inventory)
  if changes is None:
    print('No new or modified artifacts found — nothing to upload.')
    save_local_snapshot(local_inventory, cluster_id)
    return

  num_changes = len(changes)
  print(f'Found {num_changes} artifact(s) to upload.')

  # Step 4: get source/destination paths for every changed artifact
  upload_paths = get_checkpoints_from_changes(changes, cluster_id)
  api = HfApi()
  tmp_files_to_clean: list[str] = []

  try:
    # Step 4 loop — upload each artifact
    print('\n=== Step 4: Uploading artifacts to HuggingFace ===')
    for i in range(num_changes):
      arch_id = str(changes.iloc[i]['arch_id'])
      ckpt_src = upload_paths['checkpoints_src'][i]
      metrics_src = upload_paths['metrics_src'][i]
      ckpt_dest = upload_paths['checkpoints_dest'][i]
      metrics_dest = upload_paths['metrics_dest'][i]

      print(
        f'\n--- Artifact {i + 1}/{num_changes}: {arch_id} / {changes.iloc[i]["n"]} / gbs_{changes.iloc[i]["gbs"]} / lr_{changes.iloc[i]["lr"]} / {changes.iloc[i]["d"]} ---'
      )

      # 4a: ensure HF repo exists
      repo_id = ensure_hf_repo_exists(api, arch_id)

      # 4b: extract state_dict to temp file
      if not os.path.exists(ckpt_src):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_src}')
      tmp_ckpt = extract_model_state_dict_to_tmpfile(ckpt_src, cluster_id)
      tmp_files_to_clean.append(tmp_ckpt)

      # 4c: upload state_dict checkpoint
      upload_file_to_hf(api, repo_id, tmp_ckpt, ckpt_dest)

      # 4d: upload metrics
      if not os.path.exists(metrics_src):
        raise FileNotFoundError(f'Metrics file not found: {metrics_src}')
      upload_file_to_hf(api, repo_id, metrics_src, metrics_dest)

      print(f'  ✓ Uploaded {arch_id} artifact successfully.')

    # Step 5: update HF inventory (only after all uploads succeed)
    print('\n=== Step 5: Updating HF inventory ===')
    update_hf_inventory(changes)

  except Exception as e:
    print(f'\n!!! Upload failed: {e}')
    print('HF inventory was NOT updated — artifacts will be retried on next sync.')
    raise

  finally:
    # Clean up temp state_dict files
    for tmp_file in tmp_files_to_clean:
      if os.path.exists(tmp_file):
        os.remove(tmp_file)
        print(f'Cleaned up temp file: {tmp_file}')

  # Step 6: save local inventory snapshot
  print('\n=== Step 6: Saving local snapshot ===')
  save_local_snapshot(local_inventory, cluster_id)

  print('\n=== Sync complete! ===')
  print('Remember to: git add services/inventories/ && git commit && git push')


if __name__ == '__main__':
  cfg = tyro.cli(SyncServiceConfig)
  main(cfg)
