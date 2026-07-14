import os
from collections import defaultdict
from dataclasses import dataclass

import torch
import tyro
from huggingface_hub import CommitOperationAdd, HfApi

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
_HF_USERNAME = 'jaisidhsingh'


@dataclass
class SyncServiceConfig:
  cluster_id: str  # one of: mpi, tud
  batch_size: int = 50  # files per commit to stay under HF's 128 commits/hour limit


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


def ensure_hf_repo_exists(api: HfApi, arch_id: str) -> str:
  """Make sure the HF dataset repo for `arch_id` exists; create it if not.

  Returns the repo_id.
  """
  repo_id = f'{_HF_USERNAME}/OpenThesis_{arch_id}'
  repo_id = repo_id.replace('+', '-')
  print(f'Ensuring HF repo exists: {repo_id} ...')
  api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)
  return repo_id


def extract_model_state_dict_to_tmpfile(
  checkpoint_path: str, cluster_id: str, arch_id: str, n: str, gbs: int, lr: float
) -> str:
  """Load a full training checkpoint, extract only `state_dict`, save to tmp.

  Returns the path to the temporary file.
  """
  _cluster_key = 'capella' if cluster_id == 'tud' else 'mpi'
  tmp_dir = os.path.join(
    TMP_FOLDER_FOR_UPLOAD[_cluster_key], f'{arch_id}_n-{n}_gbs-{gbs}_lr-{str(lr).replace(".", "p")}'
  )
  os.makedirs(tmp_dir, exist_ok=True)

  print(f'  Extracting state_dict from {checkpoint_path} ...')
  ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
  state_dict = ckpt['state_dict']

  fname = os.path.basename(checkpoint_path)
  tmpfile_path = os.path.join(tmp_dir, fname)
  torch.save(state_dict, tmpfile_path)
  print(f'  Saved state_dict to {tmpfile_path}')
  return tmpfile_path


def upload_batch_to_hf(api: HfApi, batch_items: list[tuple[str, str, str]], batch_label: str) -> None:
  """Upload a batch of files grouped by repo_id using a single commit per repo.

  Args:
      api: HuggingFace API client.
      batch_items: List of (repo_id, src_path, dest_path) tuples.
      batch_label: Human-readable label for the commit message (e.g. "1–50").
  """
  by_repo: dict[str, list[tuple[str, str]]] = defaultdict(list)
  for repo_id, src_path, dest_path in batch_items:
    by_repo[repo_id].append((src_path, dest_path))

  for repo_id, files in by_repo.items():
    operations = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=src) for src, dest in files]
    print(f'  Creating commit on {repo_id} with {len(files)} file(s) ...')
    api.create_commit(
      repo_id=repo_id,
      repo_type='dataset',
      operations=operations,
      commit_message=f'Upload batch {batch_label} ({len(files)} files)',
    )
    print(f'  Commit created.')


def append_to_hf_inventory(state: ArtifactState) -> None:
  """Append a single successfully-uploaded artifact to hf_inventory.json."""
  hf_inventory = Inventory()
  if os.path.exists(_HF_INVENTORY_PATH):
    hf_inventory.load(_HF_INVENTORY_PATH)
  hf_inventory.push(state)
  os.makedirs(_INVENTORIES_DIR, exist_ok=True)
  hf_inventory.save(_HF_INVENTORY_PATH, format='json')


def save_local_snapshot(local_inventory: Inventory, cluster_id: str) -> None:
  """Save the full local inventory to a per-cluster snapshot file."""
  snapshot_path = os.path.join(_INVENTORIES_DIR, f'local_inventory_{cluster_id}.json')
  os.makedirs(_INVENTORIES_DIR, exist_ok=True)
  local_inventory.save(snapshot_path, format='json')
  print(f'Saved local snapshot ({len(local_inventory)} entries) to {snapshot_path}')


def main(cfg: SyncServiceConfig) -> None:
  assert cfg.cluster_id in ['mpi', 'tud'], 'Unsupported value of `--cluster_id` provided.'
  cluster_id = cfg.cluster_id
  batch_size = cfg.batch_size

  # Step 1: take inventory of the cluster
  local_inventory = take_inventory(cluster_id)

  # Step 2: load canonical HF state
  hf_inventory = load_hf_state()

  # Step 3: diff
  changes = difference(local_inventory, hf_inventory)
  if changes is None:
    print('No new or modified artifacts found — nothing to upload.')
    return

  save_local_snapshot(local_inventory, cluster_id)

  num_changes = len(changes)
  print(f'Found {num_changes} artifact(s) to upload.')
  return

  # Step 4: get source/destination paths for every changed artifact
  upload_paths = get_checkpoints_from_changes(changes, cluster_id)
  api = HfApi()

  # Step 5: upload in batches
  num_batches = (num_changes + batch_size - 1) // batch_size
  print(f'Uploading in {num_batches} batch(es) of up to {batch_size} files each ...')

  for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = min(start + batch_size, num_changes)
    batch_label = f'{start + 1}–{end}'
    tmp_files_to_clean: list[str] = []

    print(f'\n=== Batch {batch_idx + 1}/{num_batches} (artifacts {batch_label}) ===')

    try:
      batch_items: list[tuple[str, str, str]] = []

      for i in range(start, end):
        arch_id = str(changes.iloc[i]['arch_id'])
        n = str(changes.iloc[i]['n'])
        gbs = int(changes.iloc[i]['gbs'])
        lr = float(changes.iloc[i]['lr'])
        ckpt_src = upload_paths['checkpoints_src'][i]
        metrics_src = upload_paths['metrics_src'][i]
        ckpt_dest = upload_paths['checkpoints_dest'][i]
        metrics_dest = upload_paths['metrics_dest'][i]

        print(
          f'  Artifact {i + 1}/{num_changes}: {arch_id} / '
          f'{changes.iloc[i]["n"]} / gbs_{changes.iloc[i]["gbs"]} / '
          f'lr_{changes.iloc[i]["lr"]} / {changes.iloc[i]["d"]}'
        )

        repo_id = f'{_HF_USERNAME}/OpenThesis_{arch_id}'
        repo_id = repo_id.replace('+', '-')

        # Extract state_dict to temp file
        if not os.path.exists(ckpt_src):
          raise FileNotFoundError(f'Checkpoint not found: {ckpt_src}')
        tmp_ckpt = extract_model_state_dict_to_tmpfile(ckpt_src, cluster_id, arch_id, n, gbs, lr)
        tmp_files_to_clean.append(tmp_ckpt)

        # Add checkpoint to batch
        batch_items.append((repo_id, tmp_ckpt, ckpt_dest))

        # Add metrics to batch
        if not os.path.exists(metrics_src):
          raise FileNotFoundError(f'Metrics file not found: {metrics_src}')
        batch_items.append((repo_id, metrics_src, metrics_dest))

      # Upload the entire batch in a single commit per repo
      upload_batch_to_hf(api, batch_items, batch_label)

      # Persist all artifacts in this batch to hf_inventory
      for i in range(start, end):
        state = ArtifactState(
          arch_id=str(changes.iloc[i]['arch_id']),
          n=str(changes.iloc[i]['n']),
          d=str(changes.iloc[i]['d']),
          gbs=int(str(changes.iloc[i]['gbs'])),
          lr=float(str(changes.iloc[i]['lr'])),
          checkpoint_filename=str(changes.iloc[i]['checkpoint_filename']),
          metrics_filename=str(changes.iloc[i]['metrics_filename']),
          cluster_location=str(changes.iloc[i]['cluster_location']),
          mtime_spec=str(changes.iloc[i]['mtime_spec']),
        )
        append_to_hf_inventory(state)

      print(f'Batch {batch_idx + 1} uploaded and recorded successfully.')

    except Exception as e:
      print(f'\n!!! Upload failed in batch {batch_idx + 1}: {e}')
      print('Artifacts uploaded in this batch were saved to HF inventory — remaining will be retried on next sync.')
      raise

    finally:
      # Clean up temp state_dict files for this batch
      for tmp_file in tmp_files_to_clean:
        if os.path.exists(tmp_file):
          os.remove(tmp_file)
          print(f'  Cleaned up temp file: {tmp_file}')

  # Step 6: save local inventory snapshot
  save_local_snapshot(local_inventory, cluster_id)
  print('Sync complete!')


if __name__ == '__main__':
  cfg = tyro.cli(SyncServiceConfig)
  main(cfg)
