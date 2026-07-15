"""Rebuild hf_inventory.json from files actually present on HuggingFace.

Cross-references with local inventory snapshots to recover metadata
(mtime_spec, cluster_location, etc.) that can't be derived from HF alone.

Usage:
    python -m services.rebuild_hf_inventory
"""

import os

import pandas as pd
from huggingface_hub import HfApi

from services.utils import ArtifactState, Inventory

_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_INVENTORIES_DIR = os.path.join(_SERVICES_DIR, 'inventories')
_HF_INVENTORY_PATH = os.path.join(_INVENTORIES_DIR, 'hf_inventory.json')
_HF_USERNAME = 'jaisidhsingh'


def load_local_snapshots() -> pd.DataFrame | None:
  """Load and combine all local_inventory_*.json snapshots."""
  dfs = []
  for fname in sorted(os.listdir(_INVENTORIES_DIR)):
    if not (fname.startswith('local_inventory_') and fname.endswith('.json')):
      continue
    path = os.path.join(_INVENTORIES_DIR, fname)
    inv = Inventory()
    inv.load(path)
    df = pd.DataFrame(inv.data)
    if not df.empty:
      dfs.append(df)
      print(f'  Loaded {len(df)} entries from {fname}')

  if not dfs:
    print('  No local snapshots found.')
    return None

  return pd.concat(dfs, ignore_index=True)


def discover_hf_artifacts(api: HfApi, arch_ids: list[str]) -> list[dict]:
  """Probe each known arch_id on HF and scan its files.

  Returns a list of dicts with keys: arch_id, n, gbs, lr, d.
  """
  artifacts = []

  for arch_id in sorted(arch_ids):
    repo_id = f'{_HF_USERNAME}/OpenThesis_{arch_id}'.replace('+', '-')

    try:
      files = list(api.list_repo_files(repo_id, repo_type='dataset'))
    except Exception:
      print(f'  Repo not found or inaccessible: {repo_id}  — skipping')
      continue

    if not files:
      print(f'  Repo is empty: {repo_id}  — skipping')
      continue

    # Only look at checkpoint files to avoid double-counting
    # (each artifact has both ckpt_*.pt and metrics_*.json)
    for f in files:
      fname = f.split('/')[-1]
      if not fname.startswith('ckpt_'):
        continue

      # Path format: {n}/gbs_{gbs}/lr_{lr}/ckpt_decayed_to_{d}.pt
      parts = f.split('/')
      if len(parts) != 4:
        print(f'  UNEXPECTED path in {repo_id}: {f}  — skipping')
        continue

      n = parts[0]
      gbs = int(parts[1].replace('gbs_', ''))
      lr = float(parts[2].replace('lr_', '').replace('p', '.'))
      d = fname.replace('ckpt_decayed_to_', '').replace('.pt', '').replace('p', '.')

      artifacts.append(
        {
          'arch_id': arch_id,
          'n': n,
          'gbs': gbs,
          'lr': lr,
          'd': d,
        }
      )
      print(f'    {arch_id} / {n} / gbs_{gbs} / lr_{lr} / {d}')

    print(f'  → {len([a for a in artifacts if a["arch_id"] == arch_id])} artifacts in {repo_id}')

  return artifacts


def main() -> None:
  print('=== Step 1: Loading local snapshots ===')
  local_df = load_local_snapshots()
  if local_df is None:
    print('Cannot proceed without local snapshots.')
    return

  arch_ids = sorted(local_df['arch_id'].unique())
  print(f'  Found arch_ids: {arch_ids}')

  api = HfApi()

  print('\n=== Step 2: Scanning HF repos ===')
  hf_artifacts = discover_hf_artifacts(api, arch_ids)
  if not hf_artifacts:
    print('No artifacts found on HF — nothing to rebuild.')
    return

  hf_df = pd.DataFrame(hf_artifacts)
  print(f'\n  Total HF artifacts found: {len(hf_df)}')

  print('\n=== Step 3: Matching HF artifacts to local entries ===')
  match_keys = ['arch_id', 'n', 'gbs', 'lr', 'd']

  merged = hf_df.merge(local_df, on=match_keys, how='left')

  unmatched = merged[merged['mtime_spec'].isna()]
  if len(unmatched) > 0:
    print(f'\n  WARNING: {len(unmatched)} HF artifacts could not be matched to local entries:')
    for _, row in unmatched.iterrows():
      print(f'    {row["arch_id"]} / {row["n"]} / gbs_{row["gbs"]} / lr_{row["lr"]} / {row["d"]}')
    print('  These will be skipped.\n')

  matched = merged.dropna(subset=['mtime_spec'])
  print(f'  Matched {len(matched)} of {len(hf_df)} artifacts.')

  if matched.empty:
    print('No artifacts could be matched — aborting.')
    return

  print('\n=== Step 4: Building hf_inventory.json ===')
  hf_inventory = Inventory()
  for _, row in matched.iterrows():
    state = ArtifactState(
      arch_id=str(row['arch_id']),
      n=str(row['n']),
      d=str(row['d']),
      gbs=int(row['gbs']),
      lr=float(row['lr']),
      checkpoint_filename=str(row['checkpoint_filename']),
      metrics_filename=str(row['metrics_filename']),
      cluster_location=str(row['cluster_location']),
      mtime_spec=str(row['mtime_spec']),
    )
    hf_inventory.push(state)

  os.makedirs(_INVENTORIES_DIR, exist_ok=True)
  hf_inventory.save(_HF_INVENTORY_PATH, format='json')
  print(f'  Saved hf_inventory.json with {len(hf_inventory)} entries.')


if __name__ == '__main__':
  main()
