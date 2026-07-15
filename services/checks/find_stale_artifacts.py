"""Find artifacts that match on (arch_id, n, d, gbs, lr) between local and HF,
but have differing mtime_spec — i.e., artifacts that need re-uploading.

Usage:
    python -m services.find_stale_artifacts [--cluster_id tud|mpi]
"""

from dataclasses import dataclass

import pandas as pd
import tyro

from services.sync import _HF_INVENTORY_PATH, _INVENTORIES_DIR, load_hf_state
from services.utils import Inventory, take_inventory

_IDENTITY_COLS = ['arch_id', 'n', 'd', 'gbs', 'lr']


@dataclass
class Config:
  cluster_id: str = 'tud'  # one of: mpi, tud


def main(cfg: Config) -> None:
  # Load HF inventory
  hf = load_hf_state()
  if len(hf) == 0:
    print('HF inventory is empty — nothing to compare.')
    return
  hf_df = pd.DataFrame(hf.data)
  print(f'HF inventory:  {len(hf_df)} entries')

  # Take fresh local inventory
  local = take_inventory(cfg.cluster_id)
  if len(local) == 0:
    print('Local inventory is empty — nothing to compare.')
    return
  local_df = pd.DataFrame(local.data)
  print(f'Local ({cfg.cluster_id}): {len(local_df)} entries')

  # Merge on identity columns, keeping both mtime_spec values
  merged = local_df.merge(
    hf_df,
    on=_IDENTITY_COLS,
    how='inner',
    suffixes=('_local', '_hf'),
  )

  # Find rows where mtime_spec differs
  stale = merged[merged['mtime_spec_local'] != merged['mtime_spec_hf']]

  print(f'\nMatched on config:     {len(merged)} artifacts')
  print(f'Mtime differs (stale): {len(stale)} artifact(s)')
  print(f'Mtime matches (fresh): {len(merged) - len(stale)} artifact(s)')

  if len(stale) == 0:
    return

  print('\n--- Stale artifacts (need re-upload) ---')
  for _, row in stale.iterrows():
    print(f'  {row["arch_id"]:<6s}  {row["n"]:<5s}  gbs_{int(row["gbs"]):<4d}  lr_{row["lr"]}  d={row["d"]}')
    print(f'    HF mtime:    {row["mtime_spec_hf"]}')
    print(f'    local mtime: {row["mtime_spec_local"]}')
    print()


if __name__ == '__main__':
  cfg = tyro.cli(Config)
  main(cfg)
