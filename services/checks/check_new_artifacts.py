import os
from dataclasses import dataclass

import tyro

from services.utils import (
  Inventory,
  difference,
  take_inventory,
)

_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_INVENTORIES_DIR = os.path.join(_SERVICES_DIR, 'inventories')
_HF_INVENTORY_PATH = os.path.join(_INVENTORIES_DIR, 'hf_inventory.json')


@dataclass
class SyncServiceConfig:
  cluster_id: str  # one of: mpi, tud


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


def main(cfg: SyncServiceConfig) -> None:
  assert cfg.cluster_id in ['mpi', 'tud'], 'Unsupported value of `--cluster_id` provided.'
  cluster_id = cfg.cluster_id

  # Step 1: take inventory of the cluster
  local_inventory = take_inventory(cluster_id)

  # Step 2: load canonical HF state
  hf_inventory = load_hf_state()

  # Step 3: diff
  changes = difference(local_inventory, hf_inventory)
  if changes is None:
    print('No new or modified artifacts found — nothing to upload.')
    return

  num_changes = len(changes)
  print(f'Found {num_changes} artifact(s) to upload.')


if __name__ == '__main__':
  cfg = tyro.cli(SyncServiceConfig)
  main(cfg)
