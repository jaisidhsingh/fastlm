"""Find local artifacts where checkpoint mtime differs from metrics mtime.

Usage:
    python -m services.find_mismatched_mtimes --cluster_id tud
"""

from dataclasses import dataclass
from pathlib import Path

import tyro

from services.utils import get_mtime_str
from src.constants import SCALING_RESULTS_FOLDER


@dataclass
class Config:
  cluster_id: str = 'tud'  # one of: mpi, tud


def main(cfg: Config) -> None:
  _cluster_key = 'capella' if cfg.cluster_id == 'tud' else 'mpi'
  base_root = SCALING_RESULTS_FOLDER[_cluster_key]
  root = Path(base_root)

  mismatched = []
  total = 0

  for path in root.rglob('*.json'):
    if not path.is_file():
      continue

    path_str = str(path)

    if 'metrics_decayed_to_' not in path_str:
      continue

    total += 1

    metrics_mtime = get_mtime_str(path_str)
    ckpt_path = path_str.replace('.json', '.pt').replace('metrics_', 'ckpt_')
    ckpt_mtime = get_mtime_str(ckpt_path)

    # Compare at minute granularity (ignore seconds)
    if ckpt_mtime[:-3] != metrics_mtime[:-3]:
      root_removed = path_str.split(base_root)[-1].lstrip('/')
      mismatched.append(
        {
          'path': root_removed,
          'ckpt_mtime': ckpt_mtime,
          'metrics_mtime': metrics_mtime,
        }
      )

  print(f'Scanned {total} artifacts in {base_root}')
  print(f'Mismatched mtimes: {len(mismatched)}')

  if not mismatched:
    return

  print()
  for m in mismatched:
    print(f'  {m["path"]}')
    print(f'    ckpt:    {m["ckpt_mtime"]}')
    print(f'    metrics: {m["metrics_mtime"]}')
    print()


if __name__ == '__main__':
  cfg = tyro.cli(Config)
  main(cfg)
