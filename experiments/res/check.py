import os
import json
import tyro
from dataclasses import dataclass

from src.constants import *


@dataclass
class Config:
  arch_id: str = 'attn'
  n: str = '150M'
  d: str = '15.0B'
  gbs: int = 64
  lr: float = 0.001
  cluster_id: str = 'mpi'


def main(cfg: Config):
  path = os.path.join(
    HF_METRIC_FOLDER[cfg.cluster_id],
    cfg.arch_id, cfg.n,
    f"gbs_{cfg.gbs}",
    f"lr_{str(cfg.lr).replace('.', 'p')}",
    f"metrics_decayed_to_{cfg.d.replace('.', 'p')}.json"
  )
  with open(path) as f:
    data = json.load(f)
  
  print(cfg.arch_id)
  print(data['valid/loss'][-1])


if __name__ == "__main__":
  cfg = tyro.cli(Config, default=Config())
  main(cfg)

