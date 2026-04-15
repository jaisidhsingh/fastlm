import os
from dataclasses import asdict, dataclass
from pprint import pprint

import dacite
import yaml


@dataclass
class ModelConfig:
  d_model: int = 32
  n_layers: int = 6


@dataclass
class OptimizerConfig:
  name: str = 'muon'
  momentum: float = 0.9
  ns_steps: int = 5


@dataclass
class Config:
  model: ModelConfig
  opt: OptimizerConfig


def save_config(cfg):
  pprint(cfg)
  with open('./testing/testing_results/test_config.yaml', 'w') as f:
    yaml.dump(asdict(cfg), f)


def load_config(path='./testing/testing_results/test_config.yaml'):
  with open(path) as f:
    args = yaml.safe_load(f)
  cfg = dacite.from_dict(Config, args)
  pprint(cfg)


def main():
  cfg = Config(model=ModelConfig(), opt=OptimizerConfig())
  os.makedirs('./testing/testing_results', exist_ok=True)
  save_config(cfg)
  load_config()


if __name__ == '__main__':
  main()
