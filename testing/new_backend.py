import tyro
from types import SimpleNamespace
import torch
import yaml
from transformers import AutoModelForCausalLM
from dataclasses import dataclass
from src.models import config_builder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
MODEL_DTYPE = torch.bfloat16


@dataclass
class TestingConfig:
  batch_size: int = 2
  seq_len: int = 128
  dim: int = 32
  test_config: str = "/home/jsingh/projects/fastlm/src/config/int/attn_300M.yaml"


def test_new_backend_model_init(cfg): 
  model_config = config_builder(cfg)
  model = AutoModelForCausalLM.from_config(model_config)
  model = model.to(dtype=MODEL_DTYPE, device=DEVICE)
  print(model)
  return model, model_config


def main(test_cfg):
  with open(test_cfg.test_config) as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
  
  test_new_backend_model_init(cfg)


if __name__ == "__main__":
  test_cfg = tyro.cli(TestingConfig, default=TestingConfig())
  main(test_cfg)

