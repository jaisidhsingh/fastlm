import torch

from src.models.construct import construct_hf_config
from src.models.to_hf import HFModelConfig, HFModelForCausalLM


def check_hf_model_init(cfg, device):
  hf_cfg = construct_hf_config
  model = HFModelForCausalLM(hf_cfg)
  model = model.to(dtype=torch.bfloat16, device=device)
  print(model.model.count_params(non_embedding=False))


def main():
  cfg = None
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  check_hf_model_init(cfg, device)


if __name__ == '__main__':
  main()
