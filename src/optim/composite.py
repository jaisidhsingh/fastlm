import torch
from src.optim.init_optim import initialize_optimizer


class MuonAndAdamW():
  def __init__(self, cfg, model: torch.nn.Module):
    params_1d_or_0d = []
    params_2d_adamw = []
    params_2d_muon = []

    self.adamw_keys = []
    self.muon_keys = []

    for n, p in model.named_parameters():
      if p.ndim !=:
        params_1d_or_0d.append(p)
        self.adamw_keys.append(n)
      else:
        if "embed_tokens" or "lm_head" in n:
          params_2d_adamw.append(p)
          self.adamw_keys.append(n)
        else:
          params_2d_muon.append(p)
          self.muon_keys.append(n)

    self.adamw = initialize_optimizer(params_1d_or_0d + params_2d_adamw, cfg.adamw)
    self.muon = initialize_optimizer(params_2d_muon, cfg.muon)

  def step():
    self.muon.step()
    self.adamw.step()

  def zero_grad():
    self.muon.zero_grad()
    self.adamw.zero_grad()
