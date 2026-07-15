import os
import sys
from dataclasses import dataclass

import torch
import tyro
from huggingface_hub import HfApi

from src.constants import HF_CKPT_DOWN_FOLDER, SCALING_LADDER


@dataclass
class DownloadConfig:
  arch_id: str = 'attn'
  n: str = '150M'
  d: str = '3.0B'
  gbs: int = 32
  lr: float = 0.001
  cluster_id: str = 'mpi'


def parse_arch_id(arch_id: str):
  """
  - pure attention corresponds to `arch_id = "attn"`
  - pure gdn corresponds to `arch_id = "gdn"`
  - hybrid with gdn:attn = x:1 corresponds to `arch_id: "gdn+attn_x-1"`
  - hybrid with gdn:attn = 1:x corresponds to `arch_id: "gdn+attn_1-x"`
  """
  split_id = arch_id.split('_')
  arch = split_id[0]
  ratio = 1
  if len(split_id) == 2:
    [r1, r2] = [int(x) for x in split_id[1].split('-')]
    if r2 == 1:  # spec follows x:1 format
      ratio = r1
    else:  # spec follows 1:x format
      assert r1 == 1, 'Hybridisation ratio specified incorrectly'
      ratio = int(-1 * r2)  # negative value of ratio handled in `Transformer._prepare_layers`
  return arch, ratio


def download_ckpt(cfg: DownloadConfig) -> None:
  path_in_repo = os.path.join(
    cfg.n, f'gbs_{cfg.gbs}', f'lr_{str(cfg.lr).replace(".", "p")}', f'ckpt_decayed_to_{cfg.d.replace(".", "p")}.pt'
  )
  dest_folder = os.path.join(HF_CKPT_DOWN_FOLDER[cfg.cluster_id], cfg.arch_id)
  os.makedirs(dest_folder, exist_ok=True)

  api = HfApi()
  repo_id = 'jaisidhsingh/OpenThesis_' + str(cfg.arch_id).replace('+', '-')
  save_path = api.hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=path_in_repo, local_dir=dest_folder)
  return save_path


def validate_hf_stored_ckpt(cfg, path):
  ckpt = torch.load(path, weights_only=True, map_location='cpu')

  err_msg = 'Found incorrect architecture in specified checkpoint!'
  d_model = ckpt['embed_tokens.weight'].shape[1]
  assert d_model == SCALING_LADDER['models'][cfg.n]['d_model'], err_msg

  arch, ratio = parse_arch_id(cfg.arch_id)

  if '+' not in arch:
    for i in range(4):
      if arch == 'attn':
        assert f'layers.{i}.token_mixer.w_qkv.weight' in ckpt, err_msg
      else:  # gdn
        assert f'layers.{i}.token_mixer.A_log' in ckpt, err_msg
    return

  if ratio > 0:
    token_mixers = arch.split('+')  # [gdn, attn]
  else:
    token_mixers = arch.split('+')
    token_mixers.reverse()  # [attn, gdn]

  for i in range(4):
    if ratio > 0:
      if (i + 1) % (ratio + 1) == 0:  # on attn
        assert f'layers.{i}.token_mixer.w_qkv.weight' in ckpt, err_msg
      else:  # on gdn
        assert f'layers.{i}.token_mixer.A_log' in ckpt, err_msg

    else:
      if i % (abs(ratio) + 1) == 0:  # on gdn
        assert f'layers.{i}.token_mixer.A_log' in ckpt, err_msg
      else:  # on attn
        assert f'layers.{i}.token_mixer.w_qkv.weight' in ckpt, err_msg


def main(cfg):
  path = download_ckpt(cfg)
  validate_hf_stored_ckpt(cfg, path)
  print(path)


if __name__ == '__main__':
  cfg = tyro.cli(DownloadConfig)
  main(cfg)
