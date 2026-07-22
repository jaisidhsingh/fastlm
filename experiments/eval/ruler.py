import json
import os
import yaml
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import tyro
from absl import app, flags

from src.constants import *
from src.models.construct import *
from src.models.to_hf import HFModelForCausalLM, load_checkpoint_into_hf, register
from src.utils.base_utils import load_config, parse_arch_id, rm_rf_folder

register()

DEFAULT_MODEL_CONFIG = {
  'arch_id': 'attn',
  'param_scale_id': '20M',
  'model': 'transformer',
  'd_model': 256,
  'mlp_class': 'glu',
  'dtype': 'bfloat16',
  'expand': 3.0,
  'n_layers': 8,
  'n_heads': 4,
  'rms_norm': True,
  'tie_embeddings': True,
  'torch_compile': True,
  'token_mixer': 'attn',
  'hybrid_mixer_ratio': 1,
  'layer_norm_scaling': False,
  'residual_connection': 'add',
  'attn_gate': True,
  'attn_qk_norm': True,
  'gdn_conv_size': 4,
  'gdn_gate': True,
  'gdn_neg_eigval': True,
  'use_flex_attention': True,
  'vocab_size': 50304,
  'intra_doc_masking': False,
}


@dataclass
class EvalConfig:
  config: str | None = None
  ckpt_path: str | None = None
  job_idx: int | None = None
  job_cluster: int | None = None
  arch_id: str | None = None
  n: str = '150M'
  d: str = '3.0B'
  gbs: int = 32
  lr: float = 0.001
  cluster_id: str = 'mpi'
  tokenizer_path: str = '/fast/jsingh/saved_tokenizers/better-gpt2/'


def setup_model_config(cfg):
  mcfg_from_ladder = SCALING_LADDER['models'][cfg.n]
  arch, ratio = parse_arch_id(cfg.arch_id)
  model_cfg = SimpleNamespace(**DEFAULT_MODEL_CONFIG)
  model_cfg.token_mixer = arch
  model_cfg.hybrid_mixer_ratio = ratio
  model_cfg.param_scale_id = cfg.n
  model_cfg.arch_id = cfg.arch_id
  model_cfg.d_model = mcfg_from_ladder['d_model']
  model_cfg.n_layers = mcfg_from_ladder['n_layers']
  model_cfg.n_heads = mcfg_from_ladder['n_heads']
  model_cfg.seq_len = 2048
  return model_cfg


def setup_model_and_save(cfg, hf_model_save_folder):
  mcfg = setup_model_config(cfg)
  hf_cfg = construct_hf_config(mcfg)
  model = HFModelForCausalLM._from_config(hf_cfg)
  print(model)
  load_checkpoint_into_hf(model, cfg.ckpt_path)
  model.save_pretrained(hf_model_save_folder)
  del model


def get_save_path(cfg):
  folder = os.path.join(
    EVAL_RESULT_PATHS[cfg.cluster_id],
    cfg.arch_id,
    cfg.n,
    f'gbs_{cfg.gbs}__lr_{str(cfg.lr).replace(".", "p")}',
  )
  os.makedirs(folder, exist_ok=True)
  return os.path.join(folder, f'ruler__d-{cfg.d.replace(".", "p")}.json')


def parse_input(cfg):
  if cfg.config is None:
    assert cfg.arch_id is not None, 'Something must be given to run the eval.'
  if cfg.config is not None:
    assert cfg.job_idx is not None, 'job_idx needed if config is not None.'
    with open(cfg.config, 'r') as f:
      config_dict = yaml.safe_load(f)
    assert isinstance(config_dict, dict), "What"
    cfg.arch_id = config_dict['arch_id']
    cfg.n = config_dict['n']
    cfg.d = config_dict['d'][int(cfg.job_idx)]
    cfg.gbs = config_dict['gbs']
    cfg.lr = config_dict['lr']


@torch.inference_mode()
def main(cfg):
  # lazy import lm_eval because of `register()`
  from lm_eval import evaluator
  from lm_eval.tasks import TaskManager
  from transformers import AutoTokenizer

  # incl_folder = LM_EVAL_INCLUDE_PATHS[cfg.cluster_id]
  # task_manager = TaskManager(include_path=incl_folder)

  parse_input(cfg)
  assert os.path.exists(cfg.ckpt_path), 'Provided checkpoint path does not exist!'

  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda'
  elif torch.mps.is_available():
    device = 'mps'

  hf_model_save_folder = os.path.join(
    HF_TMP_SAVE_MODEL_FOLDER[cfg.cluster_id], f'{cfg.arch_id}_n-{cfg.n}_d-{cfg.d.replace(".", "p")}_gbs-{cfg.gbs}'
  )
  os.makedirs(hf_model_save_folder, exist_ok=True)
  setup_model_and_save(cfg, hf_model_save_folder)

  tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, model_max_length=2048)
  tokenizer.save_pretrained(hf_model_save_folder)
  del tokenizer

  results = {}
  results = evaluator.simple_evaluate(
    model='hf',
    model_args=f'pretrained={hf_model_save_folder}',
    tasks=['ruler'],
    batch_size='auto',
    device=device,
  )

  output_path = get_save_path(cfg)
  res2save = {k: str(v) for k, v in results['results'].items()}
  with open(output_path, 'w') as f:
    json.dump(res2save, f)

  print(f'RULER results saved to {output_path}')
  print('Evaluation complete.')

  rm_rf_folder(hf_model_save_folder)


if __name__ == '__main__':
  cfg = tyro.cli(EvalConfig)
  main(cfg)
