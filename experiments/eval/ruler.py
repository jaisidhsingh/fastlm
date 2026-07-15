import json
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import tyro
from absl import app, flags

from src.constants import *
from src.models.construct import *
from src.models.to_hf import HFModelForCausalLM, load_checkpoint_into_hf, register
from src.utils.base_utils import load_config, parse_arch_id

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
  arch_id: str
  ckpt_path: str
  n: str = '150M'
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
  hf_cfg = construct_hf_config_from_mcfg(mcfg)
  model = HFModelForCausalLM._from_config(hf_cfg)
  load_checkpoint_into_hf(model, cfg.ckpt_path)
  print(model._tied_weights_keys)
  print(getattr(model, '_dynamic_tied_weights_keys', None))
  print(model.config.tie_word_embeddings)
  model.save_pretrained(hf_model_save_folder)
  del model


def _get_results_base(cfg):
  return os.path.join('./results', cfg.arch_id)


def _get_save_prefix(cfg):
  return 'ruler'


def main(cfg):
  # lazy import lm_eval because of `register()`
  from lm_eval import evaluator
  from transformers import AutoTokenizer

  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda'
  elif torch.mps.is_available():
    device = 'mps'

  base_folder = _get_results_base(cfg)
  save_prefix = _get_save_prefix(cfg)

  # Save model and tokenizer to a temp folder
  hf_model_save_folder = os.path.dirname(cfg.ckpt_path)
  setup_model_and_save(cfg, hf_model_save_folder)

  tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, model_max_length=2048)
  tokenizer.save_pretrained(hf_model_save_folder)
  del model, tokenizer

  results = evaluator.simple_evaluate(
    model='hf',
    model_args=f'pretrained={hf_save_folder}',
    tasks=['ruler'],
    batch_size='auto',
    device=device,
  )

  output_path = os.path.join(base_folder, 'results_ruler.json')
  with open(output_path, 'w') as f:
    json.dump(results, f)

  print(f'RULER results saved to {output_path}')
  print('Evaluation complete.')


if __name__ == '__main__':
  cfg = tyro.cli(EvalConfig)
  main(cfg)
