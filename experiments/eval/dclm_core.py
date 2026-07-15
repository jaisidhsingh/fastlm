import csv
import json
import os
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import tyro
import yaml
from absl import app, flags

from src.constants import *
from src.models.construct import *
from src.models.to_hf import register

# from src.utils.base_utils import load_config

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


class HFTokenizerAdapter:
  """
  Wraps an HF AutoTokenizer so it works with core_eval.py functions.
  core_eval expects:
    tokenizer(prompts, prepend=<bos_token_id>) -> list[list[int]]
    tokenizer.get_bos_token_id() -> int
  """

  def __init__(self, hf_tokenizer):
    self._tok = hf_tokenizer

  def get_bos_token_id(self):
    return self._tok.bos_token_id

  def __call__(self, prompts, prepend=None):
    if isinstance(prompts, str):
      prompts = [prompts]
    result = []
    for prompt in prompts:
      ids = []
      if prepend is not None:
        ids.append(prepend if isinstance(prepend, int) else prepend[0])
      ids.extend(self._tok.encode(prompt, add_special_tokens=False))
      result.append(ids)
    return result


def setup_model_config(cfg):
  mcfg_from_ladder = SCALING_LADDER['models'][cfg.n]
  model_cfg = SimpleNamespace(**DEFAULT_MODEL_CONFIG)
  model_cfg.arch_id = cfg.arch_id
  model_cfg.d_model = mcfg_from_ladder['d_model']
  model_cfg.n_layers = mcfg_from_ladder['n_layers']
  model_cfg.n_heads = mcfg_from_ladder['n_heads']
  model_cfg.seq_len = 2048
  return model_cfg


def setup_model(cfg):
  model_cfg = setup_model_config(cfg)
  model, model_cfg = construct_model(model_cfg)
  ckpt = torch.load(cfg.ckpt_path, weights_only=True, map_location='cpu')
  model.load_state_dict(ckpt)
  model.max_seq_len = model_cfg.seq_len
  return model


def _get_results_base(cfg):
  return os.path.join('./results', cfg.arch_id)


def _get_save_prefix(cfg):
  return 'results'


def _get_eval_bundle_dir(cfg):
  if cfg.cluster_id == 'mpi':
    return '/fast/jsingh/data/nanochat-dclm-core/eval_bundle'
  elif cfg.cluster_id in ['alpha', 'capella']:
    return '/data/horse/ws/jasi149i-fastlm/data/nanochat-dclm-core/eval_bundle'
  else:
    raise ValueError('Unsupported value found for --cluster_id')


def eval_dclm_core(cfg):
  from transformers import AutoTokenizer

  from src.eval.core_eval import evaluate_task

  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda'
  elif torch.mps.is_available():
    device = 'mps'

  base_folder = _get_results_base(cfg)
  save_prefix = _get_save_prefix(cfg)

  model = setup_model(cfg)
  model = model.to(device=device, dtype=torch.bfloat16)
  model.eval()
  print(model)

  hf_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
  tokenizer = HFTokenizerAdapter(hf_tokenizer)

  eval_bundle_dir = _get_eval_bundle_dir(cfg)
  config_path = os.path.join(eval_bundle_dir, 'core.yaml')
  data_base_path = os.path.join(eval_bundle_dir, 'eval_data')
  eval_meta_path = os.path.join(eval_bundle_dir, 'eval_meta_data.csv')

  with open(config_path, 'r') as f:
    dclm_config = yaml.safe_load(f)
  tasks = dclm_config['icl_tasks']

  # Load random baselines
  random_baselines = {}
  with open(eval_meta_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      random_baselines[row['Eval Task']] = float(row['Random baseline'])

  # Evaluate each task
  results = {}
  centered_results = {}
  for task in tasks:
    start_time = time.time()
    label = task['label']
    task_meta = {
      'task_type': task['icl_task_type'],
      'dataset_uri': task['dataset_uri'],
      'num_fewshot': task['num_fewshot'][0],
      'continuation_delimiter': task.get('continuation_delimiter', ' '),
    }

    data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
    with open(data_path, 'r') as f:
      data = [json.loads(line.strip()) for line in f]

    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(data)

    print(f'Evaluating: {label} ({task_meta["num_fewshot"]}-shot, type: {task_meta["task_type"]})...')
    accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
    results[label] = accuracy
    random_baseline = random_baselines[label]
    centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
    centered_results[label] = centered_result
    elapsed = time.time() - start_time
    print(f'  accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s')

  core_metric = sum(centered_results.values()) / len(centered_results)
  print(f'DCLM CORE metric: {core_metric:.4f}')

  # Save results
  output = {
    'results': results,
    'centered_results': centered_results,
    'core_metric': core_metric,
  }
  output_path = os.path.join(base_folder, f'{save_prefix}_dclm_core.json')
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
  print(f'DCLM CORE results saved to {output_path}')


if __name__ == '__main__':
  cfg = tyro.cli(EvalConfig)
  eval_dclm_core(cfg)
