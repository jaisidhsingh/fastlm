import csv
import json
import os
import random
import time

import torch
import yaml
from absl import app, flags

from src.constants import SCALING_RESULTS_FOLDER
from src.models.construct import construct_hf_config
from src.models.to_hf import HFModelForCausalLM, load_checkpoint_into_hf, register
from src.utils.base_utils import load_config, print_master

register()

flags.DEFINE_string('config', 'src/config/cfg_eval.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS


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


def setup_model(cfg):
  model = construct_model(cfg)
  return model


def _get_results_base(cfg):
  arch_id = cfg.arch_id
  gbs = cfg.global_batch_size
  return os.path.join(SCALING_RESULTS_FOLDER, arch_id, 'gbs_wise_results', f'gbs_{gbs}')


def _get_save_prefix(cfg):
  lr = cfg.lr
  return f'eval_lr-{str(lr).replace(".", "p")}'


def _get_eval_bundle_dir():
  """Locate the dclm-core/eval_bundle directory relative to this file."""
  # experiments/eval.py -> experiments -> fastlm -> code -> dclm-core/eval_bundle
  code_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  return os.path.join(code_root, 'dclm-core', 'eval_bundle')


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
  model.max_seq_len = cfg.seq_len  # for truncation logic in core_eval
  model = model.to(device)
  model.eval()

  hf_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
  tokenizer = HFTokenizerAdapter(hf_tokenizer)

  eval_bundle_dir = _get_eval_bundle_dir()
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

    print_master(f'Evaluating: {label} ({task_meta["num_fewshot"]}-shot, type: {task_meta["task_type"]})...')
    accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
    results[label] = accuracy
    random_baseline = random_baselines[label]
    centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
    centered_results[label] = centered_result
    elapsed = time.time() - start_time
    print_master(f'  accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s')

  core_metric = sum(centered_results.values()) / len(centered_results)
  print_master(f'DCLM CORE metric: {core_metric:.4f}')

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
  print_master(f'DCLM CORE results saved to {output_path}')


def main(argv):
  cfg_path = FLAGS.config
  cfg, sweep_size = load_config(cfg_path)

  print_master(f'Config loaded from {cfg_path}')

  eval_dclm_core(cfg)
  print_master('Evaluation complete.')


if __name__ == '__main__':
  app.run(main)
