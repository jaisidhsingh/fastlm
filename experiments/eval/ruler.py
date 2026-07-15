import json
import os

import torch
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


def setup_model_and_save(cfg):
  hf_cfg = construct_hf_config(cfg)
  model = HFModelForCausalLM._from_config(hf_cfg)
  model = load_checkpoint_into_hf(model, cfg.checkpoint_path)
  model.save_pretrained(cfg.hf_model_save_folder)
  del model


def _get_results_base(cfg):
  arch_id = cfg.arch_id
  gbs = cfg.global_batch_size
  return os.path.join(SCALING_RESULTS_FOLDER, arch_id, 'gbs_wise_results', f'gbs_{gbs}')


def _get_save_prefix(cfg):
  lr = cfg.lr
  return f'eval_lr-{str(lr).replace(".", "p")}'


def main(argv):
  # lazy import lm_eval because of `register()`
  from lm_eval import evaluator
  from transformers import AutoTokenizer

  cfg_path = FLAGS.config
  cfg, sweep_size = load_config(cfg_path)

  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda'
  elif torch.mps.is_available():
    device = 'mps'

  base_folder = _get_results_base(cfg)
  save_prefix = _get_save_prefix(cfg)

  # Save model and tokenizer to a temp folder
  setup_model_and_save(cfg)
  tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
  tokenizer.save_pretrained(cfg.hf_model_save_folder)
  del model, tokenizer

  results = evaluator.simple_evaluate(
    model='hf',
    model_args=f'pretrained={eval_ckpt_folder}',
    tasks=['ruler'],
    batch_size='auto',
    device=device,
  )

  output_path = os.path.join(base_folder, f'{save_prefix}_{"__".join(tasks)}.json')
  with open(output_path, 'w') as f:
    json.dump(results, f)

  print_master(f'RULER results saved to {output_path}')
  print_master('Evaluation complete.')


if __name__ == '__main__':
  app.run(main)
