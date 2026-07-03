import torch
from absl import app, flags
from lm_eval import evaluator

import src.utils as utils

flags.DEFINE_string('config', 'eval_config.yaml', 'Path to the eval config.')
flags.DEFINE_string('model', 'path/to/my/model', 'Path to your model folder.')
flags.DEFINE_string('tasks', 'hellaswag,piqa,lambada', 'Comma-separated benchmarks to eval.')
flags.DEFINE_string(
  'array_seq',
  'sequential',
  'If `parallel`: split the tasks up into parallel job-arrays, else `sequential`: iterate over tasks in on job-array.',
)
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS


def make_custom_ckpt_hf(cfg, model):
  ckpt = torch.load(cfg.ckpt_path, weights_only=False, map_location='cpu')
  model.model.load_state_dict(ckpt['state_dict'])
  model.save_pretrained(cfg.eval_intermediate_model_folder)
  return model


def main(argv):
  cfg = utils.load_config(FLAGS.config)
  model = None
  model = make_custom_ckpt_hf(cfg, model)
  results = evaluator.simple_evaluate(
    model='hf', model_args=cfg.eval_intermediate_model_folder, tasks=FLAGS.tasks.split(',')
  )
  print(results)


if __name__ == '__main__':
  app.run(main)
