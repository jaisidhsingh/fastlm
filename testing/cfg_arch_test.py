import torch
from absl import app, flags

from src.models import construct_model
from src.utils import load_config

flags.DEFINE_string(
  'config',
  '/lustre/home/jsingh/projects/fastlm/execs/gdn/20M/cfg-main_gbs-16_lr-all_parallel.yaml',
  'Path to config.yaml file.',
)
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS


def main(argv):
  cfg, _ = load_config(FLAGS.config)
  model, model_config = construct_model(cfg)
  print(model)


if __name__ == '__main__':
  app.run(main)
