import json
import os
import typing as tp
from copy import deepcopy
from types import SimpleNamespace

import yaml

from src.constants import DEFAULT_CONFIG, SCALING_LADDER


class ExperimentManager:
  def __init__(self, parameter_scale_id: str, experiment_folder: str):
    self.parameter_scale_id = parameter_scale_id
    self.experiment_folder = experiment_folder
    self.info = SCALING_LADDER

    self.max_tokens_billions = SCALING_LADDER['models'][parameter_scale_id]['max_tokens_billions']
    self.num_token_budgets = SCALING_LADDER['models'][parameter_scale_id]['num_token_budgets']

    self.token_budget_ids = list(SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_grid'].keys())
    self.token_budget_map = {x: float(x[:-1]) * 1e9 for x in self.token_budget_ids}
    self.gbs_to_token_map = SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_runs']

    self.warmup_steps = SCALING_LADDER['warmup_steps']
    self.cooldown_steps = SCALING_LADDER['cooldown_steps']
    self.seq_len = SCALING_LADDER['seq_len']
    self.throughput_analysis_steps = SCALING_LADDER['throughput_analysis_steps']

  def get_steps_per_gbs(self, dp: tp.Union[int, tp.Dict]) -> dict | None:
    results = {}
    folder = os.path.join(self.experiment_folder, 'gbs_wise_results')
    os.makedirs(folder, exist_ok=True)

    for gbs in self.info['batch_sizes']:
      gbs_folder = os.path.join(folder, f'gbs_{gbs}')
      os.makedirs(gbs_folder, exist_ok=True)
      tp_result_path = os.path.join(gbs_folder, 'throughput_analysis.json')

      if os.path.exists(tp_result_path) and results is not None:
        with open(tp_result_path, 'r') as f:
          analysis = json.load(f)
          w = dp if isinstance(dp, int) else dp[gbs]
          results[gbs] = int(
            analysis['throughput_metrics']['tokens_per_sec']['mean']
            * w
            * analysis['throughput_metrics']['step_time']['mean']
          )

      elif not os.path.exists(os.path.join(gbs_folder, 'throughput_analysis_configs')):
        tp_config_paths = self.make_throughput_analysis_configs(gbs)
        print(f'You must execute `experiments/throughput_analysis.py` for {self.parameter_scale_id} and GBS={bs}')

    return results if len(results) > 0 else None

  def check_warmup_cooldown(self, dp: tp.Union[int, tp.Dict]) -> bool:
    steps_per_gbs = self.get_steps_per_gbs(dp)
    all_good = True
    for gbs, steps in steps_per_gbs.items():
      if not (self.warmup_steps + self.cooldown_steps < steps):
        print(f'Warmup (2000) + cooldown (4000) steps not accomodated in for {self.parameter_scale_id} & GBS={gbs}')
        print(f'Total steps = {steps}')
        all_good = False
    return all_good

  def check_throughputs(self, dp: tp.Union[int, tp.Dict]) -> dict | None:
    results = {}
    folder = os.path.join(self.experiment_folder, 'gbs_wise_results')
    os.makedirs(folder, exist_ok=True)

    for gbs in self.info['batch_sizes']:
      gbs_folder = os.path.join(folder, f'gbs_{gbs}')
      os.makedirs(gbs_folder, exist_ok=True)
      tp_result_path = os.path.join(gbs_folder, 'throughput_analysis.json')

      if os.path.exists(tp_result_path) and results is not None:
        with open(tp_result_path, 'r') as f:
          analysis = json.load(f)
          w = dp if isinstance(dp, int) else dp[gbs]
          results[gbs] = analysis['throughput_metrics']['tokens_per_sec']['mean'] * w

      elif not os.path.exists(os.path.join(gbs_folder, 'throughput_analysis_configs')):
        tp_config_paths = self.make_throughput_analysis_configs(gbs)
        print(f'You must execute `experiments/throughput_analysis.py` for {self.parameter_scale_id} and GBS={bs}')

    return results if len(results) > 0 else None


def display_june_scaling_plan():
  lrs = SCALING_LADDER['learning_rates']
  num_lrs = len(lrs)

  mp = SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_runs']
  scales = list(SCALING_LADDER['models'].keys())

  dp_mp = {16: 1, 32: 1, 64: 4, 128: 4, 256: 4, 512: 4}

  print('GBS :\t', f'Time x {num_lrs} LRs\t', f'  : DP : \t', f'GPU hours x {num_lrs} LRs')
  print('----' * 15)

  total_time, total_gpu_hours = 0, 0

  for sc in scales:
    print(sc)
    print('----')

    folder = f'/fast/jsingh/projects/fastlm/june/results/attn/{sc}'
    manager = ExperimentManager(sc, folder)
    acc_tps = manager.check_throughputs(dp=dp_mp)
    raw_tps = manager.check_throughputs(dp=1)

    for k, v in acc_tps.items():
      tokens = float(mp[k][:-1]) * 1e9
      hours = tokens / (v * 3600)
      total_time += hours

      gpu_hours = tokens / (raw_tps[k] * 3600)
      total_gpu_hours += gpu_hours
      print(
        k, ': \t', f'{round(hours, 2)} x {num_lrs}', f'hours    : {dp_mp[k]} : \t', f'{round(gpu_hours, 2)} x {num_lrs}'
      )

    print(' ')

  print('----' * 15)
  print('Total time = ', round(total_time * 5, 2), 'hours, assuming no runs in parallel')
  print('Total GPU hours =', round(total_gpu_hours * 5, 2))


if __name__ == '__main__':
  display_june_scaling_plan()
