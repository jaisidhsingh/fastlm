import json
import os
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

  def check_steps_accomodate_warmup(self):
    for gbs in self.info['batch_sizes']:
      budget = self.gbs_to_token_map[gbs]
      total_tokens = self.token_budget_map[budget]
      steps = int(total_tokens // (gbs * self.seq_len))
      if steps < self.warmup_steps or self.warmup_steps + self.cooldown_steps >= steps:
        print(
          f'Bad combination of warmup steps={self.warmup_steps} and cooldown steps={self.cooldown_steps} with total steps={steps} for GBS={gbs}'
        )
      else:
        print(
          f'Warmup steps={self.warmup_steps} and cooldown steps={self.cooldown_steps} accomodated by total steps={steps} for GBS={gbs} under token budget={budget}'
        )

  def make_throughput_analysis_configs(self, gbs: int) -> list:
    dp = self.info['models'][self.parameter_scale_id]['dp']
    cfg = SimpleNamespace(**DEFAULT_CONFIG)

    save_folder = ''
    for gas in self.info['gradient_accumulation_steps']:
      gas = int(gas)
      if dp * gas > gbs:
        break
      mbs = int(gbs / (dp * gas))
      cfg.micro_batch_size = mbs
      cfg.grad_accumulation_steps = gas

      save_folder = os.path.join(
        self.experiment_folder, 'gbs_wise_results', f'gbs_{gbs}', 'throughput_analysis_configs'
      )
      os.makedirs(save_folder, exist_ok=True)

      with open(os.path.join(save_folder, f'mbs-{mbs}_gas-{gas}.yaml'), 'w') as f:
        yaml.safe_dump(vars(cfg), f, sort_keys=False)

    return [os.path.join(save_folder, fname) for fname in os.listdir(save_folder)]

  def check_throughputs(self) -> dict | None:
    results = {}
    folder = os.path.join(self.experiment_folder, 'gbs_wise_results')
    os.makedirs(folder, exist_ok=True)

    for gbs in self.info['batch_sizes']:
      gbs_folder = os.path.join(folder, f'gbs_{gbs}')
      os.makedirs(gbs_folder, exist_ok=True)
      tp_result_path = os.path.join(gbs_folder, 'throughput_analysis.json')

      tp = None
      if os.path.exists(tp_result_path) and results is not None:
        with open(tp_result_path, 'r') as f:
          tp = json.load(f)

      elif not os.path.exists(os.path.join(gbs_folder, 'throughput_analysis_configs')):
        tp_config_paths = self.make_throughput_analysis_configs(gbs)
        print(f'Created configs for throughput analysis of {self.parameter_scale_id} on gbs={gbs}.')
        print('You must execute `experiments/throughput_analysis.py` on these configs:')
        for path in tp_config_paths:
          print(f'\t-> {path}')
        results = None

      if tp is not None:
        results[gbs] = tp['fastest']

    return results

  def create_configs_for_sweep(self):
    folder = os.path.join(self.experiment_folder, 'gbs_wise_results')
    os.makedirs(folder, exist_ok=True)
    checked = self.check_throughputs()

    if checked is None:
      print('Cannot create configs for sweep before throughput analysis is done. Exiting.')
      return

    for gbs in self.info['batch_sizes']:
      tmp_cfg = deepcopy(DEFAULT_CONFIG)
      tmp_cfg['micro_batch_size'] = checked[gbs]['mbs']
      tmp_cfg['grad_accumulation_steps'] = checked[gbs]['gas']

      save_folder = os.path.join(folder, f'gbs_{gbs}', 'lr_wise_configs')
      os.makedirs(save_folder)
      for lr in self.info['learning_rates']:
        tmp_cfg['lr'] = lr

        name = 'config_lr_' + str(lr).replace('.', 'p') + '.yaml'
        with open(os.path.join(save_folder, name), 'w') as f:
          yaml.safe_dump(tmp_cfg, f, sort_keys=False)


if __name__ == '__main__':
  scales = list(SCALING_LADDER['models'].keys())
  folder = '/Users/jaisidhsingh/Code/tuebingen/thesis/code/fastlm/manager_check'
  for sc in scales:
    manager = ExperimentManager(sc, folder)
    manager.check_steps_accomodate_warmup()
