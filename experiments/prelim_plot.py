import json
import math
import os
import pickle
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.constants import SCALING_LADDER, SCALING_RESULTS_FOLDER
from src.metric_tensor import ScalingMetricTensor
from src.plotting.learning_rate import loss_vs_lr_heat_d, loss_vs_lr_heat_n

sns.set_palette(sns.color_palette('rocket'))

SQL = SCALING_LADDER['seq_len']
PARAM_SCALES = [k for k in SCALING_LADDER['models'].keys()]
TOKEN_BUDGETS = [k for k in SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_grid'].keys()]
BATCH_SIZES = SCALING_LADDER['batch_sizes']
LEARNING_RATES = SCALING_LADDER['learning_rates']

P = PARAM_SCALES[:2]
Q = TOKEN_BUDGETS[:3]  # when heat_d
B = BATCH_SIZES[:2]  # when heat_d
H = LEARNING_RATES

COORDS = {
  'n': P,
  'd': Q,
  'gbs': B,
  'lr': H,
}

CMAPS = {
  '20M': sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True, reverse=True),
  '50M': sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True, reverse=True),
  '150M': sns.cubehelix_palette(as_cmap=True, reverse=True),
  '300M': sns.color_palette('rocket_r', as_cmap=True),
}


def load_data(ns, ds, gbss, lrs, arch_id):
  def load_one(n, d, gbs, lr, arch_id):
    folder = os.path.join(
      SCALING_RESULTS_FOLDER,
      arch_id,
      n,
      'gbs_wise_results',
      f'gbs_{gbs}',
      'checkpoints',
      f'lr_{str(lr).replace(".", "p")}',
    )
    fname = f'metrics_decayed_to_{d.replace(".", "p")}.json'

    ref_d = SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_runs'][gbs]
    if float(d[:-1]) <= float(ref_d[:-1]):
      with open(os.path.join(folder, fname)) as f:
        da = json.load(f)
      return min(da['valid/loss'])

  data = np.ones((len(ns), len(ds), len(gbss), len(lrs))) * 1e6
  tensor = ScalingMetricTensor(data, COORDS)

  for n in ns:
    for d in ds:
      for gbs in gbss:
        for lr in lrs:
          val = load_one(n, d, gbs, lr, arch_id)
          tensor.set(val, n=n, d=d, gbs=gbs, lr=lr)

  return tensor


if __name__ == '__main__':
  t = load_data(ns=P, ds=Q, gbss=B, lrs=H, arch_id='attn')
  with open('/home/jsingh/projects/fastlm/execs/mts/attn_j17.pkl', 'wb') as f:
    pickle.dump(t, f)
