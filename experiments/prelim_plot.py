import json
import math
import os
import pickle
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgb

from src.constants import SCALING_LADDER, SCALING_RESULTS_FOLDER
from src.metric_tensor import ScalingMetricTensor
from src.plotting.learning_rate import loss_vs_lr_heat_d, loss_vs_lr_heat_n

sns.set_palette(sns.color_palette('rocket'))

SQL = SCALING_LADDER['seq_len']
PARAM_SCALES = [k for k in SCALING_LADDER['models'].keys()]
TOKEN_BUDGETS = [k for k in SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_grid'].keys()]
BATCH_SIZES = SCALING_LADDER['batch_sizes']
LEARNING_RATES = SCALING_LADDER['learning_rates']

P = PARAM_SCALES[:4]
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
  '20M': sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=False, reverse=False),
  '50M': sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=False, reverse=False),
  '150M': sns.color_palette('ch:s=-.2,r=.6', as_cmap=False),
  '300M': sns.color_palette('rocket_r', as_cmap=False),
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
      if not os.path.exists(os.path.join(folder, fname)):
        print(n, d, gbs, lr)

      if os.path.exists(os.path.join(folder, fname)):
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


# with open('/home/jsingh/projects/fastlm/execs/mts/attn_j17.pkl', 'wb') as f:
# pickle.dump(t, f)

# %%

BASE_COLORS = {
  '20M': '#2166ac',  # blue
  '50M': '#762a83',  # wine/purple
  '150M': '#d73027',  # red
  '300M': '#FFA500',  # orange
}


def lighten(color, amount):
  c = np.array(to_rgb(color))
  return tuple(c + (1 - c) * amount)


LIGHTNESS = {'0.5B': 0.0, '1.0B': 0.4, '3.0B': 0.8}

arch_id = 'gdn'
t = load_data(ns=P, ds=Q, gbss=B, lrs=H, arch_id=arch_id)

for gbs in B:
  for n in P:
    for d in Q:
      color = lighten(BASE_COLORS[n], LIGHTNESS[d])
      y = t.at(n=n, d=d, gbs=gbs)
      if y._hdata.sum() > 0 and not (gbs == 32 and d == '0.5B'):
        best_idx = np.argmin(y._hdata)
        best_lr = t.stored_coords['lr'][best_idx]
        best_loss = y._hdata[best_idx]

        plt.plot(t.stored_coords['lr'], y, label=f'{n} - {d}', color=color)
        plt.scatter(best_lr, best_loss, s=40, zorder=10, color=color, marker='*')
      else:
        print(n, d, gbs)

  plt.xlim([2 ** (-13), 2 ** (-6)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, GBS={gbs}')
  plt.grid(True)
  plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0)
  plt.tight_layout()
  plt.savefig(f'{arch_id}_{gbs}.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()

# %%
