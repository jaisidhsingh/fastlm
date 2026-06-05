import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.constants import ANALYSIS_RESULTS_FOLDER, SCALING_RESULTS_FOLDER
from src.metric_tensor import ScalingMetricTensor

sns.set_palette(sns.color_palette('rocket'))


def load_data(ns, ds, gbss, lrs, arch_id, coords):
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
    with open(os.path.join(folder, fname)) as f:
      da = json.load(f)
    return min(da['valid/loss'])

  data = np.ones((len(ns), len(ds), len(gbss), len(lrs))) * 1e6
  tensor = ScalingMetricTensor(data, coords)

  for n in ns:
    for d in ds:
      for gbs in gbss:
        for lr in lrs:
          val = load_one(n, d, gbs, lr, arch_id)
          tensor.set(val, P=n, Q=d, B=gbs, H=lr)

  return tensor


def loss_vs_lr_heat_gbs(lrs, metric_tensor, n, d, arch_id, gbss):
  for gbs in gbss:
    plt.plot(lrs, metric_tensor.at(P=n, Q=d, B=gbs), marker='o', label=r'$b=$' + str(gbs))

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, N={n}, D={d}')
  plt.grid(True)
  plt.legend()
  plt.savefig(f'./drawings/plots/{arch_id.upper()}_loss_vs_lr_heat_gbs_N={n}_D={d}.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_d(lrs, metric_tensor, n, gbs, arch_id, ds):
  for d in ds:
    plt.plot(lrs, metric_tensor.at(P=n, Q=d, B=gbs), marker='o', label=r'$D=$' + str(d))

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, N={n}, ' + r'$b=$' + str(gbs))
  plt.grid(True)
  plt.legend()
  plt.savefig(f'./drawings/plots/{arch_id.upper()}_loss_vs_lr_heat_d_N={n}_gbs={gbs}.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_gbs_and_d(lrs, metric_tensor, n, arch_id, ds, gbss):
  for d in ds:
    for gbs in gbss:
      plt.plot(lrs, metric_tensor.at(P=n, Q=d, B=gbs), marker='o', label=r'$D=$' + str(d) + r', $b=$' + str(gbs))

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, N={n}')
  plt.grid(True)
  plt.legend()
  plt.savefig(f'./drawings/plots/{arch_id.upper()}_loss_vs_lr_heat_gbs_and_d_N={n}.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_n(lrs, metric_tensor, gbs, d, arch_id, ns):
  for n in ns:
    plt.plot(lrs, metric_tensor.at(P=n, Q=d, B=gbs), marker='o', label=r'$N=$' + str(n))

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, D={d}, ' + r'$b=$' + str(gbs))
  plt.grid(True)
  plt.legend()
  plt.savefig(f'./drawings/plots/{arch_id.upper()}_loss_vs_lr_heat_n_gbs={gbs}_d={d}.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_n_and_gbs(lrs, metric_tensor, d, arch_id, ns, gbss):
  for n in ns:
    for gbs in gbss:
      plt.plot(lrs, metric_tensor.at(P=n, Q=d, B=gbs), marker='o', label=r'$N=$' + str(n) + r', $b=$' + str(gbs))

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, D={d}')
  plt.grid(True)
  plt.legend()
  plt.savefig(f'./drawings/plots/{arch_id.upper()}_loss_vs_lr_heat_n_and_gbs_d={d}.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_n_and_d(lrs, metric_tensor, gbs, arch_id, ns, ds):
  for n in ns:
    for d in ds:
      plt.plot(lrs, metric_tensor.at(P=n, Q=d, B=gbs), marker='o', label=r'$N=$' + str(n) + r', $D=$' + str(d))

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, ' + r'$b=$' + str(gbs))
  plt.grid(True)
  plt.legend()
  plt.savefig(f'./drawings/plots/{arch_id.upper()}_loss_vs_lr_heat_n_and_d_gbs={gbs}.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()


def loss_vs_compute(metric_tensor, arch_id, ns, ds):
  xs = []
  ys = []
  for n in ds:
    for d in ds:
      nval = float(n[:-1]) * 1e6
      dval = float(d[:-1]) * 1e9
      c = nval * dval * 6
      xs.append(c)
      lossval = metric_tensor.at(P=n, Q=d, B=32, H=0.0025)
      ys.append(lossval)
      plt.scatter(c, lossval)

  xs = np.array(xs)
  ys = np.array(ys)

  logx = np.log(xs)
  logy = np.log(ys)
  b, a = np.polyfit(logx, logy, deg=1)

  xfit = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 200)
  yfit = np.exp(a) * xfit**b
  plt.plot(xfit, yfit, label='fit')

  plt.xscale('log', base=10)
  plt.yscale('log', base=10)
  plt.xlabel(r'Compute')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, ' + r'$b=$' + str(32) + r', $\eta=$' + str(0.0025))
  plt.grid(True)
  plt.savefig(f'./drawings/plots/{arch_id.upper()}_loss_vs_compute.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()
