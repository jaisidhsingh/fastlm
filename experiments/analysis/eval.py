import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')
sns.set_style('whitegrid')
sns.set_palette(sns.color_palette('Set2'))


@dataclass
class Config:
  arch_id: str = 'attn'
  n: str = '150M'
  d: str = '15.0B'
  gbs: int = 64
  lr: float = 0.001


compute_n_map = {'20M': 7345408, '150M': 58739200, '50M': 26552832}

# dclm-core first


import numpy as np
from scipy.optimize import curve_fit


def fit_log_curve(x, y):
  """
  Fit y = a*ln(x) + b to scatter data.

  Parameters
  ----------
  x, y : array-like
      Data points. All x must be > 0.

  Returns
  -------
  dict with:
      a, b        : fitted parameters
      a_err, b_err: standard errors on a, b
      r_squared   : goodness of fit
      predict     : callable, predict(x_new) -> y_new
  """
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)

  def model(x, a, b):
    return a * np.log(x) + b

  popt, pcov = curve_fit(model, x, y)
  a, b = popt
  a_err, b_err = np.sqrt(np.diag(pcov))

  residuals = y - model(x, *popt)
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((y - np.mean(y)) ** 2)
  r_squared = 1 - ss_res / ss_tot

  return {
    'a': a,
    'b': b,
    'a_err': a_err,
    'b_err': b_err,
    'r_squared': r_squared,
    'predict': lambda x_new: model(np.asarray(x_new, dtype=float), a, b),
  }


lbl_map = {'attn': r'$rho=0$', 'gdn': r'$rho=1$', 'gdn+attn_3-1': r'$rho=0.75$'}


def load_data(cfg):
  name = f'dclm-core__d-{cfg.d.replace(".", "p")}.json'
  folder = f'/fast/jsingh/projects/fastlm/hf_sync_results/evals/{cfg.arch_id}/{cfg.n}/gbs_{cfg.gbs}__lr_{str(cfg.lr).replace(".", "p")}'
  path = os.path.join(folder, name)
  with open(path, 'r') as f:
    data = json.load(f)

  return data


def get_compute(cfg):
  return compute_n_map[cfg.n] * float(cfg.d[:-1]) * 1e9 * 6


def one_lr_core_plot():
  arch_ids = ['attn', 'gdn+attn_3-1', 'gdn']
  ns = ['150M']
  ds = ['0.5B', '1.0B', '3.0B', '7.5B', '15.0B']
  gbs = 64
  lr = 0.001

  cfg = Config(gbs=gbs, lr=lr)
  for idx, aid in enumerate(arch_ids):
    cfg.arch_id = aid
    lbl = lbl_map[aid]
    entry = True
    xs, ys = [], []

    for n in ns:
      cfg.n = n

      for d in ds:
        cfg.d = d
        try:
          core_val = load_data(cfg)['core_metric']
          compute = get_compute(cfg)
          if entry:
            kwargs = dict(c=f'C{idx}', s=100, label=lbl)
          else:
            kwargs = dict(c=f'C{idx}', s=100)
          entry = False
          plt.scatter(compute, core_val, **kwargs)
          xs.append(compute)
          ys.append(core_val)
        except FileNotFoundError:
          print('Skipping current setting cause it was not found')

  #    x_p = np.linspace(1e17, 5e18, 1000)
  #    fit = fit_log_curve(xs, ys)
  #    y_p = fit['predict'](x_p)
  #    plt.plot(x_p, y_p, c=f'C{idx}')

  plt.xlabel('Compute')
  plt.ylabel('CORE Metric')
  plt.xscale('log')
  plt.legend()
  plt.savefig('./results/core_eval.png', dpi=300, bbox_inches='tight')


def multi_lr_core_plot():
  # arch_ids = ['attn', 'gdn+attn_3-1', 'gdn']
  arch_ids = ['attn']
  ns = ['150M']
  # ds = ['7.5B', '15.0B']
  ds = ['15.0B']
  gbs = 128
  lrs = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008]
  markers = ['o', 's', '*', 'P', '^', 'X']

  cfg = Config(gbs=gbs, lr=0.0)
  for idx, aid in enumerate(arch_ids):
    cfg.arch_id = aid
    lbl = lbl_map[aid]
    entry = True
    xs, ys = [], []

    for n in ns:
      cfg.n = n

      for d in ds:
        cfg.d = d

        mean_core_val = 0.0
        for jdx, lr in enumerate(lrs):
          cfg.lr = lr
          try:
            core_val = load_data(cfg)['core_metric']
            mean_core_val += core_val
            compute = get_compute(cfg)
            kwargs = dict(c=f'C{idx}', s=100, marker=markers[jdx])
            entry = False
            plt.scatter(compute, core_val, **kwargs)
          except FileNotFoundError:
            print('Skipping current setting cause it was not found')
        mean_core_val /= len(lrs)
        plt.scatter(compute, mean_core_val, c='black', s=50, marker='D')

    for kdx, m in enumerate(markers):
      plt.scatter([], [], c='black', marker=m, label=lrs[kdx])
    plt.scatter([], [], c='black', marker='D', label='mean across LRs')

    plt.xlabel('Compute')
    plt.ylabel('CORE Metric')
    plt.xscale('log')
    plt.legend()
    plt.title(f'{aid.upper()} GBS={gbs}')
    plt.savefig(f'./results/dclm_core_150M_all_lrs_gbs-{gbs}_{aid}.png', dpi=300, bbox_inches='tight')
    plt.cla()
    plt.clf()


if __name__ == '__main__':
  multi_lr_core_plot()
