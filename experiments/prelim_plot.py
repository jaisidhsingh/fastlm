import json
import math
import os
import pickle
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb

from src.constants import SCALING_LADDER, SCALING_RESULTS_FOLDER
from src.metric_tensor import ScalingMetricTensor
from src.plotting.learning_rate import loss_vs_lr_heat_d, loss_vs_lr_heat_n

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

BASE_COLORS = {
  '20M': '#2166ac',  # blue
  '50M': '#762a83',  # wine/purple
  '150M': '#d73027',  # red
  '300M': '#FFA500',  # orange
}

LIGHTNESS = {'0.5B': 0.0, '1.0B': 0.35, '3.0B': 0.6}


def lighten(color, amount):
  c = np.array(to_rgb(color))
  return tuple(c + (1 - c) * amount)


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
        # print(n, d, gbs, lr)
        return 0.0

      if os.path.exists(os.path.join(folder, fname)):
        with open(os.path.join(folder, fname)) as f:
          da = json.load(f)
        return da['valid/loss'][-1]

  data = np.ones((len(ns), len(ds), len(gbss), len(lrs))) * 1e6
  tensor = ScalingMetricTensor(data, COORDS)

  for n in ns:
    for d in ds:
      for gbs in gbss:
        for lr in lrs:
          val = load_one(n, d, gbs, lr, arch_id)
          tensor.set(val, n=n, d=d, gbs=gbs, lr=lr)

  return tensor


def plot_lr_sweep(arch_id):
  t = load_data(ns=P, ds=Q, gbss=B, lrs=H, arch_id=arch_id)
  with open(f'/home/jsingh/projects/fastlm/execs/mts/{arch_id}_j17.pkl', 'wb') as f:
    pickle.dump(t, f)

  fig, ax = plt.subplots(
    len(B[:]),
    len(P),
    figsize=(24, 10),  # scale this up — width per column ~5-6in, height per row ~5in
    sharey=False,
  )

  for ii, gbs in enumerate(B[:]):
    for jj, n in enumerate(P):
      for d in Q:
        color = lighten(BASE_COLORS[n], LIGHTNESS[d])
        y = t.at(n=n, d=d, gbs=gbs)

        if y._hdata.sum() > 0:
          yy = y._hdata[y._hdata > 0]
          lrx = np.array(t.stored_coords['lr'])[y._hdata > 0]
          ax[ii, jj].scatter(lrx, yy, label=f'N={n}, D={d}', color=color, marker='o')

          X = np.log2(lrx)
          A = np.vstack([X**2, X, np.ones_like(X)]).T
          a, b, c = np.linalg.lstsq(A, yy, rcond=None)[0]
          x_fit = np.linspace(0.75 * (2 ** (-12)), 0.75 * (2 ** (-6)), 200)
          logx = np.log2(x_fit)
          y_fit = a * logx**2 + b * logx + c
          ax[ii, jj].plot(x_fit, y_fit, color=color)

          best_idx = np.argmin(y_fit)
          best_lr = x_fit[best_idx]
          best_loss = y_fit[best_idx]

          ax[ii, jj].scatter(
            best_lr,
            best_loss,
            zorder=10,
            color=color,
            marker='*',
            s=200,
            # edgecolor='black',
          )

      ax[ii, jj].set_xlim([0.75 * 2 ** (-12), 0.75 * 2 ** (-6)])
      ax[ii, jj].set_ylim([3.0, 4.4])
      ax[ii, jj].set_xscale('log', base=2)
      ax[ii, jj].set_xlabel(r'$\eta$')
      ax[ii, jj].set_ylabel(r'$\mathcal{L}_{\text{valid}}$')

      ax[ii, jj].set_title(f'GBS={gbs}')
      ax[ii, jj].grid(True)
      ax[ii, jj].legend(loc='upper right')

  plt.suptitle(f'ARCH={arch_id}')
  plt.savefig(f'plots/{arch_id}.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()


def plot_n_d_laws(arch_id):
  t = load_data(ns=P, ds=Q, gbss=B, lrs=H, arch_id=arch_id)

  fig, ax = plt.subplots(
    1,
    2,
    figsize=(10, 6),
    sharey=False,
  )
  n, gbs = '150M', 32

  d_lrs = []
  d_lvs = []

  for d in Q:
    color = lighten(BASE_COLORS[n], LIGHTNESS[d])
    y = t.at(n=n, d=d, gbs=gbs)

    if y._hdata.sum() > 0:
      yy = y._hdata[y._hdata > 0]
      lrx = np.array(t.stored_coords['lr'])[y._hdata > 0]
      ax[0].scatter(lrx, yy, color=color, marker='o', label=d)

      X = np.log2(lrx)
      A = np.vstack([X**2, X, np.ones_like(X)]).T
      a, b, c = np.linalg.lstsq(A, yy, rcond=None)[0]
      x_fit = np.linspace(0.75 * (2 ** (-12)), 0.75 * (2 ** (-6)), 200)
      logx = np.log2(x_fit)
      y_fit = a * logx**2 + b * logx + c

      ax[0].plot(x_fit, y_fit, color=color)

      best_idx = np.argmin(y_fit)
      best_lr = x_fit[best_idx]
      best_loss = y_fit[best_idx]
      ax[0].scatter(
        best_lr,
        best_loss,
        zorder=10,
        color=color,
        marker='*',
        s=200,
        # edgecolor='black',
      )

      d_lrs.append(best_lr)
      d_lvs.append(best_loss)

  X = np.log2(d_lrs)
  A = np.vstack([X, np.ones_like(X)]).T
  a, b = np.linalg.lstsq(A, d_lvs, rcond=None)[0]
  x_fit = np.linspace(0.75 * (2 ** (-12)), 0.75 * (2 ** (-6)), 200)
  logx = np.log2(x_fit)
  y_fit = a * logx + b

  ax[0].plot(x_fit, y_fit, linestyle='--', color='black', label=rf'{round(a, 3)} * $\log x + $ {round(b, 3)}')
  ax[0].set_title(f'N={n}, GBS={gbs}')
  ax[0].set_xscale('log', base=2)
  ax[0].set_xlim([0.75 * 2 ** (-13), 0.75 * 2 ** (-6)])
  ax[0].set_ylim([3.0, 4.4])
  ax[0].set_xlabel(r'$\eta$')
  ax[0].set_ylabel(r'$\mathcal{L}_{\text{valid}}$')
  ax[0].legend(loc='upper left', fontsize=11)

  d = '3.0B'
  n_lrs = []
  n_lvs = []
  ns = P
  for n in ns:
    color = BASE_COLORS[n]
    color = np.array(to_rgb(color))
    y = t.at(n=n, d=d, gbs=gbs)

    if y._hdata.sum() > 0:
      yy = y._hdata[y._hdata > 0]
      lrx = np.array(t.stored_coords['lr'])[y._hdata > 0]
      ax[1].scatter(lrx, yy, label=f'{n}', color=color, marker='o')

      X = np.log2(lrx)
      A = np.vstack([X**2, X, np.ones_like(X)]).T
      a, b, c = np.linalg.lstsq(A, yy, rcond=None)[0]
      x_fit = np.linspace(0.75 * (2 ** (-12)), 0.75 * (2 ** (-6)), 200)
      logx = np.log2(x_fit)
      y_fit = a * logx**2 + b * logx + c

      ax[1].plot(x_fit, y_fit, color=color)

      best_idx = np.argmin(y_fit)
      best_lr = x_fit[best_idx]
      best_loss = y_fit[best_idx]
      ax[1].scatter(
        best_lr,
        best_loss,
        zorder=10,
        color=color,
        marker='*',
        s=200,
        # edgecolor='black',
      )

      n_lrs.append(best_lr)
      n_lvs.append(best_loss)

  X = np.log2(n_lrs)
  A = np.vstack([X, np.ones_like(X)]).T
  a, b = np.linalg.lstsq(A, n_lvs, rcond=None)[0]
  x_fit = np.linspace(0.75 * (2 ** (-12)), 0.75 * (2 ** (-6)), 200)
  logx = np.log2(x_fit)
  y_fit = a * logx + b

  ax[1].plot(x_fit, y_fit, linestyle='--', color='black', label=rf'{round(a, 3)} * $\log x + $ {round(b, 3)}')
  ax[1].set_title(f'D={d}, GBS={gbs}')
  ax[1].set_xscale('log', base=2)
  ax[1].set_xlim([0.75 * 2 ** (-12), 0.75 * 2 ** (-6)])
  ax[1].set_ylim([3.0, 4.4])
  ax[1].set_xlabel(r'$\eta$')
  ax[1].set_ylabel(r'$\mathcal{L}_{\text{valid}}$')
  ax[1].legend(loc='upper left', fontsize=11)

  plt.suptitle(f'ARCH={arch_id}')
  plt.savefig(f'plots/{arch_id}_2.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()


def find_opt_lr_law(arch_id):
  gbs = 32
  print(f'ARCH={arch_id}, GBS={gbs}')
  t = load_data(ns=P, ds=Q, gbss=B, lrs=H, arch_id=arch_id)
  t = t.at(n=P, gbs=gbs)
  _d = []

  for jj, n in enumerate(P):
    for d in Q:
      y = t.at(n=n, d=d)

      if y._hdata.sum() > 0:
        yy = y._hdata[y._hdata > 0]
        lrx = np.array(t.stored_coords['lr'])[y._hdata > 0]

        X = np.log2(lrx)
        A = np.vstack([X**2, X, np.ones_like(X)]).T
        a, b, c = np.linalg.lstsq(A, yy, rcond=None)[0]
        x_fit = np.linspace(0.75 * (2 ** (-12)), 0.75 * (2 ** (-6)), 200)
        logx = np.log2(x_fit)
        y_fit = a * logx**2 + b * logx + c

        best_idx = np.argmin(y_fit)
        best_lr = x_fit[best_idx]
        best_loss = y_fit[best_idx]
        _d.append([best_lr, float(n[:-1]) * 1e6, float(d[:-1]) * 1e9])

  _d = np.array(_d).astype(np.float32)
  _lr = np.array([dd[0] for dd in _d])
  _N = np.array([dd[1] for dd in _d])
  _D = np.array([dd[2] for dd in _d])

  X = np.column_stack([np.log(_N), np.log(_D), np.ones_like(_N)])
  y = np.log(_lr)

  coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
  a, b, c = coeffs

  print(f'a = {a:.4f}, b = {b:.4f}, c = {c:.4f}')

  log_lr_pred = X @ coeffs
  lr_pred = np.exp(log_lr_pred)

  ss_res = np.sum((y - log_lr_pred) ** 2)
  ss_tot = np.sum((y - y.mean()) ** 2)
  r2 = 1 - ss_res / ss_tot
  print(f'R^2 = {r2:.4f}')

  labels = ['a', 'b', 'c', r'$R^2$']
  values = [a, b, c, r2]
  values = [round(float(xv), 3) for xv in values]
  a = round(float(a), 3)
  b = round(float(b), 3)
  c = round(float(c), 3)
  r2 = round(float(r2), 3)

  fig, ax = plt.subplots(1, 2, figsize=(10, 6))
  df = pd.DataFrame({'LR': _lr, 'N': _N, 'D': _D})

  sns.lmplot(data=df, x='N', y='LR', hue='D', palette='rocket', markers='o', ci=None, legend=True)
  plt.legend(title='D')
  plt.title(
    f'ARCH={arch_id}, GBS={gbs}' + '\n' + rf'{round(a, 3)} * $\log N +$ {round(b, 3)} * $\log D + $ {round(c, 3)}'
  )
  plt.savefig(f'./plots/{arch_id}_3.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()

  sns.lmplot(data=df, x='D', y='LR', hue='N', palette='mako', markers='o', ci=None, legend=True)
  plt.legend(title='N')
  plt.title(
    f'ARCH={arch_id}, GBS={gbs}' + '\n' + rf'{round(a, 3)} * $\log N +$ {round(b, 3)} * $\log D + $ {round(c, 3)}'
  )
  plt.savefig(f'./plots/{arch_id}_4.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()

  def predict_lr(N_new, D_new):
    return np.exp(a * np.log(N_new) + b * np.log(D_new) + c)


def main(arch_id):
  plot_lr_sweep(arch_id)
  plot_n_d_laws(arch_id)
  find_opt_lr_law(arch_id)


if __name__ == '__main__':
  sns.set_style('whitegrid')
  main('attn')
  #main('gdn')
