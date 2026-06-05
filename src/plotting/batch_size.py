import math
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from src.constants import ANALYSIS_RESULTS_FOLDER, SCALING_RESULTS_FOLDER
from src.metric_tensor import ScalingMetricTensor


def loss_vs_gbs_heat_lr(metric_tensor: ScalingMetricTensor, n: str, d: str, arch_id: str) -> None:
  for lr in metric_tensor.stored_coords['lr']:
    plt.plot(
      metric_tensor.stored_coords['gbs'], metric_tensor.at(n=n, d=d, lr=lr), marker='o', label=r'$\eta=$' + str(lr)
    )

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$b$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, N={n}, D={d}')
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_gbs_heat_lr_N={n}_D={d}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_gbs_heat_d(metric_tensor: ScalingMetricTensor, n: str, lr: float, arch_id: str) -> None:
  for d in metric_tensor.stored_coords['d']:
    plt.plot(metric_tensor.stored_coords['gbs'], metric_tensor.at(n=n, d=d, lr=lr), marker='o', label=r'$D=$' + str(d))

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$b$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, N={n}, ' + r'$\eta=$' + str(lr))
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_gbs_heat_d_N={n}_lr={lr}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_gbs_heat_lr_and_d(metric_tensor: ScalingMetricTensor, n: str, arch_id: str) -> None:
  for d in metric_tensor.stored_coords['d']:
    for lr in metric_tensor.stored_coords['lr']:
      plt.plot(
        metric_tensor.stored_coords['gbs'],
        metric_tensor.at(n=n, d=d, lr=lr),
        marker='o',
        label=r'$D=$' + str(d) + r', $\eta=$' + str(lr),
      )

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$b$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, N={n}')
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_gbs_heat_lr_and_d_N={n}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_gbs_heat_n(metric_tensor: ScalingMetricTensor, lr: float, d: str, arch_id: str) -> None:
  for n in metric_tensor.stored_coords['n']:
    plt.plot(metric_tensor.stored_coords['gbs'], metric_tensor.at(n=n, d=d, lr=lr), marker='o', label=r'$N=$' + str(n))

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$b$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, D={d}, ' + r'$\eta=$' + str(gbs))
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_gbs_heat_n_lr={lr}_d={d}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_gbs_heat_n_and_lr(metric_tensor: ScalingMetricTensor, d: str, arch_id: str) -> None:
  for n in metric_tensor.stored_coords['n']:
    for lr in metric_tensor.stored_coords['lr']:
      plt.plot(
        metric_tensor.stored_coords['gbs'],
        metric_tensor.at(n=n, d=d, lr=lr),
        marker='o',
        label=r'$N=$' + str(n) + r', $\eta=$' + str(lr),
      )

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$b$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, D={d}')
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_gbs_heat_n_and_lr_d={d}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_gbs_heat_n_and_d(metric_tensor: ScalingMetricTensor, lr: float, arch_id: str) -> None:
  for n in metric_tensor.stored_coords['n']:
    for d in metric_tensor.stored_coords['d']:
      plt.plot(
        metric_tensor.stored_coords['gbs'],
        metric_tensor.at(n=n, d=d, lr=lr),
        marker='o',
        label=r'$N=$' + str(n) + r', $D=$' + str(d),
      )

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$b$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, ' + r'$\eta=$' + str(lr))
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_gbs_heat_n_and_d_lr={lr}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()
