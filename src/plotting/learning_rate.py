import math
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from src.constants import ANALYSIS_RESULTS_FOLDER, SCALING_RESULTS_FOLDER
from src.metric_tensor import ScalingMetricTensor


def loss_vs_lr_heat_gbs(metric_tensor: ScalingMetricTensor, n: str, d: str, arch_id: str) -> None:
  for gbs in metric_tensor.stored_coords['gbs']:
    plt.plot(
      metric_tensor.stored_coords['lrs'], metric_tensor.at(n=n, d=d, gbs=gbs), marker='o', label=r'$b=$' + str(gbs)
    )

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, N={n}, D={d}')
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_lr_heat_gbs_N={n}_D={d}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_d(metric_tensor: ScalingMetricTensor, n: str, gbs: int, arch_id: str) -> None:
  for d in metric_tensor.stored_coords['d']:
    plt.plot(metric_tensor.stored_coords['lr'], metric_tensor.at(n=n, d=d, gbs=gbs), marker='o', label=r'$D=$' + str(d))

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, N={n}, ' + r'$b=$' + str(gbs))
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_lr_heat_d_N={n}_gbs={gbs}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_gbs_and_d(metric_tensor: ScalingMetricTensor, n: str, arch_id: str) -> None:
  for d in metric_tensor.stored_coords['d']:
    for gbs in metric_tensor.stored_coords['gbs']:
      plt.plot(
        metric_tensor.stored_coords['lr'],
        metric_tensor.at(n=n, d=d, gbs=gbs),
        marker='o',
        label=r'$D=$' + str(d) + r', $b=$' + str(gbs),
      )

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, N={n}')
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_lr_heat_gbs_and_d_N={n}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_n(metric_tensor: ScalingMetricTensor, gbs: int, d: str, arch_id: str) -> None:
  for n in metric_tensor.stored_coords['n']:
    plt.plot(metric_tensor.stored_coords['lr'], metric_tensor.at(n=n, d=d, gbs=gbs), marker='o', label=r'$N=$' + str(n))

  plt.xlim([2 ** (-13), 2 ** (-6)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, D={d}, ' + r'$b=$' + str(gbs))
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_lr_heat_n_gbs={gbs}_d={d}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_n_and_gbs(metric_tensor: ScalingMetricTensor, d: str, arch_id: str) -> None:
  for n in metric_tensor.stored_coords['n']:
    for gbs in metric_tensor.stored_coords['gbs']:
      plt.plot(
        metric_tensor.stored_coords['lr'],
        metric_tensor.at(n=n, d=d, gbs=gbs),
        marker='o',
        label=r'$N=$' + str(n) + r', $b=$' + str(gbs),
      )

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, D={d}')
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_lr_heat_n_and_gbs_d={d}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()


def loss_vs_lr_heat_n_and_d(metric_tensor: ScalingMetricTensor, gbs: int, arch_id: str) -> None:
  for n in metric_tensor.stored_coords['n']:
    for d in metric_tensor.stored_coords['d']:
      plt.plot(
        metric_tensor.stored_coords['lr'],
        metric_tensor.at(n=n, d=d, gbs=gbs),
        marker='o',
        label=r'$N=$' + str(n) + r', $D=$' + str(d),
      )

  plt.xlim([2 ** (-9), 2 ** (-3)])
  plt.xscale('log', base=2)
  plt.xlabel(r'$\eta$')
  plt.ylabel(r'$\mathcal{L}_{\text{valid}}$')

  plt.title(f'{arch_id.upper()}, ' + r'$b=$' + str(gbs))
  plt.grid(True)
  plt.legend()
  plt.savefig(
    f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_lr_heat_n_and_d_gbs={gbs}.png',
    dpi=300,
    bbox_inches='tight',
  )
  plt.cla()
  plt.clf()
