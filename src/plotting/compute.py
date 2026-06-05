import math

import matplotlib.pyplot as plt
import numpy as np

from src.constants import ANALYSIS_RESULTS_FOLDER, SCALING_RESULTS_FOLDER
from src.metric_tensor import ScalingMetricTensor


def loss_vs_compute_loglog_fit(metric_tensor: ScalingMetricTensor, arch_id: str) -> None:
  xs = []
  ys = []
  for n in metric_tensor.stored_coords['n']:
    for d in metric_tensor.stored_coords['d']:
      nval = float(n[:-1]) * 1e6
      dval = float(d[:-1]) * 1e9
      c = nval * dval * 6
      xs.append(c)
      lossval = metric_tensor.at(P=n, Q=d).min()
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

  plt.title(f'{arch_id.upper()}')
  plt.grid(True)
  plt.savefig(f'{ANALYSIS_RESULTS_FOLDER}/plots/{arch_id.upper()}_loss_vs_compute.png', dpi=300, bbox_inches='tight')
  plt.cla()
  plt.clf()
