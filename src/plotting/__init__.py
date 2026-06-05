import json
import math
import os

import numpy as np

from src.constants import ANALYSIS_RESULTS_FOLDER, SCALING_RESULTS_FOLDER
from src.metric_tensor import ScalingMetricTensor

from .batch_size import *
from .compute import *
from .learning_rate import *


def load_data_for_plotting(ns, ds, gbss, lrs, arch_id, coords):
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
