import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import SCALING_LADDER, SCALING_RESULTS_FOLDER
from src.metric_tensor import ScalingMetricTensor


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
    data = json.load(f)
  return data
