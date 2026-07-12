import json
import os
import typing as tp
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from beartype import beartype

from src.constants import PROJECT_REPO_ROOT, SCALING_RESULTS_FOLDER


@beartype
@dataclass
class ArtifactState:
  arch_id: str
  n: str
  d: str
  gbs: int
  lr: float
  checkpoint_filename: str
  metrics_filename: str
  cluster_location: str


class Inventory:
  def __init__(self, data: defaultdict | None = None):
    self.data = defaultdict(list) if data is None else data

  def __len__(self):
    k = list(self.data.keys())[0]
    return len(self.data[k])

  def push(self, state: ArtifactState) -> None:
    for k, v in asdict(state).items():
      self.data[k].append(v)

  def pop(self, state: ArtifactState) -> None:
    for k, v in asdict(state).items():
      self.data[k].remove(v)

  def save(self, path: str, format: str = 'csv') -> None:
    assert format in ['csv', 'json'], 'Incorrect saving format provided'
    if format == 'json':
      with open(path, 'w') as f:
        json.dump(self.data, f)
    else:
      df = pd.DataFrame(self.data)
      df.to_csv(path)

  def load(self, path: str) -> None:
    if path.endswith('.json'):
      with open(path) as f:
        self.data = defaultdict(list, json.load(f))
    elif path.endswith('.csv'):
      self.data = defaultdict(list)
      df = pd.read_csv(path)
      for _, row in df.iterrows():
        self.data[row['key']].append(row['value'])
    else:
      raise ValueError('Unsupported file path provided. Path must either be of a `json` or `csv` file.')


def union(i1: Inventory, i2: Inventory) -> pd.DataFrame:
  df1 = pd.DataFrame(i1.data)
  df2 = pd.DataFrame(i2.data)
  return pd.concat([df1, df2]).drop_duplicates(ignore_index=True)


def intersection(i1: Inventory, i2: Inventory) -> pd.DataFrame:
  df1 = pd.DataFrame(i1.data)
  df2 = pd.DataFrame(i2.data)
  return df1.merge(df2).drop_duplicates(ignore_index=True)


def difference(i1: Inventory, i2: Inventory) -> pd.DataFrame | None:
  # Returns rows in df1 that are not in df2. Equivalent to df1 - intersection(df1, df2)
  df1 = pd.DataFrame(i1.data)
  df2 = pd.DataFrame(i2.data)
  result = (
    df1.merge(df2, how='left', indicator=True)
    .query('_merge == "left_only"')
    .drop(columns='_merge')
    .reset_index(drop=True)
  )
  return None if result.empty else result


def take_inventory(cluster_id):
  base_root = SCALING_RESULTS_FOLDER[cluster_id]
  root = Path(base_root)

  inventory = Inventory()

  # if metrics_xyz.json exists then ckpt_xyz.pt exists too
  for path in root.rglob('*.json'):
    if path.is_file():
      path_str = str(path)
      root_removed_path = path_str.split(base_root)[-1]
      # `root_removed_path` looks like:
      # attn/20M/gbs_wise_results/gbs_16/checkpoints/lr_0p008/metrics_decayed_to_0p5B.json

      entries = root_removed_path.split('/')
      entries.remove('gbs_wise_results')
      entries.remove('checkpoints')

      [arch_id, n, gbs_ext, lr_ext, metrics_fname] = entries
      gbs = int(gbs_ext.split('_')[-1])
      lr = float(lr_ext.split('_')[-1].replace('p', '.'))
      d = metrics_fname.split('_')[-1].replace('p', '.')
      ckpt_fname = metrics_fname.replace('metrics_', 'ckpt_').replace('.json', '.pt')

      cluster_location = 'tud' if cluster_id in ['capella', 'alpha'] else 'mpi'
      artifact_state = ArtifactState(
        arch_id=arch_id,
        n=n,
        d=d,
        gbs=gbs,
        lr=lr,
        checkpoint_filename=ckpt_fname,
        metrics_filename=metrics_fname,
        cluster_location=cluster_location,
      )
      inventory.push(artifact_state)

  return inventory


def get_checkpoints_from_changes(changes: pd.DataFrame, cluster_id: str) -> defaultdict:
  base_folder = SCALING_RESULTS_FOLDER[cluster_id]
  paths = defaultdict(list)

  for i in range(len(changes)):
    src_folder = os.path.join(
      base_folder,
      str(changes[i]['arch_id']),
      str(changes[i]['n']),
      'gbs_wise_results',
      f'gbs_{changes[i]["gbs"]}',
      'checkpoints',
      f'lr_{str(changes[i]["lr"]).replace(".", "p")}',
    )
    src_ckpt_path = os.path.join(src_folder, f'ckpt_decayed_to_{changes[i]["d"].replace(".", "p")}.pt')
    paths['checkpoints_src'].append(src_ckpt_path)

    src_metrics_path = os.path.join(src_folder, f'metrics_decayed_to_{changes[i]["d"].replace(".", "p")}.json')
    paths['metrics_src'].append(src_metrics_path)

    dest_folder = os.path.join(
      str(changes[i]['n']),
      f'gbs_{changes[i]["gbs"]}',
      f'lr_{str(changes[i]["lr"]).replace(".", "p")}',
    )

    dest_ckpt_path = os.path.join(dest_folder, f'ckpt_decayed_to_{changes[i]["d"].replace(".", "p")}.pt')
    paths['checkpoints_dest'].append(dest_ckpt_path)

    dest_metrics_path = os.path.join(dest_folder, f'metrics_decayed_to_{changes[i]["d"].replace(".", "p")}.json')
    paths['metrics_dest'].append(dest_metrics_path)

    paths['dest_repo'].append(f'OpenThesis_{changes[i]["arch_id"]}')

  return paths
