import json
import os
from dataclasses import dataclass

import torch
import tyro
from huggingface_hub import HfApi

# 0. before every run, we make a `git pull` to update inventories
# 1. after each run, we want to take inventory on the cluster the run finished on.
# 2. then, we want to compare the inventory with the state of our HF storage.
# 3. if there is an aritifact of a new (N,D,GBS,LR,rho) on the cluster, upload it to HF storage.
# 4. else if there was a change to an existing (N,D,GBS,LR,rho) on the cluster, upload->overwrite.
# 5. otherwise do nothing (run was not supposed to generate any artifacts)
# 6. finally, make a `git push`


def dummy_upload():
  api = HfApi()
  api.upload_file(
    path_or_fileobj='/path/to/model.pt',
    path_in_repo='data/model.pt',
    repo_id='username/my-dataset',
    repo_type='dataset',
  )


class SyncServiceConfig:
  cluster_id: str


def take_inventory(cluster_id):
  pass


def load_hf_state():
  pass


def compare_inventory_with_hf_state(inventory, hf_state):
  pass


def extract_model_state_dict_to_tmpfile(checkpoint_path):
  folder = 'some_prefix_to_fast_or_horse'
  model_state_dict = torch.load(checkpoint_path, weights_only=False)['state_dict']
  tmpfile_path = os.path.join(folder, 'tmpfile_for_upload_to_hf.pt')
  torch.save(model_state_dict, tmpfile_path)
  return tmpfile_path


def get_checkpoint_from_changes(changes):
  return None, None


def make_update_from_checkpoint_id(checkpoint_id):
  return None


def update_hf_state(old_hf_state, update):
  return None


def upload_checkpoint_to_hf(checkpoint_id, checkpoint_path):
  pass


def main(cfg: SyncServiceConfig):
  inventory = take_inventory(cfg.cluster_id)
  hf_state = load_hf_state()

  changes = compare_inventory_with_hf_state(inventory, hf_state)
  if changes is None:
    return

  checkpoint_id, checkpoint_path = get_checkpoint_from_changes(changes)
  update_to_hf_state = make_update_from_checkpoint_id(checkpoint_id)

  try:
    upload_checkpoint_to_hf(checkpoint_id, checkpoint_path)
  except Exception as e:
    print(e)
  finally:
    update_hf_state(hf_state, update_to_hf_state)


if __name__ == '__main__':
  cfg = tyro.cli(SyncServiceConfig)
  main(cfg)
