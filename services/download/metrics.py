import os
import sys

from huggingface_hub import snapshot_download

from src.constants import ARCH_IDS, HF_METRIC_FOLDER


def download(arch_id, cluster_id):
  hf_metric_folder = os.path.join(HF_METRIC_FOLDER[cluster_id], arch_id)
  os.makedirs(hf_metric_folder, exist_ok=True)

  repo_id = 'jaisidhsingh/OpenThesis_' + str(arch_id).replace('+', '-')
  snapshot_download(repo_id=repo_id, repo_type='dataset', allow_patterns='*.json', local_dir=hf_metric_folder)


if __name__ == '__main__':
  for arch_id in ARCH_IDS:
    download(arch_id, sys.argv[1])
