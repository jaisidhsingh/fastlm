import os
import sys

from huggingface_hub import snapshot_download


HF_METRIC_FOLDER = "./data"
ARCH_IDS = ["attn", "gdn", "gdn+attn_3-1", "gdn+attn_1-1", "gdn+attn_1-3"]


def download(arch_id):
  hf_metric_folder = os.path.join(HF_METRIC_FOLDER, arch_id)
  os.makedirs(hf_metric_folder, exist_ok=True)

  repo_id = 'jaisidhsingh/OpenThesis_' + str(arch_id).replace('+', '-')
  snapshot_download(repo_id=repo_id, repo_type='dataset', allow_patterns='*.json', local_dir=hf_metric_folder)


if __name__ == '__main__':
  for arch_id in ["attn"]:
    download(arch_id)
