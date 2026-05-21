import random

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data import get_dataloaders
from src.data.datasamplers import StatefulRandomSampler

OWT_PATH = '/fast/jsingh/data/owt-tokenized-9b-train-nn'
TOKENIZER_PATH = '/home/jsingh/projects/fastlm/tokenizer/better-gpt2'

NEMO_PATH = '/fast/jsingh/data/nemotron-cc-sample-mtsynth/tokenized_gpt2/ctx_4096/train'


def view_dataset():
  dataset = load_from_disk(NEMO_PATH)
  print(dataset[0])
  return
  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
  sampler = StatefulRandomSampler(data_source=dataset, batch_size=16, shuffle=True, seed=0)

  # def collate_fn(batch):
  #   return torch.stack([torch.tensor(item['tokens'], dtype=torch.long) for item in batch])

  loader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=16,
    collate_fn=collate_fn,
  )

  batch = next(iter(loader))
  print(batch.shape)


p_to_tps_map = {300: 37000, 150: 72000, 50: 106000, 20: 180000, 10: 250000}


def get_chincilla_tokens(params_in_m):
  return 20 * params_in_m * 1e6


def get_time_to_train(params_in_m):
  tps = p_to_tps_map[params_in_m]
  s = get_chincilla_tokens(params_in_m) / tps
  return s / 3600


if __name__ == '__main__':
  for p in p_to_tps_map.keys():
    t = get_time_to_train(p)
    print(f'GPU hours needed to train a {p}M dense LLM', t)
