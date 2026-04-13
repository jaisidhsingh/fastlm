import random

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import get_dataloaders
from data.datasamplers import StatefulRandomSampler

OWT_PATH = '/fast/jsingh/data/owt-tokenized-9b-train-nn'
TOKENIZER_PATH = '/home/jsingh/projects/fastlm/tokenizer/better-gpt2'


def view_dataset():
  dataset = load_from_disk(OWT_PATH)
  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
  sampler = StatefulRandomSampler(data_source=dataset, batch_size=16, shuffle=True, seed=0)

  def collate_fn(batch):
    return torch.stack([torch.tensor(item['tokens'], dtype=torch.long) for item in batch])

  loader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=16,
    collate_fn=collate_fn,
  )

  batch = next(iter(loader))
  print(batch.shape)


def main():
  view_dataset()


if __name__ == '__main__':
  main()
