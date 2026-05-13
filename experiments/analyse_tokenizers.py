import multiprocessing as mp
import os
import typing as tp

import datasets
import tiktoken
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from transformers import AutoTokenizer

mp.set_start_method('spawn', force=True)


def get_tokenizer(tokenizer_id: str):
  tokenizer_folder = os.path.join('/lustre/home/jsingh/projects/fastlm/tokenizer', tokenizer_id)

  if tokenizer_id == 'mistral-v3':
    tokenizer = Tekkenizer.from_file(os.path.join(tokenizer_folder, 'tekken.json'))

  elif tokenizer_id == 'gpt-neox' or tokenizer_id == 'better-gpt2':
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder)

  elif tokenizer_id == 'gpt4':
    tokenizer = tiktoken.encoding_for_model('gpt-4o')

  else:
    tokenizer = None
    raise NotImplementedError('The tokenizer you specified is not currently supported')

  return tokenizer


def map_ds(tokenizer, sample, tokenizer_kwargs):
  tokens = tokenizer.encode(sample['text'], **tokenizer_kwargs)
  return {'input_ids': tokens}


def main():
  nemotron_cc_sample_path = '/fast/jsingh/data/nemotron-cc-sample-mtsynth/raw_dataset'
  shard_fname = 'data-00200-of-01194.arrow'
  dataset = datasets.Dataset.from_file(os.path.join(nemotron_cc_sample_path, shard_fname))
  dataset = dataset.take(40000)

  tokenizer_ids = ['mistral-v3', 'gpt-neox', 'better-gpt2', 'gpt4']
  tokenizer_ids.reverse()
  tokenizer_kwargs = {
    'mistral-v3': {'bos': True, 'eos': True},
    'gpt-neox': {'add_special_tokens': True},
    'better-gpt2': {'add_special_tokens': True},
    'gpt4': {},
  }

  for tokenizer_id in tokenizer_ids:
    tokenizer = get_tokenizer(tokenizer_id)
    print(tokenizer_id)
    if hasattr(tokenizer, '__len__'):
      print(len(tokenizer))
    elif hasattr(tokenizer, 'vocab_size'):
      print(tokenizer.vocab_size)
    elif hasattr(tokenizer, 'n_vocab'):
      print(tokenizer.n_vocab)
    elif hasattr(tokenizer, 'n_words'):
      print(tokenizer.n_words)

  tokenized_dataset = dataset.map(
    lambda sample: map_ds(tokenizer, sample, tokenizer_kwargs[tokenizer_id]), remove_columns=['text']
  )

  avg = 0
  n = len(tokenized_dataset)
  for i in range(n):
    avg += len(tokenized_dataset[i]['input_ids'])
  avg /= n
  print('For tokenizer', tokenizer_id, 'avg sequence length ->', avg)


if __name__ == '__main__':
  main()
