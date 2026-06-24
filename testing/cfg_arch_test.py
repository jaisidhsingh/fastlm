# import random
# import time

import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from absl import app, flags
from transformers import AutoTokenizer

import src.utils as utils
from src.checkpoint_utils import create_save_steps
from src.data import get_dataloaders
from src.models import construct_model
from src.utils import load_config

flags.DEFINE_string(
  'config',
  '/lustre/home/jsingh/projects/fastlm/execs/attn/300M/tmp.yaml',
  'Path to config.yaml file.',
)
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')

FLAGS = flags.FLAGS
SEED = 42


# def seed():
#   random.seed(SEED)
#   np.random.seed(SEED)
#   torch.manual_seed(SEED)
#   torch.cuda.manual_seed(SEED)
#   torch.cuda.manual_seed_all(SEED)
#   torch.backends.cudnn.deterministic = True
#   torch.backends.cudnn.benchmark = False
#   torch.use_deterministic_algorithms(True)


def main(argv):
  # seed()
  device = 'cuda'
  cfg, _ = load_config(FLAGS.config)
  utils.set_batch_sizes(cfg, 1)

  tokenizer = AutoTokenizer.from_pretrained('/fast/jsingh/saved_tokenizers/better-gpt2')
  print('bos', tokenizer.bos_token, tokenizer.bos_token_id)
  print('eos', tokenizer.eos_token, tokenizer.eos_token_id)

  trainloader, validloader = get_dataloaders(cfg)

  num_dls = 0
  doc_ls = 0
  for i in range(1000):
    sample = trainloader.dataset[i]
    toks = np.asarray(sample['input_ids'])
    num_dls += np.where(toks == tokenizer.eos_token_id, 1, 0).sum()
    doc_ls += len(sample['docs_lengths'].tolist())

  print('avg no. of eos tokens', num_dls / 1000)
  print('avg no. of docs per seq', doc_ls / 1000)

  # steps_budget = utils.get_steps_budget(cfg, 1)
  # cfg.steps_budget = steps_budget

  # pdict, toks = create_save_steps(cfg, 1)
  # for k, v in pdict.items():
  #   print(k, v)
  # print(toks)

  # seq_len = cfg.seq_len

  # x = torch.randint(
  #   0,
  #   cfg.vocab_size,
  #   (1, cfg.seq_len),
  #   dtype=torch.long,
  #   device=device,
  # )
  # attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(0)

  # store = []
  # for setting in [False]:
  #   profiler = torch.profiler.profile(
  #     activities=[torch.profiler.ProfilerActivity.CUDA],
  #     profile_memory=False,
  #     with_stack=False,
  #     record_shapes=False,
  #   )
  #   ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
  #   cfg.use_fast_gatedattention = setting

  #   model, model_config = construct_model(cfg)
  #   print(model)
  #   model = model.to(dtype=torch.bfloat16, device=device)
  #   model = torch.compile(model)
  #   optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

  #   # with profiler as prof:
  #   times = []
  #   for _ in range(20):
  #     torch.cuda.synchronize()
  #     start = time.perf_counter()
  #     optim.zero_grad()
  #     with ctx:
  #       out = model(x, attn_mask)
  #       loss = F.cross_entropy(out[:, :-1, :].view(-1, out.shape[-1]).contiguous(), x[:, 1:].view(-1).contiguous())
  #     loss.backward()
  #     optim.step()
  #     torch.cuda.synchronize()
  #     end = time.perf_counter()
  #     times.append(end - start)

  #   print(sum(times) / 20)


if __name__ == '__main__':
  app.run(main)
