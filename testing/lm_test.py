import random

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data.datasamplers import StatefulRandomSampler
from models.transformer import ModelConfig, Transformer
from utils import get_chincilla_details

SEED = 0
DEVICE = 'cuda:0'
MODEL_DTYPE = torch.bfloat16
INPUT_DTYPE = torch.long
BATCH_SIZE = 2
SEQ_LEN = 16
DIM = 64
EXPAND = 2.0
N_HEADS = 4
DEPTH = 6
VOCAB_SIZE = 1024
NUM_STEPS = 200
LR = 3e-4

OWT_PATH = '/fast/jsingh/data/owt-tokenized-9b-train-nn'
TOKENIZER_PATH = '/home/jsingh/projects/fastlm/tokenizer/better-gpt2'


def setup():
  random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def test_pure_attn_lm_forward_pass():
  cfg = ModelConfig(
    dim=DIM,
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    expand=EXPAND,
    n_layers=DEPTH,
    n_heads=N_HEADS,
    token_mixer='attn',
    attn_gate=True,
    attn_qk_norm=True,
  )
  model = Transformer(cfg).to(device=DEVICE, dtype=MODEL_DTYPE)
  print(model)

  x = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN)).to(device=DEVICE, dtype=INPUT_DTYPE)
  attention_mask = (
    torch.tril(torch.ones(x.shape[1], x.shape[1])).repeat(BATCH_SIZE, 1, 1).to(device=DEVICE, dtype=torch.bool)
  )
  y = model(x, attention_mask)
  print(y.shape)
  print('Pure gated attention forward pass successful\n')


@torch.inference_mode()
def test_pure_gdn_lm_forward_pass():
  cfg = ModelConfig(
    dim=64,
    vocab_size=1024,
    seq_len=16,
    expand=2.0,
    n_layers=8,
    n_heads=4,
    token_mixer='gdn',
    gdn_conv_size=4,
    gdn_gate=True,
    gdn_neg_eigval=True,
  )
  model = Transformer(cfg).to(device=DEVICE, dtype=MODEL_DTYPE)
  print(model)

  x = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN)).to(device=DEVICE, dtype=INPUT_DTYPE)
  attention_mask = torch.ones(x.shape).to(device=DEVICE, dtype=torch.bool)
  y = model(x, attention_mask)
  print(y.shape)
  print('Pure gated deltanet forward pass successful\n')


@torch.inference_mode()
def test_hybrid_lm_forward_pass():
  cfg = ModelConfig(
    dim=64,
    vocab_size=1024,
    seq_len=16,
    expand=2.0,
    n_layers=8,
    n_heads=4,
    token_mixer='gdn+attn',
    hybrid_mixer_ratio=3,
    attn_gate=True,
    attn_qk_norm=True,
    gdn_conv_size=4,
    gdn_gate=True,
    gdn_neg_eigval=True,
  )
  model = Transformer(cfg).to(device=DEVICE, dtype=MODEL_DTYPE)
  print(model)

  x = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN)).to(device=DEVICE, dtype=INPUT_DTYPE)
  attention_mask = (
    torch.tril(torch.ones(x.shape[1], x.shape[1])).repeat(BATCH_SIZE, 1, 1).to(device=DEVICE, dtype=torch.bool)
  )
  y = model(x, attention_mask)
  print(y.shape)
  print('Hybrid 3:1 [gated deltanet : gated attention] forward pass successful\n')


def overfit_one_dummy_batch(token_mixer):
  cfg = ModelConfig(
    dim=64,
    vocab_size=1024,
    seq_len=16,
    expand=2.0,
    n_layers=8,
    n_heads=4,
    token_mixer=token_mixer,
    hybrid_mixer_ratio=3,
    attn_gate=True,
    attn_qk_norm=True,
    gdn_conv_size=4,
    gdn_gate=True,
    gdn_neg_eigval=True,
  )
  model = Transformer(cfg).to(device=DEVICE, dtype=MODEL_DTYPE)
  print(model)
  model = torch.compile(model)

  x = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN)).to(device=DEVICE, dtype=INPUT_DTYPE)
  attention_mask = (
    torch.tril(torch.ones(x.shape[1], x.shape[1])).repeat(BATCH_SIZE, 1, 1).to(device=DEVICE, dtype=torch.bool)
  )
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

  bar = tqdm(total=NUM_STEPS)
  init_loss = 0.0

  for i in range(NUM_STEPS):
    optimizer.zero_grad()

    with torch.amp.autocast(DEVICE):
      y = model(x, attention_mask)

    vocab_size = y.shape[-1]
    labels = x[:, 1:].reshape(-1).contiguous()
    logits = y[:, :-1, :].reshape(-1, vocab_size).contiguous()

    loss = torch.nn.functional.cross_entropy(logits, labels)
    bar.set_postfix({'loss': round(loss.item(), 4)})
    if i == 0:
      init_loss = round(loss.item(), 4)

    loss.backward()
    optimizer.step()

    bar.update(1)
  bar.close()

  final_loss = round(loss.item(), 4)
  print(f'Loss at init. = {init_loss}. Loss after {NUM_STEPS} = {final_loss}')
  print(f'Overfitting one dummy batch for [{token_mixer}] successful\n')


def overfit_one_nlp_batch(token_mixer):
  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
  cfg = ModelConfig(
    dim=128,
    vocab_size=len(tokenizer),
    seq_len=1024,
    expand=2.0,
    n_layers=6,
    n_heads=4,
    token_mixer=token_mixer,
    hybrid_mixer_ratio=3,
    attn_gate=True,
    attn_qk_norm=True,
    gdn_conv_size=4,
    gdn_gate=True,
    gdn_neg_eigval=True,
  )
  model = Transformer(cfg).to(device=DEVICE, dtype=MODEL_DTYPE)
  non_embedding_params = model.count_params()
  total_params = model.count_params(non_embedding=False)
  print(f'Total params: {total_params}, Non-embedding params: {non_embedding_params}')

  details = get_chincilla_details(total_params)
  num_rows = details['token_count'] // 1024

  dataset = load_from_disk(OWT_PATH)
  indices = random.sample(range(len(dataset)), num_rows)
  dataset = dataset.select(indices)
  print(num_rows, len(dataset))
  sampler = StatefulRandomSampler(data_source=dataset, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)

  def collate_fn(batch):
    return torch.stack([torch.tensor(item['tokens'], dtype=torch.long) for item in batch])

  loader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
  )

  print(model)
  model = torch.compile(model)

  batch = next(iter(loader)).to(device=DEVICE, dtype=INPUT_DTYPE)
  attention_mask = (
    torch.tril(torch.ones(batch.shape[1], batch.shape[1]))
    .repeat(batch.shape[0], 1, 1)
    .to(device=DEVICE, dtype=torch.bool)
  )
  print(batch.shape, batch.device, batch.dtype)
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

  bar = tqdm(total=NUM_STEPS)
  init_loss = 0.0

  for i in range(NUM_STEPS):
    optimizer.zero_grad()

    with torch.amp.autocast(DEVICE):
      logits = model(batch, attention_mask)

    vocab_size = logits.shape[-1]
    labels = batch.clone()[:, 1:].reshape(-1).contiguous()
    logits = logits[:, :-1, :].reshape(-1, vocab_size).contiguous()

    loss = torch.nn.functional.cross_entropy(logits, labels)
    bar.set_postfix({'loss': round(loss.item(), 4)})
    if i == 0:
      init_loss = round(loss.item(), 4)

    loss.backward()
    optimizer.step()

    bar.update(1)
  bar.close()

  final_loss = round(loss.item(), 4)
  print(f'Loss at init. = {init_loss}. Loss after {NUM_STEPS} = {final_loss}')
  print(f'Overfitting one NLP batch for [{token_mixer}] successful\n')


def main():
  setup()
  test_pure_attn_lm_forward_pass()
  test_pure_gdn_lm_forward_pass()
  test_hybrid_lm_forward_pass()
  for token_mixer in ['attn', 'gdn', 'gdn+attn']:
    overfit_one_dummy_batch(token_mixer)
    overfit_one_nlp_batch(token_mixer)


if __name__ == '__main__':
  main()
