import typing as tp
from time import perf_counter

import torch

from fla.layers import GatedDeltaNet
from src.models.components import RMSNorm

DEVICE = 'cuda:0'
DTYPE = torch.bfloat16


def setup():
  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def test_gdn_forward_pass():
  start = perf_counter()
  layer = GatedDeltaNet(hidden_size=64, num_heads=4, head_dim=16, use_gate=True, allow_neg_eigval=True, layer_idx=0).to(
    device=DEVICE,
    dtype=DTYPE,
  )
  end = perf_counter()
  print('GDN init took', end - start, 'seconds')

  x = torch.randn(2, 4, 64).to(device=DEVICE, dtype=DTYPE)
  start = perf_counter()
  y = layer(x)
  end = perf_counter()
  print('GDN forward pass took', end - start, 'seconds')

  if isinstance(y, tp.Iterable):
    for item in y:
      if item is not None:
        print(item.shape)

  print(' ')


@torch.inference_mode()
def test_gdn_compile():
  device, dtype = 'cuda', torch.bfloat16
  start = perf_counter()
  layer = GatedDeltaNet(hidden_size=64, num_heads=4, head_dim=16, use_gate=True, allow_neg_eigval=True, layer_idx=0).to(
    device=DEVICE,
    dtype=DTYPE,
  )
  end = perf_counter()
  print('GDN init took', end - start, 'seconds')

  start = perf_counter()
  layer = torch.compile(layer)
  end = perf_counter()
  print('GDN compilation took', end - start, 'seconds')

  x = torch.randn(2, 4, 64).to(device=DEVICE, dtype=DTYPE)
  start = perf_counter()
  y = layer(x)
  end = perf_counter()
  print('GDN compiled forward pass took', end - start, 'seconds')


def main():
  setup()
  test_gdn_forward_pass()
  test_gdn_compile()


if __name__ == '__main__':
  main()
