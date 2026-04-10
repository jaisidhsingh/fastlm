import torch

from models.transformer import ModelConfig, Transformer

DEVICE = 'cuda:0'
DTYPE = torch.bfloat16


def setup():
  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def test_pure_attn_lm_forward_pass():
  cfg = ModelConfig(
    dim=64,
    vocab_size=1024,
    seq_len=16,
    expand=2.0,
    n_layers=8,
    n_heads=4,
    token_mixer='attn',
    attn_gate=True,
    attn_qk_norm=True,
  )
  model = Transformer(cfg).to(device=DEVICE, dtype=DTYPE)
  x = torch.randn(2, 16, 64).to(device=DEVICE, dtype=DTYPE)
  y = model(x)
  print(y.shape)


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
  model = Transformer(cfg).to(device=DEVICE, dtype=DTYPE)
  x = torch.randn(2, 16, 64).to(device=DEVICE, dtype=DTYPE)
  y = model(x)
  print(y.shape)


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
  model = Transformer(cfg).to(device=DEVICE, dtype=DTYPE)
  x = torch.randn(2, 16, 64).to(device=DEVICE, dtype=DTYPE)
  y = model(x)
  print(y.shape)


main():
  setup()
  test_pure_attn_lm_forward_pass()
  test_pure_gdn_lm_forward_pass()
  test_hybrid_lm_forward_pass()


if __name__ == "__main__":
  main()
