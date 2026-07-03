import os
import tempfile

import torch

from src.models.to_hf import (
  HFModelConfig,
  HFModelForCausalLM,
  hf_to_internal_config,
  load_checkpoint_into_hf,
)
from src.models.transformer import ModelConfig, Transformer

SEED = 0
DEVICE = 'cpu'
MODEL_DTYPE = torch.float32
BATCH_SIZE = 2
SEQ_LEN = 16


def setup():
  torch.manual_seed(SEED)


def test_hf_config_roundtrip():
  """Test that converting HFModelConfig -> ModelConfig -> HFModelConfig is lossless."""
  hf_cfg = HFModelConfig(
    vocab_size=1024,
    seq_len=32,
    dim=128,
    expand=2.0,
    n_layers=4,
    n_heads=4,
    mlp='glu',
    rmsnorm_eps=1e-6,
    tie_embeddings=True,
    token_mixer='gdn+attn',
    hybrid_mixer_ratio=3,
    layer_norm_scaling=False,
    residual_connection='add',
    attn_gate=True,
    attn_qk_norm=True,
    gdn_conv_size=4,
    gdn_gate=True,
    gdn_neg_eigval=True,
    intra_doc=False,
  )

  internal_cfg = hf_to_internal_config(hf_cfg)
  assert isinstance(internal_cfg, ModelConfig), f'Expected ModelConfig, got {type(internal_cfg)}'

  # Verify key fields match
  for field in [
    'vocab_size',
    'seq_len',
    'dim',
    'expand',
    'n_layers',
    'n_heads',
    'mlp',
    'rmsnorm_eps',
    'tie_embeddings',
    'token_mixer',
    'hybrid_mixer_ratio',
    'layer_norm_scaling',
    'residual_connection',
    'attn_gate',
    'attn_qk_norm',
    'gdn_conv_size',
    'gdn_gate',
    'gdn_neg_eigval',
    'intra_doc',
  ]:
    assert getattr(internal_cfg, field) == getattr(hf_cfg, field), (
      f'Mismatch on {field}: {getattr(internal_cfg, field)} != {getattr(hf_cfg, field)}'
    )

  print('✓ HF config roundtrip test passed')


def test_hf_model_creation():
  """Test that HFModelForCausalLM can be created with HFModelConfig."""
  hf_cfg = HFModelConfig(
    vocab_size=1024,
    seq_len=32,
    dim=128,
    expand=2.0,
    n_layers=4,
    n_heads=4,
    mlp='glu',
    token_mixer='attn',
    attn_gate=True,
    attn_qk_norm=True,
  )

  hf_model = HFModelForCausalLM(hf_cfg)
  hf_model.eval()

  assert isinstance(hf_model, HFModelForCausalLM), f'Expected HFModelForCausalLM, got {type(hf_model)}'
  assert hf_model.config_class == HFModelConfig

  total_params = hf_model.model.count_params(non_embedding=False)
  print(f'  HF model total params: {total_params:_}')
  print('✓ HF model creation test passed')


@torch.inference_mode()
def test_hf_model_forward():
  """Test that HFModelForCausalLM can do a forward pass."""
  hf_cfg = HFModelConfig(
    vocab_size=1024,
    seq_len=32,
    dim=128,
    expand=2.0,
    n_layers=4,
    n_heads=4,
    mlp='glu',
    token_mixer='attn',
    attn_gate=True,
    attn_qk_norm=True,
  )

  hf_model = HFModelForCausalLM(hf_cfg)
  hf_model.eval()

  input_ids = torch.randint(0, hf_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN))

  with torch.inference_mode():
    out = hf_model(input_ids=input_ids)

  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, hf_cfg.vocab_size), (
    f'Expected {(BATCH_SIZE, SEQ_LEN, hf_cfg.vocab_size)}, got {out.logits.shape}'
  )
  assert out.loss is None  # No labels provided

  # Test with labels
  labels = torch.randint(0, hf_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN))
  with torch.inference_mode():
    out_with_loss = hf_model(input_ids=input_ids, labels=labels)

  assert out_with_loss.loss is not None
  assert out_with_loss.loss.ndim == 0  # scalar loss

  print('✓ HF model forward pass test passed')


@torch.inference_mode()
def test_load_checkpoint_into_hf():
  """Test that an internal Transformer's state dict can be loaded into HFModelForCausalLM."""
  # Create an internal model and save its state
  internal_cfg = ModelConfig(
    vocab_size=1024,
    seq_len=32,
    dim=128,
    expand=2.0,
    n_layers=4,
    n_heads=4,
    mlp='glu',
    token_mixer='attn',
    attn_gate=True,
    attn_qk_norm=True,
  )
  internal_model = Transformer(internal_cfg)
  internal_model.eval()

  # Do a forward pass to generate some gradients and ensure all params are used
  x = torch.randint(0, internal_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN))
  with torch.inference_mode():
    y_internal = internal_model(x)

  # Create an HF model with matching config
  hf_cfg = HFModelConfig(
    vocab_size=1024,
    seq_len=32,
    dim=128,
    expand=2.0,
    n_layers=4,
    n_heads=4,
    mlp='glu',
    token_mixer='attn',
    attn_gate=True,
    attn_qk_norm=True,
  )
  hf_model = HFModelForCausalLM(hf_cfg)
  hf_model.eval()

  # Save internal state dict to a temp file, then load into HF model
  with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
    ckpt_path = tmp.name
    torch.save(internal_model.state_dict(), ckpt_path)

  try:
    load_checkpoint_into_hf(hf_model, ckpt_path)

    # Verify outputs match after loading
    with torch.inference_mode():
      y_hf = hf_model.model(x)

    assert torch.allclose(y_internal, y_hf, atol=1e-5), (
      f'Outputs do not match! Max diff: {(y_internal - y_hf).abs().max().item()}'
    )

    print('✓ Load checkpoint into HF model test passed')
  finally:
    os.unlink(ckpt_path)


@torch.inference_mode()
def test_load_checkpoint_from_dict():
  """Test loading from a dict with 'state_dict' key (matching checkpoint format)."""
  internal_cfg = ModelConfig(
    vocab_size=1024,
    seq_len=32,
    dim=128,
    expand=2.0,
    n_layers=4,
    n_heads=4,
    mlp='glu',
    token_mixer='attn',
    attn_gate=True,
    attn_qk_norm=True,
  )
  internal_model = Transformer(internal_cfg)
  internal_model.eval()

  x = torch.randint(0, internal_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN))
  with torch.inference_mode():
    y_internal = internal_model(x)

  hf_cfg = HFModelConfig(
    vocab_size=1024,
    seq_len=32,
    dim=128,
    expand=2.0,
    n_layers=4,
    n_heads=4,
    mlp='glu',
    token_mixer='attn',
    attn_gate=True,
    attn_qk_norm=True,
  )
  hf_model = HFModelForCausalLM(hf_cfg)
  hf_model.eval()

  # Save in checkpoint format (wrapped in dict with 'state_dict' key)
  with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
    ckpt_path = tmp.name
    torch.save({'state_dict': internal_model.state_dict()}, ckpt_path)

  try:
    load_checkpoint_into_hf(hf_model, ckpt_path)

    with torch.inference_mode():
      y_hf = hf_model.model(x)

    assert torch.allclose(y_internal, y_hf, atol=1e-5), (
      f'Outputs do not match! Max diff: {(y_internal - y_hf).abs().max().item()}'
    )

    print('✓ Load checkpoint from dict test passed')
  finally:
    os.unlink(ckpt_path)


@torch.inference_mode()
def test_hf_model_generation_prepare_inputs():
  """Test that prepare_inputs_for_generation works correctly."""
  hf_cfg = HFModelConfig(
    vocab_size=1024,
    seq_len=32,
    dim=128,
    expand=2.0,
    n_layers=4,
    n_heads=4,
    mlp='glu',
    token_mixer='attn',
    attn_gate=True,
    attn_qk_norm=True,
  )
  hf_model = HFModelForCausalLM(hf_cfg)

  input_ids = torch.randint(0, hf_cfg.vocab_size, (BATCH_SIZE, 1))
  attention_mask = torch.ones(BATCH_SIZE, 1, dtype=torch.long)

  prepared = hf_model.prepare_inputs_for_generation(
    input_ids=input_ids,
    attention_mask=attention_mask,
  )

  assert 'input_ids' in prepared
  assert 'attention_mask' in prepared
  # For pure attn, linear_mask and cu_seqlens should be None
  assert prepared['linear_mask'] is None
  assert prepared['cu_seqlens'] is None

  print('✓ HF model generation prepare_inputs test passed')


def main():
  setup()

  print('=== Testing HF Model Conversion ===\n')
  test_hf_config_roundtrip()
  test_hf_model_creation()
  test_hf_model_forward()
  test_load_checkpoint_into_hf()
  test_load_checkpoint_from_dict()
  test_hf_model_generation_prepare_inputs()

  print('\n=== All tests passed! 🎉 ===')


if __name__ == '__main__':
  main()
