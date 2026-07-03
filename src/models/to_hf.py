import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
  CausalLMOutputWithPast,
)

from src.data.data_prep_utils import cu_seqlens_from_dense_attention_mask
from src.models.transformer import ModelConfig, Transformer


class HFModelConfig(PretrainedConfig):
  model_type = 'hybrid_transformer'

  def __init__(
    self,
    vocab_size=50304,
    seq_len=2048,
    dim=768,
    expand=3.0,
    n_layers=12,
    n_heads=8,
    mlp='glu',
    rmsnorm_eps=1e-6,
    tie_embeddings=False,
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
    **kwargs,
  ):
    super().__init__(**kwargs)

    self.vocab_size = vocab_size
    self.seq_len = seq_len
    self.dim = dim
    self.expand = expand
    self.n_layers = n_layers
    self.n_heads = n_heads

    self.mlp = mlp
    self.rmsnorm_eps = rmsnorm_eps
    self.tie_embeddings = tie_embeddings

    self.token_mixer = token_mixer
    self.hybrid_mixer_ratio = hybrid_mixer_ratio
    self.layer_norm_scaling = layer_norm_scaling
    self.residual_connection = residual_connection

    self.attn_gate = attn_gate
    self.attn_qk_norm = attn_qk_norm

    self.gdn_conv_size = gdn_conv_size
    self.gdn_gate = gdn_gate
    self.gdn_neg_eigval = gdn_neg_eigval

    self.intra_doc = intra_doc


def hf_to_internal_config(cfg: HFModelConfig):
  return ModelConfig(
    vocab_size=cfg.vocab_size,
    seq_len=cfg.seq_len,
    dim=cfg.dim,
    expand=cfg.expand,
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads,
    mlp=cfg.mlp,
    rmsnorm_eps=cfg.rmsnorm_eps,
    tie_embeddings=cfg.tie_embeddings,
    token_mixer=cfg.token_mixer,
    hybrid_mixer_ratio=cfg.hybrid_mixer_ratio,
    layer_norm_scaling=cfg.layer_norm_scaling,
    residual_connection=cfg.residual_connection,
    attn_gate=cfg.attn_gate,
    attn_qk_norm=cfg.attn_qk_norm,
    gdn_conv_size=cfg.gdn_conv_size,
    gdn_gate=cfg.gdn_gate,
    gdn_neg_eigval=cfg.gdn_neg_eigval,
    intra_doc=cfg.intra_doc,
  )


class HFModelForCausalLM(PreTrainedModel):
  config_class = HFModelConfig
  base_model_prefix = 'model'

  supports_gradient_checkpointing = False

  def __init__(self, config):
    super().__init__(config)

    self.model = Transformer(hf_to_internal_config(config))

    self.post_init()

  def get_input_embeddings(self):
    return self.model.embed_tokens

  def set_input_embeddings(self, value):
    self.model.embed_tokens = value

  def get_output_embeddings(self):
    return self.model.lm_head

  def set_output_embeddings(self, new_embeddings):
    self.model.lm_head = new_embeddings

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    labels=None,
    linear_mask=None,
    cu_seqlens=None,
    **kwargs,
  ):
    logits = self.model(
      x=input_ids,
      attention_mask=attention_mask,
      linear_mask=linear_mask,
      cu_seqlens=cu_seqlens,
    )

    loss = None

    if labels is not None:
      shift_logits = logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()

      loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
      )

    return CausalLMOutputWithPast(
      loss=loss,
      logits=logits,
    )

  def prepare_inputs_for_generation(
    self,
    input_ids,
    attention_mask=None,
    **kwargs,
  ):
    linear_mask, cu_seqlens = None, None
    gdn_present = 'gdn' in self.model.cfg.token_mixer
    intra_doc = self.model.cfg.intra_doc

    if attention_mask is not None:
      if gdn_present:
        linear_mask = torch.ones_like(attention_mask.shape[:-1], dtype=torch.bool, device=input_ids.shape)
      if intra_doc:
        linear_mask, cu_seqlens = cu_seqlens_from_dense_attention_mask(attention_mask, device=input_ids.device)

    return {
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'linear_mask': linear_mask,
      'cu_seqlens': cu_seqlens,
    }


def load_checkpoint_into_hf(hf_model, ckpt_path):
  state = torch.load(ckpt_path, map_location='cpu')

  if 'state_dict' in state:
    state = state['state_dict']

  missing, unexpected = hf_model.model.load_state_dict(
    state,
    strict=True,
  )

  print('missing:', missing)
  print('unexpected:', unexpected)

  return hf_model
