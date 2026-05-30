import copy
from pathlib import Path
import yaml


# Default config template mirroring config/attn/small_scale/*.yaml
DEFAULT_TEMPLATE = {
  'deterministic': False,
  'seed': 123,
  # DATA
  'trainset_path': '/fast/jsingh/data/nemotron-cc-sample-mtsynth/tokenized_gpt2/ctx_2048/train',
  'vocab_size': 50304,
  'seq_len': 2048,
  'intra_doc_masking': False,
  'sampler': 'sequential',
  'sampler_seed': None,
  'num_workers': 4,
  'eval': True,
  'validset_path': '/fast/jsingh/data/nemotron-cc-sample-mtsynth/tokenized_gpt2/ctx_2048/valid',
  'eval_every_steps': 50,
  # MODEL
  'model': 'transformer',
  'mlp_class': 'glu',
  'expand': '3.0',
  'rms_norm': True,
  'tie_embeddings': False,
  'torch_compile': True,
  'token_mixer': 'attn',
  'hybrid_mixer_ratio': 1,
  'layer_norm_scaling': False,
  'residual_connection': 'add',
  'attn_gate': True,
  'attn_qk_norm': True,
  'gdn_conv_size': 4,
  'gdn_gate': True,
  'gdn_neg_eigval': True,
  # TRAINING (set per run)
  'steps_budget': -1,
  'chinchilla_token_multiplier': 2.5,
  'micro_batch_size': 1,
  'grad_accumulation_steps': 1,
  'dtype': 'bfloat16',
  # OPTIMIZER (set per run)
  'optim': 'adamw',
  'fused_optim': True,
  'lr': 0.01,
  'weight_decay': 0.1,
  'beta1': 0.9,
  'beta2': 0.95,
  'grad_clip': 1.0,
  # SCHEDULER
  'scheduler': 'wsd',
  'warmup_steps': 2000,
  'cooldown_steps': 0,
  'lr_start': 0.0,
  'lr_end': 1e-5,
  'lr_end_pct': None,
  # EXPERIMENT
  'log_every_steps': 1,
  'print_progress': True,
  'use_wandb': False,
  'wandb_mode': 'online',
  'wandb_project': 'scaling-law',
  'wandb_entity': 'msc-thesis-jaisidh',
  'wandb_dir': '/fast/jsingh/projects/fastlm/wandb',
  'wandb_run_name': '',
  'exp_name': '',
  'out_dir': '/fast/jsingh/projects/fastlm/experiment_logs',
  'over_write': True,
  'resume': False,
  'resume_step': None,
  'resume_exp_name': None,
  'save_last_checkpoint': False,
  'save_intermediate_checkpoints': False,
  'save_every_steps': 500,
}


def build_config(
  arch,  # dict with keys: d_model, n_layers, n_heads
  mbs,  # micro batch size
  gas,  # gradient accumulation steps
  steps_budget,  # total training steps
  eta,  # learning rate
  warmup_steps=2000,
  token_mixer='attn',
  hybrid_ratio=1,
  token_mixer_pattern=None,
):
  """Build a complete config dictionary from the structural knobs.

  Parameters
  ----------
  arch : dict
      Architecture parameters: ``d_model``, ``n_layers``, ``n_heads``.
  mbs : int
      Micro batch size per GPU.
  gas : int
      Gradient accumulation steps.
  steps_budget : int
      Total number of optimizer steps.
  eta : float
      Peak learning rate.
  warmup_steps : int
      Number of linear-warmup steps (default 2000 per strategy Sec. 2).
  token_mixer : str
      ``"attn"`` or ``"gdn+attn"``.
  hybrid_ratio : int
      Hybrid mixer ratio (only used for ``"gdn+attn"``).
  token_mixer_pattern : str or None
      Explicit per-layer pattern string.

  Returns
  -------
  dict
      Complete config as a nested dictionary.
  """
  cfg = copy.deepcopy(DEFAULT_TEMPLATE)
  cfg.update(arch)

  cfg['token_mixer'] = token_mixer
  cfg['hybrid_mixer_ratio'] = hybrid_ratio
  if token_mixer_pattern is not None:
    cfg['token_mixer_pattern'] = token_mixer_pattern

  cfg['micro_batch_size'] = mbs
  cfg['grad_accumulation_steps'] = gas
  cfg['steps_budget'] = steps_budget
  cfg['scheduler'] = 'wsd'
  cfg['warmup_steps'] = warmup_steps
  cfg['cooldown_steps'] = 0
  cfg['lr_start'] = 0.0
  cfg['lr'] = eta
  cfg['use_wandb'] = False

  return cfg


def write_config_to_disk(config_dict, output_path):
  """Write the config dictionary to a YAML file.

  Parameters
  ----------
  config_dict : dict
      Config from :func:`build_config`.
  output_path : str or Path
      Target ``.yaml`` file path.

  Returns
  -------
  Path
      The output path.
  """
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, 'w') as f:
    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
  return output_path
