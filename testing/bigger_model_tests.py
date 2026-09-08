from __future__ import annotations

import gc
from types import SimpleNamespace

from transformers import AutoModelForCausalLM

from src.constants import DEFAULT_CONFIG, SCALING_LADDER
from src.models.builder import config_builder


MODEL_SIZES = ('0.6B', '1B')
ARCHITECTURES = ('attn', 'gdn', 'gdn+attn_1-3', 'gdn+attn_1-1', 'gdn+attn_3-1')


def _build_experiment_config(model_size: str, arch_id: str) -> SimpleNamespace:
  values = {
    **DEFAULT_CONFIG,
    **SCALING_LADDER['models'][model_size],
    'param_scale_id': model_size,
    'arch_id': arch_id,
  }
  return SimpleNamespace(**values)


def _count_parameters(model) -> tuple[int, int]:
  total_parameters = sum(parameter.numel() for parameter in model.parameters())
  embedding_parameters = sum(parameter.numel() for parameter in model.get_input_embeddings().parameters())
  return total_parameters, total_parameters - embedding_parameters


def test_initialisation() -> None:
  for model_size in MODEL_SIZES:
    for arch_id in ARCHITECTURES:
      cfg = _build_experiment_config(model_size, arch_id)
      model_config = config_builder(cfg)
      model = AutoModelForCausalLM.from_config(model_config)
      total_parameters, parameters_without_embedding = _count_parameters(model)

      print(f'{model_size} {arch_id}')
      print(f'  Total parameters: {total_parameters:,}')
      print(f'  Parameters excluding embedding table: {parameters_without_embedding:,}')

      del model
      gc.collect()


if __name__ == '__main__':
  test_initialisation()

