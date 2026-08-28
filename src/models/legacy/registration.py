from transformers import AutoConfig, AutoModelForCausalLM

from src.models.to_hf import HFModelConfig, HFModelForCausalLM

AutoConfig.register(
  'hybridlm_hf',
  HFModelConfig,
)

AutoModelForCausalLM.register(
  HFModelConfig,
  HFModelForCausalLM,
)
