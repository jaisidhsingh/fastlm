from transformers import AutoConfig, AutoModelForCausalLM

from src.models.to_hf import HybridTransformerConfig, HybridTransformerForCausalLM

AutoConfig.register(
  'fastlm_hf',
  HFModelConfig,
)

AutoModelForCausalLM.register(
  HFModelConfig,
  HFModelForCausalLM,
)
