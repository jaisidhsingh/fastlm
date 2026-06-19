from transformers import AutoConfig, AutoModelForCausalLM

from src.models.to_hf import HybridTransformerConfig, HybridTransformerForCausalLM

AutoConfig.register(
  'hybrid_transformer',
  HybridTransformerConfig,
)

AutoModelForCausalLM.register(
  HybridTransformerConfig,
  HybridTransformerForCausalLM,
)
