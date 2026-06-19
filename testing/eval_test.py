from absl import app, flags
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from src.models.construct import construct_hf_config
from src.models.to_hf import HFModelConfig, HFModelForCausalLM, load_checkpoint_into_hf
from src.utils import load_config


def check_hf_model_init(cfg, device):
  hf_cfg = construct_hf_config(cfg)
  assert isinstance(hf_cfg, HFModelConfig)

  model = HFModelForCausalLM(hf_cfg)
  model = model.to(dtype=torch.bfloat16, device=device)
  print(model.model.count_params(non_embedding=False))


def create_hf_checkpoint(cfg, ckpt_path, out_path, device):
  hf_cfg = construct_hf_config(cfg)
  assert isinstance(hf_cfg, HFModelConfig)
  
  model = HFModelForCausalLM(hf_cfg)
  model = model.to(dtype=torch.bfloat16, device=device)
  print(model.model.count_params(non_embedding=False))
  
  load_checkpoint_into_hf(
    model,
    ckpt_path
  )
  model.save_pretrained(out_path)
  hf_cfg.save_pretrained(out_path)

  return model


def load_tokenizer():
  return AutoTokenizer.from_pretrained("/fast/jsingh/saved_tokenizers/better-gpt2")


@torch.no_grad()
def test_hf_model_on_one_hellaswag_sample(model, tokenizer, device):
  hellaswag_path = "asdf"
  dataset = load_from_disk(hellaswag_path)
  sample = dataset[0]

  ctx = sample['ctx']
  endings = sample['endings']
  label = int(sample['label'])

  scores = []
  for ending in endings:
    prompt = ctx + " " + ending
    inputs = tokenizer(prompt, return_tensor="pt", truncation=True, max_length=2048)
    inputs = {k:v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    logits = out.logits[:, :-1, :]
    labels = inputs['input_ids'][:, 1:]

    logprobs = torch.log_softmax(logits, dim=-1)
    sequence_logprobs = logprobs.gather(-1, labels).unsqueeze(-1).squeeze(-1)
    scores.append(sequence_logprobs.sum().item())

  pred = max(range(4), key=lambda i: scores[i])
  print("Context:", ctx)
  print(" ")
  for i, e in enumerate(endings):
    print(i+1, ":", e)
  print(" ")
  print("Prediction:", pred)
  print("Ground truth:", label) 
         

def main(argv):
  cfg = None
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  ckpt_path = "asdf"
  out_path = "asdf"

  check_hf_model_init(cfg, device)
  model = create_hf_checkpoint(cfg, ckpt_path, out_path, device)
  tokenizer = load_tokenizer()

  test_hf_model_on_one_hellaswag_sample(model, tokenizer, device)


if __name__ == '__main__':
  app.run(main)

