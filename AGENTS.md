# AGENTS.md


## Repository description

### Function

Custom codebase for creating, pretraining, and running scaling law experiments on
hybrid LLM architectures (interleaving Gated DeltaNet layers with dense (full softmax) Attention layers).

### Objective

- Find optimal hyper-parameter scaling laws for different ratios of Gated DeltaNet to Attention (called hybridisation ratio)
- Find loss-versus-compute curves for different hybridisation ratios to see which scales the best


## Tech stack
- Python 3.12, `pyproject.toml` for dependency management 
- `torch==2.10.0`
- `fla` layes and models in the `./fla` folder
- `triton` and `tilelang` and `flash-attn` for `fla` kernels
- `transformers==5.9.0`


## Don't
- Don't add comments that restate the code
- Don't use comments to put separators/headers between classes/functions
- Don't change or clean-up any code/file that is not relevant to the explicit task or change requested by the user
- Don't worry about the `./testing` folder: it does not contain tests, just scratch scripts to test discrete functions/plumbing
- Don't touch the `./results` folder
- Don't touch the `./execs` folder
- Don't touch the `cluster` folder


## Do
- Use `from __future__ import annotations` in every new file
- Use the `.agents` for skills and specific instructions for important tasks
- When editing anything inside `manager`, make sure that the desired interface of using `manager`-as given in `scripts/submit/train.sh` and `scripts/submit/eval.sh`can be used without change.
- When editing anything inside `src`, make sure that the plumbing into `experiments/train/pretrain.py` and management of experimental constants in `src/constants.py` is verified as correct.


