from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.flop_counter import FlopCounterMode

from src.data.data_prep_utils import intra_doc_causal_mask, intra_doc_masking_linear
from src.models import get_param_groups
from src.optim import initialize_scheduler, intialize_optimizer
from src.optim.lr_schedule import LinearCooldown

try:
  from torch.nn.attention.flex_attention import BlockMask, create_block_mask

  _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
  _FLEX_ATTENTION_AVAILABLE = False


def _build_flex_block_mask(docs_lengths_batch, seq_len, device):
  """Build a BlockMask for intra-document causal attention using flex_attention.

  Creates a compiled block-diagonal causal mask from document boundaries.
  This is functionally identical to the dense mask from intra_doc_causal_mask,
  but uses flex_attention's block-sparse representation for FlashAttention compatibility.

  Args:
    docs_lengths_batch: list of lists of document lengths per example.
      Each inner list contains the token counts for each document in that example.
      Lengths sum to seq_len + 1 (extra token for target shift).
    seq_len: sequence length (mask will be seq_len × seq_len).
    device: torch device to create the mask on.

  Returns:
    BlockMask object for use with flex_attention.
  """
  bsz = len(docs_lengths_batch)

  # Build document ID tensor: (bsz, seq_len)
  # Each position maps to the index of the document it belongs to.
  doc_ids = torch.zeros(bsz, seq_len, dtype=torch.int32, device=device)
  for b in range(bsz):
    pos = 0
    for doc_id, length in enumerate(docs_lengths_batch[b]):
      end = min(pos + length, seq_len)
      if pos >= seq_len:
        break
      doc_ids[b, pos:end] = doc_id
      pos += length

  # Mask function: causal AND same-document
  # This gets compiled by create_block_mask into an efficient block-sparse representation
  def intra_doc_causal_mask_fn(b, h, q_idx, kv_idx):
    causal = q_idx >= kv_idx
    same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
    return causal & same_doc

  block_mask = create_block_mask(
    intra_doc_causal_mask_fn, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=True
  )
  return block_mask


def _move_to_device(batch, seq_len, device, intra_doc_masking, use_flex_attention=True):
  """Slice batch to get inputs and targets, and move them to device."""
  bsz = batch['input_ids'].shape[0]

  inputs = batch['input_ids'][:, :seq_len]  # WE WILL CHOP OFF THE LAST ONE FROM THE LOGITS
  targets = batch['input_ids'][:, 1:seq_len]  # WE ALWAYS MAKE LABELS FROM INPUTS: HF-STYLE

  if intra_doc_masking:
    if use_flex_attention:
      attn_mask = _build_flex_block_mask(batch['docs_lengths'], seq_len, device)
    else:
      masks = [intra_doc_causal_mask(doc_lengths, seq_len + 1, device) for doc_lengths in batch['docs_lengths']]
      attn_mask = torch.stack(masks, dim=0)  # (bsz, L+1, L+1)
      attn_mask = attn_mask[:, :seq_len, :seq_len].contiguous()  # (bsz, L, L)

    linear_mask_info = [
      intra_doc_masking_linear(doc_lengths, seq_len + 1, device) for doc_lengths in batch['docs_lengths']
    ]
    linear_masks = [item[0] for item in linear_mask_info]
    linear_masks = torch.cat(linear_masks, dim=0)
    linear_masks = linear_masks[:, :seq_len]

    boundaries = [subitem for doc_lengths in batch['docs_lengths'] for subitem in [0] + doc_lengths]
    flat_lengths = [l for docs in batch['docs_lengths'] for l in docs]
    cu_seqlens = torch.zeros(len(flat_lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(flat_lengths, device=device), dim=0)

    limit = int(bsz * seq_len)
    valid = cu_seqlens <= limit
    cu_seqlens = cu_seqlens[valid]

    if cu_seqlens[-1] != limit:
      cu_seqlens = torch.tensor(cu_seqlens.tolist() + [limit]).to(dtype=torch.int32, device=device)

    assert cu_seqlens.argmax() == cu_seqlens.shape[0] - 1, cu_seqlens

  else:
    if use_flex_attention:
      attn_mask = None
    else:
      attn_mask = (
        torch.tril(torch.ones(seq_len, seq_len)).repeat(batch['input_ids'].shape[0], 1, 1).to(dtype=torch.bool)
      )
    linear_masks = torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
    cu_seqlens = None

  if 'cuda' in device:
    inputs = inputs.pin_memory().to(device, non_blocking=True)
    targets = targets.pin_memory().to(device, non_blocking=True)
  else:
    inputs, targets = inputs.to(device), targets.to(device)

  if attn_mask is not None and isinstance(attn_mask, torch.Tensor):
    attn_mask = attn_mask.to(device=device)

  return inputs, targets, attn_mask, linear_masks, cu_seqlens


def apply_compile(model: torch.nn.Module) -> torch.nn.Module:
  blocks = getattr(model, 'layers')
  assert blocks is not None, 'Error: `model.layers` is set to `None`.'

  for layer_id, block in blocks.named_children():
    block = torch.compile(block)
    blocks[int(layer_id)] = block

  if not model.cfg.tie_embeddings:
    embeddings_key = 'embed_tokens'
    embeddings = torch.compile(getattr(model, embeddings_key), fullgraph=True)
    model.register_module(embeddings_key, embeddings)

  norm_key = 'out_norm'
  norm = torch.compile(getattr(model, norm_key), fullgraph=True)
  model.register_module(norm_key, norm)

  if not model.cfg.tie_embeddings:
    lm_head_key = 'lm_head'
    lm_head = torch.compile(getattr(model, lm_head_key), fullgraph=True)
    model.register_module(lm_head_key, lm_head)

  print('Compiling the entire model with torch.compile')
  return torch.compile(model)


class TorchEngine(torch.nn.Module):
  def __init__(self, model, cfg, device, local_rank, ckpt):
    super().__init__()

    self.micro_steps = 0
    self.accumulated_samples = 0

    self.seq_len = cfg.seq_len
    self.accumulation_steps = cfg.grad_accumulation_steps
    self.grad_clip = cfg.grad_clip
    self.dtype = cfg.dtype
    self.intra_doc_masking = getattr(cfg, 'intra_doc_masking', False)
    self.use_flex_attention = getattr(cfg, 'use_flex_attention', True)

    if self.use_flex_attention and not _FLEX_ATTENTION_AVAILABLE:
      raise ImportError(
        'use_flex_attention=True requires PyTorch >= 2.5. Update PyTorch or set use_flex_attention=False.'
      )

    self.device = device

    # Load model state dict
    if cfg.resume:
      model.load_state_dict(ckpt['state_dict'])
      self.micro_steps = ckpt['step'] * cfg.grad_accumulation_steps

    # Move model to device
    self.model = model.to(device)

    # Compile
    if cfg.torch_compile:
      print('Compiling the model...')
      self.model = apply_compile(self.model)
    
    # Move to DDP after custom compile
    if torch.distributed.is_initialized():
      self.model = DDP(self.model, device_ids=[local_rank])

    # AMP
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
    self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Grad scaler if training in fp16, if enabled=False, scaler is a no-op
    self.scaler = torch.amp.GradScaler(enabled=(self.dtype == 'float16'))

    # Loss
    self.criterion = CrossEntropyLoss()

    # Optimizer
    param_groups = get_param_groups(model, cfg.weight_decay)
    self.optimizer = intialize_optimizer(param_groups, cfg)
    self.scheduler = initialize_scheduler(self.optimizer, cfg)

    if cfg.resume:
      self.optimizer.load_state_dict(ckpt['optimizer'])
      self.scaler.load_state_dict(ckpt['scaler'])
      if not cfg.cooldown_only:
        self.scheduler.load_state_dict(ckpt['scheduler'])
      else:
        cooldown_steps = (
          cfg.cooldown_steps if isinstance(cfg.cooldown_steps, int) else int(cfg.steps_budget * cfg.cooldown_steps)
        )
        self.scheduler = LinearCooldown(
          self.optimizer,
          lr_max=cfg.lr,
          lr_end=cfg.lr_end if (cfg.lr_end is not None) else (cfg.lr_end_pct * cfg.lr),
          cooldown_start_step=0,
          cooldown_steps=cooldown_steps,
        )

    # if cfg.count_flops:
    #   flop_counter = FlopCounterMode(self.model, display=False)

  def step(self, batch):
    """Wraps a fwd pass, bwd pass, and optimization step."""

    self.model.train()

    self.micro_steps += 1
    self.accumulated_samples += 1

    inputs, targets, attn_mask, linear_mask, cu_seqlens = _move_to_device(
      batch, self.seq_len, self.device, self.intra_doc_masking, self.use_flex_attention
    )

    # sync (reduce) gradients at the last accumulation step
    if torch.distributed.is_initialized():
      self.model.require_backward_grad_sync = self.accumulated_samples == self.accumulation_steps

    # forward pass with autocasting
    with self.ctx:
      output = self.model(inputs, attn_mask, linear_mask, cu_seqlens)
      logits = getattr(output, 'logits', output)
      loss = self.criterion(
        logits[:, :-1, :].reshape(-1, logits.size(-1)).contiguous(), targets.reshape(-1).contiguous()
      )
      loss = loss / self.accumulation_steps

    # detach for logging (scale up to undo the division above)
    loss_val = loss.detach() * self.accumulation_steps
    if torch.isnan(loss_val):
      raise ValueError('Train loss is nan')

    # backward pass, with gradient scaling if training in fp16
    self.scaler.scale(loss).backward()

    # step after accumulation
    if self.accumulated_samples == self.accumulation_steps:
      self.accumulated_samples = 0

      if self.grad_clip:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

      # step the optimizer, step the scaler if training in fp16
      self.scaler.step(self.optimizer)
      self.scaler.update()

      # flush the gradients
      self.optimizer.zero_grad(set_to_none=True)

      # step the scheduler
      if self.scheduler:
        self.scheduler.step()

    return loss_val

  @torch.no_grad()
  def eval(self, dataloader):
    """Evaluate model on a dataloader."""

    self.model.eval()

    # Compute loss on dataloader
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
      inputs, targets, attn_mask, linear_mask, cu_seqlens = _move_to_device(
        batch, self.seq_len, self.device, self.intra_doc_masking, self.use_flex_attention
      )
      with self.ctx:
        output = self.model(inputs, attn_mask, linear_mask, cu_seqlens)
        logits = getattr(output, 'logits', output)
        loss = self.criterion(
          logits[:, :-1, :].reshape(-1, logits.size(-1)).contiguous(), targets.reshape(-1).contiguous()
        )

      if torch.isnan(loss) or loss is None:
        raise ValueError('Validation loss is nan')

      total_loss += loss.item()
      num_batches += 1

    # reduce loss across processes
    if dist.is_initialized():
      total_loss_tensor = torch.tensor([total_loss], device=self.device)
      num_batches_tensor = torch.tensor([num_batches], device=self.device, dtype=torch.int)
      dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
      dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
      total_loss = total_loss_tensor.item()
      num_batches = num_batches_tensor.item()

    # calculate average loss
    avg_loss = total_loss / num_batches

    return avg_loss
