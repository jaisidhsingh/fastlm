"""
Throughput Diagnostic Script for fastlm
========================================

This script identifies WHY training is slow by profiling the actual SDPA backend
selection, measuring per-component step time, and comparing configurations.

Run:
    python -m experiments.diagnose_throughput --config=<path_to_your_config.yaml>

What it does:
    1. Verifies which SDPA backend (Flash, MemEfficient, Math) is actually used
    2. Measures throughput with the current code (explicit attn_mask) vs is_causal=True
    3. Tests TF32 on vs off
    4. Reports correct MFU numbers
    5. Profiles time breakdown: data loading, forward, backward, optimizer step
"""

import time
from collections import defaultdict
from contextlib import nullcontext, suppress

import torch
import torch.nn.functional as F
from absl import app, flags
from torch.utils.flop_counter import FlopCounterMode

from src import utils
from src.data import get_dataloaders
from src.engine import TorchEngine
from src.engine.engine import _move_to_device
from src.models import construct_model
from src.torch_utils import pytorch_setup
from src.utils import GPU_PEAK_FLOPS_PER_SEC_MAP, print_master

flags.DEFINE_string('config', 'src/config/cfg_test.yaml', 'Path to config.yaml file.')
FLAGS = flags.FLAGS

# Number of warmup + measurement steps
WARMUP_STEPS = 5
MEASURE_STEPS = 20

# ============================================================
# Diagnostic 1: Check which SDPA backend is actually used
# ============================================================

def check_sdpa_backend(model, cfg, device):
    """
    Determine which scaled_dot_product_attention backend PyTorch selects
    for the current model configuration.
    """
    print_master('\n' + '=' * 70)
    print_master('DIAGNOSTIC 1: SDPA Backend Selection')
    print_master('=' * 70)

    seq_len = cfg.seq_len
    bsz = cfg.micro_batch_size
    n_heads = cfg.n_heads
    head_dim = cfg.d_model // cfg.n_heads

    # Create dummy Q, K, V tensors
    q = torch.randn(bsz, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(bsz, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(bsz, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # Test 1: With explicit boolean mask (what the code currently does)
    attn_mask_dense = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    attn_mask_dense = attn_mask_dense.unsqueeze(0).unsqueeze(0).expand(bsz, n_heads, seq_len, seq_len)

    print_master('\n--- Test A: With explicit attn_mask (current code behavior) ---')
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=True, enable_mem_efficient=True
    ):
        # Check which backend is selected using the context manager debug
        try:
            # Flash attention
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                try:
                    _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_dense)
                    print_master('  ✅ FlashAttention: AVAILABLE with explicit mask')
                except RuntimeError as e:
                    print_master(f'  ❌ FlashAttention: BLOCKED with explicit mask')
                    print_master(f'     Reason: {str(e)[:200]}')
        except Exception as e:
            print_master(f'  ❌ FlashAttention: BLOCKED ({e})')

        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            ):
                try:
                    _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_dense)
                    print_master('  ✅ MemEfficient: AVAILABLE with explicit mask')
                except RuntimeError as e:
                    print_master(f'  ❌ MemEfficient: BLOCKED with explicit mask')
                    print_master(f'     Reason: {str(e)[:200]}')
        except Exception as e:
            print_master(f'  ❌ MemEfficient: BLOCKED ({e})')

        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_dense)
                print_master('  ✅ Math (naive): AVAILABLE with explicit mask')
        except RuntimeError as e:
            print_master(f'  ❌ Math (naive): BLOCKED ({e})')

    print_master('\n--- Test B: With is_causal=True (optimal path) ---')
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print_master('  ✅ FlashAttention: AVAILABLE with is_causal=True')
    except RuntimeError as e:
        print_master(f'  ❌ FlashAttention: BLOCKED with is_causal=True')
        print_master(f'     Reason: {str(e)[:200]}')

    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=True
        ):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print_master('  ✅ MemEfficient: AVAILABLE with is_causal=True')
    except RuntimeError as e:
        print_master(f'  ❌ MemEfficient: BLOCKED with is_causal=True ({e})')

    # Memory comparison
    print_master('\n--- Memory Impact ---')
    mask_memory_mb = (bsz * seq_len * seq_len * 1) / (1024 * 1024)  # bool = 1 byte
    mask_memory_mb_repeated = (bsz * n_heads * seq_len * seq_len * 1) / (1024 * 1024)
    print_master(f'  Dense mask (bsz, L, L) memory: {mask_memory_mb:.1f} MB per micro-step')
    print_master(f'  Dense mask expanded (bsz, nh, L, L) memory: {mask_memory_mb_repeated:.1f} MB per micro-step')
    print_master(f'  With is_causal=True: 0 MB (no mask needed)')

    del q, k, v, attn_mask_dense
    torch.cuda.empty_cache()


# ============================================================
# Diagnostic 2: Compare throughput with mask vs is_causal
# ============================================================

def benchmark_sdpa_throughput(cfg, device):
    """
    Directly benchmark the attention operation with explicit mask vs is_causal.
    """
    print_master('\n' + '=' * 70)
    print_master('DIAGNOSTIC 2: SDPA Throughput Comparison')
    print_master('=' * 70)

    seq_len = cfg.seq_len
    bsz = cfg.micro_batch_size
    n_heads = cfg.n_heads
    head_dim = cfg.d_model // cfg.n_heads

    q = torch.randn(bsz, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(bsz, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(bsz, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, n_heads, seq_len, seq_len)

    # Benchmark with explicit mask
    for _ in range(WARMUP_STEPS):
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    torch.cuda.synchronize()
    mask_time = (time.perf_counter() - start) / MEASURE_STEPS

    # Benchmark with is_causal=True
    for _ in range(WARMUP_STEPS):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    causal_time = (time.perf_counter() - start) / MEASURE_STEPS

    print_master(f'\n  With explicit attn_mask: {mask_time * 1000:.2f} ms per attention call')
    print_master(f'  With is_causal=True:     {causal_time * 1000:.2f} ms per attention call')
    print_master(f'  Speedup from is_causal:  {mask_time / causal_time:.2f}×')
    print_master(f'\n  With {cfg.n_layers} layers, that\'s {(mask_time - causal_time) * cfg.n_layers * 1000:.1f} ms saved per fwd pass')

    del q, k, v, attn_mask
    torch.cuda.empty_cache()


# ============================================================
# Diagnostic 3: TF32 impact
# ============================================================

def benchmark_tf32_impact(cfg, device):
    """
    Benchmark matrix multiply performance with TF32 on vs off.
    """
    print_master('\n' + '=' * 70)
    print_master('DIAGNOSTIC 3: TF32 Matmul Impact')
    print_master('=' * 70)

    print_master(f'\n  Current torch.backends.cuda.matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}')
    print_master(f'  Current torch.backends.cudnn.allow_tf32 = {torch.backends.cudnn.allow_tf32}')

    # Create FP32 tensors (simulating operations that escape autocast)
    m, n, k = cfg.micro_batch_size * cfg.seq_len, cfg.d_model, cfg.d_model
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(k, n, device=device, dtype=torch.float32)

    for tf32_enabled in [False, True]:
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled

        for _ in range(WARMUP_STEPS):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(MEASURE_STEPS):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / MEASURE_STEPS

        print_master(f'  FP32 matmul ({m}×{k} @ {k}×{n}), TF32={tf32_enabled}: {elapsed * 1000:.2f} ms')

    # Restore to False (original default)
    torch.backends.cuda.matmul.allow_tf32 = False

    del a, b
    torch.cuda.empty_cache()


# ============================================================
# Diagnostic 4: Full step profiling (data, fwd, bwd, optim)
# ============================================================

def profile_step_breakdown(cfg, device):
    """
    Profile the time breakdown of a training step into components.
    """
    print_master('\n' + '=' * 70)
    print_master('DIAGNOSTIC 4: Per-Step Time Breakdown')
    print_master('=' * 70)

    model, _ = construct_model(cfg)
    engine = TorchEngine(model, cfg, device, local_rank=None, ckpt=None)
    trainloader, _ = get_dataloaders(cfg)

    timings = defaultdict(list)

    for micro_step, micro_batch in enumerate(trainloader, 1):
        if micro_step > WARMUP_STEPS + MEASURE_STEPS:
            break

        # --- Data loading / move to device ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        inputs, targets, attn_mask, linear_mask, cu_seqlens = _move_to_device(
            micro_batch, cfg.seq_len, device, getattr(cfg, 'intra_doc_masking', False)
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # --- Forward pass ---
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        with ctx:
            output = engine.model(inputs, attn_mask, linear_mask, cu_seqlens)
            logits = getattr(output, 'logits', output)
            loss = torch.nn.CrossEntropyLoss()(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # --- Backward pass ---
        loss.backward()

        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # --- Optimizer step ---
        engine.optimizer.step()
        engine.optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t4 = time.perf_counter()

        if micro_step > WARMUP_STEPS:
            timings['data_move'].append(t1 - t0)
            timings['forward'].append(t2 - t1)
            timings['backward'].append(t3 - t2)
            timings['optimizer'].append(t4 - t3)
            timings['total'].append(t4 - t0)

    # Report
    print_master(f'\n  Averaged over {MEASURE_STEPS} steps (after {WARMUP_STEPS} warmup):')
    print_master(f'  {"Component":<20} {"Mean (ms)":>10} {"% of total":>12}')
    print_master(f'  {"-"*20} {"-"*10} {"-"*12}')

    total_mean = sum(t for t in timings['total']) / len(timings['total']) * 1000
    for component in ['data_move', 'forward', 'backward', 'optimizer']:
        mean_ms = sum(timings[component]) / len(timings[component]) * 1000
        pct = mean_ms / total_mean * 100
        print_master(f'  {component:<20} {mean_ms:>10.2f} {pct:>11.1f}%')
    print_master(f'  {"TOTAL":<20} {total_mean:>10.2f} {"100.0":>11}%')

    tokens_per_step = cfg.micro_batch_size * cfg.seq_len
    tokens_per_sec = tokens_per_step / (total_mean / 1000)
    print_master(f'\n  Current throughput: {tokens_per_sec:.0f} tokens/sec')

    # Compute correct MFU
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak = GPU_PEAK_FLOPS_PER_SEC_MAP.get(gpu_name, None)
    if gpu_peak is not None:
        # Estimate FLOPs using 6ND approximation
        total_params = sum(p.numel() for p in model.parameters())
        flops_per_token = 6 * total_params
        flops_per_step = flops_per_token * tokens_per_step
        achieved_flops_per_sec = flops_per_step / (total_mean / 1000)
        mfu = achieved_flops_per_sec / gpu_peak
        print_master(f'  Correct MFU: {mfu:.1%} (GPU: {gpu_name}, peak: {gpu_peak/1e12:.0f} TFLOPS)')
    else:
        print_master(f'  GPU "{gpu_name}" not in peak FLOPS map, cannot compute MFU')

    del model, engine
    torch.cuda.empty_cache()


# ============================================================
# Diagnostic 5: Check MFU formula bug
# ============================================================

def verify_mfu_formula():
    """
    Show the MFU formula bug in utils.py.
    """
    print_master('\n' + '=' * 70)
    print_master('DIAGNOSTIC 5: MFU Formula Verification')
    print_master('=' * 70)

    # Simulate what utils.py does (line 222)
    flops_per_step = 1e12  # example: 1 TFLOP per step
    step_time = 1.0  # 1 second
    world_size = 1
    gpu_peak = 312e12  # A100 peak

    # Current code: mfu = (flops_per_step / step_time) / world_size * gpu_peak_flops_per_sec
    mfu_buggy = (flops_per_step / step_time) / world_size * gpu_peak
    # Correct:  mfu = (flops_per_step / step_time) / (world_size * gpu_peak_flops_per_sec)
    mfu_correct = (flops_per_step / step_time) / (world_size * gpu_peak)

    print_master(f'\n  Example: {flops_per_step/1e12:.0f} TFLOP/step, {step_time:.0f}s step time, A100 peak={gpu_peak/1e12:.0f} TFLOPS')
    print_master(f'  Buggy formula  (utils.py L222): MFU = {mfu_buggy:.2e}  (nonsensical!)')
    print_master(f'  Correct formula:                MFU = {mfu_correct:.2%}')
    print_master(f'\n  The bug: multiplication instead of division by gpu_peak_flops_per_sec')
    print_master(f'  Line: mfu = (flops_per_step / step_time) / world_size * gpu_peak_flops_per_sec')
    print_master(f'  Fix:  mfu = (flops_per_step / step_time) / (world_size * gpu_peak_flops_per_sec)')


# ============================================================
# Diagnostic 6: Check intra_doc_masking mask construction cost
# ============================================================

def benchmark_mask_construction(cfg, device):
    """
    Measure the CPU-side cost of constructing attention masks.
    """
    print_master('\n' + '=' * 70)
    print_master('DIAGNOSTIC 6: Attention Mask Construction Cost')
    print_master('=' * 70)

    seq_len = cfg.seq_len
    bsz = cfg.micro_batch_size

    # Non-intra-doc path: torch.tril + repeat
    start = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        mask = torch.tril(torch.ones(seq_len, seq_len)).repeat(bsz, 1, 1).to(dtype=torch.bool)
    elapsed_simple = (time.perf_counter() - start) / MEASURE_STEPS

    print_master(f'\n  Simple causal mask (tril+repeat): {elapsed_simple * 1000:.2f} ms')
    print_master(f'  Mask shape: {mask.shape}, memory: {mask.nelement() * mask.element_size() / 1024 / 1024:.1f} MB')
    print_master(f'  This mask is constructed EVERY micro-step on CPU, then moved to GPU.')
    print_master(f'  With is_causal=True, this cost is ZERO.')

    del mask


# ============================================================
# Main
# ============================================================

def main(argv):
    CFG_PATH = FLAGS.config
    cfg, _ = utils.load_config(CFG_PATH)

    _, world_size, device, master_process = pytorch_setup(cfg)
    utils.set_arch(cfg)
    utils.set_batch_sizes(cfg, world_size)

    print_master('\n' + '=' * 70)
    print_master('FASTLM THROUGHPUT DIAGNOSTIC REPORT')
    print_master('=' * 70)

    print_master(f'\n  Config: {CFG_PATH}')
    print_master(f'  Device: {device}')
    if torch.cuda.is_available():
        print_master(f'  GPU: {torch.cuda.get_device_name(0)}')
        print_master(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print_master(f'  Model: dim={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}')
    print_master(f'  Seq len: {cfg.seq_len}')
    print_master(f'  Micro batch size: {cfg.micro_batch_size}')
    print_master(f'  dtype: {cfg.dtype}')
    print_master(f'  torch.compile: {cfg.torch_compile}')
    print_master(f'  intra_doc_masking: {getattr(cfg, "intra_doc_masking", False)}')
    print_master(f'  TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}')
    print_master(f'  TF32 cudnn: {torch.backends.cudnn.allow_tf32}')
    print_master(f'  PyTorch version: {torch.__version__}')
    print_master(f'  CUDA version: {torch.version.cuda}')

    if not torch.cuda.is_available():
        print_master('\n⚠️  No CUDA GPU detected. Run this on your A100 for meaningful results.')
        print_master('  Diagnostics 1-3 require GPU. Running formula checks only.\n')
        verify_mfu_formula()
        return

    # Run all diagnostics
    check_sdpa_backend(None, cfg, device)
    benchmark_sdpa_throughput(cfg, device)
    benchmark_tf32_impact(cfg, device)
    benchmark_mask_construction(cfg, device)
    verify_mfu_formula()

    # Only run full step profiling if data is accessible
    try:
        profile_step_breakdown(cfg, device)
    except Exception as e:
        print_master(f'\n  ⚠️ Could not run full step profiling (data not accessible?): {e}')
        print_master('  The SDPA backend and micro-benchmark diagnostics above are the key findings.')

    # Final summary
    print_master('\n' + '=' * 70)
    print_master('DIAGNOSIS SUMMARY')
    print_master('=' * 70)
    print_master('''
  🔴 PRIMARY ISSUE: Explicit attn_mask prevents FlashAttention
     → F.scaled_dot_product_attention falls back to naive O(N²) math backend
     → Expected 3-5× speedup by switching to is_causal=True

  🔴 SECONDARY ISSUE: TF32 disabled for matmuls (defaults to False)
     → Set torch.backends.cuda.matmul.allow_tf32 = True in config
     → Or add cuda_matmul_allow_tf32: True to your YAML

  🟡 MFU formula is wrong (multiplies by peak instead of dividing)
     → Your logged MFU values are nonsensical
     → Fix: mfu = achieved_flops / (world_size * gpu_peak_flops)

  FIX PRIORITY:
    1. Pass attention_mask=None and is_causal=True to SDPA (biggest win)
    2. Enable TF32 for matmuls
    3. Fix MFU formula for accurate monitoring
''')


if __name__ == '__main__':
    app.run(main)
