# Intra-document `cu_seqlens`: Legacy GDN vs FLA GDN/FlashAttention

## Data contract

Packed training examples contain `seq_len + 1` tokens—2049 when `seq_len=2048`—because the extra token is needed to construct shifted language-model targets.

The model receives only the first 2048 tokens:

- Inputs: positions `0:2048`
- Targets: positions `1:2048`

However, `docs_lengths` describes all 2049 stored tokens.

## Original boundary bug

The old implementation flattened the unmodified per-sample `docs_lengths` across the batch and only clipped the final cumulative offset to `batch_size * seq_len`.

That did not remove the extra token independently from every sample. With multiple samples:

- Sample boundaries became progressively shifted.
- Document segments could cross batch-row boundaries.
- A segment could be reported as length 2049 while the model’s maximum sequence length was 2048.

The offsets could remain monotonically increasing and end at the correct total token count, so the existing assertion did not detect the semantic corruption.

## Why legacy GDN did not fail

Legacy GDN flattens the token buffer and passes `cu_seqlens` to its segmented recurrent/convolution kernels. These kernels primarily require monotonic offsets covering valid flattened memory.

Consequently, malformed boundaries could remain memory-valid:

- The kernels reset recurrent state at incorrect positions.
- Tokens from adjacent packed samples/documents could be treated as one segment.
- Training could continue with finite loss.
- The absence of an error did not mean intra-document separation was correct.

In short, legacy GDN tolerated the malformed segmentation but silently used the wrong document boundaries.

## Why FLA dense attention produced NaNs

FLA dense attention forwards `cu_seqlens` to FlashAttention’s variable-length kernel and supplies `max_seqlen=2048`.

FlashAttention has a stricter structural contract:

- Every adjacent difference in `cu_seqlens` must describe a valid segment.
- No segment may exceed the supplied `max_seqlen`.
- Offsets must accurately describe the flattened query/key/value tensors.

A malformed 2049-token segment violated that contract. GPU kernels may not raise a clean Python exception for invalid metadata; they can instead produce undefined values, which propagate into logits and eventually trigger `Train loss is nan`.

## Current behavior

The correction is scoped to `backend == 'fla'` with intra-document masking.

For each batch sample, FLA now:

1. Processes its document lengths independently.
2. Trims the final segment so retained lengths sum to exactly `seq_len`.
3. Flattens those corrected per-sample segments.
4. Builds `int32` cumulative offsets.

This guarantees sample endpoints at:

```text
seq_len, 2 * seq_len, 3 * seq_len, ...
```

For `seq_len=2048`, batch endpoints are therefore `2048`, `4096`, `6144`, etc.

Legacy boundary construction was restored unchanged at the user’s request.

## Important benchmarking nuance

For FLA models, enabling intra-document masking now gives corrected boundaries for both:

- FLA GDN layers, which use segmented recurrent/convolution kernels.
- FLA attention layers, which use FlashAttention varlen kernels.

The `use_flex_attention` flag does not select PyTorch FlexAttention in the FLA backend. With FLA intra-document masking, the meaningful inputs are:

```text
attention_mask=None
linear_mask=<2-D real-token mask>
cu_seqlens=<corrected flattened document boundaries>
```

When comparing legacy GDN and FLA GDN throughput with intra-document masking, remember that legacy still uses the original boundary construction. A successful legacy run demonstrates kernel tolerance and throughput, not necessarily equivalent document-boundary semantics.
