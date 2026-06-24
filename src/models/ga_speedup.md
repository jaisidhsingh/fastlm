# Earlier:

## Profile:

```plaintext
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     117.389ms        81.93%     117.389ms      25.902us          4532
         triton_poi_fused__unsafe_view_mul_silu_split_8         0.00%       0.000us         0.00%       0.000us       0.000us       3.696ms         2.58%       3.696ms      15.794us           234
triton_red_fused__to_copy__unsafe_view_add_mean_pow_...         0.00%       0.000us         0.00%       0.000us       0.000us       3.229ms         2.25%       3.229ms       4.907us           658
triton_red_fused__to_copy__unsafe_view_add_mean_pow_...         0.00%       0.000us         0.00%       0.000us       0.000us       3.208ms         2.24%       3.208ms       4.875us           658
fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMem...         0.00%       0.000us         0.00%       0.000us       0.000us       3.049ms         2.13%       3.049ms     138.589us            22
triton_poi_fused__to_copy__unsafe_view_cat_mul_split...         0.00%       0.000us         0.00%       0.000us       0.000us       2.500ms         1.74%       2.500ms      10.682us           234
triton_poi_fused__to_copy_add_mul_select_stack_sub_v...         0.00%       0.000us         0.00%       0.000us       0.000us       2.448ms         1.71%       2.448ms      10.459us           234
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       2.026ms         1.41%       2.026ms       1.563us          1296
       triton_poi_fused_scalar_tensor_unsqueeze_where_5         0.00%       0.000us         0.00%       0.000us       0.000us       1.793ms         1.25%       1.793ms       8.417us           213
triton_poi_fused__unsafe_view_mul_sigmoid_transpose_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.628ms         1.14%       1.628ms       6.959us           234
              nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN         0.00%       0.000us         0.00%       0.000us       0.000us     835.163us         0.58%     835.163us      18.981us            44
                     nvjet_tst_96x128_64x7_2x1_v_bz_TNN         0.00%       0.000us         0.00%       0.000us       0.000us     568.637us         0.40%     568.637us       8.616us            66
                         Memcpy DtoH (Device -> Pinned)         0.00%       0.000us         0.00%       0.000us       0.000us     299.169us         0.21%     299.169us       2.233us           134
              nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN         0.00%       0.000us         0.00%       0.000us       0.000us     239.455us         0.17%     239.455us     239.455us             1
triton_per_fused__to_copy__unsafe_view_add_mean_mul_...         0.00%       0.000us         0.00%       0.000us       0.000us     177.023us         0.12%     177.023us       4.318us            41
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      90.489us         0.06%      90.489us       1.005us            90
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us      44.421us         0.03%      44.421us       1.010us            44
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      38.272us         0.03%      38.272us       1.531us            25
triton_per_fused__to_copy__unsafe_view_add_mean_mul_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.536us         0.00%       5.536us       5.536us             1
triton_per_fused__to_copy__unsafe_view_add_mean_mul_...         0.00%       0.000us         0.00%       0.000us       0.000us       4.704us         0.00%       4.704us       4.704us             1
triton_per_fused__to_copy__unsafe_view_add_mean_mul_...         0.00%       0.000us         0.00%       0.000us       0.000us       4.672us         0.00%       4.672us       4.672us             1
triton_per_fused__to_copy_add_embedding_mean_mul_pow...         0.00%       0.000us         0.00%       0.000us       0.000us       4.480us         0.00%       4.480us       4.480us             1
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       1.216us         0.00%       1.216us       1.216us             1
                                  cudaStreamIsCapturing         0.81%       1.855ms         0.81%       1.855ms       0.392us       0.000us         0.00%       0.000us       0.000us          4735
                                Activity Buffer Request         0.85%       1.946ms         0.85%       1.946ms       1.946ms       0.000us         0.00%       0.000us       0.000us             1
                                        cudaMemcpyAsync         3.81%       8.775ms         3.81%       8.775ms       6.031us       0.000us         0.00%       0.000us       0.000us          1455
                                  cudaStreamSynchronize         0.49%       1.121ms         0.49%       1.121ms       7.047us       0.000us         0.00%       0.000us       0.000us           159
                                       cudaLaunchKernel         8.30%      19.111ms         8.44%      19.419ms       4.142us       0.000us         0.00%       0.000us       0.000us          4688
                                  Lazy Function Loading         0.52%       1.187ms         0.52%       1.187ms      33.907us       0.000us         0.00%       0.000us       0.000us            35
                                          cudaHostAlloc         1.02%       2.343ms         1.02%       2.343ms       1.172ms       0.000us         0.00%       0.000us       0.000us             2
                                  cudaDeviceSynchronize         7.89%      18.155ms         7.89%      18.155ms     179.750us       0.000us         0.00%       0.000us       0.000us           101
                                         cuLaunchKernel         6.08%      13.993ms         6.08%      13.993ms       5.575us       0.000us         0.00%       0.000us       0.000us          2510
                                             cudaMalloc         4.34%       9.992ms         4.34%       9.992ms     114.855us       0.000us         0.00%       0.000us       0.000us            87
                                               cudaFree         7.96%      18.309ms         7.96%      18.309ms      18.309ms       0.000us         0.00%       0.000us       0.000us             1
                                 cudaDeviceGetAttribute         0.78%       1.792ms         0.78%       1.792ms      85.351us       0.000us         0.00%       0.000us       0.000us            21
                                cudaGetDriverEntryPoint         0.00%       2.399us         0.00%       2.399us       1.200us       0.000us         0.00%       0.000us       0.000us             2
                                   cudaFuncSetAttribute        40.69%      93.638ms        51.13%     117.663ms       4.903ms       0.000us         0.00%       0.000us       0.000us            24
                       Runtime Triggered Module Loading        10.42%      23.982ms        10.42%      23.982ms       3.997ms       0.000us         0.00%       0.000us       0.000us             6
              cudaOccupancyAvailableDynamicSMemPerBlock         0.00%       5.032us         0.00%       5.032us       5.032us       0.000us         0.00%       0.000us       0.000us             1
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.00%       1.614us         0.00%       1.614us       1.614us       0.000us         0.00%       0.000us       0.000us             1
                         cudaOccupancyMaxActiveClusters         0.00%      10.939us         0.00%      10.939us       0.608us       0.000us         0.00%       0.000us       0.000us            18
                                   cudaGetSymbolAddress         0.00%       8.707us         0.05%     104.909us     104.909us       0.000us         0.00%       0.000us       0.000us             1
                                       cuLaunchKernelEx         0.31%     717.869us         0.31%     717.869us       6.467us       0.000us         0.00%       0.000us       0.000us           111
                               cudaEventRecordWithFlags         4.24%       9.755ms         4.24%       9.755ms       2.112us       0.000us         0.00%       0.000us       0.000us          4620
                                         cudaEventQuery         0.78%       1.802ms         0.78%       1.802ms       0.390us       0.000us         0.00%       0.000us       0.000us          4620
                                   cudaEventElapsedTime         0.71%       1.625ms         0.71%       1.625ms       0.704us       0.000us         0.00%       0.000us       0.000us          2310
                                        cudaMemsetAsync         0.01%      13.530us         0.01%      13.530us      13.530us       0.000us         0.00%       0.000us       0.000us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
```

## Code:

```python
class GatedAttention(nn.Module):
  def __init__(self, cfg: ModelConfig):
    super().__init__()
    assert cfg.dim % cfg.n_heads == 0
    self.n_heads = cfg.n_heads
    self.head_dim = cfg.dim // cfg.n_heads

    self.w_qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
    self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    self.use_gate = cfg.attn_gate
    self.qk_norm = cfg.attn_qk_norm

    if self.use_gate:
      self.w_gate = nn.Linear(cfg.dim, cfg.dim, bias=False)
    if self.qk_norm:
      self.q_norm = RMSNorm(self.head_dim, cfg.rmsnorm_eps)
      self.k_norm = RMSNorm(self.head_dim, cfg.rmsnorm_eps)

  def forward(self, x, freqs_cis: torch.Tensor | None, attention_mask: torch.Tensor | None = None):
    bsz, seqlen, d = x.shape  # (bsz, seqlen, d)

    q, k, v = self.w_qkv(x).split(d, dim=2)  # (bsz, seqlen, d)
    q = q.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, nh, h_dim)
    k = k.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, nh, h_dim)
    v = v.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, nh, h_dim)

    if self.qk_norm:
      q = self.q_norm(q)
      k = self.k_norm(k)


    if freqs_cis is not None:
      q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)  # (bsz, seqlen, nh, h_dim)

    q = q.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    k = k.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    v = v.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)

    if attention_mask is not None:
      # attn_mask has shape (bsz, seqlen, seqlen)
      # from (bsz, L, L) to (bsz, 1, L, L) so it broadcasts over heads

      # import pdb
      # pdb.set_trace()
      attention_mask = attention_mask.unsqueeze(1)
      # pdb.set_trace()

      out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)  # (bsz, nh, seqlen, h_dim)
    else:
      out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (bsz, nh, seqlen, h_dim)

    out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d)  # (bsz, seqlen, d)

    if self.use_gate:
      gating = torch.sigmoid(self.w_gate(x))
      out = out * gating

    return self.w_out(out)
```

## Profile time:

Self CPU time total: 230.141ms

Self CUDA time total: 143.277ms

# Then:

## Profile:

```plaintext
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      55.896ms        76.38%      55.896ms      27.134us          2060
         triton_poi_fused__unsafe_view_mul_silu_split_7         0.00%       0.000us         0.00%       0.000us       0.000us       3.560ms         4.86%       3.560ms      15.215us           234
fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMem...         0.00%       0.000us         0.00%       0.000us       0.000us       3.066ms         4.19%       3.066ms     139.352us            22
         triton_poi_fused__to_copy_cat_mul_split_view_2         0.00%       0.000us         0.00%       0.000us       0.000us       2.509ms         3.43%       2.509ms      10.722us           234
triton_poi_fused__to_copy_add_mul_select_stack_sub_v...         0.00%       0.000us         0.00%       0.000us       0.000us       2.447ms         3.34%       2.447ms      10.456us           234
       triton_poi_fused_scalar_tensor_unsqueeze_where_4         0.00%       0.000us         0.00%       0.000us       0.000us       1.750ms         2.39%       1.750ms       8.216us           213
triton_poi_fused__unsafe_view_mul_sigmoid_transpose_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.501ms         2.05%       1.501ms       6.414us           234
              nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN         0.00%       0.000us         0.00%       0.000us       0.000us     838.011us         1.15%     838.011us      19.046us            44
                     nvjet_tst_96x128_64x7_2x1_v_bz_TNN         0.00%       0.000us         0.00%       0.000us       0.000us     577.150us         0.79%     577.150us       8.745us            66
                         Memcpy DtoH (Device -> Pinned)         0.00%       0.000us         0.00%       0.000us       0.000us     297.790us         0.41%     297.790us       2.222us           134
              nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN         0.00%       0.000us         0.00%       0.000us       0.000us     241.215us         0.33%     241.215us     241.215us             1
triton_per_fused__to_copy__unsafe_view_add_mean_mul_...         0.00%       0.000us         0.00%       0.000us       0.000us     185.247us         0.25%     185.247us       4.518us            41
triton_per_fused__to_copy__unsafe_view_add_cat_mean_...         0.00%       0.000us         0.00%       0.000us       0.000us      93.662us         0.13%      93.662us       4.257us            22
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      90.693us         0.12%      90.693us       1.008us            90
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us      44.221us         0.06%      44.221us       1.005us            44
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      38.143us         0.05%      38.143us       1.526us            25
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      27.201us         0.04%      27.201us       1.133us            24
triton_per_fused__to_copy__unsafe_view_add_mean_mul_...         0.00%       0.000us         0.00%       0.000us       0.000us       6.176us         0.01%       6.176us       6.176us             1
triton_per_fused__to_copy__unsafe_view_add_mean_mul_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.024us         0.01%       5.024us       5.024us             1
triton_per_fused__to_copy_add_embedding_mean_mul_pow...         0.00%       0.000us         0.00%       0.000us       0.000us       4.384us         0.01%       4.384us       4.384us             1
triton_per_fused__to_copy__unsafe_view_add_mean_mul_...         0.00%       0.000us         0.00%       0.000us       0.000us       4.096us         0.01%       4.096us       4.096us             1
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       1.184us         0.00%       1.184us       1.184us             1
                                  cudaStreamIsCapturing         0.35%     804.161us         0.35%     804.161us       0.362us       0.000us         0.00%       0.000us       0.000us          2219
                                Activity Buffer Request         1.03%       2.335ms         1.03%       2.335ms       2.335ms       0.000us         0.00%       0.000us       0.000us             1
                                        cudaMemcpyAsync         0.65%       1.485ms         0.65%       1.485ms       8.116us       0.000us         0.00%       0.000us       0.000us           183
                                  cudaStreamSynchronize         0.48%       1.104ms         0.48%       1.104ms       6.944us       0.000us         0.00%       0.000us       0.000us           159
                                       cudaLaunchKernel         4.23%       9.646ms         4.38%       9.977ms       4.502us       0.000us         0.00%       0.000us       0.000us          2216
                                  Lazy Function Loading         0.44%       1.001ms         0.44%       1.001ms      41.712us       0.000us         0.00%       0.000us       0.000us            24
                                          cudaHostAlloc         1.07%       2.434ms         1.07%       2.434ms       1.217ms       0.000us         0.00%       0.000us       0.000us             2
                                  cudaDeviceSynchronize         2.82%       6.435ms         2.82%       6.435ms     123.743us       0.000us         0.00%       0.000us       0.000us            52
                                         cuLaunchKernel         4.01%       9.131ms         4.01%       9.131ms       7.509us       0.000us         0.00%       0.000us       0.000us          1216
                                             cudaMalloc         4.39%      10.005ms         4.39%      10.005ms     109.947us       0.000us         0.00%       0.000us       0.000us            91
                                               cudaFree         6.36%      14.481ms         6.36%      14.481ms      14.481ms       0.000us         0.00%       0.000us       0.000us             1
                                 cudaDeviceGetAttribute         0.81%       1.836ms         0.81%       1.836ms      87.441us       0.000us         0.00%       0.000us       0.000us            21
                                cudaGetDriverEntryPoint         0.00%       2.120us         0.00%       2.120us       1.060us       0.000us         0.00%       0.000us       0.000us             2
                                   cudaFuncSetAttribute        59.32%     135.129ms        70.53%     160.654ms       6.694ms       0.000us         0.00%       0.000us       0.000us            24
                       Runtime Triggered Module Loading        11.20%      25.509ms        11.20%      25.509ms       4.252ms       0.000us         0.00%       0.000us       0.000us             6
              cudaOccupancyAvailableDynamicSMemPerBlock         0.00%       7.140us         0.00%       7.140us       7.140us       0.000us         0.00%       0.000us       0.000us             1
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.00%       1.920us         0.00%       1.920us       1.920us       0.000us         0.00%       0.000us       0.000us             1
                         cudaOccupancyMaxActiveClusters         0.01%      17.097us         0.01%      17.097us       0.950us       0.000us         0.00%       0.000us       0.000us            18
                                   cudaGetSymbolAddress         0.01%      12.733us         0.07%     157.472us     157.472us       0.000us         0.00%       0.000us       0.000us             1
                                       cuLaunchKernelEx         0.28%     629.971us         0.28%     629.971us       5.675us       0.000us         0.00%       0.000us       0.000us           111
                               cudaEventRecordWithFlags         1.87%       4.264ms         1.87%       4.264ms       2.030us       0.000us         0.00%       0.000us       0.000us          2100
                                         cudaEventQuery         0.36%     826.720us         0.36%     826.720us       0.394us       0.000us         0.00%       0.000us       0.000us          2100
                                   cudaEventElapsedTime         0.30%     674.168us         0.30%     674.168us       0.642us       0.000us         0.00%       0.000us       0.000us          1050
                                        cudaMemsetAsync         0.01%      14.797us         0.01%      14.797us      14.797us       0.000us         0.00%       0.000us       0.000us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 227.785ms
Self CUDA time total: 73.183ms
```

## Code:

```python
class GatedAttention(nn.Module):
  def __init__(self, cfg: ModelConfig):
    super().__init__()
    assert cfg.dim % cfg.n_heads == 0
    self.n_heads = cfg.n_heads
    self.head_dim = cfg.dim // cfg.n_heads

    self.w_qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
    self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    self.use_gate = cfg.attn_gate
    self.qk_norm = cfg.attn_qk_norm

    if self.use_gate:
      self.w_gate = nn.Linear(cfg.dim, cfg.dim, bias=False)
    if self.qk_norm:
      self.qk_norm = RMSNorm(self.head_dim * 2, cfg.rmsnorm_eps)

  def forward(self, x, freqs_cis: torch.Tensor | None, attention_mask: torch.Tensor | None = None):
    bsz, seqlen, d = x.shape  # (bsz, seqlen, d)
    qkv = self.w_qkv(x).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)

    if self.qk_norm:
      qk = self.qk_norm(torch.cat([q, k], dim=-1))
      q, k = qk.chunk(2, dim=-1)

    if freqs_cis is not None:
      q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)  # (bsz, seqlen, nh, h_dim)

    q = q.movedim(1, 2)
    k = k.movedim(1, 2)
    v = v.movedim(1, 2)

    if attention_mask is not None:
      # attn_mask has shape (bsz, seqlen, seqlen)
      # from (bsz, L, L) to (bsz, 1, L, L) so it broadcasts over heads

      # import pdb
      # pdb.set_trace()
      attention_mask = attention_mask.unsqueeze(1)
      # pdb.set_trace()

      out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)  # (bsz, nh, seqlen, h_dim)
    else:
      out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (bsz, nh, seqlen, h_dim)

    out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d)  # (bsz, seqlen, d)

    if self.use_gate:
      out = out * torch.sigmoid(self.w_gate(x))

    return self.w_out(out)
```

## Profile time:

Self CPU time total: 227.785ms

Self CUDA time total: 73.183ms

## Output tensor sum:

37376.0
