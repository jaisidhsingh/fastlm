# Strategy for conducting scaling law experiments

## 1. Setup

To obtain a scaling law of the form

$$\ell(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

we want to obtain a grid of loss values $\ell$ by varying the parameter count $N$ and number of tokens used for training $D$, that control how much compute $C = 6ND$ we use overall.

We make a setup with $P=4$ budgets for parameter count $N$, i.e., $N \in \{20\text{M}, 50\text{M}, 150\text{M}, 300\text{M}\}$ and $Q=5$ budgets for tokens, i.e, $D \in \{3\text{B},6\text{B},9\text{B},12\text{B},15\text{B}\}$.

## 2. Hyper-parameter sweep

We define a grid of $B=6$ global batch sizes $b \in \{16, 32, 64, 128, 256, 512\}$ and $H=6$ learning rates $\eta \in \{0.0025,0.005,0.01,0.02,0.04,0.08\}$. Since, training each $(b, \eta)$ configuration on all $(N, D)$ values will be too costly, we use small batch sizes for small token budgets and large batch sizes for larger ones, reducing the set of candidate batch sizes to $4$ for each token budget. Additionally, we keep at least $3$ values common between the $4$ candidates of two adjacent token budgets. This creates the _staggered_ configuration given as follows:

| **D**   |     |     | **b** |     |     |     |
| ------- | --- | --- | ----- | --- | --- | --- |
| **3B**  | 16  | 32  | 64    | 128 |     |     |
| **6B**  | 16  | 32  | 64    | 128 |     |     |
| **9B**  |     | 32  | 64    | 128 | 256 |     |
| **12B** |     |     | 64    | 128 | 256 | 512 |
| **15B** |     |     | 64    | 128 | 256 | 512 |

Since we want to use a fixed warmup of $2000$ steps, we further prune values of $b$ that will not allow this condition to be fulfilled. Hence, for each value of $N$ and $\eta$, we would only need to make a maximum of $B$ training runs for all token budgets because we use WSD as our learning rate schedule (we can arrive at intermediate token budgets by decaying saved checkpoints). Note that we could have trained the model for 15B tokens for each batch size, however, this staggered approach allows for more savings in terms of total GPU-hours.

## 3. Determining micro batch size and gradient accumulation

For each $N$, we fix the number of GPUs we want to use. We use for $dp = 4$ GPUs for $N \in \{20\text{M}, 50\text{M}\}$ and $dp=8$ GPUs for $N\in \{150\text{M}, 300\text{M}\}$. Additionally, for a particular value of $b$, i.e., the global batch size we can find the product of the micro batch size ($mbs$) and the gradient accumulation steps ($gas$) because $b$ and $dp$ are now known to us, via

$$
b = mbs \times gas \times dp
$$

Because we want to train as fast as possible, we iterate $gas$, i.e, $gas \in \{1, 2, 4, 8\}$ and for each, we set $mbs$ to be the maximum number of samples that each individual GPU can fit. Finally, we choose the $(mbs, gas)$ pair that maximizes throughput, measured via tokens per second (TPS).

This results in us finding the configuration that will allow us to train as fast as possible on $N \times B \times H$ total settings. Additionally, throughput-profiling using TPS lets us estimate the GPU-hours required in advance.

## 4. Understand hyper-parameter scaling alongside compute

This strategy allows us to create a grid of $(N,D)$ values, and look at the evolution of 3 quantities together:

1. loss $\ell(N, D \mid b*, \eta*)$
2. batch size $b$
3. learning rate $\eta$

## 5. Final goal

This strategy outlines the plan for conducting scaling law experiments to ultimately understand the nuances of hybrid LLMs w.r.t existing studies on dense LLMs. Since OLMo Hybrid was trained and scaled using ladders and hyper-parameters from OLMo 3 (a purely dense LLM), there exists a gap verifying how hybrid LLMs behave w.r.t to their hyper-parameters when scaling pretraining. Upon running this strategy for hybrid LLMs of different hybridization ratios $r$, we would ideally hope to find hyper-parameters and loss as functions of this ratio

1. $$\ell(N, D, r) = \frac{A(r)}{N^{\alpha(r)}} + \frac{B(r)}{D^{\beta(r)}} + E(r)$$
2. $$b(r) = c_1 \cdot N^{r(y_1)} \cdot D^{r(z_1)}$$
3. $$\eta(r) = c_2 \cdot N^{r(y_2)} \cdot D^{r(z_2)}$$

## Results folder structure for one parameter scale (per value of $N$)

```plaintext
parameter_scale_id/
    ├── model_config.yaml
    ├── scaling_info.json
    ├── collected_results.json
    │
    └── gbs_wise_results/
        ├── gbs_16/
        │   ├── throughput_analysis_configs/
        │   │   ├── mbs-16_gas-1.yaml
        │   │   ├── mbs-8_gas-2.yaml
        │   │   └── ...
        │   │
        │   ├── lr_wise_configs/
        │   │   ├── config_lr_0p01.yaml
        │   │   ├── config_lr_0p02.yaml
        │   │   └── ...
        │   │
        │   ├── throughput_analysis.json
        │   └── checkpoints/
        │       ├── lr_0p01/
        │       │   ├── info.txt
        │       │   ├── metrics_warmup_done.json
        │       │   ├── warmup_done.pt
        │       │   ├── metrics_decay_starts_to_0p5B.json
        │       │   ├── ckpt_decay_starts_to_0p5B.pt
        │       │   ├── metrics_decayed_to_0p5B.json
        │       │   ├── ckpt_decayed_to_0p5B.pt
        │       │   ├── metrics_decayed_to_1p0B.json
        │       │   ├── ckpt_decayed_to_1p0B.pt
        │       │   └── ...
        │       │
        │       ├── lr_0p02/
        │       │   ├── ...
        │       └── ...
        │
        ├── gbs_32/
        │   ├── lr_wise_configs/
        │   │   └── ...
        │   │
        │   ├── throughput_analysis.json
        │   └── checkpoints/
        │       └── ...
        │
        └── ...
```

## Economics

```plaintext
GBS :	 Time x 6 LRs	   : DP : 	 GPU hours x 6 LRs
------------------------------------------------------------
20M
----
16 : 	 0.91 x 6 hours    : 1 : 	 0.91 x 6
32 : 	 2.61 x 6 hours    : 1 : 	 2.61 x 6
64 : 	 3.26 x 6 hours    : 4 : 	 13.03 x 6
128 : 	 3.31 x 6 hours    : 4 : 	 13.24 x 6
256 : 	 3.27 x 6 hours    : 4 : 	 13.09 x 6
512 : 	 3.26 x 6 hours    : 4 : 	 13.06 x 6

50M
----
16 : 	 1.71 x 6 hours    : 1 : 	 1.71 x 6
32 : 	 4.91 x 6 hours    : 1 : 	 4.91 x 6
64 : 	 6.15 x 6 hours    : 4 : 	 24.61 x 6
128 : 	 6.17 x 6 hours    : 4 : 	 24.66 x 6
256 : 	 6.16 x 6 hours    : 4 : 	 24.64 x 6
512 : 	 6.16 x 6 hours    : 4 : 	 24.63 x 6

150M
----
16 : 	 2.48 x 6 hours    : 1 : 	 2.48 x 6
32 : 	 7.43 x 6 hours    : 1 : 	 7.43 x 6
64 : 	 9.28 x 6 hours    : 4 : 	 37.13 x 6
128 : 	 9.26 x 6 hours    : 4 : 	 37.02 x 6
256 : 	 9.27 x 6 hours    : 4 : 	 37.09 x 6
512 : 	 9.25 x 6 hours    : 4 : 	 37.01 x 6

300M
----
16 : 	 4.66 x 6 hours    : 1 : 	 4.66 x 6
32 : 	 13.9 x 6 hours    : 1 : 	 13.9 x 6
64 : 	 17.36 x 6 hours    : 4 : 	 69.44 x 6
128 : 	 17.36 x 6 hours    : 4 : 	 69.42 x 6
256 : 	 17.37 x 6 hours    : 4 : 	 69.49 x 6
512 : 	 17.34 x 6 hours    : 4 : 	 69.34 x 6

------------------------------------------------------------
Total time =  914.14 hours, assuming no runs in parallel
Total GPU hours = 3077.56
```
