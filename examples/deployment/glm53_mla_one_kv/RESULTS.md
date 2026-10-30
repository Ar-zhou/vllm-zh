# Single-cache ablation results

Measured on 2026-10-30 with vLLM v0.31.0 plus the GLM DSpark MLA port in this
branch. Each mode served the same 50 captured chat requests, all with thinking
enabled and temperature 0.6. Responses were capped at 256 completion tokens;
every request succeeded and generated 256 tokens. The model ran on four H800
GPUs as TP1, DP4, EP4 with a 1,048,576-token context, K=8 probabilistic draft
sampling, standard rejection sampling, FP8 target KV, BF16 draft KV, an 8,192
token batch limit, four concurrent requests, and prefix caching disabled.

| Mode | Draft KV layers | Draft forward layers | KV capacity per DP rank | Accepted / proposed | Acceptance | Mean tokens per verify | Batch wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 5 | 5 | 1,703,936 | 9,086 / 30,248 | 30.04% | 3.403 | 320.4 s |
| Share first-layer KV | 1 | 5 | 1,864,640 | 2,061 / 85,584 | 2.41% | 1.193 | 675.6 s |
| First-layer forward only | 1 | 1 | 1,864,640 | 1,461 / 90,376 | 1.62% | 1.129 | 745.1 s |

Mean tokens per verify includes the target verification token. It equals
`1 + accepted draft tokens / draft steps`. Draft acceptance excludes this
target token. The one-cache variants increased schedulable KV capacity by
160,704 tokens (9.43%) per DP rank at the same GPU memory utilization.
Baseline and shared-KV ran on separate four-GPU groups concurrently. The
first-layer run overlapped the tail of shared-KV, so wall times are directional
rather than an isolated throughput benchmark.

| Draft position | Baseline | Share first-layer KV | First-layer forward only |
| ---: | ---: | ---: | ---: |
| 1 | 75.48% | 16.90% | 11.38% |
| 2 | 55.06% | 2.23% | 1.38% |
| 3 | 39.57% | 0.11% | 0.14% |
| 4 | 26.21% | 0.02% | 0.03% |
| 5 | 18.43% | 0% | 0% |
| 6 | 12.11% | 0% | 0% |
| 7 | 7.99% | 0% | 0% |
| 8 | 5.45% | 0% | 0% |

The draft's later layers were trained to read their own latent KV. Sharing the
first layer's latent state changes those layers' inputs, which likely explains
the loss in acceptance. Running only the first layer removes further learned
transformations and has still lower acceptance. This is an architectural
inference from the ablation, not a measured attribution to any one operation.

At a fixed GPU memory utilization, vLLM assigns the freed HBM to a larger KV
pool; total HBM usage therefore does not fall. The one-layer-forward mode
still constructs and loads all five draft layers' weights in this experiment.
Output correctness beyond successful generation was not scored here.
