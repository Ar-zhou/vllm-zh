# GLM DSpark MLA draft KV-layer ablation

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
| Share first two layers' KV | 2 | 5 | 1,821,696 | 3,663 / 72,824 | 5.03% | 1.402 | 673.1 s |
| First two layers only | 2 | 2 | 1,821,696 | 2,676 / 80,680 | 3.32% | 1.265 | 675.5 s |
| Share first three layers' KV | 3 | 5 | 1,780,672 | 6,274 / 52,112 | 12.04% | 1.963 | 467.0 s |
| First three layers only | 3 | 3 | 1,780,672 | 4,502 / 66,144 | 6.81% | 1.545 | 575.1 s |
| Share first four layers' KV | 4 | 5 | 1,741,440 | 8,053 / 38,016 | 21.18% | 2.695 | 360.6 s |
| First four layers only | 4 | 4 | 1,741,440 | 6,889 / 47,200 | 14.60% | 2.168 | 486.6 s |

Mean tokens per verify includes the target verification token. It equals
`1 + accepted draft tokens / draft steps`. Draft acceptance excludes this
target token. Relative to the five-cache baseline, retaining one, two, three,
or four draft KV layers increased schedulable KV capacity by 9.43%, 6.91%,
4.50%, or 2.20% per DP rank. The two modes for each retained-layer count ran
on separate four-GPU groups concurrently. The one-layer runs overlapped only
partly, so wall times are directional rather than an isolated throughput
benchmark.

| Mode | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 75.48% | 55.06% | 39.57% | 26.21% | 18.43% | 12.11% | 7.99% | 5.45% |
| Share 1 KV | 16.90% | 2.23% | 0.11% | 0.02% | 0% | 0% | 0% | 0% |
| Forward 1 | 11.38% | 1.38% | 0.14% | 0.03% | 0% | 0% | 0% | 0% |
| Share 2 KV | 29.36% | 7.93% | 1.97% | 0.68% | 0.23% | 0.05% | 0.01% | 0% |
| Forward 2 | 20.48% | 4.83% | 0.96% | 0.21% | 0.06% | 0% | 0% | 0% |
| Share 3 KV | 52.72% | 24.65% | 10.93% | 4.62% | 2.16% | 0.84% | 0.29% | 0.09% |
| Forward 3 | 34.97% | 12.63% | 4.67% | 1.55% | 0.45% | 0.13% | 0.06% | 0% |
| Share 4 KV | 69.07% | 44.82% | 26.54% | 14.92% | 8.14% | 3.64% | 1.66% | 0.67% |
| Forward 4 | 55.95% | 30.54% | 16.34% | 8.07% | 3.36% | 1.41% | 0.76% | 0.34% |

The draft's later layers were trained to read their own latent KV. Sharing an
earlier layer's latent state changes those layers' inputs, which likely explains
the loss in acceptance. Skipping later layers removes further learned
transformations and has still lower acceptance. This is an architectural
inference from the ablation, not a measured attribution to any one operation.

Among the tested training-free variants, keeping four KV layers while running
all five forward layers came closest to baseline throughput: 21.18% draft
acceptance and 360.6 s versus 30.04% and 320.4 s, for a 2.20% capacity gain.
Retaining only three KV layers gave a 4.50% capacity gain but took 467.0 s.

At a fixed GPU memory utilization, vLLM assigns the freed HBM to a larger KV
pool; total HBM usage therefore does not fall. Every mode still constructs and
loads all five draft layers' weights in this experiment.
Output correctness beyond successful generation was not scored here.
