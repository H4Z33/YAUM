# Context-dynamics results

Appended by `python -m benchmarks.context_dynamics`.

## Run steps=1500 T=40 hidden=64 substeps=2 dt=0.5 seeds=[0, 1]

| task | model | accuracy | params | learned γ median (90th pct) | wall (s) |
|---|---|---:|---:|---:|---:|
| recall_first | `lstm` | 0.116 ± 0.003 | 21,768 | — | 9 |
| recall_first | `ham_conservative` | 1.000 ± 0.000 | 11,144 | — | 209 |
| recall_first | `ham_learned` | 1.000 ± 0.000 | 11,208 | 0.036 (0.043) | 98 |
| count | `lstm` | 1.000 ± 0.000 | 21,768 | — | 10 |
| count | `ham_conservative` | 0.719 ± 0.056 | 11,144 | — | 91 |
| count | `ham_learned` | 1.000 ± 0.000 | 11,208 | 0.150 (0.302) | 96 |
| lag | `lstm` | 1.000 ± 0.000 | 21,768 | — | 8 |
| lag | `ham_conservative` | 0.928 ± 0.003 | 11,144 | — | 82 |
| lag | `ham_learned` | 0.995 ± 0.001 | 11,208 | 0.240 (0.363) | 91 |

Chance accuracy: recall_first 0.125, lag 0.125, count ≈ 0.2. Learned γ starts at 0.05.

<!-- The runs below were made in scratch files and appended after review. -->

### Seed 2 for the first table (PyTorch-default LSTM init)
## Run steps=1500 T=40 hidden=64 substeps=2 dt=0.5 seeds=[2]

| task | model | accuracy | params | learned γ median (90th pct) | wall (s) |
|---|---|---:|---:|---:|---:|
| recall_first | `lstm` | 0.117 ± 0.000 | 21,768 | — | 9 |
| recall_first | `ham_conservative` | 1.000 ± 0.000 | 11,144 | — | 91 |
| recall_first | `ham_learned` | 1.000 ± 0.000 | 11,208 | 0.034 (0.043) | 97 |
| count | `lstm` | 1.000 ± 0.000 | 21,768 | — | 12 |
| count | `ham_conservative` | 0.777 ± 0.000 | 11,144 | — | 89 |
| count | `ham_learned` | 1.000 ± 0.000 | 11,208 | 0.157 (0.230) | 98 |
| lag | `lstm` | 1.000 ± 0.000 | 21,768 | — | 10 |
| lag | `ham_conservative` | 0.947 ± 0.000 | 11,144 | — | 104 |
| lag | `ham_learned` | 0.997 ± 0.000 | 11,208 | 0.228 (0.325) | 857 |

Chance accuracy: recall_first 0.125, lag 0.125, count ≈ 0.2. Learned γ starts at 0.05.
## Run steps=1500 T=40 hidden=64 substeps=2 dt=0.5 seeds=[0, 1, 2] lstm_forget_bias=1.0

| task | model | accuracy | params | learned γ median (90th pct) | geometry γ / stiffness λ | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| assoc | `lstm` | 0.569 ± 0.068 | 21,768 | — | — | 6 |
| assoc | `ham_conservative` | 0.398 ± 0.013 | 11,144 | — | — | 24 |
| assoc | `ham_learned` | 0.399 ± 0.014 | 11,208 | 0.085 (0.106) | — | 25 |
| assoc | `ham_geometry` | 0.404 ± 0.007 | 11,592 | 0.065 (0.089) | 0.054 / 0.983 | 44 |

Chance accuracy: recall_first, lag and assoc 0.125; count ≈ 0.2. Learned γ starts at 0.05.
## Run steps=1500 T=40 hidden=64 substeps=2 dt=0.5 seeds=[0, 1, 2] lstm_forget_bias=1.0

| task | model | accuracy | params | learned γ median (90th pct) | geometry γ / stiffness λ | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| recall_first | `lstm` | 0.120 ± 0.012 | 21,768 | — | — | 10 |
| recall_first | `ham_geometry` | 0.318 ± 0.272 | 11,592 | 0.047 (0.054) | 0.050 / 0.996 | 243 |
| count | `lstm` | 1.000 ± 0.000 | 21,768 | — | — | 13 |
| count | `ham_geometry` | 0.277 ± 0.003 | 11,592 | 0.056 (0.067) | 0.052 / 1.001 | 216 |
| lag | `lstm` | 1.000 ± 0.000 | 21,768 | — | — | 11 |
| lag | `ham_geometry` | 0.516 ± 0.033 | 11,592 | 0.052 (0.061) | 0.051 / 1.001 | 216 |

Chance accuracy: recall_first, lag and assoc 0.125; count ≈ 0.2. Learned γ starts at 0.05.

### LSTM forget-gate bias = 3.0 (tuned baseline)
## Run steps=1500 T=40 hidden=64 substeps=2 dt=0.5 seeds=[0, 1, 2] lstm_forget_bias=3.0

| task | model | accuracy | params | learned γ median (90th pct) | geometry γ / stiffness λ | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| recall_first | `lstm` | 1.000 ± 0.000 | 21,768 | — | — | 9 |
| count | `lstm` | 1.000 ± 0.000 | 21,768 | — | — | 12 |
| lag | `lstm` | 1.000 ± 0.000 | 21,768 | — | — | 10 |
| assoc | `lstm` | 0.688 ± 0.040 | 21,768 | — | — | 4 |

Chance accuracy: recall_first, lag and assoc 0.125; count ≈ 0.2. Learned γ starts at 0.05.

### LSTM tuning on recall_first (3 seeds each)

| forget bias | hidden | steps | lr | accuracy |
|---|---:|---:|---:|---:|
| PyTorch default | 64 / 44 | 1500 / 6000 | 1e-3, 3e-3, 1e-2 | 0.114–0.128 (all 12 settings) |
| 1.0 | 64 | 1500 / 6000 | 3e-3, 1e-2 | 0.118–0.121 |
| 3.0 | 64 | 1500 / 6000 | 3e-3, 1e-2 | **1.000** (all settings) |

### Consolidated, 3 seeds (Hamiltonian rows: seeds 0, 1 from the first run + seed 2)

| task | LSTM (forget bias 3) | ham_conservative | ham_learned | ham_geometry | learned γ (ham_learned) |
|---|---:|---:|---:|---:|---:|
| recall_first | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.318 ± 0.272 | 0.035 |
| count | **1.000 ± 0.000** | 0.738 ± 0.051 | **1.000 ± 0.000** | 0.277 ± 0.003 | 0.153 |
| lag | **1.000 ± 0.000** | 0.934 ± 0.012 | 0.996 ± 0.002 | 0.516 ± 0.033 | 0.236 |
| assoc | **0.688 ± 0.040** | 0.398 ± 0.013 | 0.399 ± 0.014 | 0.404 ± 0.007 | 0.085 |

`ham_geometry` gradient norms during training: median ~1e3 (vs ~2 for
`ham_learned`), spikes to 1e5. The coupled matter–geometry dynamics make
backpropagation explode, and the geometry parameters barely move from
their initial values (γ_Φ 0.05 → 0.054, λ 1.0 → 0.98).

## Run steps=1500 T=40 hidden=32 (particle dim) charge_dim=16 substeps=2 dt=0.5 seeds=[0, 1, 2] — `particles` (many-body)

Four separate invocations (one per task, run in parallel on CPU, which
inflates wall time), consolidated here.

| task | model | accuracy | params | learned γ median (90th pct) | geometry γ / stiffness λ | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| assoc | `particles` | 0.392 ± 0.231 | 2,651 | 0.064 (0.080) | — | 182 |
| recall_first | `particles` | 0.125 ± 0.015 | 2,651 | 0.051 (0.059) | — | 2466 |
| count | `particles` | 0.250 ± 0.029 | 2,651 | 0.049 (0.061) | — | 2245 |
| lag | `particles` | 0.150 ± 0.018 | 2,651 | 0.055 (0.086) | — | 2222 |

Per seed: assoc 0.501 / 0.548 / 0.127; recall_first 0.119 / 0.142 / 0.114;
count 0.216 / 0.270 / 0.263; lag 0.129 / 0.160 / 0.162.

Pre-registered hypothesis 3 is **largely not supported**: a tie with
`ham_learned` on assoc within (large) seed noise, two seeds clearly above
it (0.50, 0.55) and one stuck at chance; at chance on recall_first, count
and lag, with γ never leaving its 0.05 start on those tasks.
Chance accuracy: recall_first, lag and assoc 0.125; count ≈ 0.2. Learned γ starts at 0.05.

## Run steps=1500 T=40 hidden=64 (particle dim 32) substeps=2 dt=0.5 seeds=[0, 1, 2, 3, 4] lstm_forget_bias=3.0 — hypothesis 4 (dense birth)

| task | model | accuracy | params | learned γ median (90th pct) | geometry γ / stiffness λ | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| assoc | `lstm` | 0.678 ± 0.060 | 21,768 | — | — | 5 |
| assoc | `particles` | 0.333 ± 0.175 | 2,652 | 0.067 (0.088) | — | 103 |
| assoc | `particles_dense` | 0.571 ± 0.206 | 2,652 | 0.044 (0.054) | — | 78 |

Per seed: lstm 0.714 / 0.709 / 0.643 / 0.591 / 0.735; particles 0.483 /
0.546 / 0.127 / 0.265 / 0.243; particles_dense 0.521 / 0.929 / 0.506 /
0.395 / 0.504 (1 of 5 seeds above 0.9).

Hypothesis 4 **not supported**: dense birth is below the LSTM on average
(0.571 vs 0.678). It is above the sparse-born particle model (0.333), a
difference of about 2 combined standard errors: suggestive, not
established. The particle models have ~8× fewer parameters.

### Diagnostic (not a benchmark): Big Bang variants, embedding dim 32, 3 seeds

Script kept outside the repo; settings as above except embedding 32;
`recall_first` 600 steps, `assoc` 1500 steps. Gradient norm = max over
the first 100 steps (clipped to 1.0 for the update).

| variant | assoc | recall_first (600 steps) | max grad norm, recall |
|---|---:|---:|---:|
| base (sparse birth) | 0.281 ± 0.124 | 0.121 ± 0.008 | 2 – 279 |
| dense (spawn 0.1) | 0.812 ± 0.307 (0.98, 0.46, 0.99) | 0.150 ± 0.012 | 121 – 46,885 |
| dense + expansion (κ₀ = −0.02) | 0.618 ± 0.268 | 0.123 ± 0.005 | < 2 |
| dense + 4 primordial particles | 0.516 ± 0.125 | 0.233 ± 0.120 (0.36, 0.13, 0.21) | 2,865 – 4.4e6 |
| all three ("Big Bang") | 0.487 ± 0.014 | 0.121 ± 0.006 | < 2.1 |

With expansion allowed, learned κ went *negative* on recall (−0.03,
expanding) and back *positive* on assoc (+0.10, confining).

### Audit diagnostic 2026-11-10 (not a benchmark): `assoc` budget and a transformer baseline

`python -m benchmarks.diagnostics.assoc_audit MODEL SEED STEPS`, benchmark
settings (embedding 16, lr 3e-3, batch 64), evaluated on 4096 sequences.

| model | params | steps | accuracy per seed |
|---|---:|---:|---|
| transformer (2 layers, 2 heads) | 6,984 | 1500 | 1.000, 1.000, 1.000 (seeds 0–2) |
| lstm | 21,768 | 1500 | 0.704 (seed 0) |
| lstm | 21,768 | 6000 | 0.991, 0.990 (seeds 0, 1) |
| particles_dense | 2,652 | 1500 | 0.525, 0.929, 0.503 (seeds 0–2) |
| particles_dense | 2,652 | 6000 | 1.000, 0.859, 0.785 (seeds 0, 2, 3) |
| particles (sparse birth) | 2,652 | 6000 | 0.649 (seed 2; 0.127 at 1500) |

Reading: at 1500 steps `assoc` measures learning *speed*, not ability.
The LSTM's 0.69 is an under-trained number (0.99 at 6000 steps), and the
seeds of `particles_dense` that looked "stalled near 0.5" were still
learning (their errors are spread evenly over the four pairs, and they
reach 0.79–1.00 at 6000 steps). The bar on this task is the transformer:
1.000 in 1500 steps on every seed.

## Run 2026-11-10: length generalisation (hypothesis 5). steps=1500, trained at T=40, hidden=64, seeds=[0, 1, 2, 3, 4], lstm_forget_bias=3.0

Own readout, mean ± std over 5 seeds (four invocations, one per model, consolidated).

| task | model | T=40 | T=100 | T=200 | T=400 |
|---|---|---:|---:|---:|---:|
| recall_first | `lstm` | 1.000 ± 0.000 | 0.998 ± 0.005 | 0.918 ± 0.114 | 0.865 ± 0.168 |
| recall_first | `transformer` | 1.000 ± 0.000 | 0.942 ± 0.113 | 0.697 ± 0.307 | 0.645 ± 0.305 |
| recall_first | `ham_conservative` | 1.000 ± 0.000 | 0.146 ± 0.063 | 0.146 ± 0.019 | 0.127 ± 0.017 |
| recall_first | `ham_learned` | 1.000 ± 0.000 | 0.284 ± 0.119 | 0.243 ± 0.023 | 0.256 ± 0.003 |
| count | `lstm` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 |
| count | `transformer` | 1.000 ± 0.000 | 0.526 ± 0.032 | 0.320 ± 0.026 | 0.199 ± 0.015 |
| count | `ham_conservative` | 0.758 ± 0.023 | 0.430 ± 0.013 | 0.294 ± 0.008 | 0.222 ± 0.005 |
| count | `ham_learned` | 0.991 ± 0.016 | 0.960 ± 0.071 | 0.934 ± 0.106 | 0.917 ± 0.122 |
| lag | `lstm` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| lag | `transformer` | 1.000 ± 0.000 | 0.494 ± 0.009 | 0.311 ± 0.005 | 0.221 ± 0.006 |
| lag | `ham_conservative` | 0.933 ± 0.007 | 0.668 ± 0.003 | 0.488 ± 0.003 | 0.355 ± 0.009 |
| lag | `ham_learned` | 0.996 ± 0.001 | 0.995 ± 0.002 | 0.995 ± 0.002 | 0.994 ± 0.002 |

Hypothesis 5: first part **not supported** (`ham_conservative` is at
chance on recall by T=100; the LSTM is the most robust). Second part
supported: `ham_learned` holds `count` and `lag` far better than
`ham_conservative`. Learned γ medians were again 0.033–0.037 (recall),
0.12–0.16 (count), 0.23–0.25 (lag) on all 5 seeds.

Near the training length, `recall_first`, 3 seeds (own readout):

| model | T=20 | 30 | 36 | 38 | 39 | 40 | 41 | 42 | 44 | 50 | 60 | 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lstm` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `ham_conservative` | 0.257 | 0.998 | 0.000 | 0.539 | 0.998 | 1.000 | 0.996 | 0.221 | 0.000 | 0.954 | 0.208 | 0.049 |
| `ham_learned` | 0.057 | 0.973 | 0.000 | 0.363 | 1.000 | 1.000 | 1.000 | 0.412 | 0.000 | 0.944 | 0.115 | 0.106 |

## Run 2026-11-10: cyclic coordinates (hypothesis 6). recall_first, steps=1500, hidden=64, n_cyclic=16, seeds=[0, 1, 2, 3, 4]

Own readout, mean ± std over 5 seeds.

| training lengths | model | T=40 | T=100 | T=200 | T=400 | T=1000 |
|---|---|---:|---:|---:|---:|---:|
| 40 | `lstm` | 1.000 ± 0.000 | 0.998 ± 0.005 | 0.918 ± 0.114 | 0.865 ± 0.168 | 0.830 ± 0.195 |
| 40 | `ham_conservative` | 1.000 ± 0.000 | 0.146 ± 0.063 | 0.146 ± 0.019 | 0.127 ± 0.017 | 0.131 ± 0.008 |
| 40 | `ham_cyclic` | 1.000 ± 0.000 | 0.652 ± 0.259 | 0.353 ± 0.124 | 0.236 ± 0.072 | 0.178 ± 0.031 |
| 40 | `ham_cyclic_learned` | 1.000 ± 0.000 | 0.778 ± 0.188 | 0.459 ± 0.193 | 0.350 ± 0.218 | 0.251 ± 0.147 |
| 20–40 | `lstm` | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.957 ± 0.061 | 0.928 ± 0.103 | 0.875 ± 0.084 |
| 20–40 | `ham_conservative` | 0.661 ± 0.326 | 0.194 ± 0.094 | 0.154 ± 0.050 | 0.154 ± 0.046 | 0.140 ± 0.036 |
| 20–40 | `ham_cyclic` | 1.000 ± 0.000 | 0.900 ± 0.134 | 0.399 ± 0.196 | 0.182 ± 0.069 | 0.129 ± 0.007 |
| 20–40 | `ham_cyclic_learned` | 1.000 ± 0.000 | 0.977 ± 0.052 | 0.459 ± 0.255 | 0.252 ± 0.152 | 0.131 ± 0.011 |

Hypothesis 6 **not supported as stated** (neither condition reaches 0.9
at T=400 with the model's own readout). Cyclic coordinates do help at
T=100 (0.65–0.98 vs 0.15–0.19), and without them the conservative model
cannot even fit mixed training lengths (0.66 at T=40).

### Diagnostic (not a benchmark): what is still in the state? Fresh linear readout, trained at lengths 20–40, 5 seeds

`TRAIN_LEN_MIN=20 python -m benchmarks.diagnostics.memory_probe MODEL SEED 40 100 400 1000`.
"own" is the trained readout; "probe" is a ridge readout fitted on the
raw final state of 4096 fresh sequences of that length, tested on 2048.

| model | T=100 own / probe | T=400 own / probe | T=1000 own / probe | probe per seed at T=1000 |
|---|---|---|---|---|
| `lstm` | 1.000 / 1.000 | 0.934 / 0.986 ± 0.020 | 0.869 / 0.899 ± 0.107 | 0.986, 0.869, 0.731, 0.915, 0.995 |
| `ham_conservative` | 0.184 / 0.973 | 0.157 / 0.643 ± 0.210 | 0.140 / 0.327 ± 0.197 | 0.226, 0.661, 0.162, 0.246, 0.339 |
| `ham_cyclic` | 0.906 / 1.000 | 0.170 / 1.000 ± 0.000 | 0.125 / **1.000 ± 0.000** | 1.0 ×5 |
| `ham_cyclic_learned` | 0.979 / 1.000 | 0.240 / 1.000 ± 0.000 | 0.131 / **1.000 ± 0.000** | 1.0 ×5 |

`benchmarks.diagnostics.cyclic_probe` (2 seeds per model): a probe on the
16 cyclic velocities alone also scores 1.000 at T=1000. Impulse size in
the cyclic directions: first tokens 1.5–2.1 RMS, distractors 0.13–0.16
RMS with a mean (drift per token) of 0.012–0.019. Over 1000 tokens the
drift is ~7× the signal, which a fixed readout cannot follow and a
re-fitted one can.

## Run 2026-11-10: a selection rule for the conserved momenta (hypotheses 7–10). recall_first, steps=1500, hidden=64, n_cyclic=16, seeds=[0, 1, 2, 3, 4]

Own readout, mean ± std over 5 seeds. Criteria for all four hypotheses:
≥ 0.9 at T=400 and at least the LSTM's accuracy at T=1000.

| hyp. | training lengths | model / setting | T=40 | T=100 | T=200 | T=400 | T=1000 |
|---|---|---|---:|---:|---:|---:|---:|
| – | 20–40 | `lstm` (forget bias 3) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.957 ± 0.061 | 0.928 ± 0.103 | 0.875 ± 0.084 |
| – | 20–100 | `lstm` (forget bias 3) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.983 ± 0.038 |
| 7 | 20–40 | `ham_charge` (sigmoid charge) | 1.000 ± 0.000 | 0.809 ± 0.253 | 0.298 ± 0.228 | 0.145 ± 0.020 | 0.122 ± 0.005 |
| 7 | 20–100 | `ham_charge` | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.993 ± 0.014 | 0.630 ± 0.282 | 0.205 ± 0.057 |
| 7 | 20–100 | `ham_cyclic_learned` | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.002 | 0.650 ± 0.276 | 0.248 ± 0.175 |
| 8 | 20–40 | `ham_charge`, write cost 0.1 | 1.000 ± 0.000 | 0.997 ± 0.003 | 0.612 ± 0.337 | 0.324 ± 0.170 | 0.159 ± 0.051 |
| 8 | 20–40 | `ham_charge`, write cost 1.0 | 1.000 ± 0.000 | 0.997 ± 0.007 | 0.810 ± 0.290 | 0.490 ± 0.215 | 0.215 ± 0.123 |
| 8 | 20–40 | `ham_cyclic_learned`, write cost 0.1 | 1.000 ± 0.000 | 0.984 ± 0.023 | 0.618 ± 0.253 | 0.276 ± 0.150 | 0.171 ± 0.060 |
| 8 | 20–40 | `ham_cyclic_learned`, write cost 1.0 | 0.982 ± 0.041 | 0.869 ± 0.293 | 0.728 ± 0.359 | 0.422 ± 0.183 | 0.204 ± 0.125 |
| 9 | 20–40 | `ham_charge`, net-force penalty 10 | 1.000 ± 0.000 | 0.945 ± 0.093 | 0.614 ± 0.368 | 0.404 ± 0.262 | 0.137 ± 0.025 |
| 9 | 20–40 | `ham_charge`, net-force penalty 100 | 1.000 ± 0.000 | 0.995 ± 0.011 | 0.801 ± 0.169 | 0.375 ± 0.205 | 0.163 ± 0.051 |
| 9 | 20–40 | `ham_cyclic_learned`, net-force penalty 10 | 1.000 ± 0.000 | 0.971 ± 0.026 | 0.570 ± 0.213 | 0.320 ± 0.079 | 0.135 ± 0.009 |
| 9 | 20–40 | `ham_cyclic_learned`, net-force penalty 100 | 1.000 ± 0.000 | 0.918 ± 0.083 | 0.532 ± 0.172 | 0.324 ± 0.114 | 0.227 ± 0.091 |
| 10 | 20–40 | `ham_neutral` (ReLU charge), charge cost 0 (control) | 1.000 ± 0.000 | 0.747 ± 0.297 | 0.216 ± 0.127 | 0.154 ± 0.050 | 0.142 ± 0.048 |
| 10 | 20–40 | `ham_neutral`, charge cost 0.01 | 0.975 ± 0.057 | 0.851 ± 0.332 | 0.822 ± 0.397 | 0.822 ± 0.399 | 0.819 ± 0.405 |
| 10 | 20–40 | `ham_neutral`, charge cost 0.1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** |
| 10 | 20–40 | `ham_neutral`, charge cost 1.0 | 0.992 ± 0.018 | 0.979 ± 0.029 | 0.966 ± 0.054 | 0.964 ± 0.055 | 0.963 ± 0.058 |

Hypotheses 7, 8 and 9 **not supported**. Hypothesis 10 **supported** at
charge costs 0.1 and 1.0; the cost-0 control fails as predicted, and at
0.01 one seed of five fails at every length (the risk noted in advance).

Diagnostics behind the chain:

* `benchmarks.diagnostics.readout_swap`: replacing only the conserved
  part of the state with its long-run value breaks the readout; the
  ordinary modes are harmless. Sigmoid charges never closed (distractor
  charges 0.01–0.98).
* `benchmarks.diagnostics.transfer_probe`: a readout fitted at T=40
  fails at T=1000, and the same readout on re-centred features scores
  1.000 (4/4 runs). The failure is a deterministic drift, i.e. the mean
  distractor impulse; the random part is harmless.
* `readout_swap ham_neutral 0 40 0.1`: charges of the 8 first-token
  symbols 0.51–1.32, charges of the 8 distractor symbols all exactly
  0.0, drift per token exactly 0.

### Robustness: best LSTM setting, and 10,000 tokens

All trained on lengths 20–40, 5 seeds, own readout. The LSTM gets 13
settings; `ham_neutral` is the cost-0.1 setting from above (chosen on
the same seeds at T ≤ 1000).

| model | forget bias | lr | steps | T=40 | T=400 | T=1000 | T=4000 | T=10000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ham_neutral` (cost 0.1) | – | 3e-3 | 1500 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** |
| `lstm` | 3 | 1e-3 | 1500 | 1.000 ± 0.000 | 0.935 ± 0.069 | 0.871 ± 0.102 | 0.841 ± 0.137 | 0.834 ± 0.132 |
| `lstm` | 3 | 3e-3 | 1500 | 1.000 ± 0.000 | 0.928 ± 0.103 | 0.875 ± 0.084 | 0.835 ± 0.060 | 0.825 ± 0.070 |
| `lstm` | 3 | 1e-2 | 1500 | 1.000 ± 0.000 | 0.972 ± 0.063 | 0.965 ± 0.054 | 0.950 ± 0.069 | 0.948 ± 0.071 |
| `lstm` | 3 | 3e-3 | 6000 | 1.000 ± 0.000 | 0.926 ± 0.090 | 0.880 ± 0.088 | 0.858 ± 0.074 | 0.852 ± 0.077 |
| `lstm` | 5 | 1e-3 | 1500 | 1.000 ± 0.000 | 0.996 ± 0.007 | 0.934 ± 0.066 | 0.857 ± 0.142 | 0.842 ± 0.162 |
| `lstm` | 5 | 3e-3 | 1500 | 1.000 ± 0.000 | 0.943 ± 0.120 | 0.934 ± 0.109 | 0.903 ± 0.157 | 0.896 ± 0.173 |
| `lstm` | 5 | 1e-2 | 1500 | 1.000 ± 0.000 | 0.976 ± 0.053 | 0.953 ± 0.059 | 0.911 ± 0.061 | 0.903 ± 0.055 |
| `lstm` | 5 | 3e-3 | 6000 | 1.000 ± 0.000 | 0.961 ± 0.070 | 0.927 ± 0.107 | 0.876 ± 0.147 | 0.872 ± 0.158 |
| `lstm` | 8 | 1e-3 | 1500 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.972 ± 0.063 | 0.952 ± 0.107 | 0.934 ± 0.112 |
| `lstm` | 8 | 3e-3 | 1500 | 1.000 ± 0.000 | 0.972 ± 0.062 | 0.950 ± 0.112 | 0.948 ± 0.116 | 0.946 ± 0.120 |
| `lstm` | 8 | 1e-2 | 1500 | 1.000 ± 0.000 | 0.981 ± 0.043 | 0.922 ± 0.175 | 0.928 ± 0.162 | 0.921 ± 0.178 |
| `lstm` | 8 | 3e-3 | 6000 | 1.000 ± 0.000 | 0.994 ± 0.013 | 0.954 ± 0.104 | 0.947 ± 0.119 | 0.943 ± 0.126 |

Reading it honestly: the best LSTM settings reach 0.95 at 10,000 tokens
with a seed spread of 0.07–0.12, so on mean ± std the gap to 1.000 is
about one LSTM standard deviation (5 seeds). What differs in kind is
that `ham_neutral` is exact on every seed at every length, because
nothing touches the memory after the write. Limits: one task; the first
token and the distractors come from different symbol sets, so a
per-token charge is sufficient here by construction; the charge cost
needed a sweep (0.01 loses a seed).

## Run 2026-11-10: where a per-token charge is and is not enough (hypothesis 11). steps=1500, hidden=64, trained on 20–40, seeds=[0, 1, 2, 3, 4]

Own readout, mean ± std over 5 seeds. `ham_neutral` at charge cost 0.1.

| task | model | T=40 | T=400 | T=1000 | T=4000 |
|---|---|---:|---:|---:|---:|
| `recall_mid` | `ham_neutral` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** |
| `recall_mid` | `lstm`, forget bias 3 | 1.000 ± 0.000 | 0.984 ± 0.035 | 0.978 ± 0.045 | 0.951 ± 0.052 |
| `recall_mid` | `lstm`, forget bias 8 | 1.000 ± 0.000 | 0.994 ± 0.012 | 0.972 ± 0.034 | 0.891 ± 0.127 |
| `recall_same` | `ham_neutral` | 0.307 ± 0.097 | 0.191 ± 0.071 | 0.161 ± 0.056 | 0.146 ± 0.041 |
| `recall_same` | `lstm`, forget bias 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.001** |
| `recall_same` | `lstm`, forget bias 8 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

Hypothesis 11 **supported, both halves**. Position does not matter to
the neutral-token model. When distractors share the item's symbols it
fails outright, and does not even fit the training lengths (0.307 at
T=40), while the LSTM is exact to T=4000: its input gate depends on its
state, so it can close after the first token. The charge here depends
only on the current token.

## Run 2026-11-10: a state-dependent (exclusion) charge (hypothesis 12). hidden=64, trained on 20–40, seeds=[0, 1, 2, 3, 4]

`ham_exclusion`, own readout, mean ± std over 5 seeds. The registered
test is the 1500-step block; the 4500-step block is a follow-up added
after seeing it (research-hygiene rule 8), not part of the registration.

| steps | task | charge cost | T=40 | T=400 | T=1000 | T=4000 |
|---:|---|---:|---:|---:|---:|---:|
| 1500 | `recall_same` | 0 | 0.971 ± 0.064 | 0.972 ± 0.062 | 0.971 ± 0.065 | 0.977 ± 0.052 |
| 1500 | `recall_same` | 0.1 | 0.961 ± 0.087 | 0.957 ± 0.095 | 0.960 ± 0.090 | 0.956 ± 0.098 |
| 1500 | `recall_mid` | 0 | 0.976 ± 0.053 | 0.932 ± 0.096 | 0.909 ± 0.135 | 0.889 ± 0.156 |
| 1500 | `recall_mid` | 0.1 | 0.949 ± 0.070 | 0.948 ± 0.071 | 0.954 ± 0.064 | 0.953 ± 0.065 |
| 1500 | `recall_first` | 0 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 1500 | `recall_first` | 0.1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 4500 | `recall_same` | 0 | 0.997 ± 0.007 | 0.997 ± 0.007 | 0.994 ± 0.013 | 0.997 ± 0.007 |
| 4500 | `recall_same` | 0.1 | 0.975 ± 0.055 | 0.948 ± 0.072 | 0.949 ± 0.071 | 0.954 ± 0.062 |
| 4500 | `recall_mid` | 0 | 0.978 ± 0.048 | 0.946 ± 0.077 | 0.922 ± 0.124 | 0.903 ± 0.157 |
| 4500 | `recall_mid` | 0.1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

For reference (hypothesis 11): `ham_neutral` 0.146 and LSTM 1.000 on
`recall_same` at T=4000; `ham_neutral` 1.000 and LSTM 0.951 on
`recall_mid`.

Hypothesis 12 **not supported as stated**: `recall_same` reaches
0.96–0.98 at T=4000, not 0.99, and `recall_mid` fell from 1.000 to
0.89–0.95. What the numbers do show:

* The exclusion charge does the job it was built for. `recall_same`
  goes from 0.31 (cannot fit T=40) to 0.96–0.98, and accuracy is flat
  from T=40 to T=4000: the leak named as the risk did not appear there.
* The shortfall is at the training length and comes in steps of one
  symbol in eight. Failing seeds sit at 0.87–0.89 at T=40 (7/8 = 0.875)
  with the others at 1.000, the signature of one symbol whose charge
  died (the dead-ReLU risk noted under hypothesis 10). Not verified by
  inspecting the charges.
* No single cost passes both tasks even at 3× the steps. Cost 0 solves
  `recall_same` (0.997) but leaks on `recall_mid` (one seed decays with
  length); cost 0.1 solves `recall_mid` (1.000) but leaves one seed
  short on `recall_same`.
* The LSTM is still the better model on `recall_same` (1.000, no
  tuning).
