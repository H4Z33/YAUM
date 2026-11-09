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
