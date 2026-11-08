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
