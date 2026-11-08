# Benchmark results

Appended by `python -m benchmarks.compare_training`. Runs before the
audit fixes (November 2026) were removed: their `peak |ΔH/H|` measured how
far the loss had fallen, not integrator error, and the "Fisher" mass
was a constant pinned to its floor, so those numbers are not comparable.

## Run steps=600 seeds=[0, 1, 2]

| variant | held-out loss | ‖E−E₀‖/‖E₀‖ | max one-step \|ΔH/H\| | wall (s) |
|---|---:|---:|---:|---:|
| `frozen_E` | 0.2206 ± 0.0073 | 0.00 | — | 1.5 |
| `adam_E` | 0.1978 ± 0.0049 | 0.29 | — | 1.6 |
| `adam_E_tuned` | 0.1708 ± 0.0046 | 1.88 | — | 1.5 |
| `yaum_static` | 0.1989 ± 0.0038 | 0.33 | 3.1e-07 | 3.3 |
| `yaum_fisher` | 0.1686 ± 0.0080 | 7.61 | 5.3e-07 | 3.3 |
| `yaum_damped` | 0.1699 ± 0.0065 | 2.76 | 4.3e-05 | 3.4 |

Lowest held-out loss: `yaum_fisher`.
`yaum_static` vs `frozen_E`: -0.0216 (negative = the dynamics on E help at all).
`yaum_damped` vs `adam_E_tuned`: -0.0008 (seed std ≈ 0.0065; within seed noise — a tie).
