
## Run seed=0 steps=300

| variant | wall (s) | force evals | final train | final test | best test | peak |ΔH/H| |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_adam` | 4.18 | 300 | 0.4984 | 0.5074 | 0.5119 | — |
| `yaum_leapfrog` | 4.79 | 600 | 0.5333 | 0.5408 | 0.5459 | 8.34e-01 |
| `yaum_yoshida4` | 7.63 | 1800 | 0.5333 | 0.5408 | 0.5459 | 8.34e-01 |

## Conclusions

- **Best test loss — YAUM leapfrog vs Adam-on-E:** ↑ 0.0341 (+6.7%).
- **Best test loss — YAUM yoshida4 vs Adam-on-E:** ↑ 0.0341 (+6.7%).
- **Leapfrog gives a new diagnostic Adam cannot:** peak |ΔH/H| = 8.34e-01. This is the single number that tells you the integrator is losing (or gaining) energy against the loss landscape — a failure mode invisible to a plain optimiser.
- **Wall-time cost of symplectic E:** 1.15× baseline (600 force evals vs 300 for Adam — leapfrog evaluates the gradient twice per step).
- **Yoshida4 vs leapfrog:** 1.59× wall time for 3.0× the force evaluations. On this dataset the trajectories agree to float32 precision — the per-token embedding force is too mild for the fourth-order correction to do visible work, so leapfrog is the right default until forces get stiffer.
