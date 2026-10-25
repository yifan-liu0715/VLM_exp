# Multi-seed robustness summary

Seeds: 42, 123, 20260712. Confidence intervals use 10,000 hierarchical bootstrap draws over seeds and samples.

| Variant | Metric | Point | 95% CI | P(improves) |
|---|---|---:|---:|---:|
| `formal_v2_step-750` | sharpness_retention | 0.830 | [0.715, 0.952] | 0.002 |
| `formal_v2_step-750` | motion_retention | 0.816 | [0.681, 0.985] | 0.017 |
| `formal_v2_step-750` | warp_error_reduction | 0.584 | [0.366, 0.698] | 1.000 |
| `formal_v2_step-750` | target_motion_mae_reduction | 0.360 | [0.168, 0.495] | 1.000 |
| `formal_v2_step-750` | reference_ssim_delta | 0.001 | [0.000, 0.001] | 1.000 |
| `formal_v2_step-5250` | sharpness_retention | 0.826 | [0.648, 0.989] | 0.018 |
| `formal_v2_step-5250` | motion_retention | 0.753 | [0.599, 0.934] | 0.004 |
| `formal_v2_step-5250` | warp_error_reduction | 0.562 | [0.325, 0.695] | 1.000 |
| `formal_v2_step-5250` | target_motion_mae_reduction | 0.388 | [0.191, 0.533] | 1.000 |
| `formal_v2_step-5250` | reference_ssim_delta | 0.001 | [0.001, 0.002] | 1.000 |

## Direct comparison: formal_v2_step-750 minus formal_v2_step-5250

| Metric | Difference | 95% CI | P(left better) |
|---|---:|---:|---:|
| sharpness_retention | +0.004 | [-0.099, +0.101] | 0.517 |
| motion_retention | +0.062 | [+0.009, +0.118] | 0.988 |
| warp_error_reduction | +0.022 | [-0.027, +0.074] | 0.797 |
| target_motion_mae_reduction | -0.028 | [-0.105, +0.053] | 0.246 |
| reference_ssim_delta | -0.000 | [-0.000, +0.000] | 0.148 |
