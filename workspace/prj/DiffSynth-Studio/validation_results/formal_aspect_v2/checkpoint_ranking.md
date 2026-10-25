# Formal v2 checkpoint ranking

The numeric score is exploratory; guardrails, Pareto status, and blind review take precedence.

| Rank | Checkpoint | Guardrails | CI robust | Pareto | Sharpness (95% CI) | Motion (95% CI) | Warp reduction (95% CI) | Target-motion MAE reduction (95% CI) | Score |
|---:|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| 1 | `step-5250.safetensors` | yes | no | yes | 0.961 [0.772, 1.177] | 0.873 [0.645, 1.087] | 0.328 [0.146, 0.483] | 0.216 [-0.069, 0.449] | 0.800 |
| 2 | `step-750.safetensors` | yes | no | yes | 0.905 [0.736, 1.077] | 0.864 [0.737, 0.988] | 0.338 [0.225, 0.444] | 0.177 [-0.019, 0.352] | 0.200 |
