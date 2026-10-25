# Formal v2 checkpoint ranking

The numeric score is exploratory; guardrails, Pareto status, and blind review take precedence.

| Rank | Checkpoint | Guardrails | CI robust | Pareto | Sharpness (95% CI) | Motion (95% CI) | Warp reduction (95% CI) | Target-motion MAE reduction (95% CI) | Score |
|---:|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| 1 | `step-750.safetensors` | yes | no | yes | 0.859 [0.741, 0.959] | 0.956 [0.722, 1.198] | 0.356 [0.234, 0.472] | 0.172 [-0.004, 0.332] | 0.697 |
| 2 | `step-5250.safetensors` | yes | no | yes | 0.882 [0.754, 0.990] | 0.893 [0.621, 1.162] | 0.308 [0.205, 0.413] | 0.204 [0.033, 0.333] | 0.656 |
| 3 | `step-4500.safetensors` | yes | no | yes | 0.875 [0.740, 0.993] | 0.827 [0.542, 1.107] | 0.339 [0.232, 0.449] | 0.216 [0.062, 0.334] | 0.631 |
| 4 | `step-5986.safetensors` | yes | no | yes | 0.853 [0.736, 0.949] | 0.875 [0.584, 1.161] | 0.318 [0.218, 0.420] | 0.233 [0.065, 0.368] | 0.597 |
| 5 | `step-3000.safetensors` | yes | no | yes | 0.858 [0.732, 0.963] | 0.890 [0.598, 1.184] | 0.329 [0.218, 0.443] | 0.183 [0.033, 0.300] | 0.528 |
| 6 | `step-3750.safetensors` | yes | no | yes | 0.865 [0.729, 0.981] | 0.853 [0.568, 1.135] | 0.312 [0.208, 0.418] | 0.176 [0.021, 0.292] | 0.456 |
| 7 | `step-2250.safetensors` | yes | no | no | 0.881 [0.764, 0.979] | 0.871 [0.591, 1.150] | 0.292 [0.162, 0.419] | 0.121 [-0.025, 0.249] | 0.385 |
| 8 | `step-1500.safetensors` | no | no | yes | 0.839 [0.725, 0.945] | 0.993 [0.757, 1.236] | 0.266 [0.143, 0.388] | 0.189 [0.062, 0.292] | 0.402 |
