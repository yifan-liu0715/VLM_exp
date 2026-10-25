# Formal validation subgroup audit

A checkpoint should not be selected solely from the global mean. Subgroup failures are veto signals.

| Variant | Group | N | Sharpness | Motion | Warp reduction | Target-motion MAE reduction | Ref SSIM Δ | Guardrails |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `formal_v2_step-5250` | bucket=1280x512 | 5 | 1.266 | 0.602 | 0.522 | 0.489 | +0.0020 | fail |
| `formal_v2_step-5250` | bucket=1280x704 | 5 | 0.929 | 0.714 | 0.175 | 0.033 | +0.0015 | pass |
| `formal_v2_step-5250` | bucket=704x1280 | 5 | 0.716 | 1.316 | 0.273 | 0.157 | -0.0006 | fail |
| `formal_v2_step-5250` | content=dance | 5 | 0.716 | 1.316 | 0.273 | 0.157 | -0.0006 | fail |
| `formal_v2_step-5250` | content=talking_face | 10 | 1.116 | 0.647 | 0.380 | 0.294 | +0.0018 | fail |
| `formal_v2_step-750` | bucket=1280x512 | 5 | 1.148 | 0.714 | 0.410 | 0.455 | +0.0015 | pass |
| `formal_v2_step-750` | bucket=1280x704 | 5 | 0.925 | 0.736 | 0.232 | -0.011 | +0.0013 | pass |
| `formal_v2_step-750` | bucket=704x1280 | 5 | 0.678 | 1.140 | 0.338 | 0.117 | -0.0005 | fail |
| `formal_v2_step-750` | content=dance | 5 | 0.678 | 1.140 | 0.338 | 0.117 | -0.0005 | fail |
| `formal_v2_step-750` | content=talking_face | 10 | 1.049 | 0.723 | 0.337 | 0.256 | +0.0014 | pass |
