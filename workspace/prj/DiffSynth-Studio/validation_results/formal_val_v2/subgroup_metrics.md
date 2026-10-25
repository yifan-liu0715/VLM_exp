# Formal validation subgroup audit

A checkpoint should not be selected solely from the global mean. Subgroup failures are veto signals.

| Variant | Group | N | Sharpness | Motion | Warp reduction | Target-motion MAE reduction | Ref SSIM Δ | Guardrails |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `formal_v2_step-1500` | bucket=1280x512 | 5 | 0.927 | 0.642 | 0.345 | 0.088 | +0.0016 | fail |
| `formal_v2_step-1500` | bucket=1280x704 | 5 | 0.917 | 0.769 | 0.099 | -0.000 | +0.0013 | pass |
| `formal_v2_step-1500` | bucket=704x1280 | 5 | 0.671 | 1.480 | 0.290 | 0.271 | -0.0003 | fail |
| `formal_v2_step-1500` | content=dance | 5 | 0.671 | 1.480 | 0.290 | 0.271 | -0.0003 | fail |
| `formal_v2_step-1500` | content=talking_face | 10 | 0.923 | 0.709 | 0.242 | 0.043 | +0.0014 | pass |
| `formal_v2_step-2250` | bucket=1280x512 | 5 | 0.963 | 0.481 | 0.375 | 0.050 | +0.0021 | fail |
| `formal_v2_step-2250` | bucket=1280x704 | 5 | 0.938 | 0.648 | 0.240 | -0.059 | +0.0014 | fail |
| `formal_v2_step-2250` | bucket=704x1280 | 5 | 0.733 | 1.390 | 0.265 | 0.192 | -0.0004 | fail |
| `formal_v2_step-2250` | content=dance | 5 | 0.733 | 1.390 | 0.265 | 0.192 | -0.0004 | fail |
| `formal_v2_step-2250` | content=talking_face | 10 | 0.954 | 0.570 | 0.318 | -0.005 | +0.0017 | fail |
| `formal_v2_step-3000` | bucket=1280x512 | 5 | 0.947 | 0.457 | 0.376 | 0.041 | +0.0022 | fail |
| `formal_v2_step-3000` | bucket=1280x704 | 5 | 0.916 | 0.648 | 0.243 | 0.032 | +0.0014 | fail |
| `formal_v2_step-3000` | bucket=704x1280 | 5 | 0.702 | 1.460 | 0.339 | 0.265 | -0.0005 | fail |
| `formal_v2_step-3000` | content=dance | 5 | 0.702 | 1.460 | 0.339 | 0.265 | -0.0005 | fail |
| `formal_v2_step-3000` | content=talking_face | 10 | 0.936 | 0.558 | 0.320 | 0.037 | +0.0018 | fail |
| `formal_v2_step-3750` | bucket=1280x512 | 5 | 0.964 | 0.420 | 0.327 | 0.020 | +0.0020 | fail |
| `formal_v2_step-3750` | bucket=1280x704 | 5 | 0.915 | 0.637 | 0.224 | 0.020 | +0.0014 | fail |
| `formal_v2_step-3750` | bucket=704x1280 | 5 | 0.703 | 1.400 | 0.341 | 0.263 | -0.0003 | fail |
| `formal_v2_step-3750` | content=dance | 5 | 0.703 | 1.400 | 0.341 | 0.263 | -0.0003 | fail |
| `formal_v2_step-3750` | content=talking_face | 10 | 0.946 | 0.535 | 0.284 | 0.020 | +0.0017 | fail |
| `formal_v2_step-4500` | bucket=1280x512 | 5 | 0.969 | 0.383 | 0.354 | 0.050 | +0.0023 | fail |
| `formal_v2_step-4500` | bucket=1280x704 | 5 | 0.922 | 0.626 | 0.264 | 0.077 | +0.0015 | fail |
| `formal_v2_step-4500` | bucket=704x1280 | 5 | 0.721 | 1.370 | 0.362 | 0.301 | -0.0005 | fail |
| `formal_v2_step-4500` | content=dance | 5 | 0.721 | 1.370 | 0.362 | 0.301 | -0.0005 | fail |
| `formal_v2_step-4500` | content=talking_face | 10 | 0.952 | 0.512 | 0.316 | 0.064 | +0.0019 | fail |
| `formal_v2_step-5250` | bucket=1280x512 | 5 | 0.963 | 0.407 | 0.331 | 0.044 | +0.0020 | fail |
| `formal_v2_step-5250` | bucket=1280x704 | 5 | 0.929 | 0.714 | 0.175 | 0.033 | +0.0015 | pass |
| `formal_v2_step-5250` | bucket=704x1280 | 5 | 0.744 | 1.450 | 0.351 | 0.297 | -0.0004 | fail |
| `formal_v2_step-5250` | content=dance | 5 | 0.744 | 1.450 | 0.351 | 0.297 | -0.0004 | fail |
| `formal_v2_step-5250` | content=talking_face | 10 | 0.951 | 0.570 | 0.266 | 0.038 | +0.0018 | fail |
| `formal_v2_step-5986` | bucket=1280x512 | 5 | 0.973 | 0.444 | 0.379 | 0.050 | +0.0021 | fail |
| `formal_v2_step-5986` | bucket=1280x704 | 5 | 0.913 | 0.670 | 0.244 | 0.065 | +0.0015 | fail |
| `formal_v2_step-5986` | bucket=704x1280 | 5 | 0.655 | 1.410 | 0.314 | 0.331 | -0.0003 | fail |
| `formal_v2_step-5986` | content=dance | 5 | 0.655 | 1.410 | 0.314 | 0.331 | -0.0003 | fail |
| `formal_v2_step-5986` | content=talking_face | 10 | 0.952 | 0.564 | 0.323 | 0.057 | +0.0018 | fail |
| `formal_v2_step-750` | bucket=1280x512 | 5 | 0.955 | 0.593 | 0.358 | 0.031 | +0.0020 | fail |
| `formal_v2_step-750` | bucket=1280x704 | 5 | 0.925 | 0.736 | 0.232 | -0.011 | +0.0013 | pass |
| `formal_v2_step-750` | bucket=704x1280 | 5 | 0.686 | 1.450 | 0.407 | 0.262 | +0.0000 | fail |
| `formal_v2_step-750` | content=dance | 5 | 0.686 | 1.450 | 0.407 | 0.262 | +0.0000 | fail |
| `formal_v2_step-750` | content=talking_face | 10 | 0.944 | 0.669 | 0.305 | 0.009 | +0.0016 | fail |
