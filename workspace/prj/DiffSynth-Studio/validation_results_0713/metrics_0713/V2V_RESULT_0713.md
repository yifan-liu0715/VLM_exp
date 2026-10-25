# V2V quick validation 0713

Target is the standardized original video. Source is the recorded deterministic degraded video.

| Variant | Frame SSIM ↑ | PSNR dB ↑ | Sharpness ↑ | CLIP prompt similarity ↑ |
|---|---:|---:|---:|---:|
| target | 1.0000 | 100.00 | 685.78 | 0.2308 |
| source | 0.7552 | 21.05 | 38.22 | 0.1968 |
| base | 0.5890 | 18.15 | 53.99 | 0.2206 |
| lora | 0.6111 | 18.49 | 45.25 | 0.2025 |

Interpretation: compare base and lora only under identical source video, prompt, seed, denoising strength, and inference settings. This is a native input_video V2V restoration smoke test, not a paired source-conditioned architecture.
