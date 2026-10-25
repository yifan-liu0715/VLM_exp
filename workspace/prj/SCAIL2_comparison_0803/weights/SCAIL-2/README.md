---
license: mit
pipeline_tag: image-to-video
tags:
  - character-animation
  - video-generation
  - pose-driven
  - diffusion
library_name: scail-2
---

# SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning

<div align="center">
  <a href='https://huggingface.co/papers/2606.10804'><img src='https://img.shields.io/badge/📖 arXiv-red'>
  <a href='https://teal024.github.io/SCAIL-2'><img src='https://img.shields.io/badge/🌐 Project Page-green'></a>
  <a href='https://github.com/zai-org/SCAIL-2'><img src='https://img.shields.io/badge/💻 GitHub-black?logo=github'></a>
</div>

SCAIL-2 is an open-source model for **end-to-end controlled character animation**. It animates a reference character with a driving video, and also supports character replacement and multi-character scenarios without relying on intermediate pose representations.

<p align="center">
  <img src='https://raw.githubusercontent.com/zai-org/SCAIL-2/refs/heads/wan-scail2/resources/teaser.png' alt='Teaser' width='90%'>
</p>

## 🔎 Overview

Prior approaches to character animation depend heavily on intermediate representations such as skeleton maps or inpainting masks. These intermediates are ambiguous under complex motion, restrict driving sources to human movements, and limit the reach of replacement and multi-character animation.

SCAIL-2 removes this dependence and achieve **End-to-end Driving**. Using several off-the-shelf models (SCAIL-Preview, Wan-Animate, MoCha), 60K motion pairs were synthesized and trained through a Unified Motion Transfer Interface with dedicated masking channels and RoPE design. The reverse driving training recipe with the unification lets the model learn capabilities beyond its teacher models, yielding emergent abilities such as:

- Cross-identity character replacement
- Animal-driving scenarios
- Zero-shot support for advanced control intermediates like SAM3D-Body mesh rendering

<p align="center">
  <img src='https://raw.githubusercontent.com/zai-org/SCAIL-2/refs/heads/wan-scail2/resources/network.png' alt='pipeline' width='90%'>
</p>



## 📦 Model

| Item | Detail |
|------|--------|
| Resolutions | End-to-end driving supports both 512p and 704p; pose-driven and replacement performs better at 704p |
| Constraints | H and W must both be divisible by 32 (e.g. 704×1280) |
| Training | Mixed resolutions and fps |
| Bundled modules | Wan VAE and T5 are integrated into the checkpoint for convenience |

File layout after download:

```
SCAIL-2/
├── Wan2.1_VAE.pth
├── model
│   ├── 1
│   │   └── fsdp2_rank_0000_checkpoint.pt
│   └── latest
└── umt5-xxl
    └── ...
```

## 🚀 Usage

Inference code, environment setup, and detailed instructions are provided in the project repository. Please refer to the [Project Page](https://teal024.github.io/SCAIL-2) and the code repo for how to run the model.


## 📄 Citation

```bibtex
@misc{yan2026scail2,
      title={SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning}, 
      author={Wenhao Yan and Fengjia Guo and Zhuoyi Yang and Jie Tang},
      year={2026},
      eprint={2606.10804},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.10804}, 
}
```