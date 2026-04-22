# Enhancing Continual Learning of Vision-Language Models via Dynamic Prefix Weighting [CVPR 2026]

[![Conference](https://img.shields.io/badge/CVPR-2026-0b5fff.svg)](https://cvpr.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-4b9e5d.svg)](https://arxiv.org/abs/2604.18075)

This repository contains the official implementation of our CVPR 2026 paper:
> [**Enhancing Continual Learning of Vision-Language Models via Dynamic Prefix Weighting**](https://arxiv.org/abs/2604.18075)  
> **Hyeonseo Jang**, **Hyuk Kwon**, and **Kibok Lee**

## 📖 Overview

**DPW (Dynamic Prefix Weighting)** is a parameter-efficient framework for continual learning of vision-language models. DPW assigns fine-grained, token-level weights to prefixes and adapters:

- 🔹 A **gating module** adjusts the weight of each prefix based on the importance of the corresponding input token.
- 🔹 A **residual weighting mechanism** derives adapter weights from the residuals of the prefix weights.

## ⚙️ Installation

```bash
conda env create -f environment.yml
conda activate dpw
```

## 📂 Datasets

DPW is trained and evaluated on the MTIL benchmark from [ZSCL](https://github.com/Thunderbeee/ZSCL). We follow the dataset structure of [DIKI](https://github.com/lloongx/DIKI); please refer to their [dataset instructions](https://github.com/lloongx/DIKI/blob/main/docs/datasets.md) for setup details.

## 🚀 Training

### MTIL
Multi-domain Task Incremental Learning, where task IDs are available at test time.
```bash
bash MTIL.sh
```

### ODCL-CIL
Open-Domain Continual Learning, where task IDs are not available at test time.
```bash
bash ODCL-CIL.sh
```

### Parameter-Efficient Variants
For the reduced-parameter variants (Ours†) reported in the paper:
```bash
bash MTIL_reduced_param.sh
bash ODCL-CIL_reduced_param.sh
```

## 📝 Citation
If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{jang2026dpw,
  title={Enhancing Continual Learning of Vision-Language Models via Dynamic Prefix Weighting},
  author={Jang, Hyeonseo and Kwon, Hyuk and Lee, Kibok},
  booktitle={CVPR},
  year={2026}
}
```

## 📬 Contact

For questions or issues, please contact [jhyeonseo715@yonsei.ac.kr](mailto:jhyeonseo715@yonsei.ac.kr).

## 🙏 Acknowledgements

This codebase builds on [DIKI](https://github.com/lloongx/DIKI). We thank the authors for releasing their code.
