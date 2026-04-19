# Dynamic Prefix Weighting (DPW)

## Installation

```bash
conda env create -f environment.yml
```

## Datasets

DPW is trained and evaluated on the MTIL benchmark from [ZSCL](https://github.com/Thunderbeee/ZSCL). We follow the dataset structure of [DIKI](https://github.com/lloongx/DIKI). Please refer to [DIKI's dataset instructions](https://github.com/lloongx/DIKI/blob/main/docs/datasets.md) for detailed setup guidance.

## Training

### MTIL
```bash
bash MTIL.sh
```

### ODCL-CIL
```bash
bash ODCL-CIL.sh
```

### MTIL (Reduced Parameters)
```bash
bash MTIL_reduced_param.sh
```

### ODCL-CIL (Reduced Parameters)
```bash
bash ODCL-CIL_reduced_param.sh
```

## Citation

```bibtex
@inproceedings{jang2026dpw,
  title={Enhancing Continual Learning of Vision-Language Models via Dynamic Prefix Weighting},
  author={Jang, Hyeonseo and Kwon, Hyuk and Lee, Kibok},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Acknowledgements

This code is based on [DIKI](https://github.com/lloongx/DIKI). We thank the authors for releasing their code.
