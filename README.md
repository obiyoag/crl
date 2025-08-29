# Concept Rule Learner
Welcome👋! This repository provides the official implementation of our paper *Learning Concept-Driven Logical Rules for Interpretable and Generalizable Medical Image Classification* (**CRL**) [[arXiv](https://arxiv.org/abs/2505.14049)], which has been accepted by *MICCAI 2025*.

## 💡 TL;DR
![crl](https://cdn.jsdelivr.net/gh/obiyoag/images@main/data/crl.png)
> While concept-based models offer local concept explanations (instance-level), they often neglect the global decision logic (dataset-level). Moreover, these models often suffer from concept leakage, where unintended information within soft concept representations undermines both interpretability and generalizability. To address these limitations, we propose Concept Rule Learner (**CRL**), a novel framework to learn Boolean logical rules from binary visual concepts. CRL employs logical layers to capture concept correlations and extract clinically meaningful rules, thereby providing both local and global interpretability.

## 📦 Get started

### Environment Preparing
```
conda create -n crl python=3.10
conda activate crl
# please modify according to the CUDA version in your server
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dataset Preparing
- Download the skin datasets from [here](https://skincon-dataset.github.io).
- Download the WBC images from [here](https://data.mendeley.com/datasets/snkd93bnjr/1) and concept annotations from [here](https://github.com/apple2373/wbcatt).
- Put raw images into `[data_dir]/raw_data` and the csv files into `[data_dir]/meta_data`.
- Run `python -m data.split_dataset` to split the skin dataset into five folds.

### CRL training & testing

**train&test for skin**:
```
python main.py --configs configs/skin.yaml --data.fold [fold_idx]
```

**train&test for WBC**:
```
python main.py --configs configs/wbc.yaml
```

**OOD test with DDI dataset**
```
python test_ood.py
```

## 🙋 Feedback and Contact
- ybgao22@m.fudan.edu.cn
- zxh@fudan.edu.cn

## 🛡️ License
This project is released under the [Apache-2.0 License](LICENSE).

## 🙏 Acknowledgement
We thank the maintainers of the following resources:
- [Fitzpatrick17k dataset](https://github.com/mattgroh/fitzpatrick17k), [DDI dataset](https://ddi-dataset.github.io) and [SkinCon annotations](https://skincon-dataset.github.io)
- [PBC dataset](https://data.mendeley.com/datasets/snkd93bnjr/1) and [WBCAtt annotations](https://github.com/apple2373/wbcatt)

## 📝 Citation
If you find our work or the repo useful, please consider giving a star and citation:
```
@inproceedings{Gao2025CRL,
    author={Yibo Gao, Hangqi Zhou, Zheyao Gao, Bomin Wang, Shangqi Gao, Sihan Wang, Xiahai Zhuang},
    title={Learning Concept-Driven Logical Rules for Interpretable and Generalizable Medical Image Classification},
    booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention},
    year={2025}
}
```
