# PRoA ( A Probabilistic Robustness Assessment against Functional Perturbations)

__Tianle Zhang, Wenjie Ruan, and Jonathan E. Fieldsend__

The accompanying paper _PRoA: A Probabilistic Robustness Assessment against Functional Perturbations_ is accepted by  [European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (*ECML-PKDD*)](https://ecmlpkdd.org/).

#### Citation

```
@article
```



# Abstract

In safety-critical deep learning applications, robustness measurement is a vital pre-deployment phase. However, existing robustness verification methods do not sufficiently meet the criteria for deploying machine learning systems in the real world. On the one hand, these methods attempt to claim that no perturbations can "fool" deep neural networks (DNNs), which may be too stringent in practice. Existing works, on the other hand, rigorously consider L_p bounded additive perturbations on the pixel space, although perturbations, such as colour shifting and geometric transformations, frequently occur in the real world. Thus, from the practical standpoint, we present a novel and general  *probabilistic robustness assessment method* (PRoA) based on the adaptive concentration, and it can measure the robustness of deep learning models against functional perturbations. PRoA can provide statistical guarantees on the probabilistic robustness of a model, *i.e.*, the probability of failure encountered by the trained model after deployment. Our experiments demonstrate the effectiveness and flexibility of PRoA in terms of evaluating the probabilistic robustness against a broad range of functional perturbations, and PRoA can scale well to various large-scale deep neural networks compared to existing state-of-the-art baselines. 

# Schematic Overview



# Sample Results 



# Developer's Platform

```
cox==0.1.post3
dill==0.3.4
GitPython==3.1.27
kornia==0.6.3
matplotlib==3.4.3
numpy==1.21.2
openpyxl==3.0.9
pandas==1.2.5
Pillow==9.1.1
scikit_learn==1.1.1
scipy==1.7.1
seaborn==0.11.2
statsmodels==0.11.1
timm==0.5.0
torch==1.9.1+cu111
torchvision==0.10.1+cu111
tqdm==4.64.0
```

# Run



# Remark

This tool is under active development and maintenance, please feel free to contact us about any problem encountered.

Best regards,

[tz294@exeter.ac.uk](mailto:tz294@exeter.ac.uk)
