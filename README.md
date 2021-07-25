# GMML2021Project
Project on Skoltech course "Geometric Methods in Machine Learning" on topic "Disentangled Representation Learning as Nonlinear ICA".

| ![MNIST](https://i.imgur.com/TONdCeH.png) |
|:--:| 
| *Gradual change of one feature, responsible for line thickness. NonlinearICA model on MNIST.* |


## Authors

 - Daniil Cherniavskii (Nonlinear ICA part)
 - Ivan Matvienko (InfoGAN part)

## Requirements installation

The code was tested using Python 3.7.6; installing necessary dependancies is done by:

```
pip3 install -r requirements.txt
```

## Model training

To train Nonlinear ICA model and investigate what features may be mean, please refer to the [Nonlinear ICA](https://github.com/danchern97/GMML2021Project/blob/nonlinear_ica/nonlinear_ica/NonLinearICA.ipynb) notebook.

For InfoGAN training, please refer to [InfoGAN folder](https://github.com/danchern97/GMML2021Project/tree/main/infogan)

## Pretrained models

Already pretrained models for MNIST and CelebA can be found [here](https://drive.google.com/drive/folders/1_AehTiuZSL6mJ-9MEquCoZtRuB6t879q?usp=sharing).

## Citation 

```
@inproceedings{hyvarinen2019nonlinear,
  title={Nonlinear ICA using auxiliary variables and generalized contrastive learning},
  author={Hyvarinen, Aapo and Sasaki, Hiroaki and Turner, Richard},
  booktitle={The 22nd International Conference on Artificial Intelligence and Statistics},
  pages={859--868},
  year={2019},
  organization={PMLR}
}

@inproceedings{chen2016infogan,
  title={Infogan: Interpretable representation learning by information maximizing generative adversarial nets},
  author={Chen, Xi and Duan, Yan and Houthooft, Rein and Schulman, John and Sutskever, Ilya and Abbeel, Pieter},
  booktitle={Proceedings of the 30th International Conference on Neural Information Processing Systems},
  pages={2180--2188},
  year={2016}
}
```
