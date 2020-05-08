# [Soft Threshold Weight Reparameterization for Learnable Sparsity](https://arxiv.org/abs/2002.03231)
[Aditya Kusupati](https://homes.cs.washington.edu/~kusupati/), [Vivek Ramanujan*](https://vkramanuj.github.io/), [Raghav Somani*](http://raghavsomani.github.io/), [Mitchell Worstsman*](https://mitchellnw.github.io/), [Prateek Jain](prateekjain.org), [Sham Kakade](https://homes.cs.washington.edu/~sham/) and [Ali Farhadi](https://homes.cs.washington.edu/~ali/)

This repository contains code for the CNN experiments presented in the [paper](https://arxiv.org/abs/2002.03231) along with more functionalities.

This code base is built upon the [hidden-networks repository](https://github.com/allenai/hidden-networks) modified for [STR](https://arxiv.org/abs/2002.03231), [DNW](https://arxiv.org/abs/1906.00586) and [GMP](https://arxiv.org/abs/1710.01878) experiments.

The RNN experiments in the [paper](https://arxiv.org/abs/2002.03231) are done by modifying [`FastGRNNCell`](http://manikvarma.org/pubs/kusupati18.pdf) in [EdgeML](https://github.com/Microsoft/EdgeML) using the methods discussed in the paper.

## Set Up
0. Clone this repository.
1. Using `Python 3.6`, create a `venv` with  `python -m venv myenv` and run `source myenv/bin/activate`. You can also use `conda` to create a virtual environment.
2. Install requirements with `pip install -r requirements.txt` for `venv` and appropriate `conda` commands for `conda` environment.
3. Create a **data directory** `<data-dir>`.
To run the ImageNet experiments there must be a folder `<data-dir>/imagenet`
that contains the ImageNet `train` and `val`.

## STRConv
[`STRConv`](utils/conv_type.py#L22) along with other custom convolution modules can be found in [`utils/conv_type.py`](utils/conv_type.py). Users can take `STRConv` and use it in most of the PyTorch based models as it inherits from `nn.Conv2d` or also mentioned here as [`DenseConv`](utils/conv_type.py#L12).

## Vanilla Training
This codebase contains model architectures for [ResNet18](models/resnet.py#L156), [ResNet50](models/resnet.py#L161) and [MobileNetV1](models/mobilenetv1.py) and support to train them on ImageNet-1K. We have provided some `config` files for training [ResNet50](models/resnet.py#L161) and [MobileNetV1](models/mobilenetv1.py) which can be modified for other architectures and datasets. To support more datasets, please add new dataloaders to [`data`](data/) folder.

Training across multiple GPUs is supported, however, the user should check the minimum number of GPUs required to scale ImageNet-1K. 

### Train dense models on ImageNet-1K:

ResNet50: ```python main.py --config configs/largescale/resnet50-dense.yaml --multigpu 0,1,2,3```

MobileNetV1: ```python main.py --config configs/largescale/mobilenetv1-dense.yaml --multigpu 0,1,2,3```

### Train models with **[STR](https://arxiv.org/abs/2002.03231)** on ImageNet-1K:

ResNet50: ```python main.py --config configs/largescale/resnet50-str.yaml --multigpu 0,1,2,3```

MobileNetV1: ```python main.py --config configs/largescale/mobilenetv1-str.yaml --multigpu 0,1,2,3```

To reproduce the results in the [paper](https://arxiv.org/abs/2002.03231), please modify the `config` files appropriately using the hyperparameters from the appendix of STR paper.

### Train ResNet50 models with [DNW](https://arxiv.org/abs/1906.00586) and [GMP](https://arxiv.org/abs/1710.01878) on ImageNet-1K:

DNW: ```python main.py --config configs/largescale/resnet50-dnw.yaml --multigpu 0,1,2,3```

GMP: ```python main.py --config configs/largescale/resnet50-gmp.yaml --multigpu 0,1,2,3```

Please note that **GMP** implementation is not thoroughly tested, so caution is advised. 

Modify the `config` files to tweak the performance and sparsity levels in both DNW and GMP. 

## Models and Logging
STR models are not compatible with the traditional dense models for simple evaluation and usage as transfer learning backbones. DNW and GMP models are compatible to the dense model.

Every experiment creates a directory inside `runs` folder (which will be created automatically) along with the tensorboard logs, initial model state (for LTH experiments) and best model (`model_best.pth`).

The `runs` folder also has dumps of the csv with final and best accuracies along with layer-wise sparsity distributions and thresholds in case of STR. The code checkpoints after every epoch giving a chance to resume training when pre-empted, the extra functionalities can be explored through ```python main.py -h```. 

### Convert STR model to dense model:

ResNet50: ```python main.py --config configs/largescale/resnet50-dense.yaml --multigpu 0,1,2,3 --pretrained <ResNet50-STR-Model> --dense-conv-model```

MobileNetV1: ```python main.py --config configs/largescale/mobilenetv1-dense.yaml --multigpu 0,1,2,3 --pretrained <MobileNetV1-STR-Model> --dense-conv-model```

These models use the names provided in the corresponding `config` files being used but can also be modified using `--name` argument in the command line.

### Evaluating models on ImageNet-1K:

If you want to evaluate a [pretrained](#pretrained-models) STR model provided below, you can either use the model as is or convert it to a dense model and use the dense model evaluation. To encourage uniformity, please try to convert the STR models to dense or use the dense compatible models if provided.

Dense Model Evaluation: ```python main.py --config configs/largescale/<arch>-dense.yaml --multigpu 0,1,2,3 --pretrained <Dense-Compatible-Model> --evaluate```

STR Model Evaluation: ```python main.py --config configs/largescale/<arch>-str.yaml --multigpu 0,1,2,3 --pretrained <STR-Model> --evaluate```

## Sparsity Budget Transfer
If it is hard to hand-code all the budgets into a method like DNW, you can use the budget transfer functionalities of the repo. The pre-trained models provided have to be in the native STR model format and not in a converted/compatible Dense model format. You should change [this piece](main.py#L312) of code to support the Dense format as well.

Transfer to DNW: ```python main.py --config configs/largescale/<arch>-dnw.yaml --multigpu 0,1,2,3 --pretrained <STR-Model> --ignore-pretrained-weights --use-budget```

Transfer to GMP: ```python main.py --config configs/largescale/<arch>-gmp.yaml --multigpu 0,1,2,3 --pretrained <STR-Model> --ignore-pretrained-weights --use-budget```

You should modify the corresponding `config` files for DNW and GMP to increase accuracy by changing the hyperparameters.

## Pretrained Models
All the models provided here are trained on ImageNet-1K according to the settings in the [paper](https://arxiv.org/abs/2002.03231). 

### Fully Dense Models:

These models are straightforward to train using this repo and their pre-trained models are in most of the popular frameworks. For the sake of reproducibility, please find the ResNet50 Dense model. 

| Architecture | Params | Sparsity (%) | Top-1 Acc (%) | FLOPs | Model Links |
| ------------ | :----: | :----------: | :-----------: | :---: | :---------: |
| ResNet50     | 25.6M  | 0.00         | 77.01         | 4.09G |  [Dense](https://drive.google.com/file/d/13dEj0bSyisrYOhSsYf7mBqv_ixJlCYQa/view?usp=sharing) |
| MobileNetV1  | 4.21M  | 0.00         | 70.60         | 569M  |             |

### STR Sparse Models:

We are providing links to 6 models for ResNet50 and 2 models for MobileNetV1. These models represent the sparsity regime they belong to. Each model has two versions of model links to download, the first one is the vanilla STR model and the second one is the STR model converted to be compatible with Dense models and for transfer learning. Please contact [Aditya Kusupati](https://homes.cs.washington.edu/~kusupati/) in case you need a specific model and are not able to train it from scratch. All the sparsity budgets for every model in the [paper](https://arxiv.org/abs/2002.03231) are present in the appendix, in case all you need is the non-uniform sparsity budget.

#### ResNet50:
| No. | Params | Sparsity (%) | Top-1 Acc (%) | FLOPs | Model Links |
| --- | :----: | :----------: | :-----------: | ----: | :---------: |
| 1   | 4.47M  | 81.27        | 76.12         | 705M  | [STR](https://drive.google.com/file/d/1ztb0UfsFVJQkE2vj6eNHxnab9MXuLryk/view?usp=sharing), [Dense](https://drive.google.com/file/d/1-M2AxzKdRp1pJWI4gAdN7v3a9Oih2MdG/view?usp=sharing) |
| 2   | 2.49M  | 90.23        | 74.31         | 343M  | [STR](https://drive.google.com/file/d/19Aw8ovR52sMN9FQrSnd78WLLXeGrG67z/view?usp=sharing), [Dense](https://drive.google.com/file/d/1sya0OI1o53a64M0oXNhYRnkLGPhVRXaQ/view?usp=sharing) |
| 3   | 1.24M  | 95.15        | 70.23         | 162M  | [STR](https://drive.google.com/file/d/1WTGNMQVQFBMigN8LH3NOc0QbWFnOMiPY/view?usp=sharing), [Dense](https://drive.google.com/file/d/17iKnZUl7bd4nBMjs2emhN4kTl1xEiUS9/view?usp=sharing) |
| 4   | 0.99M  | 96.11        | 67.78         | 127M  | [STR](https://drive.google.com/file/d/10-6GaZB5KA3wRilspcl674RR80zN2pFP/view?usp=sharing), [Dense](https://drive.google.com/file/d/1sBCgtY3zQRwE5436fhiG8jQz9ql8npFg/view?usp=sharing) |
| 5   | 0.50M  | 98.05        | 61.46         | 73M   | [STR](https://drive.google.com/file/d/1wgdbeZgWWTN6baQvvmiwDf-6LVs8G7fH/view?usp=sharing), [Dense](https://drive.google.com/file/d/14R288bBVjamgW80qPy4i-tHXcfssqhnm/view?usp=sharing) |
| 6   | 0.26M  | 98.98        | 51.82         | 47M   | [STR](https://drive.google.com/file/d/1MrGL0MtqmiWvZqqa02iDk2OqVKWNiRmM/view?usp=sharing), [Dense](https://drive.google.com/file/d/1jTsYTpQ4_cybbA0Z9h7yQ3SSLNaGj6BB/view?usp=sharing) |

#### MobileNetV1  :
| No. | Params | Sparsity (%) | Top-1 Acc (%) | FLOPs | Model Links |
| --- | :----: | :----------: | :-----------: | ----: | :---------: |
| 1   | 1.04M  | 75.28        | 68.35         | 101M  | [STR](https://drive.google.com/file/d/1XgBvMN2AzIoGSEYMfpoudHH3cLee-q-x/view?usp=sharing), [Dense](https://drive.google.com/file/d/19LWzHdUMpE5gm7tW9lIDs-T7rA3mcqFh/view?usp=sharing) |
| 2   | 0.46M  | 89.01        | 62.10         | 42M  | [STR](https://drive.google.com/file/d/1_mNcVZTJB6LMfv5XrFUs2pWFMu9JQvG8/view?usp=sharing), [Dense](https://drive.google.com/file/d/1-PYY_uc-diqnfhMbZJNTYgqe95ouX7fp/view?usp=sharing) |

Note: If you find any STR model to be 2x the size of its Dense compatible model, it might be because of an old implementation that might have resulted in a model that replicated the weights.

## Sparsity Budgets

The folder [`budgets`](budgets) contains the csv files containing all the non-uniform sparsity budgets STR learnt for ResNet50 on ImageNet-1K across all the sparsity regimes along with baseline budgets for 90% sparse ResNet50 on ImageNet-1K. In case, you are not able to use the pretraining models to extract sparsity budgets, you can directly import the same budgets using these files.

## Citation

If you find this project useful in your research, please consider citing:

```
@article{Kusupati20
  author    = {Kusupati, Aditya and Ramanujan, Vivek and Somani, Raghav and Wortsman, Mitchell and Jain, Prateek and Kakade, Sham and Farhadi, Ali},
  title     = {Soft Threshold Weight Reparameterization for Learnable Sparsity},
  booktitle = {arXiv preprint arXiv:2002.03231},
  year      = {2020},
}
```
