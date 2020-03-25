# [Soft Threshold Weight Reparameterization for Learnable Sparsity](https://arxiv.org/abs/2002.03231)

This repository provides code for the experiments performed in the [paper](https://arxiv.org/abs/2002.03231) along with more functionalities for wide usage.

This code base is built upon the [hidden-networks repository](https://github.com/allenai/hidden-networks) modified for [STR](https://arxiv.org/abs/2002.03231), [DNW](https://arxiv.org/abs/1906.00586) and [GMP](https://arxiv.org/abs/1710.01878) experiments.

## Set Up
0. Clone this repository.
1. Using `Python 3.6`, create a `venv` with  `python -m venv myenv` and run `source myenv/bin/activate`. You can also use `conda` to create a virtual environment.
2. Install requirements with `pip install -r requirements.txt` for `venv` and appropriate `conda` commands for `conda` environment.
3. Create a **data directory** `<data-dir>`.
To run the ImageNet experiments there must be a folder `<data-dir>/imagenet`
that contains the ImageNet `train` and `val`.

## STRConv
[`STRConv`](utils/conv_type.py#L22) along with other custom convolution modules can be found in (`utils/conv_type.py`)[utils/conv_type.py]. Users can take `STRConv` and use it in most of the PyTorch based models as it inherits from `nn.Conv2d` or also mentioned here as [`DenseConv`](utils/conv_type.py#L12).

## Vanilla Training
This codebase contains model architectures for [ResNet18](models/resnet.py#L156), [ResNet50](models/resnet.py#L161) and [MobileNetV1](models/mobilenetv1.py) and support to train them on ImageNet-1K. We have provided some `config` files for training [ResNet50](models/resnet.py#L161) and [MobileNetV1](models/mobilenetv1.py) which can be modified for other architectures and datasets. To support more datasets, please add new dataloaders to [`data`](data\) folder.

Training across multiple GPUs is supported, however, the user should check the minimum number of GPUs required to scale ImageNet-1K. 

### Train dense models on ImageNet-1K:

ResNet50: ```python main.py --config configs/largescale/resnet50-dense.yaml --multigpu 0,1,2,3```

MobileNetV1: ```python main.py --config configs/largescale/mobilenetv1-dense.yaml --multigpu 0,1,2,3```

### Train models with **[STR](https://arxiv.org/abs/2002.03231)** on ImageNet-1K:

ResNet50: ```python main.py --config configs/largescale/resnet50-str.yaml --multigpu 0,1,2,3```

MobileNetV1: ```python main.py --config configs/largescale/mobilenetv1-str.yaml --multigpu 0,1,2,3```

To reproduce the results in the [paper](https://arxiv.org/abs/2002.03231), please modify the `config` files appropriately using the hyperparameters from the appendix of STR paper.

### Train ResNet50 models [DNW](https://arxiv.org/abs/1906.00586) and [GMP](https://arxiv.org/abs/1710.01878) on ImageNet-1K:

DNW: ```python main.py --config configs/largescale/resnet50-dnw.yaml --multigpu 0,1,2,3```

GMP: ```python main.py --config configs/largescale/resnet50-gmp.yaml --multigpu 0,1,2,3```

Please note that **GMP** implementation is not experimented exhaustively, so caution advised. 

Modify the `config` files to tweak the performance and sparsity levels in both DNW and GMP. 

## Models and Logging
STR models are not compatible with the traditional dense models for simple evaluation and usage as transfer learning backbones. DNW and GMP models are compatible to the dense model.

Every experiment creates a directory inside `runs` folder (which will be created automatically) along with the tensorboard logs, initial model state (for LTH experiments) and best model (`model_best.pth`).

The `runs` folder also has dumps of the csv with final and best accuracies along with layer-wise sparsity distributions and threhsolds in case of STR. 

### Convert STR model to dense model:

ResNet50: ```python main.py --config configs/largescale/resnet50-dense.yaml --multigpu 0,1,2,3 --pretrained <ResNet50-STR-Model> --dense-conv-model```

MobileNetV1: ```python main.py --config configs/largescale/mobilenetv1-dense.yaml --multigpu 0,1,2,3 --pretrained <MobileNetV1-STR-Model> --dense-conv-model```

These models use the names provided in the corresponding `config` files being used but can also be modified using `--name` argument in command line.

### Evaluating models on ImageNet-1K:

If you want to evaluate a pretrained STR model provided below, you can either use the model as is or convert it to a dense model and use the dense model evaluation. To encourage uniformity, please try to convert the STR models to dense or use the dense compatible models if provided.

Dense Model Evaluation: ```python main.py --config configs/largescale/<arch>-dense.yaml --multigpu 0,1,2,3 --pretrained <Dense-Compatible-Model> --evaluate```

STR Model Evaluation: ```python main.py --config configs/largescale/<arch>-str.yaml --multigpu 0,1,2,3 --pretrained <STR-Model> --evaluate```

## Sparsity Budget Transfer
If it is hard to hand-code all the budgets into a method like DNW, you can use the budget transfer functionalities of the repo. 

Transfer to DNW: ```python main.py --config configs/largescale/<arch>-dnw.yaml --multigpu 0,1,2,3 --pretrained <STR-Model> --ignore-pretrained-weights --use-budget```

Transfer to GMP: ```python main.py --config configs/largescale/<arch>-gmp.yaml --multigpu 0,1,2,3 --pretrained <STR-Model> --ignore-pretrained-weights --use-budget```

You should modify the corresponding `config` files for DNW and GMP to increase accuracy by changing the hyperparameters.

## Pretrained Models

## Citing

If you find this project useful in your research, please consider citing:

```
@article{Kusupati20
  author    = {Kusupati, Aditya and Ramanujan, Vivek and Somani, Raghav and Wortsman, Mitchell and Jain, Prateek and Kakade, Sham and Farhadi, Ali},
  title     = {Soft Threshold Weight Reparameterization for Learnable Sparsity},
  booktitle = {arXiv preprint arXiv:2002.03231},
  year      = {2020},
}
```