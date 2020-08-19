# SRDLC
The main implement of paper [A network intrusion detection method based on semantic Re-encoding and deep learning](https://www.sciencedirect.com/science/article/abs/pii/S1084804520301624)

## Installation
Clone this repo.

git clone https://github.com/alilliam/SRDLC \\
cd SRDLC/ \\
This code requires PyTorch 1.1.0 and python 3.7.6+. Please install the following dependencies: \\

* pytorch 1.1.0
* torchvision
* numpy
* scipy
* scikit-image
* tqdm 

\\
To reproduce the results reported in the paper, you need to run experiments on NVIDIA 1050ti+.

## Dataset Preparation
Please put the dataset in the path ./data/

## Train and Test Models
Run python kdd_test_std.py.


