# PUResNetV2.0
A powerful tool for predicting ligand binding sites in protein structures

## Table of Contents
- [Overview](#overview)
- [Usage](#usage)
    - [Setup Conda Environment](#setup-conda-environment)
        - [Creating environment named sparseconv](#creating-environment-named-sparseconv)
        - [Installing pytorch and cuda drivers](#installing-pytorch-and-cuda-drivers)
        - [Installing MinkowskiEngine](#installing-minkowskiengine)
        - [Installing other requirements](#installing-other-requirements)
        - [Installing PUResNetV2.0 package](#installing-puresnetv20-package)
    - [Setup Docker](#setup-docker)
- [Getting Started](#getting-started)
- [Example Usage](#example-usage)
- [Citation](#citation)
- [License](#license)

## Overview
PUResNetV2.0 is a state-of-the-art deep learning model designed to predict ligand binding sites in protein structures. Utilizing advanced sparse convolution techniques and the powerful MinkowskiEngine, PUResNetV2.0 offers fast and accurate predictions to aid in computational drug discovery.

## Usage

### Setup Conda Environment

##### Creating environment named sparseconv
```bash
conda create -n sparseconv python=3.10 -c conda-forge
conda activate sparseconv
```

##### Installing pytorch and cuda drivers
```bash
conda install openblas-devel -c anaconda
conda install pytorch=1.13.0 torchvision=0.14 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
```

##### Installing MinkowskiEngine
```bash
export CUDA_HOME=$CONDA_PREFIX
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
```

##### Installing other requirements
```bash
conda install -c conda-forge openbabel
conda install -c anaconda scikit-learn
```

##### Installing PUResNetV2.0 package
```bash
pip install puresnet==0.1
```
### Setup Docker
##### JupyterLab with CUDA, PyTorch, and Python 3.10

This Docker image provides a ready-to-use JupyterLab environment with CUDA, PyTorch, and Python 3.10.

##### Prerequisites

- Docker installed on your system (https://docs.docker.com/get-docker/)
- NVIDIA GPU with compatible CUDA drivers (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

##### How to Run the Docker Image

1. Pull the Docker image from Docker Hub:

```bash
docker pull jivankandel/puresnet:latest
```

2. Run the Docker container, exposing the JupyterLab port (8888) and enabling GPU access:

```bash
docker run --gpus all -it --user root -p 8888:8888 -v "$(pwd)":/work --workdir /work jivankandel/puresnet:latest
```

GOTO folder on your local machine where you want to store your notebooks and data and run above command.

To run Examples
```bash
docker run --gpus all -it --user root -p 8888:8888 --workdir /Example jivankandel/puresnet:latest
```

3. Open your web browser and navigate to `http://localhost:8888`. JupyterLab should be running without requiring any authentication.

##### Stopping the Docker Container

To stop the running Docker container, find the container ID using the following command:

```bash
docker ps
```

Take note of the `CONTAINER ID` corresponding to your running image. Then, stop the container using the following command:

```bash
docker stop <container_id>
```

Replace `<container_id>` with the appropriate `CONTAINER ID` from the previous step.


## Getting Started
After installing PUResNetV2.0, you can start predicting ligand binding sites for your protein structures. Follow the instructions in the [Example Usage](#example-usage) section to learn how to use the tool effectively.

## Example Usage
Inside Example explore following notebook files:
1. Creating sparse tensor.ipynb
2. Predicting.ipynb
3. Training.ipynb
## Citation
1. Kandel, J., Tayara, H. & Chong, K.T. PUResNet: prediction of protein-ligand binding sites using deep residual neural network. J Cheminform 13, 65 (2021). 
https://doi.org/10.1186/s13321-021-00547-7

## License
MIT License

Copyright (c) 2023 Kandel Jeevan

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.