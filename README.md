```
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
- [Getting Started](#getting-started)
- [Example Usage](#example-usage)
- [Citation](#citation)
- [License](#license)

## Overview
PUResNetV2.0 is a state-of-the-art deep learning model designed to predict ligand binding sites in protein structures. Utilizing advanced sparse convolution techniques and the powerful MinkowskiEngine, PUResNetV2.0 offers fast and accurate predictions to aid in computational drug discovery.

## Usage

### Setup Conda Environment

#### Creating environment named sparseconv
```bash
conda create -n sparseconv python=3.10 -c conda-forge
conda activate sparseconv
```

#### Installing pytorch and cuda drivers
```bash
conda install openblas-devel -c anaconda
conda install pytorch=1.13.0 torchvision=0.14 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
```

#### Installing MinkowskiEngine
```bash
export CUDA_HOME=$CONDA_PREFIX
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
```

#### Installing other requirements
```bash
conda install -c conda-forge openbabel
conda install -c anaconda scikit-learn
```

### Installing PUResNetV2.0 package
```bash
pip install puresnet==1.0.0
```

## Getting Started
After installing PUResNetV2.0, you can start predicting ligand binding sites for your protein structures. Follow the instructions in the [Example Usage](#example-usage) section to learn how to use the tool effectively.

## Example Usage
To use PUResNetV2.0 for predicting ligand binding sites, follow the steps below:

1. Prepare your protein structure in PDB format.

2. Run the following command, replacing 'input.pdb' with the path to your protein structure file and 'output.txt' with the desired path for the output file:

```bash
puresnet predict -i input.pdb -o output.txt
```

3. The results will be saved in the specified output file. The output will contain the predicted binding site residues and their corresponding scores.

For more information on additional options and parameters, refer to the [PUResNetV2.0 documentation](https://github.com/username/puresnet/wiki).

## Citation
If you use PUResNetV2.0 in your research, please cite the following paper:

_TODO: Add citation information._

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
```
