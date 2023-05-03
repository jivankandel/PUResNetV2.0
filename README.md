# PUResNetV2.0
Prediction of ligand binding site

<h1>Usage</h1>
1. Clone this repository
<pre>
git clone https://github.com/jivankandel/PUResNetV2.0.git
cd PUResNet
</pre>
2. Setup Environment
<pre>
conda create -n sparseconv python=3.10 -c conda-forge
conda install openblas-devel -c anaconda
conda install pytorch=1.13.0 torchvision=0.14 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
conda install -c conda-forge openbabel
conda install -c anaconda scikit-learn
pip install puresnet==1.0.0
</pre>
3. Install puresnet package from PyPI
<pre>
pip install puresnet==1.0.0
</pre>
