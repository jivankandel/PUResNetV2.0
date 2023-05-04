# PUResNetV2.0
Prediction of ligand binding site

<h1>Usage</h1>
# 1. Setup Environment
<h2>Creating environment named sparseconv </h2>
<pre>
conda create -n sparseconv python=3.10 -c conda-forge
conda activate sparseconv
</pre>
<h2>Installing pytorch and cuda drivers</h2>
<pre>
conda install openblas-devel -c anaconda
conda install pytorch=1.13.0 torchvision=0.14 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
</pre>
<h2> Installing MinkowskiEngine </h2>
<pre>
export CUDA_HOME=$CONDA_PREFIX
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
</pre>
<h2> Installing other requirements </h2>
<pre>
conda install -c conda-forge openbabel
conda install -c anaconda scikit-learn
</pre>
<h2> Installing PUResNetV2.0 package </h2>
<pre>
pip install puresnet==1.0.0
</pre>

