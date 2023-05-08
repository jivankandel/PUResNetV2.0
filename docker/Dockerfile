# Use use previous versions, modify these variables
ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    cmake \
    build-essential \
    libopenblas-dev \
    xterm \
    xauth \
    openssh-server \
    tmux \
    wget \
    mate-desktop-environment-core \
    libeigen3-dev \
    swig \
    python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the latest Open Babel release from GitHub
RUN git clone https://github.com/openbabel/openbabel.git /openbabel && \
    cd /openbabel && \
    git fetch --tags && \
    latest_tag=$(git describe --tags `git rev-list --tags --max-count=1`) && \
    git checkout $latest_tag

# Build and install Open Babel
WORKDIR /openbabel/build
RUN cmake .. \
    -DPYTHON_BINDINGS=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Install MinkowskiEngine and other Python packages
RUN pip install scikit-learn jupyter jupyterlab puresnet==0.1 openbabel
ENV MAX_JOBS=$(nproc)
WORKDIR /MinkowskiEngine
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" && \
    cd MinkowskiEngine && \
    python setup.py install --force_cuda --blas=openblas

# Set the working directory
COPY Example /workspace/Example
WORKDIR /workspace/Example

# Start JupyterLab server by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''","--no-browser"]
