# Adapted from: https://raw.githubusercontent.com/CosmiQ/solaris/master/docker/gpu/Dockerfile
# docker build -t sn7_baseline_image /path_to_docker/
# NV_GPU=0 nvidia-docker run -it -v /local_data:/local_data -v  --rm -ti --ipc=host --name sn7_baseline_gpu0 sn7_baseline_image
# updated 20250212

# Use a stable CUDA base image
FROM nvidia/cuda:10.2-devel-ubuntu16.04
LABEL maintainer="avanetten <avanetten@iqt.org>"

ENV CUDNN_VERSION 7.6.0.64
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
ARG SOLARIS_BRANCH='master'

# Prep apt-get and install CuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
        libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    unzip \
    vim \
    wget \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Use bash as default shell
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Configure Conda channels
RUN conda update -n base conda && \
    conda config --prepend channels conda-forge && \
    conda config --prepend channels pytorch

# Clone Solaris repository and create Conda environment
WORKDIR /tmp/
RUN git clone https://github.com/cosmiq/solaris.git && \
    cd solaris && \
    git checkout ${SOLARIS_BRANCH} && \
    cp environment-gpu.yml /tmp/environment.yml

# Use environment.yml to create the Conda environment
COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml

# Ensure Conda environment is activated
ENV PATH /opt/conda/envs/solaris/bin:$PATH

# Install Solaris package
RUN cd solaris && pip install .

# Install additional Conda dependencies
RUN conda install -n solaris -y \
    jupyter \
    jupyterlab \
    ipykernel

# Add Jupyter kernel for Solaris environment
RUN source activate solaris && python -m ipykernel.kernelspec --name solaris --display-name "Python (solaris)"

# Install additional Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Paddle (ensure compatibility with GPU)
RUN pip install paddlepaddle-gpu==1.8.2.post107 -i https://mirror.baidu.com/pypi/simple
RUN pip install scipy==1.3.2

# Set working directory
WORKDIR /work
COPY . /work/
RUN chmod 777 train.sh test.sh download.sh pretrained_model/download_model.py

# Open ports for JupyterLab and TensorBoard
EXPOSE 8888 6006

# Default to Bash shell
CMD ["/bin/bash"]