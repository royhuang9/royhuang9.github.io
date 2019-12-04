# Setup container on Google Cloud for Deep Learning

## Prepare docker file
This dockfile is for pytorch-transformer
```
# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       1.1    (pip)
# tensorflow    1.14   (pip)
# keras         latest (pip)
# ==================================================================
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn \
        matplotlib \
        Cython \
        && \

# ==================================================================
# pytorch
# ------------------------------------------------------------------

    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
    	torch \
	torchvision \
        && \

# ==================================================================
# tensorflow
# ------------------------------------------------------------------
#
#    $PIP_INSTALL \
#        tensorflow-gpu==1.14 \
#        && \

# ==================================================================
# keras
# ------------------------------------------------------------------

#    $PIP_INSTALL \
#        h5py \
#        keras \
#        && \
# ============================
# pytorch_transformer python packages
# ----------------------------

    $PIP_INSTALL \
        tqdm \
        pytorch-transformers \
        && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 6006

```

## Compile image from Dockfile on local computer
Confirm the container works on local computer
```
docker build -t transformer .
```

Check the image, there should be an image "transformer"
```
docker images
```

run the docker
```
docker run --gpus all --ipc=host -it -v /data:/data 
```

## Build the image on Google Cloud
replace the PROJECT_ID with a real project id
```
gcloud builds submit --tag gcr.io/[PROJECT_ID]/quickstart-image .
```

## Setup a Computing Engine
Install a Computing Engine with GPU from Marketplace. When create VM, search Deep Learning VM.

## Install Docker
```
# SET UP THE REPOSITORY
sudo apt-get -y update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    software-properties-common

curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/debian \
   $(lsb_release -cs) \
   stable"

# INSTALL DOCKER ENGINE - COMMUNITY
sudo apt-get -y update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io
sudo docker -y run hello-world
```

## Install Nvidia Docker
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get -y update

sudo apt-get -y install nvidia-docker2
sudo pkill -SIGHUP dockerd

```

## upload programs to VM
```


```

## Run docker image on VM
```
gcloud auth configure-docker
docker run --gpus all --ipc=host -it -v ~:/data gcr.io/copper-triumph-248309/quickstart-image bash
```