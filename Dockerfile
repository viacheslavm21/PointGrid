ARG UBUNTU_VER=18.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.6
ARG TF_VER=1.14

FROM ubuntu:${UBUNTU_VER}

# System packages 
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -yq curl wget jq vim
RUN apt-get install -y unzip git python3-pip

# Use the above args during building https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

ARG PY_VER
ARG TF_VER
# Install packages from conda and downgrade py (optional).
RUN conda install -c anaconda -y python=${PY_VER}
RUN pip install numpy pandas scipy
RUN conda install -c anaconda -y tensorflow-mkl=${TF_VER}

RUN pip uninstall -y tensorflow
RUN pip install tensorflow==1.13.2

WORKDIR /
RUN git clone https://github.com/viacheslavm21/PointGrid.git

WORKDIR /PointGrid
RUN chmod +x download.sh
