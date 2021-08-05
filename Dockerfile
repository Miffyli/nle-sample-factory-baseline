FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 AS nhc-dev
ENV DEBIAN_FRONTEND=noninteractive

COPY apt.txt /tmp/apt.txt
RUN apt -qq update && apt -qq install -y --no-install-recommends `cat /tmp/apt.txt` \
 && rm -rf /var/cache/*

# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Create user home directory
ENV USER_NAME aicrowd
ENV HOME_DIR /home/$USER_NAME

# Replace HOST_UID/HOST_GUID with your user / group id
ENV HOST_UID 1001
ENV HOST_GID 1001

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

# Set up user
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${HOST_UID} \
    ${USER_NAME}

WORKDIR /opt/

# Install anaconda
ENV CONDA_DIR /opt/conda
RUN wget -nv -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && bash miniconda.sh -b -p ${CONDA_DIR} \
 && . ${CONDA_DIR}/etc/profile.d/conda.sh \
 && conda clean -y -a \
 && rm -rf miniconda.sh
ENV PATH ${CONDA_DIR}/bin:${PATH}

RUN conda install cmake cudatoolkit=10.2 pytorch -y -c pytorch -c nvidia && conda clean -y -a

# Install AICrowd items
WORKDIR ${HOME_DIR}
COPY --chown=1001:1001 requirements.txt ${HOME_DIR}/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# SUBMISSION IMAGE - change user and copy files
FROM nhc-dev AS nhc-submit
COPY --chown=1001:1001 . ${HOME_DIR}
USER ${USER_NAME}

