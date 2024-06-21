ARG BASE_IMAGE=python:3.10
# the only other valid option is 'nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04' for gpu support
# (TPU support does not require a specific image)

FROM ${BASE_IMAGE} AS trix_base

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    git \
    unzip \
    aria2 \
    wget \
    gnupg && \
    rm -rf /var/lib/apt/lists/*

# Install Micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Install gcloud
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && \
    apt-get install -y \
    cloudsql-proxy \
    google-cloud-cli \
    google-cloud-cli-cloud-run-proxy && \
    rm -rf /var/lib/apt/lists/*
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

RUN mkdir /app
WORKDIR /app

# Install conda dependencies
RUN apt-get update
COPY requirements.txt /tmp/requirements.txt
COPY environment.yaml /tmp/environment.yaml
ARG BUILD_FOR_TPU="false"
RUN if [ ${BUILD_FOR_TPU} = "false" ] ; then echo "Not building for tpu" ; \
    else sed -i 's/jax==/jax[tpu]==/g' /tmp/requirements.txt ; fi
ARG BUILD_FOR_GPU="false"
RUN if [ ${BUILD_FOR_GPU} = "false" ] ; then echo "Not building for gpu" ; \
    else sed -i 's/jax==/jax[cuda11_cudnn82]==/g' /tmp/requirements.txt && \
    sed -i 's/libtpu_releases\.html/jax_cuda_releases\.html/g' /tmp/environment.yaml; fi
RUN micromamba create -y --file /tmp/environment.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete
ENV PATH=/opt/conda/envs/trix/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/trix/lib/:$LD_LIBRARY_PATH

# Install InstaDeep's trix pip package with no dependencies
# as they are currently pinned to specific versions which conflicts with our
# own (example: python3.10)
ARG GITLAB_USERNAME
ARG GITLAB_ACCESS_TOKEN
ARG TRIX_COMMIT_SHA
ARG PIP_EXTRA_INDEX_URL

# Specify location of GCP credentials to get access to SQL database and runs GCP bucket
ARG GOOGLE_APPLICATION_CREDENTIALS="/app/int-research-multiomics-09c9f9d72651.json"
ENV GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS}"

# Add symlink to the python package
RUN ln -s /app/trix /opt/conda/envs/trix/lib/python3.10/site-packages/


FROM trix_base AS trix_gpu
# this is the working GPU image.
# Do avoid using the devel image as base (this is a very heavy image), we just copy the binary
# from it that we are missing.
COPY --from=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
