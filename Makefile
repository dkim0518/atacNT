SHELL := /bin/bash

# Load environment variables from .env file
include .env

# Variables
WORK_DIR = $(PWD)
PORT = 8891
PORT_DOC = 8000
ACCELERATOR = TPU

DOCKER_RUN_FLAGS_CPU = --rm --shm-size=1024m \
	-v $(WORK_DIR):/app \
	--env-file=.env

DOCKER_RUN_FLAGS_GPU = ${DOCKER_RUN_FLAGS_CPU} --gpus all --env-file=.env

DOCKER_RUN_FLAGS_TPU = --rm --privileged -p 6006:6006 \
	-v $(WORK_DIR):/app \
	--user root \
	--env-file=.env \
	--ipc="host" \
	--network host

DOCKER_IMAGE_NAME_CPU = instadeep/trix-cpu:$(USER)
DOCKER_IMAGE_NAME_GPU = instadeep/trix-gpu:$(USER)
DOCKER_IMAGE_NAME_TPU = instadeep/trix-tpu:$(USER)

USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

DOCKER_BUILD_FLAGS = --build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID) \
	--build-arg GITLAB_USERNAME=$(GITLAB_USERNAME) \
	--build-arg GITLAB_ACCESS_TOKEN=$(GITLAB_ACCESS_TOKEN) \
	--build-arg PIP_EXTRA_INDEX_URL=$(PIP_EXTRA_INDEX_URL) \
	--build-arg TRIX_COMMIT_SHA=$(TRIX_COMMIT_SHA)

# Pass NO if you do not have root permissions
SUDO = YES
ifeq ($(SUDO),YES)
SUDO_FLAG = sudo
else
SUDO_FLAG =
endif

# Makefile
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rfv
	find . | grep -E ".pytest_cache" | xargs rm -rfv
	find . | grep -E "nul" | xargs rm -rfv

ifeq ($(ACCELERATOR),GPU)
.PHONY: build
build:
	sudo docker build --target trix_gpu \
		-t $(DOCKER_IMAGE_NAME_GPU) \
		-f build-source/dev.Dockerfile build-source/ $(DOCKER_BUILD_FLAGS) \
		--build-arg BUILD_FOR_GPU="true" \
		--build-arg DEFAULT_JAX_PLATFORM_NAME="gpu" \
		--build-arg BASE_IMAGE="nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04" \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID)


.PHONY: dev_container
dev_container: build
	sudo docker run -it $(DOCKER_RUN_FLAGS_GPU) $(DOCKER_IMAGE_NAME_GPU) /bin/bash


.PHONY: notebook
notebook: build
	echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
	$(SUDO_FLAG) docker run -p $(PORT):$(PORT) -it $(DOCKER_RUN_FLAGS_GPU) \
		$(DOCKER_VARS_TO_PASS) $(DOCKER_IMAGE_NAME_GPU) \
		jupyter lab --port=$(PORT) --no-browser --ip=0.0.0.0 --allow-root
		notebook: build


else ifeq ($(ACCELERATOR),CPU)
.PHONY: build
build:
	$(SUDO_FLAG) docker build --target trix_base \
		-t $(DOCKER_IMAGE_NAME_CPU) \
		-f build-source/dev.Dockerfile build-source/ $(DOCKER_BUILD_FLAGS)

.PHONY: dev_container
dev_container: build
	sudo docker run -it $(DOCKER_RUN_FLAGS_CPU) $(DOCKER_IMAGE_NAME_CPU) /bin/bash

.PHONY: dev_docs
dev_docs: build
	sudo docker run -p $(PORT_DOC):$(PORT_DOC) -it $(DOCKER_RUN_FLAGS_CPU) $(DOCKER_IMAGE_NAME_CPU) /bin/bash


.PHONY: notebook
notebook: build
	echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
	$(SUDO_FLAG) docker run -p $(PORT):$(PORT) -it $(DOCKER_RUN_FLAGS_CPU) \
		$(DOCKER_VARS_TO_PASS) $(DOCKER_IMAGE_NAME_CPU) \
		jupyter lab --port=$(PORT) --no-browser --ip=0.0.0.0 --allow-root

else ifeq ($(ACCELERATOR),TPU)
.PHONY: build
build:
	sudo docker build --target trix_base -t $(DOCKER_IMAGE_NAME_TPU) \
		-f build-source/dev.Dockerfile build-source/ $(DOCKER_BUILD_FLAGS) \
		--build-arg BUILD_FOR_TPU="true"

.PHONY: dev_container
dev_container: build
	sudo docker run -it --rm $(DOCKER_RUN_FLAGS_TPU) $(DOCKER_IMAGE_NAME_TPU) /bin/bash

.PHONY: notebook
notebook: build
	echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
	$(SUDO_FLAG) docker run -p $(PORT):$(PORT) -it $(DOCKER_RUN_FLAGS_TPU) \
	$(DOCKER_VARS_TO_PASS) $(DOCKER_IMAGE_NAME_TPU) \
	jupyter lab --port=$(PORT) --no-browser --ip=0.0.0.0 --allow-root
endif

attach_disk:
	$(BASE_CMD) attach-disk $(NAME) --zone $(ZONE) --project $(PROJECT) --disk $(DISK_NAME) --mode read-write

mount_disk_and_format:
	$(BASE_CMD) ssh $(NAME) --zone $(ZONE) --project $(PROJECT) --command="sudo mkdir -p /mnt/disks/persist && sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb && sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist"
