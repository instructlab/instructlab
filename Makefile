# SPDX-License-Identifier: Apache-2.0

BUILD_ARGS =
CENGINE ?= podman
CONTAINER_PREFIX ?= localhost/instructlab
TOOLBOX ?= instructlab

NULL =
COMMON_DEPS = \
	$(CURDIR)/requirements.txt \
	$(NULL)

CUDA_CONTAINERFILE = $(CURDIR)/containers/cuda/Containerfile
CUDA_DEPS = \
	$(CUDA_CONTAINERFILE) \
	$(COMMON_DEPS) \
	$(NULL)

HPU_CONTAINERFILE = $(CURDIR)/containers/hpu/Containerfile
HPU_DEPS = \
	$(HPU_CONTAINERFILE) \
	$(CURDIR)/requirements-hpu.txt \
	$(COMMON_DEPS) \
	$(NULL)

ROCM_CONTAINERFILE = containers/rocm/Containerfile
ROCM_DEPS = \
	$(ROCM_CONTAINERFILE) \
	$(COMMON_DEPS) \
	containers/rocm/remove-gfx.sh \
	$(NULL)

# so users can do "make rocm-gfx1100 rocm-toolbox"
.NOTPARALLEL:
.PHONY: all
all: help

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build

.PHONY: images
images: ## Get the current controller, set the path, and build the Containerfile image. (auto-detect the compatible controller)
	@if lspci -d '10DE:*:0300'| grep -q "NVIDIA"; then \
		$(MAKE) cuda; \
	elif lspci -d '1002:*:0300' | grep -q "AMD"; then \
		$(MAKE) rocm; \
	else \
		echo "ERROR: Unable to detect AMD / Nvidia GPU" >&2; \
		exit 2; \
	fi

.PHONY: cuda
cuda: $(CUDA_DEPS)  ## Build container for NVidia CUDA
	$(CENGINE) build $(BUILD_ARGS) \
		-t $(CONTAINER_PREFIX):$@ \
		-f $(CUDA_CONTAINERFILE) \
		.

# The base container uses uids beyond 65535. Rootless builds may not work
# unless the user account has extended subordinate ids up to 2**24 - 1.
.PHONY: hpu
hpu: $(HPU_DEPS)  ## Build container for Intel Gaudi HPU
	$(CENGINE) build $(BUILD_ARGS) \
		-t $(CONTAINER_PREFIX):$@ \
		-f $< \
		.

define build-rocm =
	@echo "Building ROCm container for GPU arch '$(1)', version '$(2)'"
	$(CENGINE) build $(BUILD_ARGS) \
		--build-arg AMDGPU_ARCH=$(1) \
		--build-arg HSA_OVERRIDE_GFX_VERSION=$(2) \
		-t $(CONTAINER_PREFIX):rocm-$(1) \
		-t $(CONTAINER_PREFIX):rocm \
		-f $(ROCM_CONTAINERFILE) \
		.
endef

.PHONY: rocm
rocm:   ## Build container for AMD ROCm (auto-detect the ROCm GPU arch)
	@# amdgpu-arch requires clang-tools-extra and rocm-hip-devel to work
	@# rocminfo parsing is more messy, but requires less dependencies.
	@arch=$(shell rocminfo | grep -o -m1 'gfx[1-9][0-9a-f]*'); \
	    if test -z $$arch; then echo "Unable to detect AMD GPU arch"; exit 2; fi; \
	    echo "Auto-detected GPU arch '$$arch'"; \
	    $(MAKE) rocm-$${arch}

# RDNA3 / Radeon RX 7000 series
.PHONY: rocm-gfx1100 rocm-rdna3 rocm-gfx1101 rocm-gfx1102
rocm-rdna3 rocm-gfx1101 rocm-gfx1102: rocm-gfx1100

rocm-gfx1100: $(ROCM_DEPS)  ## Build container for AMD ROCm RDNA3 (Radeon RX 7000 series)
	$(call build-rocm,gfx1100,11.0.0)

# RDNA2 / Radeon RX 6000 series
.PHONY: rocm-gfx1030 rocm-rdna2 rocm-gfx1031
rocm-rdna2 rocm-gfx1031: rocm-gfx1030

rocm-gfx1030: $(ROCM_DEPS)  ## Build container for AMD ROCm RDNA2 (Radeon RX 6000 series)
	$(call build-rocm,gfx1030,10.3.0)

# untested older cards
# Fedora
.PHONY: rocm-gfx90a
rocm-gfx90a: $(ROCM_DEPS)  ## Build container for AMD ROCm gfx90a (untested)
	$(call build-rocm,gfx90a,9.0.10)

.PHONY: rocm-gfx908
rocm-gfx908: $(ROCM_DEPS)  ## Build container for AMD ROCm gfx908 (untested)
	$(call build-rocm,gfx908,9.0.8)

.PHONY: rocm-gfx906
rocm-gfx906: $(ROCM_DEPS)  ## Build container for AMD ROCm gfx906 (untested)
	$(call build-rocm,gfx906,9.0.6)

.PHONY: rocm-gfx900
rocm-gfx900: $(ROCM_DEPS)  ## Build container for AMD ROCm gfx900 (untested)
	$(call build-rocm,gfx900,9.0.0)

.PHONY: rocm-toolbox
toolbox-rocm:  ## Create AMD ROCm toolbox from container
	toolbox create --image $(CONTAINER_PREFIX):rocm $(TOOLBOX)

.PHONY: toolbox-rm
toolbox-rm:  ## Stop and remove toolbox container
	toolbox rm -f $(TOOLBOX)

##@ Development

.PHONY: tests
tests: ## Run tox -e unit against code
	tox -e py3-unit

.PHONY: verify
verify: ## Run tox -e fmt,lint,spellcheck against code
	tox p -e ruff,fastlint,spellcheck

.PHONY: spellcheck-sort
spellcheck-sort: .spellcheck-en-custom.txt
	sort -d -o $< $<

.PHONY: man
man:
	tox -e docs

.PHONY: docs
docs: man ## Run tox -e docs against code

##@ Linting

#
# If you want to see the full commands, run:
#   NOISY_BUILD=y make
#
ifeq ($(NOISY_BUILD),)
    ECHO_PREFIX=@
    CMD_PREFIX=@
    PIPE_DEV_NULL=> /dev/null 2> /dev/null
else
    ECHO_PREFIX=@\#
    CMD_PREFIX=
    PIPE_DEV_NULL=
endif

.PHONY: md-lint
md-lint: ## Lint markdown files
	$(ECHO_PREFIX) printf "  %-12s ./...\n" "[MD LINT]"
	$(CMD_PREFIX) podman run --rm -v $(CURDIR):/workdir --security-opt label=disable docker.io/davidanson/markdownlint-cli2:v0.12.1 > /dev/null
