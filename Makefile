BUILD_ARGS = --ssh=default
CENGINE = podman
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

ROCM_CONTAINERFILE = containers/rocm/Containerfile
ROCM_DEPS = \
	$(ROCM_CONTAINERFILE) \
	$(COMMON_DEPS) \
	containers/rocm/remove-gfx.sh \
	$(NULL)

# so users can do "make rocm-gfx1100 rocm-toolbox"
.NOTPARALLEL:
.PHONY: all
all: images tests functional verify

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build

.PHONY: images
images: ## Get the current controller, set the path, and build the Containerfile image. (auto-detect the compatible controller)
	@if lspci | grep -q "NVIDIA"; then \
		$(CENGINE) build --ssh=default -f containers/cuda/Containerfile; \
		cuda \
		break; \
	elif lspci | grep -q "AMD"; then \
		$(CENGINE) build --ssh=default -f containers/rocm/Containerfile; \
		rocm \
		break; \
	fi

.PHONY: cuda
cuda: $(CUDA_DEPS)  ## Build container for NVidia CUDA
	$(CENGINE) build $(BUILD_ARGS) \
		-t $(CONTAINER_PREFIX):$@ \
		-f $(CUDA_CONTAINERFILE) \
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
	    $(MAKE) --silent rocm-$(AMDGPU_ARCH)

# RDNA3 / Radeon RX 7000 series
.PHONY: rocm-gf1100 rocm-rdna3 rocm-gfx1101 rocm-gfx1102
rocm-rdna3 rocm-gfx1101 rocm-gfx1102: rocm-gfx1100

rocm-gfx1100: $(ROCM_DEPS)  ## Build container for AMD ROCm RDNA3 (Radeon RX 7000 series)
	$(call build-rocm,gfx1100,11.0.0)

# RDNA2 / Radeon RX 6000 series
.PHONY: rocm-gfx1030 rocm-rdna2 rocm-gfx1031
rocm-rdna2 rocm-gfx1031: rocm-gfx1030

rocm-gfx1030: $(ROCM_DEPS)  ## Build container for AMD ROCm RDNA2 (Radeon RX 6000 series)
	$(call build-rocm,gfx1030,10.3.0)

# untested older cards
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
	tox -e unit

.PHONY: verify
verify: ## Run tox -e fmt, lint against code
	tox -e fmt
	tox -e lint