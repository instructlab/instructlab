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
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build

.PHONY: images
images: ## Get the current controller, set the path, and build the Containerfile image (auto-detect the compatible controller)
	@if ! command -v lspci &> /dev/null; then \
		echo "ERROR: Unable to detect GPU, lspci is not installed" >&2; \
		exit 2; \
	elif lspci -d '10DE:*:0300'| grep -q "NVIDIA"; then \
		$(MAKE) cuda; \
	elif lspci -d '1002:*:0300' | grep -q "AMD"; then \
		$(MAKE) rocm; \
	else \
		echo "ERROR: Unable to detect AMD / Nvidia GPU" >&2; \
		exit 2; \
	fi

.PHONY: cuda
cuda: check-engine $(CUDA_DEPS)  ## Build container for Nvidia CUDA
	$(CENGINE) build $(BUILD_ARGS) \
		-t $(CONTAINER_PREFIX):$@ \
		-f $(CUDA_CONTAINERFILE) \
		.

# The base container uses uids beyond 65535. Rootless builds may not work
# unless the user account has extended subordinate ids up to 2**24 - 1.
.PHONY: hpu
hpu: $(HPU_DEPS) check-engine ## Build container for Intel Gaudi HPU
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
rocm: check-rocm  ## Build container for AMD ROCm (auto-detect the ROCm GPU arch)
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
toolbox-rocm: check-toolbox ## Create AMD ROCm toolbox from container
	toolbox create --image $(CONTAINER_PREFIX):rocm $(TOOLBOX)

.PHONY: toolbox-rm
toolbox-rm: check-toolbox ## Stop and remove toolbox container
	toolbox rm -f $(TOOLBOX)

##@ Development

.PHONY: tests
tests: check-tox ## Run unit and type checks
	tox -e py3-unit,mypy

.PHONY: regenerate-testdata
regenerate-testdata: check-tox ## Run unit tests and regenerate test data
	tox -e py3-unit -- --regenerate-testdata

.PHONY: verify
verify: check-tox ## Run linting and formatting checks via tox
	tox p -m fastverify

.PHONY: fix
fix: check-tox ## Fix formatting and linting violation with Ruff
	tox -e fix

.PHONY: spellcheck
spellcheck: .spellcheck.yml ## Spellcheck markdown files
	pyspelling --config $<

.PHONY: spellcheck-sort
spellcheck-sort: .spellcheck-en-custom.txt ## Sort and remove duplicate from the spellcheck custom file
	sort --dictionary-order --unique --output $< $<

.PHONY: docs
docs: check-tox  ## Generate Sphinx docs and man pages
	tox -e docs
	@echo
	@echo "Sphinx: docs/build/html/index.html"
	@echo "man pages: man/"

.PHONY: man
man: docs

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
md-lint: check-engine ## Lint markdown files
	$(ECHO_PREFIX) printf "  %-12s ./...\n" "[MD LINT]"
	$(CMD_PREFIX) $(CENGINE) run --rm -v $(CURDIR):/workdir --security-opt label=disable docker.io/davidanson/markdownlint-cli2:v0.12.1 > /dev/null

.PHONY: toml-lint
toml-lint: check-engine ## Lint pyproject.toml
	$(ECHO_PREFIX) printf "  %-12s ./...\n" "[TOML LINT]"
	$(CMD_PREFIX) $(CENGINE) run --rm -v $(CURDIR):/workdir --security-opt label=disable docker.io/tamasfe/taplo:0.8.1 lint /workdir/pyproject.toml

.PHONY: toml-fmt
toml-fmt: check-engine ## Format pyproject.toml
	$(ECHO_PREFIX) printf "  %-12s ./...\n" "[TOML FMT]"
	$(CMD_PREFIX) $(CENGINE) run --rm -v $(CURDIR):/workdir --security-opt label=disable docker.io/tamasfe/taplo:0.8.1 fmt /workdir/pyproject.toml

.PHONY: check-tox
check-tox:
	@command -v tox &> /dev/null || (echo "'tox' is not installed" && exit 1)

.PHONY: check-toolbox
check-toolbox:
	@command -v toolbox &> /dev/null || (echo "'toolbox' is not installed" && exit 1)

.PHONY: check-engine
check-engine:
	@command -v $(CENGINE) &> /dev/null || (echo "'$(CENGINE)' container engine is not installed, you can override it with the 'CENGINE' variable" && exit 1)

.PHONY: check-rocm
check-rocm:
	@command -v rocm &> /dev/null || (echo "'rocm' is not installed" && exit 1)

.PHONY: action-lint actionlint
action-lint: actionlint
actionlint: ## Lint GitHub Action workflows
	$(ECHO_PREFIX) printf "  %-12s .github/...\n" "[ACTION LINT]"
	$(CMD_PREFIX) if ! command -v actionlint $(PIPE_DEV_NULL) ; then \
		echo "Please install actionlint." ; \
		echo "go install github.com/rhysd/actionlint/cmd/actionlint@latest" ; \
		exit 1 ; \
	fi
	$(CMD_PREFIX) if ! command -v shellcheck $(PIPE_DEV_NULL) ; then \
		echo "Please install shellcheck." ; \
		echo "https://github.com/koalaman/shellcheck#user-content-installing" ; \
		exit 1 ; \
	fi
	$(CMD_PREFIX) actionlint -color
