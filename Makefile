.PHONY: all
all: images tests functional verify

.PHONY: images
images: # Get the current controller, set the container path, and build the Containerfile image
	@if lspci | grep -q "NVIDIA"; then \
		podman build --ssh=default -f containers/cuda/Containerfile; \
		break; \
	elif lspci | grep -q "AMD"; then \
		podman build --ssh=default -f containers/rocm/Containerfile; \
		break; \
	fi

.PHONY: tests
tests: # Run the unit tests
	tox -e unit

.PHONY: verify
verify: # Run the formatters and linters
	tox -e fmt
	tox -e lint