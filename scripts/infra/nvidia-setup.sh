#!/bin/bash
# shellcheck disable=SC2155,SC2116,SC2086,SC2155,SC2318,SC2206,SC2126,SC2001,SC1091
# Derived from:
# github.com/containers/ai-lab-recipes/blob/main/training/nvidia-bootc/Containerfile
DRIVER_VERSION="570.86.15"
CUDA_VERSION='12.8.0'
BASE_URL='https://us.download.nvidia.com/tesla'

if [[ $(id -u) != "0" ]]; then
    echo "you must run this script as root."
    exit 1
fi
set -x

dnf config-manager --save \
  --setopt=skip_missing_names_on_install=False \
  --setopt=install_weak_deps=False

cat <<FILEEOF > x509-configuration.ini
[ req ]
default_bits = 4096
distinguished_name = req_distinguished_name
prompt = no
string_mask = utf8only
x509_extensions = myexts
[ req_distinguished_name ]
O = Project Magma
CN = Project Magma
emailAddress = magma@acme.com
[ myexts ]
basicConstraints=critical,CA:FALSE
keyUsage=digitalSignature
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid
FILEEOF

mkdir -p /usr/lib/systemd/system
cat <<FILEEOF > /usr/lib/systemd/system/nvidia-toolkit-setup.service
[Unit]
Description=Generate /etc/cdi/nvidia.yaml

[Service]
Type=oneshot
ExecStart=nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
FILEEOF

rm -rf yum-packaging-precompiled-kmod
dnf install libicu podman skopeo git-core rpm-build make openssl elfutils-libelf-devel python3.12 python3.12-devel -y
if [ -z "${KERNEL_VERSION}" ]; then
      KERNELS=$(rpm -q --qf="%{BUILDTIME} %{VERSION}-%{RELEASE}\n" kernel-core | sort -n)
      KERNEL_COUNT=$(echo "${KERNELS}" | wc -l)
      LATEST_KERNEL=$(echo "${KERNELS}" | tail -1)
      export KERNEL_VERSION=${LATEST_KERNEL#* }
      if [[ "$KERNEL_COUNT" -gt 1 ]]; then
        echo Multiple kernels found, building for latest ${KERNEL_VERSION}.
      fi
fi
dnf install -y "kernel-devel-${KERNEL_VERSION}" \
    && if [ "${OS_VERSION_MAJOR}" == "" ]; then \
        . /etc/os-release \
	&& export OS_ID="$(echo ${ID})" \
        && export OS_VERSION_MAJOR="$(echo ${VERSION} | cut -d'.' -f 1)" ;\
       fi \
    && if [ "${BUILD_ARCH}" == "" ]; then \
        export BUILD_ARCH=$(arch) \
        && export TARGET_ARCH=$(echo "${BUILD_ARCH}" | sed 's/+64k//') ;\
        fi \
    && export KVER=$(echo ${KERNEL_VERSION} | cut -d '-' -f 1) \
    && KREL=$(echo ${KERNEL_VERSION} | cut -d '-' -f 2 | sed 's/\.el._*.*\..\+$//' | cut -d'.' -f 1) \
    && if [ "${OS_ID}" == "rhel" ]; then \
		KDIST="."$(echo ${KERNEL_VERSION} | cut -d '-' -f 2 | cut -d '.' -f 2-) ;\
	else \
		KDIST="."$(echo ${KERNEL_VERSION} | cut -d '-' -f 2 | sed 's/^.*\(\.el._*.*\)\..\+$/\1/' | cut -d'.' -f 2) ;\
	fi \
    && DRIVER_STREAM=$(echo ${DRIVER_VERSION} | cut -d '.' -f 1) \
    && git clone --depth 1 --single-branch -b rhel${OS_VERSION_MAJOR} https://github.com/NVIDIA/yum-packaging-precompiled-kmod \
    && cd yum-packaging-precompiled-kmod \
    && mkdir BUILD BUILDROOT RPMS SRPMS SOURCES SPECS \
    && mkdir nvidia-kmod-${DRIVER_VERSION}-${BUILD_ARCH} \
    && curl -sLOf ${BASE_URL}/${DRIVER_VERSION}/NVIDIA-Linux-${TARGET_ARCH}-${DRIVER_VERSION}.run \
    && sh ./NVIDIA-Linux-${TARGET_ARCH}-${DRIVER_VERSION}.run --extract-only --target tmp \
    && mv tmp/kernel-open nvidia-kmod-${DRIVER_VERSION}-${BUILD_ARCH}/kernel \
    && tar -cJf SOURCES/nvidia-kmod-${DRIVER_VERSION}-${BUILD_ARCH}.tar.xz nvidia-kmod-${DRIVER_VERSION}-${BUILD_ARCH} \
    && mv kmod-nvidia.spec SPECS/ \
    && openssl req -x509 -new -nodes -utf8 -sha256 -days 36500 -batch \
      -config "$(pwd)/../x509-configuration.ini" \
      -outform DER -out SOURCES/public_key.der \
      -keyout SOURCES/private_key.priv \
    && rpmbuild \
        --define "% _arch ${BUILD_ARCH}" \
        --define "%_topdir $(pwd)" \
        --define "debug_package %{nil}" \
        --define "kernel ${KVER}" \
        --define "kernel_release ${KREL}" \
        --define "kernel_dist ${KDIST}" \
        --define "driver ${DRIVER_VERSION}" \
        --define "driver_branch ${DRIVER_STREAM}" \
        -v -bb SPECS/kmod-nvidia.spec \
&& cd .. \
&& mkdir -p /lib/firmware/nvidia/${DRIVER_VERSION} \
&& cp -rp yum-packaging-precompiled-kmod/tmp/firmware/*.bin /lib/firmware/nvidia/${DRIVER_VERSION}/ \
    && dnf install -y yum-packaging-precompiled-kmod/RPMS/${BUILD_ARCH}/kmod-nvidia-*.rpm \
    && if [ "${TARGET_ARCH}" == "" ]; then \
        export TARGET_ARCH="$(arch)" ;\
        fi \
    && if [ "${OS_VERSION_MAJOR}" == "" ]; then \
        . /etc/os-release \
        && export OS_VERSION_MAJOR="$(echo ${VERSION} | cut -d'.' -f 1)" ;\
       fi \
    && export DRIVER_STREAM=$(echo ${DRIVER_VERSION} | cut -d '.' -f 1) \
        CUDA_VERSION_ARRAY=(${CUDA_VERSION//./ }) \
        CUDA_DASHED_VERSION=${CUDA_VERSION_ARRAY[0]}-${CUDA_VERSION_ARRAY[1]} \
        CUDA_MAJOR_MINOR=${CUDA_VERSION_ARRAY[0]}.${CUDA_VERSION_ARRAY[1]}
        CUDA_REPO_ARCH=${TARGET_ARCH} \
    && if [ "${TARGET_ARCH}" == "aarch64" ]; then CUDA_REPO_ARCH="sbsa"; fi \
    && cp -a /etc/dnf/dnf.conf{,.tmp} && mv /etc/dnf/dnf.conf{.tmp,} \
    && dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION_MAJOR}/${CUDA_REPO_ARCH}/cuda-rhel${OS_VERSION_MAJOR}.repo \
    && dnf module reset -y nvidia-driver \
    && dnf -y module enable nvidia-driver:${DRIVER_STREAM}-open/default \
    && export NCCL_PACKAGE=$(dnf search libnccl --showduplicates 2>/dev/null | grep ${CUDA_MAJOR_MINOR} | awk '{print $1}' | grep libnccl-2 | tail -1) \
    && dnf install -y \
        cloud-init \
        pciutils \
        tmux \
        nvidia-driver-cuda-${DRIVER_VERSION} \
        nvidia-driver-libs-${DRIVER_VERSION} \
        libnvidia-ml-${DRIVER_VERSION} \
        cuda-compat-${CUDA_DASHED_VERSION} \
        cuda-cudart-${CUDA_DASHED_VERSION} \
        cuda-cudart-devel-${CUDA_DASHED_VERSION} \
        libcublas-${CUDA_DASHED_VERSION} \
        libcublas-devel-${CUDA_DASHED_VERSION} \
        cuda-nvcc-${CUDA_DASHED_VERSION} \
        cuda-nvtx-${CUDA_DASHED_VERSION} \
        cuda-cupti-${CUDA_DASHED_VERSION} \
        libcusparse-${CUDA_DASHED_VERSION} \
        libcusparse-devel-${CUDA_DASHED_VERSION} \
        libcusolver-${CUDA_DASHED_VERSION} \
        libcusolver-devel-${CUDA_DASHED_VERSION} \
        libcufft-${CUDA_DASHED_VERSION} \
        libcurand-${CUDA_DASHED_VERSION} \
        libnvjitlink-${CUDA_DASHED_VERSION} \
        cudnn9-cuda-${CUDA_DASHED_VERSION} \
        cuda-nvrtc-${CUDA_DASHED_VERSION} \
        cuda-nvrtc-devel-${CUDA_DASHED_VERSION} \
        ${NCCL_PACKAGE} \
        libcudnn8 \
        nvidia-persistenced-${DRIVER_VERSION} \
        nvidia-container-toolkit \
        rsync \
        ${EXTRA_RPM_PACKAGES} \
    && if [ "$DRIVER_TYPE" != "vgpu" ] && [ "$TARGET_ARCH" != "arm64" ]; then \
        versionArray=(${DRIVER_VERSION//./ }); \
        DRIVER_BRANCH=${versionArray[0]}; \
        dnf module reset -y nvidia-driver && \
        dnf module enable -y nvidia-driver:${DRIVER_BRANCH}-open && \
        dnf install -y nvidia-fabric-manager-${DRIVER_VERSION} libnvidia-nscq-${DRIVER_BRANCH}-${DRIVER_VERSION} ; \
    fi \
    && . /etc/os-release && if [ "${ID}" == "rhel" ]; then \
        dnf install -y rhc rhc-worker-playbook; \
        sed -i -e "/^VARIANT=/ {s/^VARIANT=.*/VARIANT=\"RHEL AI\"/; t}" -e "\$aVARIANT=\"RHEL AI\"" /usr/lib/os-release; \
        sed -i -e "/^VARIANT_ID=/ {s/^VARIANT_ID=.*/VARIANT_ID=rhel_ai/; t}" -e "\$aVARIANT_ID=rhel_ai" /usr/lib/os-release; \
        sed -i -e "/^BUILD_ID=/ {s/^BUILD_ID=.*/BUILD_ID='${IMAGE_VERSION}'/; t}" -e "\$aBUILD_ID='${IMAGE_VERSION}'" /usr/lib/os-release; \
        fi \
    && dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm \
    && dnf install -y epel-release epel-next-release \
    && dnf install -y nvtop \
    && dnf clean all \
    && echo "blacklist nouveau" > /etc/modprobe.d/blacklist_nouveau.conf \
    && ln -f -s /usr/lib/systemd/system/nvidia-toolkit-setup.service /usr/lib/systemd/system/basic.target.wants/nvidia-toolkit-setup.service \
    && ln -f -s /usr/lib/systemd/system/nvidia-persistenced.service /etc/systemd/system/multi-user.target.wants/nvidia-persistenced.service
    systemctl daemon-reload
    systemctl enable --now nvidia-toolkit-setup.service

# Ensure the next kernel that we boot will match the driver we installed.
EXPECTED_VMLINUZ=/boot/vmlinuz-${KERNEL_VERSION}.$(uname -m)
grubby --set-default=$EXPECTED_VMLINUZ

# If the driver does not match our installed kernel, warn the user to reboot.
RUNNING_KERNEL=$(uname -r | sed 's/\.[a-z0-9_-]*$//')
if [[ "$RUNNING_KERNEL" != "$KERNEL_VERSION" ]]; then
    # The running kernel does not match the nvidia driver we installed above.
    echo "You are running kernel $RUNNING_KERNEL. Reboot into kernel $KERNEL_VERSION."
fi
