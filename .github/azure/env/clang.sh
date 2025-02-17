#! /bin/bash

#===============================================================================
# Copyright 2019-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

VERSION=${1//[^0-9.]/}
if [ -z "${VERSION}" ]; then
    VERSION=9
fi

UBUNTU_DISTRO="$(cat /etc/lsb-release | grep CODENAME | sed 's/.*=//g')"

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 15CF4D18AF4F7421

sudo add-apt-repository "deb http://apt.llvm.org/${UBUNTU_DISTRO}/ llvm-toolchain-${UBUNTU_DISTRO}-${VERSION} main"
sudo add-apt-repository "deb-src http://apt.llvm.org/${UBUNTU_DISTRO}/ llvm-toolchain-${UBUNTU_DISTRO}-${VERSION} main"
sudo apt update && sudo apt install -y "clang-${VERSION}" "lldb-${VERSION}" "lld-${VERSION}" "clang-format-${VERSION}"  "libomp-${VERSION}-dev" hwloc

sudo update-alternatives --install /usr/bin/clang clang "/usr/bin/clang-${VERSION}" 100
sudo update-alternatives --install /usr/bin/clang++ clang++ "/usr/bin/clang++-${VERSION}" 100
sudo update-alternatives --install /usr/bin/clang-format clang-format "/usr/bin/clang-format-${VERSION}" 100
sudo update-alternatives --set clang "/usr/bin/clang-${VERSION}"
sudo update-alternatives --set clang++ "/usr/bin/clang++-${VERSION}"
sudo update-alternatives --set clang-format "/usr/bin/clang-format-${VERSION}"
