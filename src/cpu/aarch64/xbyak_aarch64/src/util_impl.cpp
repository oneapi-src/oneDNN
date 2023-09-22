/*******************************************************************************
 * Copyright 2020-2023 FUJITSU LIMITED
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xbyak_aarch64_err.h"
#include "xbyak_aarch64_util.h"

#include "util_impl.h"

#if defined(__linux__)
#include "util_impl_linux.h"
#elif defined(__APPLE__)
#include "util_impl_mac.h"
#elif defined(_M_ARM64)
#include "util_impl_windows.h"
#else
#error "Unsupported OS"
#endif

namespace Xbyak_aarch64 {
namespace util {
void CpuInfo::dumpCacheInfo() const {
  printf("numCores=%d\n", numCores_[/* Phisical cores */ 1]);
  for (size_t i = 0; i < maxCacheLevel; i++) {
    auto cache = &cacheInfo_.levelCache[i];
    printf("L%zd, %d, %d, %d, %d, %d, %d, %d\n", i + 1, cache->type, cache->size[0], cache->size[1], cache->size[2], cache->sharingCores[0], cache->sharingCores[1], cache->sharingCores[2]);
  }
}

int CpuInfo::getCacheSize(cacheType_t type, uint32_t level) const {
  if (level <= maxCacheLevel) {
    auto cache = &cacheInfo_.levelCache[level - 1];
    switch (type) {
    case ICache:
      return cache->size[0];
      break;
    case DCache:
      return cache->size[1];
      break;
    case UCache:
      return cache->size[2];
      break;
    default:
      throw Error(ERR_BAD_PARAMETER);
      break;
    }
  } else {
    throw Error(ERR_BAD_PARAMETER);
  }
}

Arm64CacheType CpuInfo::getCacheType(int level) const { return cacheInfo_.levelCache[level - 1].type; }
int CpuInfo::getCodeCacheSize(int level) const { return cacheInfo_.levelCache[level - 1].size[0]; }

int CpuInfo::getCoresSharingDataCache(int level) const {
  auto cache = &cacheInfo_.levelCache[level - 1];
  int cores;

  switch (cache->type) {
  case DataCacheOnly:
  case SeparateCache:
    cores = cache->sharingCores[1];
    break;
  case UnifiedCache:
    cores = cache->sharingCores[2];
    break;
  default:
    cores = 0;
    break;
  }

  return cores;
}

int CpuInfo::getDataCacheSize(int level) const { return cacheInfo_.levelCache[level - 1].size[1]; }

const char *CpuInfo::getImplementer() const { return implementer_; }
int CpuInfo::getLastDataCacheLevel() const { return lastDataCacheLevel_; }

int CpuInfo::getNumCores(Arm64CpuTopologyLevel level) const {
  switch (level) {
  case SmtLevel:
    return numCores_[0];
    break;
  case CoreLevel:
    return numCores_[1];
    break;
  default:
    return 0;
  }
}

uint64_t CpuInfo::getSveLen() const { return sveLen_; }
Type CpuInfo::getType() const { return type_; }
int CpuInfo::getUnifiedCacheSize(int level) const { return cacheInfo_.levelCache[level - 1].size[2]; }

void CpuInfo::init() {
  for (size_t i = 0; i < maxCacheLevel; i++) {
    auto cache = &cacheInfo_.levelCache[i];
    cache->type = NoCache;
    cache->size[0] = cache->size[1] = cache->size[2] = 0;
    cache->sharingCores[0] = cache->sharingCores[1] = cache->sharingCores[2] = 0;
  }
}

void CpuInfo::put() const {
  printf("numCores=%d\n", numCores_[0]);
  for (int level = 1; level <= 3; level++) {
    printf("L%d unified size = %d\n", level, getUnifiedCacheSize(level));
    printf("L%d code size = %d\n", level, getCodeCacheSize(level));
    printf("L%d data size = %d\n", level, getDataCacheSize(level));
  }
}

void CpuInfo::setImplementer() {
  const uint32_t id = (cacheInfo_.midr_el1 >> 24) & 0xff;
  const int lastId = sizeof(implementers) / sizeof(implementer_t);

  for (int i = 0; i < lastId; i++) {
    if (implementers[i].id == id) {
      implementer_ = implementers[i].implementer;
      return;
    }
  }
  implementer_ = (char *)implementers[lastId - 1].implementer;
}

void CpuInfo::setLastDataCacheLevel() {
  for (uint32_t i = 0; i < maxCacheLevel; i++) {
    const Arm64CacheType type = cacheInfo_.levelCache[i].type;
    if (type == DataCacheOnly || type == SeparateCache || type == UnifiedCache)
      lastDataCacheLevel_ = i + 1;
  }
}

Cpu::Cpu() {
#if defined(__linux__)
  info = new CpuInfoLinux();
#elif defined(__APPLE__)
  info = new CpuInfoMac();
#elif defined(_M_ARM64)
  info = new CpuInfoWindows();
#endif
}

void Cpu::dumpCacheInfo() const { return info->dumpCacheInfo(); }

Arm64CacheType Cpu::getCacheType(const Arm64CacheLevel i) const { return info->getCacheType(i); }

uint32_t Cpu::getCoresSharingDataCache(const Arm64CacheLevel i) const { return info->getCoresSharingDataCache(i); }

uint32_t Cpu::getDataCacheSize(const Arm64CacheLevel i) const {
  uint32_t size;
  switch (info->getCacheType(i)) {
  case DataCacheOnly:
  case SeparateCache:
    size = info->getDataCacheSize(i);
    break;
  case UnifiedCache:
    size = info->getUnifiedCacheSize(i);
    break;
  default:
    size = 0;
    break;
  }

  return size;
}

const char *Cpu::getImplementer() const { return info->getImplementer(); }

uint32_t Cpu::getLastDataCacheLevel() const { return info->getLastDataCacheLevel(); }
uint32_t Cpu::getNumCores(Arm64CpuTopologyLevel level) const { return info->getNumCores(level); }
uint64_t Cpu::getSveLen() const { return info->getSveLen(); }
Type Cpu::getType() const { return info->getType(); }
bool Cpu::has(Type type) const { return (type & info->getType()) != 0; }
bool Cpu::isAtomicSupported() const { return info->getType() & (Type)XBYAK_AARCH64_HWCAP_ATOMIC; }
bool Cpu::isBf16Supported() const { return info->getType() & (Type)XBYAK_AARCH64_HWCAP_BF16; }

} // namespace util
} // namespace Xbyak_aarch64
