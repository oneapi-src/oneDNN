/*******************************************************************************
 * Copyright 2020-2022 FUJITSU LIMITED
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
#include "xbyak_aarch64_util.h"

namespace Xbyak_aarch64 {
namespace util {

void Cpu::setCacheHierarchy() {
  /* Cache size of AArch64 CPUs are described in the system registers,
     which can't be read from user-space applications.
     Linux provides `sysconf` API and `/sys/devices/system/cpu/`
     device files to get cache size, but they dosen't always return
     correct values. It may depend on Linux kernel version and
     support status of CPUs. To avoid this situation, cahche size is
     firstly read from `cacheInfoDict`, secondly get by `sysconf`.

     `sysconf` example
     #include <unistd.h>
     int main() {
       reutrn sysconf(_SC_LEVEL1_DCACHE_SIZE);
     }
   */
  const cacheInfo_t *c = nullptr;

  for (size_t j = 0; j < sizeof(cacheInfoDict) / sizeof(cacheInfo_t); j++) {
    if (cacheInfoDict[j].midr_el1 == midr_el1_) {
      c = cacheInfoDict + j;
      break;
    }
  }

  if (c != nullptr) {
    dataCacheLevel_ = c->dataCacheLevel;
    for (uint32_t i = 0; i < maxNumberCacheLevel; i++) {
      if (i < c->highestInnerCacheLevel)
        dataCacheSize_[i] = c->dataCacheSize[i];
      else
        dataCacheSize_[i] = c->dataCacheSize[i] / coresSharingDataCache_[i];
    }
  } else {
    /**
     * @ToDo Get cache information by `sysconf`
     * for the case thd dictionary is unavailable.
     */

// _SC_LEVEL<L>_DCACHE_SIZE macros are not defined on macOS.
#if defined(__APPLE__)
#define GET_CACHE_SIZE(ID) 0
#else
#define GET_CACHE_SIZE(ID) sysconf(ID)
#endif

#define XBYAK_AARCH64_CACHE_SIZE(LEVEL, SIZE, ID, CORES, VAL)                                                                                                                                                                                                                                              \
  cache_size = GET_CACHE_SIZE(ID);                                                                                                                                                                                                                                                                         \
  VAL[LEVEL] = cache_size ? (cache_size / (CORES)) : ((SIZE) / (CORES));

    uint32_t cache_size;

    /* If `sysconf` returns zero as cache sizes, 32KiB, 1MiB, 0 and 0 is set as
       1st, 2nd, 3rd and 4th level cache sizes. 2nd cache is assumed as sharing cache. */
    XBYAK_AARCH64_CACHE_SIZE(0, 1024 * 32, _SC_LEVEL1_DCACHE_SIZE, 1, coresSharingDataCache_);
    XBYAK_AARCH64_CACHE_SIZE(1, 1024 * 1024, _SC_LEVEL2_CACHE_SIZE, 1, coresSharingDataCache_);
    XBYAK_AARCH64_CACHE_SIZE(2, 0, _SC_LEVEL3_CACHE_SIZE, 1, coresSharingDataCache_);
    XBYAK_AARCH64_CACHE_SIZE(3, 0, _SC_LEVEL4_CACHE_SIZE, 1, coresSharingDataCache_);

    XBYAK_AARCH64_CACHE_SIZE(0, 1024 * 32, _SC_LEVEL1_DCACHE_SIZE, 1, dataCacheSize_);
    XBYAK_AARCH64_CACHE_SIZE(1, 1024 * 1024, _SC_LEVEL2_CACHE_SIZE, 8, dataCacheSize_);
    XBYAK_AARCH64_CACHE_SIZE(2, 0, _SC_LEVEL3_CACHE_SIZE, 1, dataCacheSize_);
    XBYAK_AARCH64_CACHE_SIZE(3, 0, _SC_LEVEL4_CACHE_SIZE, 1, dataCacheSize_);
#undef XBYAK_AARCH64_CACHE_SIZE
  }
}

void Cpu::setNumCores() {
#ifdef __linux__
  /**
   * @ToDo There are some methods to get # of cores.
   Considering various kernel versions and CPUs, a combination of
   multiple methods may be required.
   1) sysconf(_SC_NPROCESSORS_ONLN)
   2) /sys/devices/system/cpu/online
   3) std::thread::hardware_concurrency()
  */
  numCores_[0] = numCores_[1] = sysconf(_SC_NPROCESSORS_ONLN);
  coresSharingDataCache_[0] = 1;

  /* # of numa nodes: /sys/devices/system/node/node[0-9]+
     # of cores for each numa node: /sys/devices/system/node/node[0-9]+/cpu[0-9]+
     It is assumed L2 cache is shared by each numa node. */
  const int nodes = getFilePathMaxTailNumPlus1(XBYAK_AARCH64_PATH_NODES);
  int cores = 1;

  if (nodes > 0) {
    cores = getFilePathMaxTailNumPlus1(XBYAK_AARCH64_PATH_CORES);
    coresSharingDataCache_[1] = (cores > 0) ? cores : 1;
  } else {
    coresSharingDataCache_[1] = 1;
  }
#else
  numCores_[0] = numCores_[1] = 1;
  for (unsigned int i = 0; i < maxNumberCacheLevel; i++)
    coresSharingDataCache_[i] = 1;

  coresSharingDataCache_[1] = 8; // Set possible value.
#endif
}

void Cpu::setSysRegVal() {
#ifdef __linux__
  XBYAK_AARCH64_READ_SYSREG(midr_el1_, MIDR_EL1);
#endif
}

/**
 * Return directory path
 * @param[in] path ex. /sys/devices/system/node/node
 * @param[out] buf ex. /sys/devices/system/node
 */
int Cpu::getRegEx(char *buf, const char *path, const char *regex) {
  regex_t regexBuf;
  regmatch_t match[1];

  if (regcomp(&regexBuf, regex, REG_EXTENDED) != 0)
    throw ERR_INTERNAL;

  const int retVal = regexec(&regexBuf, path, 1, match, 0);
  regfree(&regexBuf);

  if (retVal != 0)
    return -1;

  const int startIdx = match[0].rm_so;
  const int endIdx = match[0].rm_eo;

  /* Something wrong (multiple match or not match) */
  if (startIdx == -1 || endIdx == -1 || (endIdx - startIdx - 1) < 1)
    return -1;

  strncpy(buf, path + startIdx, endIdx - startIdx);
  buf[endIdx - startIdx] = '\0';

  return 0;
}

int Cpu::getFilePathMaxTailNumPlus1(const char *path) {
#ifdef __linux__
  char dir_path[max_path_len];
  char file_pattern[max_path_len];
  int retVal = 0;

  getRegEx(dir_path, path, "/([^/]+/)+");
  /* Remove last '/'. */
  dir_path[strlen(dir_path) - 1] = '\0';
  getRegEx(file_pattern, path, "[^/]+$");
  strncat(file_pattern, "[0-9]+", 16);

  fflush(stdout);

  DIR *dir = opendir(dir_path);
  struct dirent *dp;

  dp = readdir(dir);
  while (dp != NULL) {
    if (getRegEx(dir_path, dp->d_name, file_pattern) == 0)
      retVal++;
    dp = readdir(dir);
  }

  if (dir != NULL)
    closedir(dir);

  return retVal;
#else
  return 0;
#endif
}

Cpu::Cpu() : type_(tNONE), sveLen_(SVE_NONE) {
#ifdef __linux__
  unsigned long hwcap = getauxval(AT_HWCAP);
  if (hwcap & HWCAP_ATOMICS) {
    type_ |= tATOMIC;
  }

  if (hwcap & HWCAP_FP) {
    type_ |= tFP;
  }
  if (hwcap & HWCAP_ASIMD) {
    type_ |= tADVSIMD;
  }
#ifdef HWCAP_SVE
  /* Some old <sys/auxv.h> may not define HWCAP_SVE.
     In that case, SVE is treated as if it were not supported. */
  if (hwcap & HWCAP_SVE) {
    type_ |= tSVE;
    // svcntb(); if arm_sve.h is available
    sveLen_ = (sveLen_t)prctl(51); // PR_SVE_GET_VL
  }
#endif
#elif defined(__APPLE__)
  size_t val = 0;
  size_t len = sizeof(val);

  if (sysctlbyname(hw_opt_atomics, &val, &len, NULL, 0) != 0)
    throw Error(ERR_INTERNAL);
  else
    type_ |= (val == 1) ? tATOMIC : 0;

  if (sysctlbyname(hw_opt_fp, &val, &len, NULL, 0) != 0)
    throw Error(ERR_INTERNAL);
  else
    type_ |= (val == 1) ? tFP : 0;

  if (sysctlbyname(hw_opt_neon, &val, &len, NULL, 0) != 0)
    throw Error(ERR_INTERNAL);
  else
    type_ |= (val == 1) ? tADVSIMD : 0;
#endif

  setSysRegVal();
  setNumCores();
  setCacheHierarchy();
}

Type Cpu::getType() const { return type_; }
bool Cpu::has(Type type) const { return (type & type_) != 0; }
uint64_t Cpu::getSveLen() const { return sveLen_; }
bool Cpu::isAtomicSupported() const { return type_ & tATOMIC; }
const char *Cpu::getImplementer() const {
  uint64_t implementer = (midr_el1_ >> 24) & 0xff;

  for (size_t i = 0; i < sizeof(implementers) / sizeof(implementer_t); i++) {
    if (implementers[i].id == implementer)
      return implementers[i].implementer;
  }

  return nullptr;
}

uint32_t Cpu::getCoresSharingDataCache(uint32_t i) const {
  if (i >= dataCacheLevel_)
    throw Error(ERR_BAD_PARAMETER);
  return coresSharingDataCache_[i];
}

uint32_t Cpu::getDataCacheLevels() const { return dataCacheLevel_; }

uint32_t Cpu::getDataCacheSize(uint32_t i) const {
  if (i >= dataCacheLevel_)
    throw Error(ERR_BAD_PARAMETER);
  return dataCacheSize_[i];
}
uint32_t Cpu::getNumCores(Arm64CpuTopologyLevel level) const {
  switch (level) {
  case CoreLevel:
    return numCores_[level - 1];
  default:
    throw Error(ERR_BAD_PARAMETER);
  }
}
} // namespace util
} // namespace Xbyak_aarch64
