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
#ifndef __APPLE__
#error "Something wrong"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/sysctl.h>

#include "xbyak_aarch64_err.h"
#include "xbyak_aarch64_util.h"

namespace Xbyak_aarch64 {
namespace util {
constexpr char hw_cacheconfig[] = "hw.cacheconfig";
constexpr char hw_l1icachesize[] = "hw.l1icachesize";
constexpr char hw_l1dcachesize[] = "hw.l1dcachesize";
constexpr char hw_l2cachesize[] = "hw.l2cachesize";
constexpr char hw_l3cachesize[] = "hw.l3cachesize";
constexpr char hw_ncpu[] = "hw.ncpu";
constexpr char hw_opt_atomics[] = "hw.optional.armv8_1_atomics";
constexpr char hw_opt_fp[] = "hw.optional.floatingpoint";
constexpr char hw_opt_neon[] = "hw.optional.neon";
constexpr char hw_perflevel1_logicalcpu[] = "hw.perflevel1.logicalcpu";

class CpuInfoMac : public CpuInfo {
public:
  CpuInfoMac() {
    init();
    cacheInfo_.midr_el1 = 0xFE << 24;
    setNumCores();
    setHwCap();
    setCacheHierarchy();
    setImplementer();
  }

private:
  uint8_t sysInfoBuf_[128];

  int getSysInfo(char const *name, const size_t len) {
    size_t len_ = len;
    int retVal;

    if ((retVal = sysctlbyname(name, sysInfoBuf_, &len_, NULL, 0)) != 0)
      memset(sysInfoBuf_, 0, sizeof(sysInfoBuf_));

    return retVal;
  }

  void setCacheHierarchy() {
    // L1 cache
    cacheInfo_.levelCache[0].type = SeparateCache;
    getSysInfo(hw_l1icachesize, sizeof(sysInfoBuf_));
    cacheInfo_.levelCache[0].size[0] = ((int64_t *)sysInfoBuf_)[0];
    getSysInfo(hw_l1dcachesize, sizeof(sysInfoBuf_));
    cacheInfo_.levelCache[0].size[1] = ((int64_t *)sysInfoBuf_)[0];

    // L2 cache
    cacheInfo_.levelCache[1].type = UnifiedCache;
    getSysInfo(hw_l2cachesize, sizeof(sysInfoBuf_));
    cacheInfo_.levelCache[1].size[2] = ((int64_t *)sysInfoBuf_)[0];

    // L3 cache
    cacheInfo_.levelCache[2].type = UnifiedCache;
    getSysInfo(hw_l3cachesize, sizeof(sysInfoBuf_));
    cacheInfo_.levelCache[2].size[2] = ((int64_t *)sysInfoBuf_)[0];

    for (size_t i = 0; i < maxCacheLevel; i++) {
      const auto cache = &cacheInfo_.levelCache[i];
      if (cache->size[1] || cache->size[2])
        lastDataCacheLevel_ = i + 1;
    }

    getSysInfo(hw_cacheconfig, sizeof(sysInfoBuf_));
    // L1
    cacheInfo_.levelCache[0].sharingCores[0] = ((uint64_t *)sysInfoBuf_)[1];
    cacheInfo_.levelCache[0].sharingCores[1] = ((uint64_t *)sysInfoBuf_)[1];
    // L2
    cacheInfo_.levelCache[1].sharingCores[2] = ((uint64_t *)sysInfoBuf_)[2];
    // L3
    cacheInfo_.levelCache[2].sharingCores[2] = ((uint64_t *)sysInfoBuf_)[3];
  }

  void setHwCap() {
    size_t val = 0;
    size_t len = sizeof(val);

    /* There are platforms with /sys not mounted. skip
     * handling HW caps for such platforms.
     */
    if (sysctlbyname(hw_opt_atomics, &val, &len, NULL, 0) != 0)
      type_ = 0;
    else
      type_ |= (val == 1) ? (Type)XBYAK_AARCH64_HWCAP_ATOMIC : 0;

    if (sysctlbyname(hw_opt_fp, &val, &len, NULL, 0) != 0)
      type_ = 0;
    else
      type_ |= (val == 1) ? (Type)XBYAK_AARCH64_HWCAP_FP : 0;

    if (sysctlbyname(hw_opt_neon, &val, &len, NULL, 0) != 0)
      type_ = 0;
    else
      type_ |= (val == 1) ? (Type)XBYAK_AARCH64_HWCAP_ADVSIMD : 0;
  }

  void setNumCores() {
    // ToDo: Distinguish physical and logical cores
    getSysInfo(hw_ncpu, sizeof(int32_t));
    numCores_[0] = numCores_[1] = ((int32_t *)sysInfoBuf_)[0];
  }
};

} // namespace util
} // namespace Xbyak_aarch64
