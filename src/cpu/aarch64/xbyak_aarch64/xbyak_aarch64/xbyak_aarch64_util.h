#pragma once
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
#ifndef XBYAK_AARCH64_UTIL_H_
#define XBYAK_AARCH64_UTIL_H_

#include <dirent.h>
#include <regex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <unistd.h>

/* In old Linux such as Ubuntu 16.04, HWCAP_ATOMICS, HWCAP_FP, HWCAP_ASIMD
   can not be found in <bits/hwcap.h> which is included from <sys/auxv.h>.
   Xbyak_aarch64 uses <asm/hwcap.h> as an alternative.
 */
#ifndef HWCAP_FP
#include <asm/hwcap.h>
#endif

#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "xbyak_aarch64_err.h"

#define XBYAK_AARCH64_MIDR_EL1(I, V, A, P, R) ((I << 24) | (V << 20) | (A << 16) | (P << 4) | (R << 0))
#define XBYAK_AARCH64_PATH_NODES "/sys/devices/system/node/node"
#define XBYAK_AARCH64_PATH_CORES "/sys/devices/system/node/node0/cpu"
#define XBYAK_AARCH64_READ_SYSREG(var, ID) asm("mrs %0, " #ID : "=r"(var));

namespace Xbyak_aarch64 {
namespace util {
typedef uint64_t Type;

constexpr uint32_t maxNumberCacheLevel = 4;
constexpr uint32_t maxTopologyLevel = 2;
constexpr uint32_t max_path_len = 1024;

enum Arm64CpuTopologyLevel { SmtLevel = 1, CoreLevel = 2 };

enum sveLen_t {
  SVE_NONE = 0,
  SVE_128 = 16 * 1,
  SVE_256 = 16 * 2,
  SVE_384 = 16 * 3,
  SVE_512 = 16 * 4,
  SVE_640 = 16 * 5,
  SVE_768 = 16 * 6,
  SVE_896 = 16 * 7,
  SVE_1024 = 16 * 8,
  SVE_1152 = 16 * 9,
  SVE_1280 = 16 * 10,
  SVE_1408 = 16 * 11,
  SVE_1536 = 16 * 12,
  SVE_1664 = 16 * 13,
  SVE_1792 = 16 * 14,
  SVE_1920 = 16 * 15,
  SVE_2048 = 16 * 16,
};

struct implementer_t {
  uint32_t id;
  const char *implementer;
};

struct cacheInfo_t {
  uint64_t midr_el1;
  uint32_t dataCacheLevel;
  uint32_t highestInnerCacheLevel;
  uint32_t dataCacheSize[maxNumberCacheLevel];
};

#ifdef __APPLE__
constexpr char hw_opt_atomics[] = "hw.optional.armv8_1_atomics";
constexpr char hw_opt_fp[] = "hw.optional.floatingpoint";
constexpr char hw_opt_neon[] = "hw.optional.neon";
#endif

const struct implementer_t implementers[] = {{0x00, "Reserved for software use"},
                                             {0xC0, "Ampere Computing"},
                                             {0x41, "Arm Limited"},
                                             {0x42, "Broadcom Corporation"},
                                             {0x43, "Cavium Inc."},
                                             {0x44, "Digital Equipment Corporation"},
                                             {0x46, "Fujitsu Ltd."},
                                             {0x49, "Infineon Technologies AG"},
                                             {0x4D, "Motorola or Freescale Semiconductor Inc."},
                                             {0x4E, "NVIDIA Corporation"},
                                             {0x50, "Applied Micro Circuits Corporation"},
                                             {0x51, "Qualcomm Inc."},
                                             {0x56, "Marvell International Ltd."},
                                             {0x69, "Intel Corporation"}};

/**
   CPU detection class
*/
class Cpu {
  uint64_t type_;
  sveLen_t sveLen_;

private:
  const struct cacheInfo_t cacheInfoDict[2] = {
      {/* A64FX */ XBYAK_AARCH64_MIDR_EL1(0x46, 0x1, 0xf, 0x1, 0x0), 2, 1, {1024 * 64, 1024 * 1024 * 8 * 4, 0, 0}},
      {/* A64FX */ XBYAK_AARCH64_MIDR_EL1(0x46, 0x2, 0xf, 0x1, 0x0), 2, 1, {1024 * 64, 1024 * 1024 * 8 * 4, 0, 0}},
  };

  uint32_t coresSharingDataCache_[maxNumberCacheLevel];
  uint32_t dataCacheSize_[maxNumberCacheLevel];
  uint32_t dataCacheLevel_;
  uint64_t midr_el1_;
  uint32_t numCores_[maxTopologyLevel];

  void setCacheHierarchy();
  void setNumCores();
  void setSysRegVal();
  int getRegEx(char *buf, const char *path, const char *regex);
  int getFilePathMaxTailNumPlus1(const char *path);

public:
  static const Type tNONE = 0;
  static const Type tADVSIMD = 1 << 1;
  static const Type tFP = 1 << 2;
  static const Type tSVE = 1 << 3;
  static const Type tATOMIC = 1 << 4;

  Cpu();

  Type getType() const;
  bool has(Type type) const;
  uint64_t getSveLen() const;
  bool isAtomicSupported() const;
  const char *getImplementer() const;
  uint32_t getCoresSharingDataCache(uint32_t i) const;
  uint32_t getDataCacheLevels() const;
  uint32_t getDataCacheSize(uint32_t i) const;
  uint32_t getNumCores(Arm64CpuTopologyLevel level) const;
};

} // namespace util
} // namespace Xbyak_aarch64
#endif
