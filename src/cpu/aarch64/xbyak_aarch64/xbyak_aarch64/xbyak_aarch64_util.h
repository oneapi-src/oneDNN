#pragma once
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
#ifndef XBYAK_AARCH64_UTIL_H_
#define XBYAK_AARCH64_UTIL_H_

#if !(defined(__linux__) || defined(__APPLE__) || defined(_M_ARM64))
#error "Unsupported OS"
#endif

#include <stdint.h>

namespace Xbyak_aarch64 {
namespace util {
typedef uint64_t Type;

constexpr uint32_t maxCacheLevel = 7; // Specification of Armv9
constexpr uint32_t maxTopologyLevel = 2;
constexpr uint32_t max_path_len = 1024;

enum Arm64CpuTopologyLevel { SmtLevel = 1, CoreLevel = 2 };
enum Arm64CacheType { NoCache = 0, InstCacheOnly = 1, DataCacheOnly = 2, SeparateCache = 3, UnifiedCache = 4, OtherCache = 5 };
enum Arm64CacheLevel { L1 = 1, L2, L3, L4, L5, L6, L7 };
enum cacheType_t { ICache = 0, DCache = 1, UCache = 2 };

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

enum hwCap_t {
  XBYAK_AARCH64_HWCAP_NONE = 0,
  XBYAK_AARCH64_HWCAP_ADVSIMD = 1 << 1,
  XBYAK_AARCH64_HWCAP_FP = 1 << 2,
  XBYAK_AARCH64_HWCAP_SVE = 1 << 3,
  XBYAK_AARCH64_HWCAP_ATOMIC = 1 << 4,
  XBYAK_AARCH64_HWCAP_BF16 = 1 << 5,
};

struct implementer_t {
  uint32_t id;
  const char *implementer;
};

/* 2023.02.11
   cacheInfo_t does not need to be disclosed. */
#if defined(__GNUC__) || defined(__clang_version__)
struct __attribute__((deprecated)) cacheInfo_t {
#elif defined(_MSC_VER)
struct __declspec(deprecated) cacheInfo_t {
#endif
  uint64_t midr_el1;
  uint32_t dataCacheLevel;
  uint32_t highestInnerCacheLevel;
  uint32_t dataCacheSize[maxCacheLevel];
};

/**
   CPU detection class
*/
class CpuInfo;
class Cpu {
private:
  CpuInfo *info;

public:
  Cpu();

  void dumpCacheInfo() const;
  Arm64CacheType getCacheType(const Arm64CacheLevel i) const;
  uint32_t getCoresSharingDataCache(const Arm64CacheLevel i) const;
  /* 2023.02.11 */
#if defined(__GNUC__) || defined(__clang_version__)
  uint32_t __attribute__((deprecated("Please use getLastDataCacheLevel()"))) getDataCacheLevels() const;
#endif
  uint32_t getDataCacheSize(const Arm64CacheLevel i) const;
  const char *getImplementer() const;
  uint32_t getLastDataCacheLevel() const;
  uint32_t getNumCores(Arm64CpuTopologyLevel level) const;
  Type getType() const;
  uint64_t getSveLen() const;
  bool has(Type type) const;
  bool isAtomicSupported() const;
  bool isBf16Supported() const;
};

} // namespace util
} // namespace Xbyak_aarch64
#endif
