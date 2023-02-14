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
namespace Xbyak_aarch64 {
namespace util {

struct levelCacheInfo_t {
  Arm64CacheType type;
  /* I cache size, D cache size and Unified cache size.
     Example1: some cache level of a CPU core has separate caches of
     32KB I$ and 64KB D$, size[] = {1024 * 32, 1024 * 64, 0},
     sharingCores[] = 1, 1, 0}.
     Example2: some cache level of CPU cores has a 1MB unified cache,
     and it is shared by 12 CPU cores,
     size[] = {0, 0, 1024 * 1024}, sharingCores[] = {0, 0, 12}.
   */
  uint32_t size[3];         // I cache, D cache, Unified cache
  uint32_t sharingCores[3]; // I cache, D cache, Unified cache
};

struct cacheInfo_v2_t {
  uint64_t midr_el1; // used as table index
  levelCacheInfo_t levelCache[maxCacheLevel];
};

const struct implementer_t implementers[] = {
    {0x00, "Reserved for software use"},
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
    {0x69, "Intel Corporation"},
    {0xFE, "Apple Inc."}, // Xbyak_aarch64 original definition
    {0xFF, "Cannot identified"},
};

class CpuInfo {
protected:
  int numCores_[2] = {}; // [0]:SmtLevel, [1], CoreLevel
  cacheInfo_v2_t cacheInfo_;
  uint32_t lastDataCacheLevel_;
  Type type_ = 0;
  uint64_t sveLen_ = 0;
  const char *implementer_ = nullptr;

  void init();
  void setImplementer();
  void setLastDataCacheLevel();

public:
  CpuInfo() {}
  void dumpCacheInfo() const;
  int getCacheSize(cacheType_t type, uint32_t level) const;
  Arm64CacheType getCacheType(int level) const;
  int getCodeCacheSize(int level) const;
  int getCoresSharingDataCache(int level) const;
  int getDataCacheSize(int level) const;
  const char *getImplementer() const;
  int getLastDataCacheLevel() const;
  int getNumCores(Arm64CpuTopologyLevel level = CoreLevel) const;
  uint64_t getSveLen() const;
  Type getType() const;
  int getUnifiedCacheSize(int level) const;
  void put() const;
};
} // namespace util
} // namespace Xbyak_aarch64
