#pragma once
/*******************************************************************************
 * Copyright 2020-2021 FUJITSU LIMITED
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

#include <stdint.h>
#ifdef __linux__
#include <sys/prctl.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "xbyak_aarch64_err.h"

namespace Xbyak_aarch64 {
namespace util {

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

struct Type_id_aa64isar0_el1 {
  int resv0 : 4;
  int aes : 4;
  int sha1 : 4;
  int sha2 : 4;
  int crc32 : 4;
  int atomic : 4;
  int resv1 : 4;
  int rdm : 4;
  int resv2 : 12;
  int dp : 4;
  int resv3 : 16;
};

inline Type_id_aa64isar0_el1 get_id_aa64isar0_el1() {
  Type_id_aa64isar0_el1 x;
  asm __volatile__("mrs %0, id_aa64isar0_el1" : "=r"(x));
  return x;
}

struct Type_id_aa64pfr0_el1 {
  int el0 : 4;
  int el1 : 4;
  int el2 : 4;
  int el3 : 4;
  int fp : 4;
  int advsimd : 4;
  int gic : 4;
  int ras : 4;
  int sve : 4;
  int resv0 : 28;
};

inline Type_id_aa64pfr0_el1 get_id_aa64pfr0_el1() {
  Type_id_aa64pfr0_el1 x;
  asm __volatile__("mrs %0, id_aa64pfr0_el1" : "=r"(x));
  return x;
}

#ifdef __APPLE__
constexpr char hw_opt_atomics[] = "hw.optional.armv8_1_atomics";
constexpr char hw_opt_fp[] = "hw.optional.floatingpoint";
constexpr char hw_opt_neon[] = "hw.optional.neon";
#endif

/**
   CPU detection class
*/
class Cpu {
  uint64_t type_;
  sveLen_t sveLen_;

public:
  typedef uint64_t Type;

  static const Type tNONE = 0;
  static const Type tADVSIMD = 1 << 1;
  static const Type tFP = 1 << 2;
  static const Type tSVE = 1 << 3;
  static const Type tATOMIC = 1 << 4;

  static const uint64_t ZCR_EL1_LEN_SHIFT = 0;
  static const uint64_t ZCR_EL1_LEN_MASK = 0xf;

  Cpu() : type_(tNONE), sveLen_(SVE_NONE) {
#ifdef __linux__
    Type_id_aa64isar0_el1 isar0 = get_id_aa64isar0_el1();
    if (isar0.atomic == 2) {
      type_ |= tATOMIC;
    }

    Type_id_aa64pfr0_el1 pfr0 = get_id_aa64pfr0_el1();
    if (pfr0.fp == 1) {
      type_ |= tFP;
    }
    if (pfr0.advsimd == 1) {
      type_ |= tADVSIMD;
    }
    if (pfr0.sve == 1) {
      type_ |= tSVE;
      // svcntb(); if arm_sve.h is available
      sveLen_ = (sveLen_t)prctl(51); // PR_SVE_GET_VL
    }
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
  }

  Type getType() const { return type_; }
  bool has(Type type) const { return (type & type_) != 0; }
  uint64_t getSveLen() const { return sveLen_; }
  bool isAtomicSupported() const { return type_ & tATOMIC; }
};
} // namespace util
} // namespace Xbyak_aarch64
#endif
