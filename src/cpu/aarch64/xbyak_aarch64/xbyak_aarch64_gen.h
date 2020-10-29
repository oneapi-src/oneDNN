/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#pragma once
/*******************************************************************************
 * Copyright 2019-2020 FUJITSU LIMITED
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

#include "xbyak_aarch64_adr.h"
#include "xbyak_aarch64_code_array.h"
#include "xbyak_aarch64_err.h"
#include "xbyak_aarch64_label.h"
#include "xbyak_aarch64_reg.h"

enum BarOpt {
  SY = 0xf,
  ST = 0xe,
  LD = 0xd,
  ISH = 0xb,
  ISHST = 0xa,
  ISHLD = 0x9,
  NSH = 0x7,
  NSHST = 0x6,
  NSHLD = 0x5,
  OSH = 0x3,
  OSHST = 0x2,
  OSHLD = 0x1
};

enum PStateField { SPSel, DAIFSet, DAIFClr, UAO, PAN, DIT };

enum Cond {
  EQ = 0x0,
  NE = 0x1,
  CS = 0x2,
  HS = 0x2,
  CC = 0x3,
  LO = 0x3,
  MI = 0x4,
  PL = 0x5,
  VS = 0x6,
  VC = 0x7,
  HI = 0x8,
  LS = 0x9,
  GE = 0xa,
  LT = 0xb,
  GT = 0xc,
  LE = 0xd,
  AL = 0xe,
  NV = 0xf
};

enum Prfop {
  PLDL1KEEP = (0x0 << 3) + (0x0 << 1) + 0x0,
  PLDL1STRM = (0x0 << 3) + (0x0 << 1) + 0x1,
  PLDL2KEEP = (0x0 << 3) + (0x1 << 1) + 0x0,
  PLDL2STRM = (0x0 << 3) + (0x1 << 1) + 0x1,
  PLDL3KEEP = (0x0 << 3) + (0x2 << 1) + 0x0,
  PLDL3STRM = (0x0 << 3) + (0x2 << 1) + 0x1,
  PLIL1KEEP = (0x1 << 3) + (0x0 << 1) + 0x0,
  PLIL1STRM = (0x1 << 3) + (0x0 << 1) + 0x1,
  PLIL2KEEP = (0x1 << 3) + (0x1 << 1) + 0x0,
  PLIL2STRM = (0x1 << 3) + (0x1 << 1) + 0x1,
  PLIL3KEEP = (0x1 << 3) + (0x2 << 1) + 0x0,
  PLIL3STRM = (0x1 << 3) + (0x2 << 1) + 0x1,
  PSTL1KEEP = (0x2 << 3) + (0x0 << 1) + 0x0,
  PSTL1STRM = (0x2 << 3) + (0x0 << 1) + 0x1,
  PSTL2KEEP = (0x2 << 3) + (0x1 << 1) + 0x0,
  PSTL2STRM = (0x2 << 3) + (0x1 << 1) + 0x1,
  PSTL3KEEP = (0x2 << 3) + (0x2 << 1) + 0x0,
  PSTL3STRM = (0x2 << 3) + (0x2 << 1) + 0x1
};

enum PrfopSve {
  PLDL1KEEP_SVE = 0x0,
  PLDL1STRM_SVE = 0x1,
  PLDL2KEEP_SVE = 0x2,
  PLDL2STRM_SVE = 0x3,
  PLDL3KEEP_SVE = 0x4,
  PLDL3STRM_SVE = 0x5,
  PSTL1KEEP_SVE = 0x8,
  PSTL1STRM_SVE = 0x9,
  PSTL2KEEP_SVE = 0xa,
  PSTL2STRM_SVE = 0xb,
  PSTL3KEEP_SVE = 0xc,
  PSTL3STRM_SVE = 0xd
};

enum Pattern {
  POW2 = 0x0,
  VL1 = 0x1,
  VL2 = 0x2,
  VL3 = 0x3,
  VL4 = 0x4,
  VL5 = 0x5,
  VL6 = 0x6,
  VL7 = 0x7,
  VL8 = 0x8,
  VL16 = 0x9,
  VL32 = 0xa,
  VL64 = 0xb,
  VL128 = 0xc,
  VL256 = 0xd,
  MUL4 = 0x1d,
  MUL3 = 0x1e,
  ALL = 0x1f
};

enum IcOp {
  ALLUIS = inner::genSysInstOp(0, 7, 1, 0), // op1=0, CRn=7, CRm=1, op2=0
  ALLU = inner::genSysInstOp(0, 7, 5, 0),   // op1=0, CRn=7, CRm=5, op2=0
  VAU = inner::genSysInstOp(3, 7, 5, 0)     // op1=3, CRn=7, CRm=5, op2=1
};

enum DcOp {
  IVAC = inner::genSysInstOp(0, 7, 0x6, 1),  // op1=0, CRn=7, CRm=0x6, op2=1
  ISW = inner::genSysInstOp(0, 7, 0x6, 2),   // op1=0, CRn=7, CRm=0x6, op2=2
  CSW = inner::genSysInstOp(0, 7, 0xA, 2),   // op1=0, CRn=7, CRm=0xA, op2=2
  CISW = inner::genSysInstOp(0, 7, 0xE, 2),  // op1=0, CRn=7, CRm=0xE, op2=2
  ZVA = inner::genSysInstOp(3, 7, 0x4, 1),   // op1=3, CRn=7, CRm=0x4, op2=1
  CVAC = inner::genSysInstOp(3, 7, 0xA, 1),  // op1=3, CRn=7, CRm=0xA, op2=1
  CVAU = inner::genSysInstOp(3, 7, 0xB, 1),  // op1=3, CRn=7, CRm=0xB, op2=1
  CIVAC = inner::genSysInstOp(3, 7, 0xE, 1), // op1=3, CRn=7, CRm=0xE, op2=1
  CVAP = inner::genSysInstOp(3, 7, 0xC, 1)   // op1=3, CRn=7, CRm=0xC, op2=1
};

enum AtOp {
  S1E1R = inner::genSysInstOp(0, 7, 0x8, 0),  // op1=0, CRn=7, CRm=0x8, op2=0
  S1E1W = inner::genSysInstOp(0, 7, 0x8, 1),  // op1=0, CRn=7, CRm=0x8, op2=1
  S1E0R = inner::genSysInstOp(0, 7, 0x8, 2),  // op1=0, CRn=7, CRm=0x8, op2=2
  S1E0W = inner::genSysInstOp(0, 7, 0x8, 3),  // op1=0, CRn=7, CRm=0x8, op2=3
  S1E2R = inner::genSysInstOp(4, 7, 0x8, 0),  // op1=4, CRn=7, CRm=0x8, op2=0
  S1E2W = inner::genSysInstOp(4, 7, 0x8, 1),  // op1=4, CRn=7, CRm=0x8, op2=1
  S12E1R = inner::genSysInstOp(4, 7, 0x8, 4), // op1=4, CRn=7, CRm=0x8, op2=4
  S12E1W = inner::genSysInstOp(4, 7, 0x8, 5), // op1=4, CRn=7, CRm=0x8, op2=5
  S12E0R = inner::genSysInstOp(4, 7, 0x8, 6), // op1=4, CRn=7, CRm=0x8, op2=6
  S12E0W = inner::genSysInstOp(4, 7, 0x8, 7), // op1=4, CRn=7, CRm=0x8, op2=7
  S1E3R = inner::genSysInstOp(6, 7, 0x8, 0),  // op1=6, CRn=7, CRm=0x8, op2=0
  S1E3W = inner::genSysInstOp(6, 7, 0x8, 1),  // op1=6, CRn=7, CRm=0x8, op2=1
  S1E1RP = inner::genSysInstOp(0, 7, 0x9, 0), // op1=0, CRn=7, CRm=0x9, op2=0
  S1E1WP = inner::genSysInstOp(0, 7, 0x9, 1), // op1=0, CRn=7, CRm=0x9, op2=1
};

enum TlbiOp {
  VMALLE1IS = inner::genSysInstOp(0, 7, 3, 0),  // op1=0, CRn=7, CRm=0x3, op2=0
  VAE1IS = inner::genSysInstOp(0, 7, 3, 1),     // op1=0, CRn=7, CRm=0x3, op2=1
  ASIDE1IS = inner::genSysInstOp(0, 7, 3, 2),   // op1=0, CRn=7, CRm=0x3, op2=2
  VAAE1IS = inner::genSysInstOp(0, 7, 3, 3),    // op1=0, CRn=7, CRm=0x3, op2=3
  VALE1IS = inner::genSysInstOp(0, 7, 3, 5),    // op1=0, CRn=7, CRm=0x3, op2=5
  VAALE1IS = inner::genSysInstOp(0, 7, 3, 7),   // op1=0, CRn=7, CRm=0x3, op2=7
  VMALLE1 = inner::genSysInstOp(0, 7, 7, 0),    // op1=0, CRn=7, CRm=0x7, op2=0
  VAE1 = inner::genSysInstOp(0, 7, 7, 1),       // op1=0, CRn=7, CRm=0x7, op2=1
  ASIDE1 = inner::genSysInstOp(0, 7, 7, 2),     // op1=0, CRn=7, CRm=0x7, op2=2
  VAAE1 = inner::genSysInstOp(0, 7, 7, 3),      // op1=0, CRn=7, CRm=0x7, op2=3
  VALE1 = inner::genSysInstOp(0, 7, 7, 5),      // op1=0, CRn=7, CRm=0x7, op2=5
  VAALE1 = inner::genSysInstOp(0, 7, 7, 7),     // op1=0, CRn=7, CRm=0x7, op2=7
  IPAS2E1IS = inner::genSysInstOp(4, 7, 0, 1),  // op1=4, CRn=7, CRm=0x0, op2=1
  IPAS2LE1IS = inner::genSysInstOp(4, 7, 0, 5), // op1=4, CRn=7, CRm=0x0, op2=5
  ALLE2IS = inner::genSysInstOp(4, 7, 3, 0),    // op1=4, CRn=7, CRm=0x3, op2=0
  VAE2IS = inner::genSysInstOp(4, 7, 3, 1),     // op1=4, CRn=7, CRm=0x3, op2=1
  ALLE1IS = inner::genSysInstOp(4, 7, 3, 4),    // op1=4, CRn=7, CRm=0x3, op2=4
  VALE2IS = inner::genSysInstOp(4, 7, 3, 5),    // op1=4, CRn=7, CRm=0x3, op2=5
  VMALLS12E1IS =
      inner::genSysInstOp(4, 7, 3, 6),           // op1=4, CRn=7, CRm=0x3, op2=6
  IPAS2E1 = inner::genSysInstOp(4, 7, 4, 1),     // op1=4, CRn=7, CRm=0x4, op2=1
  IPAS2LE1 = inner::genSysInstOp(4, 7, 4, 5),    // op1=4, CRn=7, CRm=0x4, op2=5
  ALLE2 = inner::genSysInstOp(4, 7, 7, 0),       // op1=4, CRn=7, CRm=0x7, op2=0
  VAE2 = inner::genSysInstOp(4, 7, 7, 1),        // op1=4, CRn=7, CRm=0x7, op2=1
  ALLE1 = inner::genSysInstOp(4, 7, 7, 4),       // op1=4, CRn=7, CRm=0x7, op2=4
  VALE2 = inner::genSysInstOp(4, 7, 7, 5),       // op1=4, CRn=7, CRm=0x7, op2=5
  VMALLS12E1 = inner::genSysInstOp(4, 7, 7, 6),  // op1=4, CRn=7, CRm=0x7, op2=6
  ALLE3IS = inner::genSysInstOp(6, 7, 3, 0),     // op1=6, CRn=7, CRm=0x3, op2=0
  VAE3IS = inner::genSysInstOp(6, 7, 3, 1),      // op1=6, CRn=7, CRm=0x3, op2=1
  VALE3IS = inner::genSysInstOp(6, 7, 3, 5),     // op1=6, CRn=7, CRm=0x3, op2=5
  ALLE3 = inner::genSysInstOp(6, 7, 7, 0),       // op1=6, CRn=7, CRm=0x7, op2=0
  VAE3 = inner::genSysInstOp(6, 7, 7, 1),        // op1=6, CRn=7, CRm=0x7, op2=1
  VALE3 = inner::genSysInstOp(6, 7, 7, 5),       // op1=6, CRn=7, CRm=0x7, op2=5
  VMALLE1OS = inner::genSysInstOp(0, 7, 1, 0),   // op1=0, CRn=7, CRm=0x1, op2=0
  VAE1OS = inner::genSysInstOp(0, 7, 1, 1),      // op1=0, CRn=7, CRm=0x1, op2=1
  ASIDE1OS = inner::genSysInstOp(0, 7, 1, 2),    // op1=0, CRn=7, CRm=0x1, op2=2
  VAAE1OS = inner::genSysInstOp(0, 7, 1, 3),     // op1=0, CRn=7, CRm=0x1, op2=3
  VALE1OS = inner::genSysInstOp(0, 7, 1, 5),     // op1=0, CRn=7, CRm=0x1, op2=5
  VAALE1OS = inner::genSysInstOp(0, 7, 1, 7),    // op1=0, CRn=7, CRm=0x1, op2=7
  RVAE1IS = inner::genSysInstOp(0, 7, 2, 1),     // op1=0, CRn=7, CRm=0x2, op2=1
  RVAAE1IS = inner::genSysInstOp(0, 7, 2, 3),    // op1=0, CRn=7, CRm=0x2, op2=3
  RVALE1IS = inner::genSysInstOp(0, 7, 2, 5),    // op1=0, CRn=7, CRm=0x2, op2=5
  RVAALE1IS = inner::genSysInstOp(0, 7, 2, 7),   // op1=0, CRn=7, CRm=0x2, op2=7
  RVAE1OS = inner::genSysInstOp(0, 7, 5, 1),     // op1=0, CRn=7, CRm=0x5, op2=1
  RVAAE1OS = inner::genSysInstOp(0, 7, 5, 3),    // op1=0, CRn=7, CRm=0x5, op2=3
  RVALE1OS = inner::genSysInstOp(0, 7, 5, 5),    // op1=0, CRn=7, CRm=0x5, op2=5
  RVAALE1OS = inner::genSysInstOp(0, 7, 5, 7),   // op1=0, CRn=7, CRm=0x5, op2=7
  RVAE1 = inner::genSysInstOp(0, 7, 6, 1),       // op1=0, CRn=7, CRm=0x6, op2=1
  RVAAE1 = inner::genSysInstOp(0, 7, 6, 3),      // op1=0, CRn=7, CRm=0x6, op2=3
  RVALE1 = inner::genSysInstOp(0, 7, 6, 5),      // op1=0, CRn=7, CRm=0x6, op2=5
  RVAALE1 = inner::genSysInstOp(0, 7, 6, 7),     // op1=0, CRn=7, CRm=0x6, op2=7
  RIPAS2E1IS = inner::genSysInstOp(4, 7, 0, 2),  // op1=4, CRn=7, CRm=0x0, op2=2
  RIPAS2LE1IS = inner::genSysInstOp(4, 7, 0, 6), // op1=4, CRn=7, CRm=0x0, op2=6
  ALLE2OS = inner::genSysInstOp(4, 7, 1, 0),     // op1=4, CRn=7, CRm=0x1, op2=0
  VAE2OS = inner::genSysInstOp(4, 7, 1, 1),      // op1=4, CRn=7, CRm=0x1, op2=1
  ALLE1OS = inner::genSysInstOp(4, 7, 1, 4),     // op1=4, CRn=7, CRm=0x1, op2=4
  VALE2OS = inner::genSysInstOp(4, 7, 1, 5),     // op1=4, CRn=7, CRm=0x1, op2=5
  VMALLS12E1OS =
      inner::genSysInstOp(4, 7, 1, 6),           // op1=4, CRn=7, CRm=0x1, op2=6
  RVAE2IS = inner::genSysInstOp(4, 7, 2, 1),     // op1=4, CRn=7, CRm=0x2, op2=1
  RVALE2IS = inner::genSysInstOp(4, 7, 2, 5),    // op1=4, CRn=7, CRm=0x2, op2=5
  IPAS2E1OS = inner::genSysInstOp(4, 7, 4, 0),   // op1=4, CRn=7, CRm=0x4, op2=0
  RIPAS2E1 = inner::genSysInstOp(4, 7, 4, 2),    // op1=4, CRn=7, CRm=0x4, op2=2
  RIPAS2E1OS = inner::genSysInstOp(4, 7, 4, 3),  // op1=4, CRn=7, CRm=0x4, op2=3
  IPAS2LE1OS = inner::genSysInstOp(4, 7, 4, 4),  // op1=4, CRn=7, CRm=0x4, op2=4
  RIPAS2LE1 = inner::genSysInstOp(4, 7, 4, 6),   // op1=4, CRn=7, CRm=0x4, op2=6
  RIPAS2LE1OS = inner::genSysInstOp(4, 7, 4, 7), // op1=4, CRn=7, CRm=0x4, op2=7
  RVAE2OS = inner::genSysInstOp(4, 7, 5, 1),     // op1=4, CRn=7, CRm=0x5, op2=1
  RVALE2OS = inner::genSysInstOp(4, 7, 5, 5),    // op1=4, CRn=7, CRm=0x5, op2=5
  RVAE2 = inner::genSysInstOp(4, 7, 6, 1),       // op1=4, CRn=7, CRm=0x6, op2=1
  RVALE2 = inner::genSysInstOp(4, 7, 6, 5),      // op1=4, CRn=7, CRm=0x6, op2=5
  ALLE3OS = inner::genSysInstOp(6, 7, 1, 0),     // op1=6, CRn=7, CRm=0x1, op2=0
  VAE3OS = inner::genSysInstOp(6, 7, 1, 1),      // op1=6, CRn=7, CRm=0x1, op2=1
  VALE3OS = inner::genSysInstOp(6, 7, 1, 5),     // op1=6, CRn=7, CRm=0x1, op2=5
  RVAE3IS = inner::genSysInstOp(6, 7, 2, 1),     // op1=6, CRn=7, CRm=0x2, op2=1
  RVALE3IS = inner::genSysInstOp(6, 7, 2, 5),    // op1=6, CRn=7, CRm=0x1, op2=5
  RVAE3OS = inner::genSysInstOp(6, 7, 5, 1),     // op1=6, CRn=7, CRm=0x5, op2=1
  RVALE3OS = inner::genSysInstOp(6, 7, 5, 5),    // op1=6, CRn=7, CRm=0x5, op2=5
  RVAE3 = inner::genSysInstOp(6, 7, 6, 1),       // op1=6, CRn=7, CRm=0x6, op2=1
  RVALE3 = inner::genSysInstOp(6, 7, 6, 5)       // op1=6, CRn=7, CRm=0x6, op2=5
};

/////////////////////////////////////////////////////////////
//            encoding helper class
/////////////////////////////////////////////////////////////
class CodeGenUtil {
public:
  /////////////// bit operation ////////////////////
  inline uint64_t lsb(uint64_t v) { return v & 0x1; }

  inline uint64_t msb(uint64_t v, uint32_t size) {
    uint32_t shift = (size == 0) ? 0 : size - 1;
    return (v >> shift) & 0x1;
  }

  inline uint32_t field(uint64_t v, uint32_t mpos, uint32_t lpos) {
    return static_cast<uint32_t>((v >> lpos) & ones(mpos - lpos + 1));
  }

  inline uint64_t ones(uint32_t size) {
    return (size == 64) ? 0xffffffffffffffff : ((uint64_t)1 << size) - 1;
  }

  inline uint64_t rrotate(uint64_t v, uint32_t size, uint32_t num) {
    uint32_t shift = (size == 0) ? 0 : (num % size);
    v &= ones(size);
    return (v >> shift) | ((v & ones(shift)) << (size - shift));
  }

  inline uint64_t lrotate(uint64_t v, uint32_t size, uint32_t num) {
    uint32_t shift = (size == 0) ? 0 : (num % size);
    v &= ones(size);
    return ((v << shift) | ((v >> (size - shift)))) & ones(size);
  }

  inline uint64_t replicate(uint64_t v, uint32_t esize, uint32_t size) {
    uint64_t result = 0;
    for (uint32_t i = 0; i < 64 / esize; ++i) {
      result |= v << (esize * i);
    }
    return result & ones(size);
  }

  /////////////// ARMv8/SVE psuedo code function ////////////////
  bool checkPtn(uint64_t v, uint32_t esize, uint32_t size) {
    std::vector<uint64_t> ptns;
    uint32_t max_num = size / esize;
    for (uint32_t i = 0; i < max_num; ++i) {
      ptns.push_back((v >> (esize * i)) & ones(esize));
    }
    return std::all_of(ptns.begin(), ptns.end(),
                       [&ptns](uint64_t x) { return x == ptns[0]; });
  }

  uint32_t getPtnSize(uint64_t v, uint32_t size) {
    uint32_t esize;
    for (esize = 2; esize <= size; esize <<= 1) {
      if (checkPtn(v, esize, size))
        break;
    }
    return esize;
  }

  uint32_t getPtnRotateNum(uint64_t ptn, uint32_t ptn_size) {
    assert(ptn != 0 && (ptn & ones(ptn_size)) != ones(ptn_size));
    uint32_t num;
    for (num = 0; msb(ptn, ptn_size) || !lsb(ptn); ++num) {
      ptn = lrotate(ptn, ptn_size, 1);
    }
    return num;
  }

  uint32_t countOneBit(uint64_t v, uint32_t size) {
    uint64_t num = 0;
    for (uint32_t i = 0; i < size; ++i) {
      num += lsb(v);
      v >>= 1;
    };
    return static_cast<uint32_t>(num);
  }

  uint32_t countSeqOneBit(uint64_t v, uint32_t size) {
    uint32_t num;
    for (num = 0; num < size && lsb(v); ++num) {
      v >>= 1;
    };
    return num;
  }

  uint32_t compactImm(double imm, uint32_t size) {
    uint32_t sign = (imm < 0) ? 1 : 0;

    imm = std::abs(imm);
    int32_t max_digit = static_cast<int32_t>(std::floor(std::log2(imm)));

    int32_t n = (size == 16) ? 7 : (size == 32) ? 10 : 13;
    int32_t exp = (max_digit - 1) + (1 << n);

    imm -= pow(2, max_digit);
    uint32_t frac = 0;
    for (int i = 0; i < 4; ++i) {
      if (pow(2, max_digit - 1 - i) <= imm) {
        frac |= 1 << (3 - i);
        imm -= pow(2, max_digit - 1 - i);
      }
    }
    uint32_t imm8 = concat({F(sign, 7), F(field(~exp, n, n), 6),
                            F(field(exp, 1, 0), 4), F(frac, 0)});
    return imm8;
  }

  uint32_t compactImm(uint64_t imm) {
    uint32_t imm8 = 0;
    for (uint32_t i = 0; i < 64; i += 8) {
      uint32_t bit = (imm >> i) & 0x1;
      imm8 |= bit << (i >> 3);
    }
    return imm8;
  }

  bool isCompact(uint64_t imm, uint32_t imm8) {
    bool result = true;
    for (uint32_t i = 0; i < 64; ++i) {
      uint32_t bit = (imm >> i) & 0x1;
      uint32_t rbit = (imm8 >> (i >> 3)) & 0x1;
      result &= (bit == rbit);
    }
    return result;
  }

  uint64_t genMoveMaskPrefferd(uint64_t imm) {
    bool chk_result = true;
    if (field(imm, 7, 0) != 0) {
      if (field(imm, 63, 7) == 0 || field(imm, 63, 7) == ones(57))
        chk_result = false;
      if ((field(imm, 63, 32) == field(imm, 31, 0)) &&
          (field(imm, 31, 7) == 0 || field(imm, 31, 7) == ones(25)))
        chk_result = false;
      if ((field(imm, 63, 32) == field(imm, 31, 0)) &&
          (field(imm, 31, 16) == field(imm, 15, 0)) &&
          (field(imm, 15, 7) == 0 || field(imm, 15, 7) == ones(9)))
        chk_result = false;
      if ((field(imm, 63, 32) == field(imm, 31, 0)) &&
          (field(imm, 31, 16) == field(imm, 15, 0)) &&
          (field(imm, 15, 8) == field(imm, 7, 0)))
        chk_result = false;
    } else {
      if (field(imm, 63, 15) == 0 || field(imm, 63, 15) == ones(49))
        chk_result = false;
      if ((field(imm, 63, 32) == field(imm, 31, 0)) &&
          (field(imm, 31, 7) == 0 || field(imm, 31, 7) == ones(25)))
        chk_result = false;
      if ((field(imm, 63, 32) == field(imm, 31, 0)) &&
          (field(imm, 31, 16) == field(imm, 15, 0)))
        chk_result = false;
    }
    return (chk_result) ? imm : 0;
  }

  Cond invert(Cond cond) {
    uint32_t inv_val = (uint32_t)cond ^ 1;
    return (Cond)(inv_val & ones(4));
  }

  uint32_t genHw(uint64_t imm, uint32_t size) {
    if (imm == 0)
      return 0;

    uint32_t hw = 0;
    uint32_t times = (size == 32) ? 1 : 3;
    for (uint32_t i = 0; i < times; ++i) {
      if (field(imm, 15, 0) != 0)
        break;
      ++hw;
      imm >>= 16;
    }
    return hw;
  }

  /////////////// ARM8/SVE encoding helper function ////////////////
  constexpr const uint32_t F(uint32_t val, uint32_t pos) const {
    return val << pos;
  }

  uint32_t concat(const std::initializer_list<uint32_t> list) {
    uint32_t result = 0;
    for (auto f : list) {
      result |= f;
    }
    return result;
  }

  uint32_t genSf(const RReg &Reg) { return (Reg.getBit() == 64) ? 1 : 0; }

  uint32_t genQ(const VRegVec &Reg) {
    return (Reg.getBit() * Reg.getLane() == 128) ? 1 : 0;
  }

  uint32_t genQ(const VRegElem &Reg) {
    uint32_t pos = 0;
    switch (Reg.getBit()) {
    case 8:
      pos = 3;
      break;
    case 16:
      pos = 2;
      break;
    case 32:
      pos = 1;
      break;
    case 64:
      pos = 0;
      break;
    default:
      pos = 0;
    }
    return field(Reg.getElemIdx(), pos, pos);
  }

  uint32_t genSize(const Reg &Reg) {
    uint32_t size = 0;
    switch (Reg.getBit()) {
    case 8:
      size = 0;
      break;
    case 16:
      size = 1;
      break;
    case 32:
      size = 2;
      break;
    case 64:
      size = 3;
      break;
    default:
      size = 0;
    }
    return size;
  }

  uint32_t genSizeEnc(const VRegElem &Reg) {
    uint32_t size = 0;
    switch (Reg.getBit()) {
    case 8:
      size = field(Reg.getElemIdx(), 1, 0);
      break;
    case 16:
      size = field(Reg.getElemIdx(), 0, 0) << 1;
      break;
    case 32:
      size = 0;
      break;
    case 64:
      size = 1;
      break;
    default:
      size = 0;
    }
    return size;
  }

  uint32_t genSize(uint32_t dtype) {
    uint32_t size = (dtype == 0xf)
                        ? 3
                        : (dtype == 0x4 || dtype == 0xa || dtype == 0xb)
                              ? 2
                              : (5 <= dtype && dtype <= 9) ? 1 : 0;
    return size;
  }

  uint32_t genS(const VRegElem &Reg) {
    uint32_t s = 0;
    switch (Reg.getBit()) {
    case 8:
      s = field(Reg.getElemIdx(), 2, 2);
      break;
    case 16:
      s = field(Reg.getElemIdx(), 1, 1);
      break;
    case 32:
      s = field(Reg.getElemIdx(), 0, 0);
      break;
    case 64:
      s = 0;
      break;
    default:
      s = 0;
    }
    return s;
  }
};

class CodeGenerator : public CodeGenUtil, public CodeArray {
  struct CodeInfo {
    size_t code_idx;
    std::string file;
    size_t line;
    std::string func;

    void set(size_t idx, const std::string &file, size_t line,
             const std::string &func) {
      this->code_idx = idx;
      this->file = file;
      this->line = line;
      this->func = func;
    }

    std::string header() { return "index:   mnemonic location of define\n"; }

    std::string str() {
      std::stringstream ss("");
      ss << std::setw(5) << code_idx << ": " << std::setw(10) << std::right
         << func << " (" << file << ":" << line << ")\n";
      return ss.str();
    }
  };

  CodeInfo cinfo_;
  std::deque<CodeInfo> codeInfoHist_;

  LabelManager labelMgr_;

  // set infomation of instruction code
  void setCodeInfo(const std::string &file, size_t line,
                   const std::string &func) {
    cinfo_.set(getSize(), file, line, func);
    updateCodeInfoHist();
  }

  // update history of instruction code
  void updateCodeInfoHist() {
    codeInfoHist_.push_back(cinfo_);
    if (codeInfoHist_.size() > 5) {
      codeInfoHist_.pop_front();
    }
  }

  // code history to string
  std::string genCodeHistStr() {
    std::string str = cinfo_.header();
    for (auto info : codeInfoHist_) {
      str += info.str();
    }
    str.replace(str.rfind("\n"), 1, " <---- Error\n");
    return str;
  }

  // ################### check function #################
  // check val (list)
  template <typename T>
  bool chkVal(T val, const std::initializer_list<T> &list) {
    return std::any_of(list.begin(), list.end(), [=](T x) { return x == val; });
  }

  // check val (range)
  template <typename T> bool chkVal(T val, T min, T max) {
    return (min <= val && val <= max);
  }

  // check val (condtional func)
  template <typename T> bool chkVal(T val, const std::function<bool(T)> &func) {
    return func(val);
  }

  // verify (include range)
  void verifyIncRange(uint64_t val, uint64_t min, uint64_t max, int err_type,
                      bool to_i = false) {
    if (to_i && !chkVal((int64_t)val, (int64_t)min, (int64_t)max)) {
      throw Error(err_type, genErrMsg());
    } else if (!to_i && !chkVal(val, min, max)) {
      throw Error(err_type, genErrMsg());
    }
  }

  // verify (not include range)
  void verifyNotIncRange(uint64_t val, uint64_t min, uint64_t max, int err_type,
                         bool to_i = false) {
    if (to_i && chkVal((uint64_t)val, (uint64_t)min, (uint64_t)max)) {
      throw Error(err_type, genErrMsg());
    } else if (!to_i && chkVal(val, min, max)) {
      throw Error(err_type, genErrMsg());
    }
  }

  // verify (include list)
  void verifyIncList(uint64_t val, const std::initializer_list<uint64_t> &list,
                     int err_type) {
    if (!chkVal(val, list)) {
      throw Error(err_type, genErrMsg());
    }
  }

  // verify (not include list)
  void verifyNotIncList(uint64_t val,
                        const std::initializer_list<uint64_t> &list,
                        int err_type) {
    if (chkVal(val, list)) {
      throw Error(err_type, genErrMsg());
    }
  }

  // verify (conditional function)
  void verifyCond(uint64_t val, const std::function<bool(uint64_t)> &func,
                  int err_type) {
    if (!chkVal(val, func)) {
      throw Error(err_type, genErrMsg());
    }
  }

  // verify (conditional function)
  void verifyNotCond(uint64_t val, const std::function<bool(uint64_t)> &func,
                     int err_type) {
    if (chkVal(val, func)) {
      throw Error(err_type, genErrMsg());
    }
  }

  // optional error message
  std::string genErrMsg() {
    std::string msg = "-----------------------------------------\n";
    msg += genCodeHistStr();
    return msg;
  }

  // ############### encoding helper function #############
  // generate encoded imm
  uint32_t genNImmrImms(uint64_t imm, uint32_t size) {
    // check imm
    if (imm == 0 || imm == ones(size)) {
      throw Error(ERR_ILLEGAL_IMM_VALUE, genErrMsg());
    }

    auto ptn_size = getPtnSize(imm, size);
    auto ptn = imm & ones(ptn_size);
    auto rotate_num = getPtnRotateNum(ptn, ptn_size);
    auto rotate_ptn = lrotate(ptn, ptn_size, rotate_num);
    auto one_bit_num = countOneBit(rotate_ptn, ptn_size);
    auto seq_one_bit_num = countSeqOneBit(rotate_ptn, ptn_size);

    // check ptn
    if (one_bit_num != seq_one_bit_num) {
      throw Error(ERR_ILLEGAL_IMM_VALUE, genErrMsg());
    }

    uint32_t N = (ptn_size > 32) ? 1 : 0;
    uint32_t immr = rotate_num;
    uint32_t imms = static_cast<uint32_t>(
        (ones(6) & ~ones(static_cast<uint32_t>(std::log2(ptn_size)) + 1)) |
        (one_bit_num - 1));
    return (N << 12) | (immr << 6) | imms;
  }

  // generate relative address for label offset
  uint64_t genLabelOffset(const Label &label, const JmpLabel &jmpL) {
    size_t offset = 0;
    int64_t labelOffset = 0;
    if (labelMgr_.getOffset(&offset, label)) {
      labelOffset = (offset - getSize()) * CSIZE;
    } else {
      labelMgr_.addUndefinedLabel(label, jmpL);
    }
    return labelOffset;
  }

  // ################## encoding function ##################
  // PC-rel. addressing
  uint32_t PCrelAddrEnc(uint32_t op, const XReg &rd, int64_t labelOffset) {
    int32_t imm =
        static_cast<uint32_t>((op == 1) ? labelOffset >> 12 : labelOffset);
    uint32_t immlo = field(imm, 1, 0);
    uint32_t immhi = field(imm, 20, 2);
    verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(imm, -1 * (1 << 20), ones(20), ERR_ILLEGAL_IMM_RANGE, true);
    return concat(
        {F(op, 31), F(immlo, 29), F(0x10, 24), F(immhi, 5), F(rd.getIdx(), 0)});
  }

  void PCrelAddr(uint32_t op, const XReg &rd, const Label &label) {
    auto encFunc = [&, op, rd](int64_t labelOffset) {
      return PCrelAddrEnc(op, rd, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, getSize());
    uint32_t code = PCrelAddrEnc(op, rd, genLabelOffset(label, jmpL));
    dw(code);
  }

  void PCrelAddr(uint32_t op, const XReg &rd, int64_t label) {
    uint32_t code = PCrelAddrEnc(op, rd, label);
    dw(code);
  }

  // Add/subtract (immediate)
  void AddSubImm(uint32_t op, uint32_t S, const RReg &rd, const RReg &rn,
                 uint32_t imm, uint32_t sh) {
    uint32_t sf = genSf(rd);
    uint32_t imm12 = imm & ones(12);
    uint32_t sh_f = (sh == 12) ? 1 : 0;

    verifyIncRange(imm, 0, ones(12), ERR_ILLEGAL_IMM_RANGE);
    verifyIncList(sh, {0, 12}, ERR_ILLEGAL_CONST_VALUE);

    uint32_t code =
        concat({F(sf, 31), F(op, 30), F(S, 29), F(0x11, 24), F(sh_f, 22),
                F(imm12, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Logical (immediate)
  void LogicalImm(uint32_t opc, const RReg &rd, const RReg &rn, uint64_t imm,
                  bool alias = false) {
    uint32_t sf = genSf(rd);
    uint32_t n_immr_imms = genNImmrImms(imm, rd.getBit());

    if (!alias && opc == 3)
      verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    if (!alias && opc == 1)
      verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(sf, 31), F(opc, 29), F(0x24, 23), F(n_immr_imms, 10),
                F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Move wide(immediate)
  void MvWideImm(uint32_t opc, const RReg &rd, uint32_t imm, uint32_t sh) {
    uint32_t sf = genSf(rd);
    uint32_t hw = field(sh, 5, 4);
    uint32_t imm16 = imm & 0xffff;

    if (sf == 0)
      verifyIncList(sh, {0, 16}, ERR_ILLEGAL_CONST_VALUE);
    else
      verifyIncList(sh, {0, 16, 32, 48}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(imm, 0, ones(16), ERR_ILLEGAL_IMM_RANGE);
    verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(sf, 31), F(opc, 29), F(0x25, 23), F(hw, 21),
                            F(imm16, 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Move (immediate) alias of ORR,MOVN,MOVZ
  void MvImm(const RReg &rd, uint64_t imm) {
    uint32_t rd_bit = rd.getBit();
    uint32_t hw = 0;
    uint32_t inv_hw = 0;
    uint32_t validField[4] = {0};
    uint32_t imm16 = 0;
    uint32_t inv_imm16 = 0;
    uint32_t fieldCount = 0;
    uint32_t invFieldCount = 0;

    if (imm == 0) {
      MvWideImm(2, rd, 0, 0);
      return;
    }

    if ((rd_bit == 64 && imm == ~uint64_t(0)) ||
        (rd_bit == 32 &&
         ((imm & uint64_t(0xffffffff)) == uint64_t(0xffffffff)))) {
      MvWideImm(0, rd, 0, 0);
      return;
    }

    /***** MOVZ *****/
    /* Count how many valid 16-bit field exists. */
    for (uint32_t i = 0; i < rd_bit / 16; ++i) {
      if (field(imm, 15 + i * 16, i * 16)) {
        validField[i] = 1;
        ++fieldCount;
        hw = i;
        imm16 = field(imm, 15 + 16 * i, 16 * i);
      }
    }
    if (fieldCount < 2) {
      if (!(imm16 == 0 && hw != 0)) {
        /* alias of MOVZ
   which set 16-bit immediate, bit position is indicated by (hw * 4). */
        MvWideImm(2, rd, imm16, hw << 4);
        return;
      }
    }

    /***** MOVN *****/
    /* Count how many valid 16-bit field exists. */
    for (uint32_t i = 0; i < rd_bit / 16; ++i) {
      if (field(~imm, 15 + i * 16, i * 16)) {
        ++invFieldCount;
        inv_imm16 = field(~imm, 15 + 16 * i, 16 * i);
        inv_hw = i;
      }
    }
    if (invFieldCount == 1) {
      if ((!(inv_imm16 == 0 && inv_hw != 0) && inv_imm16 != ones(16) &&
           rd_bit == 32) ||
          (!(inv_imm16 == 0 && inv_hw != 0) && rd_bit == 64)) {
        /* alias of MOVN
   which firstly, set 16-bit immediate, bit position is indicated by (hw
   * 4) then, result is inverted (NOT). */
        MvWideImm(0, rd, inv_imm16, inv_hw << 4);
        return;
      }
    }

    /***** ORR *****/
    auto ptn_size = getPtnSize(imm, rd_bit);
    auto ptn = imm & ones(ptn_size);
    auto rotate_num = getPtnRotateNum(ptn, ptn_size);
    auto rotate_ptn = lrotate(ptn, ptn_size, rotate_num);
    auto one_bit_num = countOneBit(rotate_ptn, ptn_size);
    auto seq_one_bit_num = countSeqOneBit(rotate_ptn, ptn_size);
    if (one_bit_num == seq_one_bit_num) {
      // alias of ORR
      LogicalImm(1, rd, RReg(31, rd_bit), imm, true);
      return;
    }

    /**** MOVZ followed by successive MOVK *****/
    bool isFirst = true;
    for (uint32_t i = 0; i < rd_bit / 16; ++i) {
      if (validField[i]) {
        if (isFirst) {
          MvWideImm(2, rd, field(imm, 15 + 16 * i, 16 * i), 16 * i);
          isFirst = false;
        } else {
          MvWideImm(3, rd, field(imm, 15 + 16 * i, 16 * i), 16 * i);
        }
      }
    }
  }

  // Bitfield
  void Bitfield(uint32_t opc, const RReg &rd, const RReg &rn, uint32_t immr,
                uint32_t imms, bool rn_chk = true) {
    uint32_t sf = genSf(rd);
    uint32_t N = sf;

    verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    if (rn_chk)
      verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(immr, 0, rd.getBit() - 1, ERR_ILLEGAL_IMM_RANGE);
    verifyIncRange(imms, 0, rd.getBit() - 1, ERR_ILLEGAL_IMM_RANGE);

    uint32_t code =
        concat({F(sf, 31), F(opc, 29), F(0x26, 23), F(N, 22), F(immr, 16),
                F(imms, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Extract
  void Extract(uint32_t op21, uint32_t o0, const RReg &rd, const RReg &rn,
               const RReg &rm, uint32_t imm) {
    uint32_t sf = genSf(rd);
    uint32_t N = sf;

    verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rm.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(imm, 0, rd.getBit() - 1, ERR_ILLEGAL_IMM_RANGE);

    uint32_t code = concat({F(sf, 31), F(op21, 29), F(0x27, 23), F(N, 22),
                            F(o0, 21), F(rm.getIdx(), 16), F(imm, 10),
                            F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Conditional branch (immediate)
  uint32_t CondBrImmEnc(uint32_t cond, int64_t labelOffset) {
    uint32_t imm19 = static_cast<uint32_t>((labelOffset >> 2) & ones(19));
    verifyIncRange(labelOffset, -1 * (1 << 20), ones(20), ERR_LABEL_IS_TOO_FAR,
                   true);
    return concat({F(0x2a, 25), F(imm19, 5), F(cond, 0)});
  }

  void CondBrImm(Cond cond, const Label &label) {
    auto encFunc = [&, cond](int64_t labelOffset) {
      return CondBrImmEnc(cond, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, getSize());
    uint32_t code = CondBrImmEnc(cond, genLabelOffset(label, jmpL));
    dw(code);
  }

  void CondBrImm(Cond cond, int64_t label) {
    uint32_t code = CondBrImmEnc(cond, label);
    dw(code);
  }

  // Exception generation
  void ExceptionGen(uint32_t opc, uint32_t op2, uint32_t LL, uint32_t imm) {
    uint32_t imm16 = imm & ones(16);
    verifyIncRange(imm, 0, ones(16), ERR_ILLEGAL_IMM_RANGE);
    uint32_t code =
        concat({F(0xd4, 24), F(opc, 21), F(imm16, 5), F(op2, 2), F(LL, 0)});
    dw(code);
  }

  // Hints
  void Hints(uint32_t CRm, uint32_t op2) {
    uint32_t code = concat({F(0xd5032, 12), F(CRm, 8), F(op2, 5), F(0x1f, 0)});
    dw(code);
  }

  void Hints(uint32_t imm) { Hints(field(imm, 6, 3), field(imm, 2, 0)); }

  // Barriers (option)
  void BarriersOpt(uint32_t op2, BarOpt opt, uint32_t rt) {
    if (op2 == 6)
      verifyIncList(opt, {SY}, ERR_ILLEGAL_BARRIER_OPT);
    uint32_t code = concat({F(0xd5033, 12), F(opt, 8), F(op2, 5), F(rt, 0)});
    dw(code);
  }

  // Barriers (no option)
  void BarriersNoOpt(uint32_t CRm, uint32_t op2, uint32_t rt) {
    verifyIncRange(CRm, 0, ones(4), ERR_ILLEGAL_IMM_RANGE);
    uint32_t code = concat({F(0xd5033, 12), F(CRm, 8), F(op2, 5), F(rt, 0)});
    dw(code);
  }

  // pstate
  void PState(PStateField psfield, uint32_t imm) {
    uint32_t CRm = imm & ones(4);
    uint32_t op1, op2;
    switch (psfield) {
    case SPSel:
      op1 = 0;
      op2 = 5;
      break;
    case DAIFSet:
      op1 = 3;
      op2 = 6;
      break;
    case DAIFClr:
      op1 = 3;
      op2 = 7;
      break;
    case UAO:
      op1 = 0;
      op2 = 3;
      break;
    case PAN:
      op1 = 0;
      op2 = 4;
      break;
    case DIT:
      op1 = 3;
      op2 = 2;
      break;
    default:
      op1 = 0;
      op2 = 0;
    }
    uint32_t code = concat({F(0xd5, 24), F(op1, 16), F(0x4, 12), F(CRm, 8),
                            F(op2, 5), F(0x1f, 0)});
    dw(code);
  }

  void PState(uint32_t op1, uint32_t CRm, uint32_t op2) {
    uint32_t code = concat({F(0xd5, 24), F(op1, 16), F(0x4, 12), F(CRm, 8),
                            F(op2, 5), F(0x1f, 0)});
    dw(code);
  }

  // Systtem instructions
  void SysInst(uint32_t L, uint32_t op1, uint32_t CRn, uint32_t CRm,
               uint32_t op2, const XReg &rt) {
    uint32_t code =
        concat({F(0xd5, 24), F(L, 21), F(1, 19), F(op1, 16), F(CRn, 12),
                F(CRm, 8), F(op2, 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // System register move
  void SysRegMove(uint32_t L, uint32_t op0, uint32_t op1, uint32_t CRn,
                  uint32_t CRm, uint32_t op2, const XReg &rt) {
    uint32_t code =
        concat({F(0xd5, 24), F(L, 21), F(1, 20), F(op0, 19), F(op1, 16),
                F(CRn, 12), F(CRm, 8), F(op2, 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // Unconditional branch
  void UncondBrNoReg(uint32_t opc, uint32_t op2, uint32_t op3, uint32_t rn,
                     uint32_t op4) {
    uint32_t code = concat(
        {F(0x6b, 25), F(opc, 21), F(op2, 16), F(op3, 10), F(rn, 5), F(op4, 0)});
    dw(code);
  }

  void UncondBr1Reg(uint32_t opc, uint32_t op2, uint32_t op3, const RReg &rn,
                    uint32_t op4) {
    uint32_t code = concat({F(0x6b, 25), F(opc, 21), F(op2, 16), F(op3, 10),
                            F(rn.getIdx(), 5), F(op4, 0)});
    dw(code);
  }

  void UncondBr2Reg(uint32_t opc, uint32_t op2, uint32_t op3, const RReg &rn,
                    const RReg &rm) {
    uint32_t code = concat({F(0x6b, 25), F(opc, 21), F(op2, 16), F(op3, 10),
                            F(rn.getIdx(), 5), F(rm.getIdx(), 0)});
    dw(code);
  }

  // Unconditional branch (immediate)
  uint32_t UncondBrImmEnc(uint32_t op, int64_t labelOffset) {
    verifyIncRange(labelOffset, -1 * (1 << 27), ones(27), ERR_LABEL_IS_TOO_FAR,
                   true);
    uint32_t imm26 = static_cast<uint32_t>((labelOffset >> 2) & ones(26));
    return concat({F(op, 31), F(5, 26), F(imm26, 0)});
  }

  void UncondBrImm(uint32_t op, const Label &label) {
    auto encFunc = [&, op](int64_t labelOffset) {
      return UncondBrImmEnc(op, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, getSize());
    uint32_t code = UncondBrImmEnc(op, genLabelOffset(label, jmpL));
    dw(code);
  }

  void UncondBrImm(uint32_t op, int64_t label) {
    uint32_t code = UncondBrImmEnc(op, label);
    dw(code);
  }

  // Compare and branch (immediate)
  uint32_t CompareBrEnc(uint32_t op, const RReg &rt, int64_t labelOffset) {
    verifyIncRange(labelOffset, -1 * (1 << 20), ones(20), ERR_LABEL_IS_TOO_FAR,
                   true);

    uint32_t sf = genSf(rt);
    uint32_t imm19 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(19);
    return concat(
        {F(sf, 31), F(0x1a, 25), F(op, 24), F(imm19, 5), F(rt.getIdx(), 0)});
  }

  void CompareBr(uint32_t op, const RReg &rt, const Label &label) {
    auto encFunc = [&, op](int64_t labelOffset) {
      return CompareBrEnc(op, rt, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, getSize());
    uint32_t code = CompareBrEnc(op, rt, genLabelOffset(label, jmpL));
    dw(code);
  }

  void CompareBr(uint32_t op, const RReg &rt, int64_t label) {
    uint32_t code = CompareBrEnc(op, rt, label);
    dw(code);
  }

  // Test and branch (immediate)
  uint32_t TestBrEnc(uint32_t op, const RReg &rt, uint32_t imm,
                     int64_t labelOffset) {
    verifyIncRange(labelOffset, -1 * (1 << 15), ones(15), ERR_LABEL_IS_TOO_FAR,
                   true);
    verifyIncRange(imm, 0, ones(6), ERR_ILLEGAL_IMM_RANGE);

    uint32_t b5 = field(imm, 5, 5);
    uint32_t b40 = field(imm, 4, 0);
    uint32_t imm14 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(14);

    if (b5 == 1)
      verifyIncList(rt.getBit(), {64}, ERR_ILLEGAL_IMM_VALUE);

    return concat({F(b5, 31), F(0x1b, 25), F(op, 24), F(b40, 19), F(imm14, 5),
                   F(rt.getIdx(), 0)});
  }

  void TestBr(uint32_t op, const RReg &rt, uint32_t imm, const Label &label) {
    auto encFunc = [&, op, rt, imm](int64_t labelOffset) {
      return TestBrEnc(op, rt, imm, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, getSize());
    uint32_t code = TestBrEnc(op, rt, imm, genLabelOffset(label, jmpL));
    dw(code);
  }

  void TestBr(uint32_t op, const RReg &rt, uint32_t imm, int64_t label) {
    uint32_t code = TestBrEnc(op, rt, imm, label);
    dw(code);
  }

  // Advanced SIMD load/store multipule structure
  void AdvSimdLdStMultiStructExceptLd1St1(uint32_t L, uint32_t opc,
                                          const VRegList &vt,
                                          const AdrNoOfs &adr) {
    uint32_t Q = genQ(vt);
    uint32_t size = genSize(vt);
    uint32_t len = vt.getLen();

    verifyIncRange(len, 1, 4, ERR_ILLEGAL_REG_IDX);

    opc = (opc == 0x2 && len == 1)
              ? 0x7
              : (opc == 0x2 && len == 2) ? 0xa
                                         : (opc == 0x2 && len == 3) ? 0x6 : opc;
    uint32_t code =
        concat({F(Q, 30), F(0x18, 23), F(L, 22), F(opc, 12), F(size, 10),
                F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  void AdvSimdLdStMultiStructForLd1St1(uint32_t L, uint32_t opc,
                                       const VRegList &vt,
                                       const AdrNoOfs &adr) {
    AdvSimdLdStMultiStructExceptLd1St1(L, opc, vt, adr);
  }

  // Advanced SIMD load/store multple structures (post-indexed register offset)
  void AdvSimdLdStMultiStructPostRegExceptLd1St1(uint32_t L, uint32_t opc,
                                                 const VRegList &vt,
                                                 const AdrPostReg &adr) {
    uint32_t Q = genQ(vt);
    uint32_t size = genSize(vt);

    verifyIncRange(adr.getXm().getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t len = vt.getLen();
    verifyIncRange(len, 1, 4, ERR_ILLEGAL_REG_IDX);
    opc = (opc == 0x2 && len == 1)
              ? 0x7
              : (opc == 0x2 && len == 2) ? 0xa
                                         : (opc == 0x2 && len == 3) ? 0x6 : opc;
    uint32_t code =
        concat({F(Q, 30), F(0x19, 23), F(L, 22), F(adr.getXm().getIdx(), 16),
                F(opc, 12), F(size, 10), F(adr.getXn().getIdx(), 5),
                F(vt.getIdx(), 0)});
    dw(code);
  }

  void AdvSimdLdStMultiStructPostRegForLd1St1(uint32_t L, uint32_t opc,
                                              const VRegList &vt,
                                              const AdrPostReg &adr) {
    AdvSimdLdStMultiStructPostRegExceptLd1St1(L, opc, vt, adr);
  }

  // Advanced SIMD load/store multple structures (post-indexed immediate offset)
  void AdvSimdLdStMultiStructPostImmExceptLd1St1(uint32_t L, uint32_t opc,
                                                 const VRegList &vt,
                                                 const AdrPostImm &adr) {
    uint32_t Q = genQ(vt);
    uint32_t size = genSize(vt);
    uint32_t len = vt.getLen();

    verifyIncRange(adr.getImm(), 0, ((8 * len) << Q), ERR_ILLEGAL_IMM_RANGE);
    verifyIncRange(len, 1, 4, ERR_ILLEGAL_REG_IDX);

    opc = (opc == 0x2 && len == 1)
              ? 0x7
              : (opc == 0x2 && len == 2) ? 0xa
                                         : (opc == 0x2 && len == 3) ? 0x6 : opc;
    uint32_t code =
        concat({F(Q, 30), F(0x19, 23), F(L, 22), F(0x1f, 16), F(opc, 12),
                F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD load/store multple structures (post-indexed immediate offset)
  void AdvSimdLdStMultiStructPostImmForLd1St1(uint32_t L, uint32_t opc,
                                              const VRegList &vt,
                                              const AdrPostImm &adr) {
    AdvSimdLdStMultiStructPostImmExceptLd1St1(L, opc, vt, adr);
  }

  // Advanced SIMD load/store single structures
  void AdvSimdLdStSingleStruct(uint32_t L, uint32_t R, uint32_t num,
                               const VRegElem &vt, const AdrNoOfs &adr) {
    uint32_t Q = genQ(vt);
    uint32_t S = genS(vt);
    uint32_t size = genSizeEnc(vt);
    uint32_t opc = (vt.getBit() == 8)
                       ? field(num - 1, 1, 1)
                       : (vt.getBit() == 16) ? field(num - 1, 1, 1) + 2
                                             : field(num - 1, 1, 1) + 4;
    uint32_t code =
        concat({F(Q, 30), F(0x1a, 23), F(L, 22), F(R, 21), F(opc, 13), F(S, 12),
                F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD load replication single structures
  void AdvSimdLdRepSingleStruct(uint32_t L, uint32_t R, uint32_t opcode,
                                uint32_t S, const VRegVec &vt,
                                const AdrNoOfs &adr) {
    uint32_t Q = genQ(vt);
    uint32_t size = genSize(vt);
    uint32_t code = concat({F(Q, 30), F(0x1a, 23), F(L, 22), F(R, 21),
                            F(opcode, 13), F(S, 12), F(size, 10),
                            F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD load/store single structures (post-indexed register)
  void AdvSimdLdStSingleStructPostReg(uint32_t L, uint32_t R, uint32_t num,
                                      const VRegElem &vt,
                                      const AdrPostReg &adr) {
    uint32_t Q = genQ(vt);
    uint32_t S = genS(vt);
    uint32_t size = genSizeEnc(vt);
    uint32_t opc = (vt.getBit() == 8)
                       ? field(num - 1, 1, 1)
                       : (vt.getBit() == 16) ? field(num - 1, 1, 1) + 2
                                             : field(num - 1, 1, 1) + 4;
    uint32_t code =
        concat({F(Q, 30), F(0x1b, 23), F(L, 22), F(R, 21),
                F(adr.getXm().getIdx(), 16), F(opc, 13), F(S, 12), F(size, 10),
                F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD load/store single structures (post-indexed register,
  // replicate)
  void AdvSimdLdStSingleStructRepPostReg(uint32_t L, uint32_t R,
                                         uint32_t opcode, uint32_t S,
                                         const VRegVec &vt,
                                         const AdrPostReg &adr) {
    uint32_t Q = genQ(vt);
    uint32_t size = genSize(vt);
    uint32_t code =
        concat({F(Q, 30), F(0x1b, 23), F(L, 22), F(R, 21),
                F(adr.getXm().getIdx(), 16), F(opcode, 13), F(S, 12),
                F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD load/store single structures (post-indexed immediate)
  void AdvSimdLdStSingleStructPostImm(uint32_t L, uint32_t R, uint32_t num,
                                      const VRegElem &vt,
                                      const AdrPostImm &adr) {
    uint32_t Q = genQ(vt);
    uint32_t S = genS(vt);
    uint32_t size = genSizeEnc(vt);
    uint32_t opc = (vt.getBit() == 8)
                       ? field(num - 1, 1, 1)
                       : (vt.getBit() == 16) ? field(num - 1, 1, 1) + 2
                                             : field(num - 1, 1, 1) + 4;

    verifyIncList(adr.getImm(), {num * vt.getBit() / 8}, ERR_ILLEGAL_IMM_VALUE);

    uint32_t code = concat({F(Q, 30), F(0x1b, 23), F(L, 22), F(R, 21),
                            F(0x1f, 16), F(opc, 13), F(S, 12), F(size, 10),
                            F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD load replication single structures (post-indexed immediate)
  void AdvSimdLdRepSingleStructPostImm(uint32_t L, uint32_t R, uint32_t opcode,
                                       uint32_t S, const VRegVec &vt,
                                       const AdrPostImm &adr) {
    uint32_t Q = genQ(vt);
    uint32_t size = genSize(vt);
    uint32_t len = (field(opcode, 0, 0) << 1) + R + 1;

    verifyIncList(adr.getImm(), {len << size}, ERR_ILLEGAL_IMM_VALUE);

    uint32_t code = concat({F(Q, 30), F(0x1b, 23), F(L, 22), F(R, 21),
                            F(0x1f, 16), F(opcode, 13), F(S, 12), F(size, 10),
                            F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // store exclusive
  void StExclusive(uint32_t size, uint32_t o0, const WReg ws, const RReg &rt,
                   const AdrImm &adr) {
    uint32_t L = 0;
    uint32_t o2 = 0;
    uint32_t o1 = 0;

    verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(ws.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22), F(o1, 21),
                F(ws.getIdx(), 16), F(o0, 15), F(0x1f, 10),
                F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // load exclusive
  void LdExclusive(uint32_t size, uint32_t o0, const RReg &rt,
                   const AdrImm &adr) {
    uint32_t L = 1;
    uint32_t o2 = 0;
    uint32_t o1 = 0;

    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);

    uint32_t code = concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22),
                            F(o1, 21), F(0x1f, 16), F(o0, 15), F(0x1f, 10),
                            F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // store LORelease
  void StLORelase(uint32_t size, uint32_t o0, const RReg &rt,
                  const AdrImm &adr) {
    uint32_t L = 0;
    uint32_t o2 = 1;
    uint32_t o1 = 0;

    verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22),
                            F(o1, 21), F(0x1f, 16), F(o0, 15), F(0x1f, 10),
                            F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // load LOAcquire
  void LdLOAcquire(uint32_t size, uint32_t o0, const RReg &rt,
                   const AdrImm &adr) {
    uint32_t L = 1;
    uint32_t o2 = 1;
    uint32_t o1 = 0;

    verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22),
                            F(o1, 21), F(0x1f, 16), F(o0, 15), F(0x1f, 10),
                            F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // compare and swap
  void Cas(uint32_t size, uint32_t o2, uint32_t L, uint32_t o1, uint32_t o0,
           const RReg &rs, const RReg &rt, const AdrNoOfs &adr) {
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rs.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22), F(o1, 21),
                F(rs.getIdx(), 16), F(o0, 15), F(0x1f, 10),
                F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // load/store exclusive pair
  void StExclusivePair(uint32_t L, uint32_t o1, uint32_t o0, const WReg &ws,
                       const RReg &rt1, const RReg &rt2, const AdrImm &adr) {
    uint32_t sz = (rt1.getBit() == 64) ? 1 : 0;

    verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
    verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(ws.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(1, 31), F(sz, 30), F(0x8, 24), F(0, 23), F(L, 22), F(o1, 21),
                F(ws.getIdx(), 16), F(o0, 15), F(rt2.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
    dw(code);
  }

  // load/store exclusive pair
  void LdExclusivePair(uint32_t L, uint32_t o1, uint32_t o0, const RReg &rt1,
                       const RReg &rt2, const AdrImm &adr) {
    uint32_t sz = (rt1.getBit() == 64) ? 1 : 0;

    verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
    verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(1, 31), F(sz, 30), F(0x8, 24), F(0, 23), F(L, 22), F(o1, 21),
                F(0x1f, 16), F(o0, 15), F(rt2.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
    dw(code);
  }

  // compare and swap pair
  void CasPair(uint32_t L, uint32_t o1, uint32_t o0, const RReg &rs,
               const RReg &rt, const AdrNoOfs &adr) {
    uint32_t sz = (rt.getBit() == 64) ? 1 : 0;

    verifyIncRange(rs.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(0, 31), F(sz, 30), F(0x8, 24), F(0, 23), F(L, 22), F(o1, 21),
                F(rs.getIdx(), 16), F(o0, 15), F(0x1f, 10),
                F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // LDAPR/STLR (unscaled immediate)
  void LdaprStlr(uint32_t size, uint32_t opc, const RReg &rt,
                 const AdrImm &adr) {
    int32_t simm = adr.getImm();
    uint32_t imm9 = simm & ones(9);

    verifyIncRange(simm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(size, 30), F(0x19, 24), F(opc, 22), F(imm9, 12),
                            F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // load register (literal)
  uint32_t LdRegLiteralEnc(uint32_t opc, uint32_t V, const RReg &rt,
                           int64_t labelOffset) {
    verifyIncRange(labelOffset, (-1) * (1 << 20), ones(20),
                   ERR_LABEL_IS_TOO_FAR, true);

    uint32_t imm19 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(19);
    return concat(
        {F(opc, 30), F(0x3, 27), F(V, 26), F(imm19, 5), F(rt.getIdx(), 0)});
  }

  void LdRegLiteral(uint32_t opc, uint32_t V, const RReg &rt,
                    const Label &label) {
    auto encFunc = [&, opc, V, rt](int64_t labelOffset) {
      return LdRegLiteralEnc(opc, V, rt, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, getSize());
    uint32_t code = LdRegLiteralEnc(opc, V, rt, genLabelOffset(label, jmpL));
    dw(code);
  }

  void LdRegLiteral(uint32_t opc, uint32_t V, const RReg &rt, int64_t label) {
    uint32_t code = LdRegLiteralEnc(opc, V, rt, label);
    dw(code);
  }

  // load register (SIMD&FP, literal)
  uint32_t LdRegSimdFpLiteralEnc(const VRegSc &vt, int64_t labelOffset) {
    verifyIncRange(labelOffset, -1 * (1 << 20), ones(20), ERR_LABEL_IS_TOO_FAR,
                   true);

    uint32_t opc = (vt.getBit() == 32) ? 0 : (vt.getBit() == 64) ? 1 : 2;
    uint32_t imm19 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(19);
    uint32_t V = 1;
    return concat(
        {F(opc, 30), F(0x3, 27), F(V, 26), F(imm19, 5), F(vt.getIdx(), 0)});
  }

  void LdRegSimdFpLiteral(const VRegSc &vt, const Label &label) {
    auto encFunc = [&, vt](int64_t labelOffset) {
      return LdRegSimdFpLiteralEnc(vt, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, getSize());
    uint32_t code = LdRegSimdFpLiteralEnc(vt, genLabelOffset(label, jmpL));
    dw(code);
  }

  void LdRegSimdFpLiteral(const VRegSc &vt, int64_t label) {
    uint32_t code = LdRegSimdFpLiteralEnc(vt, label);
    dw(code);
  }

  // prefetch (literal)
  uint32_t PfLiteralEnc(Prfop prfop, int64_t labelOffset) {
    verifyIncRange(labelOffset, -1 * (1 << 20), ones(20), ERR_LABEL_IS_TOO_FAR,
                   true);

    uint32_t opc = 3;
    uint32_t imm19 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(19);
    uint32_t V = 0;
    return concat({F(opc, 30), F(0x3, 27), F(V, 26), F(imm19, 5), F(prfop, 0)});
  }

  void PfLiteral(Prfop prfop, const Label &label) {
    auto encFunc = [&, prfop](int64_t labelOffset) {
      return PfLiteralEnc(prfop, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, getSize());
    uint32_t code = PfLiteralEnc(prfop, genLabelOffset(label, jmpL));
    dw(code);
  }

  void PfLiteral(Prfop prfop, int64_t label) {
    uint32_t code = PfLiteralEnc(prfop, label);
    dw(code);
  }

  // Load/store no-allocate pair (offset)
  void LdStNoAllocPair(uint32_t L, const RReg &rt1, const RReg &rt2,
                       const AdrImm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = (rt1.getBit() == 32) ? 1 : 2;

    verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE,
                   true);
    verifyCond(
        imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t opc = (rt1.getBit() == 32) ? 0 : 2;
    uint32_t imm7 = (imm >> (times + 1)) & ones(7);
    uint32_t V = 0;

    verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(L, 22),
                            F(imm7, 15), F(rt2.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
    dw(code);
  }

  // Load/store no-allocate pair (offset)
  void LdStSimdFpNoAllocPair(uint32_t L, const VRegSc &vt1, const VRegSc &vt2,
                             const AdrImm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = vt1.getBit() / 32;

    verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE,
                   true);
    verifyCond(
        imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t opc = (vt1.getBit() == 32) ? 0 : (vt1.getBit() == 64) ? 1 : 2;
    uint32_t sh = static_cast<uint32_t>(std::log2(4 * times));
    uint32_t imm7 = (imm >> sh) & ones(7);
    uint32_t V = 1;
    uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(L, 22),
                            F(imm7, 15), F(vt2.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(vt1.getIdx(), 0)});
    dw(code);
  }

  // Load/store pair (post-indexed)
  void LdStRegPairPostImm(uint32_t opc, uint32_t L, const RReg &rt1,
                          const RReg &rt2, const AdrPostImm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = (opc == 2) ? 2 : 1;

    verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE,
                   true);
    verifyCond(
        imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t imm7 = (imm >> (times + 1)) & ones(7);
    uint32_t V = 0;

    verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(1, 23),
                            F(L, 22), F(imm7, 15), F(rt2.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
    dw(code);
  }

  // Load/store pair (post-indexed)
  void LdStSimdFpPairPostImm(uint32_t L, const VRegSc &vt1, const VRegSc &vt2,
                             const AdrPostImm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = vt1.getBit() / 32;

    verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE,
                   true);
    verifyCond(
        imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t opc = (vt1.getBit() == 32) ? 0 : (vt1.getBit() == 64) ? 1 : 2;
    uint32_t sh = static_cast<uint32_t>(std::log2(4 * times));
    uint32_t imm7 = (imm >> sh) & ones(7);
    uint32_t V = 1;
    uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(1, 23),
                            F(L, 22), F(imm7, 15), F(vt2.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(vt1.getIdx(), 0)});
    dw(code);
  }

  // Load/store pair (offset)
  void LdStRegPair(uint32_t opc, uint32_t L, const RReg &rt1, const RReg &rt2,
                   const AdrImm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = (opc == 2) ? 2 : 1;

    verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE,
                   true);
    verifyCond(
        imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t imm7 = (imm >> (times + 1)) & ones(7);
    uint32_t V = 0;

    verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(2, 23),
                            F(L, 22), F(imm7, 15), F(rt2.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
    dw(code);
  }

  // Load/store pair (offset)
  void LdStSimdFpPair(uint32_t L, const VRegSc &vt1, const VRegSc &vt2,
                      const AdrImm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = vt1.getBit() / 32;

    verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE,
                   true);
    verifyCond(
        imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t opc = (vt1.getBit() == 32) ? 0 : (vt1.getBit() == 64) ? 1 : 2;
    uint32_t sh = static_cast<uint32_t>(std::log2(4 * times));
    uint32_t imm7 = (imm >> sh) & ones(7);
    uint32_t V = 1;
    uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(2, 23),
                            F(L, 22), F(imm7, 15), F(vt2.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(vt1.getIdx(), 0)});
    dw(code);
  }

  // Load/store pair (pre-indexed)
  void LdStRegPairPre(uint32_t opc, uint32_t L, const RReg &rt1,
                      const RReg &rt2, const AdrPreImm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = (opc == 2) ? 2 : 1;

    verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE,
                   true);
    verifyCond(
        imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t imm7 = (imm >> (times + 1)) & ones(7);
    uint32_t V = 0;

    verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(3, 23),
                            F(L, 22), F(imm7, 15), F(rt2.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
    dw(code);
  }

  // Load/store pair (pre-indexed)
  void LdStSimdFpPairPre(uint32_t L, const VRegSc &vt1, const VRegSc &vt2,
                         const AdrPreImm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = vt1.getBit() / 32;

    verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE,
                   true);
    verifyCond(
        imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t opc = (vt1.getBit() == 32) ? 0 : (vt1.getBit() == 64) ? 1 : 2;
    uint32_t sh = static_cast<uint32_t>(std::log2(4 * times));
    uint32_t imm7 = (imm >> sh) & ones(7);
    uint32_t V = 1;
    uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(3, 23),
                            F(L, 22), F(imm7, 15), F(vt2.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(vt1.getIdx(), 0)});
    dw(code);
  }

  // Load/store register (unscaled immediate)
  void LdStRegUnsImm(uint32_t size, uint32_t opc, const RReg &rt,
                     const AdrImm &adr) {
    int imm = adr.getImm();
    uint32_t imm9 = imm & ones(9);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t V = 0;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12),
                F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // Load/store register (SIMD&FP, unscaled immediate)
  void LdStSimdFpRegUnsImm(uint32_t opc, const VRegSc &vt, const AdrImm &adr) {
    uint32_t size = (vt.getBit() == 16)
                        ? 1
                        : (vt.getBit() == 32) ? 2 : (vt.getBit() == 64) ? 3 : 0;

    int imm = adr.getImm();
    uint32_t imm9 = adr.getImm() & ones(9);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t V = 1;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12),
                F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // prefetch register (unscaled immediate)
  void PfRegUnsImm(Prfop prfop, const AdrImm &adr) {
    uint32_t size = 3;
    uint32_t opc = 2;

    int imm = adr.getImm();
    uint32_t imm9 = imm & ones(9);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t V = 0;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12),
                F(adr.getXn().getIdx(), 5), F(prfop, 0)});
    dw(code);
  }

  // Load/store register (immediate post-indexed)
  void LdStRegPostImm(uint32_t size, uint32_t opc, const RReg &rt,
                      const AdrPostImm &adr) {
    int imm = adr.getImm();
    uint32_t imm9 = imm & ones(9);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t V = 0;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12),
                F(1, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // Load/store register (SIMD&FP, immediate post-indexed)
  void LdStSimdFpRegPostImm(uint32_t opc, const VRegSc &vt,
                            const AdrPostImm &adr) {
    uint32_t size = (vt.getBit() == 16)
                        ? 1
                        : (vt.getBit() == 32) ? 2 : (vt.getBit() == 64) ? 3 : 0;

    int imm = adr.getImm();
    uint32_t imm9 = imm & ones(9);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t V = 1;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12),
                F(1, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // Load/store register (unprivileged)
  void LdStRegUnpriv(uint32_t size, uint32_t opc, const RReg &rt,
                     const AdrImm &adr) {
    int imm = adr.getImm();
    uint32_t imm9 = imm & ones(9);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t V = 0;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12),
                F(2, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // Load/store register (immediate pre-indexed)
  void LdStRegPre(uint32_t size, uint32_t opc, const RReg &rt,
                  const AdrPreImm &adr) {
    int imm = adr.getImm();
    uint32_t imm9 = imm & ones(9);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t V = 0;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12),
                F(3, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // Load/store register (SIMD&FP, immediate pre-indexed)
  void LdStSimdFpRegPre(uint32_t opc, const VRegSc &vt, const AdrPreImm &adr) {
    uint32_t size = (vt.getBit() == 16)
                        ? 1
                        : (vt.getBit() == 32) ? 2 : (vt.getBit() == 64) ? 3 : 0;

    int imm = adr.getImm();
    uint32_t imm9 = imm & ones(9);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t V = 1;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12),
                F(3, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // Atomic memory oprations
  void AtomicMemOp(uint32_t size, uint32_t V, uint32_t A, uint32_t R,
                   uint32_t o3, uint32_t opc, const RReg &rs, const RReg &rt,
                   const AdrNoOfs &adr) {
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(A, 23), F(R, 22), F(1, 21),
                F(rs.getIdx(), 16), F(o3, 15), F(opc, 12),
                F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  void AtomicMemOp(uint32_t size, uint32_t V, uint32_t A, uint32_t R,
                   uint32_t o3, uint32_t opc, const RReg &rs, const RReg &rt,
                   const AdrImm &adr) {
    verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(A, 23), F(R, 22), F(1, 21),
                F(rs.getIdx(), 16), F(o3, 15), F(opc, 12),
                F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // load/store register (register offset)
  void LdStReg(uint32_t size, uint32_t opc, const RReg &rt, const AdrReg &adr) {
    uint32_t option = 3;
    uint32_t S =
        ((adr.getInitSh() && size == 0) || (adr.getSh() != 0 && size != 0)) ? 1
                                                                            : 0;
    uint32_t V = 0;

    verifyIncList(adr.getSh(), {0, size}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncList(adr.getMod(), {LSL}, ERR_ILLEGAL_SHMOD);

    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21),
                F(adr.getXm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10),
                F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // load/store register (register offset)
  void LdStReg(uint32_t size, uint32_t opc, const RReg &rt, const AdrExt &adr) {
    uint32_t option = adr.getMod();
    uint32_t S =
        ((adr.getInitSh() && size == 0) || (adr.getSh() != 0 && size != 0)) ? 1
                                                                            : 0;
    uint32_t V = 0;

    verifyIncList(adr.getSh(), {0, size}, ERR_ILLEGAL_CONST_VALUE);
    if (adr.getRm().getBit() == 64)
      verifyIncList(option, {SXTX}, ERR_ILLEGAL_EXTMOD);
    else
      verifyIncList(option, {UXTW, SXTW}, ERR_ILLEGAL_EXTMOD);

    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21),
                F(adr.getRm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10),
                F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
    dw(code);
  }

  // load/store register (register offset)
  void LdStSimdFpReg(uint32_t opc, const VRegSc &vt, const AdrReg &adr) {
    uint32_t size = genSize(vt);
    uint32_t option = 3;
    uint32_t vt_bit = vt.getBit();
    uint32_t S =
        ((adr.getInitSh() && vt_bit == 8) || (adr.getSh() != 0 && vt_bit != 8))
            ? 1
            : 0;
    uint32_t V = 1;

    verifyIncList(adr.getSh(), {0, size}, ERR_ILLEGAL_CONST_VALUE);

    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21),
                F(adr.getXm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10),
                F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // load/store register (register offset)
  void LdStSimdFpReg(uint32_t opc, const VRegSc &vt, const AdrExt &adr) {
    uint32_t size = genSize(vt);
    uint32_t option = adr.getMod();
    uint32_t vt_bit = vt.getBit();
    uint32_t S =
        ((adr.getInitSh() && vt_bit == 8) || (adr.getSh() != 0 && vt_bit != 8))
            ? 1
            : 0;
    uint32_t V = 1;

    uint32_t max_sh = (vt.getBit() == 128) ? 4 : size;
    verifyIncList(adr.getSh(), {0, max_sh}, ERR_ILLEGAL_CONST_VALUE);

    if (adr.getRm().getBit() == 64)
      verifyIncList(option, {SXTX}, ERR_ILLEGAL_EXTMOD);
    else
      verifyIncList(option, {UXTW, SXTW}, ERR_ILLEGAL_EXTMOD);

    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21),
                F(adr.getRm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10),
                F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // load/store register (register offset)
  void PfExt(Prfop prfop, const AdrReg &adr) {
    uint32_t size = 3;
    uint32_t opc = 2;
    uint32_t option = adr.getMod();
    uint32_t S =
        ((adr.getInitSh() && size == 0) || (adr.getSh() != 0 && size != 0)) ? 1
                                                                            : 0;
    uint32_t V = 0;

    verifyIncList(adr.getSh(), {0, 3}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncList(option, {LSL}, ERR_ILLEGAL_SHMOD);

    uint32_t ext_opt = 3;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21),
                F(adr.getXm().getIdx(), 16), F(ext_opt, 13), F(S, 12), F(2, 10),
                F(adr.getXn().getIdx(), 5), F(prfop, 0)});
    dw(code);
  }

  void PfExt(Prfop prfop, const AdrExt &adr) {
    uint32_t size = 3;
    uint32_t opc = 2;
    uint32_t option = adr.getMod();
    uint32_t S =
        ((adr.getInitSh() && size == 0) || (adr.getSh() != 0 && size != 0)) ? 1
                                                                            : 0;
    uint32_t V = 0;

    verifyIncList(adr.getSh(), {0, 3}, ERR_ILLEGAL_CONST_VALUE);

    if (adr.getRm().getBit() == 64)
      verifyIncList(option, {SXTX}, ERR_ILLEGAL_EXTMOD);
    else
      verifyIncList(option, {UXTW, SXTW}, ERR_ILLEGAL_EXTMOD);

    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21),
                F(adr.getRm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10),
                F(adr.getXn().getIdx(), 5), F(prfop, 0)});
    dw(code);
  }

  // loat/store register (pac)
  void LdStRegPac(uint32_t M, uint32_t W, const XReg &xt, const AdrImm &adr) {
    uint32_t size = 3;
    uint32_t V = 0;

    int32_t imm = adr.getImm();
    uint32_t S = (imm < 0) ? 1 : 0;
    uint32_t imm9 = (imm >> 3) & ones(9);

    verifyIncRange(imm, -4096, 4088, ERR_ILLEGAL_IMM_RANGE, true);
    verifyCond(
        std::abs(imm), [](uint64_t x) { return ((x % 8) == 0); },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(xt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(M, 23),
                            F(S, 22), F(1, 21), F(imm9, 12), F(W, 11), F(1, 10),
                            F(adr.getXn().getIdx(), 5), F(xt.getIdx(), 0)});
    dw(code);
  }

  // loat/store register (pac)
  void LdStRegPac(uint32_t M, uint32_t W, const XReg &xt,
                  const AdrPreImm &adr) {
    uint32_t size = 3;
    uint32_t V = 0;

    int32_t imm = adr.getImm();
    uint32_t S = (imm < 0) ? 1 : 0;
    uint32_t imm9 = (imm >> 3) & ones(9);

    verifyIncRange(imm, -4096, 4088, ERR_ILLEGAL_IMM_RANGE, true);
    verifyCond(
        std::abs(imm), [](uint64_t x) { return ((x % 8) == 0); },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(xt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(M, 23),
                            F(S, 22), F(1, 21), F(imm9, 12), F(W, 11), F(1, 10),
                            F(adr.getXn().getIdx(), 5), F(xt.getIdx(), 0)});
    dw(code);
  }

  // loat/store register (unsigned immediate)
  void LdStRegUnImm(uint32_t size, uint32_t opc, const RReg &rt,
                    const AdrUimm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = 1 << size;
    uint32_t imm12 = (imm >> size) & ones(12);

    verifyIncRange(imm, 0, 4095 * times, ERR_ILLEGAL_IMM_RANGE);
    verifyCond(
        imm, [=](uint64_t x) { return ((x & ones(size)) == 0); },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t V = 0;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(1, 24), F(opc, 22),
                F(imm12, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});

    dw(code);
  }

  // loat/store register (unsigned immediate)
  void LdStSimdFpUnImm(uint32_t opc, const VRegSc &vt, const AdrUimm &adr) {
    int32_t imm = adr.getImm();
    uint32_t times = vt.getBit() / 8;
    uint32_t sh = (uint32_t)std::log2(times);
    uint32_t imm12 = (imm >> sh) & ones(12);

    verifyIncRange(imm, 0, 4095 * times, ERR_ILLEGAL_IMM_RANGE);
    verifyCond(
        imm, [=](uint64_t x) { return ((x & ones(sh)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t V = 1;
    uint32_t size = genSize(vt);
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(1, 24), F(opc, 22),
                F(imm12, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
    dw(code);
  }

  // loat/store register (unsigned immediate)
  void PfRegImm(Prfop prfop, const AdrUimm &adr) {
    int32_t imm = adr.getImm();
    int32_t times = 8;
    uint32_t imm12 = (imm >> 3) & ones(12);

    verifyIncRange(imm, 0, 4095 * times, ERR_ILLEGAL_IMM_RANGE);
    verifyCond(
        imm, [=](uint64_t x) { return ((x & ones(3)) == 0); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t size = 3;
    uint32_t opc = 2;
    uint32_t V = 0;
    uint32_t code =
        concat({F(size, 30), F(0x7, 27), F(V, 26), F(1, 24), F(opc, 22),
                F(imm12, 10), F(adr.getXn().getIdx(), 5), F(prfop, 0)});
    dw(code);
  }

  // Data processing (2 source)
  void DataProc2Src(uint32_t opcode, const RReg &rd, const RReg &rn,
                    const RReg &rm) {
    uint32_t sf = genSf(rm);
    uint32_t S = 0;

    verifyCond(
        SP_IDX,
        [=](uint64_t x) {
          return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x;
        },
        ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(sf, 31), F(S, 29), F(0xd6, 21), F(rm.getIdx(), 16),
                F(opcode, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Data processing (1 source)
  void DataProc1Src(uint32_t opcode2, uint32_t opcode, const RReg &rd,
                    const RReg &rn) {
    uint32_t sf = genSf(rd);
    uint32_t S = 0;

    verifyCond(
        SP_IDX, [=](uint64_t x) { return rd.getIdx() < x || rn.getIdx() < x; },
        ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(sf, 31), F(1, 30), F(S, 29), F(0xd6, 21), F(opcode2, 16),
                F(opcode, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Data processing (1 source)
  void DataProc1Src(uint32_t opcode2, uint32_t opcode, const RReg &rd) {
    uint32_t sf = genSf(rd);
    uint32_t S = 0;

    verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(sf, 31), F(1, 30), F(S, 29), F(0xd6, 21), F(opcode2, 16),
                F(opcode, 10), F(0x1f, 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Logical (shifted register)
  void LogicalShiftReg(uint32_t opc, uint32_t N, const RReg &rd, const RReg &rn,
                       const RReg &rm, ShMod shmod, uint32_t sh) {
    uint32_t sf = genSf(rd);
    uint32_t imm6 = sh & ones(6);

    verifyIncRange(sh, 0, (32 << sf) - 1, ERR_ILLEGAL_CONST_RANGE);
    verifyCond(
        SP_IDX,
        [=](uint64_t x) {
          return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x;
        },
        ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(sf, 31), F(opc, 29), F(0xa, 24), F(shmod, 22),
                            F(N, 21), F(rm.getIdx(), 16), F(imm6, 10),
                            F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Move (register) alias of ADD,ORR
  void MvReg(const RReg &rd, const RReg &rn) {
    if (rd.getIdx() == SP_IDX || rn.getIdx() == SP_IDX) {
      // alias of ADD
      AddSubImm(0, 0, rd, rn, 0, 0);
    } else {
      // alias of ORR
      LogicalShiftReg(1, 0, rd, RReg(SP_IDX, rd.getBit()), rn, LSL, 0);
    }
  }

  // Add/subtract (shifted register)
  void AddSubShiftReg(uint32_t opc, uint32_t S, const RReg &rd, const RReg &rn,
                      const RReg &rm, ShMod shmod, uint32_t sh,
                      bool alias = false) {
    uint32_t rd_sp = (rd.getIdx() == SP_IDX);
    uint32_t rn_sp = (rn.getIdx() == SP_IDX);
    if (((rd_sp + rn_sp) >= 1 + (uint32_t)alias) && shmod == LSL) {
      AddSubExtReg(opc, S, rd, rn, rm, EXT_LSL, sh);
      return;
    }

    if (shmod == NONE)
      shmod = LSL;

    uint32_t sf = genSf(rd);
    uint32_t imm6 = sh & ones(6);

    verifyIncRange(sh, 0, (32 << sf) - 1, ERR_ILLEGAL_CONST_RANGE);
    if (!(alias && S == 1))
      verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    if (!(alias && opc == 1))
      verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(rm.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(sf, 31), F(opc, 30), F(S, 29), F(0xb, 24),
                            F(shmod, 22), F(rm.getIdx(), 16), F(imm6, 10),
                            F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Add/subtract (extended register)
  void AddSubExtReg(uint32_t opc, uint32_t S, const RReg &rd, const RReg &rn,
                    const RReg &rm, ExtMod extmod, uint32_t sh) {
    uint32_t sf = genSf(rd);
    uint32_t opt = 0;
    uint32_t imm3 = sh & ones(3);

    verifyIncRange(sh, 0, 4, ERR_ILLEGAL_CONST_RANGE);

    uint32_t option = (extmod == EXT_LSL && sf == 0)
                          ? 2
                          : (extmod == EXT_LSL && sf == 1) ? 3 : extmod;
    uint32_t code =
        concat({F(sf, 31), F(opc, 30), F(S, 29), F(0xb, 24), F(opt, 22),
                F(1, 21), F(rm.getIdx(), 16), F(option, 13), F(imm3, 10),
                F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Add/subtract (with carry)
  void AddSubCarry(uint32_t op, uint32_t S, const RReg &rd, const RReg &rn,
                   const RReg &rm) {
    uint32_t sf = genSf(rd);

    verifyCond(
        SP_IDX,
        [=](uint64_t x) {
          return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x;
        },
        ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd, 25), F(rm.getIdx(), 16),
                F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Rotate right into flags
  void RotateR(uint32_t op, uint32_t S, uint32_t o2, const XReg &xn,
               uint32_t sh, uint32_t mask) {
    uint32_t sf = genSf(xn);
    uint32_t imm6 = sh & ones(6);

    verifyIncRange(xn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(sh, 0, 63, ERR_ILLEGAL_CONST_RANGE);
    verifyIncRange(mask, 0, 15, ERR_ILLEGAL_CONST_RANGE);

    uint32_t code =
        concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd, 25), F(imm6, 15),
                F(0x1, 10), F(xn.getIdx(), 5), F(o2, 4), F(mask, 0)});
    dw(code);
  }

  // Evaluate into flags
  void Evaluate(uint32_t op, uint32_t S, uint32_t opcode2, uint32_t sz,
                uint32_t o3, uint32_t mask, const WReg &wn) {
    uint32_t sf = 0;

    verifyIncRange(wn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(mask, 0, 15, ERR_ILLEGAL_CONST_RANGE);

    uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd, 25),
                            F(opcode2, 15), F(sz, 14), F(0x2, 10),
                            F(wn.getIdx(), 5), F(o3, 4), F(mask, 0)});
    dw(code);
  }

  // Conditional compare (register)
  void CondCompReg(uint32_t op, uint32_t S, uint32_t o2, uint32_t o3,
                   const RReg &rn, const RReg &rm, uint32_t nczv, Cond cond) {
    uint32_t sf = genSf(rn);

    verifyCond(
        SP_IDX, [=](uint64_t x) { return rn.getIdx() < x || rm.getIdx() < x; },
        ERR_ILLEGAL_REG_IDX);
    verifyIncRange(nczv, 0, 15, ERR_ILLEGAL_CONST_RANGE);

    uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd2, 21),
                            F(rm.getIdx(), 16), F(cond, 12), F(o2, 10),
                            F(rn.getIdx(), 5), F(o3, 4), F(nczv, 0)});
    dw(code);
  }

  // Conditional compare (imm)
  void CondCompImm(uint32_t op, uint32_t S, uint32_t o2, uint32_t o3,
                   const RReg &rn, uint32_t imm, uint32_t nczv, Cond cond) {
    uint32_t sf = genSf(rn);
    uint32_t imm5 = imm & ones(5);

    verifyIncRange(imm, 0, 31, ERR_ILLEGAL_IMM_RANGE);
    verifyIncRange(nczv, 0, 15, ERR_ILLEGAL_CONST_RANGE);
    verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd2, 21),
                            F(imm5, 16), F(cond, 12), F(1, 11), F(o2, 10),
                            F(rn.getIdx(), 5), F(o3, 4), F(nczv, 0)});
    dw(code);
  }

  // Conditional select
  void CondSel(uint32_t op, uint32_t S, uint32_t op2, const RReg &rd,
               const RReg &rn, const RReg &rm, Cond cond) {
    uint32_t sf = genSf(rn);

    verifyCond(
        SP_IDX,
        [=](uint64_t x) {
          return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x;
        },
        ERR_ILLEGAL_REG_IDX);

    uint32_t code =
        concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd4, 21), F(rm.getIdx(), 16),
                F(cond, 12), F(op2, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Conditional select
  void DataProc3Reg(uint32_t op54, uint32_t op31, uint32_t o0, const RReg &rd,
                    const RReg &rn, const RReg &rm, const RReg &ra) {
    uint32_t sf = genSf(rd);

    verifyCond(
        SP_IDX,
        [=](uint64_t x) {
          return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x ||
                 ra.getIdx() < x;
        },
        ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(sf, 31), F(op54, 29), F(0x1b, 24), F(op31, 21),
                            F(rm.getIdx(), 16), F(o0, 15), F(ra.getIdx(), 10),
                            F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Conditional select
  void DataProc3Reg(uint32_t op54, uint32_t op31, uint32_t o0, const RReg &rd,
                    const RReg &rn, const RReg &rm) {
    uint32_t sf = genSf(rd);

    verifyCond(
        SP_IDX,
        [=](uint64_t x) {
          return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x;
        },
        ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(sf, 31), F(op54, 29), F(0x1b, 24), F(op31, 21),
                            F(rm.getIdx(), 16), F(o0, 15), F(0x1f, 10),
                            F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Cryptographic AES
  void CryptAES(uint32_t opcode, const VRegVec &vd, const VRegVec &vn) {
    uint32_t size = genSize(vd);
    uint32_t code =
        concat({F(0x4e, 24), F(size, 22), F(0x14, 17), F(opcode, 12), F(2, 10),
                F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Cryptographic three-register SHA
  void Crypt3RegSHA(uint32_t opcode, const VRegSc &vd, const VRegSc &vn,
                    const VRegVec &vm) {
    uint32_t size = 0;
    uint32_t code =
        concat({F(0x5e, 24), F(size, 22), F(vm.getIdx(), 16), F(opcode, 12),
                F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Cryptographic three-register SHA
  void Crypt3RegSHA(uint32_t opcode, const VRegVec &vd, const VRegVec &vn,
                    const VRegVec &vm) {
    uint32_t size = 0;
    uint32_t code =
        concat({F(0x5e, 24), F(size, 22), F(vm.getIdx(), 16), F(opcode, 12),
                F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Cryptographic two-register SHA
  void Crypt2RegSHA(uint32_t opcode, const Reg &vd, const Reg &vn) {
    uint32_t size = 0;
    uint32_t code =
        concat({F(0x5e, 24), F(size, 22), F(0x14, 17), F(opcode, 12), F(2, 10),
                F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD Scalar copy
  void AdvSimdScCopy(uint32_t op, uint32_t imm4, const VRegSc &vd,
                     const VRegElem &vn) {
    uint32_t sh = genSize(vd);
    uint32_t imm5 = 1 << sh | vn.getElemIdx() << (sh + 1);
    uint32_t code =
        concat({F(1, 30), F(op, 29), F(0xf, 25), F(imm5, 16), F(imm4, 11),
                F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD Scalar three same FP16
  void AdvSimdSc3SameFp16(uint32_t U, uint32_t a, uint32_t opcode,
                          const VRegSc &vd, const VRegSc &vn,
                          const VRegSc &vm) {
    uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(a, 23), F(2, 21),
                            F(vm.getIdx(), 16), F(opcode, 11), F(1, 10),
                            F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD Scalar two-register miscellaneous FP16
  void AdvSimdSc2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode,
                             const VRegSc &vd, const VRegSc &vn) {
    uint32_t code =
        concat({F(1, 30), F(U, 29), F(0xf, 25), F(a, 23), F(0xf, 19),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  void AdvSimdSc2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode,
                             const VRegSc &vd, const VRegSc &vn, double zero) {
    verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
    AdvSimdSc2RegMiscFp16(U, a, opcode, vd, vn);
  }

  // Advanced SIMD Scalar three same extra
  void AdvSimdSc3SameExtra(uint32_t U, uint32_t opcode, const VRegSc &vd,
                           const VRegSc &vn, const VRegSc &vm) {
    uint32_t size = genSize(vd);
    uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22),
                            F(vm.getIdx(), 16), F(1, 15), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD Scalar two-register miscellaneous
  void AdvSimdSc2RegMisc(uint32_t U, uint32_t opcode, const VRegSc &vd,
                         const VRegSc &vn) {
    uint32_t size = genSize(vd);
    uint32_t code =
        concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  void AdvSimdSc2RegMisc(uint32_t U, uint32_t opcode, const VRegSc &vd,
                         const VRegSc &vn, uint32_t zero) {
    verifyIncList(zero, {0}, ERR_ILLEGAL_CONST_VALUE);
    AdvSimdSc2RegMisc(U, opcode, vd, vn);
  }

  // Advanced SIMD Scalar two-register miscellaneous
  void AdvSimdSc2RegMiscSz0x(uint32_t U, uint32_t opcode, const VRegSc &vd,
                             const VRegSc &vn) {
    uint32_t size = genSize(vn) & 1;
    uint32_t code =
        concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD Scalar two-register miscellaneous
  void AdvSimdSc2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd,
                             const VRegSc &vn) {
    uint32_t size = genSize(vd);
    uint32_t code =
        concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  void AdvSimdSc2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd,
                             const VRegSc &vn, double zero) {
    verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
    AdvSimdSc2RegMiscSz1x(U, opcode, vd, vn);
  }

  // Advanced SIMD scalar pairwize
  void AdvSimdScPairwise(uint32_t U, uint32_t size, uint32_t opcode,
                         const VRegSc &vd, const VRegVec &vn) {
    uint32_t code =
        concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(3, 20),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD scalar three different
  void AdvSimdSc3Diff(uint32_t U, uint32_t opcode, const VRegSc &vd,
                      const VRegSc &vn, const VRegSc &vm) {
    uint32_t size = genSize(vn);
    uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 12),
                            F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD scalar three same
  void AdvSimdSc3Same(uint32_t U, uint32_t opcode, const VRegSc &vd,
                      const VRegSc &vn, const VRegSc &vm) {
    uint32_t size = genSize(vd);
    uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD scalar three same
  void AdvSimdSc3SameSz0x(uint32_t U, uint32_t opcode, const VRegSc &vd,
                          const VRegSc &vn, const VRegSc &vm) {
    uint32_t size = genSize(vd) & 1;
    uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD scalar three same
  void AdvSimdSc3SameSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd,
                          const VRegSc &vn, const VRegSc &vm) {
    uint32_t size = genSize(vd);
    uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD scalar shift by immediate
  void AdvSimdScShImm(uint32_t U, uint32_t opcode, const VRegSc &vd,
                      const VRegSc &vn, uint32_t sh) {
    uint32_t size = genSize(vd);

    bool lsh = (opcode == 0xa || opcode == 0xc || opcode == 0xe); // left shift
    uint32_t base = vd.getBit();
    uint32_t imm = (lsh) ? (sh + base) : ((base << 1) - sh);
    uint32_t immh = 1 << size | field(imm, size + 2, 3);
    uint32_t immb = field(imm, 2, 0);

    verifyIncRange(sh, (1 - lsh), (base - lsh), ERR_ILLEGAL_CONST_RANGE);

    uint32_t code =
        concat({F(1, 30), F(U, 29), F(0x1f, 24), F(immh, 19), F(immb, 16),
                F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD scalar x indexed element
  void AdvSimdScXIndElemSz(uint32_t U, uint32_t size, uint32_t opcode,
                           const VRegSc &vd, const VRegSc &vn,
                           const VRegElem &vm) {
    uint32_t bits = vm.getBit();
    uint32_t eidx = vm.getElemIdx();
    uint32_t H = (bits == 16)
                     ? field(eidx, 2, 2)
                     : (bits == 32) ? field(eidx, 1, 1) : field(eidx, 0, 0);
    uint32_t L =
        (bits == 16) ? field(eidx, 1, 1) : (bits == 32) ? field(eidx, 0, 0) : 0;
    uint32_t M = (bits == 16) ? field(eidx, 0, 0) : field(vm.getIdx(), 4, 4);
    uint32_t vmidx = vm.getIdx() & ones(4);

    if (bits == 16)
      verifyIncRange(vm.getIdx(), 0, 15, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(1, 30), F(U, 29), F(0x1f, 24), F(size, 22),
                            F(L, 21), F(M, 20), F(vmidx, 16), F(opcode, 12),
                            F(H, 11), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  void AdvSimdScXIndElem(uint32_t U, uint32_t opcode, const VRegSc &vd,
                         const VRegSc &vn, const VRegElem &vm) {
    uint32_t size = genSize(vm);
    AdvSimdScXIndElemSz(U, size, opcode, vd, vn, vm);
  }

  // Advanced SIMD table lookup
  void AdvSimdTblLkup(uint32_t op2, uint32_t len, uint32_t op,
                      const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
    uint32_t Q = genQ(vd);
    uint32_t code = concat({F(Q, 30), F(0xe, 24), F(op2, 22),
                            F(vm.getIdx(), 16), F(len - 1, 13), F(op, 12),
                            F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD table lookup
  void AdvSimdTblLkup(uint32_t op2, uint32_t op, const VRegVec &vd,
                      const VRegList &vn, const VRegVec &vm) {
    uint32_t Q = genQ(vd);
    uint32_t len = vn.getLen() - 1;
    uint32_t code =
        concat({F(Q, 30), F(0xe, 24), F(op2, 22), F(vm.getIdx(), 16),
                F(len, 13), F(op, 12), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD permute
  void AdvSimdPermute(uint32_t opcode, const VRegVec &vd, const VRegVec &vn,
                      const VRegVec &vm) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd);
    uint32_t code =
        concat({F(Q, 30), F(0xe, 24), F(size, 22), F(vm.getIdx(), 16),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD extract
  void AdvSimdExtract(uint32_t op2, const VRegVec &vd, const VRegVec &vn,
                      const VRegVec &vm, uint32_t index) {
    uint32_t Q = genQ(vd);
    uint32_t imm4 = index & ones(4);

    verifyIncRange(index, 0, 15, ERR_ILLEGAL_CONST_RANGE);
    if (Q == 0)
      verifyCond(
          imm4, [](int64_t x) { return (x >> 3) == 0; },
          ERR_ILLEGAL_CONST_COND);

    uint32_t code =
        concat({F(Q, 30), F(0x2e, 24), F(op2, 22), F(vm.getIdx(), 16),
                F(imm4, 11), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD copy
  void AdvSimdCopyDupElem(uint32_t op, uint32_t imm4, const VRegVec &vd,
                          const VRegElem &vn) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd);
    uint32_t imm5 = (1 << size) | (vn.getElemIdx() << (size + 1));
    uint32_t code =
        concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11),
                F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD copy
  void AdvSimdCopyDupGen(uint32_t op, uint32_t imm4, const VRegVec &vd,
                         const RReg &rn) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd);
    uint32_t imm5 = 1 << size;
    uint32_t code =
        concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11),
                F(3, 10), F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD copy
  void AdvSimdCopyMov(uint32_t op, uint32_t imm4, const RReg &rd,
                      const VRegElem &vn) {
    uint32_t Q = genSf(rd);
    uint32_t size = genSize(vn);
    uint32_t imm5 = ((1 << size) | (vn.getElemIdx() << (size + 1))) & ones(5);
    uint32_t code =
        concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11),
                F(1, 10), F(vn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD copy
  void AdvSimdCopyInsGen(uint32_t op, uint32_t imm4, const VRegElem &vd,
                         const RReg &rn) {
    uint32_t Q = 1;
    uint32_t size = genSize(vd);
    uint32_t imm5 = ((1 << size) | (vd.getElemIdx() << (size + 1))) & ones(5);
    uint32_t code =
        concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11),
                F(1, 10), F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD copy
  void AdvSimdCopyElemIns(uint32_t op, const VRegElem &vd, const VRegElem &vn) {
    uint32_t Q = 1;
    uint32_t size = genSize(vd);
    uint32_t imm5 = ((1 << size) | (vd.getElemIdx() << (size + 1))) & ones(5);
    uint32_t imm4 = (vn.getElemIdx() << size) & ones(4);
    uint32_t code =
        concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11),
                F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD three same (FP16)
  void AdvSimd3SameFp16(uint32_t U, uint32_t a, uint32_t opcode,
                        const VRegVec &vd, const VRegVec &vn,
                        const VRegVec &vm) {
    uint32_t Q = genQ(vd);
    uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(a, 23), F(2, 21),
                            F(vm.getIdx(), 16), F(opcode, 11), F(1, 10),
                            F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD two-register miscellaneous (FP16)
  void AdvSimd2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode,
                           const VRegVec &vd, const VRegVec &vn) {
    uint32_t Q = genQ(vd);
    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xe, 24), F(a, 23), F(0xf, 19),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  void AdvSimd2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode,
                           const VRegVec &vd, const VRegVec &vn, double zero) {
    verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
    AdvSimd2RegMiscFp16(U, a, opcode, vd, vn);
  }

  // Advanced SIMD three same extra
  void AdvSimd3SameExtra(uint32_t U, uint32_t opcode, const VRegVec &vd,
                         const VRegVec &vn, const VRegVec &vm) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd);
    uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22),
                            F(vm.getIdx(), 16), F(1, 15), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD three same extra
  void AdvSimd3SameExtraRotate(uint32_t U, uint32_t op32, const VRegVec &vd,
                               const VRegVec &vn, const VRegVec &vm,
                               uint32_t rotate) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd);
    uint32_t rot = rotate / 90;
    uint32_t opcode =
        (op32 == 2) ? ((op32 << 2) | rot) : ((op32 << 2) | (rot & 0x2));

    if (op32 == 2)
      verifyIncList(rotate, {0, 90, 180, 270}, ERR_ILLEGAL_CONST_VALUE);
    else if (op32 == 3)
      verifyIncList(rotate, {90, 270}, ERR_ILLEGAL_CONST_VALUE);

    uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22),
                            F(vm.getIdx(), 16), F(1, 15), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD two-register miscellaneous
  void AdvSimd2RegMisc(uint32_t U, uint32_t opcode, const VRegVec &vd,
                       const VRegVec &vn) {
    bool sel_vd = (opcode != 0x2 && opcode != 0x6);
    uint32_t Q = (sel_vd) ? genQ(vd) : genQ(vn);
    uint32_t size = (sel_vd) ? genSize(vd) : genSize(vn);
    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD two-register miscellaneous
  void AdvSimd2RegMisc(uint32_t U, uint32_t opcode, const VRegVec &vd,
                       const VRegVec &vn, uint32_t sh) {
    uint32_t Q = genQ(vn);
    uint32_t size = genSize(vn);

    verifyIncList(sh, {vn.getBit()}, ERR_ILLEGAL_CONST_VALUE);

    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD two-register miscellaneous
  void AdvSimd2RegMiscZero(uint32_t U, uint32_t opcode, const VRegVec &vd,
                           const VRegVec &vn, uint32_t zero) {
    verifyIncList(zero, {0}, ERR_ILLEGAL_CONST_VALUE);
    AdvSimd2RegMisc(U, opcode, vd, vn);
  }

  // Advanced SIMD two-register miscellaneous
  void AdvSimd2RegMiscSz(uint32_t U, uint32_t size, uint32_t opcode,
                         const VRegVec &vd, const VRegVec &vn) {
    uint32_t Q = genQ(vd);
    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD two-register miscellaneous
  void AdvSimd2RegMiscSz0x(uint32_t U, uint32_t opcode, const VRegVec &vd,
                           const VRegVec &vn) {
    bool sel_vd = (opcode == 0x17);
    uint32_t Q = (!sel_vd) ? genQ(vd) : genQ(vn);
    uint32_t size = (sel_vd) ? genSize(vd) : genSize(vn);
    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xe, 24), F((size & 1), 22), F(1, 21),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD two-register miscellaneous
  void AdvSimd2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegVec &vd,
                           const VRegVec &vn) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd);
    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  void AdvSimd2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegVec &vd,
                           const VRegVec &vn, double zero) {
    verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
    AdvSimd2RegMiscSz1x(U, opcode, vd, vn);
  }

  // Advanced SIMD across lanes
  void AdvSimdAcrossLanes(uint32_t U, uint32_t opcode, const VRegSc &vd,
                          const VRegVec &vn) {
    uint32_t Q = genQ(vn);
    uint32_t size = genSize(vn);
    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(3, 20),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD across lanes
  void AdvSimdAcrossLanesSz0x(uint32_t U, uint32_t opcode, const VRegSc &vd,
                              const VRegVec &vn) {
    uint32_t Q = genQ(vn);
    uint32_t size = 0;
    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(3, 20),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD across lanes
  void AdvSimdAcrossLanesSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd,
                              const VRegVec &vn) {
    uint32_t Q = genQ(vn);
    uint32_t size = 2;
    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(3, 20),
                F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD three different
  void AdvSimd3Diff(uint32_t U, uint32_t opcode, const VRegVec &vd,
                    const VRegVec &vn, const VRegVec &vm) {
    bool vd_sel = (opcode == 0x4 || opcode == 0x6);
    uint32_t Q = (vd_sel) ? genQ(vd) : genQ(vm);
    uint32_t size = (vd_sel) ? genSize(vd) : genSize(vm);
    uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 12),
                            F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD three same
  void AdvSimd3Same(uint32_t U, uint32_t opcode, const VRegVec &vd,
                    const VRegVec &vn, const VRegVec &vm) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd);
    uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD three same
  void AdvSimd3SameSz0x(uint32_t U, uint32_t opcode, const VRegVec &vd,
                        const VRegVec &vn, const VRegVec &vm) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd) & 1;
    uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD three same
  void AdvSimd3SameSz1x(uint32_t U, uint32_t opcode, const VRegVec &vd,
                        const VRegVec &vn, const VRegVec &vm) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd);
    uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD three same
  void AdvSimd3SameSz(uint32_t U, uint32_t size, uint32_t opcode,
                      const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
    uint32_t Q = genQ(vd);
    uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 11),
                            F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD modified immediate (vector)
  void AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegVec &vd,
                              uint32_t imm, ShMod shmod, uint32_t sh) {
    uint32_t Q = genQ(vd);
    uint32_t crmode = (vd.getBit() == 8)
                          ? 0xe
                          : (vd.getBit() == 16)
                                ? 0x8 | (sh >> 2)
                                : (vd.getBit() == 32 && shmod == LSL)
                                      ? (sh >> 2)
                                      : (vd.getBit() == 32 && shmod == MSL)
                                            ? 0xc | (sh >> 4)
                                            : 0xe;

    if (vd.getBit() == 8)
      verifyIncList(sh, {0}, ERR_ILLEGAL_CONST_VALUE);
    else if (vd.getBit() == 16)
      verifyIncList(sh, {8 * field(crmode, 1, 1)}, ERR_ILLEGAL_CONST_VALUE);
    else if (vd.getBit() == 32 && shmod == LSL)
      verifyIncList(sh, {8 * field(crmode, 2, 1)}, ERR_ILLEGAL_CONST_VALUE);
    else if (vd.getBit() == 32 && shmod == MSL)
      verifyIncList(sh, {8 * field(crmode, 0, 0) + 8}, ERR_ILLEGAL_CONST_VALUE);

    uint32_t abc = field(imm, 7, 5);
    uint32_t defgh = field(imm, 4, 0);
    uint32_t code =
        concat({F(Q, 30), F(op, 29), F(0xf, 24), F(abc, 16), F(crmode, 12),
                F(o2, 11), F(1, 10), F(defgh, 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD modified immediate (scalar)
  void AdvSimdModiImmMoviMvniEnc(uint32_t Q, uint32_t op, uint32_t o2,
                                 const Reg &vd, uint64_t imm) {
    uint32_t crmode = 0xe;
    uint32_t imm8 = compactImm(imm);

    verifyCond(
        imm, [&](uint64_t x) { return isCompact(x, imm8); },
        ERR_ILLEGAL_IMM_COND);

    uint32_t abc = field(imm8, 7, 5);
    uint32_t defgh = field(imm8, 4, 0);
    uint32_t code =
        concat({F(Q, 30), F(op, 29), F(0xf, 24), F(abc, 16), F(crmode, 12),
                F(o2, 11), F(1, 10), F(defgh, 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  void AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegSc &vd,
                              uint64_t imm) {
    uint32_t Q = 0;
    AdvSimdModiImmMoviMvniEnc(Q, op, o2, vd, imm);
  }

  void AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegVec &vd,
                              uint64_t imm) {
    uint32_t Q = genQ(vd);
    AdvSimdModiImmMoviMvniEnc(Q, op, o2, vd, imm);
  }

  // Advanced SIMD modified immediate
  void AdvSimdModiImmOrrBic(uint32_t op, uint32_t o2, const VRegVec &vd,
                            uint32_t imm, ShMod mod, uint32_t sh) {
    uint32_t Q = genQ(vd);
    uint32_t crmode = (vd.getBit() == 16) ? (0x9 | (sh >> 2)) : (1 | (sh >> 2));

    verifyIncList(mod, {LSL}, ERR_ILLEGAL_SHMOD);
    if (vd.getBit() == 16)
      verifyIncList(sh, {8 * field(crmode, 1, 1)}, ERR_ILLEGAL_CONST_VALUE);
    else if (vd.getBit() == 32)
      verifyIncList(sh, {8 * field(crmode, 2, 1)}, ERR_ILLEGAL_CONST_VALUE);

    uint32_t abc = field(imm, 7, 5);
    uint32_t defgh = field(imm, 4, 0);
    uint32_t code =
        concat({F(Q, 30), F(op, 29), F(0xf, 24), F(abc, 16), F(crmode, 12),
                F(o2, 11), F(1, 10), F(defgh, 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD modified immediate
  void AdvSimdModiImmFmov(uint32_t op, uint32_t o2, const VRegVec &vd,
                          double imm) {
    uint32_t Q = genQ(vd);
    uint32_t crmode = 0xf;
    uint32_t imm8 = compactImm(imm, vd.getBit());
    uint32_t abc = field(imm8, 7, 5);
    uint32_t defgh = field(imm8, 4, 0);
    uint32_t code =
        concat({F(Q, 30), F(op, 29), F(0xf, 24), F(abc, 16), F(crmode, 12),
                F(o2, 11), F(1, 10), F(defgh, 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD shift by immediate
  void AdvSimdShImm(uint32_t U, uint32_t opcode, const VRegVec &vd,
                    const VRegVec &vn, uint32_t sh) {
    bool vd_sel = (opcode != 0x14);
    uint32_t Q = (vd_sel) ? genQ(vd) : genQ(vn);
    uint32_t size = (vd_sel) ? genSize(vd) : genSize(vn);

    bool lsh = (opcode == 0xa || opcode == 0xc || opcode == 0xe ||
                opcode == 0x14); // left shift
    uint32_t base = (vd_sel) ? vd.getBit() : vn.getBit();
    uint32_t imm = (lsh) ? (sh + base) : ((base << 1) - sh);
    uint32_t immh = 1 << size | field(imm, size + 2, 3);
    uint32_t immb = field(imm, 2, 0);

    verifyIncRange(sh, (1 - lsh), (base - lsh), ERR_ILLEGAL_CONST_RANGE);

    uint32_t code =
        concat({F(Q, 30), F(U, 29), F(0xf, 24), F(immh, 19), F(immb, 16),
                F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Advanced SIMD vector x indexed element
  void AdvSimdVecXindElemEnc(uint32_t Q, uint32_t U, uint32_t size,
                             uint32_t opcode, const VRegVec &vd,
                             const VRegVec &vn, const VRegElem &vm) {
    bool ucmla = (U == 1 && (opcode & 0x9) == 1);
    uint32_t bits =
        (vm.getBit() == 8) ? 32 : (ucmla) ? vm.getBit() * 2 : vm.getBit();
    uint32_t eidx = vm.getElemIdx();
    uint32_t H = (bits == 16)
                     ? field(eidx, 2, 2)
                     : (bits == 32) ? field(eidx, 1, 1) : field(eidx, 0, 0);
    uint32_t L =
        (bits == 16) ? field(eidx, 1, 1) : (bits == 32) ? field(eidx, 0, 0) : 0;
    uint32_t M = (bits == 16) ? field(eidx, 0, 0) : field(vm.getIdx(), 4, 4);
    uint32_t vmidx = vm.getIdx() & ones(4);

    if (bits == 16)
      verifyIncRange(vm.getIdx(), 0, 15, ERR_ILLEGAL_REG_IDX);
    else
      verifyIncRange(vm.getIdx(), 0, 31, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(Q, 30), F(U, 29), F(0xf, 24), F(size, 22),
                            F(L, 21), F(M, 20), F(vmidx, 16), F(opcode, 12),
                            F(H, 11), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  void AdvSimdVecXindElem(uint32_t U, uint32_t opcode, const VRegVec &vd,
                          const VRegVec &vn, const VRegElem &vm) {
    bool vd_sel = (opcode == 0xe);
    uint32_t Q = (vd_sel) ? genQ(vd) : genQ(vn);
    uint32_t size = (vd_sel) ? genSize(vd) : genSize(vn);
    AdvSimdVecXindElemEnc(Q, U, size, opcode, vd, vn, vm);
  }

  // Advanced SIMD vector x indexed element
  void AdvSimdVecXindElem(uint32_t U, uint32_t opcode, const VRegVec &vd,
                          const VRegVec &vn, const VRegElem &vm,
                          uint32_t rotate) {
    uint32_t Q = genQ(vd);
    uint32_t size = genSize(vd);
    uint32_t rot = rotate / 90;

    verifyIncList(rotate, {0, 90, 180, 270}, ERR_ILLEGAL_CONST_VALUE);

    AdvSimdVecXindElemEnc(Q, U, size, (rot << 1 | opcode), vd, vn, vm);
  }

  void AdvSimdVecXindElemSz(uint32_t U, uint32_t size, uint32_t opcode,
                            const VRegVec &vd, const VRegVec &vn,
                            const VRegElem &vm) {
    uint32_t Q = genQ(vd);
    AdvSimdVecXindElemEnc(Q, U, size, opcode, vd, vn, vm);
  }

  // Cryptographic three-register, imm2
  void Crypto3RegImm2(uint32_t opcode, const VRegVec &vd, const VRegVec &vn,
                      const VRegElem &vm) {
    uint32_t imm2 = vm.getElemIdx();
    uint32_t code =
        concat({F(0x672, 21), F(vm.getIdx(), 16), F(2, 14), F(imm2, 12),
                F(opcode, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Cryptographic three-register SHA 512
  void Crypto3RegSHA512(uint32_t O, uint32_t opcode, const VRegSc &vd,
                        const VRegSc &vn, const VRegVec &vm) {
    uint32_t code =
        concat({F(0x673, 21), F(vm.getIdx(), 16), F(1, 15), F(O, 14),
                F(opcode, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Cryptographic three-register SHA 512
  void Crypto3RegSHA512(uint32_t O, uint32_t opcode, const VRegVec &vd,
                        const VRegVec &vn, const VRegVec &vm) {
    uint32_t code =
        concat({F(0x673, 21), F(vm.getIdx(), 16), F(1, 15), F(O, 14),
                F(opcode, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // XAR
  void CryptoSHA(const VRegVec &vd, const VRegVec &vn, const VRegVec &vm,
                 uint32_t imm6) {
    verifyIncRange(imm6, 0, ones(6), ERR_ILLEGAL_IMM_RANGE);
    uint32_t code = concat({F(0x674, 21), F(vm.getIdx(), 16), F(imm6, 10),
                            F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Cryptographic four-register
  void Crypto4Reg(uint32_t Op0, const VRegVec &vd, const VRegVec &vn,
                  const VRegVec &vm, const VRegVec &va) {
    uint32_t code =
        concat({F(0x19c, 23), F(Op0, 21), F(vm.getIdx(), 16),
                F(va.getIdx(), 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Cryptographic two-register SHA512
  void Crypto2RegSHA512(uint32_t opcode, const VRegVec &vd, const VRegVec &vn) {
    uint32_t code = concat(
        {F(0xcec08, 12), F(opcode, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // conversion between floating-point and fixed-point
  void ConversionFpFix(uint32_t S, uint32_t type, uint32_t rmode,
                       uint32_t opcode, const VRegSc &vd, const RReg &rn,
                       uint32_t fbits) {
    uint32_t sf = genSf(rn);
    uint32_t scale = 64 - fbits;

    verifyIncRange(fbits, 1, (32 << sf), ERR_ILLEGAL_CONST_RANGE);

    uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(rmode, 19), F(opcode, 16), F(scale, 10),
                            F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // conversion between floating-point and fixed-point
  void ConversionFpFix(uint32_t S, uint32_t type, uint32_t rmode,
                       uint32_t opcode, const RReg &rd, const VRegSc &vn,
                       uint32_t fbits) {
    uint32_t sf = genSf(rd);
    uint32_t scale = 64 - fbits;

    verifyIncRange(fbits, 1, (32 << sf), ERR_ILLEGAL_CONST_RANGE);

    uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(rmode, 19), F(opcode, 16), F(scale, 10),
                            F(vn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // conversion between floating-point and integer
  void ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode,
                       uint32_t opcode, const RReg &rd, const VRegSc &vn) {
    uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(1, 21), F(rmode, 19), F(opcode, 16),
                            F(vn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // conversion between floating-point and integer
  void ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode,
                       uint32_t opcode, const VRegSc &vd, const RReg &rn) {
    uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(1, 21), F(rmode, 19), F(opcode, 16),
                            F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // conversion between floating-point and integer
  void ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode,
                       uint32_t opcode, const RReg &rd, const VRegElem &vn) {
    uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(1, 21), F(rmode, 19), F(opcode, 16),
                            F(vn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // conversion between floating-point and integer
  void ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode,
                       uint32_t opcode, const VRegElem &vd, const RReg &rn) {
    uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(1, 21), F(rmode, 19), F(opcode, 16),
                            F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Floating-piont data-processing (1 source)
  void FpDataProc1Reg(uint32_t M, uint32_t S, uint32_t type, uint32_t opcode,
                      const VRegSc &vd, const VRegSc &vn) {
    uint32_t code =
        concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21),
                F(opcode, 15), F(1, 14), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Floating-piont compare
  void FpComp(uint32_t M, uint32_t S, uint32_t type, uint32_t op,
              uint32_t opcode2, const VRegSc &vn, const VRegSc &vm) {
    uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(op, 14), F(1, 13),
                            F(vn.getIdx(), 5), F(opcode2, 0)});
    dw(code);
  }

  // Floating-piont compare
  void FpComp(uint32_t M, uint32_t S, uint32_t type, uint32_t op,
              uint32_t opcode2, const VRegSc &vn, double imm) {
    verifyIncList(std::lround(imm), {0}, ERR_ILLEGAL_CONST_VALUE);
    uint32_t code =
        concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21),
                F(op, 14), F(1, 13), F(vn.getIdx(), 5), F(opcode2, 0)});
    dw(code);
  }

  // Floating-piont immediate
  void FpImm(uint32_t M, uint32_t S, uint32_t type, const VRegSc &vd,
             double imm) {
    uint32_t imm8 = compactImm(imm, vd.getBit());
    uint32_t code =
        concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21),
                F(imm8, 13), F(1, 12), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Floating-piont conditional compare
  void FpCondComp(uint32_t M, uint32_t S, uint32_t type, uint32_t op,
                  const VRegSc &vn, const VRegSc &vm, uint32_t nzcv,
                  Cond cond) {
    uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(cond, 12), F(1, 10),
                            F(vn.getIdx(), 5), F(op, 4), F(nzcv, 0)});
    dw(code);
  }

  // Floating-piont data-processing (2 source)
  void FpDataProc2Reg(uint32_t M, uint32_t S, uint32_t type, uint32_t opcode,
                      const VRegSc &vd, const VRegSc &vn, const VRegSc &vm) {
    uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(opcode, 12),
                            F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Floating-piont conditional select
  void FpCondSel(uint32_t M, uint32_t S, uint32_t type, const VRegSc &vd,
                 const VRegSc &vn, const VRegSc &vm, Cond cond) {
    uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22),
                            F(1, 21), F(vm.getIdx(), 16), F(cond, 12), F(3, 10),
                            F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // Floating-piont data-processing (3 source)
  void FpDataProc3Reg(uint32_t M, uint32_t S, uint32_t type, uint32_t o1,
                      uint32_t o0, const VRegSc &vd, const VRegSc &vn,
                      const VRegSc &vm, const VRegSc &va) {
    uint32_t code =
        concat({F(M, 31), F(S, 29), F(0x1f, 24), F(type, 22), F(o1, 21),
                F(vm.getIdx(), 16), F(o0, 15), F(va.getIdx(), 10),
                F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // ########################### System instruction
  // #################################
  // Instruction cache maintenance
  void InstCache(IcOp icop, const XReg &xt) {
    uint32_t code =
        concat({F(0xd5, 24), F(1, 19), F(icop, 5), F(xt.getIdx(), 0)});
    dw(code);
  }

  // Data cache maintenance
  void DataCache(DcOp dcop, const XReg &xt) {
    uint32_t code =
        concat({F(0xd5, 24), F(1, 19), F(dcop, 5), F(xt.getIdx(), 0)});
    dw(code);
  }

  // Addresss Translate
  void AddressTrans(AtOp atop, const XReg &xt) {
    uint32_t code =
        concat({F(0xd5, 24), F(1, 19), F(atop, 5), F(xt.getIdx(), 0)});
    dw(code);
  }

  // TLB Invaidate operation
  void TLBInv(TlbiOp tlbiop, const XReg &xt) {
    uint32_t code =
        concat({F(0xd5, 24), F(1, 19), F(tlbiop, 5), F(xt.getIdx(), 0)});
    dw(code);
  }

  // ################################### SVE
  // #########################################

  // SVE Integer Binary Arithmetic - Predicated Group
  void SveIntBinArPred(uint32_t opc, uint32_t type, const _ZReg &zd,
                       const _PReg &pg, const _ZReg &zn) {
    uint32_t size = genSize(zd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(type, 19), F(opc, 16),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE bitwize Logical Operation (predicated)
  void SveBitwiseLOpPred(uint32_t opc, const _ZReg &zd, const _PReg &pg,
                         const _ZReg &zn) {
    SveIntBinArPred(opc, 3, zd, pg, zn);
  }

  // SVE Integer add/subtract vectors (predicated)
  void SveIntAddSubVecPred(uint32_t opc, const _ZReg &zd, const _PReg &pg,
                           const _ZReg &zn) {
    SveIntBinArPred(opc, 0, zd, pg, zn);
  }

  // SVE Integer min/max/diffrence (predicated)
  void SveIntMinMaxDiffPred(uint32_t opc, uint32_t U, const _ZReg &zd,
                            const _PReg &pg, const _ZReg &zn) {
    SveIntBinArPred((opc << 1 | U), 1, zd, pg, zn);
  }

  // SVE Integer multiply/divide vectors (predicated)
  void SveIntMultDivVecPred(uint32_t opc, uint32_t U, const _ZReg &zd,
                            const _PReg &pg, const _ZReg &zn) {
    SveIntBinArPred((opc << 1 | U), 2, zd, pg, zn);
  }

  // SVE Integer Reduction Group
  void SveIntReduction(uint32_t opc, uint32_t type, const Reg &rd,
                       const _PReg &pg, const Reg &rn) {
    uint32_t size = genSize(rn);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(type, 19), F(opc, 16), F(1, 13),
                F(pg.getIdx(), 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // SVE bitwise logical reduction (predicated)
  void SveBitwiseLReductPred(uint32_t opc, const VRegSc &vd, const _PReg &pg,
                             const _ZReg &zn) {
    SveIntReduction(opc, 3, vd, pg, zn);
  }

  // SVE constructive prefix (predicated)
  void SveConstPrefPred(uint32_t opc, const _ZReg &zd, const _PReg &pg,
                        const _ZReg &zn) {
    SveIntReduction((opc << 1 | (static_cast<uint32_t>(pg.isM()))), 2, zd, pg,
                    zn);
  }

  // SVE integer add reduction (predicated)
  void SveIntAddReductPred(uint32_t opc, uint32_t U, const VRegSc &vd,
                           const _PReg &pg, const _ZReg &zn) {
    SveIntReduction((opc << 1 | U), 0, vd, pg, zn);
  }

  // SVE integer min/max reduction (predicated)
  void SveIntMinMaxReductPred(uint32_t opc, uint32_t U, const VRegSc &vd,
                              const _PReg &pg, const _ZReg &zn) {
    SveIntReduction((opc << 1 | U), 1, vd, pg, zn);
  }

  // SVE Bitwise Shift - Predicate Group
  void SveBitShPred(uint32_t opc, uint32_t type, const _ZReg &zdn,
                    const _PReg &pg, const _ZReg &zm) {
    uint32_t size = genSize(zdn);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(type, 19), F(opc, 16), F(4, 13),
                F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE bitwise shift by immediate (predicated)
  void SveBitwiseShByImmPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg,
                             uint32_t amount) {
    bool lsl = (opc == 3);
    uint32_t size = genSize(zdn);
    uint32_t imm =
        (lsl) ? (amount + zdn.getBit()) : (2 * zdn.getBit() - amount);
    uint32_t imm3 = imm & ones(3);
    uint32_t tsz = (1 << size) | field(imm, size + 2, 3);
    uint32_t tszh = field(tsz, 3, 2);
    uint32_t tszl = field(tsz, 1, 0);

    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(amount, (1 - lsl), (zdn.getBit() - lsl),
                   ERR_ILLEGAL_CONST_RANGE);

    uint32_t code = concat({F(0x4, 24), F(tszh, 22), F(0, 19), F(opc, 16),
                            F(4, 13), F(pg.getIdx(), 10), F(tszl, 8),
                            F(imm3, 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE bitwise shift by vector (predicated)
  void SveBitwiseShVecPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg,
                           const _ZReg &zm) {
    SveBitShPred(opc, 2, zdn, pg, zm);
  }

  // SVE bitwise shift by wide elements (predicated)
  void SveBitwiseShWElemPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg,
                             const _ZReg &zm) {
    SveBitShPred(opc, 3, zdn, pg, zm);
  }

  // SVE Integer Unary Arithmetic - Predicated Group
  void SveIntUnaryArPred(uint32_t opc, uint32_t type, const _ZReg &zd,
                         const _PReg &pg, const _ZReg &zn) {
    uint32_t size = genSize(zd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(type, 19), F(opc, 16), F(5, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE bitwise unary operations (predicated)
  void SveBitwiseUnaryOpPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg,
                             const _ZReg &zm) {
    SveIntUnaryArPred(opc, 3, zdn, pg, zm);
  }

  // SVE integer unary operations (predicated)
  void SveIntUnaryOpPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg,
                         const _ZReg &zm) {
    SveIntUnaryArPred(opc, 2, zdn, pg, zm);
  }

  // SVE integer multiply-accumulate writing addend (predicated)
  void SveIntMultAccumPred(uint32_t opc, const _ZReg &zda, const _PReg &pg,
                           const _ZReg &zn, const _ZReg &zm) {
    uint32_t size = genSize(zda);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x4, 24), F(size, 22), F(zm.getIdx(), 16),
                            F(1, 14), F(opc, 13), F(pg.getIdx(), 10),
                            F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
    dw(code);
  }

  // SVE integer multiply-add writeing multiplicand (predicated)
  void SveIntMultAddPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg,
                         const _ZReg &zm, const _ZReg &za) {
    uint32_t size = genSize(zdn);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x4, 24), F(size, 22), F(zm.getIdx(), 16),
                            F(3, 14), F(opc, 13), F(pg.getIdx(), 10),
                            F(za.getIdx(), 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE integer add/subtract vectors (unpredicated)
  void SveIntAddSubUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn,
                          const _ZReg &zm) {
    uint32_t size = genSize(zd);
    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(1, 21), F(zm.getIdx(), 16),
                F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE bitwise logical operations (unpredicated)
  void SveBitwiseLOpUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn,
                           const _ZReg &zm) {
    uint32_t code =
        concat({F(0x4, 24), F(opc, 22), F(1, 21), F(zm.getIdx(), 16),
                F(0xc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE index generation (immediate start, immediate increment)
  void SveIndexGenImmImmInc(const _ZReg &zd, int32_t imm1, int32_t imm2) {
    uint32_t size = genSize(zd);
    uint32_t imm5b = imm2 & ones(5);
    uint32_t imm5 = imm1 & ones(5);

    verifyIncRange(imm1, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(imm2, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t code = concat({F(0x4, 24), F(size, 22), F(1, 21), F(imm5b, 16),
                            F(0x10, 10), F(imm5, 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE index generation (immediate start, register increment)
  void SveIndexGenImmRegInc(const _ZReg &zd, int32_t imm, const RReg &rm) {
    uint32_t size = genSize(zd);
    uint32_t imm5 = imm & ones(5);

    verifyIncRange(imm, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(1, 21), F(rm.getIdx(), 16),
                F(0x12, 10), F(imm5, 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE index generation (register start, immediate increment)
  void SveIndexGenRegImmInc(const _ZReg &zd, const RReg &rn, int32_t imm) {
    uint32_t size = genSize(zd);
    uint32_t imm5 = imm & ones(5);

    verifyIncRange(imm, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t code = concat({F(0x4, 24), F(size, 22), F(1, 21), F(imm5, 16),
                            F(0x11, 10), F(rn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE index generation (register start, register increment)
  void SveIndexGenRegRegInc(const _ZReg &zd, const RReg &rn, const RReg &rm) {
    uint32_t size = genSize(zd);
    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(1, 21), F(rm.getIdx(), 16),
                F(0x13, 10), F(rn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE stack frame adjustment
  void SveStackFrameAdjust(uint32_t op, const XReg &xd, const XReg &xn,
                           int32_t imm) {
    uint32_t imm6 = imm & ones(6);

    verifyIncRange(imm, -32, 31, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t code = concat({F(0x8, 23), F(op, 22), F(1, 21), F(xn.getIdx(), 16),
                            F(0xa, 11), F(imm6, 5), F(xd.getIdx(), 0)});
    dw(code);
  }

  // SVE stack frame size
  void SveStackFrameSize(uint32_t op, uint32_t opc2, const XReg &xd,
                         int32_t imm) {
    uint32_t imm6 = imm & ones(6);

    verifyIncRange(imm, -32, 31, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t code = concat({F(0x9, 23), F(op, 22), F(1, 21), F(opc2, 16),
                            F(0xa, 11), F(imm6, 5), F(xd.getIdx(), 0)});
    dw(code);
  }

  // SVE bitwise shift by immediate (unpredicated)
  void SveBitwiseShByImmUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn,
                               uint32_t amount) {
    bool lsl = (opc == 3);
    uint32_t size = genSize(zd);
    uint32_t imm = (lsl) ? (amount + zd.getBit()) : (2 * zd.getBit() - amount);
    uint32_t imm3 = imm & ones(3);
    uint32_t tsz = (1 << size) | field(imm, size + 2, 3);
    uint32_t tszh = field(tsz, 3, 2);
    uint32_t tszl = field(tsz, 1, 0);

    verifyIncRange(amount, (1 - lsl), (zd.getBit() - lsl),
                   ERR_ILLEGAL_CONST_RANGE);

    uint32_t code =
        concat({F(0x4, 24), F(tszh, 22), F(1, 21), F(tszl, 19), F(imm3, 16),
                F(0x9, 12), F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE bitwise shift by wide elements (unpredicated)
  void SveBitwiseShByWideElemUnPred(uint32_t opc, const _ZReg &zd,
                                    const _ZReg &zn, const _ZReg &zm) {
    uint32_t size = genSize(zd);
    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(1, 21), F(zm.getIdx(), 16),
                F(0x8, 12), F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE address generation
  void SveAddressGen(const _ZReg &zd, const AdrVec &adr) {
    ShMod mod = adr.getMod();
    uint32_t sh = adr.getSh();
    uint32_t opc = 2 | (genSize(zd) & 0x1);
    uint32_t msz = sh & ones(2);

    verifyIncList(mod, {LSL, NONE}, ERR_ILLEGAL_SHMOD);
    verifyIncRange(sh, 0, 3, ERR_ILLEGAL_CONST_RANGE);

    uint32_t code = concat({F(0x4, 24), F(opc, 22), F(1, 21),
                            F(adr.getZm().getIdx(), 16), F(0xa, 12), F(msz, 10),
                            F(adr.getZn().getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE address generation
  void SveAddressGen(const _ZReg &zd, const AdrVecU &adr) {
    ExtMod mod = adr.getMod();
    uint32_t sh = adr.getSh();
    uint32_t opc = (mod == SXTW) ? 0 : 1;
    uint32_t msz = sh & ones(2);

    verifyIncList(mod, {UXTW, SXTW}, ERR_ILLEGAL_EXTMOD);
    verifyIncRange(sh, 0, 3, ERR_ILLEGAL_CONST_RANGE);

    uint32_t code = concat({F(0x4, 24), F(opc, 22), F(1, 21),
                            F(adr.getZm().getIdx(), 16), F(0xa, 12), F(msz, 10),
                            F(adr.getZn().getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE Integer Misc - Unpredicated Group
  void SveIntMiscUnpred(uint32_t size, uint32_t opc, uint32_t type,
                        const _ZReg &zd, const _ZReg &zn) {
    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(1, 21), F(opc, 16), F(0xb, 12),
                F(type, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE constructive prefix (unpredicated)
  void SveConstPrefUnpred(uint32_t opc, uint32_t opc2, const _ZReg &zd,
                          const _ZReg &zn) {
    SveIntMiscUnpred(opc, opc2, 3, zd, zn);
  }

  // SVE floating-point exponential accelerator
  void SveFpExpAccel(uint32_t opc, const _ZReg &zd, const _ZReg &zn) {
    uint32_t size = genSize(zd);
    SveIntMiscUnpred(size, opc, 2, zd, zn);
  }

  // SVE floating-point trig select coefficient
  void SveFpTrigSelCoef(uint32_t opc, const _ZReg &zd, const _ZReg &zn,
                        const _ZReg &zm) {
    uint32_t size = genSize(zd);
    SveIntMiscUnpred(size, zm.getIdx(), opc, zd, zn);
  }

  // SVE Element Count Group
  void SveElemCountGrp(uint32_t size, uint32_t op, uint32_t type1,
                       uint32_t type2, const Reg &rd, Pattern pat, ExtMod mod,
                       uint32_t imm) {
    uint32_t imm4 = (imm - 1) & ones(4);
    verifyIncList(mod, {MUL}, ERR_ILLEGAL_EXTMOD);
    verifyIncRange(imm, 1, 16, ERR_ILLEGAL_IMM_RANGE);
    uint32_t code =
        concat({F(0x4, 24), F(size, 22), F(type1, 20), F(imm4, 16),
                F(type2, 11), F(op, 10), F(pat, 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // SVE element count
  void SveElemCount(uint32_t size, uint32_t op, const XReg &xd, Pattern pat,
                    ExtMod mod, uint32_t imm) {
    SveElemCountGrp(size, op, 2, 0x1c, xd, pat, mod, imm);
  }

  // SVE inc/dec register by element count
  void SveIncDecRegByElemCount(uint32_t size, uint32_t D, const XReg &xd,
                               Pattern pat, ExtMod mod, uint32_t imm) {
    SveElemCountGrp(size, D, 3, 0x1c, xd, pat, mod, imm);
  }

  // SVE inc/dec vector by element count
  void SveIncDecVecByElemCount(uint32_t size, uint32_t D, const _ZReg &zd,
                               Pattern pat, ExtMod mod, uint32_t imm) {
    SveElemCountGrp(size, D, 3, 0x18, zd, pat, mod, imm);
  }

  // SVE saturating inc/dec register by element count
  void SveSatuIncDecRegByElemCount(uint32_t size, uint32_t D, uint32_t U,
                                   const RReg &rdn, Pattern pat, ExtMod mod,
                                   uint32_t imm) {
    uint32_t sf = genSf(rdn);
    SveElemCountGrp(size, U, (2 | sf), (0x1e | D), rdn, pat, mod, imm);
  }

  // SVE saturating inc/dec vector by element count
  void SveSatuIncDecVecByElemCount(uint32_t size, uint32_t D, uint32_t U,
                                   const _ZReg &zdn, Pattern pat, ExtMod mod,
                                   uint32_t imm) {
    SveElemCountGrp(size, U, 2, (0x18 | D), zdn, pat, mod, imm);
  }

  // SVE Bitwise Immeidate Group
  void SveBitwiseImm(uint32_t opc, const _ZReg &zd, uint64_t imm) {
    uint32_t imm13 = genNImmrImms(imm, zd.getBit());
    uint32_t code =
        concat({F(0x5, 24), F(opc, 22), F(imm13, 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE bitwise logical with immediate (unpredicated)
  void SveBitwiseLogicalImmUnpred(uint32_t opc, const _ZReg &zdn,
                                  uint64_t imm) {
    SveBitwiseImm(opc, zdn, imm);
  }

  // SVE broadcast bitmask immediate
  void SveBcBitmaskImm(const _ZReg &zdn, uint64_t imm) {
    SveBitwiseImm(3, zdn, imm);
  }

  // SVE copy floating-point immediate (predicated)
  void SveCopyFpImmPred(const _ZReg &zd, const _PReg &pg, double imm) {
    uint32_t size = genSize(zd);
    uint32_t imm8 = compactImm(imm, zd.getBit());
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(1, 20), F(pg.getIdx(), 16), F(6, 13),
                F(imm8, 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE copy integer immediate (predicated)
  void SveCopyIntImmPred(const _ZReg &zd, const _PReg &pg, uint32_t imm,
                         ShMod mod, uint32_t sh) {
    verifyIncList(mod, {LSL}, ERR_ILLEGAL_SHMOD);
    verifyIncList(sh, {0, 8}, ERR_ILLEGAL_CONST_VALUE);
    uint32_t size = genSize(zd);
    uint32_t imm8 = imm & ones(8);
    uint32_t type = (pg.isM() << 1) | (sh == 8);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(1, 20), F(pg.getIdx(), 16),
                F(type, 13), F(imm8, 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE extract vector (immediate offset)
  void SveExtVec(const _ZReg &zdn, const _ZReg &zm, uint32_t imm) {
    uint32_t imm8h = field(imm, 7, 3);
    uint32_t imm8l = field(imm, 2, 0);
    verifyIncRange(imm, 0, 255, ERR_ILLEGAL_IMM_RANGE);
    uint32_t code = concat({F(0x5, 24), F(1, 21), F(imm8h, 16), F(imm8l, 10),
                            F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE Permute Vector - Unpredicate Group
  void SvePerVecUnpred(uint32_t size, uint32_t type1, uint32_t type2,
                       const _ZReg &zd, const Reg &rn) {
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(1, 21), F(type1, 16), F(type2, 10),
                F(rn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE broadcast general register
  void SveBcGeneralReg(const _ZReg &zd, const RReg &rn) {
    uint32_t size = genSize(zd);
    SvePerVecUnpred(size, 0, 0xe, zd, rn);
  }

  // SVE broadcast indexed element
  void SveBcIndexedElem(const _ZReg &zd, const ZRegElem &zn) {
    uint32_t eidx = zn.getElemIdx();
    uint32_t pos = static_cast<uint32_t>(std::log2(zn.getBit()) - 2);
    uint32_t imm = (eidx << pos) | (1 << (pos - 1));
    uint32_t imm2 = field(imm, 6, 5);
    uint32_t tsz = field(imm, 4, 0);

    if (zd.getBit() == 128)
      verifyIncList(field(tsz, 4, 0), {0x10}, ERR_ILLEGAL_IMM_COND);
    else if (zd.getBit() == 64)
      verifyIncList(field(tsz, 3, 0), {0x8}, ERR_ILLEGAL_IMM_COND);
    else if (zd.getBit() == 32)
      verifyIncList(field(tsz, 2, 0), {0x4}, ERR_ILLEGAL_IMM_COND);
    else if (zd.getBit() == 16)
      verifyIncList(field(tsz, 1, 0), {0x2}, ERR_ILLEGAL_IMM_COND);
    else if (zd.getBit() == 8)
      verifyIncList(field(tsz, 0, 0), {0x1}, ERR_ILLEGAL_IMM_COND);

    SvePerVecUnpred(imm2, tsz, 0x8, zd, zn);
  }

  // SVE insert SIMD&FP scalar register
  void SveInsSimdFpSclarReg(const _ZReg &zdn, const VRegSc &vm) {
    uint32_t size = genSize(zdn);
    SvePerVecUnpred(size, 0x14, 0xe, zdn, vm);
  }

  // SVE insert general register
  void SveInsGeneralReg(const _ZReg &zdn, const RReg &rm) {
    uint32_t size = genSize(zdn);
    SvePerVecUnpred(size, 0x4, 0xe, zdn, rm);
  }

  // SVE reverse vector elements
  void SveRevVecElem(const _ZReg &zd, const _ZReg &zn) {
    uint32_t size = genSize(zd);
    SvePerVecUnpred(size, 0x18, 0xe, zd, zn);
  }

  // SVE table lookup
  void SveTableLookup(const _ZReg &zd, const _ZReg &zn, const _ZReg &zm) {
    uint32_t size = genSize(zd);
    SvePerVecUnpred(size, zm.getIdx(), 0xc, zd, zn);
  }

  // SVE unpack vector elements
  void SveUnpackVecElem(uint32_t U, uint32_t H, const _ZReg &zd,
                        const _ZReg &zn) {
    uint32_t size = genSize(zd);
    SvePerVecUnpred(size, (0x10 | (U << 1) | H), 0xe, zd, zn);
  }

  // SVE permute predicate elements
  void SvePermutePredElem(uint32_t opc, uint32_t H, const _PReg &pd,
                          const _PReg &pn, const _PReg &pm) {
    uint32_t size = genSize(pd);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(1, 21), F(pm.getIdx(), 16), F(2, 13),
                F(opc, 11), F(H, 10), F(pn.getIdx(), 5), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE reverse predicate elements
  void SveRevPredElem(const _PReg &pd, const _PReg &pn) {
    uint32_t size = genSize(pd);
    uint32_t code = concat({F(0x5, 24), F(size, 22), F(0xd1, 14),
                            F(pn.getIdx(), 5), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE unpack predicate elements
  void SveUnpackPredElem(uint32_t H, const _PReg &pd, const _PReg &pn) {
    uint32_t code = concat({F(0x5, 24), F(3, 20), F(H, 16), F(1, 14),
                            F(pn.getIdx(), 5), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE permute vector elements
  void SvePermuteVecElem(uint32_t opc, const _ZReg &zd, const _ZReg &zn,
                         const _ZReg &zm) {
    uint32_t size = genSize(zd);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(1, 21), F(zm.getIdx(), 16), F(3, 13),
                F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE compress active elements
  void SveCompressActElem(const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
    uint32_t size = genSize(zd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(1, 21), F(0xc, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE conditionally broaccast element to vector
  void SveCondBcElemToVec(uint32_t B, const _ZReg &zdn, const _PReg &pg,
                          const _ZReg &zm) {
    uint32_t size = genSize(zdn);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(0x14, 17), F(B, 16), F(0x4, 13),
                F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE conditionally extract element to SIMD&FP scalar
  void SveCondExtElemToSimdFpScalar(uint32_t B, const VRegSc &vdn,
                                    const _PReg &pg, const _ZReg &zm) {
    uint32_t size = genSize(vdn);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(0x15, 17), F(B, 16), F(0x4, 13),
                F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(vdn.getIdx(), 0)});
    dw(code);
  }

  // SVE conditionally extract element to general Reg
  void SveCondExtElemToGeneralReg(uint32_t B, const RReg &rdn, const _PReg &pg,
                                  const _ZReg &zm) {
    uint32_t size = genSize(zm);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(0x18, 17), F(B, 16), F(0x5, 13),
                F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(rdn.getIdx(), 0)});
    dw(code);
  }

  // SVE copy SIMD&FP scalar register to vector (predicated)
  void SveCopySimdFpScalarToVecPred(const _ZReg &zd, const _PReg &pg,
                                    const VRegSc &vn) {
    uint32_t size = genSize(zd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(0x10, 17), F(0x4, 13),
                F(pg.getIdx(), 10), F(vn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE copy general register to vector (predicated)
  void SveCopyGeneralRegToVecPred(const _ZReg &zd, const _PReg &pg,
                                  const RReg &rn) {
    uint32_t size = genSize(zd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(0x14, 17), F(0x5, 13),
                F(pg.getIdx(), 10), F(rn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE extract element to SIMD&FP scalar register
  void SveExtElemToSimdFpScalar(uint32_t B, const VRegSc &vd, const _PReg &pg,
                                const _ZReg &zn) {
    uint32_t size = genSize(vd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(0x11, 17), F(B, 16), F(0x4, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // SVE extract element to general register
  void SveExtElemToGeneralReg(uint32_t B, const RReg &rd, const _PReg &pg,
                              const _ZReg &zn) {
    uint32_t size = genSize(zn);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(0x10, 17), F(B, 16), F(0x5, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // SVE reverse within elements
  void SveRevWithinElem(uint32_t opc, const _ZReg &zd, const _PReg &pg,
                        const _ZReg &zn) {
    uint32_t size = genSize(zd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(0x9, 18), F(opc, 16), F(0x4, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE vector splice
  void SveSelVecSplice(const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
    uint32_t size = genSize(zd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x5, 24), F(size, 22), F(0xb, 18), F(0x4, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE select vector elements (predicated)
  void SveSelVecElemPred(const _ZReg &zd, const _PReg &pg, const _ZReg &zn,
                         const _ZReg &zm) {
    uint32_t size = genSize(zd);
    uint32_t code = concat({F(0x5, 24), F(size, 22), F(1, 21),
                            F(zm.getIdx(), 16), F(0x3, 14), F(pg.getIdx(), 10),
                            F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE Integer Compare - Vector Group
  void SveIntCompVecGrp(uint32_t opc, uint32_t ne, const _PReg &pd,
                        const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
    uint32_t size = genSize(pd);
    uint32_t code = concat({F(0x24, 24), F(size, 22), F(zm.getIdx(), 16),
                            F(opc, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5),
                            F(ne, 4), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE integer compare vectors
  void SveIntCompVec(uint32_t op, uint32_t o2, uint32_t ne, const _PReg &pd,
                     const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
    uint32_t opc = (op << 2) | o2;
    SveIntCompVecGrp(opc, ne, pd, pg, zn, zm);
  }

  // SVE integer compare with wide elements
  void SveIntCompWideElem(uint32_t op, uint32_t o2, uint32_t ne,
                          const _PReg &pd, const _PReg &pg, const _ZReg &zn,
                          const _ZReg &zm) {
    uint32_t opc = (op << 2) | 2 | o2;
    SveIntCompVecGrp(opc, ne, pd, pg, zn, zm);
  }

  // SVE integer compare with unsigned immediate
  void SveIntCompUImm(uint32_t lt, uint32_t ne, const _PReg &pd,
                      const _PReg &pg, const _ZReg &zn, uint32_t imm) {
    uint32_t size = genSize(pd);
    uint32_t imm7 = imm & ones(7);
    verifyIncRange(imm, 0, 127, ERR_ILLEGAL_IMM_RANGE);
    uint32_t code = concat({F(0x24, 24), F(size, 22), F(1, 21), F(imm7, 14),
                            F(lt, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5),
                            F(ne, 4), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE predicate logical operations
  void SvePredLOp(uint32_t op, uint32_t S, uint32_t o2, uint32_t o3,
                  const _PReg &pd, const _PReg &pg, const _PReg &pn,
                  const _PReg &pm) {
    uint32_t code =
        concat({F(0x25, 24), F(op, 23), F(S, 22), F(pm.getIdx(), 16), F(1, 14),
                F(pg.getIdx(), 10), F(o2, 9), F(pn.getIdx(), 5), F(o3, 4),
                F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE propagate break from previous partition
  void SvePropagateBreakPrevPtn(uint32_t op, uint32_t S, uint32_t B,
                                const _PReg &pd, const _PReg &pg,
                                const _PReg &pn, const _PReg &pm) {
    uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22),
                            F(pm.getIdx(), 16), F(3, 14), F(pg.getIdx(), 10),
                            F(pn.getIdx(), 5), F(B, 4), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE partition break condition
  void SvePartitionBreakCond(uint32_t B, uint32_t S, const _PReg &pd,
                             const _PReg &pg, const _PReg &pn) {
    uint32_t M = (S == 1) ? 0 : pg.isM();
    uint32_t code = concat({F(0x25, 24), F(B, 23), F(S, 22), F(2, 19), F(1, 14),
                            F(pg.getIdx(), 10), F(pn.getIdx(), 5), F(M, 4),
                            F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE propagate break to next partition
  void SvePropagateBreakNextPart(uint32_t S, const _PReg &pdm, const _PReg &pg,
                                 const _PReg &pn) {
    uint32_t code =
        concat({F(0x25, 24), F(S, 22), F(3, 19), F(1, 14), F(pg.getIdx(), 10),
                F(pn.getIdx(), 5), F(pdm.getIdx(), 0)});
    dw(code);
  }

  // SVE predicate first active
  void SvePredFirstAct(uint32_t op, uint32_t S, const _PReg &pdn,
                       const _PReg &pg) {
    uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(3, 19),
                            F(3, 14), F(pg.getIdx(), 5), F(pdn.getIdx(), 0)});
    dw(code);
  }

  // SVE predicate initialize
  void SvePredInit(uint32_t S, const _PReg &pd, Pattern pat) {
    uint32_t size = genSize(pd);
    uint32_t code = concat({F(0x25, 24), F(size, 22), F(3, 19), F(S, 16),
                            F(7, 13), F(pat, 5), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE predicate next active
  void SvePredNextAct(const _PReg &pdn, const _PReg &pg) {
    uint32_t size = genSize(pdn);
    uint32_t code = concat({F(0x25, 24), F(size, 22), F(3, 19), F(0xe, 13),
                            F(1, 10), F(pg.getIdx(), 5), F(pdn.getIdx(), 0)});
    dw(code);
  }

  // SVE predicate read from FFR (predicate)
  void SvePredReadFFRPred(uint32_t op, uint32_t S, const _PReg &pd,
                          const _PReg &pg) {
    uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(3, 19),
                            F(0xf, 12), F(pg.getIdx(), 5), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE predicate read from FFR (unpredicate)
  void SvePredReadFFRUnpred(uint32_t op, uint32_t S, const _PReg &pd) {
    uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(3, 19),
                            F(0x1f, 12), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE predicate test
  void SvePredTest(uint32_t op, uint32_t S, uint32_t opc2, const _PReg &pg,
                   const _PReg &pn) {
    uint32_t code =
        concat({F(0x25, 24), F(op, 23), F(S, 22), F(2, 19), F(3, 14),
                F(pg.getIdx(), 10), F(pn.getIdx(), 5), F(opc2, 0)});
    dw(code);
  }

  // SVE predicate zero
  void SvePredZero(uint32_t op, uint32_t S, const _PReg &pd) {
    uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(3, 19),
                            F(7, 13), F(1, 10), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE integer compare with signed immediate
  void SveIntCompSImm(uint32_t op, uint32_t o2, uint32_t ne, const _PReg &pd,
                      const _PReg &pg, const _ZReg &zn, int32_t imm) {
    uint32_t size = genSize(pd);
    uint32_t imm5 = imm & ones(5);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(imm, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);
    uint32_t code = concat({F(0x25, 24), F(size, 22), F(imm5, 16), F(op, 15),
                            F(o2, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5),
                            F(ne, 4), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE predicate count
  void SvePredCount(uint32_t opc, uint32_t o2, const RReg &rd, const _PReg &pg,
                    const _PReg &pn) {
    uint32_t size = genSize(pn);
    uint32_t code = concat({F(0x25, 24), F(size, 22), F(1, 21), F(opc, 16),
                            F(2, 14), F(pg.getIdx(), 10), F(o2, 9),
                            F(pn.getIdx(), 5), F(rd.getIdx(), 0)});
    dw(code);
  }

  // SVE Inc/Dec by Predicate Count Group
  void SveIncDecPredCount(uint32_t size, uint32_t op, uint32_t D, uint32_t opc2,
                          uint32_t type1, uint32_t type2, const Reg &rdn,
                          const _PReg &pg) {
    uint32_t code = concat({F(0x25, 24), F(size, 22), F(type1, 18), F(op, 17),
                            F(D, 16), F(type2, 11), F(opc2, 9),
                            F(pg.getIdx(), 5), F(rdn.getIdx(), 0)});
    dw(code);
  }

  // SVE inc/dec register by predicate count
  void SveIncDecRegByPredCount(uint32_t op, uint32_t D, uint32_t opc2,
                               const RReg &rdn, const _PReg &pg) {
    uint32_t size = genSize(pg);
    SveIncDecPredCount(size, op, D, opc2, 0xb, 0x11, rdn, pg);
  }

  // SVE inc/dec vector by predicate count
  void SveIncDecVecByPredCount(uint32_t op, uint32_t D, uint32_t opc2,
                               const _ZReg &zdn, const _PReg &pg) {
    uint32_t size = genSize(zdn);
    SveIncDecPredCount(size, op, D, opc2, 0xb, 0x10, zdn, pg);
  }

  // SVE saturating inc/dec register by predicate count
  void SveSatuIncDecRegByPredCount(uint32_t D, uint32_t U, uint32_t op,
                                   const RReg &rdn, const _PReg &pg) {
    uint32_t sf = genSf(rdn);
    uint32_t size = genSize(pg);
    SveIncDecPredCount(size, D, U, ((sf << 1) | op), 0xa, 0x11, rdn, pg);
  }

  // SVE saturating inc/dec vector by predicate count
  void SveSatuIncDecVecByPredCount(uint32_t D, uint32_t U, uint32_t opc,
                                   const _ZReg &zdn, const _PReg &pg) {
    uint32_t size = genSize(zdn);
    SveIncDecPredCount(size, D, U, opc, 0xa, 0x10, zdn, pg);
  }

  // SVE FFR initialise
  void SveFFRInit(uint32_t opc) {
    uint32_t code = concat({F(0x25, 24), F(opc, 22), F(0xb, 18), F(0x24, 10)});
    dw(code);
  }

  // SVE FFR write from predicate
  void SveFFRWritePred(uint32_t opc, const _PReg &pn) {
    uint32_t code = concat(
        {F(0x25, 24), F(opc, 22), F(0xa, 18), F(0x24, 10), F(pn.getIdx(), 5)});
    dw(code);
  }

  // SVE conditionally terminate scalars
  void SveCondTermScalars(uint32_t op, uint32_t ne, const RReg &rn,
                          const RReg &rm) {
    uint32_t sz = genSf(rn);
    uint32_t code =
        concat({F(0x25, 24), F(op, 23), F(sz, 22), F(1, 21), F(rm.getIdx(), 16),
                F(0x8, 10), F(rn.getIdx(), 5), F(ne, 4)});
    dw(code);
  }

  // SVE integer compare scalar count and limit
  void SveIntCompScalarCountAndLimit(uint32_t U, uint32_t lt, uint32_t eq,
                                     const _PReg &pd, const RReg &rn,
                                     const RReg &rm) {
    uint32_t size = genSize(pd);
    uint32_t sf = genSf(rn);
    uint32_t code = concat({F(0x25, 24), F(size, 22), F(1, 21),
                            F(rm.getIdx(), 16), F(sf, 12), F(U, 11), F(lt, 10),
                            F(rn.getIdx(), 5), F(eq, 4), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE broadcast floating-point immediate (unpredicated)
  void SveBcFpImmUnpred(uint32_t opc, uint32_t o2, const _ZReg &zd,
                        double imm) {
    uint32_t size = genSize(zd);
    uint32_t imm8 = compactImm(imm, zd.getBit());
    uint32_t code =
        concat({F(0x25, 24), F(size, 22), F(7, 19), F(opc, 17), F(7, 14),
                F(o2, 13), F(imm8, 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE broadcast integer immediate (unpredicated)
  void SveBcIntImmUnpred(uint32_t opc, const _ZReg &zd, int32_t imm, ShMod mod,
                         uint32_t sh) {
    verifyIncList(mod, {LSL}, ERR_ILLEGAL_SHMOD);
    verifyIncList(sh, {0, 8}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(imm, -128, 127, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t size = genSize(zd);
    uint32_t imm8 = imm & ones(8);
    uint32_t code =
        concat({F(0x25, 24), F(size, 22), F(7, 19), F(opc, 17), F(3, 14),
                F((sh == 8), 13), F(imm8, 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE integer add/subtract immediate (unpredicated)
  void SveIntAddSubImmUnpred(uint32_t opc, const _ZReg &zdn, uint32_t imm,
                             ShMod mod, uint32_t sh) {
    verifyIncList(mod, {LSL}, ERR_ILLEGAL_SHMOD);
    verifyIncList(sh, {0, 8}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(imm, 0, 255, ERR_ILLEGAL_IMM_RANGE);

    uint32_t size = genSize(zdn);
    uint32_t imm8 = imm & ones(8);
    uint32_t code =
        concat({F(0x25, 24), F(size, 22), F(4, 19), F(opc, 16), F(3, 14),
                F((sh == 8), 13), F(imm8, 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE integer min/max immediate (unpredicated)
  void SveIntMinMaxImmUnpred(uint32_t opc, uint32_t o2, const _ZReg &zdn,
                             int32_t imm) {
    if ((opc & 0x1))
      verifyIncRange(imm, 0, 255, ERR_ILLEGAL_IMM_RANGE);
    else
      verifyIncRange(imm, -128, 127, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t size = genSize(zdn);
    uint32_t imm8 = imm & ones(8);
    uint32_t code =
        concat({F(0x25, 24), F(size, 22), F(5, 19), F(opc, 16), F(3, 14),
                F(o2, 13), F(imm8, 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE integer multiply immediate (unpredicated)
  void SveIntMultImmUnpred(uint32_t opc, uint32_t o2, const _ZReg &zdn,
                           int32_t imm) {
    uint32_t size = genSize(zdn);
    uint32_t imm8 = imm & ones(8);
    verifyIncRange(imm, -128, 127, ERR_ILLEGAL_IMM_RANGE, true);
    uint32_t code =
        concat({F(0x25, 24), F(size, 22), F(6, 19), F(opc, 16), F(3, 14),
                F(o2, 13), F(imm8, 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE integer dot product (unpredicated)
  void SveIntDotProdcutUnpred(uint32_t U, const _ZReg &zda, const _ZReg &zn,
                              const _ZReg &zm) {
    uint32_t size = genSize(zda);
    uint32_t code = concat({F(0x44, 24), F(size, 22), F(zm.getIdx(), 16),
                            F(U, 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
    dw(code);
  }

  // SVE integer dot product (indexed)
  void SveIntDotProdcutIndexed(uint32_t size, uint32_t U, const _ZReg &zda,
                               const _ZReg &zn, const ZRegElem &zm) {
    uint32_t zm_idx = zm.getIdx();
    uint32_t zm_eidx = zm.getElemIdx();
    uint32_t opc = (size == 2) ? (((zm_eidx & ones(2)) << 3) | zm_idx)
                               : (((zm_eidx & ones(1)) << 4) | zm_idx);

    verifyIncRange(zm_eidx, 0, (size == 2) ? 3 : 1, ERR_ILLEGAL_REG_ELEM_IDX);
    verifyIncRange(zm_idx, 0, (size == 2) ? 7 : 15, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(0x44, 24), F(size, 22), F(1, 21), F(opc, 16),
                            F(U, 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point complex add (predicated)
  void SveFpComplexAddPred(const _ZReg &zdn, const _PReg &pg, const _ZReg &zm,
                           uint32_t ct) {
    uint32_t size = genSize(zdn);
    uint32_t rot = (ct == 270) ? 1 : 0;
    verifyIncList(ct, {90, 270}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x64, 24), F(size, 22), F(rot, 16), F(1, 15),
                F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point complex multiply-add (predicated)
  void SveFpComplexMultAddPred(const _ZReg &zda, const _PReg &pg,
                               const _ZReg &zn, const _ZReg &zm, uint32_t ct) {
    uint32_t size = genSize(zda);
    uint32_t rot = (ct / 90);
    verifyIncList(ct, {0, 90, 180, 270}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x64, 24), F(size, 22), F(zm.getIdx(), 16), F(rot, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point multiply-add (indexed)
  void SveFpMultAddIndexed(uint32_t op, const _ZReg &zda, const _ZReg &zn,
                           const ZRegElem &zm) {
    uint32_t zm_idx = zm.getIdx();
    uint32_t zm_bit = zm.getBit();
    uint32_t zm_eidx = zm.getElemIdx();
    uint32_t size = (zm_bit == 16) ? (0 | field(zm_eidx, 2, 2)) : genSize(zda);
    uint32_t opc = (zm_bit == 64) ? (((zm_eidx & ones(1)) << 4) | zm_idx)
                                  : (((zm_eidx & ones(2)) << 3) | zm_idx);

    verifyIncRange(zm_eidx, 0, ((zm_bit == 16) ? 7 : (zm_bit == 32) ? 3 : 1),
                   ERR_ILLEGAL_REG_ELEM_IDX);
    verifyIncRange(zm_eidx, 0, ((zm_bit == 64) ? 15 : 7),
                   ERR_ILLEGAL_REG_ELEM_IDX);

    uint32_t code = concat({F(0x64, 24), F(size, 22), F(1, 21), F(opc, 16),
                            F(op, 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point complex multiply-add (indexed)
  void SveFpComplexMultAddIndexed(const _ZReg &zda, const _ZReg &zn,
                                  const ZRegElem &zm, uint32_t ct) {
    uint32_t size = genSize(zda) + 1;
    uint32_t zm_idx = zm.getIdx();
    uint32_t zm_eidx = zm.getElemIdx();
    uint32_t opc = (size == 2) ? (((zm_eidx & ones(2)) << 3) | zm_idx)
                               : (((zm_eidx & ones(1)) << 4) | zm_idx);

    verifyIncRange(zm_eidx, 0, (size == 2) ? 3 : 1, ERR_ILLEGAL_REG_ELEM_IDX);
    verifyIncRange(zm_idx, 0, (size == 2) ? 7 : 15, ERR_ILLEGAL_REG_IDX);
    verifyIncList(ct, {0, 90, 180, 270}, ERR_ILLEGAL_CONST_VALUE);

    uint32_t rot = (ct / 90);
    uint32_t code =
        concat({F(0x64, 24), F(size, 22), F(1, 21), F(opc, 16), F(1, 12),
                F(rot, 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point multiply (indexed)
  void SveFpMultIndexed(const _ZReg &zd, const _ZReg &zn, const ZRegElem &zm) {
    uint32_t zm_idx = zm.getIdx();
    uint32_t zm_bit = zm.getBit();
    uint32_t zm_eidx = zm.getElemIdx();
    uint32_t size = (zm_bit == 16) ? (0 | field(zm_eidx, 2, 2)) : genSize(zd);
    uint32_t opc = (zm_bit == 64) ? (((zm_eidx & ones(1)) << 4) | zm_idx)
                                  : (((zm_eidx & ones(2)) << 3) | zm_idx);

    verifyIncRange(zm_eidx, 0, ((zm_bit == 16) ? 7 : (zm_bit == 32) ? 3 : 1),
                   ERR_ILLEGAL_REG_ELEM_IDX);
    verifyIncRange(zm_eidx, 0, ((zm_bit == 64) ? 15 : 7),
                   ERR_ILLEGAL_REG_ELEM_IDX);

    uint32_t code = concat({F(0x64, 24), F(size, 22), F(1, 21), F(opc, 16),
                            F(1, 13), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point recursive reduction
  void SveFpRecurReduct(uint32_t opc, const VRegSc vd, const _PReg &pg,
                        const _ZReg &zn) {
    uint32_t size = genSize(vd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x65, 24), F(size, 22), F(opc, 16), F(1, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(vd.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point reciprocal estimate unpredicated
  void SveFpReciproEstUnPred(uint32_t opc, const _ZReg &zd, const _ZReg &zn) {
    uint32_t size = genSize(zd);
    uint32_t code = concat({F(0x65, 24), F(size, 22), F(1, 19), F(opc, 16),
                            F(3, 12), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point compare with zero
  void SveFpCompWithZero(uint32_t eq, uint32_t lt, uint32_t ne, const _PReg &pd,
                         const _PReg &pg, const _ZReg &zn, double zero) {
    uint32_t size = genSize(pd);
    verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x65, 24), F(size, 22), F(1, 20), F(eq, 17),
                            F(lt, 16), F(1, 13), F(pg.getIdx(), 10),
                            F(zn.getIdx(), 5), F(ne, 4), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point serial resuction (predicated)
  void SveFpSerialReductPred(uint32_t opc, const VRegSc vdn, const _PReg &pg,
                             const _ZReg &zm) {
    uint32_t size = genSize(vdn);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x65, 24), F(size, 22), F(3, 19), F(opc, 16), F(1, 13),
                F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(vdn.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point arithmetic (unpredicated)
  void SveFpArithmeticUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn,
                             const _ZReg &zm) {
    uint32_t size = genSize(zd);
    uint32_t code = concat({F(0x65, 24), F(size, 22), F(zm.getIdx(), 16),
                            F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point arithmetic (predicated)
  void SveFpArithmeticPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg,
                           const _ZReg &zm) {
    uint32_t size = genSize(zdn);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x65, 24), F(size, 22), F(opc, 16), F(4, 13),
                F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point arithmetic with immediate (predicated)
  void SveFpArithmeticImmPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg,
                              float ct) {
    uint32_t size = genSize(zdn);
    uint32_t i1 = (std::lround(ct * 10) < 10) ? 0 : 1;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

    if (opc == 0 || opc == 1 || opc == 3)
      verifyIncList(std::lround(ct * 10), {5, 10}, ERR_ILLEGAL_CONST_VALUE);
    else if (opc == 2)
      verifyIncList(std::lround(ct * 10), {5, 20}, ERR_ILLEGAL_CONST_VALUE);
    else
      verifyIncList(std::lround(ct * 10), {0, 10}, ERR_ILLEGAL_CONST_VALUE);

    uint32_t code =
        concat({F(0x65, 24), F(size, 22), F(3, 19), F(opc, 16), F(4, 13),
                F(pg.getIdx(), 10), F(i1, 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point trig multiply-add coefficient
  void SveFpTrigMultAddCoef(const _ZReg &zdn, const _ZReg &zm, uint32_t imm) {
    uint32_t size = genSize(zdn);
    uint32_t imm3 = imm & ones(3);
    verifyIncRange(imm, 0, 7, ERR_ILLEGAL_IMM_RANGE);
    uint32_t code = concat({F(0x65, 24), F(size, 22), F(2, 19), F(imm3, 16),
                            F(1, 15), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point convert precision
  void SveFpCvtPrecision(uint32_t opc, uint32_t opc2, const _ZReg &zd,
                         const _PReg &pg, const _ZReg &zn) {
    uint32_t code =
        concat({F(0x65, 24), F(opc, 22), F(1, 19), F(opc2, 16), F(5, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point convert to integer
  void SveFpCvtToInt(uint32_t opc, uint32_t opc2, uint32_t U, const _ZReg &zd,
                     const _PReg &pg, const _ZReg &zn) {
    uint32_t code = concat({F(0x65, 24), F(opc, 22), F(3, 19), F(opc2, 17),
                            F(U, 16), F(5, 13), F(pg.getIdx(), 10),
                            F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point round to integral value
  void SveFpRoundToIntegral(uint32_t opc, const _ZReg &zd, const _PReg &pg,
                            const _ZReg &zn) {
    uint32_t size = genSize(zd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x65, 24), F(size, 22), F(opc, 16), F(5, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE floating-point unary operations
  void SveFpUnaryOp(uint32_t opc, const _ZReg &zd, const _PReg &pg,
                    const _ZReg &zn) {
    uint32_t size = genSize(zd);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x65, 24), F(size, 22), F(3, 18), F(opc, 16), F(5, 13),
                F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE integer convert to floationg-point
  void SveIntCvtToFp(uint32_t opc, uint32_t opc2, uint32_t U, const _ZReg &zd,
                     const _PReg &pg, const _ZReg &zn) {
    uint32_t code = concat({F(0x65, 24), F(opc, 22), F(2, 19), F(opc2, 17),
                            F(U, 16), F(5, 13), F(pg.getIdx(), 10),
                            F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
    dw(code);
  }

  // SVE floationg-point compare vectors
  void SveFpCompVec(uint32_t op, uint32_t o2, uint32_t o3, const _PReg &pd,
                    const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
    uint32_t size = genSize(pd);
    uint32_t code = concat({F(0x65, 24), F(size, 22), F(zm.getIdx(), 16),
                            F(op, 15), F(1, 14), F(o2, 13), F(pg.getIdx(), 10),
                            F(zn.getIdx(), 5), F(o3, 4), F(pd.getIdx(), 0)});
    dw(code);
  }

  // SVE floationg-point multiply-accumulate writing addend
  void SveFpMultAccumAddend(uint32_t opc, const _ZReg &zda, const _PReg &pg,
                            const _ZReg &zn, const _ZReg &zm) {
    uint32_t size = genSize(zda);
    uint32_t code = concat({F(0x65, 24), F(size, 22), F(1, 21),
                            F(zm.getIdx(), 16), F(opc, 13), F(pg.getIdx(), 10),
                            F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
    dw(code);
  }

  // SVE floationg-point multiply-accumulate writing multiplicand
  void SveFpMultAccumMulti(uint32_t opc, const _ZReg &zdn, const _PReg &pg,
                           const _ZReg &zm, const _ZReg &za) {
    uint32_t size = genSize(zdn);
    uint32_t code =
        concat({F(0x65, 24), F(size, 22), F(1, 21), F(za.getIdx(), 16),
                F(1, 15), F(opc, 13), F(pg.getIdx(), 10), F(zm.getIdx(), 5),
                F(zdn.getIdx(), 0)});
    dw(code);
  }

  // SVE 32-bit gather load (scalar plus 32-bit unscaled offsets)
  void Sve32GatherLdSc32U(uint32_t msz, uint32_t U, uint32_t ff,
                          const _ZReg &zt, const _PReg &pg,
                          const AdrSc32U &adr) {
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x42, 25), F(msz, 23), F(xs, 22), F(adr.getZm().getIdx(), 16),
                F(U, 14), F(ff, 13), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 32-bit gather load (vector plus immediate)
  void Sve32GatherLdVecImm(uint32_t msz, uint32_t U, uint32_t ff,
                           const _ZReg &zt, const _PReg &pg,
                           const AdrVecImm32 &adr) {
    uint32_t imm5 = (adr.getImm() >> msz) & ones(5);

    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(adr.getImm(), 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);

    uint32_t code = concat({F(0x42, 25), F(msz, 23), F(1, 21), F(imm5, 16),
                            F(1, 15), F(U, 14), F(ff, 13), F(pg.getIdx(), 10),
                            F(adr.getZn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 32-bit gather load halfwords (scalar plus 32-bit scaled offsets)
  void Sve32GatherLdHSc32S(uint32_t U, uint32_t ff, const _ZReg &zt,
                           const _PReg &pg, const AdrSc32S &adr) {
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x42, 25), F(1, 23), F(xs, 22), F(1, 21),
                            F(adr.getZm().getIdx(), 16), F(U, 14), F(ff, 13),
                            F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                            F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 32-bit gather load words (scalar plus 32-bit scaled offsets)
  void Sve32GatherLdWSc32S(uint32_t U, uint32_t ff, const _ZReg &zt,
                           const _PReg &pg, const AdrSc32S &adr) {
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x42, 25), F(2, 23), F(xs, 22), F(1, 21),
                            F(adr.getZm().getIdx(), 16), F(U, 14), F(ff, 13),
                            F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                            F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 32-bit gather prefetch (scalar plus 32-bit scaled offsets)
  void Sve32GatherPfSc32S(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg,
                          const AdrSc32S &adr) {
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x42, 25), F(xs, 22), F(1, 21), F(adr.getZm().getIdx(), 16),
                F(msz, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                F(prfop_sve, 0)});
    dw(code);
  }

  // SVE 32-bit gather prefetch (vector plus immediate)
  void Sve32GatherPfVecImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg,
                           const AdrVecImm32 &adr) {
    uint32_t imm5 = (adr.getImm() >> msz) & ones(5);

    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    verifyIncRange(adr.getImm(), 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);

    uint32_t code = concat({F(0x42, 25), F(msz, 23), F(imm5, 16), F(7, 13),
                            F(pg.getIdx(), 10), F(adr.getZn().getIdx(), 5),
                            F(prfop_sve, 0)});
    dw(code);
  }

  // SVE 32-bit contiguous prefetch (scalar plus immediate)
  void Sve32ContiPfScImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg,
                         const AdrScImm &adr) {
    int32_t simm = adr.getSimm();
    uint32_t imm6 = simm & ones(6);
    verifyIncRange(simm, -32, 31, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x42, 25), F(7, 22), F(imm6, 16), F(msz, 13),
                            F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                            F(prfop_sve, 0)});
    dw(code);
  }

  void Sve32ContiPfScImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg,
                         const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x42, 25), F(7, 22), F(0, 16), F(msz, 13), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(prfop_sve, 0)});
    dw(code);
  }

  // SVE 32-bit contiguous prefetch (scalar plus scalar)
  void Sve32ContiPfScSc(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg,
                        const AdrScSc &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    uint32_t code = concat(
        {F(0x42, 25), F(msz, 23), F(adr.getXm().getIdx(), 16), F(6, 13),
         F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(prfop_sve, 0)});
    dw(code);
  }

  // SVE load and broadcast element
  void SveLoadAndBcElem(uint32_t dtypeh, uint32_t dtypel, const _ZReg &zt,
                        const _PReg &pg, const AdrScImm &adr) {
    uint32_t uimm = adr.getSimm();
    uint32_t dtype = dtypeh << 2 | dtypel;
    uint32_t size = genSize(dtype);
    uint32_t imm6 = (uimm >> size) & ones(6);

    verifyIncRange(uimm, 0, 63 * (1 << size), ERR_ILLEGAL_IMM_RANGE);
    verifyCond(
        uimm,
        [=](uint64_t x) {
          return (x % ((static_cast<uint64_t>(1)) << size)) == 0;
        },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(0x42, 25), F(dtypeh, 23), F(1, 22), F(imm6, 16),
                            F(1, 15), F(dtypel, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveLoadAndBcElem(uint32_t dtypeh, uint32_t dtypel, const _ZReg &zt,
                        const _PReg &pg, const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x42, 25), F(dtypeh, 23), F(1, 22), F(0, 16),
                            F(1, 15), F(dtypel, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE load predicate register
  void SveLoadPredReg(const _PReg &pt, const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm9h = field(imm, 8, 3);
    uint32_t imm9l = field(imm, 2, 0);
    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
    uint32_t code = concat({F(0x42, 25), F(3, 23), F(imm9h, 16), F(imm9l, 10),
                            F(adr.getXn().getIdx(), 5), F(pt.getIdx(), 0)});
    dw(code);
  }

  void SveLoadPredReg(const _PReg &pt, const AdrNoOfs &adr) {
    uint32_t code = concat({F(0x42, 25), F(3, 23), F(0, 16), F(0, 10),
                            F(adr.getXn().getIdx(), 5), F(pt.getIdx(), 0)});
    dw(code);
  }

  // SVE load predicate vector
  void SveLoadPredVec(const _ZReg &zt, const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm9h = field(imm, 8, 3);
    uint32_t imm9l = field(imm, 2, 0);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t code =
        concat({F(0x42, 25), F(3, 23), F(imm9h, 16), F(1, 14), F(imm9l, 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveLoadPredVec(const _ZReg &zt, const AdrNoOfs &adr) {
    uint32_t code = concat({F(0x42, 25), F(3, 23), F(0, 16), F(1, 14), F(0, 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous first-fault load (scalar plus scalar)
  void SveContiFFLdScSc(uint32_t dtype, const _ZReg &zt, const _PReg &pg,
                        const AdrScSc &adr) {
    if (adr.getInitMod()) {
      verifyIncList(adr.getSh(), {genSize(dtype)}, ERR_ILLEGAL_CONST_VALUE);
      verifyIncList(adr.getMod(), {LSL}, ERR_ILLEGAL_SHMOD);
    }

    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat(
        {F(0x52, 25), F(dtype, 21), F(adr.getXm().getIdx(), 16), F(3, 13),
         F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveContiFFLdScSc(uint32_t dtype, const _ZReg &zt, const _PReg &pg,
                        const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(31, 16), F(3, 13),
                            F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                            F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous load (scalar plus immediate)
  void SveContiLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg,
                       const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm4 = imm & ones(4);
    verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(imm4, 16), F(5, 13),
                            F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                            F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveContiLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg,
                       const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(0, 16), F(5, 13),
                            F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                            F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous load (scalar plus scalar)
  void SveContiLdScSc(uint32_t dtype, const _ZReg &zt, const _PReg &pg,
                      const AdrScSc &adr) {
    verifyIncList(adr.getSh(), {genSize(dtype)}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat(
        {F(0x52, 25), F(dtype, 21), F(adr.getXm().getIdx(), 16), F(2, 13),
         F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous non-fault load (scalar plus immediate)
  void SveContiNFLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg,
                         const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm4 = imm & ones(4);
    verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(1, 20), F(imm4, 16),
                            F(5, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveContiNFLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg,
                         const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(1, 20), F(0, 16),
                            F(5, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous non-temporal load (scalar plus immediate)
  void SveContiNTLdScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                         const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm4 = imm & ones(4);
    verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(msz, 23), F(imm4, 16), F(7, 13),
                            F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                            F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveContiNTLdScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                         const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x52, 25), F(msz, 23), F(0, 16), F(7, 13), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous non-temporal load (scalar plus scalar)
  void SveContiNTLdScSc(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                        const AdrScSc &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat(
        {F(0x52, 25), F(msz, 23), F(adr.getXm().getIdx(), 16), F(6, 13),
         F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE load and broadcast quadword (scalar plus immediate)
  void SveLdBcQuadScImm(uint32_t msz, uint32_t num, const _ZReg &zt,
                        const _PReg &pg, const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm4 = (imm >> 4) & ones(4);
    verifyIncRange(imm, -128, 127, ERR_ILLEGAL_IMM_RANGE, true);
    verifyCond(
        imm, [](uint64_t x) { return (x % 16) == 0; }, ERR_ILLEGAL_IMM_COND);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(imm4, 16),
                            F(1, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveLdBcQuadScImm(uint32_t msz, uint32_t num, const _ZReg &zt,
                        const _PReg &pg, const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(0, 16),
                            F(1, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE load and broadcast quadword (scalar plus scalar)
  void SveLdBcQuadScSc(uint32_t msz, uint32_t num, const _ZReg &zt,
                       const _PReg &pg, const AdrScSc &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21),
                            F(adr.getXm().getIdx(), 16), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE load multiple structures (scalar plus immediate)
  void SveLdMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt,
                             const _PReg &pg, const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm4 = (imm / ((int32_t)num + 1)) & ones(4);

    verifyIncRange(imm, -8 * ((int32_t)num + 1), 7 * ((int32_t)num + 1),
                   ERR_ILLEGAL_IMM_RANGE, true);
    verifyCond(
        std::abs(imm), [=](uint64_t x) { return (x % (num + 1)) == 0; },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(imm4, 16),
                            F(7, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveLdMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt,
                             const _PReg &pg, const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(0, 16),
                            F(7, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE load multiple structures (scalar plus scalar)
  void SveLdMultiStructScSc(uint32_t msz, uint32_t num, const _ZReg &zt,
                            const _PReg &pg, const AdrScSc &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x52, 25), F(msz, 23), F(num, 21),
                F(adr.getXm().getIdx(), 16), F(6, 13), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit gather load (scalar plus unpacked 32-bit scaled offsets)
  void Sve64GatherLdSc32US(uint32_t msz, uint32_t U, uint32_t ff,
                           const _ZReg &zt, const _PReg &pg,
                           const AdrSc32US &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x62, 25), F(msz, 23), F(xs, 22), F(1, 21),
                            F(adr.getZm().getIdx(), 16), F(U, 14), F(ff, 13),
                            F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                            F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit gather load (scalar plus 64-bit scaled offsets)
  void Sve64GatherLdSc64S(uint32_t msz, uint32_t U, uint32_t ff,
                          const _ZReg &zt, const _PReg &pg,
                          const AdrSc64S &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x62, 25), F(msz, 23), F(3, 21), F(adr.getZm().getIdx(), 16),
                F(1, 15), F(U, 14), F(ff, 13), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit gather load (scalar plus 64-bit unscaled offsets)
  void Sve64GatherLdSc64U(uint32_t msz, uint32_t U, uint32_t ff,
                          const _ZReg &zt, const _PReg &pg,
                          const AdrSc64U &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x62, 25), F(msz, 23), F(2, 21), F(adr.getZm().getIdx(), 16),
                F(1, 15), F(U, 14), F(ff, 13), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit gather load (scalar plus unpacked 32-bit unscaled offsets)
  void Sve64GatherLdSc32UU(uint32_t msz, uint32_t U, uint32_t ff,
                           const _ZReg &zt, const _PReg &pg,
                           const AdrSc32UU &adr) {
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x62, 25), F(msz, 23), F(xs, 22), F(adr.getZm().getIdx(), 16),
                F(U, 14), F(ff, 13), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit gather load (vector plus immeidate)
  void Sve64GatherLdVecImm(uint32_t msz, uint32_t U, uint32_t ff,
                           const _ZReg &zt, const _PReg &pg,
                           const AdrVecImm64 &adr) {
    uint32_t imm = adr.getImm();
    uint32_t imm5 = (imm >> msz) & ones(5);

    verifyIncRange(imm, 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);
    verifyCond(
        imm,
        [=](uint64_t x) {
          return (x % ((static_cast<uint64_t>(1)) << msz)) == 0;
        },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(0x62, 25), F(msz, 23), F(1, 21), F(imm5, 16),
                            F(1, 15), F(U, 14), F(ff, 13), F(pg.getIdx(), 10),
                            F(adr.getZn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit gather load (scalar plus 64-bit scaled offsets)
  void Sve64GatherPfSc64S(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg,
                          const AdrSc64S &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x62, 25), F(3, 21), F(adr.getZm().getIdx(), 16),
                            F(1, 15), F(msz, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(prfop_sve, 0)});
    dw(code);
  }

  // SVE 64-bit gather load (scalar plus unpacked 32-bit scaled offsets)
  void Sve64GatherPfSc32US(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg,
                           const AdrSc32US &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x62, 25), F(xs, 22), F(1, 21), F(adr.getZm().getIdx(), 16),
                F(msz, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                F(prfop_sve, 0)});
    dw(code);
  }

  // SVE 64-bit gather load (vector plus immediate)
  void Sve64GatherPfVecImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg,
                           const AdrVecImm64 &adr) {
    uint32_t imm = adr.getImm();
    uint32_t imm5 = (imm >> msz) & ones(5);

    verifyIncRange(imm, 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);
    verifyCond(
        imm,
        [=](uint64_t x) {
          return (x % ((static_cast<uint64_t>(1)) << msz)) == 0;
        },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(0x62, 25), F(msz, 23), F(imm5, 16), F(7, 13),
                            F(pg.getIdx(), 10), F(adr.getZn().getIdx(), 5),
                            F(prfop_sve, 0)});
    dw(code);
  }

  // SVE 32-bit scatter store (sclar plus 32-bit scaled offsets)
  void Sve32ScatterStSc32S(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                           const AdrSc32S &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x72, 25), F(msz, 23), F(3, 21), F(adr.getZm().getIdx(), 16),
                F(1, 15), F(xs, 14), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 32-bit scatter store (sclar plus 32-bit unscaled offsets)
  void Sve32ScatterStSc32U(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                           const AdrSc32U &adr) {
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x72, 25), F(msz, 23), F(2, 21), F(adr.getZm().getIdx(), 16),
                F(1, 15), F(xs, 14), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 32-bit scatter store (vector plus immediate)
  void Sve32ScatterStVecImm(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                            const AdrVecImm32 &adr) {
    uint32_t imm = adr.getImm();
    uint32_t imm5 = (imm >> msz) & ones(5);

    verifyIncRange(imm, 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);
    verifyCond(
        imm,
        [=](uint64_t x) {
          return (x % ((static_cast<uint64_t>(1)) << msz)) == 0;
        },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(0x72, 25), F(msz, 23), F(3, 21), F(imm5, 16),
                            F(5, 13), F(pg.getIdx(), 10),
                            F(adr.getZn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit scatter store (scalar plus 64-bit scaled offsets)
  void Sve64ScatterStSc64S(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                           const AdrSc64S &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x72, 25), F(msz, 23), F(1, 21), F(adr.getZm().getIdx(), 16),
                F(5, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit scatter store (scalar plus 64-bit unscaled offsets)
  void Sve64ScatterStSc64U(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                           const AdrSc64U &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat(
        {F(0x72, 25), F(msz, 23), F(adr.getZm().getIdx(), 16), F(5, 13),
         F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit scatter store (scalar plus unpacked 32-bit scaled offsets)
  void Sve64ScatterStSc32US(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                            const AdrSc32US &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x72, 25), F(msz, 23), F(1, 21), F(adr.getZm().getIdx(), 16),
                F(1, 15), F(xs, 14), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit scatter store (scalar plus unpacked 32-bit unscaled offsets)
  void Sve64ScatterStSc32UU(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                            const AdrSc32UU &adr) {
    uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x72, 25), F(msz, 23), F(adr.getZm().getIdx(), 16), F(1, 15),
                F(xs, 14), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5),
                F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE 64-bit scatter store (vector plus immediate)
  void Sve64ScatterStVecImm(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                            const AdrVecImm64 &adr) {
    uint32_t imm = adr.getImm();
    uint32_t imm5 = (imm >> msz) & ones(5);

    verifyIncRange(imm, 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);
    verifyCond(
        imm,
        [=](uint64_t x) {
          return (x % ((static_cast<uint64_t>(1)) << msz)) == 0;
        },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(0x72, 25), F(msz, 23), F(2, 21), F(imm5, 16),
                            F(5, 13), F(pg.getIdx(), 10),
                            F(adr.getZn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous non-temporal store (scalar plus immediate)
  void SveContiNTStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                         const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm4 = imm & ones(4);
    verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x72, 25), F(msz, 23), F(1, 20), F(imm4, 16),
                            F(7, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveContiNTStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                         const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x72, 25), F(msz, 23), F(1, 20), F(0, 16),
                            F(7, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous non-temporal store (scalar plus scalar)
  void SveContiNTStScSc(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                        const AdrScSc &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat(
        {F(0x72, 25), F(msz, 23), F(adr.getXm().getIdx(), 16), F(3, 13),
         F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous store (scalar plus immediate)
  void SveContiStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                       const AdrScImm &adr) {
    uint32_t size = genSize(zt);
    int32_t imm = adr.getSimm();
    uint32_t imm4 = imm & ones(4);
    verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x72, 25), F(msz, 23), F(size, 21), F(imm4, 16),
                            F(7, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveContiStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                       const AdrNoOfs &adr) {
    uint32_t size = genSize(zt);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x72, 25), F(msz, 23), F(size, 21), F(0, 16),
                            F(7, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE contiguous store (scalar plus scalar)
  void SveContiStScSc(uint32_t msz, const _ZReg &zt, const _PReg &pg,
                      const AdrScSc &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    uint32_t size = genSize(zt);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x72, 25), F(msz, 23), F(size, 21),
                F(adr.getXm().getIdx(), 16), F(2, 13), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE store multipule structures (scalar plus immediate)
  void SveStMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt,
                             const _PReg &pg, const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm4 = (imm / ((int32_t)num + 1)) & ones(4);

    verifyIncRange(imm, -8 * ((int32_t)num + 1), 7 * ((int32_t)num + 1),
                   ERR_ILLEGAL_IMM_RANGE, true);
    verifyCond(
        std::abs(imm), [=](uint64_t x) { return (x % (num + 1)) == 0; },
        ERR_ILLEGAL_IMM_COND);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

    uint32_t code = concat({F(0x72, 25), F(msz, 23), F(num, 21), F(1, 20),
                            F(imm4, 16), F(7, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveStMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt,
                             const _PReg &pg, const AdrNoOfs &adr) {
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code = concat({F(0x72, 25), F(msz, 23), F(num, 21), F(1, 20),
                            F(0, 16), F(7, 13), F(pg.getIdx(), 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE store multipule structures (scalar plus scalar)
  void SveStMultiStructScSc(uint32_t msz, uint32_t num, const _ZReg &zt,
                            const _PReg &pg, const AdrScSc &adr) {
    verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
    uint32_t code =
        concat({F(0x72, 25), F(msz, 23), F(num, 21),
                F(adr.getXm().getIdx(), 16), F(3, 13), F(pg.getIdx(), 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  // SVE store predicate register
  void SveStorePredReg(const _PReg &pt, const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm9h = field(imm, 8, 3);
    uint32_t imm9l = field(imm, 2, 0);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t code = concat({F(0x72, 25), F(3, 23), F(imm9h, 16), F(imm9l, 10),
                            F(adr.getXn().getIdx(), 5), F(pt.getIdx(), 0)});
    dw(code);
  }

  void SveStorePredReg(const _PReg &pt, const AdrNoOfs &adr) {
    uint32_t code = concat({F(0x72, 25), F(3, 23), F(0, 16), F(0, 10),
                            F(adr.getXn().getIdx(), 5), F(pt.getIdx(), 0)});
    dw(code);
  }

  // SVE store predicate vector
  void SveStorePredVec(const _ZReg &zt, const AdrScImm &adr) {
    int32_t imm = adr.getSimm();
    uint32_t imm9h = field(imm, 8, 3);
    uint32_t imm9l = field(imm, 2, 0);

    verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

    uint32_t code =
        concat({F(0x72, 25), F(3, 23), F(imm9h, 16), F(2, 13), F(imm9l, 10),
                F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  void SveStorePredVec(const _ZReg &zt, const AdrNoOfs &adr) {
    uint32_t code = concat({F(0x72, 25), F(3, 23), F(0, 16), F(2, 13), F(0, 10),
                            F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
    dw(code);
  }

  template <class T> void putL_inner(T &label) {
    if (isAutoGrow() && size_ >= maxSize_)
      growMemory();
    UncondBrImm(0, label); // insert nemonic (B <label>)
  }

public:
  const WReg w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12;
  const WReg w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23;
  const WReg w24, w25, w26, w27, w28, w29, w30, wzr, wsp;

  const XReg x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
  const XReg x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23;
  const XReg x24, x25, x26, x27, x28, x29, x30, xzr, sp;

  const BReg b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12;
  const BReg b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23;
  const BReg b24, b25, b26, b27, b28, b29, b30, b31;

  const HReg h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;
  const HReg h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23;
  const HReg h24, h25, h26, h27, h28, h29, h30, h31;

#ifdef DNNL_AARCH64
  const SReg s0, s1, s2, s3, s4, s5, s6, s7, s8_, s9, s10, s11, s12;
#else
  const SReg s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12;
#endif
  const SReg s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23;
  const SReg s24, s25, s26, s27, s28, s29, s30, s31;

  const DReg d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12;
  const DReg d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23;
  const DReg d24, d25, d26, d27, d28, d29, d30, d31;

  const QReg q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12;
  const QReg q13, q14, q15, q16, q17, q18, q19, q20, q21, q22, q23;
  const QReg q24, q25, q26, q27, q28, q29, q30, q31;

  const VReg v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12;
  const VReg v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23;
  const VReg v24, v25, v26, v27, v28, v29, v30, v31;

  const ZReg z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12;
  const ZReg z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, z23;
  const ZReg z24, z25, z26, z27, z28, z29, z30, z31;

  const PReg p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
  const PReg p13, p14, p15;

  CodeGenerator(size_t maxSize = DEFAULT_MAX_CODE_SIZE, void *userPtr = 0,
                Allocator *allocator = 0)
      : CodeArray(maxSize, userPtr, allocator)
#if 1
        ,
        w0(0), w1(1), w2(2), w3(3), w4(4), w5(5), w6(6), w7(7), w8(8), w9(9),
        w10(10), w11(11), w12(12), w13(13), w14(14), w15(15), w16(16), w17(17),
        w18(18), w19(19), w20(20), w21(21), w22(22), w23(23), w24(24), w25(25),
        w26(26), w27(27), w28(28), w29(29), w30(30), wzr(31), wsp(31)

        ,
        x0(0), x1(1), x2(2), x3(3), x4(4), x5(5), x6(6), x7(7), x8(8), x9(9),
        x10(10), x11(11), x12(12), x13(13), x14(14), x15(15), x16(16), x17(17),
        x18(18), x19(19), x20(20), x21(21), x22(22), x23(23), x24(24), x25(25),
        x26(26), x27(27), x28(28), x29(29), x30(30), xzr(31), sp(31)

        ,
        b0(0), b1(1), b2(2), b3(3), b4(4), b5(5), b6(6), b7(7), b8(8), b9(9),
        b10(10), b11(11), b12(12), b13(13), b14(14), b15(15), b16(16), b17(17),
        b18(18), b19(19), b20(20), b21(21), b22(22), b23(23), b24(24), b25(25),
        b26(26), b27(27), b28(28), b29(29), b30(30), b31(31)

        ,
        h0(0), h1(1), h2(2), h3(3), h4(4), h5(5), h6(6), h7(7), h8(8), h9(9),
        h10(10), h11(11), h12(12), h13(13), h14(14), h15(15), h16(16), h17(17),
        h18(18), h19(19), h20(20), h21(21), h22(22), h23(23), h24(24), h25(25),
        h26(26), h27(27), h28(28), h29(29), h30(30), h31(31)

        ,
        s0(0), s1(1), s2(2), s3(3), s4(4), s5(5), s6(6), s7(7),
#ifdef DNNL_AARCH64
        s8_(8),
#else
        s8(8),
#endif
        s9(9), s10(10), s11(11), s12(12), s13(13), s14(14), s15(15), s16(16),
        s17(17), s18(18), s19(19), s20(20), s21(21), s22(22), s23(23), s24(24),
        s25(25), s26(26), s27(27), s28(28), s29(29), s30(30), s31(31)

        ,
        d0(0), d1(1), d2(2), d3(3), d4(4), d5(5), d6(6), d7(7), d8(8), d9(9),
        d10(10), d11(11), d12(12), d13(13), d14(14), d15(15), d16(16), d17(17),
        d18(18), d19(19), d20(20), d21(21), d22(22), d23(23), d24(24), d25(25),
        d26(26), d27(27), d28(28), d29(29), d30(30), d31(31)

        ,
        q0(0), q1(1), q2(2), q3(3), q4(4), q5(5), q6(6), q7(7), q8(8), q9(9),
        q10(10), q11(11), q12(12), q13(13), q14(14), q15(15), q16(16), q17(17),
        q18(18), q19(19), q20(20), q21(21), q22(22), q23(23), q24(24), q25(25),
        q26(26), q27(27), q28(28), q29(29), q30(30), q31(31)

        ,
        v0(0), v1(1), v2(2), v3(3), v4(4), v5(5), v6(6), v7(7), v8(8), v9(9),
        v10(10), v11(11), v12(12), v13(13), v14(14), v15(15), v16(16), v17(17),
        v18(18), v19(19), v20(20), v21(21), v22(22), v23(23), v24(24), v25(25),
        v26(26), v27(27), v28(28), v29(29), v30(30), v31(31)

        ,
        z0(0), z1(1), z2(2), z3(3), z4(4), z5(5), z6(6), z7(7), z8(8), z9(9),
        z10(10), z11(11), z12(12), z13(13), z14(14), z15(15), z16(16), z17(17),
        z18(18), z19(19), z20(20), z21(21), z22(22), z23(23), z24(24), z25(25),
        z26(26), z27(27), z28(28), z29(29), z30(30), z31(31)

        ,
        p0(0), p1(1), p2(2), p3(3), p4(4), p5(5), p6(6), p7(7), p8(8), p9(9),
        p10(10), p11(11), p12(12), p13(13), p14(14), p15(15)
#endif
  {
    labelMgr_.set(this);
  }

  unsigned int getVersion() const { return VERSION; }

  void L(Label &label) { labelMgr_.defineClabel(label); }
  Label L() {
    Label label;
    L(label);
    return label;
  }
  void inLocalLabel() { /*assert(NULL);*/
  }
  void outLocalLabel() { /*assert(NULL);*/
  }
  /*
        assign src to dst
        require
        dst : does not used by L()
        src : used by L()
*/
  void assignL(Label &dst, const Label &src) { labelMgr_.assign(dst, src); }
  /*
        put address of label to buffer
        @note the put size is 4(32-bit), 8(64-bit)
*/
  void putL(const Label &label) { putL_inner(label); }

  void reset() {
    resetSize();
    labelMgr_.reset();
    labelMgr_.set(this);
  }
  bool hasUndefinedLabel() const { return labelMgr_.hasUndefClabel(); }
  /*
        MUST call ready() to complete generating code if you use AutoGrow
   mode.
        It is not necessary for the other mode if hasUndefinedLabel() is true.
*/
  void ready(ProtectMode mode = PROTECT_RWE) {
    if (hasUndefinedLabel())
      throw Error(ERR_LABEL_IS_NOT_FOUND);
    if (isAutoGrow()) {
      calcJmpAddress();
      if (useProtect())
        setProtectMode(mode);
    }
  }
  // set read/exec
  void readyRE() { return ready(PROTECT_RE); }
#ifdef XBYAK_TEST
  void dump(bool doClear = true) {
    CodeArray::dump();
    if (doClear)
      size_ = 0;
  }
#endif

  WReg getTmpWReg() { return w29; }

  XReg getTmpXReg() { return x29; }

  VReg getTmpVReg() { return v31; }

  ZReg getTmpZReg() { return z31; }

  PReg getTmpPReg() { return p7; }

  /* If "imm" is "00..011..100..0" or "11..100..011..1",
   this function returns TRUE, otherwise FALSE. */
  template <typename T> bool isBitMask(T imm) {
    uint64_t bit_ptn = static_cast<uint64_t>(imm);
    int curr, prev = 0;
    uint64_t invCount = 0;

    prev = (bit_ptn & 0x1) ? 1 : 0;

    for (size_t i = 1; i < 8 * sizeof(T); i++) {
      curr = (bit_ptn & (uint64_t(1) << i)) ? 1 : 0;
      if (prev != curr) {
        invCount++;
      }
      prev = curr;
    }

    if (1 <= invCount && invCount <= 2) { // intCount == 0 means all 0 or all 1
      return true;
    }

    return false;
  }

#include "xbyak_aarch64_meta_mnemonic.h"
#include "xbyak_aarch64_mnemonic.h"

  void align(size_t x) {
    if (x == 4)
      return; // ARMv8 instructions are always 4 bytes.
    if (x < 4 || (x % 4))
      throw Error(ERR_BAD_ALIGN);

    if (isAutoGrow() && x > inner::ALIGN_PAGE_SIZE)
      fprintf(stderr, "warning:autoGrow mode does not support %d align\n",
              (int)x);

    size_t remain = size_t(getCurr());
    if (remain % 4)
      throw Error(ERR_BAD_ALIGN);
    remain = x - (remain % x);

    while (remain) {
      nop();
      remain -= 4;
    }
  }
};
