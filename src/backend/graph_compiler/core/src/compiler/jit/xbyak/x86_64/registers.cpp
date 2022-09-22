/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#include <util/utils.hpp>

#include "registers.hpp"

namespace sc {
namespace sc_xbyak {
namespace x86_64 {

Xbyak::Reg8 to_reg8(Xbyak::Reg r) {
    COMPILE_ASSERT(r.isREG(), "Not a GP reg: " << r.toString());
    return r.cvt8();
}

Xbyak::Reg16 to_reg16(Xbyak::Reg r) {
    COMPILE_ASSERT(r.isREG(), "Not a GP reg: " << r.toString());
    return r.cvt16();
}

Xbyak::Reg32 to_reg32(Xbyak::Reg r) {
    COMPILE_ASSERT(r.isREG(), "Not a GP reg: " << r.toString());
    return r.cvt32();
}

Xbyak::Reg64 to_reg64(Xbyak::Reg r) {
    COMPILE_ASSERT(r.isREG(), "Not a GP reg: " << r.toString());
    return r.cvt64();
}

Xbyak::Xmm to_xmm(Xbyak::Reg r) {
    COMPILE_ASSERT(r.isXMM() || r.isYMM() || r.isZMM(),
            "Not an [XYZ]MM reg: " << r.toString());
    return Xbyak::Xmm(r.getIdx());
}

Xbyak::Ymm to_ymm(Xbyak::Reg r) {
    COMPILE_ASSERT(r.isXMM() || r.isYMM() || r.isZMM(),
            "Not an [XYZ]MM reg: " << r.toString());
    return Xbyak::Ymm(r.getIdx());
}

Xbyak::Zmm to_zmm(Xbyak::Reg r) {
    COMPILE_ASSERT(r.isXMM() || r.isYMM() || r.isZMM(),
            "Not an [XYZ]MM reg: " << r.toString());
    return Xbyak::Zmm(r.getIdx());
}

Xbyak::Opmask to_mask(Xbyak::Reg r) {
    COMPILE_ASSERT(r.isOPMASK(), "Not an OPMASK reg: " << r.toString());
    return Xbyak::Opmask(r.getIdx());
}

Xbyak::Tmm to_tmm(Xbyak::Reg r) {
    COMPILE_ASSERT(r.isTMM(), "Not an AMX tile reg: " << r.toString());
    return Xbyak::Tmm(r.getIdx());
}

} // namespace x86_64
} // namespace sc_xbyak
} // namespace sc
