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

#include "operations.hpp"

namespace sc {
namespace sc_xbyak {

void X86_MOV(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src) {
    if (op_dst == op_src) {
        return;
    } else if (op_dst.is_reg() && op_src.is_reg()) {
        gen.mov(op_dst.get_reg(), op_src.get_reg());
    } else if (op_dst.is_reg() && op_src.is_addr()) {
        gen.mov(op_dst.get_reg(), op_src.get_addr());
    } else if (op_dst.is_reg() && op_src.is_imm()) {
        gen.mov(op_dst.get_reg(), op_src.get_imm());
    } else if (op_dst.is_addr() && op_src.is_reg()) {
        gen.mov(op_dst.get_addr(), op_src.get_reg());
    } else if (op_dst.is_addr() && op_src.is_imm()) {
        gen.mov(op_dst.get_addr(), op_src.get_imm());
    } else {
        COMPILE_ASSERT(false, "Invalid x86_mov: " << op_dst << ", " << op_src);
    }
}

void AVX_MOVSS(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src) {
    if (op_dst == op_src) {
        return;
    } else if (op_dst.is_xyz() && op_src.is_xyz()) {
        gen.vmovss(op_dst.get_xmm(), op_src.get_xmm());
    } else if (op_dst.is_xyz() && op_src.is_addr()) {
        gen.vmovss(op_dst.get_xmm(), op_src.get_addr());
    } else if (op_dst.is_addr() && op_src.is_xyz()) {
        gen.vmovss(op_dst.get_addr(), op_src.get_xmm());
    } else {
        COMPILE_ASSERT(
                false, "Invalid avx_movss: " << op_dst << ", " << op_src);
    }
}

void AVX_MOVPS(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src) {
    if (op_dst == op_src) {
        return;
    } else if (op_dst.is_xyz() && op_src.is_xyz()) {
        gen.vmovaps(op_dst.get_xyz(), op_src.get_xyz());
    } else if (op_dst.is_xyz() && op_src.is_addr()) {
        gen.vmovups(op_dst.get_xyz(), op_src.get_addr());
    } else if (op_dst.is_addr() && op_src.is_xyz()) {
        gen.vmovups(op_dst.get_addr(), op_src.get_xyz());
    } else {
        COMPILE_ASSERT(
                false, "Invalid avx_movps: " << op_dst << ", " << op_src);
    }
}

void AVX_MOVPI8(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src) {
    if (op_dst == op_src) {
        return;
    } else if (op_dst.is_xyz() && op_src.is_xyz()) {
        gen.vmovdqu8(op_dst.get_xyz(), op_src.get_xyz());
    } else if (op_dst.is_xyz() && op_src.is_addr()) {
        gen.vmovdqu8(op_dst.get_xyz(), op_src.get_addr());
    } else if (op_dst.is_addr() && op_src.is_xyz()) {
        gen.vmovdqu8(op_dst.get_addr(), op_src.get_xyz());
    } else {
        COMPILE_ASSERT(
                false, "Invalid avx_movps: " << op_dst << ", " << op_src);
    }
}

void AVX_MOVPI16(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src) {
    if (op_dst == op_src) {
        return;
    } else if (op_dst.is_xyz() && op_src.is_xyz()) {
        gen.vmovdqu16(op_dst.get_xyz(), op_src.get_xyz());
    } else if (op_dst.is_xyz() && op_src.is_addr()) {
        gen.vmovdqu16(op_dst.get_xyz(), op_src.get_addr());
    } else if (op_dst.is_addr() && op_src.is_xyz()) {
        gen.vmovdqu16(op_dst.get_addr(), op_src.get_xyz());
    } else {
        COMPILE_ASSERT(
                false, "Invalid avx_movps: " << op_dst << ", " << op_src);
    }
}

void AVX_MOVPI32(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src) {
    if (op_dst == op_src) {
        return;
    } else if (op_dst.is_xyz() && op_src.is_xyz()) {
        gen.vmovdqa32(op_dst.get_xyz(), op_src.get_xyz());
    } else if (op_dst.is_xyz() && op_src.is_addr()) {
        gen.vmovdqu32(op_dst.get_xyz(), op_src.get_addr());
    } else if (op_dst.is_addr() && op_src.is_xyz()) {
        gen.vmovdqu32(op_dst.get_addr(), op_src.get_xyz());
    } else {
        COMPILE_ASSERT(
                false, "Invalid avx_movps: " << op_dst << ", " << op_src);
    }
}

} // namespace sc_xbyak
} // namespace sc
