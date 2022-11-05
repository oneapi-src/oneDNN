/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
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

#include <cassert>

#include "cpu/aarch64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace io {

io_conf_t::io_conf_t(const bool nt_stores_enabled)
    : nt_stores_enabled_(nt_stores_enabled) {}

io_tail_conf_t::io_tail_conf_t(const std::size_t simd_w,
        const std::size_t tail_size, const Xbyak_aarch64::PReg &tail_opmask,
        const int tail_vmm_mask_idx, const Xbyak_aarch64::XReg &reg_tmp,
        const Xbyak_aarch64::XReg &reg_tmp1)
    : simd_w_(simd_w)
    , tail_size_(tail_size)
    , tail_opmask_(tail_opmask)
    , tail_vmm_mask_idx_(tail_vmm_mask_idx)
    , reg_tmp_(reg_tmp)
    , reg_tmp1_(reg_tmp1) {}

io_saturation_conf_t::io_saturation_conf_t(const int vreg_zero_saturation_idx,
        const int vreg_saturation_ubound_idx,
        const Xbyak_aarch64::XReg &reg_tmp)
    : vreg_zero_saturation_idx_(vreg_zero_saturation_idx)
    , vreg_saturation_ubound_idx_(vreg_saturation_ubound_idx)
    , reg_tmp_(reg_tmp) {}

io_gather_conf_t::io_gather_conf_t(const std::size_t simd_w,
        const Xbyak_aarch64::PReg &full_opmask, const int full_vmm_mask_idx,
        const Xbyak_aarch64::XReg &reg_tmp, const Xbyak_aarch64::XReg &reg_tmp1,
        const utils::optional_t<int> &vmm_tmp_idx)
    : simd_w_(simd_w)
    , full_opmask_(full_opmask)
    , full_vmm_mask_idx_(full_vmm_mask_idx)
    , reg_tmp_(reg_tmp)
    , reg_tmp1_(reg_tmp1)
    , vmm_tmp_idx_(vmm_tmp_idx) {}

template <typename Vmm>
jit_io_helper_t<Vmm>::jit_io_helper_t(jit_generator *host, const cpu_isa_t &isa,
        const data_type_t &data_type, const io_conf_t &io_conf,
        const utils::optional_t<io_tail_conf_t> &tail_conf,
        const utils::optional_t<io_saturation_conf_t> &saturation_conf,
        const utils::optional_t<io_gather_conf_t> &gather_conf)
    : host_(host)
    , isa_(isa)
    , data_type_(data_type)
    , io_conf_(io_conf)
    , tail_conf_(tail_conf)
    , saturation_conf_(saturation_conf)
    , gather_conf_(gather_conf) {

    assert(utils::one_of(data_type_, data_type::f32, data_type::s8,
                   data_type::u8, data_type::s32)
            && "Supported data types f32, s8, u8, s32");

    static constexpr bool is_zmm
            = std::is_same<Vmm, Xbyak_aarch64::ZReg>::value;
    MAYBE_UNUSED(is_zmm);
    assert(IMPLICATION(!is_superset(isa_, sve_128), !is_zmm)
            && "This architecture does not support z registers.");
}

template <typename Vmm>
jit_io_helper_t<Vmm>::~jit_io_helper_t() = default;

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_opmask(
        const std::size_t how_many_bits_to_set,
        const Xbyak_aarch64::XReg &reg_tmp0,
        const Xbyak_aarch64::XReg &reg_tmp1, const Xbyak_aarch64::PReg &mask) {
    host_->mov_imm(reg_tmp0, 0);

    host_->mov_imm(host_->X_TMP_2, how_many_bits_to_set);
    host_->whilelt(mask.s, reg_tmp0, host_->X_TMP_2);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_tail_mask() {
    assert(tail_conf_.has_value() && "Config for tail processing is not set.");

    if (!tail_conf_->tail_size_) return;

    assert(is_superset(isa_, sve_128));

    prepare_opmask(tail_conf_->tail_size_, tail_conf_->reg_tmp_,
            tail_conf_->reg_tmp1_, tail_conf_->tail_opmask_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    assert(is_superset(isa_, sve_128));

    prepare_opmask(gather_conf_->simd_w_, gather_conf_->reg_tmp_,
            gather_conf_->reg_tmp1_, gather_conf_->full_opmask_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    if (isa_ == sve_256) {
        const Vmm vmm_mask = Vmm(gather_conf_->full_vmm_mask_idx_);
        host_->eor(Xbyak_aarch64::ZReg(vmm_mask.getIdx()).d,
                Xbyak_aarch64::ZReg(vmm_mask.getIdx()).d,
                Xbyak_aarch64::ZReg(vmm_mask.getIdx()).d);
        host_->mov(Xbyak_aarch64::ZReg(vmm_mask.getIdx()).s,
                host_->P_NOT_256 / Xbyak_aarch64::T_m, 0);
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_saturate_f32() const {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    if (utils::one_of(data_type_, data_type::u8, data_type::s8, data_type::s32))
        host_->init_saturate_f32(
                Vmm(saturation_conf_->vreg_zero_saturation_idx_),
                Vmm(saturation_conf_->vreg_saturation_ubound_idx_),
                saturation_conf_->reg_tmp_, data_type::f32, data_type_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::gather(const Xbyak_aarch64::XReg &src_reg,
        const Vmm &indices_vmm, const Vmm &dst_vmm, const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const Xbyak_aarch64::PReg mask
            = tail ? tail_conf_->tail_opmask_ : gather_conf_->full_opmask_;

    switch (data_type_) {
        case data_type::f32:
        case data_type::s32:
            host_->ld1w({dst_vmm.s}, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(
                            src_reg, indices_vmm.s, Xbyak_aarch64::SXTW));
            if (data_type_ == data_type::s32)
                convert_to_f32(dst_vmm, dst_vmm, data_type_);
            break;
        case data_type::s8:
            host_->ld1sb({dst_vmm.s}, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(
                            src_reg, indices_vmm.s, Xbyak_aarch64::SXTW));
            break;
        case data_type::u8:
            host_->ld1b({dst_vmm.s}, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(
                            src_reg, indices_vmm.s, Xbyak_aarch64::SXTW));
            break;
        default: assert(!"Unsupported data type.");
    }

    if (data_type_ != data_type::f32)
        convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_raw_vmm, const bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    Xbyak_aarch64::PReg mask
            = tail ? tail_conf_->tail_opmask_ : host_->P_ALL_ONE;

    switch (data_type_) {
        case data_type::f32:
            load_f32(src_addr, offt, dst_raw_vmm, tail, mask);
            break;
        case data_type::s32:
            load_s32(src_addr, offt, dst_raw_vmm, tail, mask);
            break;
        case data_type::s8:
        case data_type::u8: load_i8(src_addr, offt, dst_raw_vmm, mask); break;
        default: assert(!"Unsupported data type.");
    }
}

#if 0
/**
* load_bytes is the utility function to facilitate loading of
* load_size (0 <= load_size <= 32) many contiguous bytes into the Xmm/Ymm
* register from the memory referenced by ptr[reg + offset] address.
*
* Functionally, invocation of load_bytes is equivalent to
* the following loop:
*
* for (int idx = 0; idx < load_size; ++idx)
*     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
*
* TODO: Add an option to zero-out unloaded bytes in the Xmm register.
* TODO: Add an option for unsafe_load wherein one could read outside the
* provided memory buffer so as to minimize the total number of read
* memory instructions.
*/
static void load_bytes(jit_generator *host, const Xbyak_aarch64::VReg &vmm,
        const Xbyak_aarch64::XReg reg_addr, int load_size) {
    if (load_size == 32) {
        host->not_(host->P_TMP.b, host->P_ALL_ONE / Xbyak_aarch64::T_z,
                host->P_NOT_256.b);
        host->ld1w(Xbyak_aarch64::ZRegS(vmm.getIdx()),
                host->P_TMP / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(reg_addr));
        return;
    }
    int start_bytes = 0;
    int bytes_to_load = load_size;

    if (load_size > 16) {
        start_bytes = 16;
        bytes_to_load -= 16;
    }

    if (bytes_to_load >= 8 && bytes_to_load < 16) {
        host->add_imm(
                host->X_DEFAULT_ADDR, reg_addr, start_bytes, host->X_TMP_0);
        host->ldr(host->X_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
        host->mov(vmm.b16, vmm.b16);
        host->ins(Xbyak_aarch64::VReg2D(vmm.getIdx())[0], host->X_TMP_0);
    } else if (bytes_to_load == 16) {
        host->add_imm(
                host->X_DEFAULT_ADDR, reg_addr, start_bytes, host->X_TMP_0);
        host->ldr(Xbyak_aarch64::QReg(vmm.getIdx()),
                Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
    }

    host->add_imm(host->X_DEFAULT_ADDR, reg_addr, start_bytes, host->X_TMP_0);
    switch (bytes_to_load) {
        case 0: break;
        case 1:
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[0], host->W_TMP_0);
            break;
        case 2:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[0], host->W_TMP_0);
            break;
        case 3:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[0], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 2);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[2], host->W_TMP_0);
            break;
        case 4:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[0], host->W_TMP_0);
            break;
        case 5:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[0], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 4);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[4], host->W_TMP_0);
            break;
        case 6:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[0], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 4);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[2], host->W_TMP_0);
            break;
        case 7:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[0], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 4);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[2], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 6);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[6], host->W_TMP_0);
            break;
        case 8: break;
        case 9:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[8], host->W_TMP_0);
            break;
        case 10:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[4], host->W_TMP_0);
            break;
        case 11:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[4], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 10);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[10], host->W_TMP_0);
            break;
        case 12:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[2], host->W_TMP_0);
            break;
        case 13:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[2], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 12);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[12], host->W_TMP_0);
            break;
        case 14:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[2], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 12);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[6], host->W_TMP_0);
            break;
        case 15:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[2], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 12);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[6], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 14);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[14], host->W_TMP_0);
            break;
        case 16: break;
        default: assert(!"improper load size");
    }

    if (load_size > 16) {
        host->str(host->z31,
                Xbyak_aarch64::ptr(
                        host->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
        const Xbyak_aarch64::ZReg z_vmm(vmm.getIdx());
        const Xbyak_aarch64::ZReg z_tmp(host->z31.getIdx());
        host->ptrue(host->P_TMP.d, Xbyak_aarch64::VL2);
        host->mov(z_tmp.d, z_vmm.d);
        host->splice(z_tmp.d, host->P_TMP.d, z_vmm.d);
        host->mov(z_vmm.d, z_tmp.d);
        host->mov(z_vmm.s, host->P_NOT_256 / Xbyak_aarch64::T_m, 0);
        host->ld1w(z_tmp.d, host->P_ALL_ONE / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(reg_addr));
        host->ptrue(host->P_TMP.d, Xbyak_aarch64::VL2);
        host->sel(z_vmm.d, host->P_TMP, z_tmp.d, z_vmm.d);
        host->mov(z_vmm.s, host->P_NOT_256 / Xbyak_aarch64::T_m, 0);
        host->ldr(host->z31,
                Xbyak_aarch64::ptr(
                        host->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
    }
}

/**
* load_bytes_to_dword_extension is the utility function to facilitate
* loading of load_size (0 <= load_size <= 16) many contiguous bytes in
* the Xmm register from the memory referenced by ptr[reg + offset]
* address and then do signed/zero extension of those to double words.
*
* Functionally, invocation of load_bytes_to_dword_extension is equivalent
* to the following:
*
* for (int idx = 0; idx < load_size; ++idx)
*     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
* if (is_signed) vpmovsxbd(vmm, vmm); else vpmovzxbd(vmm, vmm);
*
* Valid values for the load_size variable are:
* [0..4] for XMM version of the function
* [0..8] for YMM version of the function.
* TODO: Implement this routine for every ISA.
*/
static void load_bytes_to_dword_extension(jit_generator *host,
        const Xbyak_aarch64::VReg &vmm, const Xbyak_aarch64::XReg &reg_addr,
        bool is_signed, int load_size) {
    if (host->cpu_sveLen == Xbyak_aarch64::util::SVE_128) {
        assert(load_size >= 0 && load_size <= 8);
    } else if (host->cpu_sveLen == Xbyak_aarch64::util::SVE_256) {
        assert(load_size >= 0 && load_size <= 4);
    } else {
        assert(!"routine is not supported for the current isa");
    }
    host->str(host->z31,
            Xbyak_aarch64::ptr(
                    host->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
    // For load_size == 8/4, do load/extension in one go
    const Xbyak_aarch64::ZReg z_tmp(host->z31.getIdx());
    if (load_size == 8) {
        const Xbyak_aarch64::ZReg z_vmm(vmm.getIdx());
        if (is_signed) {
            host->ld1sb(z_tmp.s, host->P_NOT_256, Xbyak_aarch64::ptr(reg_addr));
        } else {
            host->ld1b(z_tmp.s, host->P_NOT_256, Xbyak_aarch64::ptr(reg_addr));
        }
    } else if (load_size == 4) {
        const Xbyak_aarch64::ZReg z_vmm(vmm.getIdx());
        if (is_signed) {
            host->ld1sb(z_tmp.s, host->P_NOT_128, Xbyak_aarch64::ptr(reg_addr));
        } else {
            host->ld1b(z_tmp.s, host->P_NOT_128, Xbyak_aarch64::ptr(reg_addr));
        }
    } else {
        load_bytes(host, vmm, reg_addr, load_size);
        if (is_signed) {
            host->mov(z_tmp.d, host->P_ALL_ONE,
                    Xbyak_aarch64::ZRegD(vmm.getIdx()));
            host->sxtl(Xbyak_aarch64::VReg8H(vmm.getIdx()),
                    Xbyak_aarch64::VReg8B(vmm.getIdx()));
            host->sxtl(Xbyak_aarch64::VReg4S(vmm.getIdx()),
                    Xbyak_aarch64::VReg4H(vmm.getIdx()));
            host->mov(Xbyak_aarch64::ZRegD(vmm.getIdx()), host->P_NOT_128,
                    z_tmp.d);
        } else {
            host->mov(z_tmp.d, host->P_ALL_ONE,
                    Xbyak_aarch64::ZRegD(vmm.getIdx()));
            host->uxtl(Xbyak_aarch64::VReg8H(vmm.getIdx()),
                    Xbyak_aarch64::VReg8B(vmm.getIdx()));
            host->uxtl(Xbyak_aarch64::VReg4S(vmm.getIdx()),
                    Xbyak_aarch64::VReg4H(vmm.getIdx()));
            host->mov(Xbyak_aarch64::ZRegD(vmm.getIdx()), host->P_NOT_128,
                    z_tmp.d);
        }
    }
    host->ldr(host->z31,
            Xbyak_aarch64::ptr(
                    host->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
}
template <typename Vmm>
void load_data(jit_generator *host, data_type_t type_in, const Vmm &vmm,
        const Xbyak_aarch64::XReg &src_addr, int load_size) {

    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            load_bytes(host, Xbyak_aarch64::VReg(vmm.getIdx()), src_addr,
                    sizeof(int32_t) * load_size);
            break;
        case data_type::s8:
        case data_type::u8:
            load_bytes_to_dword_extension(host,
                    Xbyak_aarch64::VReg(vmm.getIdx()), src_addr,
                    type_in == data_type::s8, load_size);
            break;
        default: assert(!"unsupported source data type");
    }
}
#endif

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_f32(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_vmm, const bool tail,
        const Xbyak_aarch64::PReg &mask) {

    host_->ld1w(
            dst_vmm.s, mask / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(src_addr));
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::load_f32(
        const Xbyak_aarch64::XReg &src_addr, const int offt,
        const Xbyak_aarch64::VReg &dst_vmm, const bool tail,
        const Xbyak_aarch64::PReg &mask) {
    UNUSED(src_addr);
    UNUSED(offt);
    UNUSED(dst_vmm);
    UNUSED(tail);

    assert(!"under construction");
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_s32(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_vmm, const bool tail,
        const Xbyak_aarch64::PReg &mask) {
    host_->ld1w(
            dst_vmm.s, mask / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(src_addr));
    host_->scvtf(dst_vmm.s, host_->P_TMP / Xbyak_aarch64::T_m, dst_vmm.s);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_i8(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_vmm, const Xbyak_aarch64::PReg &mask) {

    if (data_type_ == data_type::s8)
        host_->ld1sb(dst_vmm.s, mask / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(src_addr));
    else
        host_->ld1b(dst_vmm.s, mask / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(src_addr));

    convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store(const Vmm &src_raw_vmm,
        const Xbyak_aarch64::XReg &dst_raw_addr, const int offt,
        const bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");
    assert(!(tail && io_conf_.nt_stores_enabled_)
            && "Usage of non-temporal stores with tail leads to a general-protection exception.");

    Xbyak_aarch64::PReg mask
            = tail ? tail_conf_->tail_opmask_ : host_->P_ALL_ONE;
    const bool is_i8 = utils::one_of(data_type_, data_type::s8, data_type::u8);

    if (data_type_ == data_type::s32 || is_i8) saturate(src_raw_vmm);

    switch (data_type_) {
        case data_type::f32:
        case data_type::s32:
            store_f32(src_raw_vmm, dst_raw_addr, offt, tail, mask);
            break;
        case data_type::s8:
        case data_type::u8:
            store_i8(src_raw_vmm, dst_raw_addr, offt, mask);
            break;
        default: assert(!"Unsupported data type.");
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::saturate(const Vmm &vmm) {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    host_->saturate_f32(vmm, Vmm(saturation_conf_->vreg_zero_saturation_idx_),
            Vmm(saturation_conf_->vreg_saturation_ubound_idx_), data_type_,
            host_->P_ALL_ONE);
    host_->frintn(vmm.s, host_->P_ALL_ONE / Xbyak_aarch64::T_m,
            vmm.s); // Round to nearest even
    host_->fcvtzs(vmm.s, host_->P_ALL_ONE / Xbyak_aarch64::T_m, vmm.s);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_f32(const Vmm &src_vmm,
        const Xbyak_aarch64::XReg &dst_addr, const int offt, const bool tail,
        const Xbyak_aarch64::PReg &mask) {
    if (io_conf_.nt_stores_enabled_) {
        host_->stnt1d(Xbyak_aarch64::ZRegD(src_vmm.getIdx()), mask,
                Xbyak_aarch64::ptr(dst_addr));
    } else if (!is_superset(isa_, sve_128) && tail) {
        // ASIMD 128-bit
        switch (tail_conf_->tail_size_) {
            case 1:
                host_->str(Xbyak_aarch64::SReg(src_vmm.getIdx()),
                        Xbyak_aarch64::ptr(dst_addr));
                break;
            case 2:
                host_->str(Xbyak_aarch64::DReg(src_vmm.getIdx()),
                        Xbyak_aarch64::ptr(dst_addr));
                break;
            case 3:
                host_->str(Xbyak_aarch64::DReg(src_vmm.getIdx()),
                        Xbyak_aarch64::ptr(dst_addr));
                host_->add(dst_addr, dst_addr, 8);
                host_->st1(Xbyak_aarch64::VReg4S(src_vmm.getIdx())[2],
                        Xbyak_aarch64::ptr(dst_addr));
                host_->sub(dst_addr, dst_addr, 8);
                break;
            default: assert(!"unreachable");
        }
    } else {
        host_->st1w(Xbyak_aarch64::ZRegS(src_vmm.getIdx()), mask,
                Xbyak_aarch64::ptr(dst_addr));
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_i8_sdb(Xbyak_aarch64::XReg addr,
        const Vmm &src_vmm, const Xbyak_aarch64::PReg &mask) {
    host_->str(host_->z31,
            Xbyak_aarch64::ptr(
                    host_->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
    const Xbyak_aarch64::ZReg z_tmp(host_->z31.getIdx());
    host_->mov(z_tmp.d, Xbyak_aarch64::ZRegD(src_vmm.getIdx()));
    host_->smin(z_tmp.s, 127);
    host_->smax(z_tmp.s, -128);
    host_->st1b(z_tmp.s, mask, Xbyak_aarch64::ptr(addr));
    host_->ldr(host_->z31,
            Xbyak_aarch64::ptr(
                    host_->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
}
template <typename Vmm>
void jit_io_helper_t<Vmm>::store_i8_udb(Xbyak_aarch64::XReg addr,
        const Vmm &src_vmm, const Xbyak_aarch64::PReg &mask) {
    host_->str(host_->z31,
            Xbyak_aarch64::ptr(
                    host_->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
    const Xbyak_aarch64::ZReg z_tmp(host_->z31.getIdx());
    host_->mov(z_tmp.d, Xbyak_aarch64::ZRegD(src_vmm.getIdx()));
    host_->umin(z_tmp.s, 255);
    host_->st1b(z_tmp.s, mask, Xbyak_aarch64::ptr(addr));
    host_->ldr(host_->z31,
            Xbyak_aarch64::ptr(
                    host_->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
}
template <typename Vmm>
void jit_io_helper_t<Vmm>::store_i8(const Vmm &src_vmm,
        const Xbyak_aarch64::XReg &dst_addr, const int offt,
        const Xbyak_aarch64::PReg &mask) {
    using namespace std::placeholders;
    static constexpr bool is_zmm
            = std::is_same<Vmm, Xbyak_aarch64::ZReg>::value;

    auto store_i8_fn = data_type_ == data_type::s8
            ? std::bind(&jit_io_helper_t::store_i8_sdb, this, _1, _2, _3)
            : std::bind(&jit_io_helper_t::store_i8_udb, this, _1, _2, _3);

    if (io_conf_.nt_stores_enabled_ && is_zmm) {
        host_->not_(
                host_->P_TMP.b, mask / Xbyak_aarch64::T_z, host_->P_NOT_128.b);
        host_->stnt1d(Xbyak_aarch64::ZRegD(src_vmm.getIdx()), mask,
                Xbyak_aarch64::ptr(dst_addr));
    } else {
        store_i8_fn(dst_addr, src_vmm, mask);
    }
}

template <typename Vmm>
void uni_vpmovsxbd(jit_generator *host_, const Vmm &dst, const Vmm &src) {
    Xbyak_aarch64::ZReg z_dst(dst.getIdx());
    Xbyak_aarch64::ZReg z_src(src.getIdx());
    host_->zip1(z_dst.b, z_src.b, z_src.b);
    host_->zip1(z_dst.h, z_dst.h, z_dst.h);
    host_->sxtb(z_dst.s, host_->P_ALL_ONE / Xbyak_aarch64::T_m, z_dst.s);
}

template <typename Vmm>
void uni_vpmovzxbd(jit_generator *host_, const Vmm &dst, const Vmm &src) {
    Xbyak_aarch64::ZReg z_dst(dst.getIdx());
    Xbyak_aarch64::ZReg z_src(src.getIdx());
    host_->zip1(z_dst.b, z_src.b, z_src.b);
    host_->zip1(z_dst.h, z_dst.h, z_dst.h);
    host_->uxtb(z_dst.s, host_->P_ALL_ONE / Xbyak_aarch64::T_m, z_dst.s);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::convert_to_f32(const Vmm &dst_vmm,
        const Vmm &src_vmm, const data_type_t src_data_type) {
    switch (src_data_type) {
        case data_type::f32: // Do nothing
            break;
        case data_type::s32: {
            assert(dst_vmm.getIdx() == src_vmm.getIdx());
            host_->uni_scvtf(dst_vmm.s, dst_vmm.s);
            break;
        }
        case data_type::s8: {
            uni_vpmovsxbd(host_, dst_vmm, src_vmm);
            host_->uni_scvtf(dst_vmm.s, dst_vmm.s);
            break;
        }
        case data_type::u8: {
            uni_vpmovzxbd(host_, dst_vmm, src_vmm);
            host_->uni_scvtf(dst_vmm.s, dst_vmm.s);
            break;
        }
        default: assert(!"Unsupported data type.");
    }
}

template <typename Vmm>
void uni_vbroadcastss(
        jit_generator *host_, const Vmm &dst, const Xbyak_aarch64::XReg &src) {
    uint8_t dstIdx = dst.getIdx();
    host_->ld1rw(Xbyak_aarch64::ZRegS(dstIdx),
            host_->P_ALL_ONE / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(src));
}
template <typename Vmm>
void uni_vbroadcastss(
        jit_generator *host_, const Vmm &dst, const Xbyak_aarch64::VReg &src) {
    uint8_t dstIdx = dst.getIdx();
    uint8_t srcIdx = src.getIdx();
    host_->dup(Xbyak_aarch64::ZRegS(dstIdx), Xbyak_aarch64::ZRegS(srcIdx)[0]);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::broadcast(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_vmm) {
    switch (data_type_) {
        case data_type::f32: uni_vbroadcastss(host_, dst_vmm, src_addr); break;
        case data_type::s32: {
            if (is_superset(isa_, sve_512)) {
                if (host_->cpu_sveLen == sve_128) {
                    host_->ld1(Xbyak_aarch64::VReg(dst_vmm.getIdx()).s4,
                            Xbyak_aarch64::ptr(src_addr));
                } else {
                    host_->ld1w(Xbyak_aarch64::ZRegD(dst_vmm.getIdx()),
                            host_->P_ALL_ONE / Xbyak_aarch64::T_z,
                            Xbyak_aarch64::ptr(src_addr));
                    host_->mov(host_->P_TMP.b, host_->P_ALL_ONE.b);
                    host_->scvtf(Xbyak_aarch64::ZReg(dst_vmm.getIdx()).s,
                            host_->P_TMP / Xbyak_aarch64::T_m,
                            Xbyak_aarch64::ZReg(dst_vmm.getIdx()).s);
                    ;
                    if (host_->cpu_sveLen == sve_256)
                        host_->mov(Xbyak_aarch64::ZReg(dst_vmm.getIdx()).s,
                                host_->P_NOT_256 / Xbyak_aarch64::T_m, 0);
                }
            } else {
                uni_vbroadcastss(host_, dst_vmm, src_addr);
                convert_to_f32(dst_vmm, dst_vmm, data_type_);
            }
            break;
        }
        case data_type::s8:
        case data_type::u8: {
            const Xbyak_aarch64::VReg dst_xmm {dst_vmm.getIdx()};
            host_->ldrb(host_->W_TMP_0, Xbyak_aarch64::ptr(src_addr));
            host_->mov(dst_xmm.b16, dst_xmm.b16);
            host_->ins(dst_xmm.b16[0], host_->W_TMP_0);
            convert_to_f32(dst_vmm, dst_vmm, data_type_);
            uni_vbroadcastss(host_, dst_vmm, dst_xmm);

            break;
        }
        default: assert(!"Unsupported data type.");
    }
}

template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::jit_io_multi_dt_helper_t(jit_generator *host,
        const cpu_isa_t &isa, const data_types_t &data_types,
        const io_conf_t &io_conf,
        const utils::optional_t<io_tail_conf_t> &tail_conf,
        const std::map<data_type_t, io_saturation_conf_t> &saturation_confs,
        const utils::optional_t<io_gather_conf_t> &gather_conf) {
    assert(!data_types.empty());
    for (const auto &dt : data_types) {
        // can be replaced by try_emplace from C++17
        if (storage_.find(dt) == storage_.cend()) {

            const auto saturation_conf = saturation_confs.find(dt);
            const bool store_saturation_needed
                    = saturation_conf != saturation_confs.cend();

            storage_.emplace(dt,
                    std::make_shared<jit_io_helper_t<Vmm>>(host, isa, dt,
                            io_conf, tail_conf,
                            store_saturation_needed ? utils::optional_t<
                                    io_saturation_conf_t> {saturation_conf
                                                                   ->second}
                                                    : utils::nullopt,
                            gather_conf));
        }
    }
}

template <typename Vmm>
std::shared_ptr<jit_io_helper_t<Vmm>> jit_io_multi_dt_helper_t<Vmm>::at(
        const data_type_t dt) const {
    const auto it = storage_.find(dt);
    if (it != storage_.cend()) return it->second;

    return nullptr;
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::prepare_tail_mask() {
    return storage_.cbegin()->second->prepare_tail_mask();
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::prepare_full_mask() {
    return storage_.cbegin()->second->prepare_full_mask();
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::init_saturate_f32(
        const data_types_t &store_data_types) {
    for (const auto &dt : store_data_types) {
        const auto it = storage_.find(dt);
        if (it != storage_.cend()) {
            if (it->second->saturation_conf_.has_value())
                it->second->init_saturate_f32();
        }
    }
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::init_full_mask() {
    return storage_.cbegin()->second->init_full_mask();
}

template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::~jit_io_multi_dt_helper_t() = default;

template class jit_io_helper_t<Xbyak_aarch64::ZReg>;
template class jit_io_multi_dt_helper_t<Xbyak_aarch64::ZReg>;

} // namespace io
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
