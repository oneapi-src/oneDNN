/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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
#include <type_traits>

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace io {

io_conf_t::io_conf_t(const bool nt_stores_enabled)
    : nt_stores_enabled_(nt_stores_enabled) {}

io_tail_conf_t::io_tail_conf_t(const std::size_t simd_w,
        const std::size_t tail_size, const Xbyak::Opmask &tail_opmask,
        const int tail_vmm_mask_idx, const Xbyak::Reg64 &reg_tmp)
    : simd_w_(simd_w)
    , tail_size_(tail_size)
    , tail_opmask_(tail_opmask)
    , tail_vmm_mask_idx_(tail_vmm_mask_idx)
    , reg_tmp_(reg_tmp) {}

io_tail_conf_t::io_tail_conf_t(const std::size_t simd_w,
        const std::size_t tail_size, int tail_opmask_idx,
        const int tail_vmm_mask_idx, const Xbyak::Reg64 &reg_tmp)
    : simd_w_(simd_w)
    , tail_size_(tail_size)
    , tail_opmask_(Xbyak::Opmask(tail_opmask_idx))
    , tail_vmm_mask_idx_(tail_vmm_mask_idx)
    , reg_tmp_(reg_tmp) {}

io_emu_bf16_conf_t::io_emu_bf16_conf_t(const Xbyak::Zmm &bf16_emu_reserv_1,
        const Xbyak::Zmm &bf16_emu_reserv_2,
        const Xbyak::Zmm &bf16_emu_reserv_3, const Xbyak::Reg64 &reg_tmp,
        const Xbyak::Zmm &bf16_emu_reserv_4)
    : bf16_emu_reserv_1_(bf16_emu_reserv_1)
    , bf16_emu_reserv_2_(bf16_emu_reserv_2)
    , bf16_emu_reserv_3_(bf16_emu_reserv_3)
    , reg_tmp_(reg_tmp)
    , bf16_emu_reserv_4_(bf16_emu_reserv_4) {}

io_emu_bf16_conf_t::io_emu_bf16_conf_t(int bf16_emu_reserv_1_idx,
        int bf16_emu_reserv_2_idx, int bf16_emu_reserv_3_idx,
        const Xbyak::Reg64 &reg_tmp, int bf16_emu_reserv_4_idx)
    : bf16_emu_reserv_1_(Xbyak::Zmm(bf16_emu_reserv_1_idx))
    , bf16_emu_reserv_2_(Xbyak::Zmm(bf16_emu_reserv_2_idx))
    , bf16_emu_reserv_3_(Xbyak::Zmm(bf16_emu_reserv_3_idx))
    , reg_tmp_(reg_tmp)
    , bf16_emu_reserv_4_(Xbyak::Zmm(bf16_emu_reserv_4_idx)) {}

io_emu_fp8_conf_t::io_emu_fp8_conf_t(const Xbyak::Zmm &fp8_emu_reserv_1,
        const Xbyak::Zmm &fp8_emu_reserv_2, const Xbyak::Zmm &fp8_emu_reserv_3,
        const Xbyak::Zmm &fp8_emu_reserv_4, const Xbyak::Zmm &fp8_emu_reserv_5,
        const Xbyak::Opmask &kmask_aux, const Xbyak::Reg64 &reg_tmp)
    : fp8_emu_reserv_1_(fp8_emu_reserv_1)
    , fp8_emu_reserv_2_(fp8_emu_reserv_2)
    , fp8_emu_reserv_3_(fp8_emu_reserv_3)
    , fp8_emu_reserv_4_(fp8_emu_reserv_4)
    , fp8_emu_reserv_5_(fp8_emu_reserv_5)
    , kmask_aux_(kmask_aux)
    , reg_tmp_(reg_tmp) {}

io_emu_fp8_conf_t::io_emu_fp8_conf_t(int fp8_emu_reserv_1_idx,
        int fp8_emu_reserv_2_idx, int fp8_emu_reserv_3_idx,
        int fp8_emu_reserv_4_idx, int fp8_emu_reserv_5_idx,
        int fp8_emu_kmask_aux_idx, const Xbyak::Reg64 &reg_tmp)
    : fp8_emu_reserv_1_(Xbyak::Zmm(fp8_emu_reserv_1_idx))
    , fp8_emu_reserv_2_(Xbyak::Zmm(fp8_emu_reserv_2_idx))
    , fp8_emu_reserv_3_(Xbyak::Zmm(fp8_emu_reserv_3_idx))
    , fp8_emu_reserv_4_(Xbyak::Zmm(fp8_emu_reserv_4_idx))
    , fp8_emu_reserv_5_(Xbyak::Zmm(fp8_emu_reserv_5_idx))
    , kmask_aux_(Xbyak::Opmask(fp8_emu_kmask_aux_idx))
    , reg_tmp_(reg_tmp) {}

io_saturation_conf_t::io_saturation_conf_t(const int vreg_zero_saturation_idx,
        const int vreg_saturation_ubound_idx, const Xbyak::Reg64 &reg_tmp)
    : vreg_zero_saturation_idx_(vreg_zero_saturation_idx)
    , vreg_saturation_ubound_idx_(vreg_saturation_ubound_idx)
    , reg_tmp_(reg_tmp) {}

io_gather_conf_t::io_gather_conf_t(const std::size_t simd_w,
        const Xbyak::Opmask &full_opmask, const int full_vmm_mask_idx,
        const Xbyak::Reg64 &reg_tmp, const Xbyak::Reg64 &reg_tmp1,
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
        const utils::optional_t<io_emu_bf16_conf_t> &bf16_conf,
        const utils::optional_t<io_saturation_conf_t> &saturation_conf,
        const utils::optional_t<io_gather_conf_t> &gather_conf,
        const utils::optional_t<io_emu_fp8_conf_t> &fp8_conf)
    : host_(host)
    , isa_(isa)
    , data_type_(data_type)
    , bf16_supported_(is_data_type_supported(data_type::bf16))
    , f16_supported_(is_data_type_supported(data_type::f16))
    , fp8_supported_(
              utils::one_of(true, is_data_type_supported(data_type::f8_e5m2),
                      is_data_type_supported(data_type::f8_e4m3)))
    , bf16_emu_(nullptr)
    , io_conf_(io_conf)
    , tail_conf_(tail_conf)
    , bf16_conf_(bf16_conf)
    , saturation_conf_(saturation_conf)
    , gather_conf_(gather_conf)
    , fp8_conf_(fp8_conf) {

    if (data_type_ == data_type::bf16
            && !(is_superset(isa_, avx512_core_bf16)
                    || is_superset(isa_, avx2_vnni_2))) {
        assert(bf16_conf.has_value()
                && "Config for bf16 emulation is not set.");
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(host_,
                bf16_conf->bf16_emu_reserv_1_, bf16_conf->bf16_emu_reserv_2_,
                bf16_conf->bf16_emu_reserv_3_, bf16_conf->reg_tmp_,
                bf16_conf->bf16_emu_reserv_4_);
    }

    if (utils::one_of(data_type_, data_type::f8_e5m2, data_type::f8_e4m3)
            && fp8_supported_) {
        assert(fp8_conf.has_value() && "Config for fp8 emulation is not set.");
        switch (data_type_) {
            case data_type::f8_e5m2:
                fp8_emu_ = utils::make_unique<fp8_emulation_e5m2_t>(host_,
                        fp8_conf->fp8_emu_reserv_1_,
                        fp8_conf->fp8_emu_reserv_2_,
                        fp8_conf->fp8_emu_reserv_3_, fp8_conf->kmask_aux_,
                        fp8_conf->reg_tmp_);
                break;
            case data_type::f8_e4m3:
                fp8_emu_ = utils::make_unique<fp8_emulation_e4m3_t>(host_,
                        fp8_conf->fp8_emu_reserv_1_,
                        fp8_conf->fp8_emu_reserv_2_,
                        fp8_conf->fp8_emu_reserv_3_,
                        fp8_conf->fp8_emu_reserv_4_,
                        fp8_conf->fp8_emu_reserv_5_, fp8_conf->reg_tmp_);
                break;
            default: assert(!"Unreachable.");
        }
    }

    assert(utils::one_of(data_type_, data_type::f16, data_type::bf16,
                   data_type::f32, data_type::f8_e5m2, data_type::f8_e4m3, data_type::s8, data_type::u8, data_type::s32)
            && is_data_type_supported(data_type_)
            && "Supported data types f16, bf16, f32, f8_e5m2, f8_e4m3, s8, u8, s32");

    /*
     * vpmovsxbd, vpmovzxbd for AVX are defined only for XMM. Since AVX2
     * they are defined also for YMM. In order to avoid workaround with
     * potential performance penalty AVX with s8u8 disabled with YMM.
     */
    static constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    const bool is_avx_u8s8 = (isa_ == avx
            && utils::one_of(data_type_, data_type::s8, data_type::u8));
    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_avx_u8s8);

    assert(IMPLICATION(is_avx_u8s8, is_xmm)
            && "s8u8 with AVX should be used with XMM vreg");

    static constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;
    MAYBE_UNUSED(is_zmm);
    assert(IMPLICATION(!is_superset(isa_, avx512_core), !is_zmm)
            && "This architecture does not support zmms.");
}

template <typename Vmm>
jit_io_helper_t<Vmm>::~jit_io_helper_t() = default;

template <typename Vmm>
bool jit_io_helper_t<Vmm>::is_data_type_supported(const data_type_t dt) {
    switch (dt) {
        case data_type::f32:
        case data_type::s32:
        case data_type::u8:
        case data_type::s8: return true;
        case data_type::bf16:
            return is_superset(isa_, avx512_core) || isa_ == avx2_vnni_2;
        case data_type::f16:
            return is_superset(isa_, avx512_core_fp16) || isa_ == avx2_vnni_2;
        case data_type::f8_e4m3:
        case data_type::f8_e5m2: return is_superset(isa_, avx512_core_fp16);
        default: assert(!"Unsupported data type");
    }
    return false;
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_bf16() {
    if (bf16_emu_) {
        assert(bf16_conf_.has_value()
                && "Config for bf16 emulation is not set.");
        bf16_emu_->init_vcvtneps2bf16();
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_opmask(
        const std::size_t how_many_bits_to_set, const Xbyak::Reg64 &reg_tmp,
        const Xbyak::Opmask &mask) {
    const int mask_f32 = (1 << how_many_bits_to_set) - 1;
    const Xbyak::Reg32 regw_tmp = reg_tmp.cvt32();
    host_->mov(regw_tmp, mask_f32);
    host_->kmovw(mask, regw_tmp);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_vmm_mask(
        const std::size_t how_many_bits_to_set, const std::size_t simd_w,
        const Xbyak::Reg64 &reg_tmp, const Vmm &mask) {
    static const uint32_t mask_f32[14]
            = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

    if (how_many_bits_to_set < simd_w) {
        host_->mov(reg_tmp,
                reinterpret_cast<size_t>(&mask_f32[7 - how_many_bits_to_set]));
        host_->uni_vmovups(mask, host_->ptr[reg_tmp]);
    } else if (how_many_bits_to_set == simd_w) {
        host_->uni_vcmpps(mask, mask, mask, jit_generator::_cmp_eq_oq);
    } else {
        assert(!"Can't set so many bits.");
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_i8_data_to_store(const Vmm &i8_vmm) {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    static constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;

    host_->uni_vpackssdw(
            i8_vmm, i8_vmm, Vmm(saturation_conf_->vreg_zero_saturation_idx_));
    if (is_ymm) {
        // dst[63:0] = src[63:0]
        // dst[127:64] = src[191:128]
        // dst[191:128] = src[127:64]
        // dst[255:192] = src[127:64]
        const auto src_ymm = Xbyak::Ymm(i8_vmm.getIdx());
        host_->vpermq(src_ymm, src_ymm, 0x58);
    }

    if (data_type_ == data_type::s8)
        host_->uni_vpacksswb(i8_vmm, i8_vmm,
                Vmm(saturation_conf_->vreg_zero_saturation_idx_));
    else
        host_->uni_vpackuswb(i8_vmm, i8_vmm,
                Vmm(saturation_conf_->vreg_zero_saturation_idx_));
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_xf16_data_to_store(const Vmm &vmm) {
    const auto &cvt_lower_vmm =
            typename vreg_traits<Vmm>::Vmm_lower_t(vmm.getIdx());

    if (data_type_ == data_type::bf16)
        host_->vcvtneps2bf16(cvt_lower_vmm, vmm, host_->get_encoding());
    else
        host_->uni_vcvtps2phx(cvt_lower_vmm, vmm);
}

template <>
void jit_io_helper_t<Xbyak::Zmm>::emu_gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Zmm &indices_vmm, const Xbyak::Zmm &dst_vmm,
        const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(gather_conf_->vmm_tmp_idx_.has_value()
            && "Temporary vreg is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const Xbyak::Xmm xmm_tmp = Xbyak::Xmm(gather_conf_->full_vmm_mask_idx_);
    const Xbyak::Xmm xmm_dst = Xbyak::Xmm(*gather_conf_->vmm_tmp_idx_);
    const Xbyak::Ymm dst_ymm;

    host_->mov(gather_conf_->reg_tmp_, 0);
    host_->mov(gather_conf_->reg_tmp1_, src_reg);

    // The conversion of bf16->f32 here is split into two parts.
    // Here while loading words of bf16, the words are interleaved,
    // and in convert_to_f32, they are shifted-left to finally convert to f32
    // For f16 we do not need such interleaving.
    const int xmm_size_elem = (data_type_ == data_type::f16) ? 8 : 4;
    const int number_of_xmms = tail
            ? utils::div_up(tail_conf_->tail_size_, xmm_size_elem)
            : utils::div_up(gather_conf_->simd_w_, xmm_size_elem);
    const int num_indices_in_xmm = 16 / sizeof(int);
    for (int i = 0, idx = 0; i < number_of_xmms; i++) {

        const int number_of_values_to_load = i == number_of_xmms - 1 && tail
                        && tail_conf_->tail_size_ % xmm_size_elem != 0
                ? tail_conf_->tail_size_ % xmm_size_elem
                : xmm_size_elem;
        for (int j = 0; j < number_of_values_to_load; j++) {

            if (j % num_indices_in_xmm == 0)
                host_->vextractf32x4(xmm_tmp, indices_vmm, idx++);
            host_->vpextrd(gather_conf_->reg_tmp_.cvt32(), xmm_tmp, j);
            host_->add(src_reg, gather_conf_->reg_tmp_);
            switch (data_type_) {
                case data_type::f16:
                    assert(f16_supported_ && "Unsupported data type.");
                    host_->vpinsrw(xmm_dst, xmm_dst, host_->ptr[src_reg], j);
                    break;
                case data_type::bf16:
                    assert(bf16_supported_ && "Unsupported data type.");
                    host_->vpinsrw(
                            xmm_dst, xmm_dst, host_->ptr[src_reg], j * 2);
                    break;
                case data_type::s8:
                case data_type::u8:
                case data_type::f8_e4m3:
                case data_type::f8_e5m2: {
                    assert(IMPLICATION(
                                   utils::one_of(data_type_, data_type::f8_e4m3,
                                           data_type::f8_e5m2),
                                   fp8_supported_)
                            && "Unsupported data type.");
                    host_->vpinsrb(xmm_dst, xmm_dst, host_->ptr[src_reg],
                            i * xmm_size_elem + j);
                    break;
                }
                default: assert(!"Unsupported data type.");
            }
            host_->mov(src_reg, gather_conf_->reg_tmp1_);
        }
        if (data_type_ == data_type::bf16) {
            host_->vinsertf32x4(dst_vmm, dst_vmm, xmm_dst, i);
            host_->vpxord(xmm_dst, xmm_dst, xmm_dst);
        } else if (data_type_ == data_type::f16) {
            host_->vinsertf32x4(dst_ymm, dst_ymm, xmm_dst, i);
            host_->vpxord(xmm_dst, xmm_dst, xmm_dst);
        }
    }

    if (data_type_ == data_type::bf16)
        convert_to_f32(dst_vmm, dst_vmm, data_type_);
    else if (data_type_ == data_type::f16)
        convert_to_f32(dst_vmm, dst_ymm, data_type_);
    else if (utils::one_of(data_type_, data_type::s8, data_type::u8,
                     data_type::f8_e4m3, data_type::f8_e5m2))
        convert_to_f32(dst_vmm, xmm_dst, data_type_);
}

template <>
void jit_io_helper_t<Xbyak::Ymm>::emu_gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Ymm &indices_vmm, const Xbyak::Ymm &dst_vmm,
        const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(gather_conf_->vmm_tmp_idx_.has_value()
            && "Temporary vreg is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const Xbyak::Xmm xmm_tmp = Xbyak::Xmm(gather_conf_->full_vmm_mask_idx_);
    const Xbyak::Xmm xmm_dst = Xbyak::Xmm(*gather_conf_->vmm_tmp_idx_);

    host_->mov(gather_conf_->reg_tmp_, 0);
    host_->mov(gather_conf_->reg_tmp1_, src_reg);

    // The conversion of bf16->f32 here is split into two parts.
    // Here while loading words of bf16, the words are interleaved,
    // and in convert_to_f32, they are shifted-left to finally convert to f32
    // For f16 we do not need such interleaving.
    const int xmm_size_elem = 4;
    const int number_of_xmms = tail
            ? utils::div_up(tail_conf_->tail_size_, xmm_size_elem)
            : utils::div_up(gather_conf_->simd_w_, xmm_size_elem);
    for (int i = 0; i < number_of_xmms; i++) {
        host_->vextractf128(xmm_tmp, indices_vmm, i);

        const int number_of_values_to_load = i == number_of_xmms - 1 && tail
                        && tail_conf_->tail_size_ % xmm_size_elem != 0
                ? tail_conf_->tail_size_ % xmm_size_elem
                : xmm_size_elem;
        for (int j = 0; j < number_of_values_to_load; j++) {
            host_->vpextrd(gather_conf_->reg_tmp_.cvt32(), xmm_tmp, j);
            host_->add(src_reg, gather_conf_->reg_tmp_);
            switch (data_type_) {
                case data_type::f32:
                case data_type::s32: {
                    host_->vpinsrd(xmm_dst, xmm_dst, host_->ptr[src_reg], j);
                    break;
                }
                case data_type::f16:
                    assert(f16_supported_ && "Unsupported data type.");
                    host_->vpinsrw(xmm_dst, xmm_dst, host_->ptr[src_reg],
                            i * xmm_size_elem + j);
                    break;
                case data_type::bf16:
                    assert(bf16_supported_ && "Unsupported data type.");
                    host_->vpinsrw(
                            xmm_dst, xmm_dst, host_->ptr[src_reg], j * 2);
                    break;
                case data_type::s8:
                case data_type::u8:
                case data_type::f8_e4m3:
                case data_type::f8_e5m2: {
                    assert(IMPLICATION(
                                   utils::one_of(data_type_, data_type::f8_e4m3,
                                           data_type::f8_e5m2),
                                   fp8_supported_)
                            && "Unsupported data type.");
                    host_->vpinsrb(xmm_dst, xmm_dst, host_->ptr[src_reg],
                            i * xmm_size_elem + j);
                    break;
                }
                default: assert(!"Unsupported data type.");
            }
            host_->mov(src_reg, gather_conf_->reg_tmp1_);
        }

        if (utils::one_of(data_type_, data_type::f32, data_type::s32,
                    data_type::bf16))
            host_->vinsertf128(dst_vmm, dst_vmm, xmm_dst, i);
    }

    if (data_type_ == data_type::s32 || data_type_ == data_type::bf16)
        convert_to_f32(dst_vmm, dst_vmm, data_type_);
    else if (utils::one_of(data_type_, data_type::s8, data_type::u8,
                     data_type::f16, data_type::f8_e4m3, data_type::f8_e5m2))
        convert_to_f32(dst_vmm, xmm_dst, data_type_);
}

template <>
void jit_io_helper_t<Xbyak::Xmm>::emu_gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Xmm &indices_vmm, const Xbyak::Xmm &dst_vmm,
        const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    host_->mov(gather_conf_->reg_tmp_, 0);
    host_->mov(gather_conf_->reg_tmp1_, src_reg);

    const unsigned xmm_size_elem = 4;
    const unsigned number_of_values_to_load
            = tail ? tail_conf_->tail_size_ : xmm_size_elem;
    for (unsigned j = 0; j < number_of_values_to_load; j++) {
        host_->pextrd(gather_conf_->reg_tmp_.cvt32(), indices_vmm, j);
        host_->add(src_reg, gather_conf_->reg_tmp_);
        switch (data_type_) {
            case data_type::f32:
            case data_type::s32: {
                host_->pinsrd(dst_vmm, host_->ptr[src_reg], j);
                break;
            }
            case data_type::f16:
                assert(f16_supported_ && "Unsupported data type.");
                host_->pinsrw(dst_vmm, host_->ptr[src_reg], j);
                break;
            case data_type::bf16:
                assert(bf16_supported_ && "Unsupported data type.");
                host_->pinsrw(dst_vmm, host_->ptr[src_reg], j * 2);
                break;
            case data_type::s8:
            case data_type::u8:
            case data_type::f8_e4m3:
            case data_type::f8_e5m2: {
                assert(IMPLICATION(utils::one_of(data_type_, data_type::f8_e4m3,
                                           data_type::f8_e5m2),
                               fp8_supported_)
                        && "Unsupported data type.");
                host_->pinsrb(dst_vmm, host_->ptr[src_reg], j);
                break;
            }
            default: assert(!"Unsupported data type.");
        }
        host_->mov(src_reg, gather_conf_->reg_tmp1_);
    }

    if (data_type_ != data_type::f32)
        convert_to_f32(dst_vmm, dst_vmm, data_type_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_tail_mask() {
    assert(tail_conf_.has_value() && "Config for tail processing is not set.");

    if (!tail_conf_->tail_size_) return;

    if (is_superset(isa_, avx512_core))
        prepare_opmask(tail_conf_->tail_size_, tail_conf_->reg_tmp_,
                tail_conf_->tail_opmask_);
    else if (is_superset(isa_, sse41))
        prepare_vmm_mask(tail_conf_->tail_size_, tail_conf_->simd_w_,
                tail_conf_->reg_tmp_, Vmm(tail_conf_->tail_vmm_mask_idx_));
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    if (utils::one_of(data_type_, data_type::f16, data_type::bf16,
                data_type::s8, data_type::u8, data_type::f8_e4m3,
                data_type::f8_e5m2))
        return;

    if (is_superset(isa_, avx512_core))
        prepare_opmask(gather_conf_->simd_w_, gather_conf_->reg_tmp_,
                gather_conf_->full_opmask_);
    else if (is_superset(isa_, avx2))
        prepare_vmm_mask(gather_conf_->simd_w_, gather_conf_->simd_w_,
                gather_conf_->reg_tmp_, Vmm(gather_conf_->full_vmm_mask_idx_));
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    if (is_superset(isa_, avx2)) {
        const Vmm vmm_mask = Vmm(gather_conf_->full_vmm_mask_idx_);
        host_->uni_vxorps(vmm_mask, vmm_mask, vmm_mask);
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
void jit_io_helper_t<Vmm>::prepare_table_fp8() {
    if (fp8_emu_) fp8_emu_->prepare_table();
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::gather(const Xbyak::Reg64 &src_reg,
        const Vmm &indices_vmm, const Vmm &dst_vmm, const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const Vmm &mask = tail ? Vmm(tail_conf_->tail_vmm_mask_idx_)
                           : Vmm(gather_conf_->full_vmm_mask_idx_);

    const Vmm dst_vmm_with_mask = tail ? dst_vmm | tail_conf_->tail_opmask_
                                       : dst_vmm | gather_conf_->full_opmask_;

    const bool is_avx512 = is_superset(isa_, avx512_core);
    const bool can_use_gather_instruction = is_superset(isa_, avx2);

    if (utils::one_of(data_type_, data_type::f32, data_type::s32)
            && can_use_gather_instruction) {
        if (data_type_ == data_type::f32) {
            if (!is_avx512)
                host_->vgatherdps(
                        dst_vmm, host_->ptr[src_reg + indices_vmm], mask);
            else
                host_->vgatherdps(
                        dst_vmm_with_mask, host_->ptr[src_reg + indices_vmm]);
        } else {
            if (!is_avx512)
                host_->vpgatherdd(
                        dst_vmm, host_->ptr[src_reg + indices_vmm], mask);
            else
                host_->vpgatherdd(
                        dst_vmm_with_mask, host_->ptr[src_reg + indices_vmm]);
            convert_to_f32(dst_vmm, dst_vmm, data_type_);
        }

        // Have to restore processing mask after gather because mask
        // was zeroed.
        if (tail)
            prepare_tail_mask();
        else
            prepare_full_mask();
    } else {
        emu_gather(src_reg, indices_vmm, dst_vmm, tail);
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load(const Xbyak::Address &src_addr,
        const Vmm &dst_raw_vmm, const bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const bool is_avx512 = is_superset(isa_, avx512_core);

    const auto dst_vmm = tail && is_avx512
            ? (dst_raw_vmm | tail_conf_->tail_opmask_ | host_->T_z)
            : dst_raw_vmm;

    const bool is_i8 = utils::one_of(data_type_, data_type::s8, data_type::u8);
    const bool is_xf16
            = utils::one_of(data_type_, data_type::bf16, data_type::f16);
    const bool is_tail_load_supported = is_avx512;
    const bool can_load_byte_by_byte = tail
            && (isa_ == sse41
                    || (!is_tail_load_supported && (is_i8 || is_xf16)));

    if (can_load_byte_by_byte) {
        load_byte_by_byte(src_addr, dst_vmm, tail_conf_->tail_size_);
    } else {
        switch (data_type_) {
            case data_type::f32: load_f32(src_addr, dst_vmm, tail); break;
            case data_type::s32: load_s32(src_addr, dst_vmm, tail); break;
            case data_type::bf16: load_bf16(src_addr, dst_vmm); break;
            case data_type::f16: load_f16(src_addr, dst_vmm); break;
            case data_type::f8_e4m3:
            case data_type::f8_e5m2: load_f8(src_addr, dst_vmm); break;
            case data_type::s8:
            case data_type::u8: load_i8(src_addr, dst_vmm); break;
            default: assert(!"Unsupported data type.");
        }
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_byte_by_byte(const Xbyak::Address &src_addr,
        const Vmm &dst_vmm, const int load_size) {
    static constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;
    UNUSED(is_zmm);
    assert(!is_zmm && "Load byte by byte is not supported for Zmms.");

    host_->load_data(data_type_, dst_vmm, src_addr, load_size);

    if (utils::one_of(data_type_, data_type::s32, data_type::s8, data_type::u8))
        convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_f32(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm, const bool tail) {
    if (tail && !is_superset(isa_, avx512_core))
        host_->vmaskmovps(
                dst_vmm, Vmm(tail_conf_->tail_vmm_mask_idx_), src_addr);
    else
        host_->uni_vmovups(dst_vmm, src_addr);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_s32(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm, const bool tail) {
    if (is_superset(isa_, avx512_core))
        host_->uni_vcvtdq2ps(dst_vmm, src_addr);
    else {
        load_f32(src_addr, dst_vmm, tail);
        convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_bf16(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm) {
    assert(bf16_supported_ && "Unsupported data type.");

    host_->vpmovzxwd(dst_vmm, src_addr);
    convert_to_f32(dst_vmm, dst_vmm, data_type::bf16);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_f16(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm) {
    assert(f16_supported_ && "Unsupported data type.");
    host_->uni_vcvtph2psx(dst_vmm, src_addr);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_f8(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm) {
    assert(fp8_supported_ && fp8_emu_
            && "Unsupported data type or emulation not available.");
    if (fp8_emu_) fp8_emu_->vcvt_f8_to_f32(dst_vmm, src_addr);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_i8(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm) {
    if (data_type_ == data_type::s8)
        host_->uni_vpmovsxbd(dst_vmm, src_addr);
    else
        host_->uni_vpmovzxbd(dst_vmm, src_addr);

    convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_two_simdw_xf16(const Xbyak::Address &src_addr,
        const Vmm &dst_even_vmm, const Vmm &dst_odd_vmm) {
    // The outputs are in odd/even interleaved layouts
    // now only support bf16/f16 w/o tail on AVX2_VNNI_2
    assert(utils::one_of(data_type_, data_type::bf16, data_type::f16)
            && isa_ == avx2_vnni_2 && "Unsupported data type.");

    if (data_type_ == data_type::bf16) {
        host_->vcvtneebf162ps(dst_even_vmm, src_addr);
        host_->vcvtneobf162ps(dst_odd_vmm, src_addr);
    } else {
        host_->vcvtneeph2ps(dst_even_vmm, src_addr);
        host_->vcvtneoph2ps(dst_odd_vmm, src_addr);
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::merge_interleaved_to_plain(
        const Vmm &vmm_even, const Vmm &vmm_odd, const Vmm &vmm_aux0) {
    // Merge inputs in odd/even interleaved layouts to plain layouts
    assert(vmm_even.isYMM() && vmm_odd.isYMM()
            && "Merge interleaved to plain only supports Ymms");
    Xbyak::Ymm ymm_even = Xbyak::Ymm(vmm_even.getIdx());
    Xbyak::Ymm ymm_odd = Xbyak::Ymm(vmm_odd.getIdx());
    Xbyak::Ymm ymm_aux0 = Xbyak::Ymm(vmm_aux0.getIdx());
    Xbyak::Ymm ymm_aux1 = Xbyak::Ymm(vmm_odd.getIdx());

    host_->vpunpckldq(ymm_aux0, ymm_even, ymm_odd);
    host_->vpunpckhdq(ymm_aux1, ymm_even, ymm_odd);
    host_->vperm2i128(ymm_even, ymm_aux0, ymm_aux1, 0x20);
    host_->vperm2i128(ymm_odd, ymm_aux0, ymm_aux1, 0x31);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store(const Vmm &src_raw_vmm,
        const Xbyak::Address &dst_raw_addr, const bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");
    assert(!(tail && io_conf_.nt_stores_enabled_)
            && "Usage of non-temporal stores with tail leads to a general-protection exception.");

    const bool is_avx512 = is_superset(isa_, avx512_core);

    const auto dst_addr = tail && is_avx512
            ? (dst_raw_addr | tail_conf_->tail_opmask_)
            : dst_raw_addr;
    const auto src_vmm = tail && is_avx512
            ? (src_raw_vmm | tail_conf_->tail_opmask_)
            : src_raw_vmm;

    const bool is_store_tail_supported = is_avx512;
    const bool is_i8 = utils::one_of(data_type_, data_type::s8, data_type::u8);
    const bool is_xf16
            = utils::one_of(data_type_, data_type::bf16, data_type::f16);

    const bool can_store_byte_by_byte
            = (tail
                      && (isa_ == sse41
                              || (!is_store_tail_supported
                                      && (is_i8 || is_xf16))))
            || (std::is_same<Vmm, Xbyak::Xmm>::value && is_xf16);

    if (data_type_ == data_type::s32 || is_i8) saturate(src_raw_vmm);

    if (can_store_byte_by_byte) {
        // TODO: Consider adding opmask to store xf16 data from Xmm.
        // This could allow to use store_bf16/store_f16 functions for isa >= avx512_core.
        const size_t xmm_length
                = vreg_traits<Xbyak::Xmm>::vlen / sizeof(int32_t);
        const size_t store_size = (tail ? tail_conf_->tail_size_ : xmm_length)
                * types::data_type_size(data_type_);
        store_byte_by_byte(src_vmm, dst_addr, store_size);
    } else {
        switch (data_type_) {
            case data_type::f32:
            case data_type::s32: store_f32(src_vmm, dst_addr, tail); break;
            case data_type::bf16: store_bf16(src_vmm, dst_addr); break;
            case data_type::f16: store_f16(src_vmm, dst_addr); break;
            case data_type::f8_e4m3:
            case data_type::f8_e5m2: store_f8(src_vmm, dst_addr); break;
            case data_type::s8:
            case data_type::u8: store_i8(src_vmm, dst_raw_addr); break;
            default: assert(!"Unsupported data type.");
        }
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::saturate(const Vmm &vmm) {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    host_->saturate_f32(vmm, Vmm(saturation_conf_->vreg_zero_saturation_idx_),
            Vmm(saturation_conf_->vreg_saturation_ubound_idx_), data_type_);
    host_->uni_vcvtps2dq(vmm, vmm);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_byte_by_byte(const Vmm &src_vmm,
        const Xbyak::Address &dst_addr, const int store_size) {
    static constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;
    UNUSED(is_zmm);
    assert(!is_zmm && "Store byte by byte is not supported for Zmms.");

    const bool is_i8 = utils::one_of(data_type_, data_type::s8, data_type::u8);
    const bool is_xf16
            = utils::one_of(data_type_, data_type::bf16, data_type::f16);
    const auto &cvt_lower_vmm =
            typename vreg_traits<Vmm>::Vmm_lower_t(src_vmm.getIdx());

    if (is_i8) prepare_i8_data_to_store(src_vmm);
    if (is_xf16) prepare_xf16_data_to_store(src_vmm);

    host_->store_bytes(is_xf16 ? cvt_lower_vmm : src_vmm, dst_addr, store_size);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_f32(
        const Vmm &src_vmm, const Xbyak::Address &dst_addr, const bool tail) {
    if (io_conf_.nt_stores_enabled_)
        host_->uni_vmovntps(dst_addr, src_vmm);
    else if (!is_superset(isa_, avx512_core) && tail)
        host_->vmaskmovps(
                dst_addr, Vmm(tail_conf_->tail_vmm_mask_idx_), src_vmm);
    else
        host_->uni_vmovups(dst_addr, src_vmm);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_bf16(
        const Vmm &src_vmm, const Xbyak::Address &dst_addr) {
    assert(bf16_supported_ && "Unsupported data type.");
    assert((src_vmm.isZMM() || src_vmm.isYMM())
            && "Store operation for bf16 is not supported for Xmms.");

    const auto &cvt_lower_vmm =
            typename vreg_traits<Vmm>::Vmm_lower_t(src_vmm.getIdx());

    if (bf16_emu_)
        bf16_emu_->vcvtneps2bf16(cvt_lower_vmm, src_vmm);
    else
        host_->vcvtneps2bf16(cvt_lower_vmm, src_vmm, host_->get_encoding());

    if (io_conf_.nt_stores_enabled_)
        host_->uni_vmovntps(dst_addr, cvt_lower_vmm);
    else
        host_->uni_vmovdqu16(dst_addr, cvt_lower_vmm);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_f16(
        const Vmm &src_vmm, const Xbyak::Address &dst_addr) {
    assert(f16_supported_ && "Unsupported data type.");
    assert((src_vmm.isZMM() || src_vmm.isYMM())
            && "Store operation for f16 is not supported for Xmms.");

    const auto &cvt_lower_vmm =
            typename vreg_traits<Vmm>::Vmm_lower_t(src_vmm.getIdx());

    host_->uni_vcvtps2phx(cvt_lower_vmm, src_vmm);

    if (io_conf_.nt_stores_enabled_)
        host_->uni_vmovntps(dst_addr, cvt_lower_vmm);
    else
        host_->uni_vmovdqu16(dst_addr, cvt_lower_vmm);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_f8(
        const Vmm &src_vmm, const Xbyak::Address &dst_addr) {
    assert(fp8_supported_ && fp8_emu_
            && "Unsupported data type or emulation not available.");

    const Xbyak::Xmm lower_xmm = Xbyak::Xmm(src_vmm.getIdx());

    if (fp8_emu_)
        fp8_emu_->vcvt_f32_to_f8(
                lower_xmm | Xbyak::Opmask(src_vmm.getOpmaskIdx()), src_vmm);

    if (io_conf_.nt_stores_enabled_)
        host_->vmovntps(dst_addr, lower_xmm);
    else
        host_->vmovdqu8(dst_addr, lower_xmm);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_i8(
        const Vmm &src_vmm, const Xbyak::Address &dst_addr) {
    if (!is_superset(isa_, avx512_core)) {
        static constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;

        prepare_i8_data_to_store(src_vmm);
        if (is_ymm)
            host_->uni_vmovq(dst_addr, Xbyak::Xmm(src_vmm.getIdx()));
        else
            host_->uni_vmovd(dst_addr, src_vmm);
    } else {
        using namespace std::placeholders;
        static constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

        auto store_i8_fn = data_type_ == data_type::s8
                ? std::bind(&jit_generator::vpmovsdb, host_, _1, _2)
                : std::bind(&jit_generator::vpmovusdb, host_, _1, _2);

        if (io_conf_.nt_stores_enabled_ && is_zmm) {
            Xbyak::Xmm src_xmm(src_vmm.getIdx());
            store_i8_fn(src_xmm, src_vmm);
            host_->uni_vmovntps(dst_addr, src_xmm);
        } else {
            store_i8_fn(dst_addr, src_vmm);
        }
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::convert_to_f32(const Vmm &dst_vmm,
        const Xbyak::Xmm &src_vmm, const data_type_t src_data_type) {
    switch (src_data_type) {
        case data_type::s32: {
            assert(dst_vmm.getIdx() == src_vmm.getIdx());
            host_->uni_vcvtdq2ps(dst_vmm, dst_vmm);
            break;
        }
        case data_type::bf16:
            assert(bf16_supported_ && "Unsupported data type.");
            host_->vpslld(dst_vmm, src_vmm, 0x10);
            break;
        case data_type::f16:
            assert(f16_supported_ && "Unsupported data type.");
            host_->vcvtph2ps(dst_vmm, src_vmm);
            break;
        case data_type::f8_e5m2:
        case data_type::f8_e4m3:
            assert(fp8_supported_ && fp8_emu_
                    && "Unsupported data type or emulation not available.");
            if (fp8_emu_) fp8_emu_->vcvt_f8_to_f32(dst_vmm, src_vmm);
            break;
        case data_type::s8: {
            host_->uni_vpmovsxbd(dst_vmm, src_vmm);
            host_->uni_vcvtdq2ps(dst_vmm, dst_vmm);
            break;
        }
        case data_type::u8: {
            host_->uni_vpmovzxbd(dst_vmm, src_vmm);
            host_->uni_vcvtdq2ps(dst_vmm, dst_vmm);
            break;
        }
        default: assert(!"Unsupported data type.");
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::broadcast(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm) {
    switch (data_type_) {
        case data_type::f32: host_->uni_vbroadcastss(dst_vmm, src_addr); break;
        case data_type::bf16:
            assert(bf16_supported_ && "Unsupported data type.");
            if (is_superset(isa_, avx2_vnni_2))
                host_->vbcstnebf162ps(dst_vmm, src_addr);
            else {
                host_->vpbroadcastw(dst_vmm, src_addr);
                convert_to_f32(dst_vmm, dst_vmm, data_type_);
            }
            break;
        case data_type::f16:
            assert(f16_supported_ && "Unsupported data type.");
            if (is_superset(isa_, avx2_vnni_2))
                host_->vbcstnesh2ps(dst_vmm, src_addr);
            else
                host_->uni_vcvtph2psx(
                        dst_vmm, host_->ptr_b[src_addr.getRegExp()]);
            break;
        case data_type::s32: {
            if (is_superset(isa_, avx512_core)) {
                host_->uni_vcvtdq2ps(
                        dst_vmm, host_->ptr_b[src_addr.getRegExp()]);
            } else {
                host_->uni_vbroadcastss(dst_vmm, src_addr);
                convert_to_f32(dst_vmm, dst_vmm, data_type_);
            }
            break;
        }
        case data_type::f8_e4m3:
        case data_type::f8_e5m2:
            assert(fp8_supported_ && fp8_emu_
                    && "Unsupported data type or emulation not available.");
            if (fp8_emu_)
                fp8_emu_->vcvt_f8_to_f32(
                        dst_vmm, host_->ptr_b[src_addr.getRegExp()]);
            break;
        case data_type::s8:
        case data_type::u8: {
            const Xbyak::Xmm dst_xmm {dst_vmm.getIdx()};
            host_->uni_vpinsrb(dst_xmm, dst_xmm, src_addr, 0);
            convert_to_f32(dst_vmm, dst_vmm, data_type_);
            host_->uni_vbroadcastss(dst_vmm, dst_xmm);

            break;
        }
        default: assert(!"Unsupported data type.");
    }
}

// Has to live here, otherwise gcc483 and gcc540 will report an error:
// `undefined reference to
// 'dnnl::impl::cpu::x64::io::jit_io_multi_dt_helper_t<Xbyak::Xmm>::
//     jit_io_multi_dt_helper_t()'`
template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::jit_io_multi_dt_helper_t() = default;

template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::jit_io_multi_dt_helper_t(jit_generator *host,
        const cpu_isa_t &isa, const data_types_t &data_types,
        const io_conf_t &io_conf,
        const utils::optional_t<io_tail_conf_t> &tail_conf,
        const utils::optional_t<io_emu_bf16_conf_t> &bf16_conf,
        const std::map<data_type_t, io_saturation_conf_t> &saturation_confs,
        const utils::optional_t<io_gather_conf_t> &gather_conf,
        const utils::optional_t<io_emu_fp8_conf_t> &fp8_conf) {
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
                            dt == data_type::bf16 ? bf16_conf : utils::nullopt,
                            store_saturation_needed ? utils::optional_t<
                                    io_saturation_conf_t> {saturation_conf
                                                                   ->second}
                                                    : utils::nullopt,
                            gather_conf,
                            utils::one_of(
                                    dt, data_type::f8_e4m3, data_type::f8_e5m2)
                                    ? fp8_conf
                                    : utils::nullopt));
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
bool jit_io_multi_dt_helper_t<Vmm>::empty() const {
    return storage_.empty();
}

template <typename Vmm>
std::shared_ptr<jit_io_helper_t<Vmm>> jit_io_multi_dt_helper_t<Vmm>::operator[](
        const data_type_t dt) const {
    auto res = this->at(dt);
    if (res == nullptr) { assert(!"data not found in io"); }
    return res;
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
void jit_io_multi_dt_helper_t<Vmm>::init_bf16() {
    const auto bf16_io_helper = at(data_type::bf16);
    if (bf16_io_helper) bf16_io_helper->init_bf16();
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::prepare_table_fp8() {
    const auto f8_e5m2_io_helper = at(data_type::f8_e5m2);
    if (f8_e5m2_io_helper) f8_e5m2_io_helper->prepare_table_fp8();
    const auto f8_e4m3_io_helper = at(data_type::f8_e4m3);
    if (f8_e4m3_io_helper) f8_e4m3_io_helper->prepare_table_fp8();
}

template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::~jit_io_multi_dt_helper_t() = default;

template class jit_io_helper_t<Xbyak::Zmm>;
template class jit_io_helper_t<Xbyak::Ymm>;
template class jit_io_helper_t<Xbyak::Xmm>;

template class jit_io_multi_dt_helper_t<Xbyak::Zmm>;
template class jit_io_multi_dt_helper_t<Xbyak::Ymm>;
template class jit_io_multi_dt_helper_t<Xbyak::Xmm>;

} // namespace io
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
