/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace io {

template <typename Vmm>
io_tail_conf_t<Vmm>::io_tail_conf_t(const std::size_t simd_w,
        const std::size_t tail_size, const Xbyak::Opmask &tail_opmask,
        const Vmm &tail_vmm_mask, const Xbyak::Reg64 &reg_tmp)
    : simd_w_(simd_w)
    , tail_size_(tail_size)
    , tail_opmask_(tail_opmask)
    , tail_vmm_mask_(tail_vmm_mask)
    , reg_tmp_(reg_tmp) {}

template <typename Vmm>
io_tail_conf_t<Vmm>::io_tail_conf_t(const io_tail_conf_t &other)
    : simd_w_(other.simd_w_)
    , tail_size_(other.tail_size_)
    , tail_opmask_(other.tail_opmask_)
    , tail_vmm_mask_(other.tail_vmm_mask_)
    , reg_tmp_(other.reg_tmp_) {}

template <typename Vmm>
io_saturation_conf_t<Vmm>::io_saturation_conf_t(const Vmm &vreg_zero_saturation,
        const Vmm &vreg_saturation_ubound, const Xbyak::Reg64 &reg_tmp)
    : vreg_zero_saturation_(vreg_zero_saturation)
    , vreg_saturation_ubound_(vreg_saturation_ubound)
    , reg_tmp_(reg_tmp) {}

template <typename Vmm>
io_saturation_conf_t<Vmm>::io_saturation_conf_t(
        const io_saturation_conf_t &other)
    : vreg_zero_saturation_(other.vreg_zero_saturation_)
    , vreg_saturation_ubound_(other.vreg_saturation_ubound_)
    , reg_tmp_(other.reg_tmp_) {}

template <typename Vmm>
io_gather_conf_t<Vmm>::io_gather_conf_t(const io_gather_conf_t &other)
    : simd_w_(other.simd_w_)
    , full_opmask_(other.full_opmask_)
    , full_vmm_mask_(other.full_vmm_mask_)
    , reg_tmp_(other.reg_tmp_)
    , reg_tmp1_(other.reg_tmp1_) {}

template <typename Vmm>
io_gather_conf_t<Vmm>::io_gather_conf_t(const std::size_t simd_w,
        const Xbyak::Opmask &full_opmask, const Vmm &full_vmm_mask,
        const Xbyak::Reg64 &reg_tmp, const Xbyak::Reg64 &reg_tmp1)
    : simd_w_(simd_w)
    , full_opmask_(full_opmask)
    , full_vmm_mask_(full_vmm_mask)
    , reg_tmp_(reg_tmp)
    , reg_tmp1_(reg_tmp1) {}

io_emu_bf16_conf_t::io_emu_bf16_conf_t() {}

io_emu_bf16_conf_t::io_emu_bf16_conf_t(const Xbyak::Zmm &bf16_emu_reserv_1,
        const Xbyak::Zmm &bf16_emu_reserv_2,
        const Xbyak::Zmm &bf16_emu_reserv_3, const Xbyak::Reg64 &reg_tmp,
        const Xbyak::Zmm &bf16_emu_reserv_4)
    : bf16_emu_reserv_1_(bf16_emu_reserv_1)
    , bf16_emu_reserv_2_(bf16_emu_reserv_2)
    , bf16_emu_reserv_3_(bf16_emu_reserv_3)
    , reg_tmp_(reg_tmp)
    , bf16_emu_reserv_4_(bf16_emu_reserv_4) {}

io_emu_bf16_conf_t::io_emu_bf16_conf_t(const io_emu_bf16_conf_t &other)
    : bf16_emu_reserv_1_(other.bf16_emu_reserv_1_)
    , bf16_emu_reserv_2_(other.bf16_emu_reserv_2_)
    , bf16_emu_reserv_3_(other.bf16_emu_reserv_3_)
    , reg_tmp_(other.reg_tmp_)
    , bf16_emu_reserv_4_(other.bf16_emu_reserv_4_) {}

template <typename Vmm>
io_conf_t<Vmm>::io_conf_t() {}

template <typename Vmm>
io_conf_t<Vmm>::io_conf_t(const bool nt_stores_enabled)
    : nt_stores_enabled_(nt_stores_enabled) {}

template <typename Vmm>
jit_io_helper_t<Vmm>::jit_io_helper_t(jit_generator *host, const cpu_isa_t &isa,
        const data_type_t &data_type, const io_conf_t<Vmm> &io_conf,
        const utils::optional_t<io_tail_conf_t<Vmm>> &tail_conf,
        const utils::optional_t<io_emu_bf16_conf_t> &bf16_conf,
        const utils::optional_t<io_saturation_conf_t<Vmm>> &saturation_conf,
        const utils::optional_t<io_gather_conf_t<Vmm>> &gather_conf)
    : host_(host)
    , isa_(isa)
    , data_type_(data_type)
    , bf16_supported_(utils::one_of(isa, avx512_core, avx512_core_bf16))
    , bf16_emu_(nullptr)
    , io_conf_(io_conf)
    , tail_conf_(tail_conf)
    , saturation_conf_(saturation_conf)
    , bf16_conf_(bf16_conf)
    , gather_conf_(gather_conf) {

    if (data_type_ == data_type::bf16 && isa == avx512_core) {
        assert(bf16_conf.has_value()
                && "Config for bf16 emulation is not set.");
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(host_,
                bf16_conf->bf16_emu_reserv_1_, bf16_conf->bf16_emu_reserv_2_,
                bf16_conf->bf16_emu_reserv_3_, bf16_conf->reg_tmp_,
                bf16_conf->bf16_emu_reserv_4_);
    }

    assert(utils::one_of(data_type_, data_type::bf16, data_type::f32,
                   data_type::s8, data_type::u8, data_type::s32)
            && "Supported data types bf16, f32, s8, u8, s32");

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
}

template <typename Vmm>
jit_io_helper_t<Vmm>::~jit_io_helper_t() = default;

template <>
void jit_io_helper_t<Xbyak::Zmm>::init_bf16() {
    if (bf16_emu_) {
        assert(bf16_conf_.has_value()
                && "Config for bf16 emulation is not set.");
        bf16_emu_->init_vcvtneps2bf16();
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_bf16() {}

template <>
void jit_io_helper_t<Xbyak::Zmm>::prepare_tail_mask() {
    assert(tail_conf_.has_value() && "Config for tail processing is not set.");

    if (!tail_conf_->tail_size_) return;
    prepare_mask(tail_conf_->tail_size_, tail_conf_->simd_w_,
            tail_conf_->reg_tmp_, tail_conf_->tail_opmask_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_tail_mask() {
    assert(tail_conf_.has_value() && "Config for tail processing is not set.");

    if (!tail_conf_->tail_size_) return;
    prepare_mask(tail_conf_->tail_size_, tail_conf_->simd_w_,
            tail_conf_->reg_tmp_, tail_conf_->tail_vmm_mask_);
}

template <>
void jit_io_helper_t<Xbyak::Zmm>::prepare_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    if (data_type_ == data_type::bf16 || data_type_ == data_type::s8
            || data_type_ == data_type::u8)
        return;
    prepare_mask(gather_conf_->simd_w_, gather_conf_->simd_w_,
            gather_conf_->reg_tmp_, gather_conf_->full_opmask_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    if (isa_ != avx2
            && (data_type_ == data_type::s8 || data_type_ == data_type::u8))
        return;

    prepare_mask(gather_conf_->simd_w_, gather_conf_->simd_w_,
            gather_conf_->reg_tmp_, gather_conf_->full_vmm_mask_);
}

template <>
void jit_io_helper_t<Xbyak::Ymm>::init_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    if (isa_ == avx2) {
        host_->uni_vxorps(gather_conf_->full_vmm_mask_,
                gather_conf_->full_vmm_mask_, gather_conf_->full_vmm_mask_);
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_saturate_f32() const {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    if (utils::one_of(data_type_, data_type::u8, data_type::s8, data_type::s32))
        host_->init_saturate_f32(saturation_conf_->vreg_zero_saturation_,
                saturation_conf_->vreg_saturation_ubound_,
                saturation_conf_->reg_tmp_, data_type::f32, data_type_);
}

template <>
void jit_io_helper_t<Xbyak::Zmm>::gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Zmm &indices_vmm, const Xbyak::Zmm &dst_raw_vmm,
        bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    if (data_type_ == data_type::f32 || data_type_ == data_type::s32) {
        const auto &mask
                = tail ? tail_conf_->tail_opmask_ : gather_conf_->full_opmask_;
        const auto dst_vmm = tail
                ? (dst_raw_vmm | tail_conf_->tail_opmask_ | host_->T_z)
                : dst_raw_vmm;

        if (data_type_ == data_type::f32)
            host_->vgatherdps(
                    dst_vmm | mask, host_->ptr[src_reg + indices_vmm]);
        else
            host_->vpgatherdd(
                    dst_vmm | mask, host_->ptr[src_reg + indices_vmm]);

        // Have to restore processing mask after gather because mask
        // was zeroed after vgatherdps.
        if (tail)
            prepare_tail_mask();
        else
            prepare_full_mask();
    } else {
        emu_gather(src_reg, indices_vmm, dst_raw_vmm, tail);
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::gather(const Xbyak::Reg64 &src_reg,
        const Vmm &indices_vmm, const Vmm &dst_vmm, bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const Vmm &mask
            = tail ? tail_conf_->tail_vmm_mask_ : gather_conf_->full_vmm_mask_;

    const Vmm dst_vmm_with_mask = tail
            ? dst_vmm | tail_conf_->tail_opmask_ | host_->T_z
            : dst_vmm | gather_conf_->full_opmask_;

    if ((data_type_ == data_type::f32 || data_type_ == data_type::s32)
            && isa_ == avx2) {
        if (data_type_ == data_type::f32) {
            if (isa_ == avx2)
                host_->vgatherdps(
                        dst_vmm, host_->ptr[src_reg + indices_vmm], mask);
            else
                host_->vgatherdps(
                        dst_vmm_with_mask, host_->ptr[src_reg + indices_vmm]);
        } else {
            if (isa_ == avx2)
                host_->vpgatherdd(
                        dst_vmm, host_->ptr[src_reg + indices_vmm], mask);
            else
                host_->vpgatherdd(
                        dst_vmm_with_mask, host_->ptr[src_reg + indices_vmm]);
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

template <>
void jit_io_helper_t<Xbyak::Zmm>::load(const Xbyak::Address &src_addr,
        const Xbyak::Zmm &dst_raw_vmm, bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const auto dst_vmm = tail
            ? (dst_raw_vmm | tail_conf_->tail_opmask_ | host_->T_z)
            : dst_raw_vmm;

    switch (data_type_) {
        case data_type::f32: host_->uni_vmovups(dst_vmm, src_addr); break;
        case data_type::s32: host_->uni_vcvtdq2ps(dst_vmm, src_addr); break;
        case data_type::bf16:
            if (bf16_supported_) {
                host_->vpmovzxwd(dst_vmm, src_addr);
                convert_to_f32(dst_vmm, dst_vmm, data_type_);
            } else {
                assert(!"unsupported data type");
            }
            break;
        case data_type::s8: {
            host_->uni_vpmovsxbd(dst_vmm, src_addr);
            convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
            break;
        }
        case data_type::u8: {
            host_->uni_vpmovzxbd(dst_vmm, src_addr);
            convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
            break;
        }
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm, bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    if (tail
            && (isa_ == sse41
                    || utils::one_of(
                            data_type_, data_type::s8, data_type::u8))) {
        host_->uni_vxorps(dst_vmm, dst_vmm, dst_vmm);
        host_->load_data(data_type_, dst_vmm, src_addr, tail_conf_->tail_size_);
    } else if (utils::one_of(data_type_, data_type::f32, data_type::s32)) {
        if (tail)
            host_->vmaskmovps(dst_vmm, tail_conf_->tail_vmm_mask_, src_addr);
        else
            host_->uni_vmovups(dst_vmm, src_addr);
    } else if (data_type_ == data_type::s8) {
        host_->uni_vpmovsxbd(dst_vmm, src_addr);
    } else if (data_type_ == data_type::u8) {
        host_->uni_vpmovzxbd(dst_vmm, src_addr);
    } else
        assert(!"unsupported data type");

    if (data_type_ != data_type::f32)
        convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
}

template <>
void jit_io_helper_t<Xbyak::Zmm>::store(const Xbyak::Zmm &src_raw_vmm,
        const Xbyak::Address &dst_raw_addr, bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const auto src_vmm
            = tail ? (src_raw_vmm | tail_conf_->tail_opmask_) : src_raw_vmm;

    if (utils::one_of(
                data_type_, data_type::s32, data_type::s8, data_type::u8)) {
        assert(saturation_conf_.has_value()
                && "Config for saturation is not set.");

        host_->saturate_f32(src_raw_vmm,
                saturation_conf_->vreg_zero_saturation_,
                saturation_conf_->vreg_saturation_ubound_, data_type_);
        host_->uni_vcvtps2dq(src_vmm, src_raw_vmm);
    }

    const auto dst_addr
            = tail ? (dst_raw_addr | tail_conf_->tail_opmask_) : dst_raw_addr;

    switch (data_type_) {
        case data_type::f32:
        case data_type::s32:
            if (io_conf_.nt_stores_enabled_) {
                host_->uni_vmovntps(dst_raw_addr, src_raw_vmm);
            } else {
                host_->uni_vmovups(dst_addr, src_raw_vmm);
            }
            break;
        case data_type::bf16: {
            if (bf16_supported_) {
                const Xbyak::Ymm src_ymm {src_raw_vmm.getIdx()};
                if (bf16_emu_)
                    bf16_emu_->vcvtneps2bf16(src_ymm, src_raw_vmm);
                else
                    host_->vcvtneps2bf16(src_ymm, src_raw_vmm);
                if (io_conf_.nt_stores_enabled_) {
                    host_->uni_vmovntps(dst_raw_addr, src_ymm);
                } else {
                    host_->vmovdqu16(dst_addr, src_ymm);
                }
            } else {
                assert(!"unsupported data type");
            }
            break;
        }
        case data_type::s8: {
            if (io_conf_.nt_stores_enabled_) {
                Xbyak::Xmm src_xmm(src_vmm.getIdx());
                host_->vpmovsdb(src_xmm, src_vmm);
                host_->uni_vmovntps(dst_raw_addr, src_xmm);
            } else {
                host_->vpmovsdb(dst_raw_addr, src_vmm);
            }
            break;
        }
        case data_type::u8: {
            if (io_conf_.nt_stores_enabled_) {
                Xbyak::Xmm src_xmm(src_vmm.getIdx());
                host_->vpmovusdb(src_xmm, src_vmm);
                host_->uni_vmovntps(dst_raw_addr, src_xmm);
            } else {
                host_->vpmovusdb(dst_raw_addr, src_vmm);
            }
            break;
        }
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store(
        const Vmm &src_vmm, const Xbyak::Address &dst_addr, bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    static constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;

    if (data_type_ != data_type::f32) {
        assert(saturation_conf_.has_value()
                && "Config for saturation is not set.");

        host_->saturate_f32(src_vmm, saturation_conf_->vreg_zero_saturation_,
                saturation_conf_->vreg_saturation_ubound_, data_type_);
        host_->uni_vcvtps2dq(src_vmm, src_vmm);
    }

    const auto prepare_bytes_to_store = [&]() {
        host_->uni_vpackssdw(
                src_vmm, src_vmm, saturation_conf_->vreg_zero_saturation_);
        if (is_ymm) {
            const auto src_ymm = Xbyak::Ymm(src_vmm.getIdx());
            host_->vpermq(src_ymm, src_ymm, 0x58);
        }

        if (data_type_ == data_type::s8)
            host_->uni_vpacksswb(
                    src_vmm, src_vmm, saturation_conf_->vreg_zero_saturation_);
        else
            host_->uni_vpackuswb(
                    src_vmm, src_vmm, saturation_conf_->vreg_zero_saturation_);
    };

    const auto store_tail = [&] {
        switch (data_type_) {
            case data_type::f32:
            case data_type::s32: {
                if (isa_ == sse41)
                    host_->store_bytes(src_vmm, dst_addr,
                            tail_conf_->tail_size_ * sizeof(int32_t));
                else
                    host_->vmaskmovps(
                            dst_addr, tail_conf_->tail_vmm_mask_, src_vmm);
                break;
            }
            case data_type::s8:
            case data_type::u8: {
                prepare_bytes_to_store();
                host_->store_bytes(src_vmm, dst_addr, tail_conf_->tail_size_);
                break;
            }
            default: assert(!"unsupported data type");
        }
    };

    const auto store_no_tail = [&]() {
        switch (data_type_) {
            case data_type::f32:
            case data_type::s32:
                if (io_conf_.nt_stores_enabled_)
                    host_->uni_vmovntps(dst_addr, src_vmm);
                else
                    host_->uni_vmovups(dst_addr, src_vmm);
                break;
            case data_type::s8:
            case data_type::u8: {
                prepare_bytes_to_store();
                if (is_ymm)
                    host_->uni_vmovq(dst_addr, Xbyak::Xmm(src_vmm.getIdx()));
                else if (isa_ == sse41)
                    host_->movd(dst_addr, src_vmm);
                else
                    host_->vmovd(dst_addr, src_vmm);
                break;
            }
            default: assert(!"unsupported data type");
        }
    };

    if (tail)
        store_tail();
    else
        store_no_tail();
}

template <>
void jit_io_helper_t<Xbyak::Zmm>::prepare_mask(
        const std::size_t how_many_bits_to_set, const std::size_t simd_w,
        const Xbyak::Reg64 &reg_tmp, const Xbyak::Operand &mask) {
    const Xbyak::Opmask opmask(mask.getOpmaskIdx());

    const int mask_f32 = (1 << how_many_bits_to_set) - 1;
    const Xbyak::Reg32 regw_tmp = reg_tmp.cvt32();
    host_->mov(regw_tmp, mask_f32);
    host_->kmovw(opmask, regw_tmp);
}

template <>
void jit_io_helper_t<Xbyak::Ymm>::prepare_mask(
        const std::size_t how_many_bits_to_set, const std::size_t simd_w,
        const Xbyak::Reg64 &reg_tmp, const Xbyak::Operand &mask) {
    static const uint32_t mask_f32[14]
            = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};
    const Xbyak::Ymm mask_ymm(mask.getIdx());

    if (how_many_bits_to_set < simd_w) {
        host_->mov(reg_tmp,
                reinterpret_cast<size_t>(&mask_f32[7 - how_many_bits_to_set]));
        host_->vmovups(mask_ymm, host_->ptr[reg_tmp]);
    } else if (how_many_bits_to_set == simd_w) {
        host_->vcmpps(mask_ymm, mask_ymm, mask_ymm, jit_generator::_cmp_eq_oq);
    } else {
        assert(!"Can't set so many bits.");
    }
}

template <>
void jit_io_helper_t<Xbyak::Xmm>::prepare_mask(
        const std::size_t how_many_bits_to_set, const std::size_t simd_w,
        const Xbyak::Reg64 &reg_tmp, const Xbyak::Operand &mask) {}

template <>
void jit_io_helper_t<Xbyak::Zmm>::emu_gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Zmm &indices_vmm, const Xbyak::Zmm &dst_vmm, bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const Xbyak::Xmm xmm_tmp
            = Xbyak::Xmm(gather_conf_->full_vmm_mask_.getIdx());
    const Xbyak::Xmm xmm_dst = Xbyak::Xmm(dst_vmm.getIdx());

    host_->mov(gather_conf_->reg_tmp_, 0);
    host_->mov(gather_conf_->reg_tmp1_, src_reg);

    constexpr unsigned xmm_size_elem = 4;

    const unsigned number_of_xmms = tail
            ? utils::div_up(tail_conf_->tail_size_, xmm_size_elem)
            : utils::div_up(gather_conf_->simd_w_, xmm_size_elem);
    for (unsigned i = 0; i < number_of_xmms; i++) {
        host_->vextractf32x4(xmm_tmp, indices_vmm, i);

        const unsigned number_of_values_to_load = i == number_of_xmms - 1
                        && tail && tail_conf_->tail_size_ % xmm_size_elem != 0
                ? tail_conf_->tail_size_ % xmm_size_elem
                : xmm_size_elem;
        for (unsigned j = 0; j < number_of_values_to_load; j++) {

            host_->vpextrd(gather_conf_->reg_tmp_.cvt32(), xmm_tmp, j);
            host_->add(src_reg, gather_conf_->reg_tmp_);
            switch (data_type_) {
                case data_type::bf16:
                    host_->vpinsrw(
                            xmm_dst, xmm_dst, host_->ptr[src_reg], j * 2);
                    break;
                case data_type::s8:
                case data_type::u8:
                    host_->vpinsrb(xmm_dst, xmm_dst, host_->ptr[src_reg],
                            i * xmm_size_elem + j);
                    break;
                default: assert(!"unsupported data type");
            }
            host_->mov(src_reg, gather_conf_->reg_tmp1_);
        }

        if (data_type_ == data_type::bf16) {
            host_->vinsertf32x4(dst_vmm, dst_vmm, xmm_dst, i);
        }
    }

    convert_to_f32(dst_vmm, xmm_dst, data_type_);
}

template <>
void jit_io_helper_t<Xbyak::Ymm>::emu_gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Ymm &indices_vmm, const Xbyak::Ymm &dst_vmm, bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const Xbyak::Xmm xmm_tmp
            = Xbyak::Xmm(gather_conf_->full_vmm_mask_.getIdx());
    const Xbyak::Xmm xmm_dst = Xbyak::Xmm(dst_vmm.getIdx());

    host_->mov(gather_conf_->reg_tmp_, 0);
    host_->mov(gather_conf_->reg_tmp1_, src_reg);

    constexpr int xmm_size_elem = 4;

    const int number_of_xmms = tail
            ? utils::div_up(tail_conf_->tail_size_, xmm_size_elem)
            : utils::div_up(gather_conf_->simd_w_, xmm_size_elem);
    for (int i = number_of_xmms - 1; i >= 0; i--) {
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
                case data_type::s8:
                case data_type::u8: {
                    host_->vpinsrb(xmm_dst, xmm_dst, host_->ptr[src_reg],
                            i * xmm_size_elem + j);
                    break;
                }
                default: assert(!"unsupported data type");
            }
            host_->mov(src_reg, gather_conf_->reg_tmp1_);
        }

        if (data_type_ == data_type::f32 || data_type_ == data_type::s32) {
            host_->vinsertf128(dst_vmm, dst_vmm, xmm_dst, i);
        }
    }

    if (data_type_ != data_type::f32)
        convert_to_f32(dst_vmm, xmm_dst, data_type_);
}

template <>
void jit_io_helper_t<Xbyak::Xmm>::emu_gather(const Xbyak::Reg64 &src_reg,
        const Xbyak::Xmm &indices_vmm, const Xbyak::Xmm &dst_vmm, bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    host_->mov(gather_conf_->reg_tmp_, 0);
    host_->mov(gather_conf_->reg_tmp1_, src_reg);

    constexpr unsigned xmm_size_elem = 4;

    const unsigned number_of_values_to_load
            = tail ? tail_conf_->tail_size_ : xmm_size_elem;
    for (unsigned j = 0; j < number_of_values_to_load; j++) {
        host_->pextrd(gather_conf_->reg_tmp_.cvt32(), indices_vmm, j);
        host_->add(src_reg, gather_conf_->reg_tmp_);
        switch (data_type_) {
            case data_type::f32:
            case data_type::s32: {
                host_->vpinsrd(dst_vmm, dst_vmm, host_->ptr[src_reg], j);
                break;
            }
            case data_type::s8:
            case data_type::u8: {
                host_->vpinsrb(dst_vmm, dst_vmm, host_->ptr[src_reg], j);
                break;
            }
            default: assert(!"unsupported data type");
        }
        host_->mov(src_reg, gather_conf_->reg_tmp1_);
    }

    if (data_type_ != data_type::f32)
        convert_to_f32(dst_vmm, dst_vmm, data_type_);
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
            if (bf16_supported_) {
                host_->vpslld(dst_vmm, Vmm(src_vmm.getIdx()), 0x10);
            } else {
                assert(!"unsupported data type");
            }
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
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::broadcast(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm) {
    switch (data_type_) {
        case data_type::f32: host_->uni_vbroadcastss(dst_vmm, src_addr); break;
        case data_type::bf16:
            if (bf16_supported_) {
                host_->vpbroadcastw(dst_vmm, src_addr);
                host_->vpslld(dst_vmm, dst_vmm, 0x10);
            } else {
                assert(!"unsupported data type");
            }
            break;
        case data_type::s32: {
            if (is_superset(isa_, avx512_common)) {
                host_->uni_vcvtdq2ps(
                        dst_vmm, host_->ptr_b[src_addr.getRegExp()]);
            } else {
                host_->uni_vbroadcastss(dst_vmm, src_addr);
                host_->uni_vcvtdq2ps(dst_vmm, dst_vmm);
            }
            break;
        }
        case data_type::s8:
        case data_type::u8: {
            const Xbyak::Xmm dst_xmm {dst_vmm.getIdx()};
            host_->uni_vpinsrb(dst_xmm, dst_xmm, src_addr, 0);

            if (data_type_ == data_type::s8)
                host_->uni_vpmovsxbd(dst_xmm, dst_xmm);
            else
                host_->uni_vpmovzxbd(dst_xmm, dst_xmm);

            host_->uni_vcvtdq2ps(dst_xmm, dst_xmm);
            host_->uni_vbroadcastss(dst_vmm, dst_xmm);

            break;
        }
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::jit_io_multi_dt_helper_t(jit_generator *host,
        const cpu_isa_t &isa,
        const std::unordered_set<data_type_t, std::hash<int>> &data_types,
        const io_conf_t<Vmm> &io_conf,
        const utils::optional_t<io_tail_conf_t<Vmm>> &tail_conf,
        const utils::optional_t<io_emu_bf16_conf_t> &bf16_conf,
        const std::map<data_type_t, io_saturation_conf_t<Vmm>>
                &saturation_confs,
        const utils::optional_t<io_gather_conf_t<Vmm>> &gather_conf) {
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
                            dt == data_type::bf16 ? bf16_conf : utils::null_opt,
                            store_saturation_needed
                                    ? utils::optional_t<io_saturation_conf_t<
                                            Vmm>> {saturation_conf->second}
                                    : utils::null_opt,
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
        const std::unordered_set<data_type_t, std::hash<int>>
                &store_data_types) {

    for (const auto &dt : store_data_types) {
        const auto it = storage_.find(dt);
        if (it != storage_.cend()) {
            if (it->second->saturation_conf_.has_value()) it->second->init_saturate_f32();
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
jit_io_multi_dt_helper_t<Vmm>::~jit_io_multi_dt_helper_t() = default;

template class io_conf_t<Xbyak::Zmm>;
template class io_conf_t<Xbyak::Ymm>;
template class io_conf_t<Xbyak::Xmm>;

template class io_tail_conf_t<Xbyak::Zmm>;
template class io_tail_conf_t<Xbyak::Ymm>;
template class io_tail_conf_t<Xbyak::Xmm>;

template class io_saturation_conf_t<Xbyak::Zmm>;
template class io_saturation_conf_t<Xbyak::Ymm>;
template class io_saturation_conf_t<Xbyak::Xmm>;

template class io_gather_conf_t<Xbyak::Zmm>;
template class io_gather_conf_t<Xbyak::Ymm>;
template class io_gather_conf_t<Xbyak::Xmm>;

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