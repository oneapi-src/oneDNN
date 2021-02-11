
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include <algorithm>
#include <cassert>
#include <set>
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "jit_prelu_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace prelu {

cpu_isa_t get_supported_isa() {
    if (mayiuse(avx512_core_bf16))
        return avx512_core_bf16;
    else if (mayiuse(avx512_core))
        return avx512_core;
    else if (mayiuse(avx512_common))
        return avx512_common;
    else if (mayiuse(avx2))
        return avx2;
    else if (mayiuse(avx))
        return avx;
    else if (mayiuse(sse41))
        return sse41;

    return isa_any;
}

static int get_vlen(const cpu_isa_t &isa) noexcept {
    if (isa == avx512_core_bf16)
        return cpu_isa_traits<avx512_core_bf16>::vlen;
    else if (isa == avx512_core)
        return cpu_isa_traits<avx512_core>::vlen;
    else if (isa == avx512_common)
        return cpu_isa_traits<avx512_common>::vlen;
    else if (isa == avx2)
        return cpu_isa_traits<avx2>::vlen;
    else if (isa == avx)
        return cpu_isa_traits<avx>::vlen;
    return cpu_isa_traits<sse41>::vlen;
}

int get_n_vregs(const cpu_isa_t &isa) noexcept {
    if (isa == avx512_core_bf16)
        return cpu_isa_traits<avx512_core_bf16>::n_vregs;
    else if (isa == avx512_core)
        return cpu_isa_traits<avx512_core>::n_vregs;
    else if (isa == avx512_common)
        return cpu_isa_traits<avx512_common>::n_vregs;
    else if (isa == avx2)
        return cpu_isa_traits<avx2>::n_vregs;
    else if (isa == avx)
        return cpu_isa_traits<avx>::n_vregs;
    return cpu_isa_traits<sse41>::n_vregs;
}

bool is_s8u8(const std::set<data_type_t> &tensor_data_types) noexcept {
    return std::any_of(tensor_data_types.cbegin(), tensor_data_types.cend(),
            [](const data_type_t &dt) {
                return utils::one_of(dt, data_type::s8, data_type::u8);
            });
}

int get_simd_w(const std::set<data_type_t> &tensor_data_types) noexcept {
    const auto &isa = prelu::get_supported_isa();

    return (isa == avx && is_s8u8(tensor_data_types))
            ? vmm_traits_t<Xbyak::Xmm>::vlen / sizeof(float)
            : prelu::get_vlen(isa) / sizeof(float);
}

static bool dims_equal(
        const dims_t &lhs_dims, const dims_t &rhs_dims, const dim_t ndims) {

    for (dim_t i = 0; i < ndims; ++i) {
        if (lhs_dims[i] != rhs_dims[i]) return false;
    }

    return true;
}

static bool is_full_bcast(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs) {
    const auto lhs_ndims = lhs.ndims();
    const auto rhs_ndims = rhs.ndims();
    const dims_t &lhs_dims = lhs.dims();
    const dims_t &rhs_dims = rhs.dims();

    if (lhs_ndims == rhs_ndims && dims_equal(lhs_dims, rhs_dims, lhs_ndims)
            && lhs.format_kind() == rhs.format_kind()) {

        if (lhs.is_blocking_desc()) {
            const auto &lhs_bd = lhs.blocking_desc();
            const auto &rhs_bd = rhs.blocking_desc();
            const dims_t &lhs_strides = lhs_bd.strides;
            const dims_t &rhs_strides = rhs_bd.strides;
            const dims_t &lhs_inner_blks = lhs_bd.inner_blks;
            const dims_t &rhs_inner_blks = rhs_bd.inner_blks;
            const dims_t &lhs_inner_idxs = lhs_bd.inner_idxs;
            const dims_t &rhs_inner_idxs = rhs_bd.inner_idxs;

            return lhs_bd.inner_nblks == rhs_bd.inner_nblks
                    && dims_equal(lhs_strides, rhs_strides, lhs_ndims)
                    && dims_equal(lhs_inner_blks, rhs_inner_blks, lhs_ndims)
                    && dims_equal(lhs_inner_idxs, rhs_inner_idxs, lhs_ndims);
        }

        return true;
    }

    return false;
}

static bool is_per_oc_bcast(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs) {

    const auto &rhs_dims = rhs.dims();
    const auto &lhs_dims = lhs.dims();
    const auto &rhs_ndims = rhs.ndims();

    bool bcast_per_oc_exists = rhs_dims[0] == 1 && rhs_dims[1] == lhs_dims[1];

    if (bcast_per_oc_exists) {
        for (int dim_id = 2; dim_id < rhs_ndims; ++dim_id) {
            bcast_per_oc_exists = bcast_per_oc_exists && rhs_dims[dim_id] == 1;
        }
    }

    return bcast_per_oc_exists;
}

bcast get_bcast_type(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs) {

    if (is_full_bcast(lhs, rhs)) return bcast::full;
    const auto &lhs_ndims = lhs.ndims();
    const auto &rhs_ndims = rhs.ndims();

    if (lhs_ndims != rhs_ndims || lhs_ndims < 2) return bcast::unsupported;

    if (is_per_oc_bcast(lhs, rhs)) {
        const auto &strides = lhs.blocking_desc().strides;

        if (!lhs.is_plain())
            return bcast::per_oc_blocked;
        else if (strides[1] == 1)
            return bcast::per_oc_n_spatial_c;
        else if (strides[0] >= strides[1]
                && IMPLICATION(lhs_ndims >= 3, strides[1] >= strides[2]))
            return bcast::per_oc_n_c_spatial;
    }

    return bcast::unsupported;
}

bool dt_supported(const std::set<data_type_t> &tensor_data_types) noexcept {

    const bool tensor_dt_valid = std::all_of(tensor_data_types.cbegin(),
            tensor_data_types.cend(), [](const data_type_t &dt) {
                return utils::one_of(dt, data_type::bf16, data_type::f32,
                        data_type::s32, data_type::u8, data_type::s8);
            });

    if (tensor_dt_valid) {
        const bool any_tensor_bf16 = std::any_of(tensor_data_types.cbegin(),
                tensor_data_types.cend(),
                [](const data_type_t &dt) { return dt == data_type::bf16; });

        return IMPLICATION(any_tensor_bf16, mayiuse(avx512_core));
    }

    return false;
}

template <typename Vmm>
jit_prelu_io_helper_t<Vmm>::jit_prelu_io_helper_t(jit_generator *host,
        const cpu_isa_t &isa, const data_type_t &data_type,
        std::size_t tail_size, const Xbyak::Opmask &tail_opmask,
        const Vmm &tail_vmm_mask, const Vmm &vreg_zero_saturation,
        const Vmm &vreg_saturation_ubound, const Xbyak::Reg64 &reg_tmp)
    : host_(host)
    , isa_(isa)
    , data_type_(data_type)
    , tail_size_(tail_size)
    , tail_opmask_(tail_opmask)
    , vreg_zero_saturation_(vreg_zero_saturation)
    , vreg_saturation_ubound_(vreg_saturation_ubound)
    , tail_vmm_mask_(tail_vmm_mask)
    , reg_tmp_(reg_tmp)
    , bf16_supported_(utils::one_of(isa, avx512_core, avx512_core_bf16))
    , bf16_emu_(data_type_ == data_type::bf16 && isa == avx512_core
                      ? utils::make_unique<bf16_emulation_t>(host_,
                              host_->zmm28, host_->zmm29, host_->zmm30,
                              host_->rax, host_->zmm31)
                      : nullptr) {

    assert(utils::one_of(data_type_, data_type::bf16, data_type::f32,
                   data_type::s8, data_type::u8, data_type::s32)
            && "Supported data types bf16, f32, s8, u8,s32");

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

    if (bf16_emu_) bf16_emu_->init_vcvtneps2bf16();
}

template <typename Vmm>
jit_prelu_io_helper_t<Vmm>::jit_prelu_io_helper_t(jit_generator *host,
        const cpu_isa_t &isa, const data_type_t &data_type,
        std::size_t tail_size, const Xbyak::Opmask &tail_opmask,
        const Vmm &tail_vmm_mask, const Xbyak::Reg64 &reg_tmp)
    : jit_prelu_io_helper_t(host, isa, data_type, tail_size, tail_opmask,
            tail_vmm_mask, Vmm(0), Vmm(0), reg_tmp) {

    assert(utils::one_of(data_type_, data_type::bf16, data_type::f32)
            && "In case of u8s8s32 dt saturation regs must be passed");
}

template <typename Vmm>
jit_prelu_io_helper_t<Vmm>::~jit_prelu_io_helper_t() = default;

template <>
void jit_prelu_io_helper_t<Xbyak::Zmm>::prepare_tail_mask() {
    if (!tail_size_) return;

    const int mask_f32 = (1 << tail_size_) - 1;
    const Xbyak::Reg32 regw_tmp = reg_tmp_.cvt32();
    host_->mov(regw_tmp, mask_f32);
    host_->kmovw(tail_opmask_, regw_tmp);
}

template <typename Vmm>
void jit_prelu_io_helper_t<Vmm>::prepare_tail_mask() {
    if (!tail_size_ || isa_ == sse41) return;

    static const uint32_t mask_f32[14]
            = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

    host_->mov(reg_tmp_, reinterpret_cast<size_t>(&mask_f32[7 - tail_size_]));
    host_->vmovups(tail_vmm_mask_, host_->ptr[reg_tmp_]);
}

template <typename Vmm>
void jit_prelu_io_helper_t<Vmm>::init_saturate_f32() {
    if (utils::one_of(data_type_, data_type::u8, data_type::s8, data_type::s32))
        host_->init_saturate_f32(vreg_zero_saturation_, vreg_saturation_ubound_,
                reg_tmp_, data_type::f32, data_type_);
}

template <>
void jit_prelu_io_helper_t<Xbyak::Zmm>::load(const Xbyak::Address &src_addr,
        const Xbyak::Zmm &dst_raw_vmm, bool tail) {

    const auto dst_vmm
            = tail ? (dst_raw_vmm | tail_opmask_ | host_->T_z) : dst_raw_vmm;

    switch (data_type_) {
        case data_type::f32: host_->uni_vmovups(dst_vmm, src_addr); break;
        case data_type::s32: host_->uni_vcvtdq2ps(dst_vmm, src_addr); break;
        case data_type::bf16:
            if (bf16_supported_) {
                host_->vpmovzxwd(dst_vmm, src_addr);
                host_->vpslld(dst_raw_vmm, dst_raw_vmm, 0x10);
                break;
            }
        case data_type::s8: {
            host_->uni_vpmovsxbd(dst_vmm, src_addr);
            host_->uni_vcvtdq2ps(dst_vmm, dst_raw_vmm);
            break;
        }
        case data_type::u8: {
            host_->uni_vpmovzxbd(dst_vmm, src_addr);
            host_->uni_vcvtdq2ps(dst_vmm, dst_raw_vmm);
            break;
        }
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
void jit_prelu_io_helper_t<Vmm>::load(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm, bool tail) {

    if (tail
            && (isa_ == sse41
                    || utils::one_of(
                            data_type_, data_type::s8, data_type::u8))) {
        host_->uni_vxorps(dst_vmm, dst_vmm, dst_vmm);
        host_->load_data(data_type_, dst_vmm, src_addr, tail_size_);
    } else if (utils::one_of(data_type_, data_type::f32, data_type::s32)) {
        if (tail)
            host_->vmaskmovps(dst_vmm, tail_vmm_mask_, src_addr);
        else
            host_->uni_vmovups(dst_vmm, src_addr);
    } else if (data_type_ == data_type::s8)
        host_->uni_vpmovsxbd(dst_vmm, src_addr);
    else if (data_type_ == data_type::u8)
        host_->uni_vpmovzxbd(dst_vmm, src_addr);
    else
        assert(!"unsupported data type");

    if (utils::one_of(
                data_type_, data_type::s32, data_type::u8, data_type::s8)) {
        host_->uni_vcvtdq2ps(dst_vmm, dst_vmm);
    }
}

template <>
void jit_prelu_io_helper_t<Xbyak::Zmm>::store(const Xbyak::Zmm &src_raw_vmm,
        const Xbyak::Address &dst_raw_addr, bool tail) {

    const auto src_vmm = tail ? (src_raw_vmm | tail_opmask_) : src_raw_vmm;

    if (utils::one_of(
                data_type_, data_type::s32, data_type::s8, data_type::u8)) {
        host_->saturate_f32(src_raw_vmm, vreg_zero_saturation_,
                vreg_saturation_ubound_, data_type_);
        host_->uni_vcvtps2dq(src_vmm, src_raw_vmm);
    }

    const auto dst_addr = tail ? (dst_raw_addr | tail_opmask_) : dst_raw_addr;

    switch (data_type_) {
        case data_type::f32:
        case data_type::s32: host_->uni_vmovups(dst_addr, src_raw_vmm); break;
        case data_type::bf16: {
            if (bf16_supported_) {
                const Xbyak::Ymm src_ymm {src_raw_vmm.getIdx()};
                if (bf16_emu_)
                    bf16_emu_->vcvtneps2bf16(src_ymm, src_raw_vmm);
                else
                    host_->vcvtneps2bf16(src_ymm, src_raw_vmm);
                host_->vmovdqu16(dst_addr, src_ymm);
                break;
            }
        }
        case data_type::s8: host_->vpmovsdb(dst_raw_addr, src_vmm); break;
        case data_type::u8: host_->vpmovusdb(dst_raw_addr, src_vmm); break;
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
void jit_prelu_io_helper_t<Vmm>::store(
        const Vmm &src_vmm, const Xbyak::Address &dst_addr, bool tail) {

    static constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;

    if (data_type_ != data_type::f32) {
        host_->saturate_f32(src_vmm, vreg_zero_saturation_,
                vreg_saturation_ubound_, data_type_);
        host_->uni_vcvtps2dq(src_vmm, src_vmm);
    }

    const auto prepare_bytes_to_store = [&]() {
        host_->uni_vpackssdw(src_vmm, src_vmm, vreg_zero_saturation_);
        if (is_ymm) {
            const auto src_ymm = Xbyak::Ymm(src_vmm.getIdx());
            host_->vpermq(src_ymm, src_ymm, 0x58);
        }

        if (data_type_ == data_type::s8)
            host_->uni_vpacksswb(src_vmm, src_vmm, vreg_zero_saturation_);
        else
            host_->uni_vpackuswb(src_vmm, src_vmm, vreg_zero_saturation_);
    };

    const auto store_tail = [&] {
        switch (data_type_) {
            case data_type::f32:
            case data_type::s32: {
                if (isa_ == sse41)
                    host_->store_bytes(
                            src_vmm, dst_addr, tail_size_ * sizeof(int32_t));
                else
                    host_->vmaskmovps(dst_addr, tail_vmm_mask_, src_vmm);
                break;
            }
            case data_type::s8:
            case data_type::u8: {
                prepare_bytes_to_store();
                host_->store_bytes(src_vmm, dst_addr, tail_size_);
                break;
            }
            default: assert(!"unsupported data type");
        }
    };

    const auto store_no_tail = [&]() {
        switch (data_type_) {
            case data_type::f32:
            case data_type::s32: host_->uni_vmovups(dst_addr, src_vmm); break;
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

template <typename Vmm>
void jit_prelu_io_helper_t<Vmm>::broadcast(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm) {
    switch (data_type_) {
        case data_type::f32: host_->uni_vbroadcastss(dst_vmm, src_addr); break;
        case data_type::bf16:
            if (bf16_supported_) {
                host_->vpbroadcastw(dst_vmm, src_addr);
                host_->vpslld(dst_vmm, dst_vmm, 0x10);
                break;
            }
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
jit_prelu_io_multi_dt_helper_t<Vmm>::jit_prelu_io_multi_dt_helper_t(
        jit_generator *host, const cpu_isa_t &isa,
        const std::unordered_set<data_type_t, std::hash<int>> &data_types,
        std::size_t tail_size, const Xbyak::Opmask &tail_opmask,
        const Vmm &tail_vmm_mask, const Xbyak::Reg64 &reg_tmp,
        const std::map<data_type_t, std::pair<Vmm, Vmm>> &saturation_vmms) {

    assert(!data_types.empty());
    for (const auto &dt : data_types) {
        // can be replaced by try_emplace from C++17
        if (storage_.find(dt) == storage_.cend()) {

            const auto saturation = saturation_vmms.find(dt);
            const bool store_saturation_needed
                    = saturation != saturation_vmms.cend();

            if (store_saturation_needed) {
                const auto &lower_bound_vmm = saturation->second.first;
                const auto &upper_bound_vmm = saturation->second.second;
                storage_.emplace(dt,
                        std::make_shared<jit_prelu_io_helper_t<Vmm>>(host, isa,
                                dt, tail_size, tail_opmask, tail_vmm_mask,
                                lower_bound_vmm, upper_bound_vmm, reg_tmp));

            } else {
                storage_.emplace(dt,
                        std::make_shared<jit_prelu_io_helper_t<Vmm>>(host, isa,
                                dt, tail_size, tail_opmask, tail_vmm_mask,
                                reg_tmp));
            }
        }
    }
}

template <typename Vmm>
std::shared_ptr<jit_prelu_io_helper_t<Vmm>>
jit_prelu_io_multi_dt_helper_t<Vmm>::at(const data_type_t dt) const {
    const auto it = storage_.find(dt);
    if (it != storage_.cend()) return it->second;

    return nullptr;
}

template <typename Vmm>
void jit_prelu_io_multi_dt_helper_t<Vmm>::prepare_tail_mask() {
    return storage_.cbegin()->second->prepare_tail_mask();
}

template <typename Vmm>
void jit_prelu_io_multi_dt_helper_t<Vmm>::init_saturate_f32(
        const std::unordered_set<data_type_t, std::hash<int>>
                &store_data_types) {

    for (const auto &dt : store_data_types) {
        const auto it = storage_.find(dt);
        if (it != storage_.cend()) { it->second->init_saturate_f32(); }
    }
}

template <typename Vmm>
jit_prelu_io_multi_dt_helper_t<Vmm>::~jit_prelu_io_multi_dt_helper_t()
        = default;

template class jit_prelu_io_helper_t<Xbyak::Zmm>;
template class jit_prelu_io_helper_t<Xbyak::Ymm>;
template class jit_prelu_io_helper_t<Xbyak::Xmm>;

template class jit_prelu_io_multi_dt_helper_t<Xbyak::Zmm>;
template class jit_prelu_io_multi_dt_helper_t<Xbyak::Ymm>;
template class jit_prelu_io_multi_dt_helper_t<Xbyak::Xmm>;

} // namespace prelu
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
