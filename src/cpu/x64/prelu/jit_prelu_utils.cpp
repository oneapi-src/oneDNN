
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
#include "jit_prelu_utils.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"

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

int get_vlen(const cpu_isa_t &isa) noexcept {
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

bcast get_bcast_type(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs) {

    if (lhs == rhs) return bcast::full;
    const auto &lhs_ndims = lhs.ndims();
    const auto &rhs_ndims = rhs.ndims();

    if (lhs_ndims != rhs_ndims || lhs_ndims < 2) return bcast::unsupported;

    const auto &rhs_dims = rhs.dims();
    const auto &lhs_dims = lhs.dims();

    bool bcast_per_oc_exists = rhs_dims[0] == 1 && rhs_dims[1] == lhs_dims[1];

    if (bcast_per_oc_exists) {
        for (int dim_id = 2; dim_id < rhs_ndims; ++dim_id) {
            bcast_per_oc_exists = bcast_per_oc_exists && rhs_dims[dim_id] == 1;
        }
    }

    if (bcast_per_oc_exists) {
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

template <typename Vmm>
jit_prelu_io_helper<Vmm>::jit_prelu_io_helper(jit_generator *host,
        const cpu_isa_t &isa, const data_type_t &data_type,
        std::size_t tail_size, const Xbyak::Opmask &tail_opmask,
        const Vmm &tail_vmm_mask, const Xbyak::Reg64 &reg_tmp)
    : host_(host)
    , isa_(isa)
    , data_type_(data_type)
    , tail_size_(tail_size)
    , tail_opmask_(tail_opmask)
    , tail_vmm_mask_(tail_vmm_mask)
    , reg_tmp_(reg_tmp)
    , bf16_supported_(utils::one_of(isa, avx512_core, avx512_core_bf16))
    , bf16_emu_(data_type_ == data_type::bf16 && isa == avx512_core
                      ? utils::make_unique<bf16_emulation_t>(host_,
                              host_->zmm28, host_->zmm29, host_->zmm30,
                              host_->rax, host_->zmm31)
                      : nullptr) {

    if (bf16_emu_) bf16_emu_->init_vcvtneps2bf16();
}

template <typename Vmm>
jit_prelu_io_helper<Vmm>::~jit_prelu_io_helper() = default;

template <>
void jit_prelu_io_helper<Xbyak::Zmm>::prepare_tail_mask() {
    if (!tail_size_) return;

    const int mask_f32 = (1 << tail_size_) - 1;
    const Xbyak::Reg32 regw_tmp = reg_tmp_.cvt32();
    host_->mov(regw_tmp, mask_f32);
    host_->kmovw(tail_opmask_, regw_tmp);
}

template <typename Vmm>
void jit_prelu_io_helper<Vmm>::prepare_tail_mask() {
    if (!tail_size_ || isa_ == sse41) return;

    static const uint32_t mask_f32[14]
            = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

    host_->mov(reg_tmp_, reinterpret_cast<size_t>(&mask_f32[7 - tail_size_]));
    host_->vmovups(tail_vmm_mask_, host_->ptr[reg_tmp_]);
}

template <typename Vmm>
void jit_prelu_io_helper<Vmm>::load(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm, bool tail) {
    if (tail)
        load_tail(src_addr, dst_vmm);
    else
        switch (data_type_) {
            case data_type::f32: host_->uni_vmovups(dst_vmm, src_addr); break;
            case data_type::bf16:
                if (bf16_supported_) {
                    host_->vpmovzxwd(dst_vmm, src_addr);
                    host_->vpslld(dst_vmm, dst_vmm, 0x10);
                    break;
                }
            default: assert(!"unsupported data type");
        }
}

template <>
void jit_prelu_io_helper<Xbyak::Zmm>::load_tail(
        const Xbyak::Address &src_addr, const Xbyak::Zmm &dst_vmm) {
    switch (data_type_) {
        case data_type::f32:
            host_->uni_vmovups_tail(dst_vmm, tail_opmask_, src_addr);
            break;
        case data_type::bf16:
            if (bf16_supported_) {
                host_->vpmovzxwd(dst_vmm | tail_opmask_, src_addr);
                host_->vpslld(dst_vmm, dst_vmm, 0x10);
            }
            break;
        default: assert(!"unsupported data type");
    }
}

template <>
void jit_prelu_io_helper<Xbyak::Ymm>::load_tail(
        const Xbyak::Address &src_addr, const Xbyak::Ymm &dst_vmm) {
    host_->uni_vmovups_tail(dst_vmm, tail_vmm_mask_, src_addr);
}

template <>
void jit_prelu_io_helper<Xbyak::Xmm>::load_tail(
        const Xbyak::Address &src_addr, const Xbyak::Xmm &dst_vmm) {
    if (isa_ == sse41) {
        for (size_t tail_elem = 0; tail_elem < tail_size_; tail_elem++)
            host_->pinsrd(dst_vmm,
                    host_->ptr[src_addr.getRegExp()
                            + Xbyak::RegExp(tail_elem * sizeof(float))],
                    tail_elem);
    } else
        host_->vmaskmovps(dst_vmm, tail_vmm_mask_, src_addr);
}

template <>
void jit_prelu_io_helper<Xbyak::Zmm>::store_tail(
        const Xbyak::Zmm &src_vmm, const Xbyak::Address &dst_addr) {
    switch (data_type_) {
        case data_type::f32:
            host_->uni_vmovups_tail(dst_addr, tail_opmask_, src_vmm);
            break;
        case data_type::bf16: {
            if (bf16_supported_) {
                const Xbyak::Ymm ymm_src {src_vmm.getIdx()};
                if (bf16_emu_)
                    bf16_emu_->vcvtneps2bf16(ymm_src, src_vmm);
                else
                    host_->vcvtneps2bf16(ymm_src, src_vmm);
                host_->vmovdqu16(dst_addr | tail_opmask_, ymm_src);
                break;
            }
        }
        default: assert(!"unsupported data type");
    }
}

template <>
void jit_prelu_io_helper<Xbyak::Ymm>::store_tail(
        const Xbyak::Ymm &src_vmm, const Xbyak::Address &dst_addr) {
    host_->uni_vmovups_tail(dst_addr, tail_vmm_mask_, src_vmm);
}

template <>
void jit_prelu_io_helper<Xbyak::Xmm>::store_tail(
        const Xbyak::Xmm &src_vmm, const Xbyak::Address &dst_addr) {
    if (isa_ == sse41) {
        for (size_t tail_elem = 0; tail_elem < tail_size_; tail_elem++)
            host_->pextrd(host_->ptr[dst_addr.getRegExp()
                                  + Xbyak::RegExp(tail_elem * sizeof(float))],
                    src_vmm, tail_elem);
    } else {
        host_->vmaskmovps(dst_addr, tail_vmm_mask_, src_vmm);
    }
}

template <>
void jit_prelu_io_helper<Xbyak::Zmm>::store(
        const Xbyak::Zmm &src_vmm, const Xbyak::Address &dst_addr, bool tail) {
    if (tail)
        store_tail(src_vmm, dst_addr);
    else
        switch (data_type_) {
            case data_type::f32: host_->uni_vmovups(dst_addr, src_vmm); break;
            case data_type::bf16: {
                if (bf16_supported_) {
                    const Xbyak::Ymm src_ymm {src_vmm.getIdx()};
                    if (bf16_emu_)
                        bf16_emu_->vcvtneps2bf16(src_ymm, src_vmm);
                    else
                        host_->vcvtneps2bf16(src_ymm, src_vmm);
                    host_->vmovdqu16(dst_addr, src_ymm);
                    break;
                }
            }
            default: assert(!"unsupported data type");
        }
}

template <typename Vmm>
void jit_prelu_io_helper<Vmm>::store(
        const Vmm &src_vmm, const Xbyak::Address &dst_addr, bool tail) {
    if (tail)
        store_tail(src_vmm, dst_addr);
    else
        host_->uni_vmovups(dst_addr, src_vmm);
}

template <typename Vmm>
void jit_prelu_io_helper<Vmm>::broadcast(
        const Xbyak::Address &src_addr, const Vmm &dst_vmm) {
    switch (data_type_) {
        case data_type::f32: host_->uni_vbroadcastss(dst_vmm, src_addr); break;
        case data_type::bf16:
            if (bf16_supported_) {
                host_->vpbroadcastw(dst_vmm, src_addr);
                host_->vpslld(dst_vmm, dst_vmm, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
    }
}

template class jit_prelu_io_helper<Xbyak::Zmm>;
template class jit_prelu_io_helper<Xbyak::Ymm>;
template class jit_prelu_io_helper<Xbyak::Xmm>;

} // namespace prelu
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
