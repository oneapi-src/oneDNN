/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include <float.h>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_brgemm_weights_decompression_kernel.hpp"

#define GET_OFF(field) offsetof(weights_decompression_runtime_params_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;
using namespace std::placeholders;

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::init_decomp_params(std::function<Vmm(int)> vmm_params, Xbyak::Reg64 reg_params, bool broadcast_values, data_type_t element_type) {
    size_t oc_blocks_num = div_up(jcp_.oc_size, vec_size);
    for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
        if (broadcast_values) {
            switch (element_type) {
                case data_type::f32: {
                    uni_vbroadcastss(vmm_params(ocb), ptr[reg_params]);
                    break;
                }
                case data_type::u8: {
                    auto xmm_params = Xmm(vmm_params(ocb).getIdx());
                    auto reg_tmp_32 = Reg32(reg_tmp.getIdx());
                    movzx(reg_tmp_32, ptr[reg_params]);
                    uni_vmovq(xmm_params, reg_tmp);
                    uni_vcvtdq2ps(xmm_params, xmm_params);
                    uni_vbroadcastss(vmm_params(ocb), xmm_params);
                    break;
                }
                default: assert(!"unsupported data type");
            }
        } else {
            const auto load_addr = ptr[reg_params + ocb * vec_size * types::data_type_size(element_type)];
            switch (element_type) {
                case data_type::f32: {
                    uni_vmovups(vmm_params(ocb), load_addr);
                    break;
                }
                case data_type::u8: {
                    uni_vpmovzxbd(vmm_params(ocb), load_addr);
                    uni_vcvtdq2ps(vmm_params(ocb), vmm_params(ocb));
                    break;
                }
                default: assert(!"unsupported data type");
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::load_weights(Vmm vmm_load, const Xbyak::Address& addr, int ic) {
    switch (jcp_.weights_dt) {
        case data_type::u8: {
            uni_vpmovzxbd(vmm_load, addr);
            uni_vcvtdq2ps(vmm_load, vmm_load);
            break;
        }
        case data_type::s8: {
            uni_vpmovsxbd(vmm_load, addr);
            uni_vcvtdq2ps(vmm_load, vmm_load);
            break;
        }
        case data_type::u4: {
            uni_vpmovzxbd(vmm_load, addr);
            if (ic % 2 == 0) {
                uni_vpsrld(vmm_load, vmm_load, 4);
            } else {
                uni_vpslld(vmm_load, vmm_load, 28);
                uni_vpsrld(vmm_load, vmm_load, 28);
            }
            uni_vcvtdq2ps(vmm_load, vmm_load);
            break;
        }
        case data_type::s4: {
            uni_vpmovsxbd(vmm_load, addr);
            if (ic % 2 == 0) {
                vpsrad(vmm_load, vmm_load, 4);
            } else {
                uni_vpslld(vmm_load, vmm_load, 28);
                vpsrad(vmm_load, vmm_load, 28);
            }
            uni_vcvtdq2ps(vmm_load, vmm_load);
            break;
        }
        case data_type::nf4: {
            uni_vpmovzxbd(vmm_load, addr);
            if (ic % 2 == 0) {
                uni_vpsrld(vmm_load, vmm_load, 4);
            } else {
                uni_vpslld(vmm_load, vmm_load, 28);
                uni_vpsrld(vmm_load, vmm_load, 28);
            }

            if (isa == avx2) {
                auto res = vmm_weights(1);
                auto mask = vmm_weights(2);
                vpcmpgtd(mask, vmm_load, vmm_mask7());
                vpermd(res, vmm_load, vmm_lookup_low());
                vpsubd(vmm_load, vmm_load, vmm_mask8());
                vpermd(vmm_load, vmm_load, vmm_lookup_high());
                vblendvps(vmm_load, res, vmm_load, mask);
            } else {
                vpermd(vmm_load, vmm_load, vmm_lookup());
            }
            break;
        }
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::store_weights(const Xbyak::Address& addr, Vmm vmm_store) {
    switch (jcp_.decomp_buffer_dt) {
        case data_type::f32: {
            uni_vmovups(addr, vmm_store);
            break;
        }
        case data_type::bf16: {
            Ymm ymm_store = Ymm(vmm_store.getIdx());
            vcvtneps2bf16(ymm_store, vmm_store);
            vmovdqu16(addr, ymm_store);
            break;
        }
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::generate() {
    preamble();

    mov(reg_weights, ptr[param1 + GET_OFF(weights_ptr)]);
    mov(reg_decomp_buffer, ptr[param1 + GET_OFF(decomp_buffer_ptr)]);
    if (jcp_.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(scales_ptr)]);
    }
    if (jcp_.with_zero_points) {
        mov(reg_zero_points, ptr[param1 + GET_OFF(zero_points_ptr)]);
    }
    mov(reg_ic_size, ptr[param1 + GET_OFF(ic_size)]);

    if (jcp_.weights_dt == data_type::nf4) {
        static const float lookup[16] = {
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0
        };

        static const int32_t mask8[16] = {
            8, 8, 8, 8, 8, 8, 8, 8
        };
        static const int32_t mask7[16] = {
            7, 7, 7, 7, 7, 7, 7, 7
        };

        if (isa == avx2) {
            mov(reg_tmp, (size_t)lookup);
            uni_vmovups(vmm_lookup_low(), ptr[reg_tmp]);
            uni_vmovups(vmm_lookup_high(), ptr[reg_tmp + 8 * sizeof(float)]);
            mov(reg_tmp, (size_t)mask8);
            uni_vmovups(vmm_mask8(), ptr[reg_tmp]);
            mov(reg_tmp, (size_t)mask7);
            uni_vmovups(vmm_mask7(), ptr[reg_tmp]);
        } else {
            mov(reg_tmp, (size_t)lookup);
            uni_vmovups(vmm_lookup(), ptr[reg_tmp]);
        }
    }

    if (jcp_.with_scales)
        init_decomp_params(std::bind(&jit_brgemm_weights_decompression_kernel_t::vmm_scales, this, _1), reg_scales, jcp_.broadcast_scales, data_type::f32);

    if (jcp_.with_zero_points)
        init_decomp_params(std::bind(&jit_brgemm_weights_decompression_kernel_t::vmm_zero_points, this, _1), reg_zero_points, jcp_.broadcast_zero_points, jcp_.zero_points_dt);

    size_t oc_blocks_num = div_up(jcp_.oc_size, vec_size);

    Xbyak::Label ic_loop_label;
    Xbyak::Label ic_end_label;

    size_t weights_dt_size = types::data_type_size(jcp_.weights_dt);
    size_t typesize_scale = one_of(jcp_.weights_dt, data_type::nf4, data_type::s4, data_type::u4) ? 2 : 1;
    size_t decomp_buf_dt_size = types::data_type_size(jcp_.decomp_buffer_dt);

    L(ic_loop_label);
    {
        cmp(reg_ic_size, 1);
        jl(ic_end_label, T_NEAR);

        if (jcp_.decomp_buffer_dt == data_type::bf16) {
            for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
                for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                    size_t weights_offset;
                    if (jcp_.weights_dt == data_type::u8 || jcp_.weights_dt == data_type::s8)
                        weights_offset = (ic * jcp_.oc_size + ocb * vec_size) * weights_dt_size / typesize_scale;
                    else
                        weights_offset = ocb * jcp_.ic_internal_size * vec_size * weights_dt_size / typesize_scale;
                    auto vmm_load = vmm_weights(ic);
                    const auto load_addr = ptr[reg_weights + weights_offset];
                    load_weights(vmm_load, load_addr, ic);

                    if (jcp_.with_zero_points)
                        uni_vsubps(vmm_load, vmm_load, vmm_zero_points(ocb));
                    if (jcp_.with_scales)
                        uni_vmulps(vmm_load, vmm_load, vmm_scales(ocb));
                }

                auto ymm_store0 = Ymm(vmm_weights(0).getIdx());
                auto ymm_store1 = Ymm(vmm_weights(1).getIdx());
                auto ymm_aux0 = Ymm(vmm_weights(2).getIdx());
                auto ymm_aux1 = Ymm(vmm_weights(3).getIdx());

                vcvtneps2bf16(ymm_store0, vmm_weights(0));
                vcvtneps2bf16(ymm_store1, vmm_weights(1));
                vpunpcklwd(ymm_aux0, ymm_store0, ymm_store1);
                vpunpckhwd(ymm_aux1, ymm_store0, ymm_store1);
                vperm2i128(ymm_store0, ymm_aux0, ymm_aux1, 0x20);
                vperm2i128(ymm_store1, ymm_aux0, ymm_aux1, 0x31);

                for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                    auto ymm_store = Ymm(vmm_weights(ic).getIdx());
                    size_t decomp_buffer_offset = (ocb * jcp_.ic_internal_size + ic) * vec_size * decomp_buf_dt_size;
                    const auto decomp_buffer_addr = ptr[reg_decomp_buffer + decomp_buffer_offset];
                    vmovdqu16(decomp_buffer_addr, ymm_store);
                }
            }
        } else {
            for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
                for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                    size_t weights_offset = ocb * jcp_.ic_internal_size * vec_size * weights_dt_size / typesize_scale;
                    const auto weights_addr = ptr[reg_weights + weights_offset];
                    load_weights(vmm_weights(0), weights_addr, ic);

                    if (jcp_.with_zero_points)
                        uni_vsubps(vmm_weights(0), vmm_weights(0), vmm_zero_points(ocb));
                    if (jcp_.with_scales)
                        uni_vmulps(vmm_weights(0), vmm_weights(0), vmm_scales(ocb));

                    size_t decomp_buffer_offset = (ic * jcp_.oc_size + ocb * vec_size) * decomp_buf_dt_size;
                    const auto decomp_buffer_addr = ptr[reg_decomp_buffer + decomp_buffer_offset];
                    store_weights(decomp_buffer_addr, vmm_weights(0));
                }
            }
        }

        dec(reg_ic_size);
        add(reg_weights, weights_dt_size * jcp_.oc_size * jcp_.ic_internal_size / typesize_scale);
        add(reg_decomp_buffer, decomp_buf_dt_size * jcp_.oc_size * jcp_.ic_internal_size);

        jmp(ic_loop_label, T_NEAR);
    }
    L(ic_end_label);

    postamble();
}

template struct jit_brgemm_weights_decompression_kernel_t<avx512_core>;
template struct jit_brgemm_weights_decompression_kernel_t<avx2>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl