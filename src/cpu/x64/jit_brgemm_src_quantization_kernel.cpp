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

#include "cpu/x64/jit_brgemm_src_quantization_kernel.hpp"

#define GET_OFF(field) offsetof(src_quantization_runtime_params_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;
using namespace std::placeholders;

template <cpu_isa_t isa>
void jit_brgemm_src_quantization_kernel_t<isa>::load_src(Vmm vmm_load, const Xbyak::Address& addr) {
    switch (jcp_.src_dt) {
        case data_type::f32: {
            uni_vmovups(vmm_load, addr);
            break;
        }
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_brgemm_src_quantization_kernel_t<isa>::generate() {
    preamble();

    mov(reg_src, ptr[param1 + GET_OFF(src_ptr)]);
    mov(reg_qsrc, ptr[param1 + GET_OFF(qsrc_ptr)]);
    mov(reg_src_scales, ptr[param1 + GET_OFF(src_scales_ptr)]);
    mov(reg_ic_size, ptr[param1 + GET_OFF(ic_size)]);

    Xbyak::Label ic_loop_label;
    Xbyak::Label ic_end_label;

    size_t src_dt_size = types::data_type_size(jcp_.src_dt);
    size_t qsrc_dt_size = types::data_type_size(jcp_.qsrc_dt);
    size_t src_scales_dt_size = types::data_type_size(data_type::f32);

    static const float negative_zero[16] = {
        -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f,
        -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f
    };

    static const float positive_one[16] = {
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f
    };

    static const float int8_max[16] = {
        127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f,
        127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f
    };

    mov(reg_tmp, (size_t)negative_zero);
    uni_vmovups(vmm_sign_bit_mask(), ptr[reg_tmp]);

    mov(reg_tmp, (size_t)positive_one);
    uni_vmovups(vmm_one(), ptr[reg_tmp]);

    mov(reg_tmp, (size_t)int8_max);
    uni_vmovups(vmm_int8_max(), ptr[reg_tmp]);

    L(ic_loop_label);
    {
        cmp(reg_ic_size, jcp_.ic_quant_block);
        jl(ic_end_label, T_NEAR);

        assert(!(jcp_.ic_quant_block % vec_size));

        int ic_blocks = jcp_.ic_quant_block / vec_size;
        uni_vpxor(vmm_max(), vmm_max(), vmm_max());
        for (int icb = 0; icb < ic_blocks; icb++) {
            load_src(vmm_src(), ptr[reg_src + icb * vec_size * src_dt_size]);
            vandnps(vmm_src(), vmm_sign_bit_mask(), vmm_src());
            uni_vmaxps(vmm_max(), vmm_max(), vmm_src());
        }

        if (isa == avx512_core) {
            Xbyak::Zmm max_zmm = Xbyak::Zmm(vmm_max().getIdx());
            Xbyak::Zmm aux_zmm = Xbyak::Zmm(vmm_aux().getIdx());
            vshuff32x4(aux_zmm, max_zmm, max_zmm, 0x4E);
            uni_vmaxps(max_zmm, max_zmm, aux_zmm);
            vshuff32x4(aux_zmm, max_zmm, max_zmm, 0xB1);
            uni_vmaxps(max_zmm, max_zmm, aux_zmm);
        } else if (isa == avx2) {
            Xbyak::Ymm max_ymm = Xbyak::Ymm(vmm_max().getIdx());
            Xbyak::Ymm aux_ymm = Xbyak::Ymm(vmm_aux().getIdx());
            vperm2i128(aux_ymm, max_ymm, max_ymm, 0x01);
            uni_vmaxps(max_ymm, max_ymm, aux_ymm);
        } else {
            assert(!"unsupported isa");
        }
        uni_vshufps(vmm_aux(), vmm_max(), vmm_max(), 0x4E);
        uni_vmaxps(vmm_max(), vmm_max(), vmm_aux());
        uni_vshufps(vmm_aux(), vmm_max(), vmm_max(), 0xB1);
        uni_vmaxps(vmm_max(), vmm_max(), vmm_aux());

        auto vmm_dscale = vmm_max();
        uni_vbroadcastss(vmm_dscale, Xmm(vmm_dscale.getIdx()));
        uni_vdivps(vmm_dscale, vmm_dscale, vmm_int8_max());

        // todo: check zero case ( (dscale != 0) ? (1.0f / dscale) : 0;)
        uni_vdivps(vmm_qscale(), vmm_one(), vmm_dscale);

        uni_vmovss(ptr[reg_src_scales], Xmm(vmm_dscale.getIdx()));
        for (int icb = 0; icb < ic_blocks; icb++) {
            load_src(vmm_src(), ptr[reg_src + icb * vec_size * src_dt_size]);
            uni_vmulps(vmm_src(), vmm_src(), vmm_qscale());
            uni_vcvtps2dq(vmm_src(), vmm_src());

            if (isa == avx512_core) {
                vpmovsdb(ptr[reg_qsrc + icb * vec_size * qsrc_dt_size], vmm_src());
            } else {
                uni_vpackssdw(vmm_src(), vmm_src(), vmm_src());
                vpermq(Ymm(vmm_src().getIdx()), Ymm(vmm_src().getIdx()), 0x08);
                uni_vpacksswb(vmm_src(), vmm_src(), vmm_src());
                vmovq(ptr[reg_qsrc + icb * vec_size * qsrc_dt_size], Xmm(vmm_src().getIdx()));
            }
        }

        sub(reg_ic_size, jcp_.ic_quant_block);
        add(reg_src, src_dt_size * jcp_.ic_quant_block);
        add(reg_qsrc, qsrc_dt_size * jcp_.ic_quant_block);
        add(reg_src_scales, src_scales_dt_size);

        jmp(ic_loop_label, T_NEAR);
    }
    L(ic_end_label);

    postamble();
}

template struct jit_brgemm_src_quantization_kernel_t<avx512_core>;
template struct jit_brgemm_src_quantization_kernel_t<avx2>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl