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

#include "common/float16.hpp"
#include "common/dnnl_thread.hpp"

#include "cpu/platform.hpp"
#if DNNL_X64
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_fp16cvt.hpp"
#include "cpu/x64/jit_uni_convert_xf16.hpp"
#endif

namespace dnnl {
namespace impl {

bool try_cvt_float_to_float16(float16_t *out, const float *inp) {
#if DNNL_X64
    using namespace cpu::x64;
    if (mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)) {
        cvt_xf16_support::jit_call_t p;
        p.inp = (void *)inp;
        p.out = (void *)out;
        static const jit_cvt_ps_to_xf16_t cvt_one_ps_to_f16(data_type::f16, 1);
        cvt_one_ps_to_f16(&p);
        return true;
    }
#endif
    return false;
}

void cvt_float_to_float16(float16_t *out, const float *inp, size_t nelems) {
#if DNNL_X64
    using namespace cpu::x64;
    if (mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)) {
        cvt_xf16_support::jit_call_t p_;
        p_.inp = (void *)inp;
        p_.out = (void *)out;
        p_.nelems = nelems;
        static const jit_cvt_ps_to_xf16_t cvt_ps_to_f16(data_type::f16);
        cvt_ps_to_f16(&p_);
        return;
    }
#endif

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float16_t>(inp[i]);
}

void cvt_float16_to_float(float *out, const float16_t *inp, size_t nelems) {
#if DNNL_X64
    using namespace cpu::x64;
    if (mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)) {
        static const jit_cvt_xf16_to_ps_t kernel(data_type::f16, false);
        return kernel(out, inp, nelems);
    }
#endif

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = inp[i];
}

void add_floats_and_cvt_to_float16(
        float16_t *out, const float *inp0, const float *inp1, size_t nelems) {
#if DNNL_X64
    if (cpu::x64::mayiuse(cpu::x64::cpu_isa_t::avx512_core_fp16)) {
        cpu::x64::f16_support::jit_call_t p_;
        p_.inp = (void *)inp0;
        p_.add = (void *)inp1;
        p_.out = (void *)out;
        p_.nelems = nelems;
        static const cpu::x64::jit_avx512_core_fp16_add_cvt_ps_to_f16_t
                add_cvt_ps_to_f16;
        add_cvt_ps_to_f16(&p_);
        return;
    }
#endif

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float16_t>(inp0[i] + inp1[i]);
}

} // namespace impl
} // namespace dnnl
