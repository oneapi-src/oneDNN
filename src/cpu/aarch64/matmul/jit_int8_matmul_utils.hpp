/*******************************************************************************
* Copyright 2025 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_INT8_MATMUL_UTILS_HPP
#define CPU_AARCH64_JIT_INT8_MATMUL_UTILS_HPP

#include "common/primitive.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/matmul/jit_int8_kernel_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace Xbyak_aarch64;
struct jit_int8_matmul_utils_kernel_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_matmul_utils_kernel_t);

    XReg reg_param = abi_param1;
    XReg reg_src = x3;
    XReg reg_dst = x4;
    XReg reg_scl = x5;
    XReg reg_zp = x6;
    XReg reg_tmp = x7;
    XReg reg_tmp_2 = x8;
    XReg reg_max = x9;
    XReg reg_min = x10;
    XReg reg_tmp_1 = x11;
    XReg reg_k_loop = x12;
    XReg reg_m_loop = x13;
    XReg reg_loop = x14;
    XReg reg_tail = x15;
    XReg reg_m_tail = x16;
    XReg reg_aux_b = x17;
    XReg reg_aux_a = x18;

    PReg prd_ld = p1;
    PReg prd_st = p2;
    PReg prd_p1 = p3;
    PReg prd_p2 = p4;
    PReg prd_p3 = p5;

    XReg reg_n_loop = reg_m_loop;
    XReg reg_n_tail = reg_m_tail;

    int f32_dt_sz = 4;

    void operator()(const dyn_params_t *p) {
        return jit_generator::operator()(p);
    }

    jit_int8_matmul_utils_kernel_t(const dyn_vals_t &k) : dyn_(k) {}
    ~jit_int8_matmul_utils_kernel_t() override = default;

private:
    void gen_reo_a();
    void gen_reo_b();
    void reo_A_8x8(int, int);
    void reo_B_8x24(int, int);
    void generate() override;

    dyn_vals_t dyn_;
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif