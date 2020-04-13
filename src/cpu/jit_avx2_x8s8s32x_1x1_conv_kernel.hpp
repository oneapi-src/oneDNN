/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef JIT_AVX2_X8S8S32X_1X1_CONV_KERNEL_HPP
#define JIT_AVX2_X8S8S32X_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/eltwise/jit_uni_eltwise_injector.hpp"
#include "cpu/jit_generator.hpp"
#include "cpu/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct jit_avx2_x8s8s32x_1x1_conv_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_x8s8s32x_1x1_conv_fwd_ker_t)
    jit_avx2_x8s8s32x_1x1_conv_kernel(
            const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_
                    = new jit_uni_eltwise_injector_f32<avx2>(this, jcp.eltwise);

        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *))this->getCode();
    }

    ~jit_avx2_x8s8s32x_1x1_conv_kernel() { delete eltwise_injector_; }

    static bool post_ops_ok(
            jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
            const primitive_attr_t &attr, int nthreads, bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    bool maybe_eltwise(int position);

    int get_tail_size() { return jcp.oc_without_padding % jcp.oc_block; }

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

private:
    jit_uni_eltwise_injector_f32<avx2> *eltwise_injector_;

    const Xbyak::Reg64 reg_bcast_data = r8;
    const Xbyak::Reg64 reg_ptr_scales = r8;
    const Xbyak::Reg64 reg_output_data = r9;
    const Xbyak::Reg64 reg_load_data = r10;
    const Xbyak::Reg64 reg_ptr_sum_scale = r10;
    const Xbyak::Reg64 reg_reduce_loop_work = r11;
    const Xbyak::Reg64 reg_bias_data = r12;
    const Xbyak::Reg64 reg_comp_data = r12;
    const Xbyak::Reg64 reg_init_bcast = r13;
    const Xbyak::Reg64 reg_store_bcast = r13;
    const Xbyak::Reg64 reg_reduce_loop_iter = r13;
    const Xbyak::Reg64 aux_reg_bcast_data = r14;
    const Xbyak::Reg64 aux_reg_load_data = r15;
    const Xbyak::Reg64 reg_reduce_pos_flag = rax;
    const Xbyak::Reg64 aux1_reg_bcast_data = rbx;
    const Xbyak::Reg64 reg_bcast_loop_work = rbx;
    const Xbyak::Reg64 reg_bcast_loop_iter = rdx;
    const Xbyak::Reg64 reg_load_loop_work = rsi;
    const Xbyak::Reg64 aux_reg_output_data = abi_not_param1;

    const Xbyak::Ymm ymm_tmp = Xbyak::Ymm(12);
    const Xbyak::Ymm ymm_one = Xbyak::Ymm(13);
    const Xbyak::Ymm ymm_zero = Xbyak::Ymm(14);
    const Xbyak::Ymm ymm_shift = Xbyak::Ymm(14);
    const Xbyak::Ymm ymm_bcast = Xbyak::Ymm(15);

    constexpr static int simd_w = 8;
    constexpr static int reg64_size = sizeof(int64_t);
    constexpr static int bcast_loop_work_off = 0;
    constexpr static int reg_bias_data_off = 1 * reg64_size;
    constexpr static int reg_bcast_data_off = 2 * reg64_size;
    constexpr static int reg_load_data_off = 3 * reg64_size;
    constexpr static int reg_ptr_sum_scale_off = 4 * reg64_size;
    constexpr static int reg_comp_data_off = 5 * reg64_size;
    constexpr static int stack_space_needed = 6 * reg64_size;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate();
    void cvt2ps(data_type_t type_in, const Xbyak::Ymm &ymm_in,
            const Xbyak::Reg64 &reg, int offset, int load_size);
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
