/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef CPU_X64_JIT_SSE41_1X1_CONV_KERNEL_F32_HPP
#define CPU_X64_JIT_SSE41_1X1_CONV_KERNEL_F32_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_sse41_1x1_conv_kernel_f32 : public jit_generator {
    jit_sse41_1x1_conv_kernel_f32(
            const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<sse41>(
                    this, jcp.eltwise);

        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *))this->getCode();
    }

    ~jit_sse41_1x1_conv_kernel_f32() { delete eltwise_injector_; }

    static bool post_ops_ok(
            jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
            int nthreads);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_1x1_conv_kernel_f32)

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    using xmm_t = const Xbyak::Xmm;

    reg64_t reg_bcast_data = rax;
    reg64_t reg_load_data = rsi;
    reg64_t reg_output_data = rbx;
    reg64_t aux_reg_bcast_data = rdx;
    reg64_t aux1_reg_bcast_data = abi_not_param1;
    reg64_t aux_reg_load_data = abi_param1;
    reg64_t aux_reg_output_data = rbp;
    reg64_t reg_load_loop_work = r9;
    reg64_t reg_bcast_loop_work = r10;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t load_loop_iter = r13;
    reg64_t imm_addr64 = load_loop_iter;
    reg64_t bcast_loop_iter = r14;
    reg64_t reduce_loop_iter = r15;
    reg64_t reg_reduce_pos_flag = r8;
    reg64_t reg_output_stride = r12;
    reg64_t reg_bias_data = r12;
    reg64_t reg_diff_bias_data = bcast_loop_iter;

    int reg_diff_bias_data_stack_offt = 0;
    int stack_space_needed = 8;

    xmm_t reg_bcast = xmm_t(15);

    jit_uni_eltwise_injector_f32<sse41> *eltwise_injector_;

    void generate_bcast_loop(int load_loop_blk);
    void generate_reduce_loop(int load_loop_blk, int ur);
    void generate_diff_bias_loop(int load_loop_blk);

    inline bool is_bcast_layout_nxc() {
        switch (jcp.prop_kind) {
            case prop_kind::forward_training:
            case prop_kind::forward_inference:
                return utils::one_of(jcp.src_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
            default: assert(!"invalid prop_kind"); return false;
        }
    }

    inline bool is_load_layout_nxc() {
        return jcp.prop_kind == prop_kind::backward_weights
                && utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
    }

    inline bool is_out_layout_nxc() {
        switch (jcp.prop_kind) {
            case prop_kind::forward_training:
            case prop_kind::forward_inference:
                return utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
            default: assert(!"invalid prop_kind"); return false;
        }
    }

    inline size_t get_bcast_j_offset() {
        return is_bcast_layout_nxc() ? jcp.reduce_dim : jcp.reduce_loop_unroll;
    }

    inline size_t get_bcast_offset(int u, int j) {
        if (is_bcast_layout_nxc() || u != jcp.reduce_loop_unroll) {
            return j * get_bcast_j_offset() + u;
        } else {
            return (jcp.bcast_dim + j) * get_bcast_j_offset();
        }
    }

    inline size_t get_output_i_offset() {
        if (is_out_layout_nxc()) {
            return jcp.load_block;
        } else {
            return (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) * jcp.load_block;
        }
    }

    inline size_t get_output_j_offset() {
        return is_out_layout_nxc() ? jcp.load_dim : jcp.load_block;
    }

    inline size_t get_load_loop_output_fwd_offset(int load_loop_blk) {
        size_t offset = load_loop_blk * jcp.oc_block * sizeof(float);
        if (!is_out_layout_nxc()) {
            offset *= jcp.with_dw_conv ? jcp.ow : jcp.os;
        }
        return offset;
    }

    void generate();
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
