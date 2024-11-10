/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_sse41_1x1_conv_kernel_f32.hpp"
#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::utils;

using namespace Xbyak;

jit_sse41_1x1_conv_kernel_f32::jit_sse41_1x1_conv_kernel_f32(
        const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jit_generator(jit_name(), sse41), jcp(ajcp), attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary) {
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr size_t helper_vmm_idx = 15;
        const size_t tail_size = jcp.oc_without_padding % simd_w_;
        static constexpr bool use_exact_tail_scalar_bcast = false;

        const binary_injector::rhs_arg_static_params_t rhs_arg_static_params {
                helper_vmm_idx, r13, r14, r15, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(dst_orig),
                memory_desc_wrapper(dst_md), tail_size,
                use_exact_tail_scalar_bcast};
        const binary_injector::static_params_t static_params {
                this->param1, rhs_arg_static_params};
        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<sse41>>(
                this, jcp.post_ops, static_params);
    }
}

void jit_sse41_1x1_conv_kernel_f32::generate_bcast_loop(int load_loop_blk) {
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, reg_bcast_loop_work);

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop);
    {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            generate_reduce_loop(load_loop_blk, jcp.ur);
            if (i < num_substeps - 1) {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_substep);
            } else {
                add(aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data,
                        jcp.bcast_loop_output_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_output_substep);
            }
        }
        sub(bcast_loop_iter, jcp.bcast_block);
        cmp(bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        cmp(bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        generate_reduce_loop(load_loop_blk, jcp.ur_tail);
        L(bcast_loop_tail_out);
    }
}

size_t jit_sse41_1x1_conv_kernel_f32::get_fwd_output_ptr_l_off(
        int i, int j, int n, bool ignore_dw_conv = false) const {
    return i * get_output_i_offset(jcp, ignore_dw_conv)
            + j * get_output_j_offset(jcp) + n * 4;
}

static int reg_accum_idx(
        const int load_loop_blk, const int i, const int j, const int n) {
    return 2 * j * load_loop_blk + 2 * i + n + 1;
}

template <typename F>
static void iterate(const int load_loop_blk, const int ur, const F &f) {
    for (int j = 0; j < ur; ++j)
        for (int i = 0; i < load_loop_blk; ++i)
            for (int n = 0; n < 2; n++)
                f(i, j, n);
}
void jit_sse41_1x1_conv_kernel_f32::apply_postops(
        const int load_loop_blk, const int ur) {
    injector_utils::vmm_index_set_t vmm_idxs;
    if (jcp.with_binary) {
        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
        iterate(load_loop_blk, ur, [&](const int i, const int j, const int n) {
            const bool mask_flag = (2 * i + n) == load_loop_blk - 1;
            const size_t aux_output_offset
                    = get_fwd_output_ptr_l_off(i, j, n, true) * sizeof(float);
            const auto vmm_idx = reg_accum_idx(load_loop_blk, i, j, n);
            vmm_idxs.emplace(vmm_idx);

            rhs_arg_params.vmm_idx_to_out_reg.emplace(
                    vmm_idx, aux_reg_output_data);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                    vmm_idx, aux_output_offset);
            if (mask_flag) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
        });
        const injector_utils::register_preserve_guard_t register_guard(
                this, {abi_param1, aux_reg_output_data});
        const size_t reg_guard_stack_occupied
                = register_guard.stack_space_occupied();
        if (jcp.with_dw_conv) {
            add(aux_reg_output_data,
                    ptr[rsp + reg_dw_binary_output_off
                            + reg_guard_stack_occupied]);
        }
        mov(abi_param1,
                ptr[rsp + reg_abi_param1_backup + reg_guard_stack_occupied]);

        postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params);
    } else {
        iterate(load_loop_blk, ur, [&](const int i, const int j, const int n) {
            vmm_idxs.emplace(reg_accum_idx(load_loop_blk, i, j, n));
        });
        postops_injector_->compute_vector_range(vmm_idxs);
    }
}

void jit_sse41_1x1_conv_kernel_f32::generate_reduce_loop(
        int load_loop_blk, int ur) {
    auto reg_load = [ur, load_loop_blk](int i, int n) {
        return Xmm(2 * ur * load_loop_blk + 2 * i + n + 1);
    };

    auto reg_accum = [load_loop_blk](int i, int j, int n) {
        return Xmm(reg_accum_idx(load_loop_blk, i, j, n));
    };

    auto bias_ptr = [this](int i, int n) {
        return ptr[reg_bias_data + sizeof(float) * jcp.oc_block * i
                + n * 4 * sizeof(float)];
    };

    auto bcast_ptr = [&](int u, int j) {
        assert(j < jcp.ur);
        assert(u <= jcp.reduce_loop_unroll);
        size_t offt;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            assert(jcp.reduce_loop_unroll == jcp.reduce_block);
            offt = get_bcast_offset(jcp, u, j);
        } else
            offt = u * jcp.ic_block + j;
        return ptr[aux_reg_bcast_data + offt];
    };

    auto load_ptr = [&](int u, int i, int n) {
        size_t offt;
        size_t u0 = u % jcp.reduce_loop_unroll;
        size_t u1 = u / jcp.reduce_loop_unroll;
        switch (jcp.prop_kind) {
            case backward_data:
                offt = (i * jcp.oc_block + u0) * jcp.ic_block;
                break;
            case backward_weights:
                offt = (i * jcp.os + u0) * jcp.oc_block;
                break;
            default: offt = (i * jcp.ic + u0) * jcp.oc_block;
        }
        return ptr[aux_reg_load_data + u1 * jcp.reduce_loop_load_step
                + sizeof(float) * offt + n * 4 * sizeof(float)];
    };

    auto output_ptr = [this](int i, int j, int n) {
        switch (jcp.prop_kind) {
            case backward_data:
                return ptr[aux_reg_output_data
                        + (i * jcp.is + j) * jcp.ic_block * sizeof(float)
                        + n * 4 * sizeof(float)];
            case backward_weights:
                return ptr[aux_reg_output_data
                        + (i ? reg_output_stride * i
                             : 0) // TODO: Xbyak should allow 0 scale
                        + sizeof(float) * jcp.oc_block * j
                        + n * 4 * sizeof(float)];
            default:
                return ptr[aux_reg_output_data
                        + get_fwd_output_ptr_l_off(i, j, n) * sizeof(float)];
        }
    };

    auto init = [&]() {
        Label init_done;
        Label init_zero;

        if (jcp.with_bias
                && one_of(jcp.prop_kind, forward_training, forward_inference)) {
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jz(init_zero);

            for (int i = 0; i < load_loop_blk; i++)
                for (int j = 0; j < ur; ++j) {
                    movups(reg_accum(i, j, 0), bias_ptr(i, 0));
                    movups(reg_accum(i, j, 1), bias_ptr(i, 1));
                }
            jmp(init_done);
        }

        L(init_zero);
        for (int i = 0; i < load_loop_blk; ++i)
            for (int j = 0; j < ur; ++j) {
                auto r0 = reg_accum(i, j, 0);
                auto r1 = reg_accum(i, j, 1);
                xorps(r0, r0);
                xorps(r1, r1);
            }

        L(init_done);

        // load weights
        for (int i = 0; i < load_loop_blk; ++i) {
            movups(reg_load(i, 0), load_ptr(0, i, 0));
            movups(reg_load(i, 1), load_ptr(0, i, 1));
        }

        movss(reg_bcast, bcast_ptr(0, 0));
        shufps(reg_bcast, reg_bcast, 0);
    }; // init()

    auto store = [&]() {
        Label store_noadd;

        if (!jcp.with_sum) {
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jnz(store_noadd, T_NEAR);
        }

        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
                auto r0 = reg_accum(i, j, 0);
                auto r1 = reg_accum(i, j, 1);
                addps(r0, output_ptr(i, j, 0));
                addps(r1, output_ptr(i, j, 1));
            }

        L(store_noadd);

        if (jcp.with_eltwise || jcp.with_binary) {
            assert(ur * load_loop_blk < 14);

            Label store_nopostops;
            test(reg_reduce_pos_flag, FLAG_REDUCE_LAST);
            jz(store_nopostops, T_NEAR);

            apply_postops(load_loop_blk, ur);

            L(store_nopostops);
        }

        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
                movups(output_ptr(i, j, 0), reg_accum(i, j, 0));
                movups(output_ptr(i, j, 1), reg_accum(i, j, 1));
            }
    };

    auto fma_block = [&](bool last_block) {
        for (int u = 0; u < jcp.reduce_loop_unroll; ++u) {
            for (int j = 0; j < ur; ++j) {
                for (int i = 0; i < load_loop_blk; ++i) {
                    mulps(reg_load(i, 0), reg_bcast);
                    mulps(reg_load(i, 1), reg_bcast);
                    addps(reg_accum(i, j, 0), reg_load(i, 0));
                    addps(reg_accum(i, j, 1), reg_load(i, 1));

                    if (j == ur - 1
                            && !(last_block
                                    && u == jcp.reduce_loop_unroll - 1)) {
                        movups(reg_load(i, 0), load_ptr(u + 1, i, 0));
                        movups(reg_load(i, 1), load_ptr(u + 1, i, 1));
                    }
                }
                if (j < ur - 1) {
                    movss(reg_bcast, bcast_ptr(u, j + 1));
                    shufps(reg_bcast, reg_bcast, 0);
                }
            } // for ur
            if (!last_block || u < jcp.reduce_loop_unroll - 1) {
                movss(reg_bcast, bcast_ptr(u + 1, 0));
                shufps(reg_bcast, reg_bcast, 0);
            }
        } // for reduce_loop_unroll
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);
    mov(aux_reg_bcast_data, aux1_reg_bcast_data);

    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop);
    {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
} // reduce_loop()

void jit_sse41_1x1_conv_kernel_f32::generate_diff_bias_loop(int load_loop_blk) {
    if (!jcp.with_bias || jcp.prop_kind != backward_weights) return;

    Label diff_bias_loop, diff_bias_loop_out, diff_bias_init_out;
    Label diff_bias_load;

    auto diff_bias_ptr = [this](int i, int n) {
        return ptr[reg_diff_bias_data + i * jcp.oc_block * sizeof(float)
                + 4 * n * sizeof(float)];
    };

    auto load_ptr = [this](int u, int i, int n) {
        return ptr[aux_reg_load_data
                + (i * jcp.os + u) * jcp.oc_block * sizeof(float)
                + 4 * n * sizeof(float)];
    };

    auto diff_bias_reg = [](int i, int n) { return Xmm(2 * i + n + 1); };

    mov(reg_diff_bias_data, ptr[rsp + reg_diff_bias_data_stack_offt]);
    cmp(reg_diff_bias_data, 0);
    je(diff_bias_loop_out, T_NEAR);

    test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
    jz(diff_bias_load, T_NEAR);

    for (int i = 0; i < load_loop_blk; ++i) {
        auto r0 = diff_bias_reg(i, 0);
        auto r1 = diff_bias_reg(i, 1);
        xorps(r0, r0);
        xorps(r1, r1);
    }
    jmp(diff_bias_init_out, T_NEAR);

    L(diff_bias_load);
    for (int i = 0; i < load_loop_blk; ++i) {
        movups(diff_bias_reg(i, 0), diff_bias_ptr(i, 0));
        movups(diff_bias_reg(i, 1), diff_bias_ptr(i, 1));
    }

    L(diff_bias_init_out);
    mov(aux_reg_load_data, reg_load_data);
    mov(reduce_loop_iter, reg_reduce_loop_work);
    L(diff_bias_loop);
    {
        for (int u = 0; u < jcp.reduce_loop_unroll; ++u)
            for (int i = 0; i < load_loop_blk; ++i) {
                addps(diff_bias_reg(i, 0), load_ptr(u, i, 0));
                addps(diff_bias_reg(i, 1), load_ptr(u, i, 1));
            }
        assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jnz(diff_bias_loop, T_NEAR);
    }

    for (int i = 0; i < load_loop_blk; i++) {
        movups(diff_bias_ptr(i, 0), diff_bias_reg(i, 0));
        movups(diff_bias_ptr(i, 1), diff_bias_reg(i, 1));
    }

    add(reg_diff_bias_data, load_loop_blk * jcp.oc_block * sizeof(float));
    mov(ptr[rsp + reg_diff_bias_data_stack_offt], reg_diff_bias_data);

    L(diff_bias_loop_out);
}

void jit_sse41_1x1_conv_kernel_f32::generate() {
    preamble();

    sub(rsp, stack_space_needed);
    if (jcp.with_binary) {
        // backup abi_param1 for usage in post_ops processing
        mov(ptr[rsp + reg_abi_param1_backup], abi_param1);

        if (jcp.with_dw_conv) {
            // zero initialize dw binary output offset (store on stack)
            const auto zeroed_reg = r15;
            xor_(zeroed_reg, zeroed_reg);
            mov(ptr[rsp + reg_dw_binary_output_off], zeroed_reg);
        }
    }

    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);
    if (jcp.with_bias) {
        if (jcp.prop_kind == backward_weights) {
            mov(reg_diff_bias_data, ptr[param1 + GET_OFF(bias_data)]);
            mov(ptr[rsp + reg_diff_bias_data_stack_offt], reg_diff_bias_data);
        } else
            mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    }

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(first_last_flag)]);
    if (jcp.prop_kind == backward_weights)
        mov(reg_output_stride, ptr[param1 + GET_OFF(output_stride)]);

    auto generate_load_loop_body = [&](int load_loop_blk) {
        const size_t offst_with_dw_conv
                = get_load_loop_output_fwd_offset(jcp, load_loop_blk);
        const size_t offst_wo_dw_conv
                = get_load_loop_output_fwd_offset(jcp, load_loop_blk, true);
        generate_bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        switch (jcp.prop_kind) {
            case forward_training:
            case forward_inference:
                add(reg_bias_data,
                        load_loop_blk * jcp.oc_block * sizeof(float));
                add(reg_output_data, offst_with_dw_conv);
                if (jcp.with_binary && jcp.with_dw_conv) {
                    mov(aux_reg_load_data, ptr[rsp + reg_dw_binary_output_off]);
                    add(aux_reg_load_data,
                            offst_wo_dw_conv - offst_with_dw_conv);
                    mov(ptr[rsp + reg_dw_binary_output_off], aux_reg_load_data);
                }
                break;
            case backward_data:
                add(reg_output_data,
                        load_loop_blk * jcp.is * jcp.ic_block * sizeof(float));
                break;
            case backward_weights:
                for (int i = 0; i < load_loop_blk; i++)
                    add(reg_output_data, reg_output_stride);
                break;
            default: assert(!"invalid prop_kind");
        }
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    };

    Label load_loop_blk_8;
    Label load_loop_blk_16;
    Label load_loop_blk_24;
    Label load_loop_blk_end;

    cmp(reg_load_loop_work, 8);
    jle(load_loop_blk_8, T_NEAR);

    cmp(reg_load_loop_work, 32);
    je(load_loop_blk_16, T_NEAR);

    cmp(reg_load_loop_work, 16);
    jle(load_loop_blk_16, T_NEAR);

    L(load_loop_blk_24);
    {
        generate_diff_bias_loop(3);
        generate_load_loop_body(3);
        cmp(reg_load_loop_work, 32);
        je(load_loop_blk_16);
        cmp(reg_load_loop_work, 24);
        jge(load_loop_blk_24);
    }

    cmp(reg_load_loop_work, 8);
    jle(load_loop_blk_8, T_NEAR);

    L(load_loop_blk_16);
    {
        generate_diff_bias_loop(2);
        generate_load_loop_body(2);
        cmp(reg_load_loop_work, 16);
        jge(load_loop_blk_16);
    }

    L(load_loop_blk_8);
    {
        cmp(reg_load_loop_work, 0);
        je(load_loop_blk_end, T_NEAR);
        generate_diff_bias_loop(1);
        generate_load_loop_body(1);
    }

    L(load_loop_blk_end);

    add(rsp, stack_space_needed);

    postamble();

    if (jcp.with_eltwise)
        postops_injector_->prepare_table(/* generate = */ true);
}

status_t jit_sse41_1x1_conv_kernel_f32::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, int nthreads) {
    // disabling verbose dispatch messages for unsupported isa for better readability
    if (!mayiuse(sse41)) return status::unimplemented;

    // TODO (Roma): this code is duplicated from the generic kernel; maybe the
    // configuration struct could do some stuff below
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();

    jcp.nthr = nthreads;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][0];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[0];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;

    jcp.typesize_in = sizeof(prec_traits<data_type::f32>::type);
    jcp.typesize_out = sizeof(prec_traits<data_type::f32>::type);

    const auto &post_ops = attr.post_ops_;

    const int dw_conv_ind = post_ops.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;
    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.
    jcp.with_sum = post_ops.find(primitive_kind::sum, 0, dw_conv_ind) != -1;
    const int eltwise_ind
            = post_ops.find(primitive_kind::eltwise, 0, dw_conv_ind);
    jcp.with_eltwise = eltwise_ind != -1;
    const int binary_ind
            = post_ops.find(primitive_kind::binary, 0, dw_conv_ind);
    const int prelu_ind = post_ops.find(primitive_kind::prelu, 0, dw_conv_ind);
    jcp.with_binary = !everyone_is(-1, binary_ind, prelu_ind);

    if (dw_conv_ind >= 0) {
        // dw_conv and post_ops after it are handled externally, so skip them
        jcp.post_ops.entry_.assign(post_ops.entry_.cbegin(),
                post_ops.entry_.cbegin() + dw_conv_ind);
    } else {
        jcp.post_ops = post_ops;
    }

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = true;
    static constexpr bool sum_requires_zp_zero = true;
    const bool post_ops_ok_ = post_ops_ok(post_ops_ok_args_t(sse41,
            {eltwise, binary, sum}, jcp.post_ops, &dst_d, sum_at_pos_0_only,
            sum_requires_scale_one, sum_requires_zp_zero));
    VDISPATCH_CONV_IC(post_ops_ok_, VERBOSE_UNSUPPORTED_POSTOP);

    const auto dat_tag_nxc = utils::pick(ndims - 3, nwc, nhwc);
    const auto dat_tag_blocked = utils::pick(ndims - 3, nCw8c, nChw8c);
    jcp.src_tag = src_d.mb_stride_relaxed_match(dat_tag_nxc, dat_tag_blocked);
    jcp.dst_tag = dst_d.mb_stride_relaxed_match(dat_tag_nxc, dat_tag_blocked);
    const bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
    const auto dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_blocked;

    const int is_bwd_d = jcp.prop_kind == backward_data;
    format_tag_t wei_tag = with_groups
            ? utils::pick(2 * ndims - 6 + is_bwd_d, gOIw8i8o, gOIw8o8i,
                    gOIhw8i8o, gOIhw8o8i)
            : utils::pick(2 * ndims - 6 + is_bwd_d, OIw8i8o, OIw8o8i, OIhw8i8o,
                    OIhw8o8i);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);

    bool args_ok = true && jcp.ngroups == 1 && jcp.src_tag == dat_tag
            && jcp.wei_tag == wei_tag && jcp.dst_tag == dat_tag;
    VDISPATCH_CONV_IC(args_ok, VERBOSE_UNSUPPORTED_TAG);

    const int simd_w = 4;

    jcp.ic_block = jcp.oc_block = simd_w * 2;

    args_ok = true && jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0
            && jcp.t_pad == 0 && jcp.l_pad == 0 && jcp.stride_w == 1
            && jcp.stride_h == 1 // TODO: support some strides
            && jcp.ow == jcp.iw && jcp.oh == jcp.ih // enforce rpad=0
            && jcp.kh == 1 && jcp.kw == 1;
    VDISPATCH_CONV_IC(
            args_ok, VERBOSE_BLOCKING_FAIL, "bad blocking parameters");

    jcp.ur = 1;
    if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);

    int load_blocking {0};
    int load_blocking_max {0};
    int bcast_blocking {0};
    int bcast_blocking_max {0};
    int reduce_blocking {0};

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        jcp.reduce_dim = jcp.ic;
        jcp.reduce_block = jcp.ic_block;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.is;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? 1 : jcp.is) * sizeof(float);
        jcp.reduce_loop_load_step
                = jcp.reduce_loop_unroll * jcp.oc_block * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur
                * (is_data_layout_nxc ? jcp.oc : jcp.oc_block) * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur
                * (is_data_layout_nxc ? jcp.ic : jcp.ic_block) * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.ic * jcp.oc_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        load_blocking = is_data_layout_nxc
                ? jcp.load_dim
                : 120; // assumes the kernel is jcp.ur x 3
        load_blocking_max = is_data_layout_nxc ? jcp.load_dim : 144;
        bcast_blocking = 128; // affects load balancing across threads
        bcast_blocking_max = 192;
        reduce_blocking = is_data_layout_nxc ? jcp.reduce_dim
                                             : 128; // affects L1$ utilization
    } else if (jcp.prop_kind == backward_data) {
        jcp.reduce_dim = jcp.oc;
        jcp.reduce_block = jcp.oc_block;

        jcp.load_dim = jcp.ic;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.os;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
                = jcp.reduce_loop_unroll * jcp.os * sizeof(float);
        jcp.reduce_loop_load_step
                = jcp.reduce_loop_unroll * jcp.ic * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.oc_block * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.ic_block;

        load_blocking = 96; // assumes the kernel is jcp.ur x 3
        load_blocking_max = 144;
        bcast_blocking = 128; // affects load balancing across threads
        bcast_blocking_max = 196;
        reduce_blocking = 64; // affects L1$ utilization
    } else if (jcp.prop_kind == backward_weights) {
        jcp.reduce_dim = jcp.os;
        jcp.reduce_block = 1;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
                = jcp.reduce_loop_unroll * jcp.ic_block * sizeof(float);
        jcp.reduce_loop_load_step
                = jcp.reduce_loop_unroll * jcp.oc_block * sizeof(float);

        jcp.bcast_loop_output_step
                = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = jcp.oc_block * jcp.ur * sizeof(float);
        jcp.bcast_loop_bcast_step = jcp.ic_block * jcp.is * sizeof(float);
        jcp.bcast_loop_bcast_substep = jcp.ur * sizeof(float);

        jcp.load_loop_load_step = jcp.oc_block * jcp.os * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        while (true) {
            if (load_blocking <= 32)
                break;
            else if (load_blocking % 2 == 0)
                load_blocking /= 2;
            else if (load_blocking % 3 == 0)
                load_blocking /= 3;
            else
                break;
        }
        load_blocking *= jcp.load_block;
        load_blocking_max = load_blocking;
        assert(jcp.load_dim % load_blocking == 0);

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        while (true) {
            if (bcast_blocking <= 9)
                break;
            else if (bcast_blocking % 2 == 0)
                bcast_blocking /= 2;
            else if (bcast_blocking % 3 == 0)
                bcast_blocking /= 3;
            else
                break;
        }
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(jcp.bcast_dim % bcast_blocking == 0);

        reduce_blocking = 128; // affects L1$ utilization
    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);

    assert(jcp.bcast_block % jcp.ur == 0);
    jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
