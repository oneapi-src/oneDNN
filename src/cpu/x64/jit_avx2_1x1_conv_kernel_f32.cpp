/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#include <assert.h>
#include <limits>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_avx2_1x1_conv_kernel_f32.hpp"
#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;

using namespace Xbyak;

jit_avx2_1x1_conv_kernel_f32::jit_avx2_1x1_conv_kernel_f32(
        const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jit_generator(jit_name(), avx2), jcp(ajcp), attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr size_t helper_vmm_idx = 15;
        static constexpr bool use_exact_tail_scalar_bcast = false;
        const size_t tail_size = jcp.oc_without_padding % isa_simd_width_;

        rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx, r13, r14,
                r15, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(dst_orig),
                memory_desc_wrapper(dst_md), tail_size,
                use_exact_tail_scalar_bcast};
        static_params_t static_params {this->param1, rhs_arg_static_params};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx2>>(
                this, jcp.post_ops, static_params);
    }
}

void jit_avx2_1x1_conv_kernel_f32::generate_bcast_loop(int load_loop_blk) {
    mov(aux1_reg_bcast_data, ptr[rsp + reg_bcast_data_off]);
    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, reg_bcast_loop_work);

    Label bcast_loop, bcast_loop_tail, large_tail;

    cmp(bcast_loop_iter, jcp.bcast_block);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop);
    {
        assert(jcp.bcast_block % jcp.ur == 0);
        const int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            if (i == num_substeps - 1) L(large_tail);
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
            sub(bcast_loop_iter, jcp.ur);
        }
        cmp(bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        if (jcp.ur_tail >= jcp.ur) {
            cmp(bcast_loop_iter, jcp.ur);
            jge(large_tail, T_NEAR);
        }
        if (jcp.ur_tail % jcp.ur > 0) {
            cmp(bcast_loop_iter, 0);
            jle(bcast_loop_tail_out, T_NEAR);
            generate_reduce_loop(load_loop_blk, jcp.ur_tail % jcp.ur);
            L(bcast_loop_tail_out);
        }
    }
}

static int vreg_accum_idx(const int load_loop_blk, int i, int j) {
    return (j * load_loop_blk + i);
}

static Ymm vreg_accum(const int load_loop_blk, int i, int j) {
    return Ymm(vreg_accum_idx(load_loop_blk, i, j));
}

template <typename F>
void iterate(const int load_loop_blk, const int ur, const int load_dim_tail,
        const F &f) {
    for (int i = 0; i < load_loop_blk; ++i) {
        const bool mask_flag = (load_dim_tail > 0) && (i == load_loop_blk - 1);
        for (int j = 0; j < ur; ++j)
            f(mask_flag, i, j);
    }
}
template <typename F>
void iterate(const int load_loop_blk, const int ur, const F &f) {
    iterate(load_loop_blk, ur, 0, f);
}

void jit_avx2_1x1_conv_kernel_f32::apply_postops(
        const int load_loop_blk, const int ur, const int load_dim_tail) {
    if (jcp.with_eltwise || jcp.with_binary) {
        assert(ur * load_loop_blk < 14);

        Label store_nopost_ops;
        test(reg_reduce_pos_flag, FLAG_REDUCE_LAST);
        jz(store_nopost_ops, T_NEAR);

        injector_utils::vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params,
                    rhs_arg_params_tail;

            iterate(load_loop_blk, ur, load_dim_tail,
                    [&](const bool mask_flag, const int i, const int j) {
                        const size_t aux_output_offset
                                = (i * get_output_i_offset(jcp, true)
                                          + j * get_output_j_offset(jcp))
                                * sizeof(float);
                        const auto vmm_idx
                                = vreg_accum_idx(load_loop_blk, i, j);
                        vmm_idxs.emplace(vmm_idx);

                        rhs_arg_params_tail.vmm_idx_to_out_reg.emplace(
                                vmm_idx, aux_reg_output_data);
                        rhs_arg_params_tail.vmm_idx_to_out_elem_off_val.emplace(
                                vmm_idx, aux_output_offset);
                        if (mask_flag)
                            rhs_arg_params_tail.vmm_tail_idx_.emplace(vmm_idx);
                    });
            rhs_arg_params = rhs_arg_params_tail;
            rhs_arg_params.vmm_tail_idx_.clear();

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
                    ptr[rsp + reg_abi_param1_backup
                            + reg_guard_stack_occupied]);

            Label postops_done;
            if (load_dim_tail) {
                Label postops_no_tail;
                cmp(reg_load_loop_work,
                        load_loop_blk * jcp.load_loop_iter_step);
                jge(postops_no_tail, T_NEAR);
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params_tail);
                jmp(postops_done, T_NEAR);
                L(postops_no_tail);
            }
            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params);
            L(postops_done);
        } else {
            iterate(load_loop_blk, ur, load_dim_tail,
                    [&](const bool, const int i, const int j) {
                        vmm_idxs.emplace(vreg_accum_idx(load_loop_blk, i, j));
                    });
            postops_injector_->compute_vector_range(vmm_idxs);
        }
        L(store_nopost_ops);
    }
};

void jit_avx2_1x1_conv_kernel_f32::generate_reduce_loop(
        int load_loop_blk, int ur) {
    const int load_dim_tail
            = ((jcp.with_binary
                       && one_of(jcp.prop_kind, forward_training,
                               forward_inference))
                              ? jcp.oc_without_padding
                              : jcp.load_dim)
            % jcp.load_block;
    const int reduce_dim_tail = jcp.reduce_dim % jcp.reduce_block;

    auto vreg_load = [ur, load_loop_blk](
                             int i) { return Ymm(ur * load_loop_blk + i); };

    auto bias_ptr = [this](int i) {
        return ptr[reg_bias_data + sizeof(float) * jcp.oc_block * i];
    };

    auto bcast_ptr = [this](int u, int j) {
        assert(j < jcp.ur);
        assert(u <= jcp.reduce_loop_unroll);
        const size_t offset = get_bcast_offset(jcp, u, j);
        return make_safe_addr(aux_reg_bcast_data, offset, reg_long_offt);
    };

    auto get_load_offset_bwd_w = [this](int u, int i) {
        size_t u0 = u % jcp.reduce_loop_unroll;
        size_t u1 = u / jcp.reduce_loop_unroll;
        return u1 * jcp.reduce_loop_load_step
                + sizeof(float) * get_load_bwd_w_offset(jcp, i, u0);
    };

    auto load_ptr = [this](int u, int i) {
        size_t offt;
        size_t u0 = u % jcp.reduce_loop_unroll;
        size_t u1 = u / jcp.reduce_loop_unroll;
        switch (jcp.prop_kind) {
            case backward_data:
                offt = (i * jcp.oc_block + u0) * jcp.ic_block;
                break;
            case backward_weights:
                offt = get_load_bwd_w_offset(jcp, i, u0);
                break;
            default:
                offt = (i * rnd_up(jcp.ic, jcp.ic_block) + u0) * jcp.oc_block;
        }
        return ptr[aux_reg_load_data + u1 * jcp.reduce_loop_load_step
                + sizeof(float) * offt];
    };

    auto get_output_offset = [this](int i, int j) {
        switch (jcp.prop_kind) {
            case backward_weights: return sizeof(float) * jcp.oc_block * j;
            default:
                return (i * get_output_i_offset(jcp)
                               + j * get_output_j_offset(jcp))
                        * sizeof(float);
        }
    };

    auto output_ptr = [this, get_output_offset](int i, int j) {
        switch (jcp.prop_kind) {
            case backward_weights:
                return ptr[aux_reg_output_data
                        + (i ? reg_output_stride * i
                             : 0) // TODO: Xbyak should allow 0 scale
                        + sizeof(float) * jcp.oc_block * j];
            default:
                const size_t off = get_output_offset(i, j);
                return make_safe_addr(aux_reg_output_data, off, reg_long_offt);
        }
    };

    auto init = [&]() {
        Label init_done, init_zero;

        if (jcp.with_bias
                && one_of(jcp.prop_kind, forward_training, forward_inference)) {
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jz(init_zero, T_NEAR);

            for (int i = 0; i < load_loop_blk; i++) {
                for (int j = 0; j < ur; ++j) {
                    if (load_dim_tail > 0 && i == load_loop_blk - 1) {
                        Label load_bias_tail, load_bias_done;
                        cmp(reg_load_loop_work,
                                load_loop_blk * jcp.load_loop_iter_step);
                        jl(load_bias_tail);
                        vmovups(vreg_accum(load_loop_blk, i, j), bias_ptr(i));
                        jmp(load_bias_done);

                        L(load_bias_tail);
                        load_bytes(vreg_accum(load_loop_blk, i, j),
                                reg_bias_data, i * jcp.oc_block * sizeof(float),
                                load_dim_tail * sizeof(float));
                        L(load_bias_done);
                    } else {
                        vmovups(vreg_accum(load_loop_blk, i, j), bias_ptr(i));
                    }
                }
            }
            jmp(init_done);
        }

        L(init_zero);
        for (int i = 0; i < load_loop_blk; ++i)
            for (int j = 0; j < ur; ++j) {
                auto r = vreg_accum(load_loop_blk, i, j);
                vxorps(r, r, r);
            }

        L(init_done);
        for (int i = 0; i < load_loop_blk; ++i) {
            if (jcp.prop_kind == backward_weights && load_dim_tail > 0
                    && i == load_loop_blk - 1) {
                Label load_init_tail, load_init_done;
                cmp(reg_load_loop_work,
                        load_loop_blk * jcp.load_loop_iter_step);
                jl(load_init_tail);
                vmovups(vreg_load(i), load_ptr(0, i));
                jmp(load_init_done);

                L(load_init_tail);
                load_bytes(vreg_load(i), aux_reg_load_data,
                        get_load_offset_bwd_w(0, i),
                        load_dim_tail * sizeof(float));
                L(load_init_done);
            } else {
                vmovups(vreg_load(i), load_ptr(0, i));
            }
        }
        vbroadcastss(vreg_bcast, bcast_ptr(0, 0));
    };

    auto store = [&]() {
        Label store_noadd;

        if (!jcp.with_sum) {
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jnz(store_noadd, T_NEAR);
        }

        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
                auto r = vreg_accum(load_loop_blk, i, j);
                if (jcp.with_sum && load_dim_tail > 0
                        && i == load_loop_blk - 1) {
                    Label sum_tail, sum_done;
                    cmp(reg_load_loop_work,
                            load_loop_blk * jcp.load_loop_iter_step);
                    jl(sum_tail);
                    vaddps(r, r, output_ptr(i, j));
                    jmp(sum_done);

                    L(sum_tail);
                    load_bytes(vtmp, aux_reg_output_data,
                            get_output_offset(i, j),
                            load_dim_tail * sizeof(float));
                    vaddps(r, r, vtmp);
                    L(sum_done);
                } else {
                    vaddps(r, r, output_ptr(i, j));
                }
            }

        L(store_noadd);

        apply_postops(load_loop_blk, ur, load_dim_tail);

        if (jcp.prop_kind == backward_weights && load_dim_tail > 0) {
            push(aux_reg_bcast_data);
        }

        const auto is_padding = jcp.oc_without_padding != jcp.oc;
        if (is_padding) uni_vxorps(vtmp, vtmp, vtmp);
        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
                if (load_dim_tail > 0 && i == load_loop_blk - 1) {
                    Label store_tail, store_done;
                    cmp(reg_load_loop_work,
                            load_loop_blk * jcp.load_loop_iter_step);
                    jl(store_tail);
                    vmovups(output_ptr(i, j), vreg_accum(load_loop_blk, i, j));
                    jmp(store_done);

                    L(store_tail);
                    if (jcp.prop_kind == backward_weights) {
                        if (i) {
                            xor_(reg_tmp, reg_tmp); // rdx
                            mov(reg_tmp_output_stride,
                                    reg_output_stride); // rax
                            mov(reg_output_stride_scale, i);
                            imul(reg_output_stride_scale);
                        } else {
                            xor_(reg_tmp_output_stride, reg_tmp_output_stride);
                        }
                        lea(reg_tmp,
                                ptr[aux_reg_output_data
                                        + reg_tmp_output_stride]);
                        vmovups(output_ptr(i, j),
                                vreg_accum(load_loop_blk, i, j));
                    } else {
                        if (is_padding && jcp.with_binary) {
                            vmovups(ptr[aux_reg_output_data
                                            + get_output_offset(i, j)],
                                    vtmp);
                        }
                        store_bytes(vreg_accum(load_loop_blk, i, j),
                                aux_reg_output_data, get_output_offset(i, j),
                                load_dim_tail * sizeof(float));
                    }
                    L(store_done);
                } else {
                    vmovups(output_ptr(i, j), vreg_accum(load_loop_blk, i, j));
                }
            }

        if (jcp.prop_kind == backward_weights && load_dim_tail > 0) {
            pop(aux_reg_bcast_data);
        }
    };

    auto fma_block = [&](bool last_block) {
        const bool is_tail = reduce_dim_tail && last_block;
        const int u_end = is_tail ? reduce_dim_tail : jcp.reduce_loop_unroll;
        for (int u = 0; u < u_end; ++u) {
            for (int j = 0; j < ur; ++j) {
                for (int i = 0; i < load_loop_blk; ++i) {
                    if (jcp.isa == avx2)
                        vfmadd231ps(vreg_accum(load_loop_blk, i, j),
                                vreg_load(i), vreg_bcast);
                    else { // Intel(R) Advanced Vector Extensions (Intel(R) AVX) support
                        vmulps(vtmp, vreg_bcast, vreg_load(i));
                        vaddps(vreg_accum(load_loop_blk, i, j),
                                vreg_accum(load_loop_blk, i, j), vtmp);
                    }
                    if (j == ur - 1 && !(last_block && u == u_end - 1)) {
                        if (jcp.prop_kind == backward_weights
                                && load_dim_tail > 0
                                && i == load_loop_blk - 1) {
                            Label fma_load_tail, fma_load_done;
                            cmp(reg_load_loop_work,
                                    load_loop_blk * jcp.load_loop_iter_step);
                            jl(fma_load_tail);
                            vmovups(vreg_load(i), load_ptr(u + 1, i));
                            jmp(fma_load_done);

                            L(fma_load_tail);
                            load_bytes(vreg_load(i), aux_reg_load_data,
                                    get_load_offset_bwd_w(u + 1, i),
                                    load_dim_tail * sizeof(float));
                            L(fma_load_done);
                        } else {
                            vmovups(vreg_load(i), load_ptr(u + 1, i));
                        }
                    }
                }
                if (j < ur - 1) vbroadcastss(vreg_bcast, bcast_ptr(u, j + 1));
            }
            if (!last_block || u < u_end - 1)
                vbroadcastss(vreg_bcast, bcast_ptr(u + 1, 0));
        }
    };

    Label reduce_loop, reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);
    mov(aux_reg_bcast_data, aux1_reg_bcast_data);

    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop);
    {
        fma_block(false);
        safe_add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step, reg_long_offt);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
}

void jit_avx2_1x1_conv_kernel_f32::generate_diff_bias_loop(int load_loop_blk) {
    if (!jcp.with_bias || jcp.prop_kind != backward_weights) return;

    Label diff_bias_loop, diff_bias_loop_out, diff_bias_init_out;
    Label diff_bias_load;

    auto diff_bias_ptr = [this](int i) {
        return ptr[reg_diff_bias_data + i * jcp.oc_block * sizeof(float)];
    };

    auto load_ptr = [this](int u, int i) {
        return ptr[aux_reg_load_data
                + (i * jcp.os + u) * jcp.oc_block * sizeof(float)];
    };

    auto diff_bias_reg = [](int i) { return Ymm(i); };

    mov(reg_diff_bias_data, ptr[rsp + reg_diff_bias_data_stack_offt]);
    cmp(reg_diff_bias_data, 0);
    je(diff_bias_loop_out, T_NEAR);

    test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
    jz(diff_bias_load, T_NEAR);

    for (int i = 0; i < load_loop_blk; ++i) {
        auto r = diff_bias_reg(i);
        vxorps(r, r, r);
    }
    jmp(diff_bias_init_out, T_NEAR);

    L(diff_bias_load);
    for (int i = 0; i < load_loop_blk; ++i)
        vmovups(diff_bias_reg(i), diff_bias_ptr(i));

    L(diff_bias_init_out);
    mov(aux_reg_load_data, reg_load_data);
    mov(reduce_loop_iter, reg_reduce_loop_work);
    L(diff_bias_loop);
    {
        for (int u = 0; u < jcp.reduce_loop_unroll; ++u)
            for (int i = 0; i < load_loop_blk; ++i)
                vaddps(diff_bias_reg(i), diff_bias_reg(i), load_ptr(u, i));
        assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jnz(diff_bias_loop, T_NEAR);
    }

    for (int i = 0; i < load_loop_blk; i++)
        vmovups(diff_bias_ptr(i), diff_bias_reg(i));
    add(reg_diff_bias_data, load_loop_blk * jcp.oc_block * sizeof(float));
    mov(ptr[rsp + reg_diff_bias_data_stack_offt], reg_diff_bias_data);

    L(diff_bias_loop_out);
}

void jit_avx2_1x1_conv_kernel_f32::generate() {
    preamble();

    sub(rsp, stack_space_needed);

    if (jcp.with_binary) {
        mov(ptr[rsp + reg_abi_param1_backup], abi_param1);
        if (jcp.with_dw_conv) {
            const auto zeroed_reg = r15;
            xor_(zeroed_reg, zeroed_reg);
            mov(ptr[rsp + reg_dw_binary_output_off], zeroed_reg);
        }
    }

    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(ptr[rsp + reg_bcast_data_off], reg_bcast_data);
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
        generate_bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        const size_t offst_with_dw_conv
                = get_load_loop_output_fwd_offset(jcp, load_loop_blk);
        const size_t offst_wo_dw_conv
                = get_load_loop_output_fwd_offset(jcp, load_loop_blk, true);
        switch (jcp.prop_kind) {
            case forward_training:
            case forward_inference:
                add(reg_bias_data,
                        load_loop_blk * jcp.oc_block * sizeof(float));
                safe_add(reg_output_data, offst_with_dw_conv, reg_long_offt);
                if (jcp.with_binary && jcp.with_dw_conv) {
                    mov(aux_reg_load_data, ptr[rsp + reg_dw_binary_output_off]);
                    add(aux_reg_load_data,
                            offst_wo_dw_conv - offst_with_dw_conv);
                    mov(ptr[rsp + reg_dw_binary_output_off], aux_reg_load_data);
                }
                break;
            case backward_data:
                safe_add(reg_output_data,
                        get_load_loop_output_bwd_d_offset(jcp, load_loop_blk),
                        reg_long_offt);
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
        jle(load_loop_blk_end, T_NEAR);
        generate_diff_bias_loop(1);
        generate_load_loop_body(1);
    }

    L(load_loop_blk_end);

    add(rsp, stack_space_needed);

    postamble();

    if (jcp.with_eltwise)
        postops_injector_->prepare_table(/* generate = */ true);
}

status_t jit_avx2_1x1_conv_kernel_f32::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr) {
    // disabling verbose dispatch messages for unsupported isa for better readability
    if (!mayiuse(avx)) return status::unimplemented;
    jcp.isa = mayiuse(avx2) ? avx2 : avx;

    // TODO (Roma): this code is duplicated from the generic kernel; maybe the
    // configuration struct could do some stuff below
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();

    jcp.nthr = dnnl_get_max_threads();

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc = jcp.oc_without_padding;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.ic = jcp.ic_without_padding;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.with_bias = pick_by_prop_kind(jcp.prop_kind, cd.bias_desc.format_kind,
                            format_kind::undef, cd.diff_bias_desc.format_kind)
            != format_kind::undef;

    jcp.os = static_cast<dim_t>(jcp.od) * jcp.oh * jcp.ow;
    jcp.is = static_cast<dim_t>(jcp.id) * jcp.ih * jcp.iw;

    jcp.typesize_in = sizeof(prec_traits<data_type::f32>::type);
    jcp.typesize_out = sizeof(prec_traits<data_type::f32>::type);

    const auto &post_ops = attr.post_ops_;
    const int dw_conv_ind = post_ops.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;

    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.
    const int sum_ind = post_ops.find(primitive_kind::sum, 0, dw_conv_ind);
    jcp.with_sum = sum_ind != -1;
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

    const auto dat_tag_nxc = utils::pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx8c = utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    jcp.src_tag = src_d.mb_stride_relaxed_match(dat_tag_nxc, dat_tag_nCx8c);
    jcp.dst_tag = dst_d.mb_stride_relaxed_match(dat_tag_nxc, dat_tag_nCx8c);
    const bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
    const auto dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;

    const int is_bwd_d = jcp.prop_kind == backward_data;
    format_tag_t wei_tag = with_groups
            ? utils::pick(2 * ndims - 6 + is_bwd_d, gOIw8i8o, gOIw8o8i,
                    gOIhw8i8o, gOIdhw8o8i, gOIhw8i8o, gOIdhw8o8i)
            : utils::pick(2 * ndims - 6 + is_bwd_d, OIw8i8o, OIw8o8i, OIhw8i8o,
                    OIhw8o8i, OIdhw8i8o, OIdhw8o8i);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);

    const int simd_w = 8;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    if (jcp.with_eltwise || jcp.with_binary)
        VDISPATCH_CONV_IC(jcp.isa >= avx2, VERBOSE_UNSUPPORTED_FEATURE,
                "eltwise and binary post-ops not implemented on isa");

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = true;
    static constexpr bool sum_requires_zp_zero = true;
    const bool post_ops_ok_ = post_ops_ok(post_ops_ok_args_t(jcp.isa,
            {eltwise, binary, sum}, jcp.post_ops, &dst_d, sum_at_pos_0_only,
            sum_requires_scale_one, sum_requires_zp_zero));
    VDISPATCH_CONV_IC(post_ops_ok_, VERBOSE_UNSUPPORTED_POSTOP);

    bool args_ok = true && jcp.ngroups == 1 && jcp.src_tag == dat_tag
            && jcp.wei_tag == wei_tag && jcp.dst_tag == dat_tag;

    VDISPATCH_CONV_IC(args_ok, VERBOSE_UNSUPPORTED_TAG);

    args_ok = true && jcp.id == jcp.od && jcp.ih == jcp.oh && jcp.iw == jcp.ow
            && IMPLICATION(!is_data_layout_nxc,
                    jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0)
            && jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0
            && jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1
            && jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1;
    VDISPATCH_CONV_IC(args_ok, VERBOSE_BAD_PARAM, "");

    // TODO: remove this restriction
    // optimized 1x1 bwd_w does not support Intel AVX
    VDISPATCH_CONV_IC(!(jcp.prop_kind == backward_weights && jcp.isa != avx2),
            VERBOSE_UNSUPPORTED_ISA);

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.ur = jcp.isa == avx2 ? 4 : 3; // Intel AVX support
    if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);

    int load_blocking {0};
    int load_blocking_max {0};
    int bcast_blocking {0};
    int bcast_blocking_max {0};
    int reduce_blocking {0};
    int reduce_blocking_max {0};

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

        jcp.load_loop_load_step
                = rnd_up(jcp.ic, jcp.ic_block) * jcp.oc_block * sizeof(float);
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
        jcp.load_block = jcp.ic_block;

        jcp.bcast_dim = jcp.os;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? 1 : jcp.os) * sizeof(float);
        jcp.reduce_loop_load_step = jcp.reduce_loop_unroll
                * rnd_up(jcp.ic, jcp.ic_block) * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur
                * (is_data_layout_nxc ? jcp.ic : jcp.ic_block) * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur
                * (is_data_layout_nxc ? jcp.oc : jcp.oc_block) * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.ic_block;

        load_blocking = is_data_layout_nxc
                ? jcp.load_dim
                : 96; // assumes the kernel is jcp.ur x 3
        load_blocking_max = is_data_layout_nxc ? jcp.load_dim : 144;

        bcast_blocking = 128; // affects load balancing across threads
        bcast_blocking_max = 196;
        reduce_blocking = is_data_layout_nxc ? jcp.reduce_dim
                                             : 64; // affects L1$ utilization
    } else if (jcp.prop_kind == backward_weights) {
        jcp.reduce_dim = jcp.os;
        jcp.reduce_block = 1;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? jcp.ic : jcp.ic_block) * sizeof(float);
        jcp.reduce_loop_load_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? jcp.oc : jcp.oc_block) * sizeof(float);

        jcp.bcast_loop_output_step
                = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = jcp.oc_block * jcp.ur * sizeof(float);
        jcp.bcast_loop_bcast_step = jcp.ic_block
                * (is_data_layout_nxc ? 1 : jcp.is) * sizeof(float);
        jcp.bcast_loop_bcast_substep = jcp.ur * sizeof(float);

        jcp.load_loop_load_step = jcp.oc_block
                * (is_data_layout_nxc ? 1 : jcp.os) * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        const bool no_load_tail = jcp.load_dim % jcp.load_block == 0;
        const bool modify_load_blocking
                = IMPLICATION(is_data_layout_nxc, no_load_tail);
        while (modify_load_blocking) {
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
        assert(IMPLICATION(
                !is_data_layout_nxc, jcp.load_dim % load_blocking == 0));

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        const int bcast_blocking_lim = is_data_layout_nxc ? 17 : 9;
        const bool no_bcast_tail = jcp.bcast_dim % jcp.bcast_block == 0;
        const bool small_size_for_bcast
                = static_cast<dim_t>(jcp.id) * jcp.ih * jcp.iw <= 1024;

        // TODO Verify if the size limitation helps for blocked format as well
        const bool modify_bcast_blocking = IMPLICATION(
                is_data_layout_nxc, no_bcast_tail && small_size_for_bcast);

        while (modify_bcast_blocking) {
            if (bcast_blocking <= bcast_blocking_lim)
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
        assert(IMPLICATION(
                !is_data_layout_nxc, jcp.bcast_dim % bcast_blocking == 0));

        reduce_blocking = is_data_layout_nxc
                ? rnd_up(nstl::min(jcp.ow, 128), jcp.reduce_block)
                : 128; // affects L1$ utilization
        reduce_blocking_max = rnd_dn(reduce_blocking * 3 / 2, jcp.reduce_block);
    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);

    assert(jcp.bcast_block % jcp.ur == 0);
    jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.bcast_block;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = div_up(load_blocking, jcp.load_block);
    jcp.nb_load_blocking_max = div_up(load_blocking_max, jcp.load_block);
    jcp.nb_reduce_blocking = div_up(reduce_blocking, jcp.reduce_block);
    jcp.nb_reduce_blocking_max = div_up(reduce_blocking_max, jcp.reduce_block);

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    if (jcp.prop_kind == backward_weights) {
        const auto mb_with_nb_reduce
                = static_cast<dim_t>(jcp.mb) * jcp.nb_reduce;
        // prevent too large argument to cpu reducer
        VDISPATCH_CONV_IC(mb_with_nb_reduce <= std::numeric_limits<int>::max(),
                VERBOSE_BLOCKING_FAIL, "bad argument for cpu reducer");
    }

    return status::success;
}

void jit_avx2_1x1_conv_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp) {
    using namespace dnnl::impl::memory_tracking::names;

    if (jcp.with_bias && jcp.prop_kind != backward_data
            && (jcp.oc != jcp.oc_without_padding // blocked format
                    || (jcp.prop_kind == backward_weights // nxc format
                            && jcp.oc % jcp.oc_block != 0))) {
        const size_t nelems_padded_bias
                = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block);
        scratchpad.book<float>(key_conv_padded_bias, nelems_padded_bias);
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
