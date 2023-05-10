/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include <stdint.h>
#include "config.hpp"
#include "impl_type.hpp"
#include "util.hpp"
#include <runtime/data_type.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/dynamic_dispatch/op_dispatch_tables.hpp>
#include <runtime/dynamic_dispatch/utils.hpp>
#include <runtime/target_machine.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
static int check_and_set_matmul_core_impl(
        runtime::op_dispatch_tables_t *op_table,
        runtime::dispatch_key *data_fmt_st,
        runtime::dispatch_key *weight_fmt_st, runtime::dispatch_key *out_fmt_st,
        runtime::dynamic_tensor_t *data_dyn_tsr,
        runtime::dynamic_tensor_t *weight_dyn_tsr,
        runtime::dynamic_tensor_t *out_dyn_tsr, int M_blk, int N_blk,
        int K_blk) {
    // query impl kind here. default return normal impl kind.
    auto &impl_kind_table = op_table->impl_kind_table_;
    if (impl_kind_table) {
        uint64_t keys[3] = {static_cast<uint64_t>(M_blk),
                static_cast<uint64_t>(N_blk), static_cast<uint64_t>(K_blk)};
        void *value = impl_kind_table->get(keys, 3);
        assert(value);
        int impl = *reinterpret_cast<int *>(value);
        data_fmt_st->set_impl_alg(impl);
        weight_fmt_st->set_impl_alg(impl);
        out_fmt_st->set_impl_alg(impl);
        return impl;
    }

    return impl_kind_t::normal;
}

static int check_and_set_managed_matmul_core_impl(
        runtime::op_dispatch_tables_t *op_table,
        runtime::dispatch_key *data_fmt_st,
        runtime::dispatch_key *weight_fmt_st, runtime::dispatch_key *out_fmt_st,
        runtime::dynamic_tensor_t *data_dyn_tsr,
        runtime::dynamic_tensor_t *weight_dyn_tsr,
        runtime::dynamic_tensor_t *out_dyn_tsr, int M_blk, int N_blk,
        int K_blk) {
    // query impl kind here.
    auto &impl_kind_table = op_table->impl_kind_table_;
    assert(impl_kind_table);
    int M_split_num, N_split_num, M_sub_block, N_sub_block, K_sub_block,
            im_loop_order;
    bool is_int8 = utils::is_one_of(sc_data_etype(data_dyn_tsr->dtype_),
            sc_data_etype::U8, sc_data_etype::S8);
    bool is_f32 = sc_data_etype(data_dyn_tsr->dtype_) == sc_data_etype::F32;
    const int M = utils::divide_and_ceil(data_dyn_tsr->dims_[0], M_blk) * M_blk;
    const int N
            = utils::divide_and_ceil(weight_dyn_tsr->dims_[1], N_blk) * N_blk;
    const int K = utils::divide_and_ceil(data_dyn_tsr->dims_[1], K_blk) * K_blk;
    const int sizeofdtypeA
            = utils::get_sizeof_etype(sc_data_etype(data_dyn_tsr->dtype_));
    const int sizeofdtypeC
            = utils::get_sizeof_etype(sc_data_etype(out_dyn_tsr->dtype_));
    get_managed_matmul_config(runtime::get_runtime_target_machine(),
            M_split_num, N_split_num, M_sub_block, N_sub_block, K_sub_block,
            im_loop_order, M, N, K, M_blk, N_blk, K_blk, sizeofdtypeA,
            sizeofdtypeC, is_int8, is_f32,
            /*is_dynamic*/ true);
    uint64_t keys[6] = {static_cast<uint64_t>(M_split_num),
            static_cast<uint64_t>(N_split_num),
            static_cast<uint64_t>(M_sub_block),
            static_cast<uint64_t>(N_sub_block),
            static_cast<uint64_t>(K_sub_block),
            static_cast<uint64_t>(im_loop_order)};
    void *value = impl_kind_table->get(keys, 6);
    assert(value);
    int impl = *reinterpret_cast<int *>(value);
    data_fmt_st->set_impl_alg(impl);
    weight_fmt_st->set_impl_alg(impl);
    out_fmt_st->set_impl_alg(impl);
    return impl;
}

extern "C" void infer_shape_matmul_op(void *out, void *data, void *weight) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *data_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(data);
    runtime::dynamic_tensor_t *weight_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(weight);
    int data_ndims = data_dyn_tsr->ndims_;
    int weight_ndims = weight_dyn_tsr->ndims_;
    int &out_ndims = out_dyn_tsr->ndims_;
    // Currently not support  2D x ND
    assert(data_ndims >= weight_ndims);
    out_dyn_tsr->ndims_ = data_ndims;

    // batch dims
    int64_t *batch_dims = data_dyn_tsr->dims_;
    for (int i = 0; i < out_ndims - 2; i++) {
        out_dyn_tsr->dims_[i] = batch_dims[i];
    }
    out_dyn_tsr->dims_[out_ndims - 2] = data_dyn_tsr->dims_[data_ndims - 2];
    out_dyn_tsr->dims_[out_ndims - 1] = weight_dyn_tsr->dims_[weight_ndims - 1];
}
typedef int (*impl_set_func)(runtime::op_dispatch_tables_t *,
        runtime::dispatch_key *, runtime::dispatch_key *,
        runtime::dispatch_key *, runtime::dynamic_tensor_t *,
        runtime::dynamic_tensor_t *, runtime::dynamic_tensor_t *, int, int,
        int);
extern "C" void query_format_matmul_common_process(void *table, void *out,
        void *data, void *weight, void *ori_data, void *ori_weight,
        uint64_t *out_fmt, uint64_t *data_fmt, uint64_t *weight_fmt,
        uint64_t *ori_data_fmt, uint64_t *ori_weight_fmt, uint64_t *out_size,
        void *kernel, int *impl_alg, impl_set_func impl_func) {
    // update output shape and mask.
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *data_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(data);
    runtime::dynamic_tensor_t *weight_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(weight);
    runtime::dynamic_tensor_t *ori_data_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(ori_data);
    runtime::dynamic_tensor_t *ori_weight_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(ori_weight);
    runtime::deep_copy_dynamic_tensor(data_dyn_tsr, ori_data_dyn_tsr);
    runtime::deep_copy_dynamic_tensor(weight_dyn_tsr, ori_weight_dyn_tsr);

    int data_ndims = data_dyn_tsr->ndims_;
    int weight_ndims = weight_dyn_tsr->ndims_;
    int &out_ndims = out_dyn_tsr->ndims_;

    int64_t M = data_dyn_tsr->dims_[data_ndims - 2];
    int64_t K = data_dyn_tsr->dims_[data_ndims - 1];
    int64_t K1 = weight_dyn_tsr->dims_[weight_ndims - 2];
    int64_t N = weight_dyn_tsr->dims_[weight_ndims - 1];
    assert(K == K1);
    // infer shape
    infer_shape_matmul_op(out, data, weight);
    // update dyn_mask
    out_dyn_tsr->dyn_mask_
            = data_dyn_tsr->dyn_mask_ | weight_dyn_tsr->dyn_mask_;
    // M mask
    uint8_t M_mask = (data_dyn_tsr->dyn_mask_ & (1 << (data_ndims - 2)))
            | ~(1 << (data_ndims - 2));
    out_dyn_tsr->dyn_mask_ &= M_mask;
    // N mask
    uint8_t N_mask = (weight_dyn_tsr->dyn_mask_ & (1 << (weight_ndims - 1)))
            | ~(1 << (weight_ndims - 1));
    out_dyn_tsr->dyn_mask_ &= N_mask;

    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);

    // query format
    bool is_M_dynamic = data_dyn_tsr->dyn_mask_ & (1 << (data_ndims - 2));
    bool is_N_dynamic = weight_dyn_tsr->dyn_mask_ & (1 << (weight_ndims - 1));
    bool is_K_dynamic = data_dyn_tsr->dyn_mask_ & (1 << (data_ndims - 1));
    assert(is_K_dynamic
            == (bool)(weight_dyn_tsr->dyn_mask_ & (1 << (weight_ndims - 2))));
    auto cp_data_fmt = *ori_data_fmt;
    auto cp_weight_fmt = *ori_weight_fmt;
    auto data_fmt_st = reinterpret_cast<runtime::dispatch_key *>(&cp_data_fmt);
    auto weight_fmt_st
            = reinterpret_cast<runtime::dispatch_key *>(&cp_weight_fmt);
    int a = weight_fmt_st->get(0);

    int M_blk, N_blk, K_blk;
    M_blk = get_matmul_dyn_cfg_single(M, true);
    K_blk = get_matmul_dyn_cfg_single(K);
    N_blk = get_matmul_dyn_cfg_single(N);

    if (data_fmt_st->is_plain() || data_fmt_st->ndims() == data_ndims) {
        data_fmt_st->set_block1(M_blk);
        data_fmt_st->set_block2(K_blk);
        if (M % M_blk || K % K_blk || !data_fmt_st->is_plain()) {
            if (!data_fmt_st->is_plain()) {
                for (int i = 0; i < data_ndims; i++) {
                    data_fmt_st->set(i, i);
                }
            }
            data_fmt_st->set(data_ndims,
                    data_fmt_st->get(data_ndims - 2)); // M block
            data_fmt_st->set(data_ndims + 1,
                    data_fmt_st->get(data_ndims - 1)); // K block
            data_fmt_st->is_plain_ = 0;
        }
    } else {
        assert(data_fmt_st->ndims() == data_ndims + 2);
        // reuse last blocking.
        K_blk = data_fmt_st->get_block2();
    }
    bool is_vnni = weight_dyn_tsr->dtype_ == uint32_t(sc_data_etype::U8)
            || weight_dyn_tsr->dtype_ == uint32_t(sc_data_etype::S8)
            || weight_dyn_tsr->dtype_ == uint32_t(sc_data_etype::BF16);
    if (weight_fmt_st->is_plain() || weight_fmt_st->ndims() == weight_ndims) {
        weight_fmt_st->set_block1(K_blk);
        weight_fmt_st->set_block2(N_blk);
        if (N % N_blk || K % K_blk || is_vnni || !weight_fmt_st->is_plain()) {
            if (!weight_fmt_st->is_plain()) {
                for (int i = 0; i < weight_ndims; i++) {
                    weight_fmt_st->set(i, i);
                }
            }
            int K_axis = weight_fmt_st->get(weight_ndims - 2);
            int N_axis = weight_fmt_st->get(weight_ndims - 1);
            weight_fmt_st->set(weight_ndims - 2, N_axis);
            weight_fmt_st->set(weight_ndims - 1, K_axis);
            weight_fmt_st->set(weight_ndims, K_axis);
            weight_fmt_st->set(weight_ndims + 1, N_axis);
            if (is_vnni) { weight_fmt_st->set(weight_ndims + 2, K_axis); }
            weight_fmt_st->is_plain_ = 0;
        }
    } else {
        assert((!is_vnni && weight_fmt_st->ndims() == weight_ndims + 2)
                || (is_vnni && weight_fmt_st->ndims() == weight_ndims + 3));
        // reuse last blocking.
    }
    auto &format_table = op_table->format_table_;
    if (format_table) {
        uint64_t fmt_keys[2] = {cp_data_fmt, cp_weight_fmt};
        void *value = format_table->get(fmt_keys, 2);
        assert(value);
        *out_fmt = reinterpret_cast<uint64_t *>(value)[0];
    }
    // query kernel, need determine the impl alg first.
    uint64_t cp_out_fmt = *out_fmt;
    auto *out_fmt_st = reinterpret_cast<runtime::dispatch_key *>(&cp_out_fmt);
    auto &kernel_table = op_table->kernel_table_;
    if (kernel_table) {
        impl_func(op_table, data_fmt_st, weight_fmt_st, out_fmt_st,
                data_dyn_tsr, weight_dyn_tsr, out_dyn_tsr, M_blk, N_blk, K_blk);
        uint64_t keys[3] = {cp_data_fmt, cp_weight_fmt, cp_out_fmt};
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(), keys, 3);
        assert(func);
        data_fmt_st->reset_blocks_and_impl();
        weight_fmt_st->reset_blocks_and_impl();
        *reinterpret_cast<void **>(kernel) = func;
    } else {
        assert(impl_alg);
        *impl_alg = impl_func(op_table, data_fmt_st, weight_fmt_st, out_fmt_st,
                data_dyn_tsr, weight_dyn_tsr, out_dyn_tsr, M_blk, N_blk, K_blk);
    }
    // avoid internal status change in multi thread case.
    *data_fmt = cp_data_fmt;
    *weight_fmt = cp_weight_fmt;

    // query inplace
    *out_size = calculate_blocking_dims(out_dyn_tsr, out_fmt);
}

extern "C" void query_format_matmul_core_op(void *table, void *out, void *data,
        void *weight, void *ori_data, void *ori_weight, uint64_t *out_fmt,
        uint64_t *data_fmt, uint64_t *weight_fmt, uint64_t *ori_data_fmt,
        uint64_t *ori_weight_fmt, uint64_t *out_size, void *kernel,
        int *impl_alg) {
    query_format_matmul_common_process(table, out, data, weight, ori_data,
            ori_weight, out_fmt, data_fmt, weight_fmt, ori_data_fmt,
            ori_weight_fmt, out_size, kernel, impl_alg,
            check_and_set_matmul_core_impl);
}

extern "C" void query_format_managed_matmul_core_op(void *table, void *out,
        void *data, void *weight, void *ori_data, void *ori_weight,
        uint64_t *out_fmt, uint64_t *data_fmt, uint64_t *weight_fmt,
        uint64_t *ori_data_fmt, uint64_t *ori_weight_fmt, uint64_t *out_size,
        void *kernel, int *impl_alg) {
    query_format_matmul_common_process(table, out, data, weight, ori_data,
            ori_weight, out_fmt, data_fmt, weight_fmt, ori_data_fmt,
            ori_weight_fmt, out_size, kernel, impl_alg,
            check_and_set_managed_matmul_core_impl);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
