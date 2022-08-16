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
#include <stdint.h>
#include "impl_type.hpp"
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/dynamic_dispatch/op_dispatch_tables.hpp>
#include <runtime/dynamic_dispatch/utils.hpp>
#include <runtime/target_machine.hpp>
namespace sc {
static void check_and_set_matmul_impl(runtime::dynamic_tensor_t *data_dyn_tsr,
        runtime::dynamic_tensor_t *weight_dyn_tsr,
        runtime::dispatch_key *data_fmt_st,
        runtime::dispatch_key *weight_fmt_st,
        runtime::dispatch_key *out_fmt_st) {
    // Currently we only check padding or not
    int impl_alg = impl_etype_t::no_padding;
    auto simd_length = std::min(UINT64_C(16),
            static_cast<uint64_t>(
                    runtime::get_runtime_target_machine()
                            .cpu_flags_.get_max_vector_lanes(
                                    sc_data_etype(data_dyn_tsr->dtype_))));
    for (int i = 0; i < data_dyn_tsr->ndims_; i++) {
        if (data_dyn_tsr->dims_[i] % simd_length) {
            impl_alg = impl_etype_t::normal;
            break;
        }
    }
    if (impl_alg == impl_etype_t::no_padding) {
        for (int i = 0; i < weight_dyn_tsr->ndims_; i++) {
            if (weight_dyn_tsr->dims_[i] % simd_length) {
                impl_alg = impl_etype_t::normal;
                break;
            }
        }
    }
    data_fmt_st->set_impl_alg(impl_alg);
    weight_fmt_st->set_impl_alg(impl_alg);
    out_fmt_st->set_impl_alg(impl_alg);
}
extern "C" void query_format_matmul_core_op(void *table, void *out, void *data,
        void *weight, void *ori_data, void *ori_weight, uint64_t *out_fmt,
        uint64_t *data_fmt, uint64_t *weight_fmt, uint64_t *ori_data_fmt,
        uint64_t *ori_weight_fmt, uint64_t *out_size, void *kernel) {
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
    // Currently not support  2D x ND
    assert(data_ndims == weight_ndims);
    out_dyn_tsr->ndims_ = data_ndims;
    int64_t M = data_dyn_tsr->dims_[data_ndims - 2];
    int64_t K = data_dyn_tsr->dims_[data_ndims - 1];
    int64_t K1 = weight_dyn_tsr->dims_[weight_ndims - 2];
    int64_t N = weight_dyn_tsr->dims_[weight_ndims - 1];
    assert(K == K1);
    // batch dims
    int64_t *batch_dims = data_dyn_tsr->dims_;
    for (int i = 0; i < out_ndims - 2; i++) {
        out_dyn_tsr->dims_[i] = batch_dims[i];
    }
    out_dyn_tsr->dims_[out_ndims - 2] = M;
    out_dyn_tsr->dims_[out_ndims - 1] = N;
    // mask
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
    assert(is_M_dynamic || is_N_dynamic || is_K_dynamic);
    *data_fmt = *ori_data_fmt;
    *weight_fmt = *ori_weight_fmt;
    auto data_fmt_st = reinterpret_cast<runtime::dispatch_key *>(data_fmt);
    auto weight_fmt_st = reinterpret_cast<runtime::dispatch_key *>(weight_fmt);
    int a = weight_fmt_st->get(0);
    int M_blk, N_blk;
    if (data_fmt_st->is_plain()) {
        int K_blk;
        M_blk = runtime::get_dyn_cfg_single(M, true);
        data_fmt_st->set_block1(M_blk);
        K_blk = runtime::get_dyn_cfg_single(K, true);
        data_fmt_st->set_block2(K_blk);
        if (M % M_blk || K % K_blk) {
            int ndims = data_fmt_st->ndims();
            data_fmt_st->set(ndims, data_fmt_st->get(ndims - 2));
            data_fmt_st->set(ndims + 1, data_fmt_st->get(ndims - 1));
            data_fmt_st->is_plain_ = 0;
        }
    } else {
        // reuse last blocking
    }
    if (weight_fmt_st->is_plain()) {
        int K_blk;
        K_blk = runtime::get_dyn_cfg_single(K, true);
        weight_fmt_st->set_block1(K_blk);
        N_blk = runtime::get_dyn_cfg_single(N, true);
        weight_fmt_st->set_block2(N_blk);
        if (N % N_blk || K % K_blk) {
            int ndims = data_fmt_st->ndims();
            int K_axis = weight_fmt_st->get(ndims - 2);
            int N_axis = weight_fmt_st->get(ndims - 1);
            weight_fmt_st->set(ndims - 2, N_axis);
            weight_fmt_st->set(ndims - 1, K_axis);
            weight_fmt_st->set(ndims, K_axis);
            weight_fmt_st->set(ndims + 1, N_axis);
            weight_fmt_st->is_plain_ = 0;
        }
    } else {
        // reuse last blocking
    }
    auto &format_table = op_table->format_table_;
    uint64_t fmt_keys[2] = {*data_fmt, *weight_fmt};
    void *value = format_table->get(fmt_keys, 2);
    assert(value);
    *out_fmt = reinterpret_cast<uint64_t *>(value)[0];
    // query kernel, need determine the impl alg first.
    auto *out_fmt_st = reinterpret_cast<runtime::dispatch_key *>(out_fmt);
    check_and_set_matmul_impl(data_dyn_tsr, weight_dyn_tsr, data_fmt_st,
            weight_fmt_st, out_fmt_st);
    auto &kernel_table = op_table->kernel_table_;
    uint64_t keys[3] = {*data_fmt, *weight_fmt, *out_fmt};

    void *func = op_table->kernel_dispatch_func_(kernel_table.get(), keys, 3);
    assert(func);
    data_fmt_st->reset_blocks_and_impl();
    weight_fmt_st->reset_blocks_and_impl();
    out_fmt_st->reset_blocks_and_impl();
    *reinterpret_cast<void **>(kernel) = func;
    // query inplace
    *out_size = calculate_blocking_dims(out_dyn_tsr, out_fmt);
}
} // namespace sc
