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
#include <compiler/config/context.hpp>
#include <runtime/config.hpp>
#include <runtime/data_type.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/dynamic_dispatch/op_dispatch_tables.hpp>
#include <runtime/dynamic_dispatch/ops/runtime_op_info.hpp>
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
        runtime::dynamic_tensor_t *out_dyn_tsr, int M_blk, int N_blk, int K_blk,
        int &internal_impl) {
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
        runtime::dynamic_tensor_t *out_dyn_tsr, int M_blk, int N_blk, int K_blk,
        int &internal_impl) {
    // query impl kind here.
    auto &impl_kind_table = op_table->impl_kind_table_;
    assert(impl_kind_table);
    int M_split_num, N_split_num, M_sub_block, N_sub_block, K_sub_block,
            im_loop_order;
    bool is_int8 = utils::is_one_of(sc_data_etype(data_dyn_tsr->dtype_),
            sc_data_etype::U8, sc_data_etype::S8);
    bool is_f32 = sc_data_etype(data_dyn_tsr->dtype_) == sc_data_etype::F32;
    bool no_vnni_f16 = get_default_context()->machine_.cpu_flags_.fAVX512FP16
            && sc_data_etype(data_dyn_tsr->dtype_) == sc_data_etype::F16;
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
            sizeofdtypeC, is_int8, is_f32 || no_vnni_f16,
            /*is_dynamic*/ true);
    uint64_t keys[6] = {static_cast<uint64_t>(M_split_num),
            static_cast<uint64_t>(N_split_num),
            static_cast<uint64_t>(M_sub_block),
            static_cast<uint64_t>(N_sub_block),
            static_cast<uint64_t>(K_sub_block),
            static_cast<uint64_t>(im_loop_order)};
    void *value = impl_kind_table->get(keys, 6);
    assert(value);
    internal_impl = *reinterpret_cast<int *>(value);
    const int num_threads = runtime_config_t::get().get_num_threads();
    const int K_split_num = num_threads / M_split_num / N_split_num;
    if (K_split_num > 1) { return mmm_impl_kind_t::is_partial; }
    return mmm_impl_kind_t::full_k;
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
        runtime::dynamic_tensor_t *, runtime::dynamic_tensor_t *, int, int, int,
        int &);
extern "C" void query_format_matmul_common_process(void *table, void *out,
        void *data, void *weight, void *ori_data, void *ori_weight,
        uint64_t *out_fmt, uint64_t *data_fmt, uint64_t *weight_fmt,
        uint64_t *ori_data_fmt, uint64_t *ori_weight_fmt, uint64_t *out_size,
        void *kernel, int *impl_alg, impl_set_func impl_func,
        bool is_mmm = false) {
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
        // todo: find the better way for s8 with amx process.
        if ((!is_mmm && (M % M_blk || K % K_blk || !data_fmt_st->is_plain()))
                || (is_mmm
                        && data_dyn_tsr->dtype_ == uint32_t(sc_data_etype::S8)
                        && K % K_blk)) {
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
    int internal_impl;
    int impl = impl_func(op_table, data_fmt_st, weight_fmt_st, out_fmt_st,
            data_dyn_tsr, weight_dyn_tsr, out_dyn_tsr, M_blk, N_blk, K_blk,
            internal_impl);
    if (!impl_alg) {
        // single op query.
        uint64_t keys[3] = {cp_data_fmt, cp_weight_fmt, cp_out_fmt};
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(), keys, 3);
        assert(func);
        data_fmt_st->reset_blocks_and_impl();
        weight_fmt_st->reset_blocks_and_impl();
        *reinterpret_cast<void **>(kernel) = func;
    } else {
        *impl_alg = impl;
    }
    // impl func
    if (is_mmm) {
        assert(kernel_table);
        runtime::dispatch_key impl_fmt_st
                = runtime::get_impl_dispatch_key(internal_impl);
        uint64_t keys[3] = {impl_fmt_st, impl_fmt_st, impl_fmt_st};
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(), keys, 3);
        assert(func);
        *(reinterpret_cast<void **>(kernel)
                + static_cast<int>(impl_alg == nullptr))
                = func;
    }
    // avoid internal status change in multi thread case.
    *data_fmt = cp_data_fmt;
    *weight_fmt = cp_weight_fmt;

    // query inplace
    *out_size = calculate_blocking_dims(out_dyn_tsr, out_fmt);
}

static int check_and_set_conv_fwd_impl(runtime::op_dispatch_tables_t *op_table,
        runtime::dispatch_key *data_fmt_st,
        runtime::dispatch_key *weight_fmt_st, runtime::dispatch_key *out_fmt_st,
        int N, int P, int Q, int K, int C, int k_blk, bool is_bf16, bool dyn_bs,
        bool dyn_h, bool dyn_w, bool is_conv_1x1) {
    // query impl kind here. default return normal impl kind.
    int impl = impl_kind_t::normal;
    auto &impl_kind_table = op_table->impl_kind_table_;
    int num_threads = runtime_config_t::get().get_num_threads();
    int max_thr = 1;
    size_t num_threads_candidates = 4;
    std::vector<int> threads_candidates = utils::get_factors(num_threads);
    std::vector<int> h_threads_ = {};
    for (size_t i = 0; i < num_threads_candidates; ++i) {
        h_threads_.push_back(
                threads_candidates.size() > i && P > threads_candidates[i]
                        ? threads_candidates[i]
                        : max_thr);
        max_thr = std::max(max_thr, h_threads_[i]);
    }

    std::vector<int> oc_threads_ = {1, 4, 8};
    if (impl_kind_table) {
        int h_threads = 1;
        int oc_threads = 1;
        int im_w_blk = 64;
        int im_h_blk = 1;
        // large channel size, split oc first
        if (N < num_threads) {
            if (K >= 512) {
                oc_threads = *(std::find_if(oc_threads_.rbegin(),
                        oc_threads_.rend(), [&](int split) {
                            return split == 1
                                    || (K / k_blk % split == 0
                                            && num_threads % split == 0);
                        }));
            }
            num_threads /= oc_threads;

            if (N == 1) {
                h_threads = num_threads;
            } else {
                for (int i = h_threads_.size() - 1; i >= 0; --i) {
                    if (P > (2 ^ (i + 2))) {
                        h_threads = std::max(h_threads,
                                num_threads % h_threads_[3] == 0 ? h_threads_[3]
                                                                 : 1);
                    }
                }
            }
            num_threads = runtime_config_t::get().get_num_threads();
            if (num_threads < 5 && N == 1) {
                if (!is_conv_1x1) {
                    if ((K >= 256) && K % num_threads == 0
                            && K / num_threads % k_blk == 0) {
                        oc_threads = num_threads;
                        h_threads = 1;
                    } else {
                        h_threads = P / num_threads >= 2 ? num_threads : 1;
                        oc_threads = 1;
                    }
                } else {
                    if ((K >= 512) && K % num_threads == 0) {
                        oc_threads = num_threads;
                        h_threads = 1;
                    } else {
                        h_threads = P / num_threads >= 2 ? num_threads : 1;
                        oc_threads = 1;
                    }
                }
            }
        }

        im_w_blk = dyn_w ? 64 : std::min(Q, 64);
        if (P % h_threads == 0 && num_threads <= 4) {
            if (Q <= 16)
                im_h_blk = P / h_threads % 4 == 0 ? 4
                        : P / h_threads % 2 == 0  ? 2
                                                  : 1;
            else if (Q <= 32)
                im_h_blk = P / h_threads % 2 == 0 && P % h_threads == 0 ? 2 : 1;
        }

        uint64_t keys[4] = {static_cast<uint64_t>(h_threads),
                static_cast<uint64_t>(oc_threads),
                static_cast<uint64_t>(im_h_blk),
                static_cast<uint64_t>(im_w_blk)};
        void *value = impl_kind_table->get(keys, 4);
        assert(value);
        impl = *reinterpret_cast<int *>(value);
        data_fmt_st->set_impl_alg(impl);
        weight_fmt_st->set_impl_alg(impl);
        out_fmt_st->set_impl_alg(impl);
    }

    return impl;
}

extern "C" void infer_shape_conv_fwd_op(void *out, void *data, void *weight,
        dyn_conv_fwd_runtime_info_t &op_info) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *data_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(data);
    runtime::dynamic_tensor_t *weight_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(weight);
    int data_ndims = data_dyn_tsr->ndims_;
    int weight_ndims = weight_dyn_tsr->ndims_;
    int &out_ndims = out_dyn_tsr->ndims_;

    out_dyn_tsr->ndims_ = data_ndims;

    int64_t OC = weight_ndims == 4 ? weight_dyn_tsr->dims_[0]
                                   : weight_dyn_tsr->dims_[0]
                    * weight_dyn_tsr->dims_[weight_ndims - 1];

    int64_t *data_dims = data_dyn_tsr->dims_;
    out_dyn_tsr->dims_[0] = data_dims[0];

    int strides[3] = {op_info.stride_d, op_info.stride_h, op_info.stride_w};
    int pads_begin[3] = {
            op_info.pads_begin_d, op_info.pads_begin_h, op_info.pads_begin_w};
    int pads_end[3]
            = {op_info.pads_end_d, op_info.pads_end_h, op_info.pads_end_w};

    //  P = (H + padding_h * 2 - R) / stride_h + 1;
    //  Q = (W + padding_w * 2 - S) / stride_w + 1;
    int offset = data_ndims == 5 ? -2 : -1;
    for (int i = 2; i < out_ndims; i++) {
        out_dyn_tsr->dims_[i]
                = (data_dims[i] + pads_begin[i + offset] + pads_end[i + offset]
                          - weight_dyn_tsr->dims_[i])
                        / strides[i + offset]
                + 1;
    }
    out_dyn_tsr->dims_[1] = OC;
}

extern "C" void query_format_conv_fwd_core_op(void *table, void *out,
        void *data, void *weight, void *ori_data, void *ori_weight,
        uint64_t *out_fmt, uint64_t *data_fmt, uint64_t *weight_fmt,
        uint64_t *ori_data_fmt, uint64_t *ori_weight_fmt, uint64_t *out_size,
        void *kernel, int *impl_alg) {
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

    int64_t BS = data_dyn_tsr->dims_[0];
    int64_t IC = data_dyn_tsr->dims_[1];
    int64_t IH = data_dyn_tsr->dims_[data_ndims - 2];
    int64_t IW = data_dyn_tsr->dims_[data_ndims - 1];

    int64_t OC = weight_ndims == 4 ? weight_dyn_tsr->dims_[0]
                                   : weight_dyn_tsr->dims_[0]
                    * weight_dyn_tsr->dims_[weight_ndims - 1];
    int64_t IC1 = weight_ndims == 4 ? weight_dyn_tsr->dims_[1]
                                    : weight_dyn_tsr->dims_[1]
                    * weight_dyn_tsr->dims_[weight_ndims - 2];
    assert(IC == IC1);
    // infer shape
    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    dyn_conv_fwd_runtime_info_t info
            = *op_table->op_info_
                       .unchecked_get_as<dyn_conv_fwd_runtime_info_t>();
    infer_shape_conv_fwd_op(out, data, weight, info);

    int64_t OH = weight_ndims == 4 ? out_dyn_tsr->dims_[data_ndims - 2]
                                   : out_dyn_tsr->dims_[data_ndims - 3];
    int64_t OW = weight_ndims == 4 ? out_dyn_tsr->dims_[data_ndims - 1]
                                   : out_dyn_tsr->dims_[data_ndims - 2];
    // update dyn_mask
    out_dyn_tsr->dyn_mask_ = data_dyn_tsr->dyn_mask_;
    // query format
    bool is_BS_dynamic = data_dyn_tsr->dyn_mask_ & 1;
    bool is_IH_dynamic = data_dyn_tsr->dyn_mask_ & (1 << 2);
    bool is_IW_dynamic = data_dyn_tsr->dyn_mask_ & (1 << 3);
    auto cp_data_fmt = *ori_data_fmt;
    auto cp_weight_fmt = *ori_weight_fmt;
    auto data_fmt_st = reinterpret_cast<runtime::dispatch_key *>(&cp_data_fmt);
    auto weight_fmt_st
            = reinterpret_cast<runtime::dispatch_key *>(&cp_weight_fmt);
    int a = weight_fmt_st->get(0);

    // 4. according to the dynamic dim and dynamic var to get the dispatch
    // key (blocking size)
    bool is_conv_1x1
            = weight_dyn_tsr->dims_[2] == 1 && weight_dyn_tsr->dims_[3] == 1;
    bool has_pad = info.pads_begin_h > 0 || info.pads_begin_w > 0
            || info.pads_begin_d > 0;
    bool is_f32 = sc_data_etype(data_dyn_tsr->dtype_) == sc_data_etype::F32;
    bool no_vnni_f16 = get_default_context()->machine_.cpu_flags_.fAVX512FP16
            && sc_data_etype(data_dyn_tsr->dtype_) == sc_data_etype::F16;
    auto default_block = get_dyn_conv_default_block(is_conv_1x1,
            utils::get_sizeof_etype(sc_data_etype(data_dyn_tsr->dtype_)),
            has_pad, is_f32 || no_vnni_f16);
    int k_block = utils::get_blocks(OC, 1, default_block).back();
    int c_block = utils::get_blocks(IC, 1, default_block).back();
    auto &format_table = op_table->format_table_;
    if (data_fmt_st->is_plain()) {
        int n_aix = data_fmt_st->get(0);
        int c_axis = data_fmt_st->get(1);
        int h_axis = data_fmt_st->get(data_ndims - 2);
        int w_axis = data_fmt_st->get(data_ndims - 1);
        data_fmt_st->set(0, n_aix);
        data_fmt_st->set(data_ndims - 1, c_axis);
        data_fmt_st->set(data_ndims - 3, h_axis);
        data_fmt_st->set(data_ndims - 2, w_axis);
        if (data_ndims == 5) {
            int d_axis = data_fmt_st->get(data_ndims - 3);
            data_fmt_st->set(data_ndims - 4, d_axis);
        }
        data_fmt_st->is_plain_ = 0;
    }
    data_fmt_st->impl_alg_ = 0;
    if (weight_ndims == 4) {
        bool is_vnni = weight_dyn_tsr->dtype_ == uint32_t(sc_data_etype::U8)
                || weight_dyn_tsr->dtype_ == uint32_t(sc_data_etype::S8)
                || weight_dyn_tsr->dtype_ == uint32_t(sc_data_etype::BF16);
        if (weight_fmt_st->ndims() == weight_ndims) {
            weight_fmt_st->set_block1(c_block);
            weight_fmt_st->set_block2(k_block);
            if (!weight_fmt_st->is_plain()) {
                for (int i = 0; i < weight_ndims; i++) {
                    weight_fmt_st->set(i, i);
                }
            }
            int c_axis = weight_fmt_st->get(1);
            int k_axis = weight_fmt_st->get(0);
            weight_fmt_st->set(0, k_axis);
            weight_fmt_st->set(1, c_axis);
            weight_fmt_st->set(weight_ndims, c_axis);
            weight_fmt_st->set(weight_ndims + 1, k_axis);
            if (is_vnni) { weight_fmt_st->set(weight_ndims + 2, c_axis); }
            weight_fmt_st->is_plain_ = 0;
        } else {
            assert((!is_vnni && weight_fmt_st->ndims() == weight_ndims + 2)
                    || (is_vnni && weight_fmt_st->ndims() == weight_ndims + 3));
            // reuse last blocking.
        }
    }
    weight_fmt_st->impl_alg_ = 0;
    uint64_t fmt_keys[2] = {cp_data_fmt, cp_weight_fmt};
    void *value = format_table->get(fmt_keys, 2);
    assert(value);
    *out_fmt = reinterpret_cast<uint64_t *>(value)[0];
    // query kernel, need determine the impl alg first.
    uint64_t cp_out_fmt = *out_fmt;
    auto *out_fmt_st = reinterpret_cast<runtime::dispatch_key *>(&cp_out_fmt);
    auto &kernel_table = op_table->kernel_table_;
    bool is_bf16_weight
            = weight_dyn_tsr->dtype_ == uint32_t(sc_data_etype::BF16);
    if (kernel_table) {
        check_and_set_conv_fwd_impl(op_table, data_fmt_st, weight_fmt_st,
                out_fmt_st, BS, OH, OW, OC, IC, k_block, is_bf16_weight,
                is_BS_dynamic, is_IH_dynamic, is_IW_dynamic, is_conv_1x1);
        uint64_t keys[3] = {cp_data_fmt, cp_weight_fmt, cp_out_fmt};
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(), keys, 3);
        assert(func);
        data_fmt_st->reset_blocks_and_impl();
        weight_fmt_st->reset_blocks_and_impl();
        *reinterpret_cast<void **>(kernel) = func;
    } else {
        assert(impl_alg);
        *impl_alg = check_and_set_conv_fwd_impl(op_table, data_fmt_st,
                weight_fmt_st, out_fmt_st, BS, OH, OW, OC, IC, k_block,
                is_bf16_weight, is_BS_dynamic, is_IH_dynamic, is_IW_dynamic,
                is_conv_1x1);
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
            check_and_set_managed_matmul_core_impl, true);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
