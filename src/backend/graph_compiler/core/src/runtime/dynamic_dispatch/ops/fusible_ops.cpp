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
static void check_and_set_fusible_impl(runtime::dynamic_tensor_t *in0,
        runtime::dynamic_tensor_t *in1, runtime::dispatch_key *in0_fmt_st,
        runtime::dispatch_key *in1_fmt_st, runtime::dispatch_key *out_fmt_st) {
    int impl_alg = impl_kind_t::no_padding;
    int simd_length = std::min(UINT64_C(16),
            static_cast<uint64_t>(runtime::get_runtime_target_machine()
                                          .cpu_flags_.get_max_vector_lanes(
                                                  sc_data_etype(in0->dtype_))));
    int ndims = in0->ndims_;
    if (in0->dims_[ndims - 2] < in0_fmt_st->get_block1()
            || in1->dims_[ndims - 1] < in1_fmt_st->get_block2()) {
        impl_alg = impl_kind_t::normal;
    }
    for (int i = 0; i < in0->ndims_; i++) {
        if (!(in0->dims_[i] == 1 || in0->dims_[i] % simd_length == 0)) {
            impl_alg = impl_kind_t::normal;
            break;
        }
    }
    if (impl_alg == impl_kind_t::no_padding && in1) {
        for (int i = 0; i < in1->ndims_; i++) {
            if (!(in1->dims_[i] == 1 || in1->dims_[i] % simd_length == 0)) {
                impl_alg = impl_kind_t::normal;
                break;
            }
        }
    }
    in0_fmt_st->set_impl_alg(impl_alg);
    out_fmt_st->set_impl_alg(impl_alg);
    if (in1_fmt_st) { in1_fmt_st->set_impl_alg(impl_alg); }
}

extern "C" void query_format_unary_fusible_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel) {
    // update output shape and mask.
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);
    runtime::deep_copy_dynamic_tensor(out_dyn_tsr, in_dyn_tsr);

    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query format
    auto &format_table = op_table->format_table_;
    void *value = format_table->get(in_fmt, 1);
    assert(value);
    *out_fmt = reinterpret_cast<uint64_t *>(value)[1];
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    if (kernel_table) {
        uint64_t keys[2] = {*in_fmt, *out_fmt};
        void *func
                = op_table->kernel_dispatch_func_(kernel_table.get(), keys, 2);
        assert(func);
        *reinterpret_cast<void **>(kernel) = func;
    }
    // query inplace
    *out_size = calculate_blocking_dims(out_dyn_tsr, out_fmt);
}

// we have partern like a fused op connected before two reorders, when we query
// the first reorder, we query the fused op first, the shape of dyn tsr of
// second reorder's output is unknown.
extern "C" void query_format_binary_fusible_op(void *table, void *out,
        void *in0, void *in1, uint64_t *out_fmt, uint64_t *in0_fmt,
        uint64_t *in1_fmt, uint64_t *out_size, void *kernel) {
    // update output shape and mask.
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in0_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in0);
    runtime::dynamic_tensor_t *in1_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in1);
    bool dims_equal = in0_dyn_tsr->ndims_ == in1_dyn_tsr->ndims_;
    assert(dims_equal || in1_dyn_tsr->ndims_ == 1);
    out_dyn_tsr->ndims_ = in0_dyn_tsr->ndims_;
    out_dyn_tsr->dyn_mask_ = in0_dyn_tsr->dyn_mask_ | in1_dyn_tsr->dyn_mask_;
    for (int i = 0; i < in0_dyn_tsr->ndims_; i++) {
        if (dims_equal) {
            out_dyn_tsr->dims_[i]
                    = std::max(in0_dyn_tsr->dims_[i], in1_dyn_tsr->dims_[i]);
        } else {
            out_dyn_tsr->dims_[i] = in0_dyn_tsr->dims_[i];
        }
    }

    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query format
    auto &format_table = op_table->format_table_;
    assert(format_table);
    uint64_t fmt_keys[2] = {*in0_fmt, *in1_fmt};
    void *value = format_table->get(fmt_keys, 2);
    assert(value);
    *in0_fmt = reinterpret_cast<uint64_t *>(value)[0];
    *in1_fmt = reinterpret_cast<uint64_t *>(value)[1];
    *out_fmt = reinterpret_cast<uint64_t *>(value)[2];
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    if (kernel_table) {
        uint64_t keys[3] = {*in0_fmt, *in1_fmt, *out_fmt};
        void *func
                = op_table->kernel_dispatch_func_(kernel_table.get(), keys, 3);
        assert(func);
        *reinterpret_cast<void **>(kernel) = func;
    }
    // query inplace
    *out_size = calculate_blocking_dims(out_dyn_tsr, out_fmt);
}

// actually reorder op does not need to query format, we only query kernel here.
extern "C" void query_format_reorder_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel) {
    // update output shape and mask.
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);
    out_dyn_tsr->ndims_ = in_dyn_tsr->ndims_;
    out_dyn_tsr->dyn_mask_ = in_dyn_tsr->dyn_mask_;
    for (int i = 0; i < in_dyn_tsr->ndims_; i++) {
        out_dyn_tsr->dims_[i] = in_dyn_tsr->dims_[i];
    }

    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    // reset blocks for plain format
    if (kernel_table) {
        uint64_t cp_in_fmt = *in_fmt, cp_out_fmt = *out_fmt;
        auto *in_fmt_st = reinterpret_cast<runtime::dispatch_key *>(&cp_in_fmt);
        auto *out_fmt_st
                = reinterpret_cast<runtime::dispatch_key *>(&cp_out_fmt);
        check_and_set_fusible_impl(
                in_dyn_tsr, out_dyn_tsr, in_fmt_st, out_fmt_st, out_fmt_st);
        uint64_t keys[2] = {*in_fmt_st, *out_fmt_st};
        void *func
                = op_table->kernel_dispatch_func_(kernel_table.get(), keys, 2);
        assert(func);
        *reinterpret_cast<void **>(kernel) = func;
        in_fmt_st->reset_blocks_and_impl();
        out_fmt_st->reset_blocks_and_impl();
    }
    // query inplace
    if (*in_fmt == uint64_t(out_fmt)) {
        *out_size = 0;
    } else {
        *out_size = calculate_blocking_dims(out_dyn_tsr, out_fmt);
    }
}

extern "C" void query_format_reduce_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel) {
    // update output shape and mask.
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);

    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query format
    auto &format_table = op_table->format_table_;
    assert(format_table);
    void *value = format_table->get(in_fmt, 1);
    assert(value);
    *out_fmt = reinterpret_cast<uint64_t *>(value)[1];
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    // reset blocks for plain format
    runtime::dispatch_key tmp_fmt = *out_fmt;
    //     if (tmp_fmt.is_plain()) { tmp_fmt.reset_blocks(); }
    if (kernel_table) {
        uint64_t keys[2] = {*in_fmt, tmp_fmt};
        void *func
                = op_table->kernel_dispatch_func_(kernel_table.get(), keys, 2);
        assert(func);
        *reinterpret_cast<void **>(kernel) = func;
    }
    // query inplace
    *out_size = calculate_blocking_dims(out_dyn_tsr, out_fmt);
}

extern "C" void query_format_tensor_view_op(
        void *table, void *out, void *in, uint64_t *out_fmt, uint64_t *in_fmt) {
    // only query format for tensor view
    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query format
    auto &format_table = op_table->format_table_;
    assert(format_table);
    void *value = format_table->get(in_fmt, 1);
    assert(value);
    *out_fmt = reinterpret_cast<uint64_t *>(value)[1];
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    assert(!kernel_table);
}
} // namespace sc
