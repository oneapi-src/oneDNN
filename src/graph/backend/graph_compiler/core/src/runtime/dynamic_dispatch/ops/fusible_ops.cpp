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
#include "impl_type.hpp"
#include "util.hpp"
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/dynamic_dispatch/op_dispatch_tables.hpp>
#include <runtime/dynamic_dispatch/ops/runtime_op_info.hpp>
#include <runtime/dynamic_dispatch/utils.hpp>
#include <runtime/target_machine.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
static bool is_block_produce_padding(
        runtime::dynamic_tensor_t *in, runtime::dispatch_key *in_fmt_st) {
    if (!in_fmt_st->is_plain()) {
        int dim_count[runtime::dispatch_key::meta::MAX_DIMS] = {0};
        bool first_block = true;
        for (int i = 0; i < in_fmt_st->ndims(); i++) {
            int ori_dim = in_fmt_st->get(i);
            dim_count[ori_dim]++;
            if (dim_count[ori_dim] == 2) {
                if (first_block) {
                    if (in->dims_[ori_dim] % in_fmt_st->get_block1() != 0) {
                        return true;
                    }
                    first_block = false;
                } else {
                    if (in->dims_[ori_dim] % in_fmt_st->get_block2() != 0) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

static bool is_fast_transpose_padding(runtime::dynamic_tensor_t *in,
        runtime::dispatch_key *in_fmt_st, runtime::dispatch_key *out_fmt_st) {
    int ori_in_dim = in_fmt_st->get(in_fmt_st->ndims() - 1);
    int ori_out_dim = out_fmt_st->get(out_fmt_st->ndims() - 1);
    uint32_t etype = in->dtype_;
    if (ori_in_dim != ori_out_dim) {
        if (etype == uint32_t(sc_data_etype::F32)
                && (in->dims_[ori_in_dim] % 8 != 0
                        || in->dims_[ori_out_dim] % 8 != 0)) {
            return true;
        }
        if (etype == uint32_t(sc_data_etype::BF16)
                && (in->dims_[ori_in_dim] % 8 != 0
                        || in->dims_[ori_in_dim] % 32 != 0)) {
            return true;
        }
    }
    return false;
}

static int check_and_set_reorder_impl(runtime::dynamic_tensor_t *in,
        runtime::dispatch_key *in_fmt_st, runtime::dispatch_key *out_fmt_st) {
    int impl_alg = impl_kind_t::no_padding;
    int ndims = in->ndims_;
    if (is_block_produce_padding(in, in_fmt_st)
            || is_block_produce_padding(in, out_fmt_st)) {
        impl_alg = impl_kind_t::normal;
    }
    if (impl_alg == impl_kind_t::no_padding
            && is_fast_transpose_padding(in, in_fmt_st, out_fmt_st)) {
        impl_alg = impl_kind_t::normal;
    }
    in_fmt_st->set_impl_alg(impl_alg);
    out_fmt_st->set_impl_alg(impl_alg);
    return impl_alg;
}

extern "C" void infer_shape_unary_fusible_op(void *out, void *in) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);
    runtime::deep_copy_dynamic_tensor(out_dyn_tsr, in_dyn_tsr);
}

extern "C" void infer_shape_padding_fusible_op(
        void *out, void *in, dyn_padding_runtime_info_t &op_info) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);

    int data_ndims = in_dyn_tsr->ndims_;
    out_dyn_tsr->ndims_ = data_ndims;
    int64_t *data_dims = in_dyn_tsr->dims_;
    out_dyn_tsr->dims_[0] = data_dims[0];
    out_dyn_tsr->dims_[1] = data_dims[1];
    int pads_begin[3] = {
            op_info.pads_begin_d, op_info.pads_begin_h, op_info.pads_begin_w};
    int pads_end[3]
            = {op_info.pads_end_d, op_info.pads_end_h, op_info.pads_end_w};
    int offset = data_ndims == 5 ? -2 : -1;
    for (int i = 2; i < data_ndims; i++) {
        out_dyn_tsr->dims_[i]
                = data_dims[i] + pads_begin[i + offset] + pads_end[i + offset];
    }
}

extern "C" void query_format_unary_fusible_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel) {
    // infer shape
    infer_shape_unary_fusible_op(out, in);
    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query format
    auto &format_table = op_table->format_table_;
    if (format_table) {
        void *value = format_table->get(in_fmt, 1);
        assert(value);
        *out_fmt = reinterpret_cast<uint64_t *>(value)[1];
    }
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    if (kernel_table) {
        uint64_t keys[2] = {*in_fmt, *out_fmt};
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(), keys, 2);
        *reinterpret_cast<void **>(kernel) = func;
    }
    // query inplace
    *out_size = runtime::calculate_blocking_dims(out, out_fmt);
}

extern "C" void infer_shape_binary_fusible_op(void *out, void *in0, void *in1) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in0_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in0);
    runtime::dynamic_tensor_t *in1_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in1);
    bool dims_equal = in0_dyn_tsr->ndims_ == in1_dyn_tsr->ndims_;
    assert(dims_equal || in1_dyn_tsr->ndims_ == 1);
    out_dyn_tsr->ndims_ = in0_dyn_tsr->ndims_;
    for (int i = 0; i < in0_dyn_tsr->ndims_; i++) {
        if (dims_equal) {
            out_dyn_tsr->dims_[i]
                    = std::max(in0_dyn_tsr->dims_[i], in1_dyn_tsr->dims_[i]);
        } else {
            out_dyn_tsr->dims_[i] = in0_dyn_tsr->dims_[i];
        }
    }
}

// we have partern like a fused op connected before two reorders, when we query
// the first reorder, we query the fused op first, the shape of dyn tsr of
// second reorder's output is unknown.
extern "C" void query_format_binary_fusible_op(void *table, void *out,
        void *in0, void *in1, uint64_t *out_fmt, uint64_t *in0_fmt,
        uint64_t *in1_fmt, uint64_t *out_size, void *kernel) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in0_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in0);
    runtime::dynamic_tensor_t *in1_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in1);
    // infer shape
    infer_shape_binary_fusible_op(out, in0, in1);
    // update dyn_mask
    out_dyn_tsr->dyn_mask_ = in0_dyn_tsr->dyn_mask_ | in1_dyn_tsr->dyn_mask_;

    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query format
    auto &format_table = op_table->format_table_;
    if (format_table) {
        uint64_t fmt_keys[2] = {*in0_fmt, *in1_fmt};
        void *value = format_table->get(fmt_keys, 2);
        assert(value);
        *in0_fmt = reinterpret_cast<uint64_t *>(value)[0];
        *in1_fmt = reinterpret_cast<uint64_t *>(value)[1];
        *out_fmt = reinterpret_cast<uint64_t *>(value)[2];
    }
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    if (kernel_table) {
        uint64_t keys[3] = {*in0_fmt, *in1_fmt, *out_fmt};
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(), keys, 3);
        assert(func);
        *reinterpret_cast<void **>(kernel) = func;
    }
    // query inplace
    *out_size = runtime::calculate_blocking_dims(out, out_fmt);
}

// actually reorder op does not need to query format, we only query kernel here.
extern "C" void query_format_reorder_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel,
        int *impl_alg) {
    // infer shape
    infer_shape_unary_fusible_op(out, in);
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);
    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    // reset blocks for plain format
    uint64_t cp_in_fmt = *in_fmt, cp_out_fmt = *out_fmt;
    auto *in_fmt_st = reinterpret_cast<runtime::dispatch_key *>(&cp_in_fmt);
    auto *out_fmt_st = reinterpret_cast<runtime::dispatch_key *>(&cp_out_fmt);
    // reset before for some plain in/out formats.
    in_fmt_st->reset_blocks_and_impl();
    out_fmt_st->reset_blocks_and_impl();
    auto tmp_impl_alg
            = check_and_set_reorder_impl(in_dyn_tsr, in_fmt_st, out_fmt_st);
    if (impl_alg) { *impl_alg = tmp_impl_alg; }
    if (kernel_table) {
        uint64_t keys[2] = {*in_fmt_st, *out_fmt_st};
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(), keys, 2);
        assert(func);
        *reinterpret_cast<void **>(kernel) = func;
    }
    in_fmt_st->reset_blocks_and_impl();
    out_fmt_st->reset_blocks_and_impl();
    // query inplace
    if (*in_fmt == uint64_t(out_fmt)) {
        *out_size = 0;
    } else {
        *out_size = runtime::calculate_blocking_dims(out, out_fmt);
    }
}

extern "C" void query_format_padding_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel,
        int *impl_alg) {
    // infer shape
    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);

    dyn_padding_runtime_info_t info
            = *op_table->op_info_
                       .unchecked_get_as<dyn_padding_runtime_info_t>();
    infer_shape_padding_fusible_op(out, in, info);
    // query format
    auto &format_table = op_table->format_table_;
    if (format_table) {
        void *value = format_table->get(in_fmt, 1);
        assert(value);
        *out_fmt = reinterpret_cast<uint64_t *>(value)[1];
    }
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
    *out_size = runtime::calculate_blocking_dims(out, out_fmt);
}

extern "C" void infer_shape_reduce_op(
        void *out, void *in, int *rd_axis, int num_axis) {
    // todo: currently we only support keep dimension reduce.
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);
    assert(num_axis > 0);
    assert(in_dyn_tsr->ndims_ >= num_axis);
    out_dyn_tsr->ndims_ = in_dyn_tsr->ndims_;
    for (int i = 0; i < in_dyn_tsr->ndims_; i++) {
        out_dyn_tsr->dims_[i] = in_dyn_tsr->dims_[i];
    }
    for (int i = 0; i < num_axis; i++) {
        out_dyn_tsr->dims_[rd_axis[i]] = 1;
    }
}

extern "C" void query_format_reduce_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel) {
    // check the output shape should be infered before query.
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    for (int i = 0; i < out_dyn_tsr->ndims_; i++) {
        assert(out_dyn_tsr->dims_[i] > 0);
    }
    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query format
    auto &format_table = op_table->format_table_;
    if (format_table) {
        void *value = format_table->get(in_fmt, 1);
        assert(value);
        *out_fmt = reinterpret_cast<uint64_t *>(value)[1];
    }
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    // reset blocks for plain format
    runtime::dispatch_key tmp_fmt = *out_fmt;
    //     if (tmp_fmt.is_plain()) { tmp_fmt.reset_blocks(); }
    if (kernel_table) {
        uint64_t keys[2] = {*in_fmt, tmp_fmt};
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(), keys, 2);
        assert(func);
        *reinterpret_cast<void **>(kernel) = func;
    }
    // query inplace
    *out_size = runtime::calculate_blocking_dims(out_dyn_tsr, out_fmt);
}

extern "C" void query_format_tensor_view_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel) {
    // only query format for tensor view
    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query format
    auto &format_table = op_table->format_table_;
    if (format_table) {
        void *value = format_table->get(in_fmt, 1);
        assert(value);
        *out_fmt = reinterpret_cast<uint64_t *>(value)[1];
    }
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    assert(!kernel_table);
    // query inplace
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    *out_size = runtime::calculate_blocking_dims(out_dyn_tsr, out_fmt);
}

extern "C" void query_format_select_op(void *table, void *out, void *in0,
        void *in1, void *in2, uint64_t *out_fmt, uint64_t *in0_fmt,
        uint64_t *in1_fmt, uint64_t *in2_fmt, uint64_t *out_size,
        void *kernel) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in1_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in1);
    runtime::dynamic_tensor_t *in2_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in2);
    // update dyn_mask
    out_dyn_tsr->dyn_mask_ = in1_dyn_tsr->dyn_mask_ | in2_dyn_tsr->dyn_mask_;

    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    // query format
    auto &format_table = op_table->format_table_;
    if (format_table) {
        uint64_t fmt_keys[3] = {0, 0, *in2_fmt};
        void *value = format_table->get(fmt_keys, 3);
        assert(value);
        *in0_fmt = reinterpret_cast<uint64_t *>(value)[0];
        *in1_fmt = reinterpret_cast<uint64_t *>(value)[1];
        *out_fmt = reinterpret_cast<uint64_t *>(value)[3];
    }
    // query kernel
    auto &kernel_table = op_table->kernel_table_;
    if (kernel_table) {
        uint64_t keys[4] = {*in0_fmt, *in1_fmt, *in2_fmt, *out_fmt};
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(), keys, 4);
        assert(func);
        *reinterpret_cast<void **>(kernel) = func;
    }
    // query inplace
    *out_size = runtime::calculate_blocking_dims(out, out_fmt);
}

extern "C" void infer_shape_transpose_op(
        void *out, void *in, int *tr_axis, int num_axis) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);
    assert(in_dyn_tsr->ndims_ == num_axis || num_axis == 0);
    out_dyn_tsr->ndims_ = in_dyn_tsr->ndims_;
    if (num_axis == 0) {
        for (int i = 0; i < in_dyn_tsr->ndims_; i++) {
            out_dyn_tsr->dims_[i] = in_dyn_tsr->dims_[i];
        }
    } else {
        for (int i = 0; i < num_axis; i++) {
            out_dyn_tsr->dims_[i] = in_dyn_tsr->dims_[tr_axis[i]];
        }
    }
}

extern "C" void infer_shape_tensor_view_op(void *out, void *in,
        int64_t *old_axis, int num_old_axis, int64_t *new_axis,
        int num_new_axis) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);
    assert(num_old_axis > 0 && num_new_axis > 0);
    out_dyn_tsr->ndims_ = num_new_axis;
    int old_idx = 0, new_idx = 0;
    for (; new_idx < num_new_axis; new_idx++) {
        if (new_axis[new_idx] < 0) {
            while (old_idx < num_old_axis && old_axis[old_idx] > 0) {
                old_idx++;
            }
            assert(old_idx < num_old_axis);
            new_axis[new_idx] = in_dyn_tsr->dims_[old_idx];
            old_idx++;
        }
        out_dyn_tsr->dims_[new_idx] = new_axis[new_idx];
    }
}

extern "C" void infer_shape_select_op(
        void *out, void *in0, void *in1, void *in2) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in0_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in0);
    runtime::dynamic_tensor_t *in1_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in1);
    runtime::dynamic_tensor_t *in2_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in2);
    assert(in1_dyn_tsr->ndims_ == 1 && in1_dyn_tsr->dims_[0] == 1);
    int lhs_ndims = in0_dyn_tsr->ndims_;
    int rhs_ndims = in2_dyn_tsr->ndims_;
    int max_ndims = std::max(lhs_ndims, rhs_ndims);
    int lhs_offset = max_ndims - lhs_ndims;
    int rhs_offset = max_ndims - rhs_ndims;
    out_dyn_tsr->ndims_ = max_ndims;
    for (int i = 0; i < max_ndims; i++) {
        int lhs = i >= lhs_offset ? in0_dyn_tsr->dims_[i - lhs_offset] : 1;
        int rhs = i >= rhs_offset ? in2_dyn_tsr->dims_[i - rhs_offset] : 1;
        out_dyn_tsr->dims_[i] = std::max(lhs, rhs);
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
