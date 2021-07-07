/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef BACKEND_DNNL_OPERATORS_CONCAT_HPP
#define BACKEND_DNNL_OPERATORS_CONCAT_HPP

#include <algorithm>
#include <vector>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/legacy.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

struct concat : public dnnl::concat, public kernel_base {
    using super = dnnl::concat;

private:
    primitive_desc pd_;
    dnnl::concat prim_;
    exec_args concat_args_;

    memory dst_opt_mem_;

    dnnl::engine engine_;
    dnnl::stream stream_;
    impl::allocator_t *alloc_ {nullptr};

    bool first_iteration_ {true};
    void *dst_opt_buf_ {nullptr};

    dnnl::reorder::primitive_desc dst_reorder_pd_;

    bool shapes_match(const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs,
            const int64_t axis, const int32_t rank) {
        using lt = impl::logical_tensor_t;

        const auto same_rank = std::all_of(inputs.cbegin(), inputs.cend(),
                [rank](const lt &in) { return rank == in.ndims; });
        if (!same_rank) return false;

        for (int32_t i = 0; i < rank; ++i) {
            if (axis == static_cast<int64_t>(i)) continue;
            const auto d = outputs.front().dims[i];
            const auto same_d = std::all_of(inputs.cbegin(), inputs.cend(),
                    [d, i](const lt &in) { return d == in.dims[i]; });
            if (!same_d) return false;
        }

        const auto out_axis_dim = outputs.front().dims[axis];
        const auto in_axis_dim_sum = std::accumulate(inputs.cbegin(),
                inputs.cend(), 0L, [axis](int64_t acc, const lt &in) {
                    return acc + in.dims[axis];
                });
        if (out_axis_dim != in_axis_dim_sum) return false;

        // all checks passed
        return true;
    }

    void free_buffer_if_allocated() {
        if (dst_opt_buf_) {
            allocator::free(dst_opt_buf_, engine_, alloc_);
            dst_opt_buf_ = nullptr;
        }
    }

public:
    ~concat() override { free_buffer_if_allocated(); }

    impl::status_t compile_impl(const op_t *op, const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using lt = impl::logical_tensor_t;

        engine_ = make_dnnl_engine(*g_engine);

        const size_t expected_outs = 1;
        if (inputs.empty() || expected_outs != outputs.size())
            return status::invalid_argument;

        const auto rank = outputs.front().ndims;
        const auto res = try_reverse_axis(op->get_attr<int64_t>("axis"), rank);
        if (!res.first) return status::invalid_argument;
        const auto axis = res.second;

        if (!shapes_match(inputs, outputs, axis, rank))
            return status::invalid_shape;

        std::vector<memory::desc> src_mds;
        src_mds.reserve(inputs.size());
        std::for_each(inputs.cbegin(), inputs.cend(), [&src_mds](const lt &in) {
            src_mds.push_back(make_dnnl_memory_desc(in));
        });
        auto dst_md = make_dnnl_memory_desc(outputs.front());
        // we need to let oneDNN choose the optimal tag for dst
        memory::desc dst_any_md {
                dst_md.dims(), dst_md.data_type(), format_tag::any};

        pd_ = primitive_desc(
                dst_any_md, static_cast<int>(axis), src_mds, engine_);
        prim_ = super(pd_);

        lt *orig_dst_lt = const_cast<lt *>(&outputs.front());
        fill_layout_info(orig_dst_lt, pd_.dst_desc());

        first_iteration_ = true;

        return status::success;
    }

    impl::status_t execute_impl(const op_t *op, const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);

        alloc_ = g_stream->get_engine()->get_allocator();
        stream_ = make_dnnl_stream(engine_, *g_stream);

        std::vector<memory> src_mems;
        src_mems.reserve(inputs.size());
        std::for_each(inputs.cbegin(), inputs.cend(),
                [&src_mems, this](const impl::tensor_t &in) {
                    src_mems.push_back(make_dnnl_memory(in, engine_));
                });
        memory dst_mem = make_dnnl_memory(outputs.front(), engine_);

        auto pd_dst_desc = pd_.dst_desc();
        if (pd_dst_desc != dst_mem.get_desc()) {
            if (first_iteration_) {
                // in case dst_opt_buf_ already holds the data
                free_buffer_if_allocated();
                dst_opt_buf_ = allocator::malloc(
                        pd_dst_desc.get_size(), engine_, alloc_);
                dst_opt_mem_ = memory(pd_dst_desc, engine_, dst_opt_buf_);

                dst_reorder_pd_ = dnnl::reorder::primitive_desc(
                        engine_, pd_dst_desc, engine_, dst_mem.get_desc());
            }
        } else {
            if (first_iteration_) dst_opt_mem_ = dst_mem;
            dst_opt_mem_.set_data_handle(dst_mem.get_data_handle());
        }

        if (first_iteration_) {
            for (int i = 0; i < inputs.size(); ++i)
                concat_args_[DNNL_ARG_MULTIPLE_SRC + i] = src_mems[i];
            concat_args_[DNNL_ARG_DST] = dst_opt_mem_;
        }

        prim_.execute(stream_, concat_args_);

        if (dst_mem.get_desc() != pd_dst_desc) {
            dnnl::reorder(dst_reorder_pd_)
                    .execute(stream_, dst_opt_mem_, dst_mem);
        }

        first_iteration_ = false;

        return status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
