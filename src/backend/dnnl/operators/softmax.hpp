/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_DNNL_OPERATORS_SOFTMAX_HPP
#define BACKEND_DNNL_OPERATORS_SOFTMAX_HPP

#include <vector>

#include "backend/dnnl/tensor.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace softmax {
enum softmax_inputs { kSrc };
enum softmax_outputs { kDst };
} // namespace softmax

namespace softmax_bwd {
enum softmax_bwd_inputs { kDiff_dst, kDst };
enum softmax_bwd_outputs { kDiff_src };
} // namespace softmax_bwd

struct softmax_forward : public dnnl::softmax_forward, public kernel_base {
    using super = dnnl::softmax_forward;

private:
    primitive_desc pd_;
    int64_t axis_;
    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    void compute(const tensor &src, tensor &dst, const dnnl::engine &p_engine,
            impl::allocator_t *alc, const dnnl::stream &p_stream) {
        auto expected_src = src.reorder_if_differ_in(p_stream, pd_.src_desc());
        tensor expected_dst = dst;
        if (pd_.dst_desc() != dst.get_desc()) {
            expected_dst = tensor {pd_.dst_desc(), p_engine, alc};
        }

        super(pd_).execute(p_stream,
                {{DNNL_ARG_SRC, expected_src}, {DNNL_ARG_DST, expected_dst}});

        if (expected_dst != dst) {
            dnnl::reorder(expected_dst, dst)
                    .execute(p_stream, expected_dst, dst);
        }
    }

    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // prepare the inputs and outputs' tensors' descs
        const desc src {inputs.at(softmax::kSrc)};
        impl::logical_tensor_t *dst_lt
                = const_cast<logical_tensor_t *>(&outputs.at(softmax::kDst));

        p_engine_ = make_dnnl_engine(*g_engine);
        axis_ = op->get_attr<int64_t>("axis");
        if (axis_ < 0) { axis_ += src.get_ndims(); }
        BACKEND_DNNL_ENFORCE(
                axis_ >= 0 && axis_ < src.get_ndims(), "Invalid softmax axis.");

        pd_ = primitive_desc(
                {prop_kind::forward, src, static_cast<int>(axis_)}, p_engine_);
        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        fill_layout_info(dst_lt, optimal_dst_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        tensor src_ts {inputs.at(softmax::kSrc), p_engine_, alc};
        tensor dst_ts {outputs.at(softmax::kDst), p_engine_, alc};
        softmax_forward::compute(src_ts, dst_ts, p_engine_, alc, p_stream_);
        return impl::status::success;
    }
};

struct softmax_backward : public dnnl::softmax_backward, public kernel_base {
    using super = dnnl::softmax_backward;

private:
    primitive_desc pd_;
    int64_t axis_;
    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    void compute(const tensor &dst, const tensor &diff_dst, tensor &diff_src,
            impl::allocator_t *alc, const dnnl::stream &p_stream) {
        UNUSED(alc);
        auto expected_dst = dst.reorder_if_differ_in(p_stream, pd_.dst_desc());
        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(p_stream, pd_.diff_dst_desc());
        auto expected_diff_src
                = diff_src.reorder_if_differ_in(p_stream, pd_.diff_src_desc());

        super(pd_).execute(p_stream,
                {{DNNL_ARG_DST, expected_dst},
                        {DNNL_ARG_DIFF_DST, expected_diff_dst},
                        {DNNL_ARG_DIFF_SRC, expected_diff_src}});

        if (expected_diff_src != diff_src) {
            dnnl::reorder(expected_diff_src, diff_src)
                    .execute(p_stream, expected_diff_src, diff_src);
        }
    }

    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // prepare the outputs' tensors' descs
        const desc dst {inputs.at(softmax_bwd::kDst)};
        const desc diff_dst {inputs.at(softmax_bwd::kDiff_dst)};

        impl::logical_tensor_t *diff_src_lt = const_cast<logical_tensor_t *>(
                &outputs.at(softmax_bwd::kDiff_src));

        axis_ = op->get_attr<int64_t>("axis");
        if (axis_ < 0) { axis_ += dst.get_ndims(); }
        BACKEND_DNNL_ENFORCE(
                axis_ >= 0 && axis_ < dst.get_ndims(), "Invalid softmax axis.");

        p_engine_ = make_dnnl_engine(*g_engine);
        auto forward_hints = softmax_forward::primitive_desc(
                {prop_kind::forward_training, dst, static_cast<int>(axis_)},
                p_engine_);
        pd_ = primitive_desc({diff_dst, dst, static_cast<int>(axis_)},
                p_engine_, forward_hints);

        const tensor::desc optimal_diff_src_desc {pd_.diff_src_desc()};
        fill_layout_info(diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        tensor dst {inputs.at(softmax_bwd::kDst), p_engine_, alc};
        tensor diff_dst {inputs.at(softmax_bwd::kDiff_dst), p_engine_, alc};
        tensor diff_src {outputs.at(softmax_bwd::kDiff_src), p_engine_, alc};

        softmax_backward::compute(dst, diff_dst, diff_src, alc, p_stream_);
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
