/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_BACKEND_DNNL_OPERATORS_SOFTMAX_HPP
#define LLGA_BACKEND_DNNL_OPERATORS_SOFTMAX_HPP

#include <vector>

#include "backend/dnnl/tensor.hpp"
#include "common.hpp"

namespace llga {
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

public:
    void compute(const tensor &src, tensor &dst, const engine &aengine) {
        auto expected_src = src.reorder_if_differ_in(pd_.src_desc());
        tensor expected_dst = dst;
        if (pd_.dst_desc() != dst.get_desc()) {
            expected_dst = tensor {pd_.dst_desc(), aengine};
        }

        stream s(aengine);
        super(pd_).execute(s,
                {{DNNL_ARG_SRC, expected_src}, {DNNL_ARG_DST, expected_dst}});
        s.wait();

        if (expected_dst != dst) {
            dnnl::reorder(expected_dst, dst).execute(s, expected_dst, dst);
            s.wait();
        }
    }

    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // prepare the inputs and outputs' tensors' descs
        const desc src {inputs.at(softmax::kSrc)};
        impl::logical_tensor_t *dst_lt
                = const_cast<logical_tensor_t *>(&outputs.at(softmax::kDst));

        auto eng = engine_manager::get()->get_engine(*aengine);
        axis_ = anode->get_attr<int64_t>("axis");

        pd_ = primitive_desc({prop_kind::forward, src, axis_}, *eng);
        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        fill_layout_info(dst_lt, optimal_dst_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));
        tensor src_ts {inputs.at(softmax::kSrc), *eng};
        tensor dst_ts {outputs.at(softmax::kDst), *eng};
        softmax_forward::compute(src_ts, dst_ts, *eng);
        return impl::status::success;
    }
};

struct softmax_backward : public dnnl::softmax_backward, public kernel_base {
    using super = dnnl::softmax_backward;

private:
    primitive_desc pd_;
    int64_t axis_;

public:
    void compute(const tensor &dst, const tensor &diff_dst, tensor &diff_src,
            const engine &aengine) {
        auto expected_dst = dst.reorder_if_differ_in(pd_.dst_desc());
        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(pd_.diff_dst_desc());
        auto expected_diff_src
                = diff_src.reorder_if_differ_in(pd_.diff_src_desc());

        stream s(aengine);
        super(pd_).execute(s,
                {{DNNL_ARG_DST, expected_dst},
                        {DNNL_ARG_DIFF_DST, expected_diff_dst},
                        {DNNL_ARG_DIFF_SRC, expected_diff_src}});
        s.wait();

        if (expected_diff_src != diff_src) {
            dnnl::reorder(expected_diff_src, diff_src)
                    .execute(s, expected_diff_src, diff_src);
            s.wait();
        }
    }

    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // prepare the outputs' tensors' descs
        const desc dst {inputs.at(softmax_bwd::kDst)};
        const desc diff_dst {inputs.at(softmax_bwd::kDiff_dst)};

        impl::logical_tensor_t *diff_src_lt = const_cast<logical_tensor_t *>(
                &outputs.at(softmax_bwd::kDiff_src));

        auto eng = engine_manager::get()->get_engine(*aengine);
        axis_ = anode->get_attr<int64_t>("axis");

        auto forward_hints = softmax_forward::primitive_desc(
                {prop_kind::forward_training, dst, axis_}, *eng);
        pd_ = primitive_desc({diff_dst, dst, axis_}, *eng, forward_hints);

        const tensor::desc optimal_diff_src_desc {pd_.diff_src_desc()};
        fill_layout_info(diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));
        tensor dst {inputs.at(softmax_bwd::kDst), *eng};
        tensor diff_dst {inputs.at(softmax_bwd::kDiff_dst), *eng};
        tensor diff_src {outputs.at(softmax_bwd::kDiff_src), *eng};

        softmax_backward::compute(dst, diff_dst, diff_src, *eng);
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif
