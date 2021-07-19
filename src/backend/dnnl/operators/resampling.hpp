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

#ifndef BACKEND_DNNL_OPERATORS_RESAMPLING_HPP
#define BACKEND_DNNL_OPERATORS_RESAMPLING_HPP

#include <string>
#include <vector>

#include "backend/dnnl/tensor.hpp"
#include "interface/backend.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace resampling {
enum resampling_inputs { kSrc, kSizes_Scales, kAxes };
enum resampling_outputs { kDst };
} // namespace resampling

namespace resampling_bwd {
enum resampling_bwd_inputs { kSrc, kDiff_dst, kSizes_Scales, kAxes };
enum resampling_bwd_outputs { kDiff_src, kDiff_Scales };
} // namespace resampling_bwd

struct resampling_forward : public dnnl::resampling_forward,
                            public kernel_base {
    using super = dnnl::resampling_forward;

private:
    primitive_desc pd_;
    dnnl::resampling_forward prim_;
    std::string mode_;
    algorithm alg_;
    dnnl::engine p_engine_;

public:
    void compute(const tensor &src, tensor &dst, impl::allocator_t *alc,
            const dnnl::stream &p_stream) const {
        tensor expected_src, expected_dst;

        if (src.get_desc() != pd_.src_desc()) {
            if (expected_src.is_empty()) {
                expected_src = tensor {pd_.src_desc(), p_engine_, alc};
            }
            src.reorder_to(p_stream, expected_src);
        } else {
            expected_src = src;
        }

        if (dst.get_desc() != pd_.dst_desc()) {
            if (expected_dst.is_empty()) {
                expected_dst = tensor {pd_.dst_desc(), p_engine_, alc};
            }
        } else
            expected_dst = dst;

        prim_.execute(p_stream,
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
        const desc src {inputs.at(resampling::kSrc)};
        const desc dst {outputs.at(resampling::kDst)};

        p_engine_ = make_dnnl_engine(*g_engine);
        mode_ = op->get_attr<std::string>("mode");
        if (mode_ == "nearest") {
            alg_ = algorithm::resampling_nearest;
        } else if (mode_ == "linear") {
            alg_ = algorithm::resampling_linear;
        } else {
            BACKEND_DNNL_ENFORCE(0, "Unsupported resampling mode.");
        }

        pd_ = primitive_desc({prop_kind::forward, alg_, src, dst}, p_engine_);
        prim_ = super(pd_);
        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        impl::logical_tensor_t *dst_lt
                = const_cast<logical_tensor_t *>(&outputs.at(resampling::kDst));
        fill_layout_info(dst_lt, optimal_dst_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        tensor src_ts {inputs.at(resampling::kSrc), p_engine_, alc};
        tensor dst_ts {outputs.at(resampling::kDst), p_engine_, alc};
        resampling_forward::compute(src_ts, dst_ts, alc, p_stream);
        return impl::status::success;
    }
};

struct resampling_backward : public dnnl::resampling_backward,
                             public kernel_base {
    using super = dnnl::resampling_backward;

private:
    primitive_desc pd_;
    resampling_forward::primitive_desc forward_hints_;
    std::string mode_;
    algorithm alg_;

    tensor expected_src_;
    tensor expected_diff_src_;
    tensor expected_diff_dst_;

    // TODO(Jihui): dnnl backward not support calculate diff_scales
    std::string shape_calculation_mode_;
    tensor expected_diff_scales_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    void compute(const tensor &src, const tensor &diff_dst, tensor &diff_src,
            impl::allocator_t *alc) {
        if (src.get_desc() != forward_hints_.src_desc()) {
            if (expected_src_.is_empty()) {
                expected_src_
                        = tensor {forward_hints_.src_desc(), p_engine_, alc};
            }
            src.reorder_to(p_stream_, expected_src_);
        } else {
            expected_src_ = src;
        }

        if (diff_dst.get_desc() != pd_.diff_dst_desc()) {
            if (expected_diff_dst_.is_empty()) {
                expected_diff_dst_
                        = tensor {pd_.diff_dst_desc(), p_engine_, alc};
            }
            diff_dst.reorder_to(p_stream_, expected_diff_dst_);
        } else
            expected_diff_dst_ = diff_dst;

        if (diff_src.get_desc() != pd_.diff_src_desc()) {
            if (expected_diff_src_.is_empty()) {
                expected_diff_src_
                        = tensor {pd_.diff_src_desc(), p_engine_, alc};
            }
        } else {
            expected_diff_src_ = diff_src;
        }

        exec_args args;

        args.insert({DNNL_ARG_SRC, expected_src_});
        args.insert({DNNL_ARG_DIFF_SRC, expected_diff_src_});
        args.insert({DNNL_ARG_DIFF_DST, expected_diff_dst_});

        super(pd_).execute(p_stream_, args);

        if (expected_diff_src_ != diff_src) {
            dnnl::reorder(expected_diff_src_, diff_src)
                    .execute(p_stream_, expected_diff_src_, diff_src);
        }
    }

    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // prepare the outputs' tensors' descs
        const desc src {inputs.at(resampling_bwd::kSrc)};
        const desc diff_dst {inputs.at(resampling_bwd::kDiff_dst)};
        const desc diff_src {outputs.at(resampling_bwd::kDiff_src)};

        mode_ = op->get_attr<std::string>("mode");
        if (mode_ == "nearest") {
            alg_ = algorithm::resampling_nearest;
        } else if (mode_ == "linear") {
            alg_ = algorithm::resampling_linear;
        } else {
            BACKEND_DNNL_ENFORCE(0, "Unsupported resampling mode.");
        }

        p_engine_ = make_dnnl_engine(*g_engine);
        forward_hints_ = resampling_forward::primitive_desc(
                {prop_kind::forward_training, alg_, src, diff_dst}, p_engine_);

        pd_ = primitive_desc(
                {alg_, diff_src, diff_dst}, p_engine_, forward_hints_);

        const tensor::desc optimal_diff_src_desc {pd_.diff_src_desc()};
        impl::logical_tensor_t *diff_src_lt = const_cast<logical_tensor_t *>(
                &outputs.at(resampling_bwd::kDiff_src));
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

        tensor src {inputs.at(resampling_bwd::kSrc), p_engine_, alc};
        tensor diff_dst {inputs.at(resampling_bwd::kDiff_dst), p_engine_, alc};
        tensor diff_src {outputs.at(resampling_bwd::kDiff_src), p_engine_, alc};

        resampling_backward::compute(src, diff_dst, diff_src, alc);
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
