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

#ifndef BACKEND_DNNL_OPERATORS_POOL_HPP
#define BACKEND_DNNL_OPERATORS_POOL_HPP

#include <string>
#include <vector>

#include "backend/dnnl/tensor.hpp"
#include "backend/dnnl/utils.hpp"
#include "common.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace pool {
enum pool_inputs { kSrc };
enum pool_outputs { kDst };
} // namespace pool

namespace pool_bwd {
enum pool_bwd_inputs { kSrc, kDiff_dst };
enum pool_bwd_outputs { kDiff_src };
} // namespace pool_bwd

namespace pool_bwd_with_indices {
enum maxpool_bwd_inputs { kSrc, kIndices, kDiff_dst };
} // namespace pool_bwd_with_indices

struct pooling_forward : public dnnl::pooling_v2_forward, public kernel_base {
    using super = dnnl::pooling_v2_forward;
    using dims_t = std::vector<dnnl::graph::impl::dim_t>;

private:
    primitive_desc pd_;

    algorithm algo_;

    bool is_training_ {false};
    prop_kind prop_kind_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    void compute(const tensor &src, const dims &output_sizes, tensor &dst,
            const dnnl::engine &p_engine, impl::allocator_t *alc,
            const dnnl::stream &p_stream) {
        UNUSED(output_sizes);
        bool with_workspace = prop_kind_ == prop_kind::forward_training
                && algo_ == dnnl::algorithm::pooling_max;

        auto expected_src = src.reorder_if_differ_in(p_stream, pd_.src_desc());

        tensor expected_dst = dst;
        if (pd_.dst_desc() != dst.get_desc()) {
            expected_dst = tensor {pd_.dst_desc(), p_engine, alc};
        }
        exec_args args {
                {DNNL_ARG_SRC, expected_src}, {DNNL_ARG_DST, expected_dst}};
        if (with_workspace) {
            expected_dst.init_workspace(pd_.workspace_desc());
            args.insert({DNNL_ARG_WORKSPACE, expected_dst.get_workspace()});
        }

        super(pd_).execute(p_stream, args);

        // if output layout has been set and different from optimal layout
        // we have to do reorder
        if (expected_dst != dst) {
            dnnl::reorder(expected_dst, dst)
                    .execute(p_stream, expected_dst, dst);
        }
    }

    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        dims strides = anode->get_attr<dims>("strides");
        dims kernel = anode->get_attr<dims>("kernel");
        dims pads_begin = anode->get_attr<dims>("pads_begin");
        dims pads_end = anode->get_attr<dims>("pads_end");
        std::string data_format = anode->get_attr<std::string>("data_format");
        // "NXC" format will be converted to "NCX" format
        impl::logical_tensor_t src_lt = inputs.at(pool::kSrc);
        impl::logical_tensor_t dst_lt = outputs.at(pool::kDst);

        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper(&inputs.at(pool::kSrc))
                             .reorder_data_dims_strides();
            dst_lt = impl::logical_tensor_wrapper(&outputs.at(pool::kDst))
                             .reorder_data_dims_strides();
        }
        // prepare the inputs and outputs' tensors' descs
        const desc src {src_lt};
        const desc dst {dst_lt};

        op_kind_t kind = anode->get_op_kind();
        dims dilations;
        if (kind == op_kind::MaxPool) {
            algo_ = algorithm::pooling_max;
            dilations = anode->get_attr<dims>("dilations");
            // default dilations are all 1s but in primitive, they're 0s.
            std::for_each(dilations.begin(), dilations.end(),
                    [](dim_t &v) { v -= 1; });
        } else if (kind == op_kind::AvgPool) {
            dilations = dims(strides.size(), 0);
            bool exclude_pad = anode->get_attr<bool>("exclude_pad");
            algo_ = exclude_pad ? algorithm::pooling_avg_exclude_padding
                                : algorithm::pooling_avg_include_padding;
        } else {
            BACKEND_DNNL_ENFORCE(0, "Unsupported pool op.");
        }

        p_engine_ = make_dnnl_engine(*g_engine);

        // workaround: use src once issue intel/mkl-dnn#588 is
        // resolved
        auto expected_src = src.is_4c_blocked() ? src.to_default_format() : src;
        auto any_dst = dst.to_format_any();

        prop_kind_ = is_training_ ? prop_kind::forward
                                  : prop_kind::forward_inference;

        pd_ = primitive_desc({prop_kind_, algo_, expected_src, any_dst, strides,
                                     kernel, dilations, pads_begin, pads_end},
                p_engine_);

        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        impl::logical_tensor_t *ori_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(pool::kDst));
        fill_layout_info(ori_dst_lt, optimal_dst_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        std::string data_format = anode->get_attr<std::string>("data_format");
        auto &src_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(pool::kSrc).get_logical_tensor());
        auto &dst_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(pool::kDst).get_logical_tensor());
        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper(src_lt)
                             .reorder_data_dims_strides();
            dst_lt = impl::logical_tensor_wrapper(dst_lt)
                             .reorder_data_dims_strides();
        }

        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        tensor x {src_lt, p_engine_, alc,
                inputs.at(pool::kSrc).get_data_handle()};
        tensor y {dst_lt, p_engine_, alc,
                outputs.at(pool::kDst).get_data_handle()};

        dims outsize = y.get_dims();
        pooling_forward::compute(x, outsize, y, p_engine_, alc, p_stream_);
        return impl::status::success;
    }
};

struct pooling_backward : public dnnl::pooling_v2_backward, public kernel_base {
    using super = dnnl::pooling_v2_backward;

private:
    dnnl::pooling_v2_forward::primitive_desc forward_hints_;
    primitive_desc pd_;
    op_kind_t kind_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    void compute(const tensor &diff_dst, const tensor &src, tensor &diff_src,
            const dnnl::engine &p_engine, impl::allocator_t *alc,
            const dnnl::stream &p_stream, tensor indices = tensor {}) {
        // generate indices tensor from src when it's needed
        // but can't get from function parameters
        if (kind_ == op_kind::MaxPoolBackprop && indices.is_empty()) {
            auto expected_src = src.reorder_if_differ_in(
                    p_stream, forward_hints_.src_desc());
            auto expected_dst
                    = tensor {forward_hints_.dst_desc(), p_engine, alc};
            indices = tensor {forward_hints_.workspace_desc(), p_engine, alc};
            exec_args args {{DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_DST, expected_dst},
                    {DNNL_ARG_WORKSPACE, indices}};

            dnnl::pooling_v2_forward(forward_hints_).execute(p_stream, args);
        }

        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(p_stream, pd_.diff_dst_desc());
        auto expected_diff_src
                = diff_src.reorder_if_differ_in(p_stream, pd_.diff_src_desc());

        exec_args args = exec_args {
                {DNNL_ARG_DIFF_DST, expected_diff_dst},
                {DNNL_ARG_DIFF_SRC, expected_diff_src},
        };

        if (!indices.is_empty()) { args.insert({DNNL_ARG_WORKSPACE, indices}); }

        super(pd_).execute(p_stream, args);

        if (expected_diff_src != diff_src) {
            dnnl::reorder(expected_diff_src, diff_src)
                    .execute(p_stream, expected_diff_src, diff_src);
        }
    }

    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // prepare the inputs and outputs' tensors' descs
        const desc src {inputs.at(pool_bwd::kSrc)};
        const desc diff_dst {inputs.at(pool_bwd::kDiff_dst)};
        impl::logical_tensor_t *diff_src_lt
                = const_cast<impl::logical_tensor_t *>(
                        &outputs.at(pool_bwd::kDiff_src));
        const desc diff_src {*diff_src_lt};

        dims strides = anode->get_attr<dims>("strides");
        dims kernel = anode->get_attr<dims>("kernel");
        dims pads_begin = anode->get_attr<dims>("pads_begin");
        dims pads_end = anode->get_attr<dims>("pads_end");

        kind_ = anode->get_op_kind();
        algorithm algo = algorithm::undef;
        dims dilations {};
        if (kind_ == op_kind::AvgPoolBackprop) {
            bool exclude_pad = anode->get_attr<bool>("exclude_pad");
            algo = exclude_pad ? algorithm::pooling_avg_exclude_padding
                               : algorithm::pooling_avg_include_padding;
            dilations = dims(strides.size(), 0);
        } else if (kind_ == op_kind::MaxPoolBackprop) {
            algo = algorithm::pooling_max;
            dilations = anode->get_attr<dims>("dilations");
            // default dilations are all 1s but in primitive, they're 0s.
            std::for_each(dilations.begin(), dilations.end(),
                    [](dim_t &v) { v -= 1; });
        } else {
            return status::unsupported;
        }

        p_engine_ = make_dnnl_engine(*g_engine);
        forward_hints_ = dnnl::pooling_v2_forward::primitive_desc(
                {prop_kind::forward_training, algo, src, diff_dst, strides,
                        kernel, dilations, pads_begin, pads_end},
                p_engine_);

        pd_ = primitive_desc({algo, src, diff_dst, strides, kernel, dilations,
                                     pads_begin, pads_end},
                p_engine_, forward_hints_);

        const tensor::desc optimal_diff_src_desc {pd_.diff_src_desc()};
        fill_layout_info(diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        tensor src {inputs.at(pool_bwd::kSrc), p_engine_, alc};
        tensor diff_dst {};
        tensor indices {};
        if (anode->get_op_kind() == op_kind::MaxPoolBackprop
                && inputs.size() > pool_bwd_with_indices::kDiff_dst) {
            diff_dst = tensor {inputs.at(pool_bwd_with_indices::kDiff_dst),
                    p_engine_, alc};
            indices = tensor {
                    inputs.at(pool_bwd_with_indices::kIndices), p_engine_, alc};
        } else {
            diff_dst = tensor {inputs.at(pool_bwd::kDiff_dst), p_engine_, alc};
        }

        tensor diff_src {outputs.at(pool_bwd::kDiff_src), p_engine_, alc};
        pooling_backward::compute(
                diff_dst, src, diff_src, p_engine_, alc, p_stream_, indices);
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
