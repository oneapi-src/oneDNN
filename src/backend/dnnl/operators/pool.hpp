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

#ifndef LLGA_BACKEND_DNNL_OPERATORS_POOL_HPP
#define LLGA_BACKEND_DNNL_OPERATORS_POOL_HPP

#include <string>
#include <vector>

#include "backend/dnnl/tensor.hpp"
#include "backend/dnnl/utils.hpp"
#include "common.hpp"

namespace llga {
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

struct pooling_forward : public dnnl::pooling_forward, public kernel_base {
    using super = dnnl::pooling_forward;
    using dims_t = std::vector<llga::impl::dim_t>;

private:
    primitive_desc pd_;

    algorithm algo_;

    bool is_training_ {false};
    prop_kind prop_kind_;

public:
    void compute(const tensor &src, const dims &output_sizes, tensor &dst,
            const engine &aengine) {
        bool with_workspace = prop_kind_ == prop_kind::forward_training
                && algo_ == dnnl::algorithm::pooling_max;

        auto expected_src = src.reorder_if_differ_in(pd_.src_desc());

        tensor expected_dst = dst;
        if (pd_.dst_desc() != dst.get_desc()) {
            expected_dst = tensor {pd_.dst_desc(), aengine};
        }
        exec_args args {
                {DNNL_ARG_SRC, expected_src}, {DNNL_ARG_DST, expected_dst}};
        if (with_workspace) {
            expected_dst.init_workspace(pd_.workspace_desc());
            args.insert({DNNL_ARG_WORKSPACE, expected_dst.get_workspace()});
        }

        stream s(aengine);
        super(pd_).execute(s, args);
        s.wait();
        // if output layout has been set and different from optimal layout
        // we have to do reorder
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
        if (kind == op_kind::MaxPool) {
            algo_ = algorithm::pooling_max;
        } else if (kind == op_kind::AvgPool) {
            bool exclude_pad = anode->get_attr<bool>("exclude_pad");
            algo_ = exclude_pad ? algorithm::pooling_avg_exclude_padding
                                : algorithm::pooling_avg_include_padding;
        } else {
            BACKEND_DNNL_ENFORCE(0, "Unsupported pool op.");
        }

        auto eng = engine_manager::get()->get_engine(*aengine);

        auto expected_src = src.is_4c_blocked() ? src.to_default_format() : src;
        auto any_dst = dst.to_format_any();

        prop_kind_ = is_training_ ? prop_kind::forward
                                  : prop_kind::forward_inference;

        pd_ = primitive_desc({prop_kind_, algo_, expected_src, any_dst, strides,
                                     kernel, pads_begin, pads_end},
                *eng);

        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        impl::logical_tensor_t *ori_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(pool::kDst));
        fill_layout_info(ori_dst_lt, optimal_dst_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
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

        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));
        tensor x {src_lt, inputs.at(pool::kSrc).get_data_handle(), *eng};
        tensor y {dst_lt, outputs.at(pool::kDst).get_data_handle(), *eng};

        dims outsize = y.get_dims();
        pooling_forward::compute(x, outsize, y, *eng);
        return impl::status::success;
    }
};

struct pooling_backward : public dnnl::pooling_backward, public kernel_base {
    using super = dnnl::pooling_backward;

private:
    dnnl::pooling_forward::primitive_desc forward_hints_;
    primitive_desc pd_;
    op_kind_t kind_;
    bool is_training_ {true};

public:
    void compute(const tensor &diff_dst, const tensor &src, tensor &diff_src,
            const engine &aengine, tensor indices = tensor {}) {
        stream s(aengine);
        // generate indices tensor from src when it's needed
        // but can't get from function parameters
        if (kind_ == op_kind::MaxPoolBackprop && indices.is_empty()) {
            auto expected_src
                    = src.reorder_if_differ_in(forward_hints_.src_desc());
            auto expected_dst = tensor {forward_hints_.dst_desc(), aengine};
            indices = tensor {forward_hints_.workspace_desc(), aengine};
            exec_args args {{DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_DST, expected_dst},
                    {DNNL_ARG_WORKSPACE, indices}};

            dnnl::pooling_forward(forward_hints_).execute(s, args);
            s.wait();
        }

        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(pd_.diff_dst_desc());
        auto expected_diff_src
                = diff_src.reorder_if_differ_in(pd_.diff_src_desc());

        exec_args args = exec_args {
                {DNNL_ARG_DIFF_DST, expected_diff_dst},
                {DNNL_ARG_DIFF_SRC, expected_diff_src},
        };

        if (!indices.is_empty()) { args.insert({DNNL_ARG_WORKSPACE, indices}); }

        super(pd_).execute(s, args);
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
        algorithm algo;
        if (kind_ == op_kind::AvgPoolBackprop) {
            bool exclude_pad = anode->get_attr<bool>("exclude_pad");
            algo = exclude_pad ? algorithm::pooling_avg_exclude_padding
                               : algorithm::pooling_avg_include_padding;
        } else if (kind_ == op_kind::MaxPoolBackprop) {
            algo = algorithm::pooling_max;
        } else {
            BACKEND_DNNL_ENFORCE(0, "Unsupported pool_backward op.");
        }

        auto eng = engine_manager::get()->get_engine(*aengine);

        forward_hints_ = dnnl::pooling_forward::primitive_desc(
                {prop_kind::forward_training, algo, src, diff_dst, strides,
                        kernel, pads_begin, pads_end},
                *eng);

        pd_ = primitive_desc(
                {algo, src, diff_dst, strides, kernel, pads_begin, pads_end},
                *eng, forward_hints_);

        const tensor::desc optimal_diff_src_desc {pd_.diff_src_desc()};
        fill_layout_info(diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));
        tensor src {inputs.at(pool_bwd::kSrc), *eng};
        tensor diff_dst {};
        tensor indices {};
        if (anode->get_op_kind() == op_kind::MaxPoolBackprop
                && inputs.size() > pool_bwd_with_indices::kDiff_dst) {
            diff_dst = tensor {
                    inputs.at(pool_bwd_with_indices::kDiff_dst), *eng};
            indices = tensor {inputs.at(pool_bwd_with_indices::kIndices), *eng};
        } else {
            diff_dst = tensor {inputs.at(pool_bwd::kDiff_dst), *eng};
        }

        tensor diff_src {outputs.at(pool_bwd::kDiff_src), *eng};
        pooling_backward::compute(diff_dst, src, diff_src, *eng, indices);
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif
