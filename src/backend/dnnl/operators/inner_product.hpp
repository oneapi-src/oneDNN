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

#ifndef BACKEND_DNNL_OPERATORS_INNER_PRODUCT_HPP
#define BACKEND_DNNL_OPERATORS_INNER_PRODUCT_HPP

#include <vector>

#include "backend/dnnl/tensor.hpp"
#include "interface/backend.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace inner_product {
enum inner_product_inputs { kSrc, kWeight, kBias };
enum inner_product_outputs { kDst };
} // namespace inner_product

struct inner_product_forward : public dnnl::inner_product_forward,
                               public kernel_base {
    using super = dnnl::inner_product_forward;

private:
    primitive_desc pd_;
    attr_t attr_;
    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;

        auto op_kind = op->get_kind();
        bool with_relu = op_kind == op_kind::matmul_relu ? true : false;

        p_engine_ = make_dnnl_engine(*g_engine);
        // prepare the inputs and outputs tensors' descs
        desc src {inputs.at(inner_product::kSrc)};
        desc weight {inputs.at(inner_product::kWeight)};
        const bool with_bias = inputs.size() > inner_product::kBias;
        desc bias
                = with_bias ? desc {inputs.at(inner_product::kBias)} : desc {};
        impl::logical_tensor_t *dst_lt = const_cast<impl::logical_tensor_t *>(
                &outputs.at(inner_product::kDst));

        auto dst_dims = impl::logical_tensor_wrapper(dst_lt).vdims();
        BACKEND_DNNL_ENFORCE(
                dst_dims[0] == src.get_dim(0), "Incorrect dims in output");
        BACKEND_DNNL_ENFORCE(
                dst_dims[1] == weight.get_dim(0), "Incorrect dims in output");

        // prepare the operator attributes
        attr_ = with_relu ? attr_t::fuse_eltwise() : attr_t {};

        BACKEND_DNNL_ENFORCE(utils::one_of(weight.get_data_type(),
                                     data_type::f32, data_type::bf16),
                "Incorrect data type in weight");

        // align weight data type with src
        data_type dst_data_type = src.get_data_type() == data_type::bf16
                ? data_type::bf16
                : data_type::f32;
        src = src.to_type(dst_data_type);
        weight = weight.to_type(dst_data_type);
        if (with_bias) {
            BACKEND_DNNL_ENFORCE(utils::one_of(bias.get_data_type(),
                                         data_type::f32, data_type::bf16),
                    "Incorrect data type in bias");
            bias = bias.to_format_any();
        }

        tensor::desc dst_desc(dst_dims, dst_data_type, format_tag::any);
        pd_ = with_bias
                ? primitive_desc(
                        {prop_kind::forward, src, weight, bias, dst_desc},
                        attr_, p_engine_)
                : primitive_desc({prop_kind::forward, src, weight, dst_desc},
                        attr_, p_engine_);

        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        fill_layout_info(dst_lt, optimal_dst_desc);

        // prepacking
        const tensor::desc optimal_weight_desc {pd_.weights_desc()};
        impl::logical_tensor_t *ori_weight_desc
                = const_cast<impl::logical_tensor_t *>(
                        &inputs.at(inner_product::kWeight));
        fill_layout_info(ori_weight_desc, optimal_weight_desc);
        if (with_bias) {
            const tensor::desc optimal_bias_desc {pd_.bias_desc()};
            impl::logical_tensor_t *ori_bias_desc
                    = const_cast<impl::logical_tensor_t *>(
                            &inputs.at(inner_product::kBias));
            fill_layout_info(ori_bias_desc, optimal_bias_desc);
        }
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);

        tensor x {inputs.at(inner_product::kSrc), p_engine_, alc};
        tensor w {inputs.at(inner_product::kWeight), p_engine_, alc};
        tensor y {outputs.at(inner_product::kDst), p_engine_, alc};

        if (inputs.size() > inner_product::kBias) {
            tensor b {inputs.at(inner_product::kBias), p_engine_, alc};
            compute(x, w, b, y, alc, p_stream_, prop_kind::forward);
        } else {
            compute(x, w, y, alc, p_stream_, prop_kind::forward);
        }
        return impl::status::success;
    }

private:
    void compute(const tensor &src, const tensor &weights, const tensor &bias,
            tensor &dst, impl::allocator_t *alc, const dnnl::stream &p_stream,
            const prop_kind aprop_kind = prop_kind::forward) {
        compute_impl</*with_bias=*/true>(
                src, weights, bias, dst, aprop_kind, alc, p_stream);
    }

    void compute(const tensor &src, const tensor &weights, tensor &dst,
            impl::allocator_t *alc, const dnnl::stream &p_stream,
            const prop_kind aprop_kind = prop_kind::forward) {
        static tensor dummy_bias;
        compute_impl</*with_bias=*/false>(
                src, weights, dummy_bias, dst, aprop_kind, alc, p_stream);
    }

    static tensor::desc expected_weights_desc(const dims &weights_dims,
            const dnnl::engine &p_engine, const dims &src_dims = dims(),
            data_type dtype = data_type::f32,
            data_type x_dtype = data_type::f32,
            prop_kind aprop_kind = prop_kind::forward) {
        auto x_dims = weights_dims;
        x_dims[0] = src_dims.empty() ? 1 : src_dims[0];
        auto y_dims = {x_dims[0], weights_dims[0]};

        BACKEND_DNNL_ENFORCE(x_dims.size() == weights_dims.size(),
                "Invalid dims for data and weights");
        tensor::desc src_desc(x_dims, x_dtype, tag::any);
        tensor::desc dst_desc(y_dims, dtype, tag::any);
        tensor::desc weights_desc(weights_dims, dtype, tag::any);
        auto pd = primitive_desc(
                {aprop_kind, src_desc, weights_desc, dst_desc}, p_engine);
        return pd.weights_desc();
    }

    template <bool with_bias>
    void compute_impl(const tensor &src, const tensor &weights,
            const tensor &bias, tensor &dst, const prop_kind aprop_kind,
            impl::allocator_t *alc, const dnnl::stream &p_stream) {
        // workaround: src and weights from caffe2 may have different dims.
        // It would be better for caffe2 to do this reshape anyway.
        auto src_ = src;
        if (src.ndims() != weights.ndims()) {
            auto new_dims = weights.get_dims();
            new_dims[0] = src.get_dim(0);
            src_.reshape(p_stream, new_dims);
        }
        compute_impl_<with_bias>(
                src_, weights, bias, dst, aprop_kind, alc, p_stream);
    }

    template <bool with_bias>
    void compute_impl_(const tensor &src, const tensor &weights,
            const tensor &bias, tensor &dst, const prop_kind aprop_kind,
            impl::allocator_t *alc, const dnnl::stream &p_stream) {
        UNUSED(aprop_kind);
        UNUSED(alc);
        tensor::desc src_desc, weights_desc, bias_desc;
        attr_t op_attr, src_attr, weights_attr, bias_attr;

        auto expected_src
                = src.reorder_if_differ_in(p_stream, pd_.src_desc(), src_attr);
        auto expected_weights = weights.reorder_if_differ_in(
                p_stream, pd_.weights_desc(), weights_attr);
        dst.reinit_if_possible(p_stream, pd_.dst_desc());

        if (with_bias) {
            auto expected_bias = bias.reorder_if_differ_in(
                    p_stream, pd_.bias_desc(), bias_attr);
            super(pd_).execute(p_stream,
                    {{DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_WEIGHTS, expected_weights},
                            {DNNL_ARG_BIAS, expected_bias},
                            {DNNL_ARG_DST, dst}});
        } else {
            super(pd_).execute(p_stream,
                    {{DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_WEIGHTS, expected_weights},
                            {DNNL_ARG_DST, dst}});
        }
    }
};

struct inner_product_backward_data : public dnnl::inner_product_backward_data {
    using super = dnnl::inner_product_backward_data;

    static void compute(const tensor &diff_dst, const tensor &weights,
            const dims &diff_src_dims, tensor &diff_src,
            const dnnl::engine &p_engine, impl::allocator_t *alc,
            const dnnl::stream &p_stream) {
        auto weights_ = weights;
        if (diff_dst.get_data_type() == data_type::bf16) {
            weights_ = tensor(
                    weights.get_desc().to_type(data_type::bf16), p_engine, alc);
            weights_.reorder_from(p_stream, weights);
        }

        // workaround: diff_src and weights from caffe2 may have different dims.
        // It would be better for caffe2 to do this reshape anyway.
        if (diff_src_dims.size() != weights.ndims()) {
            auto new_dims = diff_src_dims;
            new_dims[0] = weights.get_dim(0);
            weights_.reshape(p_stream, new_dims);
        }

        auto diff_dst_desc = diff_dst.get_desc().to_format_any();
        auto weights_desc = weights_.get_desc();
        auto diff_src_desc = tensor::desc(
                diff_src_dims, diff_dst.get_data_type(), tag::any);

        auto forward_hints = inner_product_forward::primitive_desc(
                {prop_kind::forward, diff_src_desc, weights_desc,
                        diff_dst_desc},
                p_engine);

        auto pd = primitive_desc({diff_src_desc, weights_desc, diff_dst_desc},
                p_engine, forward_hints);

        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(p_stream, pd.diff_dst_desc());
        auto expected_weights
                = weights_.reorder_if_differ_in(p_stream, pd.weights_desc());
        diff_src.reinit_if_possible(p_stream, pd.diff_src_desc());

        super(pd).execute(p_stream,
                {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                        {DNNL_ARG_WEIGHTS, expected_weights},
                        {DNNL_ARG_DIFF_SRC, diff_src}});
    }
};

struct inner_product_backward_weights
    : public dnnl::inner_product_backward_weights {

    using super = dnnl::inner_product_backward_weights;

    static void compute(const tensor &src, const tensor &diff_dst,
            tensor &diff_weights, tensor &diff_bias,
            const dnnl::engine &p_engine, const dnnl::stream &p_stream,
            const data_type diff_weight_type = data_type::undef) {
        compute_impl</*with_diff_bias=*/true>(src, diff_dst, diff_weights,
                diff_bias, diff_weight_type, p_engine, p_stream);
    }

    static void compute(const tensor &src, const tensor &diff_dst,
            tensor &diff_weights, const dnnl::engine &p_engine,
            const dnnl::stream &p_stream,
            const data_type diff_weight_type = data_type::undef) {
        static tensor dummy_diff_bias;
        compute_impl</*with_diff_bias=*/false>(src, diff_dst, diff_weights,
                dummy_diff_bias, diff_weight_type, p_engine, p_stream);
    }

private:
    template <bool with_diff_bias = true>
    static void compute_impl(const tensor &src, const tensor &diff_dst,
            tensor &diff_weights, tensor &diff_bias,
            const data_type diff_weight_type, const dnnl::engine &p_engine,
            const dnnl::stream &p_stream) {
        auto src_desc = src.get_desc().to_format_any();
        auto diff_dst_desc = diff_dst.get_desc().to_format_any();
        auto diff_weights_dims = src.get_dims();
        diff_weights_dims[0] = diff_dst.get_dim(1);
        data_type diff_dst_type = diff_dst.get_data_type();
        data_type diff_weight_type_in = data_type::undef == diff_weight_type
                ? diff_dst_type
                : diff_weight_type;
        auto diff_weights_desc = tensor::desc(
                diff_weights_dims, diff_weight_type_in, tag::any);

        auto diff_bias_desc = tensor::desc(
                {diff_dst.get_dim(1)}, diff_weight_type_in, tag::any);

        // for forward hint, weights_desc should have same data_type
        // with other input desc, expect for bias_desc
        auto weights_desc = diff_weights_desc;
        if (diff_weight_type_in != diff_dst_type) {
            weights_desc = weights_desc.to_type(diff_dst_type);
        }
        auto forward_hints = with_diff_bias
                ? inner_product_forward::primitive_desc(
                        {prop_kind::forward, src_desc, weights_desc,
                                diff_bias_desc, diff_dst_desc},
                        p_engine)
                : inner_product_forward::primitive_desc(
                        {prop_kind::forward, src_desc, weights_desc,
                                diff_dst_desc},
                        p_engine);
        auto pd = with_diff_bias
                ? primitive_desc({src_desc, diff_weights_desc, diff_bias_desc,
                                         diff_dst_desc},
                        p_engine, forward_hints)
                : primitive_desc({src_desc, diff_weights_desc, diff_dst_desc},
                        p_engine, forward_hints);

        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(p_stream, pd.diff_dst_desc());
        auto expected_src = src.reorder_if_differ_in(p_stream, pd.src_desc());
        diff_weights.reinit_if_possible(p_stream, pd.diff_weights_desc());

        exec_args args {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                {DNNL_ARG_SRC, expected_src},
                {DNNL_ARG_DIFF_WEIGHTS, diff_weights}};

        if (with_diff_bias) {
            diff_bias.reinit_if_possible(p_stream, pd.diff_bias_desc());
            args.insert({DNNL_ARG_DIFF_BIAS, diff_bias});
        }

        super(pd).execute(p_stream, args);
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
