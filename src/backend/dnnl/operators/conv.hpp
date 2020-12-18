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

#ifndef LLGA_BACKEND_DNNL_OPERATORS_CONV_HPP
#define LLGA_BACKEND_DNNL_OPERATORS_CONV_HPP

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "backend/dnnl/abstract_types.hpp"
#include "backend/dnnl/tensor.hpp"

#include "bn_fusion.hpp"
#include "common.hpp"

namespace llga {
namespace impl {
namespace dnnl_impl {

namespace conv {
enum conv_inputs { kSrc, kWeight, kBias };
enum fused_bn_inputs { kScale, kShift, kMean, kVariance };
enum conv_outputs { kDst };
} // namespace conv

namespace conv_bwd_data {
enum conv_bwd_inputs { kDiffdst, kWeight };
enum conv_bwd_outputs { kDiffsrc };
} // namespace conv_bwd_data

namespace conv_bwd_filter {
enum conv_bwd_inputs { kSrc, kDiffdst };
enum conv_bwd_outputs { kDiffweight, kDiffbias };
} // namespace conv_bwd_filter

struct convolution_forward_params {
    dnnl::convolution_forward::primitive_desc pd;
    int64_t groups;
    tensor scratchpad;
};

/**
 * Convolution operators' set for checking some feature, fusion support for a
 * specific convolutional op_kind_t.
 */
struct convolution_op_set {
    /**
     * Check if convolution operator has bias add.
     *
     * @param kind operator kind
     * @return whether the operator has bias add
     */
    static bool with_bias(op_kind_t kind) {
        static const std::set<op_kind_t> with_bias_set {op_kind::conv_bias,
                op_kind::conv_bias_add, op_kind::conv_bias_add_elu,
                op_kind::conv_bias_add_relu, op_kind::conv_bias_add_relu6,
                op_kind::conv_bias_bn, op_kind::conv_bias_elu,
                op_kind::conv_bias_relu, op_kind::conv_bias_sigmoid,
                op_kind::conv_bias_hardtanh, op_kind::conv_bias_relu6,
                op_kind::conv_bias_square, op_kind::conv_bias_tanh,
                op_kind::conv_bias_abs, op_kind::conv_bias_sqrt,
                op_kind::conv_bias_bn_relu, op_kind::conv_bias_bn_add,
                op_kind::conv_bias_bn_add_relu,
                op_kind::conv_bwd_f_biasadd_bwd};
        return with_bias_set.count(kind);
    }

    /**
     * Check if convolution operator fuses batchnorm
     *
     * @param kind operator kind
     * @return whether the operator fuses batchnorm
     */
    static bool fuse_batchnorm(op_kind_t kind) {
        static const std::set<op_kind_t> with_batchnorm_set {
                op_kind::conv_bias_bn, op_kind::conv_bias_bn_add,
                op_kind::conv_bias_bn_add_relu, op_kind::conv_bias_bn_relu,
                op_kind::conv_bn, op_kind::conv_bn_add,
                op_kind::conv_bn_add_relu, op_kind::conv_bn_relu};
        return with_batchnorm_set.count(kind);
    }

    /**
     * Check if convolution operator fuses sum
     *
     * @param kind operator kind
     * @return whether the operator fused sum
     */
    static bool fuse_sum(op_kind_t kind) {
        static const std::set<op_kind_t> with_sum_set {op_kind::conv_add,
                op_kind::conv_add_relu, op_kind::conv_bias_add,
                op_kind::conv_bias_add_elu, op_kind::conv_bias_add_relu,
                op_kind::conv_bias_add_relu6, op_kind::conv_bias_bn_add,
                op_kind::conv_bias_bn_add_relu, op_kind::conv_bn_add,
                op_kind::conv_bn_add_relu};
        return with_sum_set.count(kind);
    }

    /**
     * Check if convolution operator fuses activation relu
     *
     * @param kind operator kind
     * @return whether the operator fused activation relu
     */
    static bool fuse_eltwise(op_kind_t kind) {
        static const std::set<op_kind_t> with_eltwise_set {
                op_kind::conv_add_relu,
                op_kind::conv_bias_add_elu,
                op_kind::conv_bias_add_relu,
                op_kind::conv_bias_add_relu6,
                op_kind::conv_bias_bn_add_relu,
                op_kind::conv_bias_bn_relu,

                op_kind::conv_bias_abs,
                op_kind::conv_bias_elu,
                op_kind::conv_bias_hardtanh,
                op_kind::conv_bias_relu6,
                op_kind::conv_bias_sigmoid,
                op_kind::conv_bias_sqrt,
                op_kind::conv_bias_square,
                op_kind::conv_bias_tanh,

                op_kind::conv_bn_add_relu,
                op_kind::conv_bn_relu,
                op_kind::conv_bias_relu,
                op_kind::conv_relu,
        };
        return with_eltwise_set.count(kind);
    }
};

struct convolution_forward : public dnnl::convolution_forward,
                             public kernel_base {
    using super = dnnl::convolution_forward;
    using dims_t = std::vector<llga::impl::dim_t>;

private:
    // cahced pd is in this struct
    convolution_forward_params params_;

    dims strides_;
    dims dilates_;
    dims pads_begin_;
    dims pads_end_;
    int64_t groups_;
    std::string data_format_, filter_format_;

    // cahce expected data to avoid creating memory in every iteration
    tensor expected_src_;
    tensor expected_weights_;
    tensor expected_bias_;
    tensor expected_dst_;

    tensor updated_weights_;
    tensor updated_bias_;

    attr_t attr_;

    bool with_bias_;
    bool with_sum_;
    bool with_eltwise_;
    bool with_bn_;

    float alpha_ = 0.f;
    float beta_ = 0.f;

    size_t bn_input_offset_;

    float epsilon_; // bn epsilon

    bool is_training_;

    // FIXME(qun) NOT well designed
    /// \note Currently we don't have enough information from framework to
    /// decide cache or not. Also we think that caching data in a library
    /// is not safe at this moment.
    char *val = std::getenv("DNNL_GRAPH_WEIGHT_CACHE");
    bool enable_cache_data_ = (val != nullptr && std::strcmp(val, "1") == 0);

public:
    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        const op_kind_t conv_kind = anode->get_op_kind();
        // prepare the operator attributes
        strides_ = anode->get_attr<dims>("strides");
        dilates_ = anode->get_attr<dims>("dilations");
        pads_begin_ = anode->get_attr<dims>("pads_begin");
        pads_end_ = anode->get_attr<dims>("pads_end");
        groups_ = anode->get_attr<int64_t>("groups");
        data_format_ = anode->get_attr<std::string>("data_format");
        filter_format_ = anode->get_attr<std::string>("filter_format");
        impl::logical_tensor_t src_lt = inputs.at(conv::kSrc);
        impl::logical_tensor_t weight_lt = inputs.at(conv::kWeight);
        impl::logical_tensor_t dst_lt = outputs.at(conv::kDst);

        // "NXC"
        if (data_format_ == "NXC") {
            src_lt = logical_tensor_wrapper(&inputs.at(conv::kSrc))
                             .reorder_data_dims_strides();
            dst_lt = impl::logical_tensor_wrapper(&outputs.at(conv::kDst))
                             .reorder_data_dims_strides();
        }
        // "XIO"
        if (filter_format_ == "XIO") {
            weight_lt = impl::logical_tensor_wrapper(&inputs.at(conv::kWeight))
                                .reorder_weight_dims_strides();
        }

        // Check fused post_ops
        with_sum_ = convolution_op_set::fuse_sum(conv_kind);
        with_eltwise_ = convolution_op_set::fuse_eltwise(conv_kind);
        with_bn_ = convolution_op_set::fuse_batchnorm(conv_kind);
        // A convolution operator has bias if
        // - Op name has bias
        // - W/O fused batchnorm and post-op sum, there are 3 inputs:
        //  conv_src, conv_wei, *conv_bias*
        // - W/ fused batchnorm and W/O post-op sum, there are 7 inputs:
        //  conv_src, conv_wei, *conv_bias*, bn_scale, bn_shift, bn_mean, bn_var
        // - W/O fused batachnorm and W/ post-op sum, there are 4 inputs:
        //  conv_src, conv_wei, *conv_bias*, post-src
        // - W/ fused batchnorm and W/ post-op sum, there are 8 inputs:
        //  conv_src, conv_wei, *conv_bias*, bn_scale, bn_shift, bn_mean, bn_var
        //  , post-src
        with_bias_ = convolution_op_set::with_bias(conv_kind)
                || (!with_bn_ && !with_sum_ && inputs.size() == 3)
                || (with_bn_ && !with_sum_ && inputs.size() == 7)
                || (!with_bn_ && with_sum_ && inputs.size() == 4)
                || (with_bn_ && with_sum_ && inputs.size() == 8);

        // set attrs of eltwise
        if (anode->has_attr("alpha")) {
            alpha_ = anode->get_attr<float>("alpha");
        } else if (anode->has_attr("min")) {
            alpha_ = anode->get_attr<float>("min");
        }

        if (anode->has_attr("beta")) {
            beta_ = anode->get_attr<float>("beta");
        } else if (anode->has_attr("max")) {
            beta_ = anode->get_attr<float>("max");
        }

        // the bn inputs offset (if exist)
        if (with_bn_) bn_input_offset_ = with_bias_ ? 3 : 2;

        // prepare the inputs and outputs tensors' descs
        const desc src {src_lt};
        const desc weight {weight_lt};

        // if with_bias, we use the bias_desc directly, otherwise
        // if with_bn, we use bn's shift_desc as the bias_desc
        const desc bias = with_bias_
                ? desc {inputs.at(conv::kBias)}
                : (with_bn_ ? desc {inputs.at(bn_input_offset_ + conv::kShift)}
                            : desc {});
        const desc dst {dst_lt};

        if (with_bn_) epsilon_ = anode->get_attr<float>("epsilon");

        // append post_ops to attrs
        if (with_sum_) {
            attr_ = attr_t::fuse_sum();
            if (with_eltwise_) {
                attr_ = attr_t::residual(
                        get_eltwise_algo(conv_kind), 1.f, 1.f, alpha_, beta_);
            }
        } else if (with_eltwise_) {
            attr_ = attr_t::fuse_eltwise(
                    get_eltwise_algo(conv_kind), 1.f, alpha_, beta_);
        }

        auto eng = engine_manager::get()->get_engine(*aengine);
#define CONV_GET_CONFIG(with_bias, engine, kind, alpha, beta) \
    get_config<with_bias>(src, weight, bias, dst, strides_, dilates_, \
            pads_begin_, pads_end_, groups_, engine, attr_, \
            algorithm::convolution_direct, prop_kind::forward_inference, kind, \
            alpha, beta)

        // if with_bn, we also need to create ptimitive with bias,
        // because bn's shift will act as bias in execution
        params_ = (with_bias_ || with_bn_)
                ? CONV_GET_CONFIG(true, *eng, conv_kind, alpha_, beta_)
                : CONV_GET_CONFIG(false, *eng, conv_kind, alpha_, beta_);
#undef CONV_GET_CONFIG

        const dnnl::convolution_forward::primitive_desc &pd = params_.pd;
        const tensor::desc optimal_dst_desc {pd.dst_desc()};
        // fill_layout_info for not-copied input/outputs
        impl::logical_tensor_t *ori_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(conv::kDst));
        fill_layout_info(ori_dst_lt, optimal_dst_desc);

        // TODO(wuxun): for prepacking, temporarily skip when `with_bn_` is True
        // need to think about how to implement bn folding outside, maybe then
        // we can also remove `with_bn_` flag.
        if (!with_bn_) {
            const tensor::desc optimal_weight_desc {pd.weights_desc()};
            impl::logical_tensor_t *ori_weight_lt
                    = const_cast<impl::logical_tensor_t *>(
                            &inputs.at(conv::kWeight));
            fill_layout_info(ori_weight_lt, optimal_weight_desc);
            if (with_bias_) {
                const tensor::desc optimal_bias_desc {pd.bias_desc()};
                impl::logical_tensor_t *ori_bias_lt
                        = const_cast<impl::logical_tensor_t *>(
                                &inputs.at(conv::kBias));
                fill_layout_info(ori_bias_lt, optimal_bias_desc);
            }
        }
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        const op_kind_t conv_kind = anode->get_op_kind();
        auto &src_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv::kSrc).get_logical_tensor());
        auto &weight_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv::kWeight).get_logical_tensor());
        auto &dst_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(conv::kDst).get_logical_tensor());
        const impl::tensor &impl_bias
                = with_bias_ ? inputs.at(conv::kBias) : impl::tensor {};

        impl::logical_tensor_t post_src_lt {};
        if (with_sum_) {
            post_src_lt = const_cast<impl::logical_tensor_t &>(
                    inputs.back().get_logical_tensor());
        }

        auto &pd = params_.pd;
        auto scratchpad = params_.scratchpad;
        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));

        // "NXC"
        if (data_format_ == "NXC") {
            src_lt = impl::logical_tensor_wrapper(src_lt)
                             .reorder_data_dims_strides();
            dst_lt = impl::logical_tensor_wrapper(dst_lt)
                             .reorder_data_dims_strides();
            if (with_sum_) {
                post_src_lt = impl::logical_tensor_wrapper(post_src_lt)
                                      .reorder_data_dims_strides();
            }
        }
        // "XIO"
        if (filter_format_ == "XIO") {
            weight_lt = impl::logical_tensor_wrapper(weight_lt)
                                .reorder_weight_dims_strides();
        }

        const tensor src {
                src_lt, inputs.at(conv::kSrc).get_data_handle(), *eng};
        const tensor weight {
                weight_lt, inputs.at(conv::kWeight).get_data_handle(), *eng};
        const tensor post_src = with_sum_
                ? tensor {post_src_lt, inputs.back().get_data_handle(), *eng}
                : tensor {};
        const tensor bias = with_bias_ ? tensor {impl_bias, *eng} : tensor {};
        tensor dst {dst_lt, outputs.at(conv::kDst).get_data_handle(), *eng};

        if (expected_weights_.is_empty() || !enable_cache_data_) {
            if (with_bn_) {
                const tensor bn_scale {
                        inputs.at(bn_input_offset_ + conv::kScale), *eng};
                const tensor bn_shift {
                        inputs.at(bn_input_offset_ + conv::kShift), *eng};
                const tensor bn_mean {
                        inputs.at(bn_input_offset_ + conv::kMean), *eng};
                const tensor bn_var {
                        inputs.at(bn_input_offset_ + conv::kVariance), *eng};

                if (updated_weights_.is_empty()) {
                    updated_weights_ = tensor {weight.get_desc(), *eng};
                    updated_bias_ = tensor {bn_shift.get_desc(), *eng};
                }

                bn_fusion::folding(&updated_weights_, &updated_bias_, weight,
                        bias, bn_mean, bn_var, bn_scale, bn_shift, epsilon_,
                        *eng);

                if (updated_weights_.get_desc()
                        != pd.weights_desc()) { //need to reorder
                    if (expected_weights_.is_empty()) {
                        expected_weights_ = tensor {pd.weights_desc(), *eng};
                    }
                    updated_weights_.make_grouped_weights(params_.groups)
                            .reorder_to(expected_weights_);
                } else {
                    expected_weights_ = updated_weights_.make_grouped_weights(
                            params_.groups);
                }

                if (updated_bias_.get_desc() != pd.bias_desc()) {
                    if (expected_bias_.is_empty()) {
                        expected_bias_ = tensor {pd.bias_desc(), *eng};
                    }
                    updated_bias_.reorder_to(expected_bias_);
                } else {
                    expected_bias_ = updated_bias_;
                }
            } else if (with_bias_) {
                if (bias.get_desc() != pd.bias_desc()) {
                    if (expected_bias_.is_empty()) {
                        expected_bias_ = tensor {pd.bias_desc(), *eng};
                    }
                    bias.reorder_to(expected_bias_);
                } else {
                    expected_bias_ = bias;
                }
            }

            if (!with_bn_) {
                if (weight.get_desc() != pd.weights_desc()) { //need to reorder
                    if (expected_weights_.is_empty()) {
                        expected_weights_ = tensor {pd.weights_desc(), *eng};
                    }
                    weight.make_grouped_weights(params_.groups)
                            .reorder_to(expected_weights_);
                } else {
                    expected_weights_
                            = weight.make_grouped_weights(params_.groups);
                }
            }
        }

        if (src.get_desc() != pd.src_desc()) {
            if (expected_src_.is_empty()) {
                expected_src_ = tensor {pd.src_desc(), *eng};
            }
            src.reorder_to(expected_src_);
        } else {
            expected_src_ = src;
        }

        if (dst.get_desc() != pd.dst_desc()) {
            if (expected_dst_.is_empty()) {
                expected_dst_ = tensor {pd.dst_desc(), *eng};
            }
        } else
            expected_dst_ = dst;

        if (with_sum_
                && post_src.get_data_handle()
                        != expected_dst_.get_data_handle()) {
            post_src.reorder_to(expected_dst_);
        }

        stream s(*eng);
        if (with_bias_ || with_bn_) {
            super(pd).execute(s,
                    {{DNNL_ARG_SRC, expected_src_},
                            {DNNL_ARG_WEIGHTS, expected_weights_},
                            {DNNL_ARG_BIAS, expected_bias_},
                            {DNNL_ARG_DST, expected_dst_},
                            {DNNL_ARG_SCRATCHPAD, scratchpad}});
        } else {
            super(pd).execute(s,
                    {{DNNL_ARG_SRC, expected_src_},
                            {DNNL_ARG_WEIGHTS, expected_weights_},
                            {DNNL_ARG_DST, expected_dst_},
                            {DNNL_ARG_SCRATCHPAD, scratchpad}});
        }
        s.wait();

        if (expected_dst_ != dst) expected_dst_.reorder_to(dst);
        return impl::status::success;
    }

    template <bool with_bias>
    static primitive_desc get_primitive_desc(const tensor::desc &src_desc,
            const tensor::desc &weights_desc, const tensor::desc &bias_desc,
            const tensor::desc &dst_desc, const dims &strides,
            const dims &dilates, const dims &pads_begin, const dims &pads_end,
            const engine &aengine, const attr_t &attr = attr_t(),
            algorithm aalgorithm = algorithm::convolution_direct,
            prop_kind aprop_kind = prop_kind::forward) {
        auto src_desc_any = src_desc.to_format_any();
        auto weights_desc_any = weights_desc.to_format_any();
        auto bias_desc_any
                = with_bias ? bias_desc.to_format_any() : tensor::desc();
        auto dst_desc_any = dst_desc.to_format_any();

        if (with_bias) {
            return primitive_desc(
                    {aprop_kind, aalgorithm, src_desc_any, weights_desc_any,
                            bias_desc_any, dst_desc_any, strides, dilates,
                            pads_begin, pads_end},
                    attr, aengine);
        } else {
            return primitive_desc(
                    {aprop_kind, aalgorithm, src_desc_any, weights_desc_any,
                            dst_desc_any, strides, dilates, pads_begin,
                            pads_end},
                    attr, aengine);
        }
    }

private:
    static algorithm get_eltwise_algo(op_kind_t kind) {
        switch (kind) {
            case op_kind::conv_add_relu:
            case op_kind::conv_bias_add_relu:
            case op_kind::conv_bias_bn_add_relu:
            case op_kind::conv_bias_bn_relu:
            case op_kind::conv_bn_add_relu:
            case op_kind::conv_bn_relu:
            case op_kind::conv_bias_relu:
            case op_kind::conv_relu: return (algorithm::eltwise_relu);

            case op_kind::conv_bias_add_elu:
            case op_kind::conv_bias_elu: return (algorithm::eltwise_elu);

            case op_kind::conv_bias_add_relu6:
            case op_kind::conv_bias_relu6:
            case op_kind::conv_bias_hardtanh: return (algorithm::eltwise_clip);

            case op_kind::conv_bias_abs: return (algorithm::eltwise_abs);
            case op_kind::conv_bias_sigmoid:
                return (algorithm::eltwise_logistic);
            case op_kind::conv_bias_sqrt: return (algorithm::eltwise_sqrt);
            case op_kind::conv_bias_square: return (algorithm::eltwise_square);
            case op_kind::conv_bias_tanh: return (algorithm::eltwise_tanh);

            default: BACKEND_DNNL_ENFORCE(0, "Unsupported fused_eltwise op.");
        }
    }

    static std::vector<int64_t> get_output_dims(const tensor &x,
            const tensor &kernel, const dims &stride, const dims &pads_begin,
            const dims &pads_end, const dims &dilations, int64_t groups) {
        std::vector<int64_t> kernel_size(static_cast<size_t>(x.ndims()));
        auto dilations_ = utils::get_compatible_dilates(dilations);
        if (kernel.ndims() == x.ndims() + 1) {
            BACKEND_DNNL_ENFORCE(groups > 1,
                    "Only group dnnl_conv2d weights could have been "
                    "reordered to 5d");
            kernel_size[0] = kernel.get_dim(0) * kernel.get_dim(1);
            std::copy_n(kernel.get_dims().cbegin() + 2, x.ndims() - 1,
                    kernel_size.begin() + 1);
        } else {
            std::copy_n(
                    kernel.get_dims().cbegin(), x.ndims(), kernel_size.begin());
        }

        const memory::dims x_dims = x.get_dims();
        std::vector<int64_t> input_dims {x_dims.cbegin(), x_dims.cend()};
        auto dim = input_dims.size();
        std::vector<int64_t> output_dims(dim);
        output_dims[0] = input_dims[0];
        output_dims[1] = kernel_size[0];
        for (size_t d = 2; d < dim; ++d) {
            auto kernel = (dilations_[d - 2] + 1) * (kernel_size[d] - 1) + 1;
            output_dims[d]
                    = (input_dims[d] + (pads_begin[d - 2] + pads_end[d - 2])
                              - kernel)
                            / stride[d - 2]
                    + 1;
        }
        return output_dims;
    }

    template <bool with_bias>
    static convolution_forward_params get_config(const tensor::desc &src,
            const tensor::desc &weights, const tensor::desc &bias,
            const tensor::desc &dst, const dims &strides, const dims &dilates,
            const dims &pads_begin, const dims &pads_end, const int64_t groups,
            const engine &aengine, const attr_t &attr = attr_t(),
            const algorithm aalgorithm = algorithm::convolution_direct,
            const prop_kind aprop_kind = prop_kind::forward,
            const op_kind_t kind = op_kind::Convolution,
            const float alpha = 0.f, const float beta = 0.f) {
        tensor::desc src_desc, weights_desc, bias_desc;
        attr_t op_attr;

        // make weights and dilates compatible with oneDNN
        const tensor::desc group_weights
                = groups <= 1 ? weights : weights.to_grouped(groups);
        auto com_dilates = utils::get_compatible_dilates(dilates);

        op_attr = attr;

        BACKEND_DNNL_ENFORCE(
                utils::one_of(weights.get_data_type(), data_type::f32,
                        data_type::f16, data_type::bf16),
                "Incorrect data type in weights");

        src_desc = src.to_format_any();
        weights_desc = group_weights.to_format_any();

        if (with_bias) {
            BACKEND_DNNL_ENFORCE(
                    utils::one_of(bias.get_data_type(), data_type::f32,
                            data_type::f16, data_type::bf16),
                    "Incorrect data type in bias");
            bias_desc = bias;
        }

        op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        auto pd = get_primitive_desc<with_bias>(src_desc, weights_desc,
                bias_desc, dst, strides, com_dilates, pads_begin, pads_end,
                aengine, op_attr, aalgorithm, aprop_kind);

        // allocate scratchpad
        tensor scratchpad(pd.scratchpad_desc(), aengine);

        return {pd, groups, scratchpad};
    }

    impl::status_t prepare_inplace_pairs_impl(const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(aengine);
        if (with_sum_) {
            size_t input_idx = with_bias_ ? conv::kBias + 1 : conv::kWeight + 1;
            if (with_bn_) input_idx = bn_input_offset_ + conv::kVariance + 1;
            constexpr size_t output_idx = 0;
            if (logical_tensor_wrapper(inputs[input_idx]).is_opaque()
                    && logical_tensor_wrapper(outputs[output_idx]).is_opaque())
                inplace_pairs_.push_back(
                        {inputs[input_idx].id, outputs[output_idx].id});
        }
        return impl::status::success;
    }
};

struct convolution_backward_data : public dnnl::convolution_backward_data,
                                   public kernel_base {
    using super = dnnl::convolution_backward_data;
    using dims_t = std::vector<llga::impl::dim_t>;

private:
    dims strides_;
    dims dilates_;
    dims pads_begin_;
    dims pads_end_;
    int64_t groups_;

    primitive_desc pd_;

public:
    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // update shape
        std::string data_format = anode->get_attr<std::string>("data_format");
        std::string filter_format
                = anode->get_attr<std::string>("filter_format");
        impl::logical_tensor_t diff_src_lt
                = outputs.at(conv_bwd_data::kDiffsrc);
        impl::logical_tensor_t weight_lt = inputs.at(conv::kWeight);
        impl::logical_tensor_t diff_dst_lt = inputs.at(conv_bwd_data::kDiffdst);

        // "NXC"
        if (data_format == "NXC") {
            diff_dst_lt = logical_tensor_wrapper(
                    &inputs.at(conv_bwd_data::kDiffdst))
                                  .reorder_data_dims_strides();
            diff_src_lt = impl::logical_tensor_wrapper(
                    &outputs.at(conv_bwd_data::kDiffsrc))
                                  .reorder_data_dims_strides();
        }
        // "XIO"
        if (filter_format == "XIO") {
            weight_lt = impl::logical_tensor_wrapper(&inputs.at(conv::kWeight))
                                .reorder_weight_dims_strides();
        }

        const op_kind_t conv_kind = anode->get_op_kind();

        const desc diff_dst_desc {diff_dst_lt};

        const desc weights_desc {weight_lt};

        const desc diff_src_desc {diff_src_lt};

        // cache operator attributes
        strides_ = anode->get_attr<dims>("strides");
        dilates_ = anode->get_attr<dims>("dilations");

        // make dilation aligned with oneDNN
        dilates_ = utils::get_compatible_dilates(dilates_);

        pads_begin_ = anode->get_attr<dims>("pads_begin");
        pads_end_ = anode->get_attr<dims>("pads_end");
        groups_ = anode->get_attr<int64_t>("groups");

        auto eng = engine_manager::get()->get_engine(*aengine);

        const auto diff_src_desc_any = diff_src_desc.to_format_any();

        const auto weights_desc_any = groups_ <= 1
                ? weights_desc.to_format_any()
                : weights_desc.to_grouped(groups_).to_format_any();

        const auto diff_dst_desc_any = diff_dst_desc.to_format_any();

        auto forward_hints = convolution_forward::get_primitive_desc<
                /*with_bias=*/false>(diff_src_desc_any, weights_desc_any,
                tensor::desc(), diff_dst_desc_any, strides_, dilates_,
                pads_begin_, pads_end_, *eng, attr_t(),
                algorithm::convolution_direct, prop_kind::forward_training);

        pd_ = primitive_desc(
                {algorithm::convolution_direct, diff_src_desc_any,
                        weights_desc_any, diff_dst_desc_any, strides_, dilates_,
                        pads_begin_, pads_end_},
                *eng, forward_hints);

        const desc optimal_diff_src_desc {pd_.diff_src_desc()};
        impl::logical_tensor_t *ori_diff_src_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(conv::kDst));
        fill_layout_info(ori_diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        const op_kind_t conv_kind = anode->get_op_kind();
        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));

        auto &weight_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_data::kWeight).get_logical_tensor());
        auto &diff_dst_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_data::kDiffdst).get_logical_tensor());
        auto &diff_src_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(conv_bwd_data::kDiffsrc).get_logical_tensor());
        // "NXC"
        if (anode->get_attr<std::string>("data_format") == "NXC") {
            diff_dst_lt = impl::logical_tensor_wrapper(diff_dst_lt)
                                  .reorder_data_dims_strides();
            diff_src_lt = impl::logical_tensor_wrapper(diff_src_lt)
                                  .reorder_data_dims_strides();
        }
        // "XIO"
        if (anode->get_attr<std::string>("filter_format") == "XIO") {
            weight_lt = impl::logical_tensor_wrapper(weight_lt)
                                .reorder_data_dims_strides();
        }

        const auto &diff_src_dims
                = impl::logical_tensor_wrapper(diff_src_lt).vdims();

        const tensor diff_dst {diff_dst_lt,
                inputs.at(conv_bwd_data::kDiffdst).get_data_handle(), *eng};
        const tensor weights {weight_lt,
                inputs.at(conv_bwd_data::kWeight).get_data_handle(), *eng};
        tensor diff_src {diff_src_lt,
                outputs.at(conv_bwd_data::kDiffsrc).get_data_handle(), *eng};
        compute(diff_dst, weights, diff_src_dims, diff_src, *eng);
        return impl::status::success;
    }

private:
    void compute(const tensor &diff_dst, const tensor &weights,
            const dims &diff_src_dims, tensor &diff_src,
            const engine &aengine) {
        // make weights and dilates compatible with DNNL
        auto weights_ = weights.make_grouped_weights(groups_);

        auto diff_dst_desc = diff_dst.get_desc().to_format_any();
        // align weight data type with diff_dst for bf16
        auto weights_desc = weights_.get_desc().to_format_any().to_type(
                diff_dst.get_data_type());

        auto diff_src_desc = tensor::desc(
                diff_src_dims, diff_dst_desc.get_data_type(), tag::any);

        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(pd_.diff_dst_desc());
        auto expected_weights
                = weights_.reorder_if_differ_in(pd_.weights_desc());
        tensor expected_diff_src = diff_src;
        if (pd_.diff_src_desc() != diff_src.get_desc()) {
            expected_diff_src = tensor {pd_.diff_src_desc(), aengine};
        }

        stream s(aengine);
        super(pd_).execute(s,
                {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                        {DNNL_ARG_WEIGHTS, expected_weights},
                        {DNNL_ARG_DIFF_SRC, expected_diff_src}});
        s.wait();

        if (expected_diff_src != diff_src) {
            dnnl::reorder(expected_diff_src, diff_src)
                    .execute(s, expected_diff_src, diff_src);
            s.wait();
        }
    }
};

struct convolution_backward_weights : public dnnl::convolution_backward_weights,
                                      public kernel_base {
    using super = dnnl::convolution_backward_weights;
    using dims_t = std::vector<llga::impl::dim_t>;

private:
    dims strides_;
    dims dilates_;
    dims pads_begin_;
    dims pads_end_;
    int64_t groups_;

    bool with_diff_bias_;

    primitive_desc pd_;

public:
    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        const op_kind_t conv_kind = anode->get_op_kind();

        // update shape
        std::string data_format = anode->get_attr<std::string>("data_format");
        std::string filter_format
                = anode->get_attr<std::string>("filter_format");

        impl::logical_tensor_t src_lt = inputs.at(conv_bwd_filter::kSrc);
        impl::logical_tensor_t diff_weight_lt
                = outputs.at(conv_bwd_filter::kDiffweight);
        impl::logical_tensor_t diff_dst_lt
                = inputs.at(conv_bwd_filter::kDiffdst);

        // "NXC"
        if (data_format == "NXC") {
            diff_dst_lt = logical_tensor_wrapper(
                    &inputs.at(conv_bwd_filter::kDiffdst))
                                  .reorder_data_dims_strides();
            src_lt = impl::logical_tensor_wrapper(
                    &inputs.at(conv_bwd_filter::kSrc))
                             .reorder_data_dims_strides();
        }
        // "XIO"
        if (filter_format == "XIO") {
            diff_weight_lt = impl::logical_tensor_wrapper(
                    &outputs.at(conv_bwd_filter::kDiffweight))
                                     .reorder_weight_dims_strides();
        }

        const desc src_desc {src_lt};
        const desc diff_dst_desc {diff_dst_lt};
        const desc diff_weights_desc {diff_weight_lt};

        with_diff_bias_ = convolution_op_set::with_bias(conv_kind);

        impl::logical_tensor_t *diff_bias_lt = nullptr;
        if (with_diff_bias_) {
            diff_bias_lt = const_cast<impl::logical_tensor_t *>(
                    &outputs.at(conv_bwd_filter::kDiffbias));
        }

        // cache operator attributes
        strides_ = anode->get_attr<dims>("strides");
        dilates_ = anode->get_attr<dims>("dilations");

        // make dilates compatible with oneDNN
        dilates_ = utils::get_compatible_dilates(dilates_);

        pads_begin_ = anode->get_attr<dims>("pads_begin");
        pads_end_ = anode->get_attr<dims>("pads_end");
        groups_ = anode->get_attr<int64_t>("groups");

        auto eng = engine_manager::get()->get_engine(*aengine);

        const auto src_desc_any = src_desc.to_format_any();
        const auto diff_dst_desc_any = diff_dst_desc.to_format_any();
        const auto diff_weights_desc_any = groups_ <= 1
                ? diff_weights_desc.to_format_any()
                : diff_weights_desc.to_grouped(groups_).to_format_any();

        // for forward hint, weights_desc should have same data_type
        // with other input desc, expect for bias_desc
        auto weights_desc = diff_weights_desc;
        auto diff_weight_type_in = diff_weights_desc.get_data_type();
        auto diff_dst_type = diff_dst_desc.get_data_type();
        if (diff_weight_type_in != diff_dst_type) {
            weights_desc = weights_desc.to_type(diff_dst_type);
        }

        const desc diff_bias_desc = desc(
                {diff_dst_desc.get_dim(1)}, diff_weight_type_in, tag::any);

        if (with_diff_bias_) {
            auto forward_hints = convolution_forward::get_primitive_desc<
                    /*with_diff_bias=*/true>(src_desc_any, weights_desc,
                    diff_bias_desc, diff_dst_desc_any, strides_, dilates_,
                    pads_begin_, pads_end_, *eng, attr_t(),
                    algorithm::convolution_direct, prop_kind::forward_training);

            pd_ = primitive_desc({algorithm::convolution_direct, src_desc_any,
                                         diff_weights_desc_any, diff_bias_desc,
                                         diff_dst_desc_any, strides_, dilates_,
                                         pads_begin_, pads_end_},
                    *eng, forward_hints);
        } else {
            auto forward_hints = convolution_forward::get_primitive_desc<
                    /*with_diff_bias=*/false>(src_desc_any, weights_desc,
                    diff_bias_desc, diff_dst_desc_any, strides_, dilates_,
                    pads_begin_, pads_end_, *eng, attr_t(),
                    algorithm::convolution_direct, prop_kind::forward_training);

            pd_ = primitive_desc(
                    {algorithm::convolution_direct, src_desc_any,
                            diff_weights_desc_any, diff_dst_desc_any, strides_,
                            dilates_, pads_begin_, pads_end_},
                    *eng, forward_hints);
        }

        const desc optimal_diff_weights_desc {pd_.diff_weights_desc()};
        auto ori_diff_weights_lt = const_cast<impl::logical_tensor_t *>(
                &outputs.at(conv_bwd_filter::kDiffweight));
        fill_layout_info(ori_diff_weights_lt, optimal_diff_weights_desc);
        if (with_diff_bias_) {
            const desc optimal_diff_bias_desc {pd_.diff_bias_desc()};
            fill_layout_info(diff_bias_lt, optimal_diff_bias_desc);
        }
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        const op_kind_t conv_kind = anode->get_op_kind();
        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));

        auto &src_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_filter::kSrc).get_logical_tensor());
        auto &diff_dst_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_filter::kDiffdst).get_logical_tensor());
        auto &diff_weights_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(conv_bwd_filter::kDiffweight).get_logical_tensor());

        auto diff_bias = with_diff_bias_ ? const_cast<impl::tensor &>(
                                 outputs.at(conv_bwd_filter::kDiffbias))
                                         : impl::tensor {};

        // "NXC"
        if (anode->get_attr<std::string>("data_format") == "NXC") {
            diff_dst_lt = impl::logical_tensor_wrapper(diff_dst_lt)
                                  .reorder_data_dims_strides();
            src_lt = impl::logical_tensor_wrapper(src_lt)
                             .reorder_data_dims_strides();
        }
        // "XIO"
        if (anode->get_attr<std::string>("filter_format") == "XIO") {
            diff_weights_lt = impl::logical_tensor_wrapper(diff_weights_lt)
                                      .reorder_data_dims_strides();
        }
        const tensor src {src_lt,
                inputs.at(conv_bwd_filter::kSrc).get_data_handle(), *eng};
        const tensor diff_dst {diff_dst_lt,
                inputs.at(conv_bwd_filter::kDiffdst).get_data_handle(), *eng};
        tensor diff_weights {diff_weights_lt,
                outputs.at(conv_bwd_filter::kDiffweight).get_data_handle(),
                *eng};
        tensor diff_b = with_diff_bias_ ? tensor {diff_bias, *eng} : tensor {};

        auto diff_weights_dims
                = impl::logical_tensor_wrapper(diff_weights_lt).vdims();
        conv_backward_weights_impl(
                src, diff_dst, diff_weights, diff_b, diff_weights_dims, *eng);
        return impl::status::success;
    }

    static void compute(const tensor &src, const tensor &diff_dst,
            const dims &diff_weights_dims, tensor &diff_weights,
            tensor &diff_bias, const dims &strides, const dims &dilates,
            const dims &pads_begin, const dims &pads_end, const int64_t groups,
            const engine &aengine,
            const data_type diff_weight_type = data_type::undef,
            algorithm aalgorithm = algorithm::convolution_direct) {
        compute_impl</*with_diff_bias=*/true>(src, diff_dst, diff_weights_dims,
                diff_weights, diff_bias, strides, dilates, pads_begin, pads_end,
                groups, diff_weight_type, aalgorithm, aengine);
    }

    static void compute(const tensor &src, const tensor &diff_dst,
            const dims &diff_weights_dims, tensor &diff_weights,
            const dims &strides, const dims &dilates, const dims &pads_begin,
            const dims &pads_end, const int64_t groups, const engine &aengine,
            const data_type diff_weight_type = data_type::undef,
            algorithm aalgorithm = algorithm::convolution_direct) {
        static tensor dummy_diff_bias;
        compute_impl</*with_diff_bias=*/false>(src, diff_dst, diff_weights_dims,
                diff_weights, dummy_diff_bias, strides, dilates, pads_begin,
                pads_end, groups, diff_weight_type, aalgorithm, aengine);
    }

private:
    void conv_backward_weights_impl(const tensor &src, const tensor &diff_dst,
            tensor &diff_weights, tensor &diff_bias,
            const std::vector<dim_t> &diff_weights_dims, const engine &eng) {
        if (with_diff_bias_) {
            compute_impl</*with_diff_bias=*/true>(src, diff_dst,
                    diff_weights_dims, diff_weights, diff_bias, eng);
        } else {
            compute_impl</*with_diff_bias=*/false>(src, diff_dst,
                    diff_weights_dims, diff_weights, diff_bias, eng);
        }
    }

    template <bool with_diff_bias>
    void compute_impl(const tensor &src, const tensor &diff_dst,
            const dims &diff_weights_dims, tensor &diff_weights,
            tensor &diff_bias, const engine &aengine) {
        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(pd_.diff_dst_desc());
        auto expected_src = src.reorder_if_differ_in(pd_.src_desc());
        // embed group info into diff_weights_desc
        auto expected_diff_weights_desc
                = tensor::desc(pd_.diff_weights_desc(), groups_);

        tensor expected_diff_weights = diff_weights;
        if (pd_.diff_weights_desc() != diff_weights.get_desc()) {
            expected_diff_weights
                    = tensor {expected_diff_weights_desc, aengine};
        }
        tensor expected_diff_bias = diff_bias;

        stream s(aengine);
        if (with_diff_bias_) {
            if (pd_.diff_bias_desc() != diff_bias.get_desc()) {
                expected_diff_bias = tensor {pd_.diff_bias_desc(), aengine};
            }
            super(pd_).execute(s,
                    {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                            {DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
                            {DNNL_ARG_DIFF_BIAS, expected_diff_bias}});
        } else {
            super(pd_).execute(s,
                    {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                            {DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights}});
        }
        s.wait();

        if (expected_diff_weights != diff_weights) {
            dnnl::reorder(expected_diff_weights, diff_weights)
                    .execute(s, expected_diff_weights, diff_weights);
            s.wait();
        }

        if (with_diff_bias_ && expected_diff_bias != diff_bias) {
            dnnl::reorder(expected_diff_bias, diff_bias)
                    .execute(s, expected_diff_bias, diff_bias);
            s.wait();
        }
    }

    template <bool with_diff_bias>
    static void compute_impl(const tensor &src, const tensor &diff_dst,
            const dims &diff_weights_dims, tensor &diff_weights,
            tensor &diff_bias, const dims &strides, const dims &dilates,
            const dims &pads_begin, const dims &pads_end, const int64_t groups,
            const data_type diff_weight_type, algorithm aalgorithm,
            const engine &aengine) {
        // make diff_weights and dilates compatible with DNNL
        auto comp_dilates = utils::get_compatible_dilates(dilates);
        data_type diff_dst_type = diff_dst.get_data_type();
        data_type diff_weight_type_in = data_type::undef == diff_weight_type
                ? diff_dst_type
                : diff_weight_type;

        auto diff_weights_desc = tensor::desc(
                diff_weights_dims, diff_weight_type_in, tag::any);
        if (groups > 1) {
            diff_weights_desc
                    = diff_weights_desc.to_grouped(groups).to_format_any();
        }

        auto diff_dst_desc = diff_dst.get_desc().to_format_any();
        auto src_desc = src.get_desc().to_format_any();

        auto diff_bias_desc = tensor::desc(
                {diff_dst.get_dim(1)}, diff_weight_type_in, tag::any);

        // for forward hint, weights_desc should have same data_type
        // with other input desc, expect for bias_desc
        auto weights_desc = diff_weights_desc;
        if (diff_weight_type_in != diff_dst_type) {
            weights_desc = weights_desc.to_type(diff_dst_type);
        }
        auto forward_hints
                = convolution_forward::get_primitive_desc<with_diff_bias>(
                        src_desc, weights_desc, diff_bias_desc, diff_dst_desc,
                        strides, comp_dilates, pads_begin, pads_end, aengine,
                        attr_t(), aalgorithm, prop_kind::forward);

        auto pd = with_diff_bias
                ? primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                                         diff_bias_desc, diff_dst_desc, strides,
                                         comp_dilates, pads_begin, pads_end},
                        aengine, forward_hints)
                : primitive_desc(
                        {aalgorithm, src_desc, diff_weights_desc, diff_dst_desc,
                                strides, comp_dilates, pads_begin, pads_end},
                        aengine, forward_hints);

        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
        auto expected_src = src.reorder_if_differ_in(pd.src_desc());
        // embed group info into diff_weights_desc
        auto expected_diff_weights_desc
                = tensor::desc(pd.diff_weights_desc(), groups);
        diff_weights.reinit_if_possible(expected_diff_weights_desc);

        stream s(aengine);
        if (with_diff_bias) {
            diff_bias.reinit_if_possible(pd.diff_bias_desc());
            super(pd).execute(s,
                    {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                            {DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_DIFF_WEIGHTS, diff_weights},
                            {DNNL_ARG_DIFF_BIAS, diff_bias}});
        } else {
            super(pd).execute(s,
                    {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                            {DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_DIFF_WEIGHTS, diff_weights}});
        }
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif
