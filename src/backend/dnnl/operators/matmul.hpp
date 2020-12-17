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

#ifndef LLGA_BACKEND_DNNL_OPERATORS_MATMUL_HPP
#define LLGA_BACKEND_DNNL_OPERATORS_MATMUL_HPP

#include <functional>
#include <set>
#include <utility>
#include <vector>

#include "backend/dnnl/tensor.hpp"
#include "bn_fusion.hpp"
#include "common.hpp"

namespace llga {
namespace impl {
namespace dnnl_impl {

namespace matmul_fwd {
enum matmul_inputs { kSrc, kWeight, kBias };
enum fused_bn_inputs { kScale, kShift, kMean, kVariance };
enum matmul_outputs { kDst };
} // namespace matmul_fwd

struct matmul_op_set {
    /**
     * Check if matmul operator has bias add.
     *
     * @param kind operator kind
     * @return whether the operator has bias add
     */
    static bool with_bias(op_kind_t kind) {
        static const std::set<op_kind_t> with_bias_set {op_kind::matmul_bias,
                op_kind::matmul_bias_add, op_kind::matmul_bias_add_relu,
                op_kind::matmul_bias_bn, op_kind::matmul_bias_elu,
                op_kind::matmul_bias_hardtanh, op_kind::matmul_bias_relu6,
                op_kind::matmul_bias_relu, op_kind::matmul_bias_sigmoid};
        return with_bias_set.count(kind);
    }

    /**
     * Check if matmul operator fuses batchnorm
     *
     * @param kind operator kind
     * @return whether the operator fuses batchnorm
     */
    static bool fuse_batchnorm(op_kind_t kind) {
        return kind == op_kind::matmul_bias_bn;
    }

    /**
     * Check if matmul operator fuses sum
     *
     * @param kind operator kind
     * @return whether the operator fused sum
     */

    static bool fuse_sum(op_kind_t kind) {
        static const std::set<op_kind_t> with_sum_set {op_kind::matmul_bias_add,
                op_kind::matmul_bias_add_relu, op_kind::matmul_add,
                op_kind::matmul_add_gelu, op_kind::matmul_add_relu,
                op_kind::matmul_add_sigmoid};
        return with_sum_set.count(kind);
    }

    /**
     * Check if matmul operator fuses activation relu
     *
     * @param kind operator kind
     * @return whether the operator fused activation relu
     */
    static bool fuse_eltwise(op_kind_t kind) {
        static const std::set<op_kind_t> with_eltwise_set {op_kind::matmul_relu,
                op_kind::matmul_elu, op_kind::matmul_hardtanh,
                op_kind::matmul_gelu, op_kind::matmul_sigmoid,
                op_kind::matmul_bias_elu, op_kind::matmul_bias_hardtanh,
                op_kind::matmul_bias_relu6, op_kind::matmul_bias_relu,
                op_kind::matmul_bias_sigmoid, op_kind::matmul_bias_add_relu,
                op_kind::matmul_add_gelu, op_kind::matmul_add_relu,
                op_kind::matmul_add_sigmoid};
        return with_eltwise_set.count(kind);
    }
};

struct matmul_forward : public dnnl::matmul, public kernel_base {
    using super = dnnl::matmul;
    // TODO(qun) the inner product primitive related code will be removed,
    // onece oneDNN issues are fixed. Currently, we need this workaround to
    // improve performance and get correct results on GPU device.
    using ip_super = dnnl::inner_product_forward;
    using ip_primitive_desc = dnnl::inner_product_forward::primitive_desc;

private:
    // cached pd is in this struct
    primitive_desc pd_;
    // cache expected data to avoid creating memory in every iteration
    tensor expected_src_;
    tensor expected_weights_;
    tensor expected_bias_;
    tensor expected_dst_;
    attr_t attr_;

    tensor updated_weights_;
    tensor updated_bias_;

    bool transpose_a_ = false;
    bool transpose_b_ = false;

    bool with_bias_;
    bool with_sum_;
    bool with_eltwise_;
    bool with_bn_;
    op_kind_t kind_;
    float alpha_ = 0.f;
    float beta_ = 0.f;

    size_t bn_input_offset_;

    float epsilon_; // bn epsilon

    bool is_training_;

    // FIXME(qun) NOT well designed
    /// \note Currently we don't have enough information from framework to
    /// decide cache or not. Also we think that caching data in a library
    /// is not safe at this moment.
    bool disable_cache_data_ {true};

    // inner_product primitive_desc
    // used for ndx2d input
    ip_primitive_desc ip_pd_;

    bool is_ndx2d_ {false};

    tensor::desc ori_src_desc_ {};
    tensor::desc ori_weight_desc_ {};
    tensor::desc ori_bias_desc_ {};

public:
    void compute(const engine &aengine) {
        stream s(aengine);
        if (with_bias_) {
            super(pd_).execute(s,
                    {{DNNL_ARG_SRC, expected_src_},
                            {DNNL_ARG_WEIGHTS, expected_weights_},
                            {DNNL_ARG_BIAS, expected_bias_},
                            {DNNL_ARG_DST, expected_dst_}});
        } else {
            super(pd_).execute(s,
                    {{DNNL_ARG_SRC, expected_src_},
                            {DNNL_ARG_WEIGHTS, expected_weights_},
                            {DNNL_ARG_DST, expected_dst_}});
        }
        s.wait();
    }

    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        kind_ = anode->get_op_kind();
        with_bias_ = matmul_op_set::with_bias(kind_);
        with_sum_ = matmul_op_set::fuse_sum(kind_);
        with_eltwise_ = matmul_op_set::fuse_eltwise(kind_);
        with_bn_ = matmul_op_set::fuse_batchnorm(kind_);

        // deal with 1D add
        bool add_1d = (with_bias_ == false) && (with_sum_ == true)
                && (impl::logical_tensor_wrapper(inputs[matmul_fwd::kBias])
                                .ndims()
                        == 1);
        if (add_1d) {
            with_bias_ = true;
            with_sum_ = false;
        }

        // set attrs of eltwise
        if (with_eltwise_) {
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
        }
        // the bn inputs offset (if exist)
        if (with_bn_) bn_input_offset_ = with_bias_ ? 3 : 2;

        // prepare the inputs and outputs tensors' descs
        desc src {inputs.at(matmul_fwd::kSrc)};
        desc weight {inputs.at(matmul_fwd::kWeight)};

        dims old_weights_dims = weight.get_dims();

        //change dims and strides if tensor need to transpose
        if (anode->has_attr("transpose_a"))
            transpose_a_ = anode->get_attr<bool>("transpose_a");
        if (anode->has_attr("transpose_b"))
            transpose_b_ = anode->get_attr<bool>("transpose_b");

        if (transpose_a_ && src.get_ndims() > 1) {
            int ndims = src.get_ndims();
            dims expected_strides = src.get_strides();
            dims expected_dims = src.get_dims();
            std::swap(expected_dims[ndims - 2], expected_dims[ndims - 1]);
            std::swap(expected_strides[ndims - 2], expected_strides[ndims - 1]);
            src = desc {expected_dims, src.get_data_type(), expected_strides};
        }

        if (transpose_b_ && weight.get_ndims() > 1) {
            int ndims = weight.get_ndims();
            dims expected_strides = weight.get_strides();
            dims expected_dims = weight.get_dims();
            std::swap(expected_dims[ndims - 2], expected_dims[ndims - 1]);
            std::swap(expected_strides[ndims - 2], expected_strides[ndims - 1]);
            weight = desc {
                    expected_dims, weight.get_data_type(), expected_strides};
        }

        //if src or weight is 1-D, reshape it into 2-D for oneDNN
        if (src.get_ndims() == 1)
            src = desc {{1, src.get_dim(0)}, src.get_data_type(),
                    {src.get_dim(0), 1}};
        if (weight.get_ndims() == 1)
            weight = desc {
                    {weight.get_dim(0), 1}, weight.get_data_type(), {1, 1}};

        // if with_bias, we use the bias_desc directly, otherwise
        // if with_bn, we use bn's shift_desc as the bias_desc
        desc bias = with_bias_ ? desc {inputs.at(matmul_fwd::kBias)}
                               : (with_bn_ ? desc {inputs.at(bn_input_offset_
                                          + matmul_fwd::kShift)}
                                           : desc {});

        ori_src_desc_ = src;
        ori_weight_desc_ = weight;

        dims old_bias_dims = with_bias_ ? bias.get_dims() : dims {};

        impl::logical_tensor_t *dst_lt = const_cast<impl::logical_tensor_t *>(
                &outputs.at(matmul_fwd::kDst));

        tensor::desc dst(*dst_lt);

        // check the input dimension
        int src_ndims = src.get_ndims();
        int weight_ndims = weight.get_ndims();
        if (src_ndims > 2 && weight_ndims == 2) { is_ndx2d_ = true; }

        //if bias has different dims with src, reshape it
        if (with_bias_) {
            int bias_ndims = bias.get_ndims();
            // inner_product support 1d bias
            if (src_ndims != bias_ndims && !is_ndx2d_) {
                dims expected_dims = src.get_dims();
                dims expected_strides = src.get_strides();
                for (size_t i = 0; i < src_ndims - bias_ndims; ++i)
                    expected_dims[i] = 1;
                for (size_t i = src_ndims - bias_ndims; i < src_ndims; ++i)
                    expected_dims[i] = bias.get_dim(i - src_ndims + bias_ndims);
                expected_strides[src_ndims - 1] = 1;
                for (size_t i = src_ndims - 1; i > 0; --i)
                    expected_strides[i - 1]
                            = expected_strides[i] * expected_dims[i];
                bias = desc {
                        expected_dims, bias.get_data_type(), expected_strides};
            }
        }

        ori_bias_desc_ = bias;
        if (with_bn_) epsilon_ = anode->get_attr<float>("epsilon");

        // append post_ops to attrs
        if (with_sum_) {
            attr_ = attr_t::fuse_sum();
            if (with_eltwise_) {
                attr_ = attr_t::residual(
                        get_eltwise_algo(kind_), 1.f, 1.f, alpha_, beta_);
            }
        } else if (with_eltwise_) {
            attr_ = attr_t::fuse_eltwise(
                    get_eltwise_algo(kind_), 1.f, alpha_, beta_);
        }

        auto eng = engine_manager::get()->get_engine(*aengine);

        if (with_bias_) {
            BACKEND_DNNL_ENFORCE(
                    utils::one_of(bias.get_data_type(), data_type::f32,
                            data_type::f16, data_type::bf16),
                    "Incorrect data type in bias");
            bias = bias.to_format_any();
        }

        dims old_dst_dims = dst.get_dims();
        if (is_ndx2d_) {
            src = src.reshape(flatten_to_2d(src.get_dims()));
            dst = dst.reshape(flatten_to_2d(dst.get_dims()));
            weight = weight.permute_axes({1, 0}); // ip requires [OC, IC]
            prop_kind pkind = prop_kind::forward_inference;
            ip_pd_ = with_bias_
                    ? ip_primitive_desc(
                            {pkind, src, weight, bias, dst}, attr_, *eng)
                    : ip_primitive_desc({pkind, src, weight, dst}, attr_, *eng);
        } else {
            pd_ = with_bias_
                    ? primitive_desc({src, weight, bias, dst}, attr_, *eng)
                    : primitive_desc({src, weight, dst}, attr_, *eng);
        }

        fill_layout_info(dst_lt,
                is_ndx2d_ ? ip_pd_.dst_desc().reshape(old_dst_dims)
                          : pd_.dst_desc());

        // TODO(wuxun): for prepacking, temporarily skip when `with_bn_` is True
        // need to think about how to implement bn folding outside, maybe then
        // we can also remove `with_bn_` flag.
        if (!with_bn_) {
            impl::logical_tensor_t *ori_weight_lt
                    = const_cast<impl::logical_tensor_t *>(
                            &inputs.at(matmul_fwd::kWeight));
            // TODO(wuxun): here, we need to reshape the queried desc to the
            // original shape. However, if there is a broadcast in one dim and
            // DNNL also needs padding in this broadcast dim, reshaping will
            // fail. A possible solution is that in conversion's reorder, we
            // also add check for the broadcast-able dims.
            fill_layout_info(ori_weight_lt,
                    is_ndx2d_ ? ip_pd_.weights_desc()
                                        .permute_axes({1, 0})
                                        .reshape(old_weights_dims)
                              : pd_.weights_desc());
            if (with_bias_) {
                impl::logical_tensor_t *ori_bias_lt
                        = const_cast<impl::logical_tensor_t *>(
                                &inputs.at(matmul_fwd::kBias));
                fill_layout_info(ori_bias_lt,
                        is_ndx2d_ ? ip_pd_.bias_desc()
                                  : pd_.bias_desc().reshape(old_bias_dims));
            }
        }
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));

        auto pd_src_desc = is_ndx2d_ ? ip_pd_.src_desc() : pd_.src_desc();
        auto pd_weights_desc
                = is_ndx2d_ ? ip_pd_.weights_desc() : pd_.weights_desc();
        auto pd_bias_desc = is_ndx2d_ ? ip_pd_.bias_desc() : pd_.bias_desc();
        auto pd_dst_desc = is_ndx2d_ ? ip_pd_.dst_desc() : pd_.dst_desc();

        //create src and weight tensor, and change the desc of them if
        //they have transpose_* attr or ndims = 1
        tensor src {inputs.at(matmul_fwd::kSrc), *eng};
        if (transpose_a_ || src.ndims() == 1) {
            src = tensor {ori_src_desc_,
                    inputs.at(matmul_fwd::kSrc).get_data_handle(), *eng};
        }
        tensor weight {inputs.at(matmul_fwd::kWeight), *eng};
        if (transpose_b_ || weight.ndims() == 1) {
            weight = tensor {ori_weight_desc_,
                    inputs.at(matmul_fwd::kWeight).get_data_handle(), *eng};
        }

        tensor bias = with_bias_ ? tensor {inputs.at(matmul_fwd::kBias), *eng}
                                 : tensor {};

        if (!is_ndx2d_ && with_bias_ && bias.ndims() != src.ndims()) {
            bias = tensor {ori_bias_desc_,
                    inputs.at(matmul_fwd::kBias).get_data_handle(), *eng};
        }

        tensor post_src = with_sum_ ? tensor {inputs.back(), *eng} : tensor {};
        tensor dst {outputs.at(matmul_fwd::kDst), *eng};
        if (with_bn_) {
            const tensor bn_scale {
                    inputs.at(bn_input_offset_ + matmul_fwd::kScale), *eng};
            const tensor bn_shift {
                    inputs.at(bn_input_offset_ + matmul_fwd::kShift), *eng};
            const tensor bn_mean {
                    inputs.at(bn_input_offset_ + matmul_fwd::kMean), *eng};
            const tensor bn_var {
                    inputs.at(bn_input_offset_ + matmul_fwd::kVariance), *eng};

            if (updated_weights_.is_empty()) {
                updated_weights_ = tensor {weight.get_desc(), *eng};
                updated_bias_ = tensor {bn_shift.get_desc(), *eng};
            }

            bn_fusion::folding(&updated_weights_, &updated_bias_, weight, bias,
                    bn_mean, bn_var, bn_scale, bn_shift, epsilon_, *eng);

            if (updated_weights_.get_desc()
                    != pd_weights_desc) { //need to reorder
                if (expected_weights_.is_empty()) {
                    expected_weights_ = tensor {pd_weights_desc, *eng};
                }
                updated_weights_.reorder_to(expected_weights_);
            } else {
                expected_weights_ = updated_weights_;
            }

            if (updated_bias_.get_desc() != pd_bias_desc) {
                if (expected_bias_.is_empty()) {
                    expected_bias_ = tensor {pd_bias_desc, *eng};
                }
                updated_bias_.reorder_to(expected_bias_);
            } else {
                expected_bias_ = updated_bias_;
            }

        } else {
            if (is_ndx2d_) { weight = weight.transpose_(1, 0); }
            if (weight.get_desc() != pd_weights_desc) { //need to reorder
                if (expected_weights_.is_empty()) {
                    expected_weights_ = tensor {pd_weights_desc, *eng};
                }
                weight.reorder_to(expected_weights_);
            } else {
                expected_weights_ = weight;
            }

            if (with_bias_) {
                if (bias.get_desc() != pd_bias_desc) {
                    if (expected_bias_.is_empty()) {
                        expected_bias_ = tensor {pd_bias_desc, *eng};
                    }
                    bias.reorder_to(expected_bias_);
                } else {
                    expected_bias_ = bias;
                }
            }
        }

        if (is_ndx2d_) {
            src = src.reshape(flatten_to_2d(src.get_dims()));
            dst = dst.reshape(flatten_to_2d(dst.get_dims()));
        }

        if (src.get_desc() != pd_src_desc) {
            if (expected_src_.is_empty()) {
                expected_src_ = tensor {pd_src_desc, *eng};
            }
            src.reorder_to(expected_src_);
        } else {
            expected_src_ = src;
        }

        if (dst.get_desc() != pd_dst_desc) {
            if (expected_dst_.is_empty()) {
                expected_dst_ = tensor {pd_dst_desc, *eng};
            }
        } else {
            expected_dst_ = dst;
        }

        if (with_sum_) {
            if (is_ndx2d_)
                post_src = post_src.reshape(flatten_to_2d(post_src.get_dims()));
            post_src.reorder_to(expected_dst_);
        }

        if (is_ndx2d_) {
            stream s(*eng);
            ip_super(ip_pd_).execute(s,
                    {{DNNL_ARG_SRC, expected_src_},
                            {DNNL_ARG_WEIGHTS, expected_weights_},
                            // won't be used if not with_bias_
                            {DNNL_ARG_BIAS, expected_bias_},
                            {DNNL_ARG_DST, expected_dst_}});
            s.wait();
        } else {
            compute(*eng);
        }
        if (expected_dst_ != dst) expected_dst_.reorder_to(dst);
        return impl::status::success;
    }

private:
    // Covert a shape from nd to 2d by flatening the first n-1 dims
    dims flatten_to_2d(const dims &in) {
        assert(in.size() >= 2);
        if (in.size() == 2) return in;
        int64_t batch = std::accumulate(in.begin(), in.end() - 1,
                static_cast<int64_t>(1), std::multiplies<int64_t>());
        return {batch, in.back()};
    }

    algorithm get_eltwise_algo(op_kind_t kind) {
        switch (kind) {
            case op_kind::matmul_relu:
            case op_kind::matmul_bias_relu:
            case op_kind::matmul_bias_add_relu:
            case op_kind::matmul_add_relu: return (algorithm::eltwise_relu);

            case op_kind::matmul_elu:
            case op_kind::matmul_bias_elu: return (algorithm::eltwise_elu);

            case op_kind::matmul_hardtanh:
            case op_kind::matmul_bias_relu6:
            case op_kind::matmul_bias_hardtanh:
                return (algorithm::eltwise_clip);

            case op_kind::matmul_sigmoid:
            case op_kind::matmul_bias_sigmoid:
            case op_kind::matmul_add_sigmoid:
                return (algorithm::eltwise_logistic);

            case op_kind::matmul_gelu:
            case op_kind::matmul_add_gelu: return (algorithm::eltwise_gelu_erf);

            default:
                BACKEND_DNNL_ENFORCE(
                        0, "Unsupported fused_eltwise op for matmul.");
        }
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif
