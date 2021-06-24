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

#ifndef BACKEND_DNNL_OPERATORS_CONV_HPP
#define BACKEND_DNNL_OPERATORS_CONV_HPP

#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/legacy.hpp"

#include "bn_fusion.hpp"

namespace dnnl {
namespace graph {
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
                op_kind::conv_bias_bn_add_relu, op_kind::conv_bias_swish,
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
     * Check if convolution operator fuses add
     *
     * @param kind operator kind
     * @return whether the operator fused add
     */
    static bool fuse_add(op_kind_t kind) {
        static const std::set<op_kind_t> with_add_set {op_kind::conv_add,
                op_kind::conv_add_elu, op_kind::conv_add_relu,
                op_kind::conv_add_relu6, op_kind::conv_bias_add,
                op_kind::conv_bias_add_elu, op_kind::conv_bias_add_relu,
                op_kind::conv_bias_add_relu6, op_kind::conv_bias_bn_add,
                op_kind::conv_bias_bn_add_relu, op_kind::conv_bn_add,
                op_kind::conv_bn_add_relu};
        return with_add_set.count(kind);
    }

    /**
     * Check if convolution operator fuses activation relu
     *
     * @param kind operator kind
     * @return whether the operator fused activation relu
     */
    static bool fuse_eltwise(op_kind_t kind) {
        static const std::set<op_kind_t> with_eltwise_set {
                op_kind::conv_add_elu,
                op_kind::conv_add_relu,
                op_kind::conv_add_relu6,
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
                op_kind::conv_bias_swish,
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
    using dims_t = std::vector<dnnl::graph::impl::dim_t>;

private:
    primitive_desc pd_;
    dnnl::convolution_forward prim_;

    dims strides_;
    dims dilates_;
    dims pads_begin_;
    dims pads_end_;
    int64_t groups_;
    std::string data_format_, filter_format_;

    memory expected_src_;
    memory expected_weights_;
    memory expected_bias_;
    memory expected_dst_;

    memory updated_weights_;
    memory updated_bias_;

    memory scratchpad_;
    memory::desc scratchpad_desc_;

    memory::desc cvt_src_desc_;
    memory::desc cvt_weights_desc_;
    memory::desc cvt_bias_desc_;
    memory::desc cvt_dst_desc_;
    memory::desc cvt_post_src_desc_;

    memory::desc opt_src_desc_;
    memory::desc opt_weights_desc_;
    memory::desc opt_bias_desc_;
    memory::desc opt_dst_desc_;

    memory cvt_src_;
    memory cvt_weight_;
    memory cvt_bias_;
    memory cvt_dst_;
    memory cvt_post_src_;

    dnnl::engine p_engine_;
    impl::allocator_t *alc_ {nullptr};

    std::vector<void *> internal_buffers_;

    attr_t attr_;

    bool with_bias_;
    bool with_add_;
    bool with_post_sum_;
    bool with_post_binary_add_;
    bool with_eltwise_;
    bool with_bn_;

    float alpha_ = 0.f;
    float beta_ = 0.f;

    size_t bn_input_offset_;

    bool channel_first_ {true};

    float epsilon_; // bn epsilon

    bool first_iteration_ {true};

    exec_args conv_args_;

    dnnl::reorder::primitive_desc src_reorder_pd_;
    dnnl::reorder::primitive_desc weight_reorder_pd_;
    dnnl::reorder::primitive_desc dst_reorder_pd_;
    dnnl::reorder::primitive_desc bias_reorder_pd_;

    // FIXME(qun) NOT well designed
    /// \note Currently we don't have enough information from framework to
    /// decide cache or not. Also we think that caching data in a library
    /// is not safe at this moment.
    char *val = std::getenv("DNNL_GRAPH_WEIGHT_CACHE");
    bool enable_cache_data_ = (val != nullptr && std::strcmp(val, "1") == 0);

public:
    virtual ~convolution_forward() {
        for (auto &buf : internal_buffers_) {
            allocator::free(buf, p_engine_, alc_);
        }
    }

    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        const op_kind_t conv_kind = op->get_kind();
        // prepare the operator attributes
        strides_ = op->get_attr<dims>("strides");
        dilates_ = op->get_attr<dims>("dilations");
        pads_begin_ = op->get_attr<dims>("pads_begin");
        pads_end_ = op->get_attr<dims>("pads_end");
        groups_ = op->get_attr<int64_t>("groups");
        data_format_ = op->get_attr<std::string>("data_format");
        filter_format_ = op->get_attr<std::string>("filter_format");

        // Check fused post_ops
        with_add_ = convolution_op_set::fuse_add(conv_kind);
        with_eltwise_ = convolution_op_set::fuse_eltwise(conv_kind);
        with_bn_ = convolution_op_set::fuse_batchnorm(conv_kind);
        // A convolution operator has bias if
        // - Op name has bias
        // - W/O fused batchnorm and post-op add, there are 3 inputs:
        //  conv_src, conv_wei, *conv_bias*
        // - W/ fused batchnorm and W/O post-op add, there are 7 inputs:
        //  conv_src, conv_wei, *conv_bias*, bn_scale, bn_shift, bn_mean, bn_var
        // - W/O fused batachnorm and W/ post-op add, there are 4 inputs:
        //  conv_src, conv_wei, *conv_bias*, post-src
        // - W/ fused batchnorm and W/ post-op add, there are 8 inputs:
        //  conv_src, conv_wei, *conv_bias*, bn_scale, bn_shift, bn_mean, bn_var
        //  , post-src
        with_bias_ = convolution_op_set::with_bias(conv_kind)
                || (!with_bn_ && !with_add_ && inputs.size() == 3)
                || (with_bn_ && !with_add_ && inputs.size() == 7)
                || (!with_bn_ && with_add_ && inputs.size() == 4)
                || (with_bn_ && with_add_ && inputs.size() == 8);

        // set attrs of eltwise
        if (op->has_attr("alpha")) {
            alpha_ = op->get_attr<float>("alpha");
        } else if (op->has_attr("min")) {
            alpha_ = op->get_attr<float>("min");
        }
        // special handle for swish, spec doesn't support setting attr of alpha
        // in sigmoid op (swish op is formed by sigmoid and multiply ops)
        if (conv_kind == op_kind::conv_bias_swish) { alpha_ = 1.f; }

        if (op->has_attr("beta")) {
            beta_ = op->get_attr<float>("beta");
        } else if (op->has_attr("max")) {
            beta_ = op->get_attr<float>("max");
        }

        // the bn inputs offset (if exist)
        if (with_bn_) {
            bn_input_offset_ = with_bias_ ? 3 : 2;
            epsilon_ = op->get_attr<float>("epsilon");
        }

        // append post_ops to attrs
        if (with_add_) {
            impl::logical_tensor_t post_src_lt = inputs.back();
            impl::logical_tensor_t dst_lt = outputs.at(conv::kDst);
            if (impl::logical_tensor_wrapper(post_src_lt)
                            .has_same_shape_as(dst_lt)) {
                // if post src has the same shape of dst
                // set post sum attribute
                attr_ = attr_t::fuse_sum();
                if (with_eltwise_) {
                    attr_ = attr_t::residual(get_eltwise_algo(conv_kind), 1.f,
                            1.f, alpha_, beta_);
                }
                with_post_sum_ = true;
            } else {
                const logical_tensor_wrapper dst_lt_wrapper(dst_lt);
                int dst_lt_ndims = dst_lt_wrapper.ndims();
                memory::desc post_src = make_dnnl_memory_desc(post_src_lt);
                post_src = expand(post_src, dst_lt_ndims);
                // post binary only supports per tensor and per channel
                // broadcast, which means the expand shape of post src should
                // be all one or the post_src_dim[c_axis]==dst_dim[c_axis]
                int c_axis = (data_format_ == "NXC") ? (dst_lt_ndims - 1) : 1;
                for (int i = dst_lt_ndims - 1; i >= 0; i--) {
                    if (post_src.dims()[i] == 1) continue;

                    if (i != c_axis
                            || dst_lt_wrapper.dims()[i] != post_src.dims()[i]) {
                        return impl::status::compile_fail;
                    }
                }
                attr_ = attr_t::fuse_binary(post_src, algorithm::binary_add);
                with_post_binary_add_ = true;
            }
        } else if (with_eltwise_) {
            attr_ = attr_t::fuse_eltwise(
                    get_eltwise_algo(conv_kind), 1.f, alpha_, beta_);
        }

        memory::desc src = make_dnnl_memory_desc(inputs.at(conv::kSrc));
        memory::desc weight = make_dnnl_memory_desc(inputs.at(conv::kWeight));
        memory::desc dst = make_dnnl_memory_desc(outputs.at(conv::kDst));

        // "NXC"
        if (data_format_ == "NXC") {
            src = permute_NXC2NCX(src);
            dst = permute_NXC2NCX(dst);
        }

        // "XIO"
        if (filter_format_ == "XIO") {
            weight = permute_XIO2OIX(weight);
            channel_first_ = false;
        }

        cvt_src_desc_ = src;
        cvt_weights_desc_ = weight;
        cvt_dst_desc_ = dst;

        if (with_add_) {
            cvt_post_src_desc_ = make_dnnl_memory_desc(inputs.back());
            if (data_format_ == "NXC") {
                cvt_post_src_desc_ = permute_NXC2NCX(cvt_post_src_desc_);
            }
            if (with_post_binary_add_) {
                cvt_post_src_desc_
                        = expand(cvt_post_src_desc_, cvt_dst_desc_.data.ndims);
            }
        }

        // if with_bias, we use the bias_desc directly, otherwise
        // if with_bn, we use bn's shift_desc as the bias_desc
        memory::desc bias = with_bias_
                ? make_dnnl_memory_desc(inputs.at(conv::kBias))
                : (with_bn_ ? make_dnnl_memory_desc(
                           inputs.at(bn_input_offset_ + conv::kShift))
                            : memory::desc {});

        cvt_bias_desc_ = bias;

        p_engine_ = make_dnnl_engine(*g_engine);

#define CONV_GET_CONFIG(with_bias, p_engine) \
    get_config<with_bias>(src, weight, bias, dst, strides_, dilates_, \
            pads_begin_, pads_end_, groups_, p_engine, attr_, \
            algorithm::convolution_direct, prop_kind::forward)

        // if with_bn, we also need to create ptimitive with bias,
        // because bn's shift will act as bias in execution
        pd_ = (with_bias_ || with_bn_) ? CONV_GET_CONFIG(true, p_engine_)
                                       : CONV_GET_CONFIG(false, p_engine_);
        prim_ = super(pd_);
#undef CONV_GET_CONFIG
        // set this flag = true every time compile is called
        first_iteration_ = true;

        opt_src_desc_ = pd_.src_desc();
        opt_weights_desc_ = pd_.weights_desc();
        opt_dst_desc_ = pd_.dst_desc();
        if (with_bias_ || with_bn_) { opt_bias_desc_ = pd_.bias_desc(); }
        scratchpad_desc_ = pd_.scratchpad_desc();

        if (groups_ > 1)
            cvt_weights_desc_ = to_grouped(cvt_weights_desc_, groups_);

        if (impl::logical_tensor_wrapper(inputs.at(conv::kSrc)).is_any())
            cvt_src_desc_ = pd_.src_desc();
        if (impl::logical_tensor_wrapper(inputs.at(conv::kWeight)).is_any())
            cvt_weights_desc_ = pd_.weights_desc();
        if (impl::logical_tensor_wrapper(outputs.at(conv::kDst)).is_any()) {
            cvt_dst_desc_ = pd_.dst_desc();
        }
        if ((with_bias_ || with_bn_)
                && impl::logical_tensor_wrapper(inputs.at(conv::kBias))
                           .is_any()) {
            cvt_bias_desc_ = pd_.bias_desc();
        }

        // fill_layout_info for not-copied input/outputs
        impl::logical_tensor_t *ori_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(conv::kDst));
        if (data_format_ == "NXC") {
            memory::desc tmp = permute_NCX2NXC(pd_.dst_desc());
            fill_layout_info(ori_dst_lt, tmp);
        } else {
            fill_layout_info(ori_dst_lt, pd_.dst_desc());
        }

        // TODO(wuxun): for prepacking, temporarily skip when `with_bn_` is True
        // need to think about how to implement bn folding outside, maybe then
        // we can also remove `with_bn_` flag.
        if (!with_bn_) {
            impl::logical_tensor_t *ori_weight_lt
                    = const_cast<impl::logical_tensor_t *>(
                            &inputs.at(conv::kWeight));
            if (filter_format_ == "XIO") {
                memory::desc tmp = permute_OIX2XIO(pd_.weights_desc());
                fill_layout_info(ori_weight_lt, tmp);
            } else {
                fill_layout_info(ori_weight_lt, pd_.weights_desc());
            }
            if (with_bias_) {
                impl::logical_tensor_t *ori_bias_lt
                        = const_cast<impl::logical_tensor_t *>(
                                &inputs.at(conv::kBias));
                fill_layout_info(ori_bias_lt, pd_.bias_desc());
            }
        }
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);

        if (first_iteration_) {
            alc_ = g_stream->get_engine()->get_allocator();
        }

        // bn folding
        if (with_bn_ && (first_iteration_ || !enable_cache_data_)) {
            const memory weight
                    = make_dnnl_memory(inputs.at(conv::kWeight), p_engine_);
            const memory bias = with_bias_
                    ? make_dnnl_memory(inputs.at(conv::kBias), p_engine_)
                    : memory();
            const memory bn_scale = make_dnnl_memory(
                    inputs.at(bn_input_offset_ + conv::kScale), p_engine_);
            const memory bn_shift = make_dnnl_memory(
                    inputs.at(bn_input_offset_ + conv::kShift), p_engine_);
            const memory bn_mean = make_dnnl_memory(
                    inputs.at(bn_input_offset_ + conv::kMean), p_engine_);
            const memory bn_var = make_dnnl_memory(
                    inputs.at(bn_input_offset_ + conv::kVariance), p_engine_);

            if (first_iteration_) {
                internal_buffers_.emplace_back(allocator::malloc(
                        weight.get_desc().get_size(), p_engine_, alc_));
                updated_weights_ = make_dnnl_memory(
                        weight.get_desc(), p_engine_, internal_buffers_.back());

                internal_buffers_.emplace_back(allocator::malloc(
                        bn_shift.get_desc().get_size(), p_engine_, alc_));
                updated_bias_ = make_dnnl_memory(bn_shift.get_desc(), p_engine_,
                        internal_buffers_.back());
            }

            bn_fusion::folding(&updated_weights_, &updated_bias_, weight, bias,
                    bn_mean, bn_var, bn_scale, bn_shift, epsilon_, *g_stream);
        }

        // only create memory object in first iteration
        if (first_iteration_) {
            cvt_src_ = make_dnnl_memory(cvt_src_desc_, p_engine_, nullptr);
            cvt_weight_
                    = make_dnnl_memory(cvt_weights_desc_, p_engine_, nullptr);
            cvt_bias_ = make_dnnl_memory(cvt_bias_desc_, p_engine_, nullptr);
            cvt_dst_ = make_dnnl_memory(cvt_dst_desc_, p_engine_, nullptr);
            cvt_post_src_
                    = make_dnnl_memory(cvt_post_src_desc_, p_engine_, nullptr);
        }

        // modify the data handle by using given buffer in every iteration
        cvt_src_.set_data_handle(inputs.at(conv::kSrc).get_data_handle());
        cvt_dst_.set_data_handle(outputs.at(conv::kDst).get_data_handle());
        if (with_bn_) {
            cvt_weight_.set_data_handle(updated_weights_.get_data_handle());
            cvt_bias_.set_data_handle(updated_bias_.get_data_handle());
        } else {
            cvt_weight_.set_data_handle(
                    inputs.at(conv::kWeight).get_data_handle());
            if (with_bias_) {
                cvt_bias_.set_data_handle(
                        inputs.at(conv::kBias).get_data_handle());
            }
        }
        if (with_add_)
            cvt_post_src_.set_data_handle(inputs.back().get_data_handle());

        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        // allocate buffer for internal memory (optimal layout memory),
        // and reorder given inputs into these internal memory
        if (cvt_weights_desc_ != opt_weights_desc_) {
            if (first_iteration_) {
                internal_buffers_.emplace_back(allocator::malloc(
                        opt_weights_desc_.get_size(), p_engine_, alc_));
                expected_weights_ = make_dnnl_memory(
                        opt_weights_desc_, p_engine_, internal_buffers_.back());
                weight_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                        cvt_weights_desc_, p_engine_, opt_weights_desc_);
            }
            if (first_iteration_ || !enable_cache_data_)
                dnnl::reorder(weight_reorder_pd_)
                        .execute(p_stream, cvt_weight_, expected_weights_);
        } else {
            if (first_iteration_) expected_weights_ = cvt_weight_;
            expected_weights_.set_data_handle(cvt_weight_.get_data_handle());
        }

        if (with_bias_ || with_bn_) {
            if (cvt_bias_desc_ != opt_bias_desc_) {
                if (first_iteration_) {
                    internal_buffers_.emplace_back(allocator::malloc(
                            opt_bias_desc_.get_size(), p_engine_, alc_));
                    expected_bias_ = make_dnnl_memory(opt_bias_desc_, p_engine_,
                            internal_buffers_.back());
                    bias_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                            cvt_bias_desc_, p_engine_, opt_bias_desc_);
                }

                if (first_iteration_ || !enable_cache_data_)
                    dnnl::reorder(bias_reorder_pd_)
                            .execute(p_stream, cvt_bias_, expected_bias_);
            } else {
                if (first_iteration_) expected_bias_ = cvt_bias_;
                expected_bias_.set_data_handle(cvt_bias_.get_data_handle());
            }
        }

        if (cvt_src_desc_ != opt_src_desc_) {
            if (first_iteration_) {
                internal_buffers_.emplace_back(allocator::malloc(
                        opt_src_desc_.get_size(), p_engine_, alc_));
                expected_src_ = make_dnnl_memory(
                        opt_src_desc_, p_engine_, internal_buffers_.back());
                src_reorder_pd_ = dnnl::reorder::primitive_desc(
                        p_engine_, cvt_src_desc_, p_engine_, opt_src_desc_);
            }

            dnnl::reorder(src_reorder_pd_)
                    .execute(p_stream, cvt_src_, expected_src_);
        } else {
            if (first_iteration_) expected_src_ = cvt_src_;
            expected_src_.set_data_handle(cvt_src_.get_data_handle());
        }

        // allocate buffer for optimal output
        if (cvt_dst_desc_ != opt_dst_desc_) {
            if (first_iteration_) {
                internal_buffers_.emplace_back(allocator::malloc(
                        opt_dst_desc_.get_size(), p_engine_, alc_));
                expected_dst_ = make_dnnl_memory(
                        opt_dst_desc_, p_engine_, internal_buffers_.back());
                dst_reorder_pd_ = dnnl::reorder::primitive_desc(
                        p_engine_, opt_dst_desc_, p_engine_, cvt_dst_desc_);
            }
        } else {
            if (first_iteration_) expected_dst_ = cvt_dst_;
            expected_dst_.set_data_handle(cvt_dst_.get_data_handle());
        }

        // reorder the post_src to optimal output if they are not inplaced
        if (with_post_sum_
                && cvt_post_src_.get_data_handle()
                        != expected_dst_.get_data_handle()) {
            dnnl::reorder(cvt_post_src_, expected_dst_)
                    .execute(p_stream, cvt_post_src_, expected_dst_);
        }

        if (first_iteration_) {
            internal_buffers_.emplace_back(allocator::malloc(
                    scratchpad_desc_.get_size(), p_engine_, alc_));
            scratchpad_ = make_dnnl_memory(
                    scratchpad_desc_, p_engine_, internal_buffers_.back());

            conv_args_ = {{DNNL_ARG_SRC, expected_src_},
                    {DNNL_ARG_WEIGHTS, expected_weights_},
                    {DNNL_ARG_BIAS, expected_bias_},
                    {DNNL_ARG_DST, expected_dst_},
                    {DNNL_ARG_SCRATCHPAD, scratchpad_}};
            if (with_post_binary_add_) {
                conv_args_.insert(
                        {(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1),
                                cvt_post_src_});
            }
        }

        prim_.execute(p_stream, conv_args_);
        // reorder optimal output to given output buffer
        if (expected_dst_ != cvt_dst_) {
            dnnl::reorder(dst_reorder_pd_)
                    .execute(p_stream, expected_dst_, cvt_dst_);
        }

        first_iteration_ = false;
        return impl::status::success;
    }

    template <bool with_bias>
    static primitive_desc get_primitive_desc(const tensor::desc &src_desc,
            const tensor::desc &weights_desc, const tensor::desc &bias_desc,
            const tensor::desc &dst_desc, const dims &strides,
            const dims &dilates, const dims &pads_begin, const dims &pads_end,
            const dnnl::engine &p_engine, const attr_t &attr = attr_t(),
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
                    attr, p_engine);
        } else {
            return primitive_desc(
                    {aprop_kind, aalgorithm, src_desc_any, weights_desc_any,
                            dst_desc_any, strides, dilates, pads_begin,
                            pads_end},
                    attr, p_engine);
        }
    }

private:
    static algorithm get_eltwise_algo(op_kind_t kind) {
        switch (static_cast<int>(kind)) {
            case op_kind::conv_add_relu:
            case op_kind::conv_bias_add_relu:
            case op_kind::conv_bias_bn_add_relu:
            case op_kind::conv_bias_bn_relu:
            case op_kind::conv_bn_add_relu:
            case op_kind::conv_bn_relu:
            case op_kind::conv_bias_relu:
            case op_kind::conv_relu: return (algorithm::eltwise_relu);

            case op_kind::conv_add_elu:
            case op_kind::conv_bias_add_elu:
            case op_kind::conv_bias_elu: return (algorithm::eltwise_elu);

            case op_kind::conv_add_relu6:
            case op_kind::conv_bias_add_relu6:
            case op_kind::conv_bias_relu6:
            case op_kind::conv_bias_hardtanh: return (algorithm::eltwise_clip);

            case op_kind::conv_bias_abs: return (algorithm::eltwise_abs);
            case op_kind::conv_bias_sigmoid:
                return (algorithm::eltwise_logistic);
            case op_kind::conv_bias_sqrt: return (algorithm::eltwise_sqrt);
            case op_kind::conv_bias_square: return (algorithm::eltwise_square);
            case op_kind::conv_bias_tanh: return (algorithm::eltwise_tanh);
            case op_kind::conv_bias_swish: return (algorithm::eltwise_swish);

            default: BACKEND_DNNL_ENFORCE(0, "Unsupported fused_eltwise op.");
        }
        return algorithm::undef;
    }

    template <bool with_bias>
    static primitive_desc get_config(const memory::desc &src,
            const memory::desc &weights, const memory::desc &bias,
            const memory::desc &dst, const dims &strides, const dims &dilates,
            const dims &pads_begin, const dims &pads_end, const int64_t groups,
            const dnnl::engine &p_engine, const attr_t &attr = attr_t(),
            const algorithm aalgorithm = algorithm::convolution_direct,
            const prop_kind aprop_kind = prop_kind::forward) {
        // make weights and dilates compatible with oneDNN
        const memory::desc group_weights
                = groups <= 1 ? weights : to_grouped(weights, groups);
        auto com_dilates = get_compatible_dilates(dilates);

        BACKEND_DNNL_ENFORCE(
                impl::utils::one_of(weights.data_type(), data_type::f32,
                        data_type::f16, data_type::bf16),
                "Incorrect data type in weights");

        memory::desc src_desc = to_format_any(src);
        memory::desc weights_desc = to_format_any(group_weights);
        memory::desc dst_desc = to_format_any(dst);

        attr_t op_attr = attr;
        op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        primitive_desc pd;
        if (with_bias) {
            memory::desc bias_desc = bias;
            BACKEND_DNNL_ENFORCE(
                    impl::utils::one_of(bias.data_type(), data_type::f32,
                            data_type::f16, data_type::bf16),
                    "Incorrect data type in bias");
            pd = primitive_desc({aprop_kind, aalgorithm, src_desc, weights_desc,
                                        bias_desc, dst_desc, strides,
                                        com_dilates, pads_begin, pads_end},
                    op_attr, p_engine);
        } else {
            pd = primitive_desc(
                    {aprop_kind, aalgorithm, src_desc, weights_desc, dst_desc,
                            strides, com_dilates, pads_begin, pads_end},
                    op_attr, p_engine);
        }

        return pd;
    }

    impl::status_t prepare_inplace_pairs_impl(const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(g_engine);
        if (with_post_sum_) {
            size_t input_idx = with_bias_ ? conv::kBias + 1 : conv::kWeight + 1;
            if (with_bn_) input_idx = bn_input_offset_ + conv::kVariance + 1;
            constexpr size_t output_idx = 0;

            const logical_tensor_wrapper post_src_lt(inputs[input_idx]);
            const logical_tensor_wrapper dst_lt(outputs[output_idx]);
            if (post_src_lt.is_opaque() && dst_lt.is_opaque()
                    && post_src_lt.is_similar(dst_lt))
                inplace_pairs_.push_back(
                        {inputs[input_idx].id, outputs[output_idx].id});
        }
        return impl::status::success;
    }
};

struct convolution_backward_data : public dnnl::convolution_backward_data,
                                   public kernel_base {
    using super = dnnl::convolution_backward_data;
    using dims_t = std::vector<dnnl::graph::impl::dim_t>;

private:
    dims strides_;
    dims dilates_;
    dims pads_begin_;
    dims pads_end_;
    int64_t groups_;

    primitive_desc pd_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // update shape
        std::string data_format = op->get_attr<std::string>("data_format");
        std::string filter_format = op->get_attr<std::string>("filter_format");
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

        const desc diff_dst_desc {diff_dst_lt};

        const desc weights_desc {weight_lt};

        const desc diff_src_desc {diff_src_lt};

        // cache operator attributes
        strides_ = op->get_attr<dims>("strides");
        dilates_ = op->get_attr<dims>("dilations");

        // make dilation aligned with oneDNN
        dilates_ = get_compatible_dilates(dilates_);

        pads_begin_ = op->get_attr<dims>("pads_begin");
        pads_end_ = op->get_attr<dims>("pads_end");
        groups_ = op->get_attr<int64_t>("groups");

        p_engine_ = make_dnnl_engine(*g_engine);

        const auto diff_src_desc_any = diff_src_desc.to_format_any();

        const auto weights_desc_any = groups_ <= 1
                ? weights_desc.to_format_any()
                : weights_desc.to_grouped(groups_).to_format_any();

        const auto diff_dst_desc_any = diff_dst_desc.to_format_any();

        auto forward_hints = convolution_forward::get_primitive_desc<
                /*with_bias=*/false>(diff_src_desc_any, weights_desc_any,
                tensor::desc(), diff_dst_desc_any, strides_, dilates_,
                pads_begin_, pads_end_, p_engine_, attr_t(),
                algorithm::convolution_direct, prop_kind::forward_training);

        pd_ = primitive_desc(
                {algorithm::convolution_direct, diff_src_desc_any,
                        weights_desc_any, diff_dst_desc_any, strides_, dilates_,
                        pads_begin_, pads_end_},
                p_engine_, forward_hints);

        const desc optimal_diff_src_desc {pd_.diff_src_desc()};
        impl::logical_tensor_t *ori_diff_src_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(conv::kDst));
        fill_layout_info(ori_diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        auto &weight_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_data::kWeight).get_logical_tensor());
        auto &diff_dst_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_data::kDiffdst).get_logical_tensor());
        auto &diff_src_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(conv_bwd_data::kDiffsrc).get_logical_tensor());
        // "NXC"
        if (op->get_attr<std::string>("data_format") == "NXC") {
            diff_dst_lt = impl::logical_tensor_wrapper(diff_dst_lt)
                                  .reorder_data_dims_strides();
            diff_src_lt = impl::logical_tensor_wrapper(diff_src_lt)
                                  .reorder_data_dims_strides();
        }
        // "XIO"
        if (op->get_attr<std::string>("filter_format") == "XIO") {
            weight_lt = impl::logical_tensor_wrapper(weight_lt)
                                .reorder_data_dims_strides();
        }

        const tensor diff_dst {diff_dst_lt, p_engine_, alc,
                inputs.at(conv_bwd_data::kDiffdst).get_data_handle()};
        const tensor weights {weight_lt, p_engine_, alc,
                inputs.at(conv_bwd_data::kWeight).get_data_handle()};
        tensor diff_src {diff_src_lt, p_engine_, alc,
                outputs.at(conv_bwd_data::kDiffsrc).get_data_handle()};
        compute(diff_dst, weights, diff_src, p_engine_, alc, p_stream_);
        return impl::status::success;
    }

private:
    void compute(const tensor &diff_dst, const tensor &weights,
            tensor &diff_src, const dnnl::engine &p_engine,
            impl::allocator_t *alc, const dnnl::stream &p_stream) {
        // make weights and dilates compatible with DNNL
        auto weights_ = weights.make_grouped_weights(groups_);

        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(p_stream, pd_.diff_dst_desc());
        auto expected_weights
                = weights_.reorder_if_differ_in(p_stream, pd_.weights_desc());
        tensor expected_diff_src = diff_src;
        if (pd_.diff_src_desc() != diff_src.get_desc()) {
            expected_diff_src = tensor {pd_.diff_src_desc(), p_engine, alc};
        }

        super(pd_).execute(p_stream,
                {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                        {DNNL_ARG_WEIGHTS, expected_weights},
                        {DNNL_ARG_DIFF_SRC, expected_diff_src}});

        if (expected_diff_src != diff_src) {
            dnnl::reorder(expected_diff_src, diff_src)
                    .execute(p_stream, expected_diff_src, diff_src);
        }
    }
};

struct convolution_backward_weights : public dnnl::convolution_backward_weights,
                                      public kernel_base {
    using super = dnnl::convolution_backward_weights;
    using dims_t = std::vector<dnnl::graph::impl::dim_t>;

private:
    dims strides_;
    dims dilates_;
    dims pads_begin_;
    dims pads_end_;
    int64_t groups_;

    bool with_diff_bias_;

    primitive_desc pd_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        const op_kind_t conv_kind = op->get_kind();

        // update shape
        std::string data_format = op->get_attr<std::string>("data_format");
        std::string filter_format = op->get_attr<std::string>("filter_format");

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
        strides_ = op->get_attr<dims>("strides");
        dilates_ = op->get_attr<dims>("dilations");

        // make dilates compatible with oneDNN
        dilates_ = get_compatible_dilates(dilates_);

        pads_begin_ = op->get_attr<dims>("pads_begin");
        pads_end_ = op->get_attr<dims>("pads_end");
        groups_ = op->get_attr<int64_t>("groups");

        p_engine_ = make_dnnl_engine(*g_engine);

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
                    pads_begin_, pads_end_, p_engine_, attr_t(),
                    algorithm::convolution_direct, prop_kind::forward_training);

            pd_ = primitive_desc({algorithm::convolution_direct, src_desc_any,
                                         diff_weights_desc_any, diff_bias_desc,
                                         diff_dst_desc_any, strides_, dilates_,
                                         pads_begin_, pads_end_},
                    p_engine_, forward_hints);
        } else {
            auto forward_hints = convolution_forward::get_primitive_desc<
                    /*with_diff_bias=*/false>(src_desc_any, weights_desc,
                    diff_bias_desc, diff_dst_desc_any, strides_, dilates_,
                    pads_begin_, pads_end_, p_engine_, attr_t(),
                    algorithm::convolution_direct, prop_kind::forward_training);

            pd_ = primitive_desc(
                    {algorithm::convolution_direct, src_desc_any,
                            diff_weights_desc_any, diff_dst_desc_any, strides_,
                            dilates_, pads_begin_, pads_end_},
                    p_engine_, forward_hints);
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

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        auto &src_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_filter::kSrc).get_logical_tensor());
        auto &diff_dst_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_filter::kDiffdst).get_logical_tensor());
        auto &diff_weights_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(conv_bwd_filter::kDiffweight).get_logical_tensor());

        auto diff_bias = with_diff_bias_ ? const_cast<impl::tensor_t &>(
                                 outputs.at(conv_bwd_filter::kDiffbias))
                                         : impl::tensor_t {};

        // "NXC"
        if (op->get_attr<std::string>("data_format") == "NXC") {
            diff_dst_lt = impl::logical_tensor_wrapper(diff_dst_lt)
                                  .reorder_data_dims_strides();
            src_lt = impl::logical_tensor_wrapper(src_lt)
                             .reorder_data_dims_strides();
        }
        // "XIO"
        if (op->get_attr<std::string>("filter_format") == "XIO") {
            diff_weights_lt = impl::logical_tensor_wrapper(diff_weights_lt)
                                      .reorder_data_dims_strides();
        }
        const tensor src {src_lt, p_engine_, alc,
                inputs.at(conv_bwd_filter::kSrc).get_data_handle()};
        const tensor diff_dst {diff_dst_lt, p_engine_, alc,
                inputs.at(conv_bwd_filter::kDiffdst).get_data_handle()};
        tensor diff_weights {diff_weights_lt, p_engine_, alc,
                outputs.at(conv_bwd_filter::kDiffweight).get_data_handle()};
        tensor diff_b = with_diff_bias_ ? tensor {diff_bias, p_engine_, alc}
                                        : tensor {};

        auto diff_weights_dims
                = impl::logical_tensor_wrapper(diff_weights_lt).vdims();
        conv_backward_weights_impl(src, diff_dst, diff_weights, diff_b,
                diff_weights_dims, alc, p_stream_);
        return impl::status::success;
    }

private:
    void conv_backward_weights_impl(const tensor &src, const tensor &diff_dst,
            tensor &diff_weights, tensor &diff_bias,
            const std::vector<dim_t> &diff_weights_dims, impl::allocator_t *alc,
            const dnnl::stream &p_stream) {
        if (with_diff_bias_) {
            compute_impl</*with_diff_bias=*/true>(src, diff_dst,
                    diff_weights_dims, diff_weights, diff_bias, p_engine_, alc,
                    p_stream);
        } else {
            compute_impl</*with_diff_bias=*/false>(src, diff_dst,
                    diff_weights_dims, diff_weights, diff_bias, p_engine_, alc,
                    p_stream);
        }
    }

    template <bool with_diff_bias>
    void compute_impl(const tensor &src, const tensor &diff_dst,
            const dims &diff_weights_dims, tensor &diff_weights,
            tensor &diff_bias, const dnnl::engine &p_engine,
            impl::allocator_t *alc, const dnnl::stream &p_stream) {
        UNUSED(diff_weights_dims);
        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(p_stream, pd_.diff_dst_desc());
        auto expected_src = src.reorder_if_differ_in(p_stream, pd_.src_desc());
        // embed group info into diff_weights_desc
        auto expected_diff_weights_desc
                = tensor::desc(pd_.diff_weights_desc(), groups_);

        tensor expected_diff_weights = diff_weights;
        if (pd_.diff_weights_desc() != diff_weights.get_desc()) {
            expected_diff_weights
                    = tensor {expected_diff_weights_desc, p_engine, alc};
        }
        tensor expected_diff_bias = diff_bias;

        if (with_diff_bias_) {
            if (pd_.diff_bias_desc() != diff_bias.get_desc()) {
                expected_diff_bias
                        = tensor {pd_.diff_bias_desc(), p_engine, alc};
            }
            super(pd_).execute(p_stream,
                    {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                            {DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
                            {DNNL_ARG_DIFF_BIAS, expected_diff_bias}});
        } else {
            super(pd_).execute(p_stream,
                    {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                            {DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights}});
        }

        if (expected_diff_weights != diff_weights) {
            dnnl::reorder(expected_diff_weights, diff_weights)
                    .execute(p_stream, expected_diff_weights, diff_weights);
        }

        if (with_diff_bias_ && expected_diff_bias != diff_bias) {
            dnnl::reorder(expected_diff_bias, diff_bias)
                    .execute(p_stream, expected_diff_bias, diff_bias);
        }
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
