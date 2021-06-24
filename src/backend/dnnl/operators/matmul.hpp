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

#ifndef BACKEND_DNNL_OPERATORS_MATMUL_HPP
#define BACKEND_DNNL_OPERATORS_MATMUL_HPP

#include <functional>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>

#include "bn_fusion.hpp"

namespace dnnl {
namespace graph {
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
                op_kind::matmul_bias_relu, op_kind::matmul_bias_gelu,
                op_kind::matmul_bias_sigmoid};
        return with_bias_set.count(kind);
    }

    /**
     * Check if matmul operator fuses add
     *
     * @param kind operator kind
     * @return whether the operator fused add
     */

    static bool fuse_add(op_kind_t kind) {
        static const std::set<op_kind_t> with_add_set {op_kind::matmul_bias_add,
                op_kind::matmul_bias_add_relu, op_kind::matmul_add,
                op_kind::matmul_add_gelu, op_kind::matmul_add_relu,
                op_kind::matmul_add_sigmoid};
        return with_add_set.count(kind);
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
                op_kind::matmul_bias_gelu, op_kind::matmul_bias_sigmoid,
                op_kind::matmul_bias_add_relu, op_kind::matmul_add_gelu,
                op_kind::matmul_add_relu, op_kind::matmul_add_sigmoid};
        return with_eltwise_set.count(kind);
    }
};

struct matmul_forward : public dnnl::matmul, public kernel_base {
    using super = dnnl::matmul;

private:
    // cached pd is in this struct
    primitive_desc pd_;
    dnnl::matmul prim_;
    // cache expected data to avoid creating memory in every iteration
    memory expected_src_;
    memory expected_weights_;
    memory expected_bias_;
    memory expected_dst_;
    attr_t attr_;

    memory updated_weights_;
    memory updated_bias_;

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

    bool transpose_a_ {false};
    bool transpose_b_ {false};

    bool with_bias_ {false};
    bool with_add_ {false};
    bool with_post_sum_ {false};
    bool with_post_binary_add_ {false};
    bool with_eltwise_ {false};
    op_kind_t kind_;
    float alpha_ = 0.f;
    float beta_ = 0.f;

    dnnl::engine p_engine_;
    impl::allocator_t *alc_ {nullptr};
    std::vector<void *> internal_buffers_;

    exec_args matmul_args_;
    bool first_iteration_ = true;

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
    virtual ~matmul_forward() {
        for (auto &buf : internal_buffers_) {
            allocator::free(buf, p_engine_, alc_);
        }
    }
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        kind_ = op->get_kind();
        with_add_ = matmul_op_set::fuse_add(kind_);
        with_eltwise_ = matmul_op_set::fuse_eltwise(kind_);
        with_bias_ = matmul_op_set::with_bias(kind_)
                || (!with_add_ && inputs.size() == 3);

        // deal with 1D add
        bool add_1d = (with_bias_ == false) && (with_add_ == true)
                && (impl::logical_tensor_wrapper(inputs[matmul_fwd::kBias])
                                .ndims()
                        == 1);
        if (add_1d) {
            with_bias_ = true;
            with_add_ = false;
        }

        // set attrs of eltwise
        if (with_eltwise_) {
            if (op->has_attr("alpha")) {
                alpha_ = op->get_attr<float>("alpha");
            } else if (op->has_attr("min")) {
                alpha_ = op->get_attr<float>("min");
            }

            if (op->has_attr("beta")) {
                beta_ = op->get_attr<float>("beta");
            } else if (op->has_attr("max")) {
                beta_ = op->get_attr<float>("max");
            }
        }

        // prepare the inputs and outputs tensors' descs
        memory::desc src = make_dnnl_memory_desc(inputs.at(matmul_fwd::kSrc));
        memory::desc weight
                = make_dnnl_memory_desc(inputs.at(matmul_fwd::kWeight));

        //change dims and strides if tensor need to transpose
        if (op->has_attr("transpose_a"))
            transpose_a_ = op->get_attr<bool>("transpose_a");
        if (op->has_attr("transpose_b"))
            transpose_b_ = op->get_attr<bool>("transpose_b");

        if (transpose_a_ && src.dims().size() > 1) {
            const logical_tensor_wrapper src_lt_wrapper(
                    inputs.at(matmul_fwd::kSrc));
            const int ndims = src_lt_wrapper.ndims();
            dims expected_strides = src_lt_wrapper.vstrides();
            dims expected_dims = src_lt_wrapper.vdims();
            const auto last_dim = static_cast<dims::size_type>(ndims - 1);
            std::swap(expected_dims[last_dim - 1], expected_dims[last_dim]);
            std::swap(
                    expected_strides[last_dim - 1], expected_strides[last_dim]);
            src = memory::desc {
                    expected_dims, src.data_type(), expected_strides};
        }

        if (transpose_b_ && weight.dims().size() > 1) {
            const logical_tensor_wrapper weight_lt_wrapper(
                    inputs.at(matmul_fwd::kWeight));
            const int ndims = weight_lt_wrapper.ndims();
            dims expected_strides = weight_lt_wrapper.vstrides();
            dims expected_dims = weight_lt_wrapper.vdims();
            const auto last_dim = static_cast<dims::size_type>(ndims - 1);
            std::swap(expected_dims[last_dim - 1], expected_dims[last_dim]);
            std::swap(
                    expected_strides[last_dim - 1], expected_strides[last_dim]);
            weight = memory::desc {
                    expected_dims, weight.data_type(), expected_strides};
        }

        //if src or weight is 1-D, reshape it into 2-D for oneDNN
        if (src.dims().size() == 1)
            src = memory::desc {
                    {1, src.dims()[0]}, src.data_type(), {src.dims()[0], 1}};
        if (weight.dims().size() == 1)
            weight = memory::desc {
                    {weight.dims()[0], 1}, weight.data_type(), {1, 1}};

        // if with_bias, we use the bias_desc directly, otherwise
        memory::desc bias = with_bias_
                ? make_dnnl_memory_desc(inputs.at(matmul_fwd::kBias))
                : memory::desc {};
        memory::desc dst = make_dnnl_memory_desc(outputs.at(matmul_fwd::kDst));

        // check the input dimension
        const int src_ndims = static_cast<int>(src.dims().size());
        const int weight_ndims = static_cast<int>(weight.dims().size());
        const int bias_ndims = static_cast<int>(bias.dims().size());
        const int dst_ndims = static_cast<int>(dst.dims().size());
        const dims old_bias_dims = with_bias_ ? bias.dims() : dims {};

        // expand src or weight for broadcast
        if (src_ndims != weight_ndims) {
            if (src_ndims > weight_ndims) {
                weight = expand(weight, src_ndims);
            } else {
                src = expand(src, weight_ndims);
            }
        }

        // if bias has different dims with dst, expand
        if (with_bias_ && bias_ndims != dst_ndims) {
            bias = expand(bias, dst_ndims);
        }

        // append post_ops to attrs
        if (with_add_) {
            impl::logical_tensor_t post_src_lt = inputs.back();
            memory::desc post_src = make_dnnl_memory_desc(post_src_lt);
            const impl::logical_tensor_t dst_lt = outputs.at(matmul_fwd::kDst);
            if (impl::logical_tensor_wrapper(post_src_lt)
                            .has_same_shape_as(dst_lt)) {
                // if post src has the same shape of dst
                // set post sum attribute
                attr_ = attr_t::fuse_sum();
                if (with_eltwise_) {
                    attr_ = attr_t::residual(
                            get_eltwise_algo(kind_), 1.f, 1.f, alpha_, beta_);
                }
                with_post_sum_ = true;
            } else {
                const logical_tensor_wrapper dst_lt_wrapper(dst_lt);
                const int dst_lt_ndims = dst_lt_wrapper.ndims();
                post_src = expand(post_src, dst_lt_ndims);
                // post binary only supports per tensor and per channel
                // broadcast, which means the expand shape of post src should
                // be all one or the post_src_dim[1]==dst_dim[1]
                for (int i = dst_lt_ndims - 1; i >= 0; i--) {
                    if (post_src.dims()[i] == 1) continue;

                    if (i != 1 || dst.dims()[i] != post_src.dims()[i]) {
                        return impl::status::compile_fail;
                    }
                }

                attr_ = attr_t::fuse_binary(post_src, algorithm::binary_add);
                with_post_binary_add_ = true;
            }
            cvt_post_src_desc_ = post_src;
        } else if (with_eltwise_) {
            attr_ = attr_t::fuse_eltwise(
                    get_eltwise_algo(kind_), 1.f, alpha_, beta_);
        }

        cvt_src_desc_ = src;
        cvt_weights_desc_ = weight;
        cvt_bias_desc_ = bias;
        cvt_dst_desc_ = dst;

        p_engine_ = make_dnnl_engine(*g_engine);

        if (with_bias_) {
            BACKEND_DNNL_ENFORCE(
                    impl::utils::one_of(bias.data_type(), data_type::f32,
                            data_type::f16, data_type::bf16),
                    "Incorrect data type in bias");
        }

        pd_ = with_bias_
                ? primitive_desc({src, weight, bias, dst}, attr_, p_engine_)
                : primitive_desc({src, weight, dst}, attr_, p_engine_);
        prim_ = super(pd_);
        // set this flag = true every time compile is called
        first_iteration_ = true;

        opt_src_desc_ = pd_.src_desc();
        opt_weights_desc_ = pd_.weights_desc();
        opt_dst_desc_ = pd_.dst_desc();
        if (with_bias_) { opt_bias_desc_ = pd_.bias_desc(); }

        if (impl::logical_tensor_wrapper(inputs.at(matmul_fwd::kSrc)).is_any())
            cvt_src_desc_ = opt_src_desc_;
        if (impl::logical_tensor_wrapper(inputs.at(matmul_fwd::kWeight))
                        .is_any())
            cvt_weights_desc_ = opt_weights_desc_;
        if (impl::logical_tensor_wrapper(outputs.at(matmul_fwd::kDst))
                        .is_any()) {
            cvt_dst_desc_ = opt_dst_desc_;
        }
        if ((with_bias_)
                && impl::logical_tensor_wrapper(inputs.at(matmul_fwd::kBias))
                           .is_any()) {
            cvt_bias_desc_ = opt_bias_desc_;
        }

        if (cvt_weights_desc_ != opt_weights_desc_) { //need to reorder
            weight_reorder_pd_ = dnnl::reorder::primitive_desc(
                    p_engine_, cvt_weights_desc_, p_engine_, opt_weights_desc_);
        }

        if (with_bias_) {
            if (cvt_bias_desc_ != opt_bias_desc_) {
                bias_reorder_pd_ = dnnl::reorder::primitive_desc(
                        p_engine_, cvt_bias_desc_, p_engine_, opt_bias_desc_);
            }
        }

        if (cvt_src_desc_ != opt_src_desc_) {
            src_reorder_pd_ = dnnl::reorder::primitive_desc(
                    p_engine_, cvt_src_desc_, p_engine_, opt_src_desc_);
        }

        if (cvt_dst_desc_ != opt_dst_desc_) {
            dst_reorder_pd_ = dnnl::reorder::primitive_desc(
                    p_engine_, opt_dst_desc_, p_engine_, cvt_dst_desc_);
        }

        // fill_layout_info for not-copied input/outputs
        impl::logical_tensor_t *ori_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(conv::kDst));
        fill_layout_info(ori_dst_lt, pd_.dst_desc());

        impl::logical_tensor_t *ori_weight_lt
                = const_cast<impl::logical_tensor_t *>(
                        &inputs.at(matmul_fwd::kWeight));
        // TODO(wuxun): here, we need to reshape the queried desc to the
        // original shape. However, if there is a broadcast in one dim and
        // DNNL also needs padding in this broadcast dim, reshaping will
        // fail. A possible solution is that in conversion's reorder, we
        // also add check for the broadcast-able dims.
        fill_layout_info(ori_weight_lt, pd_.weights_desc());
        if (with_bias_) {
            impl::logical_tensor_t *ori_bias_lt
                    = const_cast<impl::logical_tensor_t *>(
                            &inputs.at(matmul_fwd::kBias));
            fill_layout_info(
                    ori_bias_lt, opt_bias_desc_.reshape(old_bias_dims));
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
        cvt_src_.set_data_handle(inputs.at(matmul_fwd::kSrc).get_data_handle());
        cvt_dst_.set_data_handle(
                outputs.at(matmul_fwd::kDst).get_data_handle());
        cvt_weight_.set_data_handle(
                inputs.at(matmul_fwd::kWeight).get_data_handle());
        if (with_bias_) {
            cvt_bias_.set_data_handle(
                    inputs.at(matmul_fwd::kBias).get_data_handle());
        }
        if (with_add_)
            cvt_post_src_.set_data_handle(inputs.back().get_data_handle());

        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        if (cvt_weights_desc_ != opt_weights_desc_) { //need to reorder
            if (first_iteration_) {
                internal_buffers_.emplace_back(allocator::malloc(
                        opt_weights_desc_.get_size(), p_engine_, alc_));
                expected_weights_ = make_dnnl_memory(
                        opt_weights_desc_, p_engine_, internal_buffers_.back());
            }

            // matmul should not cache 2nd input data unless const is set in lt
            dnnl::reorder(weight_reorder_pd_)
                    .execute(p_stream, cvt_weight_, expected_weights_);
        } else {
            if (first_iteration_) expected_weights_ = cvt_weight_;
            expected_weights_.set_data_handle(cvt_weight_.get_data_handle());
        }

        if (with_bias_) {
            if (cvt_bias_desc_ != opt_bias_desc_) {
                if (first_iteration_) {
                    internal_buffers_.emplace_back(allocator::malloc(
                            opt_bias_desc_.get_size(), p_engine_, alc_));
                    expected_bias_ = make_dnnl_memory(opt_bias_desc_, p_engine_,
                            internal_buffers_.back());
                }

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
            }

            dnnl::reorder(src_reorder_pd_)
                    .execute(p_stream, cvt_src_, expected_src_);
        } else {
            if (first_iteration_) expected_src_ = cvt_src_;
            expected_src_.set_data_handle(cvt_src_.get_data_handle());
        }

        if (cvt_dst_desc_ != opt_dst_desc_) {
            if (first_iteration_) {
                internal_buffers_.emplace_back(allocator::malloc(
                        opt_dst_desc_.get_size(), p_engine_, alc_));
                expected_dst_ = make_dnnl_memory(
                        opt_dst_desc_, p_engine_, internal_buffers_.back());
            }
        } else {
            if (first_iteration_) expected_dst_ = cvt_dst_;
            expected_dst_.set_data_handle(cvt_dst_.get_data_handle());
        }

        if (with_post_sum_
                && cvt_post_src_.get_data_handle()
                        != expected_dst_.get_data_handle()) {
            dnnl::reorder(cvt_post_src_, expected_dst_)
                    .execute(p_stream, cvt_post_src_, expected_dst_);
        }

        if (first_iteration_) {
            matmul_args_ = {{DNNL_ARG_SRC, expected_src_},
                    {DNNL_ARG_WEIGHTS, expected_weights_},
                    {DNNL_ARG_DST, expected_dst_}};

            if (with_bias_) {
                matmul_args_.insert({DNNL_ARG_BIAS, expected_bias_});
            }

            if (with_post_binary_add_) {
                matmul_args_.insert(
                        {(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1),
                                cvt_post_src_});
            }
        }

        prim_.execute(p_stream, matmul_args_);

        // reorder optimal output to given output buffer
        if (expected_dst_ != cvt_dst_) {
            dnnl::reorder(dst_reorder_pd_)
                    .execute(p_stream, expected_dst_, cvt_dst_);
        }

        first_iteration_ = false;
        return impl::status::success;
    }

private:
    algorithm get_eltwise_algo(op_kind_t kind) {
        switch (static_cast<int>(kind)) {
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
            case op_kind::matmul_bias_gelu:
            case op_kind::matmul_add_gelu: return (algorithm::eltwise_gelu_erf);

            default:
                BACKEND_DNNL_ENFORCE(
                        0, "Unsupported fused_eltwise op for matmul.");
        }
        return algorithm::undef;
    }

    impl::status_t prepare_inplace_pairs_impl(const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(g_engine);
        if (with_post_sum_) {
            size_t input_idx = with_bias_ ? matmul_fwd::kBias + 1
                                          : matmul_fwd::kWeight + 1;

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

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
