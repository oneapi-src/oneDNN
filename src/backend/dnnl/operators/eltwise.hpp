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

#ifndef BACKEND_DNNL_OPERATORS_ELTWISE_HPP
#define BACKEND_DNNL_OPERATORS_ELTWISE_HPP

#include <tuple>
#include <vector>
#include <unordered_set>

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/tensor.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace eltwise {
enum eltwise_inputs { kSrc };
enum eltwise_outputs { kDst };
} // namespace eltwise

struct eltwise_fusion_set {
    static bool with_binary(op_kind_t kind) {
        static const std::unordered_set<op_kind_t, enum_hash> with_binary_set {
                op_kind::relu_add};
        return with_binary_set.find(kind) != with_binary_set.end();
    }

    static bool get_binary_algo(op_kind_t kind, algorithm &alg) {
        static const std::unordered_set<op_kind_t, enum_hash> add_set {
                op_kind::relu_add};
        if (add_set.find(kind) != add_set.end()) {
            alg = algorithm::binary_add;
            return true;
        }
        return false;
    }
};

struct eltwise_forward : public dnnl::eltwise_forward, public kernel_base {
    using super = dnnl::eltwise_forward;

private:
    primitive_desc pd_;
    dnnl::eltwise_forward prim_;
    algorithm post_alg_;
    algorithm algo_;
    attr_t attr_;
    op_kind_t kind_;

    tensor expected_dst_;

    float alpha_ = 0.f;
    float beta_ = 0.f;
    prop_kind prop_kind_ = prop_kind::forward;
    dnnl::engine p_engine_;

public:
    void compute(const tensor &src, tensor &dst, const tensor &post_src,
            impl::allocator_t *alc, const dnnl::stream &p_stream) const {
        tensor expected_dst;
        if (dst.get_desc() != pd_.dst_desc()) {
            if (expected_dst.is_empty()) {
                expected_dst = tensor {pd_.dst_desc(), p_engine_, alc};
            }
        } else {
            expected_dst = dst;
        }

        exec_args args;

        args.insert({DNNL_ARG_SRC, src});
        args.insert({DNNL_ARG_DST, expected_dst});
        if (eltwise_fusion_set::with_binary(kind_))
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                    post_src});

        prim_.execute(p_stream, args);

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
        // prepare engine and the inputs' tensors' descs
        const desc src {inputs.at(eltwise::kSrc)};
        const desc dst {outputs.at(eltwise::kDst)};
        p_engine_ = make_dnnl_engine(*g_engine);
        // set alpha and beta
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

        kind_ = op->get_kind();
        switch (kind_) {
            case impl::op_kind::Abs: algo_ = algorithm::eltwise_abs; break;
            case impl::op_kind::Elu: algo_ = algorithm::eltwise_elu; break;
            case impl::op_kind::Exp: algo_ = algorithm::eltwise_exp; break;
            case impl::op_kind::GELU:
                algo_ = algorithm::eltwise_gelu_erf;
                break;
            case impl::op_kind::HardTanh:
                algo_ = algorithm::eltwise_clip;
                break;
            case impl::op_kind::Log: algo_ = algorithm::eltwise_log; break;
            case impl::op_kind::Pow: algo_ = algorithm::eltwise_pow; break;
            case impl::op_kind::ReLU:
            case op_kind::relu_add: algo_ = algorithm::eltwise_relu; break;
            case impl::op_kind::Sqrt: algo_ = algorithm::eltwise_sqrt; break;
            case impl::op_kind::Square:
                algo_ = algorithm::eltwise_square;
                break;
            case impl::op_kind::Tanh: algo_ = algorithm::eltwise_tanh; break;

            default: BACKEND_DNNL_ENFORCE(0, "Unsupported eltwise op.");
        }
        if (!eltwise_fusion_set::with_binary(kind_))
            pd_ = primitive_desc(
                    {prop_kind_, algo_, src, alpha_, beta_}, p_engine_);
        else {
            //eltwise op only support binary post_ops
            tensor::desc post_src {inputs.back()};

            // post binary only supports per tensor and per channel
            // broadcast, which means the expand shape of post src should
            // be all one or the post_src_dim[1]==dst_dim[1]
            if (post_src.get_ndims() == dst.get_ndims()) {
                for (int i = dst.get_ndims() - 1; i >= 0; i--) {
                    if (post_src.get_dim(i) == 1) continue;

                    if (i != 1 || dst.get_dim(i) != post_src.get_dim(i)) {
                        return impl::status::compile_fail;
                    }
                }
            } else {
                if (post_src.get_ndims() != 1 || post_src.get_dim(0) != 1)
                    return impl::status::compile_fail;
            }

            if (eltwise_fusion_set::get_binary_algo(kind_, post_alg_)) {
                attr_ = attr_t::fuse_binary(post_src, post_alg_);
                pd_ = primitive_desc({prop_kind_, algo_, src, alpha_, beta_},
                        attr_, p_engine_);
            }
        }
        prim_ = super(pd_);
        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        impl::logical_tensor_t *dst_lt = const_cast<impl::logical_tensor_t *>(
                &outputs.at(eltwise::kDst));
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

        tensor x {inputs.at(eltwise::kSrc).get_logical_tensor(), p_engine_, alc,
                inputs.at(eltwise::kSrc).get_data_handle()};
        tensor y {outputs.at(eltwise::kDst).get_logical_tensor(), p_engine_,
                alc, outputs.at(eltwise::kDst).get_data_handle()};
        const tensor post_src = eltwise_fusion_set::with_binary(kind_)
                ? tensor {inputs.back().get_logical_tensor(), p_engine_, alc,
                        inputs.back().get_data_handle()}
                : tensor {};
        compute(x, y, post_src, alc, p_stream);
        return impl::status::success;
    }
};

struct eltwise_backward : public dnnl::eltwise_backward, public kernel_base {
    using super = dnnl::eltwise_backward;
    using eltwise_argpack = std::tuple<algorithm, float, float>;

private:
    algorithm algo_;
    float alpha_;
    float beta_;
    primitive_desc pd_;
    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // prepare the input's and output's desc
        const desc src {inputs.at(eltwise::kSrc + 1)};

        op_kind_t kind = op->get_kind();
        p_engine_ = make_dnnl_engine(*g_engine);

        pd_ = get_config(src, kind, p_engine_, 0.f, 0.f);

        const desc optimal_diff_src {pd_.diff_src_desc()};
        impl::logical_tensor_t *diff_src_lt
                = const_cast<impl::logical_tensor_t *>(
                        &outputs.at(eltwise::kSrc));
        fill_layout_info(diff_src_lt, optimal_diff_src);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        tensor x1 {inputs.at(eltwise::kSrc + 1), p_engine_, alc};
        tensor x2 {inputs.at(eltwise::kDst), p_engine_, alc};
        tensor y {outputs.at(eltwise::kSrc), p_engine_, alc};
        compute(x1, x2, y, p_engine_, alc, p_stream_);
        return impl::status::success;
    }

private:
    // If grady and x had different format, performance is bad.
    // TODO(xxx): Seeking a single shot solution.
    void compute(const tensor &src, const tensor &diff_dst, tensor &diff_src,
            const dnnl::engine &aengine, impl::allocator_t *alc,
            const dnnl::stream &p_stream) {
        UNUSED(alc);
        UNUSED(aengine);
        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(p_stream, pd_.diff_dst_desc());
        auto expected_src = src.reorder_if_differ_in(p_stream, pd_.src_desc());
        diff_src.reinit_if_possible(p_stream, pd_.diff_src_desc());

        super(pd_).execute(p_stream,
                {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                        {DNNL_ARG_SRC, expected_src},
                        {DNNL_ARG_DIFF_SRC, diff_src}});
    }

    primitive_desc get_config(const tensor::desc &src, op_kind_t kind,
            const dnnl::engine &p_engine, float alpha = 0.0, float beta = 0.0) {
        switch (kind) {
            case impl::op_kind::ReLUBackprop:
                algo_ = algorithm::eltwise_relu;
                break;
            case impl::op_kind::GELUBackprop:
                algo_ = algorithm::eltwise_gelu_erf;
                break;
            default: BACKEND_DNNL_ENFORCE(0, "Unsupported eltwise backward op");
        }
        alpha_ = alpha;
        beta_ = beta;
        auto func = [&src, &p_engine](algorithm algo, float alpha, float beta) {
            auto forward_hints = eltwise_forward::primitive_desc(
                    {prop_kind::forward_training, algo, src, alpha, beta},
                    p_engine);

            return primitive_desc(
                    {algo, forward_hints.dst_desc(), src, alpha, beta},
                    p_engine, forward_hints);
        };
        return func(algo_, alpha, beta);
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
