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

#include "backend/dnnl/tensor.hpp"
#include "common.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace eltwise {
enum eltwise_inputs { kSrc };
enum eltwise_outputs { kDst };
} // namespace eltwise

struct eltwise_forward : public dnnl::eltwise_forward, public kernel_base {
    using super = dnnl::eltwise_forward;

private:
    primitive_desc pd_;
    algorithm algo_;
    float alpha_ = 0.f;
    float beta_ = 0.f;
    prop_kind prop_kind_ = prop_kind::forward_inference;
    dnnl::engine eng_;

public:
    void compute(const tensor &src, tensor &dst, const dnnl::engine &aengine,
            impl::allocator_t *alc) {
        tensor expected_src = src;
        tensor expected_dst = dst;
        if (pd_.dst_desc() != dst.get_desc()) {
            expected_dst = tensor {pd_.dst_desc(), aengine, alc};
        }

        dnnl::stream s(aengine);
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
        // prepare engine and the inputs' tensors' descs
        const desc src {inputs.at(eltwise::kSrc)};
        eng_ = make_dnnl_engine(*aengine);
        // set alpha and beta
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

        auto kind = anode->get_op_kind();
        switch (kind) {
            case op_kind::Abs: algo_ = algorithm::eltwise_abs; break;
            case op_kind::Elu: algo_ = algorithm::eltwise_elu; break;
            case op_kind::Exp: algo_ = algorithm::eltwise_exp; break;
            case op_kind::GELU: algo_ = algorithm::eltwise_gelu_erf; break;
            case op_kind::HardTanh: algo_ = algorithm::eltwise_clip; break;
            case op_kind::Log: algo_ = algorithm::eltwise_log; break;
            case op_kind::Pow: algo_ = algorithm::eltwise_pow; break;
            case op_kind::ReLU: algo_ = algorithm::eltwise_relu; break;
            case op_kind::Sqrt: algo_ = algorithm::eltwise_sqrt; break;
            case op_kind::Square: algo_ = algorithm::eltwise_square; break;
            case op_kind::Tanh: algo_ = algorithm::eltwise_tanh; break;

            default: BACKEND_DNNL_ENFORCE(0, "Unsupported eltwise op.");
        }

        pd_ = primitive_desc({prop_kind_, algo_, src, alpha_, beta_}, eng_);
        const tensor::desc optimal_dst_desc {pd_.dst_desc()};

        impl::logical_tensor_t *dst_lt = const_cast<impl::logical_tensor_t *>(
                &outputs.at(eltwise::kDst));
        fill_layout_info(dst_lt, optimal_dst_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream_t *astream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(anode);
        impl::allocator_t *alc = astream->get_engine()->get_allocator();

        tensor x {inputs.at(eltwise::kSrc), eng_, alc};
        tensor y {outputs.at(eltwise::kDst), eng_, alc};
        compute(x, y, eng_, alc);
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
    dnnl::engine eng_;

public:
    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // prepare the input's and output's desc
        const desc src {inputs.at(eltwise::kSrc + 1)};

        op_kind_t kind = anode->get_op_kind();
        eng_ = make_dnnl_engine(*aengine);

        pd_ = get_config(src, kind, eng_, 0.f, 0.f);

        const desc optimal_diff_src {pd_.diff_src_desc()};
        impl::logical_tensor_t *diff_src_lt
                = const_cast<impl::logical_tensor_t *>(
                        &outputs.at(eltwise::kSrc));
        fill_layout_info(diff_src_lt, optimal_diff_src);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream_t *astream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(anode);
        impl::allocator_t *alc = astream->get_engine()->get_allocator();

        tensor x1 {inputs.at(eltwise::kSrc + 1), eng_, alc};
        tensor x2 {inputs.at(eltwise::kDst), eng_, alc};
        tensor y {outputs.at(eltwise::kSrc), eng_, alc};
        compute(x1, x2, y, eng_, alc);
        return impl::status::success;
    }

private:
    // If grady and x had different format, performance is bad.
    // TODO(xxx): Seeking a single shot solution.
    void compute(const tensor &src, const tensor &diff_dst, tensor &diff_src,
            const dnnl::engine &aengine, impl::allocator_t *alc) {
        UNUSED(alc);
        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(pd_.diff_dst_desc());
        auto expected_src = src.reorder_if_differ_in(pd_.src_desc());
        diff_src.reinit_if_possible(pd_.diff_src_desc());

        dnnl::stream s(aengine);
        super(pd_).execute(s,
                {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                        {DNNL_ARG_SRC, expected_src},
                        {DNNL_ARG_DIFF_SRC, diff_src}});
        s.wait();
    }

    primitive_desc get_config(const tensor::desc &src, op_kind_t kind,
            const dnnl::engine &aengine, float alpha = 0.0, float beta = 0.0) {
        switch (kind) {
            case op_kind::ReLUBackprop: algo_ = algorithm::eltwise_relu; break;
            case op_kind::GELUBackprop:
                algo_ = algorithm::eltwise_gelu_erf;
                break;
            default: BACKEND_DNNL_ENFORCE(0, "Unsupported eltwise backward op");
        }
        alpha_ = alpha;
        beta_ = beta;
        auto func = [&src, &aengine](algorithm algo, float alpha, float beta) {
            auto forward_hints = eltwise_forward::primitive_desc(
                    {prop_kind::forward_training, algo, src, alpha, beta},
                    aengine);

            return primitive_desc(
                    {algo, forward_hints.dst_desc(), src, alpha, beta}, aengine,
                    forward_hints);
        };
        return func(algo_, alpha, beta);
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
