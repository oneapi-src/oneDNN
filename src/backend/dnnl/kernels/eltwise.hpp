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

#ifndef BACKEND_DNNL_KERNELS_ELTWISE_HPP
#define BACKEND_DNNL_KERNELS_ELTWISE_HPP

#include <memory>
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

template <bool quantized>
struct eltwise_fwd_t : public kernel_base_t {
private:
    dnnl::engine p_engine_;
    impl::allocator_t *g_alloc_;

    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

public:
    ~eltwise_fwd_t() override {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
    }

    impl::status_t compile_impl(const dnnl_partition_impl_t *part,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = g_engine->get_allocator();

        subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_);
        BACKEND_DNNL_CHECK(
                set_given_inputs_outputs(subgraph_, inputs, outputs));

        subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
            return this->memory_planner_.get_memory_info(val);
        });
        pass_pipeline_t pipeline(vis);

        BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);
        BACKEND_DNNL_ADD_PASS(pipeline, insert_reorder);

        BACKEND_DNNL_ADD_PASS(pipeline, infer_shape);

        pipeline.reset_visualize_arg(true, false);
        BACKEND_DNNL_ADD_PASS(pipeline, infer_type);
        BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

        // bind the memory for each op
        auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
            return memory_planner_.run(sg);
        };
        pipeline.reset_visualize_arg(true, true);
        BACKEND_DNNL_ADD_PASS(pipeline, memory_plan);
        BACKEND_DNNL_ADD_PASS(pipeline, compile_ops);

        // Run the added passes
        BACKEND_DNNL_CHECK(pipeline.run(subgraph_));

        // fill information for outputs logical tensors
        for (size_t i = 0; i < outputs.size(); i++) {
            BACKEND_DNNL_CHECK(set_shape_and_layout(
                    const_cast<impl::logical_tensor_t &>(outputs[i]),
                    subgraph_->outs_[i]));
        }

        // generate a hash key for exec_args_mgr
        resource_ctor_ = [this]() {
            return this->memory_planner_.get_exec_args_set().clone();
        };

        return impl::status::success;
    }

    impl::status_t execute_impl(const dnnl_partition_impl_t *part,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(part);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        // each thread's own local resource
        thread_local_cache_t<execution_args_set_t> res_cache;
        execution_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
            mem_idx.first.set_data_handle(
                    inputs[mem_idx.second].get_data_handle());
        }
        for (const auto &mem_idx : res->get_mems_use_external_outputs()) {
            mem_idx.first.set_data_handle(
                    outputs[mem_idx.second].get_data_handle());
        }

        temporary_scratchpad_t scratchpad(
                memory_planner_.total_internal_temporary_size(), p_engine_,
                *g_alloc_);
        assertm(scratchpad.size()
                        >= memory_planner_.total_internal_temporary_size(),
                "no enough scratchpad memory");
        grantor_t var_grantor = memory_planner_.internal_temporary_grantor(
                scratchpad.get_buffer());

        registry_t::key_t key = 0;
        for (auto &mem_offkey : res->get_mems_use_internal_temporary()) {
            mem_offkey.first.set_data_handle(
                    var_grantor.get(mem_offkey.second));
        }

        for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
            subgraph_->execs_[i]->execute(p_stream, res->get_exec_args()[i]);
        }

        return impl::status::success;
    }
};

using float_eltwise_fwd = eltwise_fwd_t</* quantized */ false>;

struct eltwise_backward : public dnnl::eltwise_backward, public kernel_base_t {
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
        using desc = dnnl_tensor_t::desc_t;
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

        dnnl_tensor_t x1 {inputs.at(eltwise::kSrc + 1), p_engine_, alc};
        dnnl_tensor_t x2 {inputs.at(eltwise::kDst), p_engine_, alc};
        dnnl_tensor_t y {outputs.at(eltwise::kSrc), p_engine_, alc};
        compute(x1, x2, y, p_engine_, alc, p_stream_);
        return impl::status::success;
    }

private:
    // If grady and x had different format, performance is bad.
    // TODO(xxx): Seeking a single shot solution.
    void compute(const dnnl_tensor_t &src, const dnnl_tensor_t &diff_dst,
            dnnl_tensor_t &diff_src, const dnnl::engine &aengine,
            impl::allocator_t *alc, const dnnl::stream &p_stream) {
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

    primitive_desc get_config(const dnnl_tensor_t::desc_t &src, op_kind_t kind,
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
