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

#ifndef BACKEND_DNNL_KERNELS_BATCHNORM_HPP
#define BACKEND_DNNL_KERNELS_BATCHNORM_HPP

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/constant_cache.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"
#include "backend/dnnl/passes/compile_ops.hpp"
#include "backend/dnnl/passes/constant_propagation.hpp"
#include "backend/dnnl/passes/infer_type.hpp"
#include "backend/dnnl/passes/insert_ops.hpp"
#include "backend/dnnl/passes/layout_propagation.hpp"
#include "backend/dnnl/passes/lower_down.hpp"
#include "backend/dnnl/passes/memory_planning.hpp"
#include "backend/dnnl/passes/op_executable.hpp"
#include "backend/dnnl/scratchpad.hpp"
#include "backend/dnnl/thread_local_cache.hpp"
#include "backend/dnnl/utils.hpp"

// should be removed later
#include "backend/dnnl/tensor.hpp"
#include "backend/dnnl/utils.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#include <dnnl_sycl.hpp>
#endif

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace batch_normalization {
enum batch_normalization_inputs { kSrc, kScale, kShift, kMean, kVariance };
enum batch_normalization_outputs {
    kDst,
    kRunning_mean,
    kRunning_variance,
    kBatch_mean,
    kBatch_variance
};
} // namespace batch_normalization

namespace batch_normalization_bwd {
enum batch_normalization_bwd_inputs {
    kSrc,
    kDiff_dst,
    kScale,
    kMean,
    kVariance
};
enum batch_normalization_bwd_outputs { kDiff_src, kDiff_scale, kDiff_shift };
} // namespace batch_normalization_bwd

struct batchnorm_fwd_t : public kernel_base_t {
private:
    dnnl::engine p_engine_;
    impl::allocator_t *g_alloc_;

    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

public:
    ~batchnorm_fwd_t() override {
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

        BACKEND_DNNL_ADD_PASS(pipeline, eltwise_canonicalization);
        BACKEND_DNNL_ADD_PASS(pipeline, batchnorm_canonicalization);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);
        BACKEND_DNNL_ADD_PASS(pipeline, insert_permute);
        BACKEND_DNNL_ADD_PASS(pipeline, infer_shape);

        pipeline.reset_visualize_arg(true, false);
        BACKEND_DNNL_ADD_PASS(pipeline, infer_type);
        BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

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

struct batch_normalization_backward : public dnnl::batch_normalization_backward,
                                      public kernel_base_t {
    using super = dnnl::batch_normalization_backward;

private:
    primitive_desc pd_;
    float epsilon_;

    dnnl_tensor_t diff_scale_shift_;
    dnnl_tensor_t original_diff_src_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = dnnl_tensor_t::desc_t;
        epsilon_ = op->get_attr<float>("epsilon");
        std::string data_format = op->get_attr<std::string>("data_format");

        impl::logical_tensor_t src_lt = inputs.at(batch_normalization::kSrc);
        impl::logical_tensor_t diff_src_lt
                = outputs.at(batch_normalization_bwd::kDiff_src);
        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper_t(&src_lt)
                             .reorder_data_dims_strides();
        }
        // prepare the inputs and outputs tensors' descs
        desc src {src_lt};

        const bool use_stats = inputs.size() > batch_normalization::kMean;

        auto flags = normalization_flag::use_scale_shift;
        if (use_stats) flags |= normalization_flag::use_global_stats;

        // workaround: use src once issue intel/mkl-dnn#588 is
        // resolved
        src = src.is_4c_blocked() ? src.to_default_format() : src;

        p_engine_ = make_dnnl_engine(*g_engine);
        impl::allocator_t *alc = g_engine->get_allocator();

        auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
                {prop_kind::forward_training, src, epsilon_, flags}, p_engine_);
        // cache diff_dst in order to compute
        pd_ = primitive_desc({prop_kind::backward, forward_hints.dst_desc(),
                                     src, epsilon_, flags},
                p_engine_, forward_hints);

        diff_scale_shift_
                = dnnl_tensor_t(pd_.diff_weights_desc(), p_engine_, alc);
        const dnnl_tensor_t::desc_t optimal_diff_src_desc {pd_.diff_src_desc()};
        fill_layout_info(&diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        std::string data_format = op->get_attr<std::string>("data_format");
        auto &src_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(batch_normalization_bwd::kSrc).get_logical_tensor());
        auto &diff_dst_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(batch_normalization_bwd::kDiff_dst)
                        .get_logical_tensor());
        auto &diff_src_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(batch_normalization_bwd::kDiff_src)
                        .get_logical_tensor());
        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper_t(src_lt)
                             .reorder_data_dims_strides();
            diff_dst_lt = impl::logical_tensor_wrapper_t(diff_dst_lt)
                                  .reorder_data_dims_strides();
            diff_src_lt = impl::logical_tensor_wrapper_t(diff_src_lt)
                                  .reorder_data_dims_strides();
        }
        dnnl_tensor_t x {src_lt, p_engine_, alc,
                inputs.at(batch_normalization_bwd::kSrc).get_data_handle()};
        dnnl_tensor_t w {
                inputs.at(batch_normalization_bwd::kScale), p_engine_, alc};
        dnnl_tensor_t m {
                inputs.at(batch_normalization_bwd::kMean), p_engine_, alc};
        dnnl_tensor_t v {
                inputs.at(batch_normalization_bwd::kVariance), p_engine_, alc};
        dnnl_tensor_t diff_dst {diff_dst_lt, p_engine_, alc,
                inputs.at(batch_normalization_bwd::kDiff_dst)
                        .get_data_handle()};
        dnnl_tensor_t diff_scale {
                outputs.at(batch_normalization_bwd::kDiff_scale), p_engine_,
                alc};
        dnnl_tensor_t diff_shift {
                outputs.at(batch_normalization_bwd::kDiff_shift), p_engine_,
                alc};
        dnnl_tensor_t diff_src {diff_src_lt, p_engine_, alc,
                outputs.at(batch_normalization_bwd::kDiff_src)
                        .get_data_handle()};
        compute(x, m, v, diff_dst, w, diff_src, diff_scale, diff_shift,
                p_engine_, alc, p_stream_);
        return impl::status::success;
    }

private:
    void compute_impl(dnnl_tensor_t &src, dnnl_tensor_t &mean,
            dnnl_tensor_t &variance, dnnl_tensor_t &diff_dst,
            const dnnl_tensor_t &scale, dnnl_tensor_t &diff_src,
            const dnnl::engine &p_engine, impl::allocator_t *alc,
            const dnnl::stream &p_stream) {
        UNUSED(alc);
        UNUSED(p_engine);
        // TODO(xxx): support no-affine model
        original_diff_src_ = diff_src;

        diff_dst.reinit_if_possible(p_stream, pd_.diff_dst_desc());
        src.reinit_if_possible(p_stream, pd_.src_desc());
        diff_src.reinit_if_possible(p_stream, pd_.diff_src_desc());
        diff_scale_shift_.reinit_if_possible(p_stream, pd_.diff_weights_desc());

        mean.reinit_if_possible(p_stream, pd_.mean_desc());
        variance.reinit_if_possible(p_stream, pd_.variance_desc());
        super(pd_).execute(p_stream,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_DIFF_DST, diff_dst},
                        {DNNL_ARG_SCALE_SHIFT, scale}, // only need scale
                        {DNNL_ARG_MEAN, mean}, {DNNL_ARG_VARIANCE, variance},
                        {DNNL_ARG_DIFF_SRC, diff_src},
                        {DNNL_ARG_DIFF_SCALE_SHIFT, diff_scale_shift_}});

        if (diff_src.get_desc() != original_diff_src_.get_desc()) {
            diff_src.reorder_to(p_stream, original_diff_src_);
        }
    }

    void compute(dnnl_tensor_t &src, dnnl_tensor_t &mean,
            dnnl_tensor_t &variance, dnnl_tensor_t &diff_dst,
            const dnnl_tensor_t &scale, dnnl_tensor_t &diff_src,
            dnnl_tensor_t &diff_scale, dnnl_tensor_t &diff_shift,
            const dnnl::engine &p_engine, impl::allocator_t *alc,
            const dnnl::stream &p_stream) {
        compute_impl(src, mean, variance, diff_dst, scale, diff_src, p_engine,
                alc, p_stream);
        diff_scale.reinit_if_possible(p_stream, scale.get_desc());
        diff_shift.reinit_if_possible(p_stream, scale.get_desc());
        auto *diff_scale_shift_buf
                = static_cast<char *>(diff_scale_shift_.get_data_handle());
        if (p_engine_.get_kind() == dnnl::engine::kind::cpu) {
#if DNNL_GRAPH_CPU_SYCL
            cl::sycl::queue q = dnnl::sycl_interop::get_queue(p_stream);
            q.memcpy(diff_scale.get_data_handle(), diff_scale_shift_buf,
                    diff_scale.get_size());
            q.memcpy(diff_shift.get_data_handle(),
                    diff_scale_shift_buf + diff_scale.get_size(),
                    diff_shift.get_size());
#else
            std::memcpy(diff_scale.get_data_handle(), diff_scale_shift_buf,
                    diff_scale.get_size());
            std::memcpy(diff_shift.get_data_handle(),
                    diff_scale_shift_buf + diff_scale.get_size(),
                    diff_shift.get_size());
#endif
        } else {
#if DNNL_GRAPH_GPU_SYCL
            cl::sycl::queue q = dnnl::sycl_interop::get_queue(p_stream);
            q.memcpy(diff_scale.get_data_handle(), diff_scale_shift_buf,
                    diff_scale.get_size());
            q.memcpy(diff_shift.get_data_handle(),
                    diff_scale_shift_buf + diff_scale.get_size(),
                    diff_shift.get_size());
#else
            std::memcpy(diff_scale.get_data_handle(), diff_scale_shift_buf,
                    diff_scale.get_size());
            std::memcpy(diff_shift.get_data_handle(),
                    diff_scale_shift_buf + diff_scale.get_size(),
                    diff_shift.get_size());
#endif
        }
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
