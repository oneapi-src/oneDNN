/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef BACKEND_DNNL_KERNELS_RESAMPLING_HPP
#define BACKEND_DNNL_KERNELS_RESAMPLING_HPP

#include <memory>
#include <string>
#include <vector>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"
#include "backend/dnnl/f32_kernel_resource.hpp"
#include "backend/dnnl/passes/compile_ops.hpp"
#include "backend/dnnl/passes/infer_type.hpp"
#include "backend/dnnl/passes/insert_ops.hpp"
#include "backend/dnnl/passes/layout_propagation.hpp"
#include "backend/dnnl/passes/lower_down.hpp"
#include "backend/dnnl/passes/memory_planning.hpp"
#include "backend/dnnl/passes/op_executable.hpp"
#include "backend/dnnl/scratchpad.hpp"
#include "backend/dnnl/thread_local_cache.hpp"
#include "backend/dnnl/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace resampling_bwd {
enum resampling_bwd_inputs { kSrc, kDiff_dst, kSizes_Scales };
enum resampling_bwd_outputs { kDiff_src };
} // namespace resampling_bwd

struct resampling_fwd_t : public kernel_base_t {
private:
    dnnl::engine p_engine_;
    impl::allocator_t *g_alloc_;

    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

    impl::status_t prepare_inplace_pairs_impl(const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(g_engine);

        op_t *interpolate_op = nullptr;
        for (auto &op : subgraph_->get_ops()) {
            if (op->get_kind() == impl::op_kind::Interpolate) {
                interpolate_op = op.get();
                break;
            }
        }

        bool with_sum = interpolate_op->has_attr("with_sum")
                ? interpolate_op->get_attr<bool>("with_sum")
                : false;

        if (with_sum) {
            // get the index of post sum input
            size_t post_sum_index = 0;
            auto &prm_attr_mgr = subgraph_->prm_attr_mgr_;
            if (interpolate_op->has_attr("primitive_attr_key")) {
                int64_t key = interpolate_op->get_attr<int64_t>(
                        "primitive_attr_key");
                const auto &pops = prm_attr_mgr.get_attr(key).get_post_ops();
                for (int n = pops.len() - 1; n >= 0; --n) {
                    if (pops.kind(n) != dnnl::primitive::kind::eltwise)
                        post_sum_index++;
                    if (pops.kind(n) == dnnl::primitive::kind::sum) break;
                }
            }

            auto val = interpolate_op->get_input_value(
                    interpolate_op->num_inputs() - post_sum_index);
            if (val->has_producer()
                    && val->get_producer().get_kind() == op_kind::permute) {
                val = val->get_producer().get_input_value(0);
            }
            size_t post_src_id = val->get_logical_tensor().id;

            // find the given post src index
            size_t idx = 0;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].id == post_src_id) {
                    idx = i;
                    break;
                }
            }

            const logical_tensor_wrapper_t post_src_lt(inputs[idx]);
            const logical_tensor_wrapper_t dst_lt(outputs[0]);
            // TODO(qun) we didn't report iplace pair if two lts have different
            // layout type because of frontend users didn't process this
            // situation at this moment. In the future, we need to fix this for
            // more inplace opportunities.
            if (((post_src_lt.is_opaque() && dst_lt.is_opaque())
                        || (post_src_lt.is_strided() && dst_lt.is_strided()))
                    && post_src_lt.is_similar(dst_lt))
                inplace_pairs_.push_back({post_src_id, outputs[0].id});
        }
        return impl::status::success;
    }

public:
    ~resampling_fwd_t() override {
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

        // Because we use binary post-ops for broadcast add and sum post-ops for
        // non-broadcast add. So we have to know concret shape before fuse
        // post-ops
        BACKEND_DNNL_ADD_PASS(pipeline, infer_shape);
        BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);
        BACKEND_DNNL_ADD_PASS(pipeline, infer_shape);
        BACKEND_DNNL_ADD_PASS(pipeline, infer_type);
        BACKEND_DNNL_ADD_PASS(pipeline, eltwise_canonicalization);

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

        // fill information for inputs logical tensors
        for (size_t i = 0; i < inputs.size(); i++) {
            BACKEND_DNNL_CHECK(set_shape_and_layout(
                    const_cast<impl::logical_tensor_t &>(inputs[i]),
                    subgraph_->ins_[i]));
        }

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

struct resampling_backward : public dnnl::resampling_backward,
                             public kernel_base_t {
    using super = dnnl::resampling_backward;

private:
    primitive_desc pd_;
    resampling_forward::primitive_desc forward_hints_;
    std::string mode_;
    algorithm alg_;

    dnnl_tensor_t expected_src_;
    dnnl_tensor_t expected_diff_src_;
    dnnl_tensor_t expected_diff_dst_;

    // TODO(Jihui): dnnl backward not support calculate diff_scales
    std::string shape_calculation_mode_;
    dnnl_tensor_t expected_diff_scales_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    void compute(const dnnl_tensor_t &src, const dnnl_tensor_t &diff_dst,
            dnnl_tensor_t &diff_src, impl::allocator_t *alc) {
        if (src.get_desc() != forward_hints_.src_desc()) {
            if (expected_src_.is_empty()) {
                expected_src_ = dnnl_tensor_t {
                        forward_hints_.src_desc(), p_engine_, alc};
            }
            src.reorder_to(p_stream_, expected_src_);
        } else {
            expected_src_ = src;
        }

        if (diff_dst.get_desc() != pd_.diff_dst_desc()) {
            if (expected_diff_dst_.is_empty()) {
                expected_diff_dst_
                        = dnnl_tensor_t {pd_.diff_dst_desc(), p_engine_, alc};
            }
            diff_dst.reorder_to(p_stream_, expected_diff_dst_);
        } else
            expected_diff_dst_ = diff_dst;

        if (diff_src.get_desc() != pd_.diff_src_desc()) {
            if (expected_diff_src_.is_empty()) {
                expected_diff_src_
                        = dnnl_tensor_t {pd_.diff_src_desc(), p_engine_, alc};
            }
        } else {
            expected_diff_src_ = diff_src;
        }

        exec_args args;

        args.insert({DNNL_ARG_SRC, expected_src_});
        args.insert({DNNL_ARG_DIFF_SRC, expected_diff_src_});
        args.insert({DNNL_ARG_DIFF_DST, expected_diff_dst_});

        super(pd_).execute(p_stream_, args);

        if (expected_diff_src_ != diff_src) {
            dnnl::reorder(expected_diff_src_, diff_src)
                    .execute(p_stream_, expected_diff_src_, diff_src);
        }
    }

    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = dnnl_tensor_t::desc_t;
        // prepare the outputs' tensors' descs
        const desc src {inputs.at(resampling_bwd::kSrc)};
        const desc diff_dst {inputs.at(resampling_bwd::kDiff_dst)};
        const desc diff_src {outputs.at(resampling_bwd::kDiff_src)};

        mode_ = op->get_attr<std::string>("mode");
        if (mode_ == "nearest") {
            alg_ = algorithm::resampling_nearest;
        } else if (mode_ == "linear") {
            alg_ = algorithm::resampling_linear;
        } else {
            BACKEND_DNNL_ENFORCE(0, "Unsupported resampling mode.");
        }

        p_engine_ = make_dnnl_engine(*g_engine);
        forward_hints_ = resampling_forward::primitive_desc(
                {prop_kind::forward_training, alg_, src, diff_dst}, p_engine_);

        pd_ = primitive_desc(
                {alg_, diff_src, diff_dst}, p_engine_, forward_hints_);

        const dnnl_tensor_t::desc_t optimal_diff_src_desc {pd_.diff_src_desc()};
        impl::logical_tensor_t *diff_src_lt = const_cast<logical_tensor_t *>(
                &outputs.at(resampling_bwd::kDiff_src));
        fill_layout_info(diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        dnnl_tensor_t src {inputs.at(resampling_bwd::kSrc), p_engine_, alc};
        dnnl_tensor_t diff_dst {
                inputs.at(resampling_bwd::kDiff_dst), p_engine_, alc};
        dnnl_tensor_t diff_src {
                outputs.at(resampling_bwd::kDiff_src), p_engine_, alc};

        resampling_backward::compute(src, diff_dst, diff_src, alc);
        return impl::status::success;
    }
};

using float_resampling_fwd = resampling_fwd_t;

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
