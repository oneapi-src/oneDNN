/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_PRIMITIVE_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_PRIMITIVE_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/primitive.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/sdpa_pd.hpp"
#include "common/sdpa_types.hpp"
#include "common/sdpa_utils.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_stream.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/ocl_stream.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "sycl/sycl_stream.hpp"
#endif

#include "graph/interface/backend.hpp"
#include "graph/interface/graph.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"
#include "graph/backend/dnnl/dnnl_partition_impl.hpp"
#include "graph/backend/dnnl/op_executable.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/thread_local_cache.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using ltw = logical_tensor_wrapper_t;
using op_ptr = std::shared_ptr<op_t>;
using registry_key = size_t;

struct sdp_primitive_config_t {
public:
    sdp_primitive_config_t() = default;

    std::shared_ptr<value_t> q_ = nullptr;
    std::shared_ptr<value_t> k_ = nullptr;
    std::shared_ptr<value_t> v_ = nullptr;
    std::shared_ptr<value_t> dst_ = nullptr;
    std::shared_ptr<value_t> scale_ = nullptr;
    std::shared_ptr<value_t> attn_mask_ = nullptr;
    bool invert_scale_ = false;

    // SDP pd and primitive.
    std::shared_ptr<primitive_desc_t> sdpa_pd_;
    std::shared_ptr<primitive_t> sdpa_prim_;

private:
    op_ptr get_post_op(const op_ptr &op) const {
        const auto out_val = op->get_output_value(0);
        const auto &consumers = out_val->get_consumers();
        if (consumers.size() != 1) return nullptr;
        return consumers[0].get_op().shared_from_this();
    }

public:
    status_t locate_io(std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) {

        using dnnl::impl::utils::one_of;

        auto follow_back = [](std::shared_ptr<value_t> val) {
            while (val->has_producer() && val->get_producer().num_inputs() == 1)
                val = val->get_producer().get_input_value(0);
            return val;
        };

        auto in_tensor_list
                = [](const value_t *val,
                          const std::vector<logical_tensor_t> &list) {
                      for (auto &t : list)
                          if (val->get_logical_tensor().id == t.id) return true;
                      return false;
                  };

        // Locate ops of interest: matmuls, scale, mask
        op_ptr mm1, mm2, scale, add, final_op;
        for (const auto &cur_op : sg->get_ops()) {
            if (in_tensor_list(cur_op->get_output_value(0).get(), outputs))
                final_op = cur_op;
            if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
            auto post_op = get_post_op(cur_op);
            if (post_op && post_op->get_kind() == op_kind::dnnl_binary) {
                if (mm1) return status::unimplemented;
                mm1 = cur_op;
                scale = post_op;

                auto scale_alg = static_cast<alg_kind_t>(
                        post_op->get_attr<int64_t>(op_attr::alg_kind));
                if (!one_of(scale_alg, alg_kind::binary_mul,
                            alg_kind::binary_div))
                    return status::unimplemented;
                invert_scale_ = (scale_alg == alg_kind::binary_div);

                if (get_post_op(post_op)->get_kind() == op_kind::dnnl_binary)
                    add = get_post_op(post_op);
            } else {
                if (mm2) return status::unimplemented;
                mm2 = cur_op;
            }
        }

        // Locate input/outputs: Q, K, V, dst, scale, mask
        if (!mm1 || !mm2 || !final_op) return status::unimplemented;
        q_ = mm1->get_input_value(0);
        k_ = mm1->get_input_value(1);
        v_ = mm2->get_input_value(1);
        dst_ = (final_op->get_kind() == op_kind::dnnl_transpose)
                ? final_op->get_input_value(0)
                : final_op->get_output_value(
                        0); /* for some reason final transpose is not fused into mm2 */

        if (scale) {
            auto s0 = follow_back(scale->get_input_value(0));
            auto s1 = follow_back(scale->get_input_value(1));
            scale_ = in_tensor_list(s1.get(), inputs) ? s1 : s0;
        }

        if (add) {
            auto m0 = add->get_input_value(0), m1 = add->get_input_value(1);
            attn_mask_ = in_tensor_list(m1.get(), inputs) ? m1 : m0;
        }

        return status::success;
    }

    // Initialize parameters and primitive.
    status_t init(std::shared_ptr<subgraph_t> &sg, const dnnl::engine &p_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) {

        CHECK(locate_io(sg, inputs, outputs));

        // Retrieve mds and create pd, primitive
        auto md_q = make_dnnl_memory_desc(q_->get_logical_tensor());
        auto md_k = make_dnnl_memory_desc(k_->get_logical_tensor());
        auto md_v = make_dnnl_memory_desc(v_->get_logical_tensor());
        auto md_dst = make_dnnl_memory_desc(dst_->get_logical_tensor());

        dnnl::memory::desc md_mask;
        if (attn_mask_)
            md_mask = make_dnnl_memory_desc(attn_mask_->get_logical_tensor());

        auto scale_dt = impl::data_type::undef;
        if (scale_) scale_dt = scale_->get_logical_tensor().data_type;

        dnnl::primitive_attr attr;

        auto &mgr = sg->fusion_info_mgr_;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        attr.set_fpmath_mode(
                static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode()));

        CHECK(create_sdpa_pd(sdpa_pd_, p_engine.get(), md_q.get(), md_k.get(),
                md_v.get(), md_dst.get(), md_mask.get(), scale_dt,
                invert_scale_, attr.get()));

        auto status = sdpa_pd_->create_primitive(sdpa_prim_, p_engine.get());

        if (status != status::success) {
            if (get_verbose(verbose_t::create_dispatch, component_t::graph)) {
                printf("onednn_verbose,graph,create:dispatch,sdpa,could not "
                       "create primitive, falling back\n");
            }
        }

        return status;
    }
};

class sdp_primitive_kernel_t : public kernel_base_t {
private:
    allocator_t *g_alloc_ = nullptr;

    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;
    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

    sdp_primitive_config_t cfg_;

public:
    sdp_primitive_kernel_t() {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.retain();
    }

    ~sdp_primitive_kernel_t() override {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
        res_cache.release();
    }

    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override {
        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = reinterpret_cast<graph::allocator_t *>(
                g_engine->get_allocator());

        // First, dry run on a deep copy
        subgraph_ = std::make_shared<subgraph_t>(
                graph_t::deep_copy(part->get_ops()), p_engine_,
                part->get_fpmath_mode(), false, true);
        CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

        subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
            return this->memory_planner_.get_memory_info(val);
        });
        pass_pipeline_t pipeline = pass_pipeline_t(vis);

        BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
        BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);
        BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_matmul);

        pipeline.reset_visualize_arg(true, false);
        BACKEND_DNNL_ADD_PASS(pipeline, infer_shape);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_transpose_to_matmul);
        BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

        // bind the memory for each op
        auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
            return memory_planner_.run(sg);
        };
        pipeline.reset_visualize_arg(true, true);
        BACKEND_DNNL_ADD_PASS(pipeline, memory_plan);

        auto modify_subgraph = [&] {
            // Run the added passes
            CHECK(pipeline.run(subgraph_));

            // fill information for inputs logical tensors
            for (size_t i = 0; i < inputs.size(); i++) {
                auto &in = const_cast<logical_tensor_t &>(inputs[i]);
                in = subgraph_->ins_[i];
            }

            // fill information for outputs logical tensors
            for (size_t i = 0; i < outputs.size(); i++) {
                auto &out = const_cast<logical_tensor_t &>(outputs[i]);
                out = subgraph_->outs_[i];
            }

            return status::success;
        };

        resource_ctor_ = [this]() {
            return this->memory_planner_.get_exec_args_set().clone();
        };

        CHECK(modify_subgraph());
        CHECK(cfg_.init(subgraph_, p_engine_, inputs, outputs));

        // Successfully created the primitive. Rerun the passes again, modifying
        //   the original ops.
        subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
                part->get_fpmath_mode(), false, true);
        CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));
        CHECK(modify_subgraph());
        CHECK(cfg_.locate_io(subgraph_, inputs, outputs));

        return status::success;
    }

    void prepare_args_set(const execution_args_set_t *res,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const scratchpad_t &scratchpad) {
        // update the data of partition in/outputs args
        for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
            mem_idx.first.set_data_handle(
                    inputs[mem_idx.second].get_data_handle());
        }
        for (const auto &mem_idx : res->get_mems_use_external_outputs()) {
            mem_idx.first.set_data_handle(
                    outputs[mem_idx.second].get_data_handle());
        }

        grantor_t var_grantor = memory_planner_.internal_temporary_grantor(
                scratchpad.get_buffer());

        for (auto &mem_offkey : res->get_mems_use_internal_temporary()) {
            mem_offkey.first.set_data_handle(
                    var_grantor.get(mem_offkey.second));
        }
    }

    status_t get_prim_exec_args(exec_args_t &args, memory (&mem_storage)[6],
            const execution_args_set_t *res) {
        bool ok = res->find_value_mem_map(cfg_.q_.get(), mem_storage[0])
                && res->find_value_mem_map(cfg_.k_.get(), mem_storage[1])
                && res->find_value_mem_map(cfg_.v_.get(), mem_storage[2])
                && res->find_value_mem_map(cfg_.dst_.get(), mem_storage[3]);

        if (cfg_.scale_)
            ok = ok
                    && res->find_value_mem_map(
                            cfg_.scale_.get(), mem_storage[4]);
        if (cfg_.attn_mask_)
            ok = ok
                    && res->find_value_mem_map(
                            cfg_.attn_mask_.get(), mem_storage[5]);

        if (!ok) return status::runtime_error;

        memory_arg_t mem_arg_q = {mem_storage[0].get(), true};
        memory_arg_t mem_arg_k = {mem_storage[1].get(), true};
        memory_arg_t mem_arg_v = {mem_storage[2].get(), true};
        memory_arg_t mem_arg_dst = {mem_storage[3].get(), false};
        memory_arg_t mem_arg_scale = {mem_storage[4].get(true), true};
        memory_arg_t mem_arg_mask = {mem_storage[5].get(true), true};

        args.clear();
        args[DNNL_ARG_QUERIES] = mem_arg_q;
        args[DNNL_ARG_KEYS] = mem_arg_k;
        args[DNNL_ARG_VALUES] = mem_arg_v;
        args[DNNL_ARG_DST] = mem_arg_dst;
        args[DNNL_ARG_SCALE] = mem_arg_scale;
        args[DNNL_ARG_ATTN_MASK] = mem_arg_mask;

        return status::success;
    }

    status_t execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override {
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        thread_local_cache_t<execution_args_set_t> res_cache;
        execution_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        temporary_scratchpad_t scratchpad(
                memory_planner_.total_internal_temporary_size(), p_engine_,
                *g_alloc_);
        assertm(scratchpad.size()
                        >= memory_planner_.total_internal_temporary_size(),
                "not enough scratchpad memory");
        prepare_args_set(res, inputs, outputs, scratchpad);

        memory mem_storage[6];
        exec_args_t args;
        CHECK(get_prim_exec_args(args, mem_storage, res));
        exec_ctx_t ctx(p_stream.get(), std::move(args));

        return cfg_.sdpa_prim_->execute(ctx);
    }

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {

        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        thread_local_cache_t<execution_args_set_t> res_cache;
        execution_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        temporary_scratchpad_t scratchpad(
                memory_planner_.total_internal_temporary_size(), p_engine_,
                *g_alloc_);
        assertm(scratchpad.size()
                        >= memory_planner_.total_internal_temporary_size(),
                "not enough scratchpad memory");
        prepare_args_set(res, inputs, outputs, scratchpad);

        memory mem_storage[6];
        exec_args_t args;
        CHECK(get_prim_exec_args(args, mem_storage, res));
        exec_ctx_t ctx(p_stream.get(), std::move(args));

        auto *sycl_stream = dnnl::impl::utils::downcast<
                dnnl::impl::sycl::sycl_stream_t *>(p_stream.get());

        sycl_stream->before_exec_hook();

        if (!sycl_deps.empty()) sycl_stream->sycl_ctx().set_deps(sycl_deps);

        auto status = cfg_.sdpa_prim_->execute(ctx);

        auto return_event = sycl_stream->get_output_event();

        scratchpad.set_deps(return_event);
        if (sycl_event) *sycl_event = return_event;

        sycl_stream->after_exec_hook();

        return status;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &cl_deps,
            cl_event *ret_event) override {

        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        thread_local_cache_t<execution_args_set_t> res_cache;
        execution_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        temporary_scratchpad_t scratchpad(
                memory_planner_.total_internal_temporary_size(), p_engine_,
                *g_alloc_);
        assertm(scratchpad.size()
                        >= memory_planner_.total_internal_temporary_size(),
                "not enough scratchpad memory");
        prepare_args_set(res, inputs, outputs, scratchpad);

        memory mem_storage[6];
        exec_args_t args;
        CHECK(get_prim_exec_args(args, mem_storage, res));
        exec_ctx_t ctx(p_stream.get(), std::move(args));

        // TODO (pc): refactor
        namespace ocl = gpu::intel::ocl;
        auto *ocl_stream = dnnl::impl::utils::downcast<ocl::ocl_stream_t *>(
                p_stream.get());

        ocl_stream->before_exec_hook();

        if (!cl_deps.empty()) {
            std::vector<xpu::ocl::wrapper_t<cl_event>> events(cl_deps.size());
            for (size_t i = 0; i < cl_deps.size(); i++)
                events[i] = xpu::ocl::wrapper_t<cl_event>(cl_deps[i], true);
            ocl_stream->ocl_ctx().set_deps(events);
        }

        auto status = cfg_.sdpa_prim_->execute(ctx);

        cl_event return_event = nullptr;
        if ((ocl_stream->flags() & stream_flags::in_order) == 0) {
            auto last = ocl_stream->get_output_event();
            return_event = last.release();
        }

        scratchpad.set_deps(return_event);
        if (ret_event) *ret_event = return_event;

        ocl_stream->after_exec_hook();

        return status;
    }
#endif
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
