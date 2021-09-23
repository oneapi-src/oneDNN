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

#ifndef BACKEND_DNNL_KERNELS_MATMUL_HPP
#define BACKEND_DNNL_KERNELS_MATMUL_HPP

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

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

template <bool quantized>
struct matmul : public kernel_base {
private:
    dnnl::engine p_engine_;
    impl::allocator_t *g_alloc_;

    primitive_attr_mgr prm_attr_mgr_;
    executable_mgr exec_mgr_;
    memory_planner_t memory_planner_;

    std::vector<op_executable *> execs_;

    std::vector<std::shared_ptr<impl::op_t>> opt_subgraph_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

    // FIXME(qun) improve the cache key
    constant_cache_t::key_t constant_key_
            = reinterpret_cast<constant_cache_t::key_t>(this);

    bool enable_constant_cache_ = utils::is_enable_constant_cache();

    std::vector<bool> is_constant_;

    pd_cache_t pd_cache_;

public:
    virtual ~matmul() {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));

        if (enable_constant_cache_) {
            constant_cache_t constant_cache;
            constant_cache.remove_if_exist(constant_key_);
        }
    }

    virtual impl::status_t compile_impl(const dnnl_partition_impl_t *part,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = g_engine->get_allocator();

        std::vector<std::shared_ptr<op_t>> subgraph = part->get_ops();

        if (quantized) {
            fuse_typecast_to_matmul(subgraph);
            fuse_typecast_to_add(subgraph);
            fuse_post_typecast_to_matmul(subgraph);
        }

        set_all_layout_to_any(subgraph);

        // have to set the given inputs and outputs before infer shape and
        // compile
        //
        // FIXME(wuxun): currently, matmul op need to check ndims when decide
        // whether to insert transpose/expand op into subgraph, however
        // PyTorch doesn't have ndims info when adding op to graph, so here
        // directly used the input logical tensors passed from compilation stage
        BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph, inputs, outputs));

        fuse_bias_add(subgraph);
        // check if bias exists
        check_with_bias(subgraph);
        fuse_mul_sigmoid_to_swish(subgraph);

        BACKEND_DNNL_CHECK(impl::graph_t(subgraph).infer_shape());

        if (quantized) {
            // split quant/dequant to pairs of mul_scales and add_zps
            split_quant_dequant(subgraph);
            BACKEND_DNNL_CHECK(impl::graph_t(subgraph).infer_shape());
            fuse_to_int8_matmul(subgraph);
            folding_mul_scales(subgraph);
            fuse_output_scales(subgraph, prm_attr_mgr_);
        }

        BACKEND_DNNL_CHECK(fuse_post_ops(subgraph, prm_attr_mgr_));

        if (quantized) {
            fuse_zero_points(subgraph, prm_attr_mgr_);
            // fuse neighboring mul_scales and zdd_zps op to quantize/dequantize
            fuse_mul_scales_add_zps(subgraph);
        }

        insert_u8_to_s8_for_matmul(subgraph, prm_attr_mgr_);
        BACKEND_DNNL_CHECK(impl::graph_t(subgraph).infer_shape());
        insert_transpose_for_matmul(subgraph);
        BACKEND_DNNL_CHECK(impl::graph_t(subgraph).infer_shape());
        insert_expand_and_squeeze_for_matmul(subgraph);
        insert_reorder(subgraph);

        subgraph_visualizer_t vis(part->id());
        vis.run(subgraph, "after_lower_down", false);

        impl::graph_t agraph(subgraph);
        BACKEND_DNNL_CHECK(agraph.infer_shape());
        BACKEND_DNNL_CHECK(infer_type(agraph));
        // do constant propagation here so that we can
        // prepare constant info for other optimizations.
        if (enable_constant_cache_) { constant_propagation(subgraph, false); }

        vis.run(subgraph, "after_infer_shape_infer_type", true);

        BACKEND_DNNL_CHECK(layout_propagation(
                subgraph, p_engine_, prm_attr_mgr_, pd_cache_));

        vis.run(subgraph, "after_layout_propagation", true);

        // fill layout information for outputs logical tensors
        for (size_t i = 0; i < outputs.size(); i++) {
            for (auto out_val : impl::graph_t(subgraph).get_output_values()) {
                auto compiled_lt = out_val->get_logical_tensor();
                if (compiled_lt.id == outputs[i].id) {
                    auto lt = const_cast<impl::logical_tensor_t *>(&outputs[i]);
                    auto md = make_dnnl_memory_desc(compiled_lt);
                    lt->ndims = compiled_lt.ndims;
                    impl::utils::array_copy(
                            lt->dims, compiled_lt.dims, DNNL_GRAPH_MAX_NDIMS);
                    impl::utils::array_copy(lt->layout.strides,
                            compiled_lt.layout.strides, DNNL_GRAPH_MAX_NDIMS);
                    fill_layout_info(lt, md);
                }
            }
        }

        // do constant propagation again since layout propagation may
        // insert/delete operators
        if (enable_constant_cache_) { constant_propagation(subgraph); }

        BACKEND_DNNL_CHECK(compile_ops(
                subgraph, p_engine_, prm_attr_mgr_, exec_mgr_, pd_cache_));

        // bind the memory for each op
        BACKEND_DNNL_CHECK(memory_planner_.run(
                subgraph, inputs, outputs, p_engine_, prm_attr_mgr_));

        vis.run(subgraph, "after_memory_planning", true, true,
                [this](const value_t *val) {
                    return this->memory_planner_.get_memory_info(val);
                });

        // topologically sort the executables
        impl::topo_order_visit(impl::graph_t(subgraph).get_output_ops(),
                [this](impl::op_t *op) {
                    auto exec_key = op->get_attr<int64_t>("executable_key");
                    auto &exec = exec_mgr_.get_executable(exec_key);
                    execs_.emplace_back(exec.get());

                    is_constant_.push_back(op->has_attr("is_constant")
                            && op->get_attr<bool>("is_constant"));
                    return impl::status::success;
                });

        opt_subgraph_ = subgraph;

        resource_ctor_ = [this]() {
            return this->memory_planner_.get_exec_args_set().clone();
        };

        return impl::status::success;
    }

    virtual impl::status_t execute_impl(const dnnl_partition_impl_t *part,
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

        if (enable_constant_cache_) {
            std::promise<constant_cache_t::cached_t> c_promise;
            constant_cache_t global_constant_cache;
            constant_cache_t::value_t cached_value
                    = global_constant_cache.get_or_add(
                            constant_key_, c_promise.get_future());
            bool is_from_cache = cached_value.valid();
            if (is_from_cache) {
                constant_cache_t::cached_t c_buffer = cached_value.get();
                grantor_t c_grantor
                        = memory_planner_.internal_persistent_grantor(
                                c_buffer->data<char>());
                registry_t::key_t key = 0;
                for (auto &mem_offkey :
                        res->get_mems_use_internal_persistent()) {
                    mem_offkey.first.set_data_handle(
                            c_grantor.get(mem_offkey.second));
                }
            } else {
                constant_cache_t::cached_t c_buffer
                        = std::make_shared<constant_buffer_t>(
                                memory_planner_
                                        .total_internal_persistent_size(),
                                p_engine_, g_alloc_);
                grantor_t c_grantor
                        = memory_planner_.internal_persistent_grantor(
                                c_buffer->data<char>());
                registry_t::key_t key = 0;
                for (auto &mem_offkey :
                        res->get_mems_use_internal_persistent()) {
                    mem_offkey.first.set_data_handle(
                            c_grantor.get(mem_offkey.second));
                }

                for (size_t i = 0; i < execs_.size(); i++) {
                    if (!is_constant_[i]) continue;
                    execs_[i]->execute(p_stream, res->get_exec_args()[i]);
                }

                c_promise.set_value(c_buffer);
            }
        }

        for (size_t i = 0; i < execs_.size(); i++) {
            if (is_constant_[i]) continue;
            execs_[i]->execute(p_stream, res->get_exec_args()[i]);
        }

        return impl::status::success;
    }

    virtual impl::status_t prepare_inplace_pairs_impl(
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(g_engine);

        op_t *matmul_op = nullptr;
        for (auto &op : opt_subgraph_) {
            if (op->get_kind() == impl::op_kind::MatMul) {
                matmul_op = op.get();
                break;
            }
        }

        bool with_sum = matmul_op->has_attr("with_sum")
                ? matmul_op->get_attr<bool>("with_sum")
                : false;

        if (with_sum) {
            // post_src should always be the last one input of matmul op
            auto val = matmul_op->get_input_value(matmul_op->num_inputs() - 1);
            if (val->has_producer()
                    && val->get_producer().get_kind() == op_kind::expand) {
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

            const logical_tensor_wrapper post_src_lt(inputs[idx]);
            const logical_tensor_wrapper dst_lt(outputs[0]);
            if (post_src_lt.is_opaque() && dst_lt.is_opaque()
                    && post_src_lt.is_similar(dst_lt))
                inplace_pairs_.push_back({post_src_id, outputs[0].id});
        }
        return impl::status::success;
    }
};

using float_matmul = matmul</* quantized */ false>;
using quantized_matmul = matmul</* quantized */ true>;

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
