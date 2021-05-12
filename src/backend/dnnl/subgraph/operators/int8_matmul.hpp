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

#ifndef BACKEND_DNNL_SUBGRAPH_OPERATORS_INT8_MATMUL_HPP
#define BACKEND_DNNL_SUBGRAPH_OPERATORS_INT8_MATMUL_HPP

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"
#include "backend/dnnl/legacy.hpp"
#include "backend/dnnl/subgraph/compile_ops.hpp"
#include "backend/dnnl/subgraph/infer_type.hpp"
#include "backend/dnnl/subgraph/layout_propagation.hpp"
#include "backend/dnnl/subgraph/memory_binding.hpp"
#include "backend/dnnl/subgraph/op_executable.hpp"
#include "backend/dnnl/subgraph/passes.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

struct quantized_matmul : public dnnl::matmul, public kernel_base {
private:
    dnnl::engine p_engine_;
    dnnl::stream p_stream_;
    impl::allocator_t *g_alloc_;

    std::vector<impl::op_t *> subgraph_;

    primitive_attr_mgr prm_attr_mgr_;
    execution_args_mgr exec_arg_mgr_;
    executable_mgr exec_mgr_;

    std::vector<op_executable *> execs_;
    std::vector<exec_args> exec_args_;

    char *val2 = std::getenv("DNNL_GRAPH_HOLD_MEMORY");
    bool enable_hold_memory_ = (val2 != nullptr && std::strcmp(val2, "1") == 0);

    bool first_iter_ {true};

    std::vector<std::shared_ptr<impl::op_t>> opt_subgraph_;

    char *internal_buf_ {nullptr};

public:
    virtual ~quantized_matmul() {
        if (!enable_hold_memory_ || !internal_buf_) return;
        allocator::free(internal_buf_, p_engine_, g_alloc_);
    }

    virtual impl::status_t compile_impl(const dnnl_partition_impl_t *part,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = g_engine->get_allocator();

        std::vector<std::shared_ptr<op_t>> subgraph = part->get_ops();

        fuse_bias_add(subgraph);

        // check if bias exists
        check_with_bias(subgraph);

        // have to set the given inputs and outputs before infer shape and
        // compile
        //
        // FIXME(wuxun): currently, matmul op need to check ndims when decide
        // whether to insert transpose/expand op into subgraph, however
        // PyTorch doesn't have ndims info when adding op to graph, so here
        // directly used the input logical tensors passed from compilation stage
        set_given_inputs_outputs(subgraph, inputs, outputs);

        // split quant/dequant to pairs of mul_scales and add_zps
        split_quant_dequant(subgraph);
        fuse_to_int8_matmul(subgraph);
        folding_mul_scales(subgraph);

        // construct fused matmul attr
        fuse_output_scales(subgraph, prm_attr_mgr_);
        BACKEND_DNNL_CHECK(fuse_post_ops(subgraph, prm_attr_mgr_));
        fuse_zero_points(subgraph, prm_attr_mgr_);

        insert_transpose_for_matmul(subgraph);
        insert_expand_for_matmul(subgraph);
        insert_reorder(subgraph);

        impl::graph_t agraph(subgraph);
        BACKEND_DNNL_CHECK(agraph.infer_shape());
        BACKEND_DNNL_CHECK(infer_type(agraph));
        BACKEND_DNNL_CHECK(
                layout_propagation(subgraph, p_engine_, prm_attr_mgr_));

        // fill layout information for outputs logical tensors
        for (size_t i = 0; i < outputs.size(); i++) {
            for (auto out_val : impl::graph_t(subgraph).get_output_values()) {
                auto compiled_lt = out_val->get_logical_tensor();
                if (compiled_lt.id == outputs[i].id) {
                    auto lt = const_cast<impl::logical_tensor_t *>(&outputs[i]);
                    auto md = make_dnnl_memory_desc(compiled_lt);
                    fill_layout_info(lt, md);
                }
            }
        }

        // bind the memory for each op
        BACKEND_DNNL_CHECK(memory_binding(subgraph, inputs, outputs, p_engine_,
                exec_arg_mgr_, prm_attr_mgr_));

        BACKEND_DNNL_CHECK(
                compile_ops(subgraph, p_engine_, prm_attr_mgr_, exec_mgr_));

        // topologically sort the prms and their args
        impl::topo_order_visit(impl::graph_t(subgraph).get_output_ops(),
                [this](impl::op_t *op) {
                    auto exec_key = op->get_attr<int64_t>("executable_key");
                    auto &exec = exec_mgr_.get_executable(exec_key);
                    execs_.emplace_back(exec.get());

                    auto arg_key = op->get_attr<int64_t>("execution_args_key");
                    auto &args = exec_arg_mgr_.get_args(arg_key);
                    exec_args_.emplace_back(args);
                });

        opt_subgraph_ = subgraph;

        return impl::status::success;
    }

    virtual impl::status_t execute_impl(const dnnl_partition_impl_t *part,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(part);
        if (first_iter_) { p_stream_ = make_dnnl_stream(p_engine_, *g_stream); }

        // update the data of partition in/outputs args
        for (size_t i = 0; i < inputs.size(); i++) {
            exec_arg_mgr_.get_external_input_mem(i).set_data_handle(
                    inputs[i].get_data_handle());
        }
        for (size_t i = 0; i < outputs.size(); i++) {
            exec_arg_mgr_.get_external_output_mem(i).set_data_handle(
                    outputs[i].get_data_handle());
        }

        // allocate buffer for internal memory object
        if (!enable_hold_memory_ || first_iter_) {
            internal_buf_ = (char *)allocator::malloc(
                    exec_arg_mgr_.get_internal_mem_size(), p_engine_, g_alloc_);
            char *start = internal_buf_;
            for (auto &mem : exec_arg_mgr_.get_internal_mems()) {
                mem.set_data_handle((void *)start);
                start += mem.get_desc().get_size();
            }
        }

        for (size_t i = 0; i < execs_.size(); i++) {
            execs_[i]->execute(p_stream_, exec_args_[i]);
        }

        // free buffer for internal memory object
        if (!enable_hold_memory_) {
            allocator::free(internal_buf_, p_engine_, g_alloc_);
        }

        first_iter_ = false;
        return impl::status::success;
    }

    virtual impl::status_t prepare_inplace_pairs_impl(
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(g_engine);

        op_t *matmul_op = nullptr;
        for (auto &op : opt_subgraph_) {
            if (op->get_kind() == op_kind::MatMul) {
                matmul_op = op.get();
                break;
            }
        }

        bool with_sum = matmul_op->has_attr("with_sum")
                ? matmul_op->get_attr<bool>("with_sum")
                : false;

        if (with_sum) {
            // post_src should always be the last one input of matmul op
            auto post_src_val
                    = matmul_op->get_input_value(matmul_op->num_inputs() - 1);
            size_t post_src_id = post_src_val->get_logical_tensor().id;

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

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
