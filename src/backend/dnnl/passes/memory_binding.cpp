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

#include <memory>
#include <vector>

#include "interface/c_types_map.hpp"
#include "interface/value.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/passes/memory_binding.hpp"
#include "backend/dnnl/passes/op_executable.hpp"
#include "backend/dnnl/passes/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using ltw = impl::logical_tensor_wrapper;

static memory bind_memory_for_value(value_t *val, const dnnl::engine &p_engine,
        execution_args_mgr &exec_arg_mgr) {
    memory mem;
    if (!exec_arg_mgr.find_value_mem_map(val, mem)) {
        mem = make_dnnl_memory(make_dnnl_memory_desc(val->get_logical_tensor()),
                p_engine, nullptr);
        exec_arg_mgr.add_value_mem_map({val, mem});

        bool is_alias = false;
        if (val->has_producer() && is_preprocess_op(val->get_producer())) {
            is_alias = true;
        }

        if (!is_alias) {
            if (ltw(val->get_logical_tensor()).property_type()
                    == impl::property_type::constant) {
                exec_arg_mgr.add_internal_constant_mem(mem);
            } else {
                exec_arg_mgr.add_internal_variable_mem(mem);
            }
        }
    }
    return mem;
}

static void bind_memory_for_conv_and_matmul(op_ptr &op,
        const dnnl::engine &p_engine, execution_args_mgr &exec_arg_mgr,
        primitive_attr_mgr &prm_attr_mgr) {
    int64_t key = exec_arg_mgr.init_args();
    op->set_attr<int64_t>("execution_args_key", key);
    auto &args = exec_arg_mgr.get_args(key);

    memory mem;
    size_t index = 0;

    // bind mem for inputs
    mem = bind_memory_for_value(
            op->get_input_value(index++).get(), p_engine, exec_arg_mgr);
    args.insert({DNNL_ARG_SRC, mem});

    mem = bind_memory_for_value(
            op->get_input_value(index++).get(), p_engine, exec_arg_mgr);
    args.insert({DNNL_ARG_WEIGHTS, mem});

    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        mem = bind_memory_for_value(
                op->get_input_value(index++).get(), p_engine, exec_arg_mgr);
        args.insert({DNNL_ARG_BIAS, mem});
    }

    dnnl::primitive_attr prm_attr = prm_attr_mgr.get_attr(
            op->get_attr<int64_t>("primitive_attr_key"));
    dnnl::post_ops pops = prm_attr.get_post_ops();
    for (int i = 0; i < pops.len(); i++) {
        if (pops.kind(i) == dnnl::primitive::kind::sum) {
            mem = bind_memory_for_value(
                    op->get_input_value(index++).get(), p_engine, exec_arg_mgr);
            args.insert({DNNL_GRAPH_ARG_POST_SRC, mem});
        } else if (pops.kind(i) == dnnl::primitive::kind::binary) {
            mem = bind_memory_for_value(
                    op->get_input_value(index++).get(), p_engine, exec_arg_mgr);
            args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, mem});
        } else {
        }
    }

    // bind mem for outputs
    mem = bind_memory_for_value(
            op->get_output_value(0).get(), p_engine, exec_arg_mgr);
    args.insert({DNNL_ARG_DST, mem});

    if (op->num_outputs() > 1) {
        mem = bind_memory_for_value(
                op->get_output_value(1).get(), p_engine, exec_arg_mgr);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }
}

// for single-input-single-output op
static void bind_memory_for_siso_op(op_ptr &op, const dnnl::engine &p_engine,
        execution_args_mgr &exec_arg_mgr, bool need_scratchpad = false,
        bool need_workspace = false) {
    int64_t key = exec_arg_mgr.init_args();
    op->set_attr<int64_t>("execution_args_key", key);
    auto &args = exec_arg_mgr.get_args(key);

    auto in_val = op->get_input_value(0);
    memory in_mem = bind_memory_for_value(in_val.get(), p_engine, exec_arg_mgr);
    args.insert({DNNL_ARG_FROM, in_mem});

    auto out_val = op->get_output_value(0);
    memory out_mem
            = bind_memory_for_value(out_val.get(), p_engine, exec_arg_mgr);
    args.insert({DNNL_ARG_TO, out_mem});

    if (need_scratchpad && op->num_outputs() > 1) {
        dnnl::memory mem = bind_memory_for_value(
                op->get_output_value(1).get(), p_engine, exec_arg_mgr);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    if (need_workspace && op->num_outputs() > 2) {
        dnnl::memory mem = bind_memory_for_value(
                op->get_output_value(2).get(), p_engine, exec_arg_mgr);
        args.insert({DNNL_ARG_WORKSPACE, mem});
    }
}

/// After doing infer shape, infer type and layout propagation passes, the
/// information of all logical tensors in the subgraph should be complete. We
/// can create dnnl::memory objects by using these logical tensors and nullptr
/// data handle. These memory object can be regard as placeholder, their data
/// handle will be set to valid buffer at execution stage.

/// We put all inputs and outputs memory objects of each op into a map. The key
/// of map is int variable, which is used to indicate each memory object's
/// semantic, such as src, dst, weights, ... The value of map is memory object.
/// The <int, memory> map will be set as an attributes of op. We call this
/// operation as memory binding.

/// During doing memory binding, we divide these memory objects into three
/// categories:

/// 1. The memory objects that correspond to partition's inputs and outputs.
/// These memory objects will be filled with valid buffer by user at the start
/// of execution. We call this kind of memory object as `external memory`.

/// 2. The memory objects that are used as scratchpad, such as memory to store
/// optimal layout weights or src/dst. These memory objects will not be filled
/// by user, we have to allocate memory for them by using given allocator. We
/// call this kind of memory object as `internal memory`.

/// 3. The memory objects that share same underlying buffer with above two kind
/// of memory objects. This kind of memory objects are usually used as output of
/// preprocess op, which will only re-parse the existing inputs buffer with
/// preprocessed memory descriptor instead of generating new data. We call this
/// kind of memory object as `shadow memory`.
impl::status_t memory_binding(std::vector<op_ptr> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs,
        const dnnl::engine &p_engine, execution_args_mgr &exec_arg_mgr,
        primitive_attr_mgr &prm_attr_mgr) {
    impl::graph_t tmp_graph(subgraph);
    std::vector<value_t *> in_vals = tmp_graph.get_input_values();
    std::vector<value_t *> out_vals = tmp_graph.get_output_values();

    // bind external memory
    for (size_t i = 0; i < inputs.size(); i++) {
        auto mem = make_dnnl_memory(
                make_dnnl_memory_desc(inputs[i]), p_engine, nullptr);
        exec_arg_mgr.add_external_input_mem(mem);
        for (auto in_val : in_vals) {
            if (in_val->get_logical_tensor().id == inputs[i].id)
                exec_arg_mgr.add_value_mem_map({in_val, mem});
        }
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto mem = make_dnnl_memory(
                make_dnnl_memory_desc(outputs[i]), p_engine, nullptr);
        exec_arg_mgr.add_external_output_mem(mem);
        for (auto out_val : out_vals) {
            if (out_val->get_logical_tensor().id == outputs[i].id)
                exec_arg_mgr.add_value_mem_map({out_val, mem});
        }
    }

    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() == impl::op_kind::Convolution
                || cur_op->get_kind() == op_kind::dnnl_convolution
                || cur_op->get_kind() == impl::op_kind::MatMul) {
            bind_memory_for_conv_and_matmul(
                    cur_op, p_engine, exec_arg_mgr, prm_attr_mgr);
        } else if (cur_op->get_kind() == impl::op_kind::MaxPool
                || cur_op->get_kind() == impl::op_kind::AvgPool
                || cur_op->get_kind() == op_kind::dnnl_pool) {
            const bool is_training = cur_op->has_attr("is_training")
                    ? cur_op->get_attr<bool>("is_training")
                    : false;
            bind_memory_for_siso_op(
                    cur_op, p_engine, exec_arg_mgr, true, is_training);
        } else if (cur_op->get_kind() == impl::op_kind::Reorder
                || cur_op->get_kind() == op_kind::mul_scales
                || cur_op->get_kind() == op_kind::permute
                || cur_op->get_kind() == op_kind::to_group
                || cur_op->get_kind() == op_kind::expand
                || cur_op->get_kind() == op_kind::dnnl_u8_to_s8) {
            bind_memory_for_siso_op(cur_op, p_engine, exec_arg_mgr);
        } else {
            assertm(false, "memory binding: unsupported op");
            return impl::status::compile_fail;
        }
    }

    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
