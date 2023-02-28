/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "compiler/config/context.hpp"
#include "compiler/ir/graph/driver.hpp"
#include "compiler/ir/graph/dynamic_utils.hpp"
#include "compiler/ir/graph/pass/pass.hpp"
#include "compiler_partition_impl.hpp"

#include "common/rw_mutex.hpp"
#include "graph/interface/graph.hpp"
#include "graph/utils/debug.hpp"
#include "graph/utils/utils.hpp"
#include "runtime/runtime.hpp"
#include "runtime/thread_locals.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {

static std::unordered_map<
        std::shared_ptr<graph::compiler_impl::compiler_graph_engine_t>, int>
        partition_count_map;
static std::unordered_map<const graph::engine_t *,
        std::shared_ptr<graph::compiler_impl::compiler_graph_engine_t>>
        engine_map;
static std::mutex global_mutex;

graph::status_t compiler_partition_impl_t::infer_shape(
        std::vector<const graph::logical_tensor_t *> &inputs,
        std::vector<graph::logical_tensor_t *> &outputs) const {
    std::lock_guard<std::mutex> lck(mtx_);
    // construct a temp graph
    copied_ops_ = graph::graph_t::deep_copy(ops_);
    graph::graph_t temp_graph(copied_ops_);
    auto output_ops = temp_graph.get_output_ops();
    auto ret = topo_order_visit(output_ops, [&](op_t *cur_op) {
        const graph::op_schema_t *cur_op_schema
                = graph::op_schema_registry_t::get_op_schema(
                        cur_op->get_kind());
        assertm(cur_op_schema, "Can't infer shape for cur op: no schema");
        auto get_logical_tensor = [&](const std::shared_ptr<value_t> &val)
                -> graph::logical_tensor_t {
            logical_tensor_t lt = val->get_logical_tensor();
            auto in_pos = std::find_if(inputs.begin(), inputs.end(),
                    [&](const graph::logical_tensor_t *alt) -> bool {
                        return alt->id == lt.id;
                    });
            if (in_pos != inputs.end()) { return **in_pos; }
            return lt;
        };
        graph::op_t temp_node = graph::op_t(cur_op->get_kind());
        temp_node.merge_attributes(cur_op->get_attributes());
        std::vector<graph::logical_tensor_t> ordered_inputs_holder
                = utils::func_map(
                        cur_op->get_input_values(), get_logical_tensor);
        std::vector<graph::logical_tensor_t> ordered_outputs_holder
                = utils::func_map(
                        cur_op->get_output_values(), get_logical_tensor);
        std::vector<graph::logical_tensor_t *> ordered_inputs;
        ordered_inputs.reserve(ordered_inputs_holder.size());
        for (auto &tsr : ordered_inputs_holder) {
            assert(tsr.layout_type == graph::layout_type::strided);
            ordered_inputs.emplace_back(&tsr);
        }
        std::vector<graph::logical_tensor_t *> ordered_outputs;
        ordered_outputs.reserve(ordered_outputs_holder.size());
        for (auto &tsr : ordered_outputs_holder) {
            ordered_outputs.emplace_back(&tsr);
        }
        graph::status_t ret = cur_op_schema->shape_infer(
                &temp_node, ordered_inputs, ordered_outputs);
        if (ret != graph::status::success) return ret;
        for (size_t i = 0; i < cur_op->get_output_values().size(); ++i) {
            auto output_lt = *ordered_outputs[i];
            auto cur_val = cur_op->get_output_values()[i];
            cur_val->set_logical_tensor(output_lt);
            // TODO(yifei): move the logic into compile() stage
            // to let compiler backend decide the optimized layout after
            // layout_propagation
            if (output_lt.layout_type != graph::layout_type::strided) {
                // force set strided layout
                graph::dims shape(
                        output_lt.dims, output_lt.dims + output_lt.ndims);
                graph::dims strides = utils::get_dense_strides(shape);
                cur_val->set_strides(strides);
            }
            auto out_pos = std::find_if(outputs.begin(), outputs.end(),
                    [&](graph::logical_tensor_t *alt) -> bool {
                        return alt->id == ordered_outputs[i]->id;
                    });
            if (out_pos != outputs.end()) {
                **out_pos = cur_val->get_logical_tensor();
            }
        }
        return graph::status::success;
    });
    return ret;
}

graph::status_t compiler_partition_impl_t::compile(
        graph::compiled_partition_t *compiled_partition,
        const std::vector<graph::logical_tensor_t> &inputs,
        const std::vector<graph::logical_tensor_t> &outputs,
        const graph::engine_t *aengine) const {
    try {
        graph::status_t res = status::success;
        // here we call infer_shape since logical tensor info
        // may be incomplete for the graph corresponding to the
        // partition
        std::vector<const graph::logical_tensor_t *> input_ref;
        std::vector<graph::logical_tensor_t *> output_ref;
        input_ref.reserve(inputs.size());
        output_ref.reserve(outputs.size());
        for (auto &t : inputs) {
            input_ref.push_back(const_cast<graph::logical_tensor_t *>(&t));
        }
        for (auto &t : outputs) {
            output_ref.push_back(const_cast<graph::logical_tensor_t *>(&t));
        }
        res = this->infer_shape(input_ref, output_ref);
        if (res != status::success) { return res; }

        std::lock_guard<std::mutex> lck(mtx_);
        std::vector<gc::runtime::dynamic_tensor_t> dyn_inputs, dyn_outputs;
        std::unordered_map<size_t, gc::sc_op_ptr> inputs_map, outputs_map;
        std::vector<gc::sc_op_ptr> sc_inputs;
        compiler_graph_impl_t sub_graph;
        size_t id = 0;
        std::vector<size_t> out_lt_ids(outputs.size());
        sc_inputs.reserve(inputs.size());
        bool is_dynamic = false;
        for (auto &in_lt : inputs) {
            gc::sc_op_ptr in_ret;
            in_ret = sub_graph.make_compiler_backend_input(in_lt);
            if (!is_dynamic && sub_graph.is_dynamic()) { is_dynamic = true; }
            inputs_map[in_lt.id] = in_ret;
            sc_inputs.emplace_back(in_ret);
            in_ret->attrs_.set("unique_id", id++);
            in_ret->attrs_.set(
                    "temp.name", "logical_tensor_" + std::to_string(in_lt.id));
        }
        if (is_dynamic) {
            dyn_inputs.resize(inputs.size());
            dyn_outputs.reserve(outputs.size());
            std::transform(sc_inputs.begin(), sc_inputs.end(),
                    dyn_inputs.begin(), [](const gc::sc_op_ptr &in) {
                        return gc::convert_graph_tensor_to_dynamic_tensor(
                                in->get_outputs()[0]);
                    });
        }
        for (auto &out_lt : outputs) {
            out_lt_ids.emplace_back(out_lt.id);
        }
        graph::graph_t temp_graph(copied_ops_);
        auto output_ops = temp_graph.get_output_ops();
        graph::status_t status;
        status = topo_order_visit(output_ops, [&](op_t *cur_op) {
            std::vector<gc::graph_tensor_ptr> producer_lt, consumer_lt;
            // translate input value
            for (auto &in_value : cur_op->get_input_values()) {
                auto backend_op = in_value->has_producer()
                        ? sub_graph.op_mapping_[in_value->get_producer()
                                                        .get_id()]
                        : nullptr;
                if (backend_op) {
                    producer_lt.emplace_back(
                            backend_op->get_info()
                                    .outputs_[in_value->get_offset()]);
                } else {
                    auto lt = in_value->get_logical_tensor();
                    assert(inputs_map.find(lt.id) != inputs_map.end());
                    producer_lt.emplace_back(
                            inputs_map[lt.id]->get_outputs()[0]);
                }
            }
            // Get consumer lt
            if (!is_dynamic) {
                for (auto &out_value : cur_op->get_output_values()) {
                    auto lt = out_value->get_logical_tensor();
                    consumer_lt.emplace_back(
                            compiler_graph_impl_t::convert_logical_tensor(lt));
                }
            }
            // translate op
            gc::sc_op_ptr ret;
            ret = sub_graph.make_backend_op(cur_op, producer_lt, consumer_lt);
            if (!ret) { return impl::status::unimplemented; }
            // translate output value
            for (size_t i = 0; i < cur_op->get_output_values().size(); i++) {
                auto &out_value = cur_op->get_output_values()[i];
                auto lt = out_value->get_logical_tensor();

                if (std::find(out_lt_ids.begin(), out_lt_ids.end(), lt.id)
                        != out_lt_ids.end()) {
                    auto out_ret
                            = sub_graph.make_output({ret->get_outputs()[i]});
                    if (is_dynamic) {
                        auto out_dyn_tsr
                                = gc::convert_graph_tensor_to_dynamic_tensor(
                                        out_ret->get_inputs()[0]);
                        dyn_outputs.emplace_back(out_dyn_tsr);
                    }
                    out_ret->attrs_.set("unique_id", id++);
                    out_ret->attrs_.set("temp.name",
                            "logical_tensor_" + std::to_string(lt.id));
                    gc::sc_data_format_t output_format
                            = out_ret->get_inputs()[0]->details_.get_format();
                    gc::sc_dims output_strides
                            = out_ret->get_inputs()[0]->details_.get_strides();
                    if (!output_format.is_any()) {
                        out_ret->attrs_.set("target_formats",
                                std::vector<gc::sc_data_format_t> {
                                        output_format});
                        out_ret->attrs_.set("target_strides",
                                std::vector<gc::sc_dims> {output_strides});
                    }
                    outputs_map[lt.id] = out_ret;
                }
            }
            sub_graph.op_mapping_[cur_op->get_id()] = ret;
            return graph::status::success;
        });
        if (status != graph::status::success) return status;

        gc::sc_graph_t &backend_graph_obj
                = *dynamic_cast<gc::sc_graph_t *>(&sub_graph);
        if (!gc::check_graph_connection(backend_graph_obj)) {
            return graph::status::invalid_graph;
        }
        backend_graph_obj.attrs_["temp.name"]
                = pname_ + "_" + std::to_string(this->id_);

        COMPILE_ASSERT(aengine->kind() == graph::engine_kind_t::dnnl_cpu,
                "Graph compiler backend only supports cpu engine");
        gc::context_ptr ctx;
        ctx = gc::get_default_context();
        std::shared_ptr<compiler_graph_engine_t> graph_engine;
        {
            std::lock_guard<std::mutex> lock(global_mutex);
            auto iter = engine_map.find(aengine);
            if (iter != engine_map.end()) {
                graph_engine = iter->second;
            } else {
                graph_engine = std::make_shared<compiler_graph_engine_t>(
                        &graph_engine_vtable,
                        static_cast<allocator_t *>(aengine->get_allocator()));
                engine_map[aengine] = graph_engine;
            }
        }
        // check engine
        auto &tls_buffer = gc::runtime::thread_local_buffer_t::tls_buffer_;
        if (tls_buffer.engine_ && tls_buffer.engine_ != graph_engine.get()) {
            gc::release_runtime_memory(tls_buffer.engine_);
        }

        ctx->engine_ = static_cast<gc::runtime::engine_t *>(graph_engine.get());

        gc::graph_driver(backend_graph_obj, 28, 10, ctx);

        std::vector<gc::sc_op_ptr> args;
        for (auto &out_lt : outputs) {
            for (const auto &op : backend_graph_obj.get_output_ops()) {
                if (op->attrs_.get<size_t>("unique_id")
                        == outputs_map[out_lt.id]->attrs_.get<size_t>(
                                "unique_id")) {
                    args.push_back(op);
                    break;
                }
            }
        }
        for (auto &in_lt : inputs) {
            for (const auto &op : backend_graph_obj.get_input_ops()) {
                if (op->attrs_.get<size_t>("unique_id")
                        == inputs_map[in_lt.id]->attrs_.get<size_t>(
                                "unique_id")) {
                    args.push_back(op);
                    break;
                }
            }
        }
        gc::ir_module_ptr ir_mod
                = gc::lower_graph(ctx, backend_graph_obj, args);

        std::shared_ptr<gc::jit_function_t> fptr
                = gc::jit_engine_t::make(ctx)->get_entry_func(ir_mod, true);
        auto pimpl = std::make_shared<compiler_compiled_partition_impl_t>(
                *aengine, inputs, outputs, fptr, graph_engine,
                std::move(dyn_inputs), std::move(dyn_outputs));
        compiled_partition->init(pimpl);
        return res;
    } catch (...) { return graph::status::unimplemented; }
}

std::shared_ptr<graph::partition_impl_t>
compiler_partition_impl_t::clone() const {
    auto ret = std::make_shared<compiler_partition_impl_t>(
            get_engine_kind(), get_fpmath_mode(), get_kind(), get_name());
    ret->ops_ = graph::graph_t::deep_copy(ops_);
    ret->inputs_ = inputs_;
    ret->outputs_ = outputs_;
    ret->id_ = id_;
    ret->is_init_ = is_init_;
    return ret;
}

bool compiler_partition_impl_t::is_initialized() const {
    return is_init_;
}

compiler_compiled_partition_impl_t::compiler_compiled_partition_impl_t(
        const graph::engine_t &engine,
        const std::vector<graph::logical_tensor_t> &inputs,
        const std::vector<graph::logical_tensor_t> &outputs,
        const std::shared_ptr<gc::jit_function_t> &jit_func,
        const std::shared_ptr<graph::compiler_impl::compiler_graph_engine_t>
                &graph_engine,
        std::vector<gc::runtime::dynamic_tensor_t> &&dyn_inputs,
        std::vector<gc::runtime::dynamic_tensor_t> &&dyn_outputs)
    : graph::compiled_partition_impl_t(engine, inputs, outputs, {})
    , jit_func_(jit_func)
    , graph_engine_(graph_engine)
    , dyn_inputs_(std::move(dyn_inputs))
    , dyn_outputs_(std::move(dyn_outputs)) {
    std::lock_guard<std::mutex> lock(global_mutex);
    partition_count_map[graph_engine_]++;
    graph_engine_->allocator_->retain();
}

compiler_compiled_partition_impl_t::~compiler_compiled_partition_impl_t() {
    std::lock_guard<std::mutex> lock(global_mutex);
    auto itr = partition_count_map.find(graph_engine_);
    if (itr != partition_count_map.end()) {
        itr->second--;
        if (itr->second == 0) {
            gc::release_runtime_memory(graph_engine_.get());
            for (auto iter = engine_map.begin(); iter != engine_map.end();) {
                if (iter->second == graph_engine_) {
                    iter = engine_map.erase(iter);
                } else {
                    ++iter;
                }
            }
        }
    }
    jit_func_ = nullptr;
    graph_engine_->allocator_->release();
}

graph::status_t compiler_compiled_partition_impl_t::execute(
        const graph::stream_t *astream,
        const std::vector<graph::tensor_t> &inputs,
        const std::vector<graph::tensor_t> &outputs) {
    // set backend runtime stream
    compiler_graph_stream_t backend_stream {graph_engine_.get()};
    std::vector<gc::generic_val> generic_args;
    if (dyn_inputs_.empty()) {
        generic_args.reserve(inputs.size() + outputs.size());
        for (auto out_tensor : outputs) {
            generic_args.emplace_back(out_tensor.get_data_handle());
        }
        for (auto in_tensor : inputs) {
            generic_args.emplace_back(in_tensor.get_data_handle());
        }
    } else {
        generic_args.resize(inputs.size() + outputs.size());
        auto trans_func = [](const graph::tensor_t &tsr,
                                  gc::runtime::dynamic_tensor_t &dyn) {
            dyn.data_ = tsr.get_data_handle();
            dyn.dims_ = const_cast<gc::sc_dim *>(
                    reinterpret_cast<const gc::sc_dim *>(
                            tsr.get_logical_tensor().dims));
            return &dyn;
        };
        auto arg_it = std::transform(outputs.begin(), outputs.end(),
                dyn_outputs_.begin(), generic_args.begin(), trans_func);
        std::transform(inputs.begin(), inputs.end(), dyn_inputs_.begin(),
                arg_it, trans_func);
    }
    jit_func_->call_generic(&backend_stream, generic_args.data());
    return status::success;
}
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
