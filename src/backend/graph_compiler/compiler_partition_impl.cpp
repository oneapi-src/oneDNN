/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#include <string>
#include <vector>
#include <unordered_map>

#include "compiler/ir/graph/driver.hpp"
#include "compiler/ir/graph/pass/pass.hpp"
#include "compiler_partition_impl.hpp"

#include "interface/graph.hpp"
#include "runtime/runtime.hpp"
#include "utils.hpp"
#include "utils/debug.hpp"
#include "utils/rw_mutex.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {

static std::unordered_map<
        std::shared_ptr<impl::compiler_impl::compiler_graph_engine_t>, int>
        partition_count_map;

impl::status_t compiler_partition_impl_t::infer_shape(
        std::vector<const impl::logical_tensor_t *> &inputs,
        std::vector<impl::logical_tensor_t *> &outputs) const {
    std::lock_guard<std::mutex> lck(mtx_);
    // construct a temp graph
    copied_ops_ = impl::graph_t::deep_copy(ops_);
    impl::graph_t temp_graph(copied_ops_);
    auto output_ops = temp_graph.get_output_ops();
    topo_order_visit(output_ops, [&](op_t *cur_op) {
        const impl::op_schema_t *cur_op_schema
                = impl::op_schema_registry_t::get_op_schema(cur_op->get_kind());
        assertm(cur_op_schema, "Can't infer shape for cur op: no schema");
        auto get_logical_tensor = [&](const std::shared_ptr<value_t> &val)
                -> impl::logical_tensor_t {
            logical_tensor_t lt = val->get_logical_tensor();
            auto in_pos = std::find_if(inputs.begin(), inputs.end(),
                    [&](const impl::logical_tensor_t *alt) -> bool {
                        return alt->id == lt.id;
                    });
            if (in_pos != inputs.end()) { return **in_pos; }
            auto out_pos = std::find_if(outputs.begin(), outputs.end(),
                    [&](impl::logical_tensor_t *alt) -> bool {
                        return alt->id == lt.id;
                    });
            if (out_pos != outputs.end()) { return **out_pos; }
            return lt;
        };
        impl::op_t temp_node = impl::op_t(cur_op->get_kind());
        temp_node.merge_attributes(cur_op->get_attributes());
        std::vector<impl::logical_tensor_t> ordered_inputs_holder
                = utils::func_map(
                        cur_op->get_input_values(), get_logical_tensor);
        std::vector<impl::logical_tensor_t> ordered_outputs_holder
                = utils::func_map(
                        cur_op->get_output_values(), get_logical_tensor);
        std::vector<impl::logical_tensor_t *> ordered_inputs;
        ordered_inputs.reserve(ordered_inputs_holder.size());
        for (auto &tsr : ordered_inputs_holder) {
            ordered_inputs.emplace_back(&tsr);
        }
        std::vector<impl::logical_tensor_t *> ordered_outputs;
        ordered_outputs.reserve(ordered_outputs_holder.size());
        for (auto &tsr : ordered_outputs_holder) {
            ordered_outputs.emplace_back(&tsr);
        }
        impl::status_t ret = cur_op_schema->shape_infer(
                &temp_node, ordered_inputs, ordered_outputs);
        if (ret != impl::status::success) return ret;
        for (size_t i = 0; i < cur_op->get_output_values().size(); ++i) {
            cur_op->get_output_values()[i]->set_logical_tensor(
                    *ordered_outputs[i]);
        }
        return impl::status::success;
    });
    // start setting outputs
    for (auto &op : output_ops) {
        for (auto &val : op->get_output_values()) {
            auto out_pos = std::find_if(outputs.begin(), outputs.end(),
                    [&](impl::logical_tensor_t *alt) -> bool {
                        return alt->id == val->get_logical_tensor().id;
                    });
            if (out_pos != outputs.end()) {
                **out_pos = val->get_logical_tensor();
            }
        }
    }
    return impl::status::success;
}

impl::status_t compiler_partition_impl_t::compile(
        impl::compiled_partition_t *compiled_partition,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs,
        const impl::engine_t *aengine) const {
    try {
        // here we call infer_shape to avoid the case when
        // all input/output logical tensors are set
        // but intermedia logical tensors have unknown shape
        std::vector<const impl::logical_tensor_t *> input_ref;
        std::vector<impl::logical_tensor_t *> output_ref;
        input_ref.reserve(inputs.size());
        output_ref.reserve(outputs.size());
        for (auto &t : inputs) {
            input_ref.push_back(const_cast<impl::logical_tensor_t *>(&t));
        }
        for (auto &t : outputs) {
            output_ref.push_back(const_cast<impl::logical_tensor_t *>(&t));
        }
        this->infer_shape(input_ref, output_ref);

        std::lock_guard<std::mutex> lck(mtx_);
        impl::status_t res = status::success;
        std::unordered_map<size_t, sc::sc_op_ptr> inputs_map, outputs_map;
        compiler_graph_impl_t sub_graph;
        size_t id = 0;
        for (auto &in_lt : inputs) {
            sc::sc_op_ptr in_ret;
            in_ret = sub_graph.make_compiler_backend_input(in_lt);
            inputs_map[in_lt.id] = in_ret;
            in_ret->attrs_.set("unique_id", id++);
        }
        impl::graph_t temp_graph(copied_ops_);
        auto output_ops = temp_graph.get_output_ops();
        topo_order_visit(output_ops, [&](op_t *cur_op) {
            std::vector<sc::graph_tensor_ptr> producer_lt, consumer_lt;
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
            // translate output value
            for (auto &out_value : cur_op->get_output_values()) {
                consumer_lt.emplace_back(
                        compiler_graph_impl_t::convert_logical_tensor(
                                out_value->get_logical_tensor()));
            }

            sc::sc_op_ptr ret;
            ret = sub_graph.make_backend_op(cur_op, producer_lt, consumer_lt);
            if (utils::is_output_op(cur_op->shared_from_this())) {
                for (size_t i = 0; i < consumer_lt.size(); ++i) {
                    auto out_ret = sub_graph.make_output({consumer_lt[i]});
                    out_ret->attrs_.set("unique_id", id++);
                    sc::sc_data_format_t output_format
                            = out_ret->get_inputs()[0]->details_.get_format();
                    if (!output_format.is_any()) {
                        out_ret->attrs_.set("target_formats",
                                std::vector<sc::sc_data_format_t> {
                                        output_format});
                    }
                    outputs_map[cur_op->get_output_values()[i]
                                        ->get_logical_tensor()
                                        .id]
                            = out_ret;
                }
            }
            sub_graph.op_mapping_[cur_op->get_id()] = ret;
            return impl::status::success;
        });
        sc::sc_graph_t &backend_graph_obj
                = *dynamic_cast<sc::sc_graph_t *>(&sub_graph);
        if (!sc::check_graph_connection(backend_graph_obj)) {
            return impl::status::invalid_graph;
        }

        COMPILE_ASSERT(aengine->kind() == impl::engine_kind_t::dnnl_graph_cpu,
                "Graph compiler backend only supports cpu engine");
        sc::context_ptr ctx;
        ctx = sc::get_default_context();
        auto graph_engine = std::make_shared<compiler_graph_engine_t>(
                &graph_engine_vtable, aengine->get_allocator());
        ctx->engine_ = static_cast<sc::runtime::engine_t *>(graph_engine.get());

        sc::graph_driver(backend_graph_obj, 28, 10, ctx);

        std::vector<sc::sc_op_ptr> args;
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
        sc::ir_module_ptr ir_mod
                = sc::lower_graph(ctx, backend_graph_obj, args);

        std::shared_ptr<sc::jit_function_t> fptr
                = sc::jit_engine_t::make(ctx)->get_entry_func(ir_mod, true);
        auto pimpl = std::make_shared<compiler_compiled_partition_impl_t>(
                *aengine, inputs, outputs, fptr, graph_engine);
        compiled_partition->init(pimpl);
        return res;
    } catch (...) { return impl::status::compile_fail; }
}

std::shared_ptr<impl::partition_impl_t> compiler_partition_impl_t::clone() {
    return std::make_shared<compiler_partition_impl_t>(*this);
}

compiler_partition_impl_t::compiler_partition_impl_t(
        const compiler_partition_impl_t &other)
    : impl::partition_impl_t(other) {
    is_init_ = other.is_init_;
}

bool compiler_partition_impl_t::is_initialized() const {
    return is_init_;
}

std::string compiler_partition_impl_t::to_string() const {
    std::ostringstream os;

    const auto dims_to_string = [&](const std::vector<int64_t> &dims) {
        std::ostringstream oss;
        oss << "(";
        const char *delimer = "";
        for (const auto &d : dims) {
            oss << delimer << d;
            delimer = "x";
        }
        oss << ")";
        return oss.str();
    };

    for (const auto &op : ops_) {
        os << " [ op: (";
        if (op) {
            os << "ID: " << op->get_id()
               << ", kind: " << impl::op_t::kind2str(op->get_kind()) << " ), ";
        }
    }
    os << " ] \n";

    os << "  [ inputs: ";
    const char *delimer = "";
    for (const auto &i : inputs_) {
        const impl::logical_tensor_wrapper_t v(i);
        os << delimer << "(ID: " << v.id() << "("
           << impl::utils::data_type2str(v.data_type()) << ":"
           << dims_to_string(v.vdims());
        delimer = ")), ";
    }
    os << " ]\n";

    os << "  [ outputs: ";
    delimer = "";
    for (const auto &o : outputs_) {
        const impl::logical_tensor_wrapper_t v(o);
        os << delimer << "(ID: " << v.id() << "("
           << impl::utils::data_type2str(v.data_type()) << ":"
           << dims_to_string(v.vdims());
        delimer = ")), ";
    }
    os << " ]\n";
    os << " ]\n";
    os << "]";

    return os.str();
}

compiler_compiled_partition_impl_t::compiler_compiled_partition_impl_t(
        const impl::engine_t &engine,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs,
        const std::shared_ptr<sc::jit_function_t> &jit_func,
        const std::shared_ptr<impl::compiler_impl::compiler_graph_engine_t>
                &graph_engine)
    : impl::compiled_partition_impl_t(engine, inputs, outputs, {})
    , jit_func_(jit_func)
    , graph_engine_(graph_engine) {
    std::lock_guard<std::mutex> lock(mtx_);
    partition_count_map[graph_engine_]++;
    graph_engine_->allocator_->retain();
}

compiler_compiled_partition_impl_t::~compiler_compiled_partition_impl_t() {
    std::lock_guard<std::mutex> lock(mtx_);
    auto itr = partition_count_map.find(graph_engine_);
    if (itr != partition_count_map.end()) {
        itr->second--;
        if (itr->second == 0) {
            sc::release_runtime_memory(graph_engine_.get());
        }
    }
    jit_func_ = nullptr;
    graph_engine_->allocator_->release();
}

impl::status_t compiler_compiled_partition_impl_t::execute(
        const impl::stream_t *astream,
        const std::vector<impl::tensor_t> &inputs,
        const std::vector<impl::tensor_t> &outputs) {
    // set backend runtime stream
    compiler_graph_stream_t backend_stream {graph_engine_.get()};

    std::vector<sc::generic_val> generic_args;
    generic_args.reserve(inputs.size() + outputs.size());
    for (auto out_tensor : outputs) {
        generic_args.emplace_back(out_tensor.get_data_handle());
    }
    for (auto in_tensor : inputs) {
        generic_args.emplace_back(in_tensor.get_data_handle());
    }
    jit_func_->call_generic(&backend_stream, generic_args.data());
    return status::success;
}
} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
