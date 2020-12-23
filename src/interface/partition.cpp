/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <cassert>
#include <cstring>
#include <limits>
#include <set>

#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph_sycl.h"

#include "c_types_map.hpp"
#include "op_schema.hpp"
#include "partition.hpp"

#include "interface/backend.hpp"
#include "interface/stream.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

namespace {

// FIXME(qun) Workaround: op in this list can have repeated inputs
static const std::set<impl::op_kind_t> s_whitelist {
        impl::op_kind::Multiply, impl::op_kind::Add};

//RCONT is for const or non-const impl::logical_tensor_t
template <typename RCONT>
impl::status_t get_ordered_inputs_outputs(const impl::node_t *work_node,
        const std::vector<impl::logical_tensor_t> &expected,
        const std::vector<RCONT *> &origin,
        std::vector<impl::logical_tensor_t *> &ordered) {
    // to support abitrary re-connection in FWK graph, we need to
    // find required and ordered input and output logical tensors from the
    // out-of-order inputs / outputs
    if (origin.size() < expected.size()
            && s_whitelist.count(work_node->get_op_kind()) == 0) {
        return impl::status::miss_ins_outs;
    }
    ordered.reserve(expected.size());
    for (auto &&val : expected) {
        auto pos = std::find_if(origin.begin(), origin.end(),
                [&val](RCONT *in) -> bool { return in->id == val.id; });
        if (pos != origin.end()) {
            ordered.emplace_back(const_cast<impl::logical_tensor_t *>(*pos));
        }
    }
    if (ordered.size() != expected.size()) return impl::status::miss_ins_outs;
    return impl::status::success;
}

impl::status_t get_ordered_inputs_outputs(const impl::node_t *work_node,
        const std::vector<impl::logical_tensor_t> &expected,
        const std::vector<impl::tensor> &origin,
        std::vector<impl::tensor> &ordered) {
    // to support abitrary re-connection in FWK graph, we need to
    // find required and ordered input and output tensors from the
    // out-of-order inputs / outputs
    if (origin.size() < expected.size()
            && s_whitelist.count(work_node->get_op_kind()) == 0) {
        return impl::status::miss_ins_outs;
    }
    ordered.reserve(expected.size());
    for (auto &&val : expected) {
        auto pos = std::find_if(origin.begin(), origin.end(),
                [&val](const impl::tensor &in) -> bool {
                    return in.get_logical_tensor().id == val.id;
                });
        if (pos != origin.end()) { ordered.emplace_back(*pos); }
    }
    if (ordered.size() != expected.size()) return impl::status::miss_ins_outs;
    return impl::status::success;
}

} // namespace

using namespace llga::impl;

status_t DNNL_GRAPH_API dnnl_graph_partition_create(
        dnnl_graph_partition_t **partition) {
    *partition = new dnnl_graph_partition();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_destroy(
        dnnl_graph_partition_t *partition) {
    delete partition;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_op_num(
        const dnnl_graph_partition_t *partition, size_t *num) {
    *num = partition->num_ops();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_ops(
        dnnl_graph_partition_t *partition, size_t num, size_t *node) {
    auto ids = partition->get_ops();
    if (ids.size() != num || node == nullptr) {
        return status::invalid_argument;
    }

    int idx = 0;
    for (auto it = ids.begin(); it != ids.end(); ++it, ++idx) {
        node[idx] = *it;
    }

    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_id(
        const dnnl_graph_partition_t *partition, size_t *id) {
    if (partition == nullptr || id == nullptr) {
        return status::invalid_argument;
    }

    *id = partition->id();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_compile(
        dnnl_graph_partition_t *partition,
        dnnl_graph_compiled_partition_t *compiled_partition, uint64_t in_num,
        const dnnl_graph_logical_tensor_t **inputs, uint64_t out_num,
        const dnnl_graph_logical_tensor_t **outputs,
        const dnnl_graph_engine_t *engine) {
    if (partition == nullptr || engine == nullptr) {
        return status::invalid_argument;
    }

    std::vector<const dnnl_graph_logical_tensor_t *> in {
            inputs, inputs + in_num};
    std::vector<const dnnl_graph_logical_tensor_t *> out {
            outputs, outputs + out_num};
    status_t ret = partition->compile(compiled_partition, in, out, engine);
    return ret;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_infer_shape(
        dnnl_graph_partition_t *partition, uint64_t in_num,
        const dnnl_graph_logical_tensor_t **inputs, uint64_t out_num,
        dnnl_graph_logical_tensor_t **outputs) {
    if (partition == nullptr) { return status::invalid_argument; }

    std::vector<const dnnl_graph_logical_tensor_t *> in {
            inputs, inputs + in_num};
    std::vector<dnnl_graph_logical_tensor_t *> out {outputs, outputs + out_num};
    status_t ret = partition->infer_shape(in, out);
    return ret;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_inputs_num(
        const dnnl_graph_partition_t *partition, uint64_t *num) {
    if (partition == nullptr || num == nullptr) {
        return status::invalid_argument;
    }

    *num = static_cast<uint64_t>(partition->get_inputs_num());
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_outputs_num(
        const dnnl_graph_partition_t *partition, uint64_t *num) {
    if (partition == nullptr || num == nullptr) {
        return status::invalid_argument;
    }

    *num = static_cast<uint64_t>(partition->get_outputs_num());
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_inputs(
        const dnnl_graph_partition_t *partition, uint64_t num,
        const dnnl_graph_logical_tensor_t **inputs) {
    if (partition == nullptr || inputs == nullptr
            || partition->get_inputs_num() != num) {
        return status::invalid_argument;
    }

    auto &in = partition->get_inputs();
    for (size_t i = 0; i < num; ++i) {
        inputs[i] = &(in[i]);
    }

    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_outputs(
        const dnnl_graph_partition_t *partition, uint64_t num,
        const dnnl_graph_logical_tensor_t **outputs) {
    if (partition == nullptr || outputs == nullptr
            || partition->get_outputs_num() != num) {
        return status::invalid_argument;
    }

    auto &out = partition->get_outputs();
    for (size_t i = 0; i < num; ++i) {
        outputs[i] = &(out[i]);
    }

    return status::success;
}

/// Initializes a conversion partition
status_t DNNL_GRAPH_API dnnl_graph_conversion_init(
        dnnl_graph_partition_t *conversion,
        const dnnl_graph_logical_tensor_t *input,
        const dnnl_graph_logical_tensor_t *output,
        impl::engine_kind_t engine_kind) {
    conversion->init(impl::op_kind::convert, engine_kind, *input, *output);
    return status::success;
}

///
/// dnnl_graph_compiled_partition_t
///
status_t DNNL_GRAPH_API dnnl_graph_compiled_partition_create(
        dnnl_graph_compiled_partition_t **created_compiled_partition,
        dnnl_graph_partition_t *partition) {
    *created_compiled_partition
            = new dnnl_graph_compiled_partition {*partition};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_compiled_partition_execute(
        const dnnl_graph_compiled_partition_t *compiled_partition,
        const dnnl_graph_stream_t *stream, const uint64_t num_inputs,
        const dnnl_graph_tensor_t **inputs, const uint64_t num_outputs,
        const dnnl_graph_tensor_t **outputs) {
    if (utils::any_null(stream, compiled_partition, inputs, outputs))
        return status::invalid_argument;

    std::vector<dnnl_graph_tensor_t> ins, outs;
    ins.reserve(num_inputs);
    outs.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i) {
        ins.emplace_back(**(inputs + i));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        outs.emplace_back(**(outputs + i));
    }

    return compiled_partition->execute(stream, ins, outs);
}

status_t DNNL_GRAPH_API dnnl_graph_sycl_interop_compiled_partition_execute(
        const dnnl_graph_compiled_partition_t *compiled_partition,
        const dnnl_graph_stream_t *stream, const uint64_t num_inputs,
        const dnnl_graph_tensor_t **inputs, const uint64_t num_outputs,
        const dnnl_graph_tensor_t **outputs, const uint64_t num_deps,
        void *deps, void *sycl_event) {
#if DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(stream, compiled_partition, inputs, outputs))
        return status::invalid_argument;

    std::vector<dnnl_graph_tensor_t> ins, outs;
    ins.reserve(num_inputs);
    outs.reserve(num_outputs);
    std::vector<cl::sycl::event> sycl_deps;
    sycl_deps.reserve(num_deps);
    for (size_t i = 0; i < num_inputs; ++i) {
        ins.emplace_back(**(inputs + i));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        outs.emplace_back(**(outputs + i));
    }
    auto sycl_deps_ptr = static_cast<cl::sycl::event *>(deps);
    for (size_t i = 0; i < num_deps; ++i)
        sycl_deps.emplace_back(*(sycl_deps_ptr + i));

    return compiled_partition->execute_sycl(
            stream, ins, outs, static_cast<cl::sycl::event *>(sycl_event));
#else
    UNUSED(compiled_partition);
    UNUSED(stream);
    UNUSED(num_inputs);
    UNUSED(inputs);
    UNUSED(num_outputs);
    UNUSED(outputs);
    UNUSED(num_deps);
    UNUSED(deps);
    UNUSED(sycl_event);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_compiled_partition_destroy(
        dnnl_graph_compiled_partition_t *compiled_partition) {
    delete compiled_partition;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_compiled_partition_query_logical_tensor(
        const dnnl_graph_compiled_partition_t *compiled_partition, size_t tid,
        dnnl_graph_logical_tensor_t *lt) {
    if (utils::any_null(compiled_partition, lt))
        return status::invalid_argument;
    return compiled_partition->query_logical_tensor(tid, lt);
}

status_t DNNL_GRAPH_API dnnl_graph_compiled_partition_get_inplace_pairs(
        const dnnl_graph_compiled_partition_t *compiled_partition,
        size_t *num_inplace_pairs, const impl::inplace_pair_t **inplace_pairs) {
    if (utils::any_null(compiled_partition, num_inplace_pairs, inplace_pairs))
        return status::invalid_argument;

    const auto &cp_inplace_pairs = compiled_partition->get_inplace_pairs();
    *num_inplace_pairs = cp_inplace_pairs.size();
    *inplace_pairs = cp_inplace_pairs.data();

    return status::success;
}

void dnnl_graph_partition::init(impl::op_kind_t op_kind,
        const impl::engine_kind_t engine_kind,
        const impl::logical_tensor_t &input,
        const impl::logical_tensor_t &output) {
    engine_kind_ = engine_kind;
    node_ = impl::utils::make_unique<impl::node_t>(op_kind);
    impl::logical_tensor_wrapper ltw {&output};
    assert(ltw.is_opaque());
    auto backend_name = impl::backend_manager::get_backend(
            static_cast<size_t>(ltw.layout_id()))
                                ->get_name();
    node_->set_attr<std::string>("backend", backend_name);
    inputs_.push_back(input);
    outputs_.push_back(output);
}

status_t dnnl_graph_partition::compile(compiled_partition *compiled_partition,
        std::vector<const impl::logical_tensor_t *> &inputs,
        std::vector<const impl::logical_tensor_t *> &outputs,
        const engine_t *aengine) {
    if (!aengine || aengine->kind() != engine_kind_)
        return status::invalid_argument;
    // we shouldn't modify partition after
    // creating from filter(). all the information generated
    // at compilation stage should be stored in the corresponding
    // compiled_partition
    // in shape_infer, node's attrs may be changed
    impl::node_t *work_node = compiled_partition->src_partition_.node_.get();

    // to support abitrary re-connection in FWK graph, we need to
    // find required input and output logical tensors from the compile
    // function's parameters
    impl::status_t ret;
    std::vector<impl::logical_tensor_t *> required_inputs, required_outputs;
    ret = get_ordered_inputs_outputs(
            work_node, inputs_, inputs, required_inputs);
    if (impl::status::success != ret) return ret;
    ret = get_ordered_inputs_outputs(
            work_node, outputs_, outputs, required_outputs);
    if (impl::status::success != ret) return ret;

    // In the phase of compilation, all the output shape should be known.
    auto pos = std::find_if(required_outputs.begin(), required_outputs.end(),
            [&](const std::vector<impl::logical_tensor_t *>::value_type &out)
                    -> bool {
                return impl::logical_tensor_wrapper(out).is_shape_unknown();
            });
    if (pos != required_outputs.end()) { return status::invalid_argument; }

    const op_schema *cur_op_schema
            = op_schema_registry::get_op_schema(work_node->get_op_kind());
    if (cur_op_schema) { // if cur_op_schema is not nullptr
        // infer attributes of the node, i.e. auto_pad
        cur_op_schema->shape_infer(
                work_node, required_inputs, required_outputs);
    }

    // store the filled logical tensor value to compiled partition
    compiled_partition->inputs_.reserve(required_inputs.size());
    for (auto r_in : required_inputs) {
        compiled_partition->inputs_.emplace_back(*r_in);
    }
    compiled_partition->outputs_.reserve(required_outputs.size());
    for (auto r_out : required_outputs) {
        compiled_partition->outputs_.emplace_back(*r_out);
    }
    compiled_partition->engine_ = *aengine;

    std::string backend_name = work_node->get_attr<std::string>("backend");
    compiled_partition->executable_
            = backend_manager::get_backend(backend_name)
                      ->compile(&(compiled_partition->src_partition_), aengine,
                              compiled_partition->inputs_,
                              compiled_partition->outputs_);
    if (!compiled_partition->executable_) return status::compile_fail;
    return status::success;
}

status_t dnnl_graph_partition::infer_shape(
        std::vector<const impl::logical_tensor_t *> &inputs,
        std::vector<impl::logical_tensor_t *> &outputs) {
    std::vector<impl::logical_tensor_t *> required_inputs, required_outputs;
    impl::status_t ret;
    ret = get_ordered_inputs_outputs(
            node_.get(), inputs_, inputs, required_inputs);
    if (impl::status::success != ret) return ret;
    ret = get_ordered_inputs_outputs(
            node_.get(), outputs_, outputs, required_outputs);
    if (impl::status::success != ret) return ret;

    // check if shape is already known, if so, no need to do shape inference
    auto pos = std::find_if(required_outputs.begin(), required_outputs.end(),
            [&](const std::vector<impl::logical_tensor_t *>::value_type &out)
                    -> bool {
                return impl::logical_tensor_wrapper(out).is_shape_unknown();
            });
    if (pos == required_outputs.end()) { return status::success; }

    const op_schema *cur_op_schema
            = op_schema_registry::get_op_schema(node_->get_op_kind());
    if (cur_op_schema) { // if cur_op_schema is not nullptr
        // shape_infer will change node attrs, so in order to keep the node_
        // in partition unchanged, create a temp_node to hold these changes
        impl::node_t temp_node = impl::node_t(node_->get_op_kind());
        temp_node.merge_attrs_map(node_->get_attrs_map());
        status_t ret = cur_op_schema->shape_infer(
                &temp_node, required_inputs, required_outputs);
        return ret;
    } else {
        return status::invalid_op;
    }
}

const std::vector<impl::inplace_pair_t> &
dnnl_graph_compiled_partition::get_inplace_pairs() const {
    return executable_->get_inplace_pairs();
}

status_t dnnl_graph_compiled_partition::execute(const impl::stream *astream,
        const std::vector<impl::tensor> &inputs,
        const std::vector<impl::tensor> &outputs) const {
    // to support abitrary re-connection in FWK graph, we need to
    // find required input and output logical tensors from the compile
    // function's parameters
    std::vector<impl::tensor> required_inputs, required_outputs;

    impl::status_t ret;
    ret = get_ordered_inputs_outputs(
            src_partition_.node_.get(), inputs_, inputs, required_inputs);
    if (impl::status::success != ret) return ret;
    ret = get_ordered_inputs_outputs(
            src_partition_.node_.get(), outputs_, outputs, required_outputs);
    if (impl::status::success != ret) return ret;

    if (!astream || !astream->get_engine()->match(engine_))
        return status::invalid_argument;

    return executable_->execute(astream, required_inputs, required_outputs);
}

#if DNNL_GRAPH_WITH_SYCL
status_t dnnl_graph_compiled_partition::execute_sycl(
        const impl::stream *astream, const std::vector<impl::tensor> &inputs,
        const std::vector<impl::tensor> &outputs,
        const cl::sycl::event *sycl_event) const {
    // to support abitrary re-connection in FWK graph, we need to
    // find required input and output logical tensors from the compile
    // function's parameters
    std::vector<impl::tensor> required_inputs, required_outputs;
    impl::status_t ret;
    ret = get_ordered_inputs_outputs(
            src_partition_.node_.get(), inputs_, inputs, required_inputs);
    if (impl::status::success != ret) return ret;
    ret = get_ordered_inputs_outputs(
            src_partition_.node_.get(), outputs_, outputs, required_outputs);
    if (impl::status::success != ret) return ret;

    // TODO(Wei, Zixuan): Here we just pass down the pointer of cl::sycl::event
    // from API interface. The pointer points a concrete object created by the
    // users. In backend operators, the event object returned by primitive
    // execution can be assigned to the users' object.
    UNUSED(sycl_event);

    if (!astream || !astream->get_engine()->match(engine_))
        return status::invalid_argument;
    return executable_->execute(astream, required_inputs, required_outputs);
}
#endif // DNNL_GRAPH_WITH_SYCL

status_t dnnl_graph_compiled_partition::query_logical_tensor(
        size_t tid, logical_tensor_t *lt) const {
    auto pos_in = std::find_if(inputs_.begin(), inputs_.end(),
            [&](const impl::logical_tensor_t &in_) -> bool {
                return in_.id == tid;
            });
    if (pos_in != inputs_.end()) {
        *lt = *pos_in;
        return status::success;
    }

    auto pos_out = std::find_if(outputs_.begin(), outputs_.end(),
            [&](const impl::logical_tensor_t &out_) -> bool {
                return out_.id == tid;
            });
    if (pos_out != outputs_.end()) {
        *lt = *pos_out;
        return status::success;
    }

    // if we didn't found the logical tensor in compiled partition's inputs_
    // and outputs_, this means the logical tensor is not required by this
    // compiled partition. this will be a common situation if FWK give abitrary
    // connection, and shouldn't be regard as an error
    std::memset(lt, 0, sizeof(logical_tensor_t));
    return status::success;
}
