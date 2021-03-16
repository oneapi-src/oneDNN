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

#include <cassert>
#include <cstring>
#include <limits>
#include <set>

#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph_sycl.h"

#include "backend.hpp"
#include "c_types_map.hpp"
#include "op_schema.hpp"
#include "partition.hpp"
#include "stream.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

using namespace dnnl::graph::impl;

status_t DNNL_GRAPH_API dnnl_graph_partition_create(partition_t **partition) {
    *partition = new partition_t();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_destroy(partition_t *partition) {
    delete partition;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_op_num(
        const partition_t *partition, size_t *num) {
    *num = partition->num_ops();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_ops(
        partition_t *partition, size_t num, size_t *ops) {
    auto ids = partition->get_op_ids();
    if (ids.size() != num || ops == nullptr) {
        return status::invalid_argument;
    }

    int idx = 0;
    for (auto it = ids.begin(); it != ids.end(); ++it, ++idx) {
        ops[idx] = *it;
    }

    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_id(
        const partition_t *partition, size_t *id) {
    if (partition == nullptr || id == nullptr) {
        return status::invalid_argument;
    }

    *id = partition->id();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_compile(partition_t *partition,
        compiled_partition_t *compiled_partition, uint64_t in_num,
        const logical_tensor_t **inputs, uint64_t out_num,
        const logical_tensor_t **outputs, const engine_t *engine) {
    if (partition == nullptr || engine == nullptr) {
        return status::invalid_argument;
    }

    std::vector<const logical_tensor_t *> in {inputs, inputs + in_num};
    std::vector<const logical_tensor_t *> out {outputs, outputs + out_num};
    status_t ret = partition->compile(compiled_partition, in, out, engine);
    return ret;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_infer_shape(partition_t *partition,
        uint64_t in_num, const logical_tensor_t **inputs, uint64_t out_num,
        logical_tensor_t **outputs) {
    if (partition == nullptr) { return status::invalid_argument; }

    std::vector<const logical_tensor_t *> in {inputs, inputs + in_num};
    std::vector<logical_tensor_t *> out {outputs, outputs + out_num};
    status_t ret = partition->infer_shape(in, out);
    return ret;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_inputs_num(
        const partition_t *partition, uint64_t *num) {
    if (partition == nullptr || num == nullptr) {
        return status::invalid_argument;
    }

    *num = static_cast<uint64_t>(partition->get_inputs_num());
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_outputs_num(
        const partition_t *partition, uint64_t *num) {
    if (partition == nullptr || num == nullptr) {
        return status::invalid_argument;
    }

    *num = static_cast<uint64_t>(partition->get_outputs_num());
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_partition_get_inputs(
        const partition_t *partition, uint64_t num,
        const logical_tensor_t **inputs) {
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
        const partition_t *partition, uint64_t num,
        const logical_tensor_t **outputs) {
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
status_t DNNL_GRAPH_API dnnl_graph_conversion_init(partition_t *conversion,
        const logical_tensor_t *input, const logical_tensor_t *output,
        engine_kind_t engine_kind) {
    logical_tensor_wrapper ltw {output};
    assert(ltw.is_opaque());
    auto backend_ptr = const_cast<backend *>(
            backend_registry::get_singleton().get_registered_backend(
                    static_cast<size_t>(ltw.layout_id())));

    auto cvs_impl
            = backend_ptr->create_conversion(engine_kind, *input, *output);
    conversion->init(cvs_impl);
    return status::success;
}

///
/// dnnl_graph_compiled_partition_t
///
status_t DNNL_GRAPH_API dnnl_graph_compiled_partition_create(
        compiled_partition_t **created_compiled_partition,
        partition_t *partition) {
    *created_compiled_partition = new compiled_partition_t {*partition};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_compiled_partition_execute(
        const compiled_partition_t *compiled_partition, const stream_t *stream,
        const uint64_t num_inputs, const tensor_t **inputs,
        const uint64_t num_outputs, const tensor_t **outputs) {
    if (utils::any_null(stream, compiled_partition, inputs, outputs))
        return status::invalid_argument;

    std::vector<tensor_t> ins, outs;
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
        const compiled_partition_t *compiled_partition, const stream_t *stream,
        const uint64_t num_inputs, const tensor_t **inputs,
        const uint64_t num_outputs, const tensor_t **outputs,
        const uint64_t num_deps, void *deps, void *sycl_event) {
#if DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(stream, compiled_partition, inputs, outputs))
        return status::invalid_argument;

    std::vector<tensor_t> ins, outs;
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
        compiled_partition_t *compiled_partition) {
    delete compiled_partition;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_compiled_partition_query_logical_tensor(
        const compiled_partition_t *compiled_partition, size_t tid,
        logical_tensor_t *lt) {
    if (utils::any_null(compiled_partition, lt))
        return status::invalid_argument;
    return compiled_partition->query_logical_tensor(tid, lt);
}

status_t DNNL_GRAPH_API dnnl_graph_compiled_partition_get_inplace_pairs(
        const compiled_partition_t *compiled_partition,
        size_t *num_inplace_pairs, const inplace_pair_t **inplace_pairs) {
    if (utils::any_null(compiled_partition, num_inplace_pairs, inplace_pairs))
        return status::invalid_argument;

    const auto &cp_inplace_pairs = compiled_partition->get_inplace_pairs();
    *num_inplace_pairs = cp_inplace_pairs.size();
    *inplace_pairs = cp_inplace_pairs.data();

    return status::success;
}

impl::status_t dnnl_graph_partition::infer_shape(
        std::vector<const impl::logical_tensor_t *> &inputs,
        std::vector<impl::logical_tensor_t *> &outputs) {
    // check if shape is already known, if so, no need to do shape inference
    auto pos = std::find_if(outputs.begin(), outputs.end(),
            [&](const std::vector<logical_tensor_t *>::value_type &out)
                    -> bool {
                return logical_tensor_wrapper(out).is_shape_unknown();
            });
    if (pos == outputs.end()) { return status::success; }

    return pimpl_->infer_shape(inputs, outputs);
}

static status_t pre_process(std::vector<logical_tensor_t> &dst,
        std::vector<const logical_tensor_t *> &src, const backend *abackend) {
    using ltw = logical_tensor_wrapper;
    dst.reserve(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst.emplace_back(*src[i]);
        if (ltw(src[i]).is_opaque()) {
            size_t layout_id = static_cast<size_t>(src[i]->layout.layout_id);
            auto pair = backend_registry::decode_layout_id(layout_id);
            if (pair.second != abackend->get_id()) {
                // given opaque layout id must be generated by this
                // backend
                return status::invalid_argument;
            }
            dst[i].layout.layout_id = static_cast<int64_t>(pair.first);
        }
    }
    return status::success;
}

static status_t post_process(std::vector<logical_tensor_t> &dst,
        std::vector<logical_tensor_t> &src, const backend *abackend) {
    using ltw = logical_tensor_wrapper;
    UNUSED(src);

    for (size_t i = 0; i < dst.size(); i++) {
        if (ltw(dst[i]).is_opaque()) {
            size_t layout_id = static_cast<size_t>(dst[i].layout.layout_id);
            dst[i].layout.layout_id
                    = static_cast<int64_t>(backend_registry::encode_layout_id(
                            layout_id, abackend->get_id()));
        }
    }
    return status::success;
}

static status_t pre_process(std::vector<tensor_t> &dst,
        const std::vector<tensor_t> &src, const backend *abackend) {
    using ltw = logical_tensor_wrapper;
    dst.reserve(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst.emplace_back(src[i]);
        auto &src_lt = src[i].get_logical_tensor();
        if (ltw(src_lt).is_opaque()) {
            size_t layout_id = static_cast<size_t>(src_lt.layout.layout_id);
            auto pair = backend_registry::decode_layout_id(layout_id);
            if (pair.second != abackend->get_id()) {
                return status::invalid_argument;
            }
            auto &dst_lt = const_cast<logical_tensor_t &>(
                    dst[i].get_logical_tensor());

            dst_lt.layout.layout_id = static_cast<int64_t>(pair.first);
        }
    }
    return status::success;
}

status_t dnnl_graph_partition::compile(compiled_partition_t *cp,
        std::vector<const impl::logical_tensor_t *> &inputs,
        std::vector<const impl::logical_tensor_t *> &outputs,
        const engine_t *aengine) {
    status_t ret;

    if (!aengine || aengine->kind() != pimpl_->get_engine_kind())
        return status::invalid_argument;

    const backend *backend = pimpl_->get_assigned_backend();
    if (!backend) return status::compile_fail;

    // Pre-process the given logical tensor. The pre-process includes
    // 1. decode backend id from the layout id and remove it
    std::vector<logical_tensor_t> tmp_inputs, tmp_outputs;
    ret = pre_process(tmp_inputs, inputs, backend);
    if (status::success != ret) return ret;

    ret = pre_process(tmp_outputs, outputs, backend);
    if (status::success != ret) return ret;

    // The impl's compile will generate the compiled_partition_impl and
    // modify the given inputs outputs logical tensor
    ret = pimpl_->compile(cp, tmp_inputs, tmp_outputs, aengine);
    if (status::success != ret) return ret;

    // Post-process the modified logical tensor and store them
    // to compiled_partition_impl. The post-process includes
    // 1. encode backend id to generated layout id
    ret = post_process(cp->get_mutable_inputs(), tmp_inputs, backend);
    if (status::success != ret) return ret;

    ret = post_process(cp->get_mutable_outputs(), tmp_outputs, backend);
    if (status::success != ret) return ret;

    if (ret != status::success || !cp->is_initialized())
        return status::compile_fail;
    return status::success;
}

status_t dnnl_graph_compiled_partition::execute(const stream_t *astream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs) const {
    if (!astream || !astream->get_engine()->match(pimpl_->get_engine()))
        return status::invalid_argument;

    status_t ret;

    const backend *backend = src_partition_.get_assigned_backend();
    if (!backend) return status::compile_fail;

    // Pre-process the given tensor. The pre-process includes
    // FIXME(xx) reduce overhead?
    // 1. decode backend id from the layout id and remove it
    std::vector<tensor_t> processed_inputs, processed_outputs;
    pre_process(processed_inputs, inputs, backend);
    pre_process(processed_outputs, outputs, backend);

    if (utils::get_verbose()) {
        double ms = utils::get_msec();
        ret = pimpl_->execute(astream, processed_inputs, processed_outputs);
        ms = utils::get_msec() - ms;
        printf("dnnl_graph_verbose,exec,%s,%g\n", this->info(), ms);
        fflush(stdout);
    } else {
        ret = pimpl_->execute(astream, processed_inputs, processed_outputs);
    }

    return ret;
}

#if DNNL_GRAPH_WITH_SYCL
status_t dnnl_graph_compiled_partition::execute_sycl(const stream_t *astream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs,
        const cl::sycl::event *sycl_event) const {
    if (!astream || !astream->get_engine()->match(pimpl_->get_engine()))
        return status::invalid_argument;

    status_t ret;

    const backend *backend = src_partition_.get_assigned_backend();
    if (!backend) return status::compile_fail;

    // Pre-process the given tensor. The pre-process includes
    // 1. decode backend id from the layout id and remove it
    std::vector<tensor_t> processed_inputs, processed_outputs;
    pre_process(processed_inputs, inputs, backend);
    pre_process(processed_outputs, outputs, backend);

    if (utils::get_verbose()) {
        double ms = utils::get_msec();
        ret = pimpl_->execute_sycl(
                astream, processed_inputs, processed_outputs, sycl_event);
        ms = utils::get_msec() - ms;
        printf("dnnl_graph_verbose,exec,%s,%g\n", this->info(), ms);
        fflush(stdout);
    } else {
        ret = pimpl_->execute_sycl(
                astream, processed_inputs, processed_outputs, sycl_event);
    }

    return ret;
}
#endif // DNNL_GRAPH_WITH_SYCL
