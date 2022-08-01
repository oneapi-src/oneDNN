/*******************************************************************************
* Copyright 2022 Intel Corporation
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

// If graph component is not built, we still need to preserve the API symbols
// but return unimplemented for each.
#ifndef BUILD_GRAPH

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph_sycl.h"

#include "graph/interface/c_types_map.hpp"

using namespace dnnl::impl::graph;

status_t DNNL_API dnnl_graph_allocator_create(allocator_t **allocator,
        host_allocate_f host_malloc, host_deallocate_f host_free) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_sycl_interop_allocator_create(
        allocator_t **allocator, sycl_allocate_f sycl_malloc,
        sycl_deallocate_f sycl_free) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_allocator_destroy(allocator_t *allocator) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_make_engine_with_allocator(engine_t **engine,
        engine_kind_t kind, size_t index, const allocator_t *alloc) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_sycl_interop_make_engine_with_allocator(
        engine_t **engine, const void *device, const void *context,
        const allocator_t *alloc) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_graph_create(
        graph_t **graph, engine_kind_t engine_kind) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_graph_create_with_fpmath_mode(
        graph_t **graph, engine_kind_t engine_kind, fpmath_mode_t mode) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_graph_destroy(graph_t *graph) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_add_op(graph_t *graph, op_t *op) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_graph_filter(
        graph_t *graph, partition_policy_t policy) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_graph_get_partition_num(
        const graph_t *graph, size_t *num) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_graph_get_partitions(
        graph_t *graph, size_t num, partition_t **partition) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_logical_tensor_init(
        logical_tensor_t *logical_tensor, size_t tid, data_type_t dtype,
        int32_t ndims, layout_type_t ltype, property_type_t ptype) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_logical_tensor_init_with_dims(
        logical_tensor_t *logical_tensor, size_t tid, data_type_t dtype,
        int32_t ndims, const dims_t dims, layout_type_t ltype,
        property_type_t ptype) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_logical_tensor_init_with_strides(
        logical_tensor_t *logical_tensor, size_t tid, data_type_t dtype,
        int32_t ndims, const dims_t dims, const dims_t strides,
        property_type_t ptype) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_logical_tensor_get_mem_size(
        const logical_tensor_t *logical_tensor, size_t *size) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_logical_tensor_has_same_layout(
        const logical_tensor_t *lt1, const logical_tensor_t *lt2,
        uint8_t *is_same) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_create(
        op_t **op, size_t id, op_kind_t kind, const char *verbose_name) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_destroy(op_t *op) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_add_input(
        op_t *op, const logical_tensor_t *input) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_add_output(
        op_t *op, const logical_tensor_t *output) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_set_attr_f32(op_t *op,
        dnnl_graph_op_attr_t name, const float *value, size_t value_len) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_set_attr_bool(op_t *op,
        dnnl_graph_op_attr_t name, const uint8_t *value, size_t value_len) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_set_attr_s64(op_t *op,
        dnnl_graph_op_attr_t name, const int64_t *value, size_t value_len) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_set_attr_str(op_t *op,
        dnnl_graph_op_attr_t name, const char *value, size_t value_len) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_get_id(const op_t *op, size_t *id) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_op_get_kind(const op_t *op, op_kind_t *kind) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_get_compiled_partition_cache_capacity(
        int *capacity) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_set_compiled_partition_cache_capacity(
        int capacity) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_create(partition_t **partition) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_create_with_op(
        partition_t **partition, const op_t *op, dnnl_engine_kind_t ekind) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_destroy(partition_t *partition) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_get_op_num(
        const partition_t *partition, size_t *num) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_get_ops(
        partition_t *partition, size_t num, size_t *ops) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_get_id(
        const partition_t *partition, size_t *id) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_compile(partition_t *partition,
        compiled_partition_t *compiled_partition, size_t in_num,
        const logical_tensor_t **inputs, size_t out_num,
        const logical_tensor_t **outputs, dnnl_engine_t engine) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_get_in_ports_num(
        const partition_t *partition, size_t *num) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_get_out_ports_num(
        const partition_t *partition, size_t *num) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_get_in_ports(
        const partition_t *partition, size_t num, logical_tensor_t *inputs) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_get_out_ports(
        const partition_t *partition, size_t num, logical_tensor_t *outputs) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_is_supported(
        const partition_t *partition, uint8_t *is_supported) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_get_engine_kind(
        const partition_t *partition, dnnl_engine_kind_t *kind) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_partition_get_kind(
        const partition_t *partition, partition_kind_t *kind) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_compiled_partition_create(
        compiled_partition_t **compiled_partition, partition_t *partition) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_compiled_partition_execute(
        const compiled_partition_t *compiled_partition, dnnl_stream_t stream,
        size_t num_inputs, const tensor_t **inputs, size_t num_outputs,
        const tensor_t **outputs) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_sycl_interop_compiled_partition_execute(
        const compiled_partition_t *compiled_partition, dnnl_stream_t stream,
        size_t num_inputs, const tensor_t **inputs, size_t num_outputs,
        const tensor_t **outputs, const void *deps, void *sycl_event) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_compiled_partition_destroy(
        compiled_partition_t *compiled_partition) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_compiled_partition_query_logical_tensor(
        const compiled_partition_t *compiled_partition, size_t tid,
        logical_tensor_t *lt) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_compiled_partition_get_inplace_ports(
        const compiled_partition_t *compiled_partition,
        size_t *num_inplace_pairs, const inplace_pair_t **inplace_pairs) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_tensor_create(tensor_t **tensor,
        const logical_tensor_t *logical_tensor, dnnl_engine_t eng,
        void *handle) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_tensor_destroy(tensor_t *tensor) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_tensor_get_data_handle(
        const tensor_t *tensor, void **handle) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_tensor_set_data_handle(
        tensor_t *tensor, void *handle) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_tensor_get_engine(
        const tensor_t *tensor, dnnl_engine_t *engine) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_get_constant_tensor_cache(int *flag) {
    return status::unimplemented;
}

status_t DNNL_API dnnl_graph_set_constant_tensor_cache(int flag) {
    return status::unimplemented;
}

#endif
