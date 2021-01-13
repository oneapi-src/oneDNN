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

/// @file
/// C API

#ifndef ONEAPI_DNNL_DNNL_GRAPH_H
#define ONEAPI_DNNL_DNNL_GRAPH_H

#include "oneapi/dnnl/dnnl_graph_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_graph_api
/// @{

/// @addtogroup dnnl_graph_api_allocator
/// @{

/// Initializes an allocator with the given allocation and deallocation
/// call-back function pointers (CPU)
///
/// @param created_allocator Output allocator
/// @param cpu_malloc A pointer to malloc function for CPU
/// @param cpu_free A pointer to free function for CPU
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_allocator_create(
        dnnl_graph_allocator_t **created_allocator,
        dnnl_graph_cpu_allocate_f cpu_malloc,
        dnnl_graph_cpu_deallocate_f cpu_free);

/// Destroys the created allocator
///
/// @param allocator The allocator to be destroyed.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_allocator_destroy(
        dnnl_graph_allocator_t *allocator);

/// @} dnnl_graph_api_allocator

/// @addtogroup dnnl_graph_api_logical_tensor
/// @{

/// Initializes a logical tensor with id, data type, ndims, and layout type.
///
/// @param created_logical_tensor Output logical tensor.
/// @param tid The unique id of output logical tensor.
/// @param dtype Elements data type.
/// @param ndims Number of dimensions, -1 means unknown.
/// @param ltype Layout type of target memory.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_logical_tensor_init(
        dnnl_graph_logical_tensor_t *created_logical_tensor, size_t tid,
        dnnl_graph_data_type_t dtype, int32_t ndims,
        dnnl_graph_layout_type_t ltype);

/// Initializes a logical tensor with basic information and dims.
///
/// @note
///     If dims contains all valid values and ltype is dnnl_graph_strided. The
///     strides field in dnnl_graph_logical_tensor_t wil be inferred in a row
///     major and contiguous way. Otherwise, Accessing the strides field will
///     be undefined behavior.
///
///     Eg. dims (2, 3, 4, 5) will get strides (60, 20, 5, 1)
///
/// @param created_logical_tensor Output logical tensor.
/// @param tid The unique id of output logical tensor.
/// @param dtype Elements data type.
/// @param ndims Number of dimensions, -1 means unknown.
/// @param dims Array of dimensions.
/// @param ltype Layout type of target memory.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_logical_tensor_init_with_dims(
        dnnl_graph_logical_tensor_t *created_logical_tensor, size_t tid,
        dnnl_graph_data_type_t dtype, int32_t ndims,
        const dnnl_graph_dims_t dims, dnnl_graph_layout_type_t ltype);

/// Initializes a logical tensor with basic information, dims, and strides.
///
/// @note
///     Once strides are explicitly provided through API, the layout_type
///     in dnnl_graph_logical_tensor_t can only be dnnl_graph_strided or
///     dnnl_graph_any.
///
/// @param created_logical_tensor Output logical tensor.
/// @param tid The unique id of output logical tensor.
/// @param dtype Elements data type.
/// @param ndims Number of dimensions, -1 means unknown.
/// @param dims Array of dimensions.
/// @param strides Array of strides
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_logical_tensor_init_with_strides(
        dnnl_graph_logical_tensor_t *created_logical_tensor, size_t tid,
        dnnl_graph_data_type_t dtype, int32_t ndims,
        const dnnl_graph_dims_t dims, const dnnl_graph_dims_t strides);

/// Returns the memory size described by the logical tensor
///
/// @note
///     If it's a strided layout, the size will be calculated by dims
///     and strides. If it's an opaque layout, the size will be queried
///     by layout_id.
///
/// @param logical_tensor Logical tensor.
/// @param size Output memory size in bytes.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_logical_tensor_get_mem_size(
        const dnnl_graph_logical_tensor_t *logical_tensor, size_t *size);

/// Compares if this and input logical tensor has the same layout
///
/// @param lt1 The handle of first logical tensor
/// @param lt2 The handle of second logical tensor
/// @param is_same @c true if these two logical tensors have the same layout or
///     data type
///     @c false if these two logical tensors have different layouts or data
///     types
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API
dnnl_graph_logical_tenosr_has_same_layout_and_dtype(
        const dnnl_graph_logical_tensor_t *lt1,
        const dnnl_graph_logical_tensor_t *lt2, uint8_t *is_same);

/// @} dnnl_graph_api_logical_tensor

/// @addtogroup dnnl_graph_api_tensor
/// @{

/// Initializes a tensor with ndims, dims, data type and data handle.
///
/// @param created_tensor Output tensor.
/// @param ndims Number of dimensions, -1 means unknown.
/// @param dims Array of dimensions.
/// @param type Data type of tensor.
/// @param data_handle Pointer to the data for this tensor.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_tensor_create(
        dnnl_graph_tensor_t **created_tensor, int64_t ndims,
        const int64_t *dims, dnnl_graph_data_type_t type, void *data_handle);

/// Initializes a tensor with logical tensor and data handle.
///
/// @param created_tensor Output tensor.
/// @param logical_tensor Description for this tensor.
/// @param data_handle Pointer to the data for this tensor.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_tensor_create_with_logical_tensor(
        dnnl_graph_tensor_t **created_tensor,
        const dnnl_graph_logical_tensor_t *logical_tensor, void *data_handle);

/// Destroys the created tensor.
///
/// @param tensor The tensor to be destroyed.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_tensor_destroy(
        dnnl_graph_tensor_t *tensor);

/// Gets data handle of tensor, if type doesn't match tensor's data type,
/// nullptr will be returned.
///
/// @param tensor The input tensor.
/// @param type Expected data type of the tensor.
/// @param data_handle Pointer to the data of input tensor.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_tensor_get_if_type(
        const dnnl_graph_tensor_t *tensor, dnnl_graph_data_type_t type,
        void **data_handle);

/// Set data handle for tensor
///
/// @param tensor The input tensor.
/// @param data_handle New data_handle for tensor.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_tensor_set_data_handle(
        dnnl_graph_tensor_t *tensor, void *data_handle);

/// Gets number of elements of the tensor.
///
/// @param tensor Input tensor.
/// @param no The number of elements of the tensor.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_tensor_get_element_num(
        const dnnl_graph_tensor_t *tensor, int64_t *no);

/// Gets the unique id of the tensor
///
/// @param tensor Input tensor.
/// @param id The unique id.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_tensor_get_id(
        const dnnl_graph_tensor_t *tensor, size_t *id);

/// @} dnnl_graph_api_tensor

/// @addtogroup dnnl_graph_api_op
/// @{

/// Initializes an op with unique id, kind and debug string
///
/// @param created_op Output op
/// @param id The unique id of this op
/// @param kind The op kind specifies which computation is represented by
///     the op, such as Convolution and ReLU.
/// @param debug_string The string added for debug
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_op_create(
        dnnl_graph_op_t **created_op, uint64_t id, dnnl_graph_op_kind_t kind,
        const char *const debug_string);

/// Destroys the created op
///
/// @param op The op to be destroyed.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_op_destroy(dnnl_graph_op_t *op);

/// Adds input logical tensor to the op
///
/// @param op Input op
/// @param input The input logical tensor to be added
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_op_add_input(
        dnnl_graph_op_t *op, const dnnl_graph_logical_tensor_t *input);

/// Adds output logical tensor to the op
///
/// @param op Input op
/// @param output The output logical tensor to be added
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_op_add_output(
        dnnl_graph_op_t *op, const dnnl_graph_logical_tensor_t *output);

/// Returns the kind of specified attribute in the Op
///
/// @param op Input op
/// @param name Name of the attribute
/// @param kind Attribute's kind
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_op_get_attr_kind(
        const dnnl_graph_op_t *op, const char *name,
        dnnl_graph_attribute_kind_t *kind);

/// Sets the attribute according to the name and kind
///
/// @param op Input op
/// @param name Attribute's name
/// @param kind The attribute's kind
/// @param attr The attribute's value
/// @param attr_no The number of attributes
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_op_add_attr(dnnl_graph_op_t *op,
        const char *name, dnnl_graph_attribute_kind_t kind, const void *attr,
        int64_t attr_no);

/// Returns the attribute according to the name and kind
///
/// @param op Input op
/// @param name Attribute's name
/// @param kind The attribute's kind
/// @param attr The attribute's value
/// @param attr_no The number of attributes
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_op_get_attr(
        const dnnl_graph_op_t *op, const char *name,
        dnnl_graph_attribute_kind_t kind, const void **attr, int64_t *attr_no);

/// Returns the unique id of the Op
///
/// @param op Input op
/// @param id The unique id
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_op_get_id(
        const dnnl_graph_op_t *op, size_t *id);

/// Returns the concrete kind of this op
///
/// @param op Input op
/// @param kind Op kind
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_op_get_kind(
        const dnnl_graph_op_t *op, dnnl_graph_op_kind_t *kind);

/// @} dnnl_graph_api_op

/// @addtogroup dnnl_graph_api_partition
/// @{

/// Creates a new empty partition.
///
/// @param partition The handle of output partition.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_create(
        dnnl_graph_partition_t **partition);

/// Destroy the target partition.
///
/// @param partition The partition to be destroyed.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_destroy(
        dnnl_graph_partition_t *partition);

/// Returns the number of ops of the partition.
///
/// @param partition The target partition.
/// @param num Output the number of ops.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_get_op_num(
        const dnnl_graph_partition_t *partition, size_t *num);

/// Returns the list of op IDs of the partition.
///
/// @param partition The target partition.
/// @param num The number of ops.
/// @param ids Output the op IDs.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_get_ops(
        dnnl_graph_partition_t *partition, size_t num, size_t *ids);

/// Returns the ID of the partition.
///
/// @param partition The target partition.
/// @param id Output the ID of the partition.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_get_id(
        const dnnl_graph_partition_t *partition, size_t *id);

/// Compile the partition with given input and output logical tensors
///
/// @param partition The target partition.
/// @param compiled_partition Output compiled partition.
/// @param in_num The number of input logical tensors.
/// @param inputs A list of input logical tensors.
/// @param out_num The number of output logical tensors.
/// @param outputs A list of output logical tensors.
/// @param engine The target engine of the compilation.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_compile(
        dnnl_graph_partition_t *partition,
        dnnl_graph_compiled_partition_t *compiled_partition, uint64_t in_num,
        const dnnl_graph_logical_tensor_t **inputs, uint64_t out_num,
        const dnnl_graph_logical_tensor_t **outputs,
        const dnnl_graph_engine_t *engine);

/// Infer the shape of the output logical tensors of a partition with given
/// input logical tensors.
///
/// @note The output logical tensors will be mutated with inferred shape. If an
///       output logical tensor is with strided layout type, its strides field
///       will be also inferred as dense strides.
///
/// @param partition The target partition.
/// @param in_num The number of input logical tensors.
/// @param inputs A list of input logical tensors.
/// @param out_num The number of output logical tensors.
/// @param outputs A list of output logical tensors.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_infer_shape(
        dnnl_graph_partition_t *partition, uint64_t in_num,
        const dnnl_graph_logical_tensor_t **inputs, uint64_t out_num,
        dnnl_graph_logical_tensor_t **outputs);

/// Returns the number of input logical tensors of the partition.
///
/// @param partition The target partition.
/// @param num Output the number of input logical tensors.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_get_inputs_num(
        const dnnl_graph_partition_t *partition, uint64_t *num);

/// Returns a list of input logical tensors of the partition.
///
/// @param partition The target partition.
/// @param num The number of input logical tensors.
/// @param inputs Output the list of input logical tensors.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_get_inputs(
        const dnnl_graph_partition_t *partition, uint64_t num,
        const dnnl_graph_logical_tensor_t **inputs);

/// Returns the number of output logical tensors of the partition.
///
/// @param partition The target partition.
/// @param num Output the number of output logical tensors.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_get_outputs_num(
        const dnnl_graph_partition_t *partition, uint64_t *num);

/// Returns a list of output logical tensors of the partition.
///
/// @param partition The target partition.
/// @param num The number of output logical tensors.
/// @param outputs Output the list of output logical tensors.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_partition_get_outputs(
        const dnnl_graph_partition_t *partition, uint64_t num,
        const dnnl_graph_logical_tensor_t **outputs);

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_conversion
/// @{

/// Initializes a conversion
///
/// @param conversion The target conversion
/// @param input The input logical tensor
/// @param output The output logical tensor
/// @param engine_kind The kind of engine
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_conversion_init(
        dnnl_graph_partition_t *conversion,
        const dnnl_graph_logical_tensor_t *input,
        const dnnl_graph_logical_tensor_t *output,
        dnnl_graph_engine_kind_t engine_kind);

/// @} dnnl_graph_api_conversion

/// @addtogroup dnnl_graph_api_compiled_partition
/// @{

/// Creates a new compiled partition.
///
/// @param compiled_partition The handle of output compiled_partition.
/// @param partition The handle of input partition.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_compiled_partition_create(
        dnnl_graph_compiled_partition_t **compiled_partition,
        dnnl_graph_partition_t *partition);

/// Execute a compiled partition.
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param stream The stream used for execution
/// @param num_inputs The number of input tensors
/// @param inputs A list of input tensors
/// @param num_outputs The number of output tensors
/// @param outputs A non-empty list of output tensors
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_compiled_partition_execute(
        const dnnl_graph_compiled_partition_t *compiled_partition,
        const dnnl_graph_stream_t *stream, const uint64_t num_inputs,
        const dnnl_graph_tensor_t **inputs, const uint64_t num_outputs,
        const dnnl_graph_tensor_t **outputs);

/// Destroy the target compiled partition.
///
/// @param compiled_partition The compiled partition to be destroyed.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_compiled_partition_destroy(
        dnnl_graph_compiled_partition_t *compiled_partition);

/// Returns the logical tensor according to tensor id
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param tid The unique id of required tensor
/// @param lt The output logical tensor
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API
dnnl_graph_compiled_partition_query_logical_tensor(
        const dnnl_graph_compiled_partition_t *compiled_partition, size_t tid,
        dnnl_graph_logical_tensor_t *lt);

/// Returns the in-place pairs.
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param num_inplace_pairs The number of in-place pairs.
/// @param inplace_pairs The handle of in-place pairs.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API
dnnl_graph_compiled_partition_get_inplace_pairs(
        const dnnl_graph_compiled_partition_t *compiled_partition,
        size_t *num_inplace_pairs,
        const dnnl_graph_inplace_pair_t **inplace_pairs);

/// @} dnnl_graph_api_compiled_partition

/// @addtogroup dnnl_graph_api_engine
/// @{

/// Creates an engine with specified engine kind and device id.
///
/// @param created_engine The handle of output engine.
/// @param engine_kind The kind of engine.
/// @param device_id The device associated to created engine.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_engine_create(
        dnnl_graph_engine_t **created_engine,
        dnnl_graph_engine_kind_t engine_kind, int32_t device_id);

/// Destroy the target engine.
///
/// @param engine The target engine.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_engine_destroy(
        dnnl_graph_engine_t *engine);

/// Set an allocator to the target engine.
///
/// @param engine The target engine.
/// @param allocator The allocator which will be set to engine.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_engine_set_allocator(
        dnnl_graph_engine_t *engine, dnnl_graph_allocator_t *allocator);

/// Get the device handle from an engine.
///
/// @param engine The target engine.
/// @param handle The output device handle.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_engine_get_device_handle(
        const dnnl_graph_engine_t *engine, void **handle);

/// Get the device id from an engine.
///
/// @param engine The target engine.
/// @param device_id The output device id.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_engine_get_device_id(
        const dnnl_graph_engine_t *engine, int32_t *device_id);

/// Get the engine kind from an engine.
///
/// @param engine The target engine.
/// @param kind The output engine kind.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_engine_get_kind(
        const dnnl_graph_engine_t *engine, dnnl_graph_engine_kind_t *kind);

/// @} dnnl_graph_api_engine

/// @addtogroup dnnl_graph_api_graph
/// @{

/// Creates a new empty graph.
///
/// @param created_graph The handle of output graph.
/// @param device_type The kind for engine, it can be #dnnl_graph_any_engine,
///     #dnnl_graph_cpu and #dnnl_graph_gpu.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_graph_create(
        dnnl_graph_graph_t **created_graph,
        dnnl_graph_engine_kind_t device_type);

/// Destroy the target graph.
///
/// @param graph The graph to be destroyed.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_graph_destroy(
        dnnl_graph_graph_t *graph);

/// Add a new op to a graph.
///
/// @param graph The graph.
/// @param op A new op to be added.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_add_op(
        dnnl_graph_graph_t *graph, dnnl_graph_op_t *op);

/// Do graph filtering and partitioning.
///
/// @param graph The graph.
/// @param policy Partition policy, it can be #dnnl_graph_partition_policy_max,
///     #dnnl_graph_partition_policy_fusion, and
///     #dnnl_graph_partition_policy_debug.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_graph_filter(
        dnnl_graph_graph_t *graph, dnnl_graph_partition_policy_t policy);

/// Gets the number of partitions of the graph.
///
/// @param graph The graph.
/// @param num Output the number of partitions.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_graph_get_partition_num(
        const dnnl_graph_graph_t *graph, uint64_t *num);

/// Gets the filtered partitions of the graph.
///
/// @param graph The graph.
/// @param num The number of partitions.
/// @param partition Output the partitions.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_graph_get_partitions(
        dnnl_graph_graph_t *graph, uint64_t num,
        dnnl_graph_partition_t **partition);

/// @} dnnl_graph_api_graph

/// @addtogroup dnnl_graph_api_threadpool
/// @{

/// Creates a thread pool
///
/// @param created_thread_pool The handle of output thread pool
/// @param num_threads Number of threads in this thread pool
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_thread_pool_create(
        dnnl_graph_thread_pool_t **created_thread_pool, int32_t num_threads);

/// Destroy the target thread pool
///
/// @param thread_pool The thread pool to be destroyed
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_thread_pool_destroy(
        dnnl_graph_thread_pool_t *thread_pool);

/// @} dnnl_graph_api_threadpool

/// @addtogroup dnnl_graph_api_stream_attr
/// @{

/// Creates a stream attribute with specified thread pool.
///
/// @param created_stream_attr The handle of output stream attribute.
/// @param thread_pool The handle of thread pool to create attribute on.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_stream_attr_create(
        dnnl_graph_stream_attr_t **created_stream_attr,
        dnnl_graph_thread_pool_t *thread_pool);

/// Destroy the target stream attribute.
///
/// @param stream_attr The target stream attribute.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_stream_attr_destroy(
        dnnl_graph_stream_attr_t *stream_attr);

/// @} dnnl_graph_api_stream_attr

/// @addtogroup dnnl_graph_api_stream
/// @{

/// Creates a stream for the specified engine.
///
/// @param created_stream The handle of output stream.
/// @param engine Engine to create the stream on.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_stream_create(
        dnnl_graph_stream_t **created_stream,
        const dnnl_graph_engine_t *engine);

/// Creates a stream for the specified engine and with behavior controlled by
/// the stream attribute.
///
/// @param created_stream The handle of output stream.
/// @param engine Engine to create the stream on.
/// @param attr The stream attribute.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_stream_create_with_attr(
        dnnl_graph_stream_t **created_stream, const dnnl_graph_engine_t *engine,
        const dnnl_graph_stream_attr_t *attr);

/// Creates a stream for a given engine associated with a SYCL queue and with
/// behavior controlled by the stream attribute.
///
/// @param created_stream The handle of output stream.
/// @param engine Engine to create the stream on.
/// @param queue SYCL queue to use.
/// @param attr The stream attribute.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_stream_create_sycl_with_attr(
        dnnl_graph_stream_t **created_stream, const dnnl_graph_engine_t *engine,
        const void *queue, const dnnl_graph_stream_attr_t *attr);

/// Destroy the target stream.
///
/// @param stream The target stream.
/// @returns #dnnl_graph_result_success on success and a status describing the
///     error otherwise.
dnnl_graph_result_t DNNL_GRAPH_API dnnl_graph_stream_destroy(
        dnnl_graph_stream_t *stream);

/// @} dnnl_graph_api_stream

/// @addtogroup dnnl_graph_api_service Service
/// @{

/// Returns library version information.
/// @returns Pointer to a constant structure containing
///  - major: major version number,
///  - minor: minor version number,
///  - patch: patch release number,
///  - hash: git commit hash.
const dnnl_graph_version_t DNNL_GRAPH_API *dnnl_graph_version(void);

/// @} dnnl_graph_api_service

/// @} dnnl_graph_api

#ifdef __cplusplus
}
#endif
#endif
