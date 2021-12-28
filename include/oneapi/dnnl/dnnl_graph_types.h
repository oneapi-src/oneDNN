/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
/// C API definitions

#ifndef ONEAPI_DNNL_DNNL_GRAPH_TYPES_H
#define ONEAPI_DNNL_DNNL_GRAPH_TYPES_H

#include <stddef.h>
#include <stdint.h>

#include "oneapi/dnnl/dnnl_graph_buildin_ops.h"

#if defined _WIN32 || defined __CYGWIN__
#define DNNL_GRAPH_HELPER_DLL_IMPORT __declspec(dllimport)
#define DNNL_GRAPH_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define DNNL_GRAPH_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define DNNL_GRAPH_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#else
#define DNNL_GRAPH_HELPER_DLL_IMPORT
#define DNNL_GRAPH_HELPER_DLL_EXPORT
#endif
#endif

#ifdef DNNL_GRAPH_DLL
#ifdef DNNL_GRAPH_DLL_EXPORTS
#define DNNL_GRAPH_API DNNL_GRAPH_HELPER_DLL_EXPORT
#else
#define DNNL_GRAPH_API DNNL_GRAPH_HELPER_DLL_IMPORT
#endif
#else
#define DNNL_GRAPH_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_graph_api
/// @{

/// @addtogroup dnnl_graph_api_logical_tensor
/// @{

/// Maximum dimensions
#define DNNL_GRAPH_MAX_NDIMS 12
/// An integer to indicate unknown number of dimensions
#define DNNL_GRAPH_UNKNOWN_NDIMS -1
/// An integer to indicate unknown dimensions
#define DNNL_GRAPH_UNKNOWN_DIM -1

/// A type to describe tensor dimension.
typedef int64_t dnnl_graph_dim_t;

/// A type to describe tensor dimensions.
typedef dnnl_graph_dim_t dnnl_graph_dims_t[DNNL_GRAPH_MAX_NDIMS];

/// Data type specifications
typedef enum {
    /// undefined data type for initialization
    dnnl_graph_data_type_undef = 0,
    /// 16-bit/half-precision floating point.
    dnnl_graph_f16 = 1,
    /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
    dnnl_graph_bf16 = 2,
    /// 32-bit/single-precision floating point.
    dnnl_graph_f32 = 3,
    /// 32-bit signed integer.
    dnnl_graph_s32 = 4,
    /// 8-bit signed integer.
    dnnl_graph_s8 = 5,
    /// 8-bit unsigned integer.
    dnnl_graph_u8 = 6,
} dnnl_graph_data_type_t;

/// @} dnnl_graph_api_logical_tensor

/// @addtogroup dnnl_graph_api_partition
/// @{

/// Policy specifications for partitioning
typedef enum {
    /// Best optimization
    dnnl_graph_partition_policy_max = 0,
    /// Have fusion
    dnnl_graph_partition_policy_fusion,
    /// No optimization
    dnnl_graph_partition_policy_debug,
} dnnl_graph_partition_policy_t;

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_utils
/// @{

/// Status values returned by the library functions.
typedef enum {
    /// The operation was successful
    dnnl_graph_result_success,
    /// The operation was not ready
    dnnl_graph_result_not_ready,
    /// The operation failed because device was not found
    dnnl_graph_result_error_device_not_found,
    /// The operation failed because requested functionality is not implemented.
    dnnl_graph_result_error_unsupported,
    /// The operation failed because of incorrect function arguments
    dnnl_graph_result_error_invalid_argument,
    /// The operation failed because of the failed compilation
    dnnl_graph_result_error_compile_fail,
    /// The operation failed because of incorrect index
    dnnl_graph_result_error_invalid_index,
    /// The operation failed because of incorrect graph
    dnnl_graph_result_error_invalid_graph,
    /// The operation failed because of incorrect shape
    dnnl_graph_result_error_invalid_shape,
    /// The operation failed because of incorrect type
    dnnl_graph_result_error_invalid_type,
    /// The operation failed because of incorrect op
    dnnl_graph_result_error_invalid_op,
    /// The operation failed because of missing inputs or outputs
    dnnl_graph_result_error_miss_ins_outs,
    /// Unknown error
    dnnl_graph_result_error_unknown = 0x7fffffff
} dnnl_graph_result_t;

/// @} dnnl_graph_api_utils

/// @addtogroup dnnl_graph_api_engine
/// @{

/// Kind for engine
typedef enum {
    /// An unspecified engine
    dnnl_graph_any_engine,
    /// CPU engine
    dnnl_graph_cpu,
    /// GPU engine
    dnnl_graph_gpu,
} dnnl_graph_engine_kind_t;

/// @} dnnl_graph_api_engine

/// @addtogroup dnnl_graph_api_op
/// @{

/// Enumeration for op kind
typedef enum {
#define DEFINE_SYMBOL(s) k##s,
    DNNL_GRAPH_FORALL_BUILDIN_OPS(DEFINE_SYMBOL)
#undef DEFINE_SYMBOL
            kLastSymbol, // where we start counting for new symbols
    kLastOp = 10000,
} dnnl_graph_op_kind_t;

/// Kind for op's attributes
typedef enum {
    /// attributes with float type
    dnnl_graph_attribute_kind_f,
    /// atributes with list of floats
    dnnl_graph_attribute_kind_fs,
    /// attributes with int64_t type
    dnnl_graph_attribute_kind_i,
    /// atributes with list of int64_t
    dnnl_graph_attribute_kind_is,
    /// attributes with string type
    dnnl_graph_attribute_kind_s,
    /// attributes with bool type
    dnnl_graph_attribute_kind_b,
} dnnl_graph_attribute_kind_t;

/// @} dnnl_graph_api_op

/// @addtogroup dnnl_graph_api_allocator
/// @{

/// An enumeration to express the lifetime management of the allocator
typedef enum {
    /// Memory allocation with persistent lifetime, need to be freed manually
    dnnl_graph_allocator_persistent = 0,
    /// Memory allocation for output tensor
    dnnl_graph_allocator_output,
    /// Memory allocation with temporary lifetime
    dnnl_graph_allocator_temp,
} dnnl_graph_allocator_lifetime_t;

/// An attribute struct associated with allocator.
typedef struct {
    /// lifetime enumertion
    dnnl_graph_allocator_lifetime_t type;
    /// alignment value
    size_t alignment;
} dnnl_graph_allocator_attr_t;

/// Allocation call-back function interface for CPU
typedef void *(*dnnl_graph_cpu_allocate_f)(size_t, dnnl_graph_allocator_attr_t);
/// Deallocation call-back function interface for CPU
typedef void (*dnnl_graph_cpu_deallocate_f)(void *);
/// Allocation call-back function interface for SYCL device
typedef void *(*dnnl_graph_sycl_allocate_f)(
        size_t, const void *, const void *, dnnl_graph_allocator_attr_t);
/// Deallocation call-back function interface for SYCL device
typedef void (*dnnl_graph_sycl_deallocate_f)(void *, const void *);

/// @} dnnl_graph_api_allocator

/// @addtogroup dnnl_graph_api_logical_tensor
/// @{

/// Layout type specification
typedef enum {
    /// undefined layout type
    dnnl_graph_layout_type_undef = 0,
    /// any means that oneDNN graph implementation needs to decide the
    /// layout for the compiled partition.
    dnnl_graph_layout_type_any = 1,
    /// strided means that the layout is determined by the strides field.
    dnnl_graph_layout_type_strided = 2,
    /// opaque means that the layout is a target-specific layout decided by
    /// oneDNN graph implementation.
    dnnl_graph_layout_type_opaque = 3,
} dnnl_graph_layout_type_t;

/// Logical tensor property
typedef enum {
    /// undefined tensor property
    dnnl_graph_tensor_property_undef = 0,
    /// variable means the tensor can be changed among iterations
    dnnl_graph_tensor_property_variable = 1,
    /// constant means the tensor will keep unchanged among iterations
    dnnl_graph_tensor_property_constant = 2,
} dnnl_graph_tensor_property_t;

/// @brief logical tensor definition
typedef struct {
    /// Unique id of each logical tensor. Provided by framework.
    size_t id;

    /// Number of dimension. Default -1 means not initialized.
    int32_t ndims;

    /// Size of each dimension. -1 means the size is unknown on the axis.
    dnnl_graph_dims_t dims;

    /// Data type of the tensor elements.
    dnnl_graph_data_type_t data_type;

    /// Tensor property: undef, variable, or constant.
    dnnl_graph_tensor_property_t property;

    /// Layout type of the tensor: any, strided, or opaque.
    dnnl_graph_layout_type_t layout_type;
    union {
        /// Valid when layout_type is `dnnl_graph_strided`.
        /// -1 means the stride is unknown on the axis.
        dnnl_graph_dims_t strides;

        /// Valid when layout_type is `dnnl_graph_opaque`.
        /// `layout_id` is generated and managed by backend.
        size_t layout_id;
    } layout;
} dnnl_graph_logical_tensor_t;

/// @} dnnl_graph_api_logical_tensor

/// @addtogroup dnnl_graph_api_compiled_partition
/// @{

/// @brief In-place pair definition
typedef struct {
    /// The id of input tensor
    size_t input;

    /// The id of output tensor
    size_t output;
} dnnl_graph_inplace_pair_t;

/// @} dnnl_graph_api_compiled_partition

/// @cond DO_NOT_DOCUMENT_THIS

/// @brief An allocator handle
struct dnnl_graph_allocator;
typedef struct dnnl_graph_allocator dnnl_graph_allocator_t;

/// @brief A tensor handle
struct dnnl_graph_tensor;
typedef struct dnnl_graph_tensor dnnl_graph_tensor_t;

/// @brief A op handle
struct dnnl_graph_op;
typedef struct dnnl_graph_op dnnl_graph_op_t;

/// @brief A partition handle
struct dnnl_graph_partition;
typedef struct dnnl_graph_partition dnnl_graph_partition_t;

/// @brief A compiled partition handle
struct dnnl_graph_compiled_partition;
typedef struct dnnl_graph_compiled_partition dnnl_graph_compiled_partition_t;

/// @brief A graph handle
struct dnnl_graph_graph;
typedef struct dnnl_graph_graph dnnl_graph_graph_t;

/// @brief An engine handle
struct dnnl_graph_engine;
typedef struct dnnl_graph_engine dnnl_graph_engine_t;

/// @brief A thread pool handle
struct dnnl_graph_thread_pool;
typedef struct dnnl_graph_thread_pool dnnl_graph_thread_pool_t;

/// @brief A stream attr handle
struct dnnl_graph_stream_attr;
typedef struct dnnl_graph_stream_attr dnnl_graph_stream_attr_t;

/// @brief A stream handle
struct dnnl_graph_stream;
typedef struct dnnl_graph_stream dnnl_graph_stream_t;

/// @endcond

/// @addtogroup dnnl_graph_api_service
/// @{

/// Structure containing version information as per [Semantic
/// Versioning](https://semver.org)
typedef struct {
    int major; ///< Major version
    int minor; ///< Minor version
    int patch; ///< Patch version
    const char *hash; ///< Git hash of the sources (may be absent)
} dnnl_graph_version_t;

/// No runtime (disabled)
#define DNNL_GRAPH_RUNTIME_NONE 0u

/// Sequential runtime (CPU only)
#define DNNL_GRAPH_RUNTIME_SEQ 1u

/// OpenMP runtime (CPU only)
#define DNNL_GRAPH_RUNTIME_OMP 2u

/// Threadpool runtime (CPU only)
#define DNNL_GRAPH_RUNTIME_THREADPOOL 8u

/// SYCL runtime
#define DNNL_GRAPH_RUNTIME_SYCL 512u

/// DPC++ runtime
#define DNNL_GRAPH_RUNTIME_DPCPP DNNL_GRAPH_RUNTIME_SYCL

/// @} dnnl_graph_api_service

/// @} dnnl_graph_api

#ifdef __cplusplus
}
#endif
#endif
