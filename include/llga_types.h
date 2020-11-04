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

#ifndef LLGA_LLGA_TYPES_H
#define LLGA_LLGA_TYPES_H

#include <stddef.h>
#include <stdint.h>

#include "buildin_ops.h"

#if defined _WIN32 || defined __CYGWIN__
#define LLGA_HELPER_DLL_IMPORT __declspec(dllimport)
#define LLGA_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define LLGA_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define LLGA_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#else
#define LLGA_HELPER_DLL_IMPORT
#define LLGA_HELPER_DLL_EXPORT
#endif
#endif

#ifdef LLGA_DLL_EXPORTS
#define LLGA_API LLGA_HELPER_DLL_EXPORT
#else
#define LLGA_API LLGA_HELPER_DLL_IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LLGA_MAX_NDIMS 12
#define LLGA_UNKNOWN_NDIMS -1
#define LLGA_UNKNOWN_DIM -1

/// A type to describe tensor dimension.
typedef int64_t llga_dim_t;

/// A type to describe tensor dimensions.
typedef llga_dim_t llga_dims_t[LLGA_MAX_NDIMS];

typedef enum llga_data_type {
    llga_data_type_undef = 0,
    /// 16-bit/half-precision floating point.
    llga_f16 = 1,
    /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
    llga_bf16 = 2,
    /// 32-bit/single-precision floating point.
    llga_f32 = 3,
    /// 32-bit signed integer.
    llga_s32 = 4,
    /// 8-bit signed integer.
    llga_s8 = 5,
    /// 8-bit unsigned integer.
    llga_u8 = 6,
} llga_data_type_t;

typedef enum llga_partition_policy {
    llga_partition_policy_max = 0,
    llga_partition_policy_fusion,
    llga_partition_policy_debug,
} llga_partition_policy_t;

typedef enum llga_result {
    llga_result_success,
    llga_result_not_ready,
    llga_result_error_device_not_found,
    llga_result_error_unsupported,
    llga_result_error_invalid_argument,
    llga_result_error_compile_fail,
    llga_result_error_invalid_index,
    llga_result_error_invalid_graph,
    llga_result_error_unknown = 0x7fffffff
} llga_result_t;

typedef enum llga_engine_kind {
    llga_any_engine,
    llga_cpu,
    llga_gpu,
} llga_engine_kind_t;

typedef enum llga_op_kind {
#define DEFINE_SYMBOL(s) k##s,
    LLGA_FORALL_BUILDIN_OPS(DEFINE_SYMBOL)
#undef DEFINE_SYMBOL
            kLastSymbol, // where we start counting for new symbols
    kLastOp = 10000,
} llga_op_kind_t;

typedef enum llga_attribute_kind {
    llga_attribute_kind_f,
    llga_attribute_kind_fs,
    llga_attribute_kind_i,
    llga_attribute_kind_is,
    llga_attribute_kind_s,
    llga_attribute_kind_b,
} llga_attribute_kind_t;

/// @brief An enumeration to express the lifetime management of the allocator
typedef enum {
    llga_allocator_persistent = 0,
    llga_allocator_output,
    llga_allocator_temp,
} llga_allocator_lifetime_t;

/// @brief An attribute struct associated with allocator.
typedef struct llga_allocator_attr {
    llga_allocator_lifetime_t type;
    size_t alignment;
} llga_allocator_attr_t;

/// @brief A allocator handle
typedef struct llga_allocator llga_allocator_t;
/// Regarding the allocator call-back function, we also provide a hint/attribute
/// to integration, which indicates that which type of memory is being requested
/// (persistent/output/temp) and the expected alignment of memory allocation.
typedef void *(*llga_cpu_allocate_f)(size_t, llga_allocator_attr_t);
typedef void (*llga_cpu_deallocate_f)(void *);
typedef void *(*llga_sycl_allocate_f)(
        size_t, const void *, const void *, llga_allocator_attr_t);
typedef void (*llga_sycl_deallocate_f)(void *, const void *);

/// Layout type specification
typedef enum {
    llga_layout_type_undef = 0,
    llga_layout_type_any = 1,
    llga_layout_type_strided = 2,
    llga_layout_type_opaque = 3,
} llga_layout_type_t;

/// @brief logical tensor definition
typedef struct {
    /// Unique id of each logical tensor. Provided by framework.
    size_t id;

    /// Number of dimension. Default -1 means not initialized.
    int32_t ndims;

    /// Size of each dimension. -1 means the size is unknown on the axis.
    llga_dims_t dims;

    /// Data type of the tensor elements.
    llga_data_type_t data_type;

    /// Layout type of the tensor: any, strided, or opaque.
    llga_layout_type_t layout_type;
    union {
        /// Valid when layout_type is `llga_strided`.
        /// -1 means the stride is unknown on the axis.
        llga_dims_t strides;

        /// Valid when layout_type is `llga_opaque`.
        /// `layout_id` is generated and managed by backend.
        size_t layout_id;
    } layout;
} llga_logical_tensor_t;

/// @brief A tensor handle
struct llga_tensor;
typedef struct llga_tensor llga_tensor_t;

/// @brief A llga op handle
struct llga_op;
typedef struct llga_op llga_op_t;

/// @brief A partition handle
struct llga_partition;
typedef struct llga_partition llga_partition_t;

/// @brief A partition_list handle
struct llga_partition_list;
typedef struct llga_partition_list llga_partition_list_t;

/// @brief A compiled partition handle
struct llga_compiled_partition;
typedef struct llga_compiled_partition llga_compiled_partition_t;

/// @brief a backend graph handle
struct llga_graph;
typedef struct llga_graph llga_graph_t;

/// @brief A engine handle
struct llga_engine;
typedef struct llga_engine llga_engine_t;

/// @brief A thread pool handle
struct llga_thread_pool;
typedef struct llga_thread_pool llga_thread_pool_t;

/// @brief A stream attr handle
struct llga_stream_attr;
typedef struct llga_stream_attr llga_stream_attr_t;

/// @brief A stream handle
struct llga_stream;
typedef struct llga_stream llga_stream_t;

#ifdef __cplusplus
}
#endif
#endif
