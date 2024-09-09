/*******************************************************************************
* Copyright 2024 Intel Corporation
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
/// ukernel C API types definitions

#ifndef ONEAPI_DNNL_DNNL_UKERNEL_TYPES_H
#define ONEAPI_DNNL_DNNL_UKERNEL_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "oneapi/dnnl/dnnl_types.h"

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_ukernel
/// @{

#ifdef DNNL_EXPERIMENTAL_UKERNEL

/// Packing specification
typedef enum {
    /// Undefined pack type. A guard value.
    dnnl_pack_type_undef = 0,
    /// Plain, not transposed layout. Similar to format_tag::ab.
    dnnl_pack_type_no_trans,
    /// Plain, transposed layout. Similar to format_tag::ba.
    dnnl_pack_type_trans,
    /// Packed by 32 bits along K dimension layout.
    dnnl_pack_type_pack32,
} dnnl_pack_type_t;

/// @struct dnnl_ukernel_attr_params
/// An opaque structure to describe ukernel attributes memory storage.
struct dnnl_ukernel_attr_params;

/// A ukernel attributes memory storage handle.
typedef struct dnnl_ukernel_attr_params *dnnl_ukernel_attr_params_t;

/// A constant ukernel attributes memory storage handle.
typedef const struct dnnl_ukernel_attr_params *const_dnnl_ukernel_attr_params_t;

/// @addtogroup dnnl_api_ukernel_brgemm
/// @{

/// @struct dnnl_brgemm
/// An opaque structure to describe a brgemm ukernel.
struct dnnl_brgemm;

/// A brgemm ukernel handle.
typedef struct dnnl_brgemm *dnnl_brgemm_t;

/// A constant brgemm ukernel handle.
typedef const struct dnnl_brgemm *const_dnnl_brgemm_t;

/// @struct dnnl_transform
/// An opaque structure to describe a transform routine.
struct dnnl_transform;

/// A transform routine handle.
typedef struct dnnl_transform *dnnl_transform_t;

/// A constant transform routine handle.
typedef const struct dnnl_transform *const_dnnl_transform_t;

/// @} dnnl_api_ukernel_brgemm
#endif

/// @} dnnl_api_ukernel

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif /* ONEAPI_DNNL_DNNL_UKERNEL_TYPES_H */
