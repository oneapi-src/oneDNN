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

#ifndef COMMON_SDPA_INTERNAL_H
#define COMMON_SDPA_INTERNAL_H

#include <oneapi/dnnl/dnnl.h>

#define DNNL_ARG_QUERIES DNNL_ARG_SRC_0
#define DNNL_ARG_KEYS DNNL_ARG_SRC_1
#define DNNL_ARG_VALUES DNNL_ARG_SRC_2
#define DNNL_ARG_ATTN_MASK DNNL_ARG_SHIFT

/// @addtogroup dnnl_api_sdpa
/// @{

/// Creates a primitive descriptor for a scaled dot product attention primitive
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param query_desc Query memory descriptor (tensor Q)
/// @param key_desc Key memory descriptor (tensor K)
/// @param value_desc Value memory descriptor (tensor V)
/// @param dst_desc Destination memory descriptor.
/// @param attn_mask_desc Attention mask memory descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @param kq_attr Attribute for the Key/Query matmul operation(can be NULL).
/// @param vs_attr Attribute for the Value/Score matmul operation(can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.

dnnl_status_t DNNL_API dnnl_sdpa_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t query_desc, const_dnnl_memory_desc_t key_desc,
        const_dnnl_memory_desc_t value_desc, const_dnnl_memory_desc_t dst_desc,
        const_dnnl_memory_desc_t mask_desc, dnnl_data_type_t scale_dt,
        bool invert_scale, dnnl_dim_t kv_head_number, bool causal_mask,
        const_dnnl_primitive_attr_t attr, const_dnnl_primitive_attr_t kq_attr,
        const_dnnl_primitive_attr_t vs_attr);

/// @} dnnl_api_sdpa

#endif
