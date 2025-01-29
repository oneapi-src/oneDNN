/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef COMMON_SDPA_INTERNAL_HPP
#define COMMON_SDPA_INTERNAL_HPP

#include "dnnl.hpp"

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

dnnl_status_t DNNL_API sdpa_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t query_desc, const_dnnl_memory_desc_t key_desc,
        const_dnnl_memory_desc_t value_desc, const_dnnl_memory_desc_t dst_desc,
        const_dnnl_memory_desc_t mask_desc, dnnl_data_type_t scale_dt,
        bool invert_scale, dnnl_dim_t kv_head_number, bool causal_mask,
        const_dnnl_primitive_attr_t attr, const_dnnl_primitive_attr_t kq_attr,
        const_dnnl_primitive_attr_t vs_attr);

namespace dnnl {
namespace impl {

/// Scaled Dot Product Attention (sdpa) internal primitive.
/// Implementing internally for more flexible validation
struct sdpa : public dnnl::primitive {
    /// Primitive descriptor for a sdpa primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        primitive_desc(const engine &aengine, const memory::desc &query_desc,
                const memory::desc &key_desc, const memory::desc &value_desc,
                const memory::desc *attn_mask_desc, memory::data_type scale_dt,
                const memory::desc &output_desc, bool invert_scale,
                memory::dim kv_head_number, bool causal_mask,
                const primitive_attr &attr = default_attr(),
                const primitive_attr &kq_attr = default_attr(),
                const primitive_attr &vs_attr = default_attr()) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = sdpa_primitive_desc_create(&pd,
                    aengine.get(), query_desc.get(), key_desc.get(),
                    value_desc.get(), output_desc.get(),
                    optional_arg(attn_mask_desc), (dnnl_data_type_t)scale_dt,
                    invert_scale, kv_head_number, causal_mask, attr.get(),
                    kq_attr.get(), vs_attr.get());

            dnnl::error::wrap_c_api(status,
                    "could not create a primitive descriptor for a sdpa "
                    "primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    sdpa() = default;

    /// Constructs a sdpa primitive.
    /// @param pd Primitive descriptor for a sdpa primitive.
    sdpa(const primitive_desc &pd) : primitive(pd) {}
};
} // namespace impl
} // namespace dnnl

#endif
