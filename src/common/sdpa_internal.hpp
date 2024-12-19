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

#ifndef COMMON_SDPA_INTERNAL_HPP
#define COMMON_SDPA_INTERNAL_HPP

#include "common/sdpa_internal.h"
#include "dnnl.hpp"

namespace dnnl {
namespace impl {

/// Scaled Dot Product Attention (sdpa) primitive.
struct sdpa : public dnnl::primitive {
    /// Primitive descriptor for a sdpa primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a sdpa primitive
        ///
        /// @param aengine Engine to use.
        /// @param query_desc Memory descriptor for query tensor.
        /// @param key_desc Memory descriptor for key tensor.
        /// @param value_desc Memory descriptor for value tensor.
        /// @param output_desc Memory descriptor for output tensor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param kq_attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param vs_attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        primitive_desc(const engine &aengine, const memory::desc &query_desc,
                const memory::desc &key_desc, const memory::desc &value_desc,
                const memory::desc &output_desc,
                const primitive_attr &attr = default_attr(),
                const primitive_attr &kq_attr = default_attr(),
                const primitive_attr &vs_attr = default_attr())
            : primitive_desc(aengine, query_desc, key_desc, value_desc, nullptr,
                    memory::data_type::undef, output_desc, false, 1, false,
                    attr, kq_attr, vs_attr) {}

        /// Constructs a primitive descriptor for a sdpa primitive
        ///
        /// @param aengine Engine to use.
        /// @param query_desc Memory descriptor for query tensor.
        /// @param key_desc Memory descriptor for key tensor.
        /// @param value_desc Memory descriptor for value tensor.
        /// @param output_desc Memory descriptor for output tensor.
        /// @param attn_mask_desc Memory descriptor for attention mask tensor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        primitive_desc(const engine &aengine, const memory::desc &query_desc,
                const memory::desc &key_desc, const memory::desc &value_desc,
                const memory::desc &attn_mask_desc,
                const memory::desc &output_desc,
                const primitive_attr &attr = default_attr(),
                const primitive_attr &kq_attr = default_attr(),
                const primitive_attr &vs_attr = default_attr())
            : primitive_desc(aengine, query_desc, key_desc, value_desc,
                    &attn_mask_desc, memory::data_type::undef, output_desc,
                    false, 1, false, attr, kq_attr, vs_attr) {}

        /// Constructs a primitive descriptor for a sdpa primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a sdpa primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::undef) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc query_desc() const {
            return query_md((dnnl::query)query::src_md, 0);
        }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc key_desc() const {
            return query_md((dnnl::query)query::src_md, 1);
        }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc value_desc() const {
            return query_md((dnnl::query)query::src_md, 2);
        }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc attn_mask_desc() const {
            return query_md((dnnl::query)query::src_md, 3);
        }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const {
            return query_md((dnnl::query)query::weights_md, 1);
        }

        ///// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return query_md((dnnl::query)query::dst_md, 0);
        }

        primitive_desc(const engine &aengine, const memory::desc &query_desc,
                const memory::desc &key_desc, const memory::desc &value_desc,
                const memory::desc *attn_mask_desc, memory::data_type scale_dt,
                const memory::desc &output_desc, bool invert_scale,
                dnnl_dim_t kv_head_number, bool causal_mask,
                const primitive_attr &attr,
                const primitive_attr &kq_attr = default_attr(),
                const primitive_attr &vs_attr = default_attr()) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_sdpa_primitive_desc_create(&pd,
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

    /// Constructs a sdpa primitive from a cache blob.
    /// @param pd Primitive descriptor for a sdpa primitive.
    /// @param cache_blob Cache blob.
    sdpa(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};
} // namespace impl
} // namespace dnnl

#endif
