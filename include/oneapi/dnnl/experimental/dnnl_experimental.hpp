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

#include <oneapi/dnnl/experimental/dnnl_experimental.h>

namespace dnnl {
namespace experimental {

/// Scaled Dot Product Attention (gmlp) primitive.
struct gmlp : public dnnl::primitive {
    /// Primitive descriptor for a gmlp primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a gmlp primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a gmlp primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::undef) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return query_md(query::src_md, 0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc W_gate_desc() const { return query_md(query::src_md, 1); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc W_up_desc() const { return query_md(query::src_md, 2); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc W_down_desc() const {
            return query_md(query::src_md, 3);
        }

        ///// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return query_md(query::dst_md, 0); }

        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &W_gate_desc, const memory::desc &W_up_desc,
                const memory::desc &W_down_desc, const memory::desc &output_desc,
                const primitive_attr &attr = default_attr(),
                const primitive_attr &gate_attr = default_attr(),
                const primitive_attr &up_attr = default_attr(),
                const primitive_attr &down_attr = default_attr()) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_gmlp_primitive_desc_create(&pd, aengine.get(),
                            src_desc.get(), W_gate_desc.get(), W_up_desc.get(),
                            W_down_desc.get(), output_desc.get(), attr.get(),
                            gate_attr.get(), up_attr.get(), down_attr.get());

            dnnl::error::wrap_c_api(status,
                    "could not create a primitive descriptor for a gmlp "
                    "primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    gmlp() = default;

    /// Constructs a gmlp primitive.
    /// @param pd Primitive descriptor for a gmlp primitive.
    gmlp(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a gmlp primitive from a cache blob.
    /// @param pd Primitive descriptor for a gmlp primitive.
    /// @param cache_blob Cache blob.
    gmlp(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

} // namespace experimental
} // namespace dnnl
