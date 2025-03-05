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

#ifndef DNNL_TEST_INTERNAL_MLP_INTERNAL_HPP
#define DNNL_TEST_INTERNAL_MLP_INTERNAL_HPP

#include "dnnl.hpp"

#define DNNL_ARG_SRC DNNL_ARG_SRC_0
#define DNNL_ARG_WTS_GATE DNNL_ARG_SRC_1
#define DNNL_ARG_WTS_UP DNNL_ARG_SRC_2
#define DNNL_ARG_WTS_DOWN DNNL_ARG_SRC_3

/// Creates a primitive descriptor for a gated mlp primitive
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param src_desc src memory descriptor.
/// @param W_gate_desc W_gate memory descriptor.
/// @param W_up_desc W_up memory descriptor.
/// @param W_down_desc W_down memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @param attr wts_gate attributes (can be NULL).
/// @param attr wts_up attributes (can be NULL).
/// @param attr wts_down attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.

dnnl_status_t DNNL_API gmlp_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t W_gate_desc,
        const_dnnl_memory_desc_t W_up_desc,
        const_dnnl_memory_desc_t W_down_desc, const_dnnl_memory_desc_t dst_desc,
        dnnl_alg_kind_t activation, const_dnnl_primitive_attr_t attr,
        const_dnnl_primitive_attr_t gate_attr,
        const_dnnl_primitive_attr_t up_attr,
        const_dnnl_primitive_attr_t down_attr);

namespace dnnl {
namespace impl {

/// Gated MLP (gmlp) internal primitive.
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

        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &W_gate_desc, const memory::desc &W_up_desc,
                const memory::desc &W_down_desc,
                const memory::desc &output_desc, const alg_kind_t &activation,
                const primitive_attr &attr = default_attr(),
                const primitive_attr &gate_attr = default_attr(),
                const primitive_attr &up_attr = default_attr(),
                const primitive_attr &down_attr = default_attr()) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = gmlp_primitive_desc_create(&pd,
                    aengine.get(), src_desc.get(), W_gate_desc.get(),
                    W_up_desc.get(), W_down_desc.get(), output_desc.get(),
                    activation, attr.get(), gate_attr.get(), up_attr.get(),
                    down_attr.get());

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
};

} // namespace impl
} // namespace dnnl

#endif
