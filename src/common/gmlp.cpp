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

#include "gated_mlp_pd.hpp"
#include "gated_mlp_utils.hpp"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

dnnl_status_t DNNL_API gmlp_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t v, const_dnnl_memory_desc_t wg,
        const_dnnl_memory_desc_t wu, const_dnnl_memory_desc_t wd,
        const_dnnl_memory_desc_t dst, dnnl_alg_kind_t activation,
        const_dnnl_primitive_attr_t attr, const_dnnl_primitive_attr_t gate_attr,
        const_dnnl_primitive_attr_t up_attr,
        const_dnnl_primitive_attr_t down_attr) {
    dnnl::impl::gated_mlp_desc_t gated_mlp_desc
            = dnnl::impl::create_gated_mlp_desc(v, wg, wu, wd, dst, activation,
                    gate_attr, up_attr, down_attr);
    return dnnl::impl::primitive_desc_create(primitive_desc_iface, engine,
            (const dnnl::impl::op_desc_t *)&gated_mlp_desc, nullptr, attr);
}
