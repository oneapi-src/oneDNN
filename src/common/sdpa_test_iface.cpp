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

#include "common/c_types_map.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/sdpa_pd.hpp"
#include "common/sdpa_types.hpp"
#include "common/sdpa_utils.hpp"
#include "opdesc.hpp"

using dnnl::impl::status_t;
using namespace dnnl::impl;

dnnl_status_t DNNL_API sdpa_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t query_desc, const_dnnl_memory_desc_t key_desc,
        const_dnnl_memory_desc_t value_desc, const_dnnl_memory_desc_t dst_desc,
        const_dnnl_memory_desc_t mask_desc, dnnl_data_type_t scale_dt,
        bool invert_scale, dnnl_dim_t kv_head_number, bool causal_mask,
        const_dnnl_primitive_attr_t attr, const_dnnl_primitive_attr_t kq_attr,
        const_dnnl_primitive_attr_t vs_attr) {
    if (auto err = sdpa_attr_check(query_desc, key_desc, value_desc, engine,
                attr, kq_attr, vs_attr)) {
        return err;
    }
    dnnl::impl::sdpa_desc_t sdpa_desc = dnnl::impl::create_sdpa_desc(query_desc,
            key_desc, value_desc, dst_desc, mask_desc,
            (dnnl::impl::data_type_t)scale_dt, invert_scale, kv_head_number,
            causal_mask, kq_attr, vs_attr);
    return dnnl::impl::primitive_desc_create(primitive_desc_iface, engine,
            (const dnnl::impl::op_desc_t *)&sdpa_desc, nullptr, attr);
}
