/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef UTILS_DNNL_QUERY_HPP
#define UTILS_DNNL_QUERY_HPP

#include <cstdint>
#include <string>

#include "oneapi/dnnl/dnnl.h"

dnnl_prop_kind_t query_prop_kind(const_dnnl_primitive_desc_t pd);
dnnl_primitive_kind_t query_prim_kind(const_dnnl_primitive_desc_t pd);

std::string query_impl_info(const_dnnl_primitive_desc_t pd);

// General interface of quering memory desc.
const dnnl_memory_desc_t &query_md(
        const_dnnl_primitive_desc_t pd, dnnl_query_t what, int index = 0);
// Particular interface of quering through execution argument and its index.
const dnnl_memory_desc_t &query_md(
        const_dnnl_primitive_desc_t pd, int index = 0);

dnnl_engine_t query_engine(const_dnnl_primitive_desc_t pd,
        dnnl_query_t engine_type = dnnl_query_engine);

int64_t query_mem_consumption(const_dnnl_primitive_desc_t pd);

int query_n_inputs(const_dnnl_primitive_desc_t pd);
int query_n_outputs(const_dnnl_primitive_desc_t pd);

const_dnnl_post_ops_t query_post_ops(const_dnnl_primitive_attr_t attr);
const_dnnl_post_ops_t query_post_ops(const_dnnl_primitive_desc_t pd);
const_dnnl_primitive_attr_t query_attr(const_dnnl_primitive_desc_t pd);
const_dnnl_primitive_desc_t query_pd(dnnl_primitive_t prim);
const_dnnl_op_desc_t query_op_desc(const_dnnl_primitive_desc_t pd);

dnnl_engine_kind_t query_engine_kind(const dnnl_engine_t &engine);

#endif
