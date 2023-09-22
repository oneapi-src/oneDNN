/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "common.hpp"

#include "utils/dnnl_query.hpp"

dnnl_prop_kind_t query_prop_kind(const_dnnl_primitive_desc_t pd) {
    dnnl_prop_kind_t prop_kind = dnnl_prop_kind_undef;
    dnnl_primitive_desc_query(pd, dnnl_query_prop_kind, 0, &prop_kind);
    return prop_kind;
}

dnnl_primitive_kind_t query_prim_kind(const_dnnl_primitive_desc_t pd) {
    dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive;
    dnnl_primitive_desc_query(pd, dnnl_query_primitive_kind, 0, &prim_kind);
    return prim_kind;
}

dnnl_alg_kind_t query_alg_kind(const_dnnl_primitive_desc_t pd) {
    dnnl_alg_kind_t alg_kind = dnnl_alg_kind_undef;
    dnnl_primitive_desc_query(pd, dnnl_query_alg_kind, 0, &alg_kind);
    return alg_kind;
}

std::string query_impl_info(const_dnnl_primitive_desc_t pd) {
    const char *str = nullptr;
    dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &str);
    std::string s(str);
    return s;
}

const_dnnl_memory_desc_t query_md(
        const_dnnl_primitive_desc_t pd, dnnl_query_t what, int index) {
    return dnnl_primitive_desc_query_md(pd, what, index);
}

const_dnnl_memory_desc_t query_md(const_dnnl_primitive_desc_t pd, int index) {
    return query_md(pd, dnnl_query_exec_arg_md, index);
}

const_dnnl_memory_desc_t query_md(const_dnnl_memory_t memory) {
    const_dnnl_memory_desc_t md = nullptr;
    auto st = dnnl_memory_get_memory_desc(memory, &md);
    if (st != dnnl_success)
        BENCHDNN_PRINT(0, "%s\n", "Error: querying md from memory failed");
    return md;
}

dnnl_engine_t query_engine(
        const_dnnl_primitive_desc_t pd, dnnl_query_t engine_type) {
    dnnl_engine_t engine;
    dnnl_primitive_desc_query(pd, engine_type, 0, &engine);
    return engine;
}

int64_t query_mem_consumption(const_dnnl_primitive_desc_t pd) {
    int64_t size = 0;
    dnnl_primitive_desc_query(pd, dnnl_query_memory_consumption_s64, 0, &size);
    return size;
}

int query_n_inputs(const_dnnl_primitive_desc_t pd) {
    return dnnl_primitive_desc_query_s32(pd, dnnl_query_num_of_inputs_s32, 0);
}

int query_n_outputs(const_dnnl_primitive_desc_t pd) {
    return dnnl_primitive_desc_query_s32(pd, dnnl_query_num_of_outputs_s32, 0);
}

const_dnnl_post_ops_t query_post_ops(const_dnnl_primitive_attr_t attr) {
    const_dnnl_post_ops_t post_ops {};
    dnnl_primitive_attr_get_post_ops(attr, &post_ops);
    return post_ops;
}

const_dnnl_post_ops_t query_post_ops(const_dnnl_primitive_desc_t pd) {
    const_dnnl_post_ops_t post_ops {};
    dnnl_primitive_attr_get_post_ops(query_attr(pd), &post_ops);
    return post_ops;
}

const_dnnl_primitive_attr_t query_attr(const_dnnl_primitive_desc_t pd) {
    const_dnnl_primitive_attr_t attr {};
    dnnl_primitive_desc_get_attr(pd, &attr);
    return attr;
}

const_dnnl_primitive_desc_t query_pd(dnnl_primitive_t prim) {
    const_dnnl_primitive_desc_t pd {};
    dnnl_primitive_get_primitive_desc(prim, &pd);
    return pd;
}

dnnl_engine_kind_t query_engine_kind(const dnnl_engine_t &engine) {
    dnnl_engine_kind_t engine_kind = dnnl_any_engine;
    dnnl_engine_get_kind(engine, &engine_kind);
    return engine_kind;
}

#ifdef DNNL_EXPERIMENTAL_SPARSE
dnnl_sparse_encoding_t query_md_sparse_encoding(const_dnnl_memory_desc_t md) {
    dnnl_sparse_encoding_t encoding = dnnl_sparse_encoding_undef;
    if (!md) return encoding;
    dnnl_memory_desc_query_v2(md, dnnl_query_sparse_encoding, 0, &encoding);
    return encoding;
}

dnnl_dim_t query_md_nnz(const_dnnl_memory_desc_t md) {
    dnnl_dim_t nnz = 0;
    if (!md) return nnz;
    dnnl_memory_desc_query_v2(md, dnnl_query_nnz_s64, 0, &nnz);
    return nnz;
}
#endif

int query_md_num_handles(const_dnnl_memory_desc_t md) {
#ifdef DNNL_EXPERIMENTAL_SPARSE
    int nhandles = 0;
    if (!md) return nhandles;
    dnnl_memory_desc_query_v2(md, dnnl_query_num_handles_s32, 0, &nhandles);
    return nhandles;
#else
    return 1;
#endif
}

int query_md_ndims(const_dnnl_memory_desc_t md) {
    int ndims = 0;
    if (!md) return ndims;
    dnnl_memory_desc_query(md, dnnl_query_ndims_s32, &ndims);
    return ndims;
}

int query_md_inner_nblks(const_dnnl_memory_desc_t md) {
    int inner_nblks = 0;
    if (!md) return inner_nblks;
    dnnl_memory_desc_query(md, dnnl_query_inner_nblks_s32, &inner_nblks);
    return inner_nblks;
}

dnnl_dim_t query_md_submemory_offset(const_dnnl_memory_desc_t md) {
    dnnl_dim_t submemory_offset = 0;
    if (!md) return submemory_offset;
    dnnl_memory_desc_query(
            md, dnnl_query_submemory_offset_s64, &submemory_offset);
    return submemory_offset;
}

dnnl_data_type_t query_md_data_type(
        const_dnnl_memory_desc_t md, int buffer_index) {
    dnnl_data_type_t dt = dnnl_data_type_undef;
    if (!md) return dt;
#ifdef DNNL_EXPERIMENTAL_SPARSE
    dnnl_memory_desc_query_v2(md, dnnl_query_data_type, buffer_index, &dt);
#else
    dnnl_memory_desc_query(md, dnnl_query_data_type, &dt);
#endif
    return dt;
}

dnnl_format_kind_t query_md_format_kind(const_dnnl_memory_desc_t md) {
    dnnl_format_kind_t format_kind = dnnl_format_kind_undef;
    if (!md) return format_kind;
    dnnl_memory_desc_query(md, dnnl_query_format_kind, &format_kind);
    return format_kind;
}

static const dnnl_dims_t &query_md_array_member(
        const_dnnl_memory_desc_t md, dnnl_query_t what) {
    static const dnnl_dims_t dummy {};
    if (!md) return dummy;
    const dnnl_dims_t *res;
    dnnl_memory_desc_query(md, what, &res);
    return *res;
}

const dnnl_dims_t &query_md_dims(const_dnnl_memory_desc_t md) {
    return query_md_array_member(md, dnnl_query_dims);
}

const dnnl_dims_t &query_md_padded_dims(const_dnnl_memory_desc_t md) {
    return query_md_array_member(md, dnnl_query_padded_dims);
}

const dnnl_dims_t &query_md_padded_offsets(const_dnnl_memory_desc_t md) {
    return query_md_array_member(md, dnnl_query_padded_offsets);
}

const dnnl_dims_t &query_md_strides(const_dnnl_memory_desc_t md) {
    return query_md_array_member(md, dnnl_query_strides);
}

const dnnl_dims_t &query_md_inner_blks(const_dnnl_memory_desc_t md) {
    return query_md_array_member(md, dnnl_query_inner_blks);
}

const dnnl_dims_t &query_md_inner_idxs(const_dnnl_memory_desc_t md) {
    return query_md_array_member(md, dnnl_query_inner_idxs);
}
