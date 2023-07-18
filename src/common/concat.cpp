/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "concat_pd.hpp"
#include "engine.hpp"
#include "impl_list_item.hpp"
#include "primitive_cache.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_hashing.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;

#define VCHECK_CONCAT(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, concat, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_CONCAT_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, concat, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);
namespace dnnl {
namespace impl {

status_t concat_primitive_desc_create(std::shared_ptr<primitive_desc_t> &pd,
        engine_t *engine, const memory_desc_t *dst_md, int n, int concat_dim,
        const memory_desc_t *const *src_mds, const primitive_attr_t *attr) {
    VCHECK_CONCAT(!any_null(src_mds) && n > 0, VERBOSE_NULL_ARG);

    if (attr == nullptr)
        attr = &default_attr();
    else {
        using smask_t = primitive_attr_t::skip_mask_t;
        VCHECK_CONCAT_UNIMPL(attr->has_default_values(smask_t::scales_runtime),
                VERBOSE_UNSUPPORTED_ATTR);
        const auto &scales = attr->scales_;
        if (!scales.has_default_values())
            for (const auto &s : scales.scales_)
                VCHECK_CONCAT_UNIMPL(
                        s.second.mask_ == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    const int ndims = src_mds[0]->ndims;
    const dims_t &dims = src_mds[0]->dims;
    const data_type_t dt = src_mds[0]->data_type;
    VCONDCHECK(primitive, create, check, concat,
            !memory_desc_wrapper(src_mds[0]).has_runtime_dims_or_strides(),
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    dim_t concat_dim_sz = dims[concat_dim];
    VCHECK_CONCAT(!memory_desc_wrapper(src_mds[0]).format_any(),
            VERBOSE_UNSUPPORTED_TAG_S, "src");

#define SRC2STR(i) (std::string("src_") + std::to_string(i)).c_str()
    for (int i = 1; i < n; ++i) {
        const memory_desc_t &src_md = *src_mds[i];
        VCHECK_CONCAT(src_md.ndims == ndims, VERBOSE_INCONSISTENT_NDIMS,
                "src_0", SRC2STR(i));
        VCHECK_CONCAT(!memory_desc_wrapper(src_md).format_any(),
                VERBOSE_UNSUPPORTED_TAG_S, SRC2STR(i));
        VCONDCHECK(primitive, create, check, concat,
                !memory_desc_wrapper(src_md).has_runtime_dims_or_strides(),
                status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

        for (int d = 0; d < ndims; ++d) {
            if (d == concat_dim) continue;
            VCHECK_CONCAT(src_md.dims[d] == dims[d], VERBOSE_INCONSISTENT_DIM,
                    "src_0", d, SRC2STR(i), d);
        }
        VCHECK_CONCAT(src_md.data_type == dt, VERBOSE_INCONSISTENT_DT, "src_0",
                SRC2STR(i));
        concat_dim_sz += src_md.dims[concat_dim];
    }
#undef SRC2STR

    memory_desc_t dummy_dst_md;
    if (dst_md) {
        VCHECK_CONCAT(dst_md->ndims == ndims, VERBOSE_INCONSISTENT_NDIMS,
                "src_0", "dst");
        VCHECK_CONCAT(
                !memory_desc_wrapper(dst_md).has_runtime_dims_or_strides(),
                VERBOSE_RUNTIMEDIM_UNSUPPORTED);
        for (int d = 0; d < ndims; ++d)
            if (d == concat_dim) {
                VCHECK_CONCAT(dst_md->dims[d] == concat_dim_sz, VERBOSE_BAD_DIM,
                        "dst", d);
            } else {
                VCHECK_CONCAT(dst_md->dims[d] == dims[d],
                        VERBOSE_INCONSISTENT_DIM, "src_0", d, "dst", d);
            }
    } else {
        dummy_dst_md = *src_mds[0];
        dummy_dst_md.dims[concat_dim] = concat_dim_sz;
        dummy_dst_md.format_kind = format_kind::any;
        dst_md = &dummy_dst_md;
    }

    auto desc = concat_desc_t(
            primitive_kind::concat, dst_md, n, concat_dim, src_mds);
    primitive_hashing::key_t key(
            engine, reinterpret_cast<op_desc_t *>(&desc), attr, 0, {});
    pd = primitive_cache().get_pd(key);

    if (pd) return success;

    concat_pd_t *concat_pd = nullptr;
    for (auto c = engine->get_concat_implementation_list(); *c; ++c) {
        if ((*c)(&concat_pd, engine, attr, dst_md, n, concat_dim, src_mds)
                == success) {
            pd.reset(concat_pd);
            return success;
        }
    }
    return unimplemented;
}

status_t concat_primitive_desc_create(std::shared_ptr<primitive_desc_t> &pd,
        engine_t *engine, const memory_desc_t *dst_md, int n, int concat_dim,
        const memory_desc_t *src_mds, const primitive_attr_t *attr) {
    std::vector<const memory_desc_t *> src_mds_ptrs(n);
    for (int i = 0; i < n; i++)
        src_mds_ptrs[i] = &src_mds[i];
    return concat_primitive_desc_create(
            pd, engine, dst_md, n, concat_dim, src_mds_ptrs.data(), attr);
}

} // namespace impl
} // namespace dnnl

status_t dnnl_concat_primitive_desc_create(
        primitive_desc_iface_t **concat_pd_iface, engine_t *engine,
        const memory_desc_t *dst_md, int n, int concat_dim,
        const memory_desc_t *const *src_mds, const primitive_attr_t *attr) {
    if (any_null(concat_pd_iface)) return invalid_arguments;

    std::shared_ptr<primitive_desc_t> pd;
    CHECK(concat_primitive_desc_create(
            pd, engine, dst_md, n, concat_dim, src_mds, attr));
    return safe_ptr_assign(
            *concat_pd_iface, new primitive_desc_iface_t(pd, engine));
}
