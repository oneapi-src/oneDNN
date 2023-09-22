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
#include "engine.hpp"
#include "impl_list_item.hpp"
#include "primitive_cache.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_hashing.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "sum_pd.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;

#define VCHECK_SUM(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, sum, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_SUM_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, sum, (cond), status::unimplemented, \
            msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {

status_t sum_primitive_desc_create(primitive_desc_iface_t **sum_pd_iface,
        const memory_desc_t *dst_md, int n, const float *scales,
        const memory_desc_t *const *src_mds, const primitive_attr_t *attr,
        engine_t *engine) {

    VCHECK_SUM(!any_null(sum_pd_iface, src_mds, scales) && n > 0,
            VERBOSE_NULL_ARG);

    if (attr == nullptr) attr = &default_attr();
    VCHECK_SUM_UNIMPL(attr->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    const int ndims = src_mds[0]->ndims;
    const dims_t &dims = src_mds[0]->dims;
    VCONDCHECK(primitive, create, check, sum,
            !memory_desc_wrapper(src_mds[0]).has_runtime_dims_or_strides(),
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    VCHECK_SUM(!memory_desc_wrapper(src_mds[0]).format_any(),
            VERBOSE_UNSUPPORTED_TAG_S, "src");

#define SRC2STR(i) (std::string("src_") + std::to_string(i)).c_str()
    for (int i = 1; i < n; ++i) {
        const memory_desc_t &src_md = *src_mds[i];
        VCHECK_SUM(src_md.ndims == ndims, VERBOSE_INCONSISTENT_NDIMS, "src_0",
                SRC2STR(i));
        VCHECK_SUM(!memory_desc_wrapper(src_md).format_any(),
                VERBOSE_UNSUPPORTED_TAG_S, SRC2STR(i));
        VCONDCHECK(primitive, create, check, sum,
                !memory_desc_wrapper(src_md).has_runtime_dims_or_strides(),
                status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);
        for (int d = 0; d < ndims; ++d)
            VCHECK_SUM(src_md.dims[d] == dims[d], VERBOSE_INCONSISTENT_DIM,
                    "src_0", d, SRC2STR(i), d);
    }
#undef SRC2STR

    memory_desc_t dummy_dst_md;
    if (dst_md) {
        VCHECK_SUM(dst_md->ndims == ndims, VERBOSE_INCONSISTENT_NDIMS, "src_0",
                "dst");
        VCHECK_SUM(!memory_desc_wrapper(dst_md).has_runtime_dims_or_strides(),
                VERBOSE_RUNTIMEDIM_UNSUPPORTED);
        for (int d = 0; d < ndims; ++d) {
            VCHECK_SUM(dst_md->dims[d] == dims[d], VERBOSE_INCONSISTENT_DIM,
                    "src_0", d, "dst", d);
        }
    } else {
        dummy_dst_md = *src_mds[0];
        dummy_dst_md.format_kind = format_kind::any;
        dst_md = &dummy_dst_md;
    }

    auto desc = sum_desc_t(primitive_kind::sum, dst_md, n, scales, src_mds);
    primitive_hashing::key_t key(
            engine, reinterpret_cast<op_desc_t *>(&desc), attr, 0, {});
    auto pd = primitive_cache().get_pd(key);

    if (pd) {
        return safe_ptr_assign(
                *sum_pd_iface, new primitive_desc_iface_t(pd, engine));
    }

    for (auto s = engine->get_sum_implementation_list(); *s; ++s) {
        sum_pd_t *sum_pd = nullptr;
        if ((*s)(&sum_pd, engine, attr, dst_md, n, scales, src_mds)
                == success) {
            pd.reset(sum_pd);
            CHECK(safe_ptr_assign(
                    *sum_pd_iface, new primitive_desc_iface_t(pd, engine)));
            return status::success;
        }
    }
    return unimplemented;
}

} // namespace impl
} // namespace dnnl

status_t dnnl_sum_primitive_desc_create(primitive_desc_iface_t **sum_pd_iface,
        engine_t *engine, const memory_desc_t *dst_md, int n,
        const float *scales, const memory_desc_t *const *src_mds,
        const primitive_attr_t *attr) {
    return sum_primitive_desc_create(
            sum_pd_iface, dst_md, n, scales, src_mds, attr, engine);
}
