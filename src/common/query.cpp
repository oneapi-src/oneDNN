/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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
#include "dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"
#if DNNL_VERBOSE_EXTRA
#include "consistency.hpp"
#define AND_(...) SCHKV(args_ok, __VA_ARGS__)
#endif

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;

status_t dnnl_primitive_desc_query(const primitive_desc_t *primitive_desc,
        query_t what, int index, void *result) {
    if (any_null(primitive_desc, result)) return invalid_arguments;

    return primitive_desc->query(what, index, result);
}

const memory_desc_t *dnnl_primitive_desc_query_md(
        const primitive_desc_t *primitive_desc, query_t what, int index) {
    const memory_desc_t *res_md = nullptr;
#if DNNL_VERBOSE_EXTRA
    Consistency args_ok("dnnl_primitive_desc_query_md args:");
    AND_(primitive_desc != nullptr);
    AND_((what & query::some_md) == query::some_md);
    AND_(what != query::some_md);
    AND_(dnnl_primitive_desc_query(primitive_desc, what, index, &res_md)
            == success);
#else
    bool args_ok = true && primitive_desc != nullptr
            && (what & query::some_md) == query::some_md
            && what != query::some_md
            && dnnl_primitive_desc_query(primitive_desc, what, index, &res_md)
                    == success;
#endif
    return args_ok ? res_md : nullptr;
}

int dnnl_primitive_desc_query_s32(
        const primitive_desc_t *primitive_desc, query_t what, int index) {
    int res_s32;
#if DNNL_VERBOSE_EXTRA
    Consistency args_ok("dnnl_primitive_desc_query_s32 args:");
    AND_(primitive_desc != nullptr);
    AND_(one_of(what, query::num_of_inputs_s32, query::num_of_outputs_s32));
    AND_(dnnl_primitive_desc_query(primitive_desc, what, index, &res_s32)
            == success);
    return args_ok ? res_s32 : 0;
#else
    bool args_ok = primitive_desc != nullptr
            && one_of(what, query::num_of_inputs_s32, query::num_of_outputs_s32)
            && dnnl_primitive_desc_query(primitive_desc, what, index, &res_s32)
                    == success;
    return args_ok ? res_s32 : 0;
#endif
}
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
