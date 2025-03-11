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

#include "oneapi/dnnl/dnnl_ukernel.h"

#include "cpu/ukernel/c_types_map.hpp"

#if DNNL_X64
#include "cpu/x64/ukernel/brgemm_api.hpp"
#endif

#ifdef DNNL_EXPERIMENTAL_UKERNEL

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::ukernel;

status_t dnnl_transform_create(transform_t **transform, dim_t K, dim_t N,
        pack_type_t in_pack_type, dim_t in_ld, dim_t out_ld, data_type_t in_dt,
        data_type_t out_dt) {
#if DNNL_X64
    return x64::ukernel::dnnl_transform_create(
            transform, K, N, in_pack_type, in_ld, out_ld, in_dt, out_dt);
#endif
    return status::unimplemented;
}

status_t dnnl_transform_generate(transform_t *transform) {
#if DNNL_X64
    return x64::ukernel::dnnl_transform_generate(transform);
#endif
    return status::unimplemented;
}

status_t dnnl_transform_execute(
        const transform_t *transform, const void *in_ptr, void *out_ptr) {
#if DNNL_X64
    return x64::ukernel::dnnl_transform_execute(transform, in_ptr, out_ptr);
#endif
    return status::unimplemented;
}

status_t dnnl_transform_destroy(transform_t *transform) {
#if DNNL_X64
    return x64::ukernel::dnnl_transform_destroy(transform);
#endif
    return status::unimplemented;
}

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
