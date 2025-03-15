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
#include "cpu/x64/ukernel/attr_params.hpp"
#endif

#ifdef DNNL_EXPERIMENTAL_UKERNEL

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::ukernel;

status_t dnnl_ukernel_attr_params_create(attr_params_t **attr_params) {
#if DNNL_X64
    return x64::ukernel::dnnl_ukernel_attr_params_create(attr_params);
#endif
    return status::unimplemented;
}

status_t dnnl_ukernel_attr_params_set_post_ops_args(
        attr_params_t *attr_params, const void **post_ops_args) {
#if DNNL_X64
    return x64::ukernel::dnnl_ukernel_attr_params_set_post_ops_args(
            attr_params, post_ops_args);
#endif
    return status::unimplemented;
}

status_t dnnl_ukernel_attr_params_set_A_scales(
        attr_params_t *attr_params, const void *a_scales) {
#if DNNL_X64
    return x64::ukernel::dnnl_ukernel_attr_params_set_A_scales(
            attr_params, a_scales);
#endif
    return status::unimplemented;
}

status_t dnnl_ukernel_attr_params_set_B_scales(
        attr_params_t *attr_params, const void *b_scales) {
#if DNNL_X64
    return x64::ukernel::dnnl_ukernel_attr_params_set_B_scales(
            attr_params, b_scales);
#endif
    return status::unimplemented;
}

status_t dnnl_ukernel_attr_params_set_D_scales(
        attr_params_t *attr_params, const void *d_scales) {
#if DNNL_X64
    return x64::ukernel::dnnl_ukernel_attr_params_set_D_scales(
            attr_params, d_scales);
#endif
    return status::unimplemented;
}

status_t dnnl_ukernel_attr_params_destroy(attr_params_t *attr_params) {
#if DNNL_X64
    return x64::ukernel::dnnl_ukernel_attr_params_destroy(attr_params);
#endif
    return status::unimplemented;
}

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
