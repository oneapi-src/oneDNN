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

#include "common/utils.hpp"

#include "cpu/x64/ukernel/attr_params.hpp"

#ifdef DNNL_EXPERIMENTAL_UKERNEL

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::ukernel;

status_t attr_params_t::set_post_ops_args(const void **post_ops_args) {
    post_ops_args_ = post_ops_args;
    return status::success;
}

status_t attr_params_t::set_scales(const void *scales, int arg) {
    switch (arg) {
        case DNNL_ARG_SRC: a_scales_ = scales; break;
        case DNNL_ARG_WEIGHTS: b_scales_ = scales; break;
        case DNNL_ARG_DST: d_scales_ = scales; break;
        default: assert(!"unsupported arg");
    }
    return status::success;
}

const void *attr_params_t::get_scales(int arg) const {
    switch (arg) {
        case DNNL_ARG_SRC: return a_scales_;
        case DNNL_ARG_WEIGHTS: return b_scales_;
        case DNNL_ARG_DST: return d_scales_;
        default: assert(!"unsupported arg");
    }
    return nullptr;
}

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ukernel {

status_t dnnl_ukernel_attr_params_create(attr_params_t **attr_params) {
    *attr_params = new attr_params_t();
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_post_ops_args(
        attr_params_t *attr_params, const void **post_ops_args) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_post_ops_args(post_ops_args));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_A_scales(
        attr_params_t *attr_params, const void *a_scales) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_scales(a_scales, DNNL_ARG_SRC));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_B_scales(
        attr_params_t *attr_params, const void *b_scales) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_scales(b_scales, DNNL_ARG_WEIGHTS));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_D_scales(
        attr_params_t *attr_params, const void *d_scales) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_scales(d_scales, DNNL_ARG_DST));
    return status::success;
}

status_t dnnl_ukernel_attr_params_destroy(attr_params_t *attr_params) {
    delete attr_params;
    return status::success;
}

} // namespace ukernel
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
