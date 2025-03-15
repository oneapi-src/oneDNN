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

#ifndef CPU_X64_UKERNEL_ATTR_PARAMS_HPP
#define CPU_X64_UKERNEL_ATTR_PARAMS_HPP

#include "common/nstl.hpp"

#include "cpu/ukernel/c_types_map.hpp"

#ifdef DNNL_EXPERIMENTAL_UKERNEL

struct dnnl_ukernel_attr_params : public dnnl::impl::c_compatible {
    dnnl_ukernel_attr_params() = default;

    dnnl::impl::status_t set_post_ops_args(const void **post_ops_args);
    const void *get_post_ops_args() const { return post_ops_args_; }

    dnnl::impl::status_t set_scales(const void *scales, int arg);
    const void *get_scales(int arg) const;

private:
    const void *post_ops_args_;
    const void *a_scales_;
    const void *b_scales_;
    const void *d_scales_;
};

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ukernel {

status_t dnnl_ukernel_attr_params_create(
        dnnl_ukernel_attr_params **attr_params);

status_t dnnl_ukernel_attr_params_set_post_ops_args(
        dnnl_ukernel_attr_params *attr_params, const void **post_ops_args);

status_t dnnl_ukernel_attr_params_set_A_scales(
        dnnl_ukernel_attr_params *attr_params, const void *a_scales);

status_t dnnl_ukernel_attr_params_set_B_scales(
        dnnl_ukernel_attr_params *attr_params, const void *b_scales);

status_t dnnl_ukernel_attr_params_set_D_scales(
        dnnl_ukernel_attr_params *attr_params, const void *d_scales);

status_t dnnl_ukernel_attr_params_destroy(
        dnnl_ukernel_attr_params *attr_params);

} // namespace ukernel
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
