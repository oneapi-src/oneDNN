/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
# Copyright 2025 Arm Ltd. and affiliates
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
#ifndef CPU_AARCH64_CAPI_CAPI_HPP
#define CPU_AARCH64_CAPI_CAPI_HPP

#include <memory>

#include "common/primitive_attr.hpp"

#include "oneapi/dnnl/dnnl_ukernel.h"
#include "oneapi/dnnl/dnnl_ukernel_types.h"

#include "cpu/aarch64/matmul/brgemm_matmul_utils.hpp"
#include "cpu/aarch64/matmul/brgemm_matmul_copy_utils.hpp"

#ifdef DNNL_EXPERIMENTAL_UKERNEL

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace capi {

using pack_type_t = dnnl_pack_type_t;
namespace pack_type {
    const pack_type_t undef = dnnl_pack_type_undef;
    const pack_type_t no_trans = dnnl_pack_type_no_trans;
    const pack_type_t trans = dnnl_pack_type_trans;
    const pack_type_t pack32 = dnnl_pack_type_pack32;
} // namespace pack_type

using attr_params_t = dnnl_ukernel_attr_params;

} // namespace capi
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

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

struct dnnl_transform : public dnnl::impl::c_compatible {
    // Ctor that follows a call to initialize matmul conf struct.
    dnnl_transform(dnnl::impl::dim_t K, dnnl::impl::dim_t N,
            dnnl_pack_type_t in_pack_type,
            dnnl::impl::dim_t in_ld, dnnl::impl::dim_t out_ld,
            dnnl::impl::data_type_t in_dt, dnnl::impl::data_type_t out_dt);

    // Generates a transform kernel.
    dnnl::impl::status_t generate();

    // Executes a transform kernel.
    dnnl::impl::status_t execute(const void *src, void *dst) const;

private:
    // User's inputs.
    dnnl::impl::dim_t K_, N_;
    dnnl::impl::dim_t in_ld_, out_ld_;
    dnnl::impl::data_type_t in_dt_, out_dt_;
    // Save `strides_` for `execute` to get proper source offset.
    dnnl::impl::dims_t strides_;

    // A transform kernel.
    // Note: though it's a generic class for any kind of transformation, so far
    // it's only matmul's copy_B.
    dnnl::impl::cpu::aarch64::matmul::brgemm_matmul_conf_t bmc_;
    // `unique_ptr` is required by API that generates a kernel.
    std::unique_ptr<dnnl::impl::cpu::aarch64::matmul::jit_brgemm_matmul_copy_b_t>
            pack_B_kernel_;

    // Creates a `verbose_info_` string once during `generate()` call, and calls
    // it during execute(). This is done to avoid string re-creation.
    dnnl::impl::status_t create_verbose_info();
    std::string verbose_info_;
};

#endif

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
