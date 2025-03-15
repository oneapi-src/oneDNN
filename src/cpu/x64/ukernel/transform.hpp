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

#ifndef CPU_X64_UKERNEL_TRANSFORM_HPP
#define CPU_X64_UKERNEL_TRANSFORM_HPP

#include <memory>

#include "cpu/ukernel/c_types_map.hpp"

#include "cpu/x64/matmul/brgemm_matmul_copy_utils.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

#ifdef DNNL_EXPERIMENTAL_UKERNEL

struct dnnl_transform : public dnnl::impl::c_compatible {
    // Ctor that follows a call to initialize matmul conf struct.
    dnnl_transform(dnnl::impl::dim_t K, dnnl::impl::dim_t N,
            dnnl::impl::cpu::ukernel::pack_type_t in_pack_type,
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
    dnnl::impl::cpu::x64::matmul::brgemm_matmul_conf_t bmc_;
    // `unique_ptr` is required by API that generates a kernel.
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>
            pack_B_kernel_;

    // Creates a `verbose_info_` string once during `generate()` call, and calls
    // it during execute(). This is done to avoid string re-creation.
    dnnl::impl::status_t create_verbose_info();
    std::string verbose_info_;
};

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ukernel {

status_t dnnl_transform_create(dnnl_transform **transform, dim_t K, dim_t N,
        dnnl::impl::cpu::ukernel::pack_type_t in_pack_type, dim_t in_ld,
        dim_t out_ld, data_type_t in_dt, data_type_t out_dt);

status_t dnnl_transform_generate(dnnl_transform *transform);

status_t dnnl_transform_execute(
        const dnnl_transform *transform, const void *in_ptr, void *out_ptr);

status_t dnnl_transform_destroy(dnnl_transform *transform);

} // namespace ukernel
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
