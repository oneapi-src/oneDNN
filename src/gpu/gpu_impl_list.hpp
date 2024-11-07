/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_GPU_IMPL_LIST_HPP
#define GPU_GPU_IMPL_LIST_HPP

#include <map>
#include <vector>

#include "common/engine.hpp"
#include "common/impl_list_item.hpp"
#include "common/impl_registration.hpp"
#include "common/sdpa_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

// There is a set of macros to instantiate implementations for different
// vendors and kernel languages to enable using a single implementation list
// (exception: implementation lists for concat, sum and reorder).
//
// oneDNN currently supports four GPU vendors:
// - INTEL
// - NVIDIA
// - AMD
// - GENERIC (standalone or in a combination with the other vendors)
//
// The macros for INTEL, NVIDIA and AMD vendors assume that all implementations
// within a single vendor can be enabled at once.
//
// The macros for the GENERIC vendor can be either truly generic or
// runtime specific:
// - GENERIC: truly generic implementation that is not tied to any vendor
//            and runtime, e.g. an implementation of the concat primitive
//            based on reorders.
// - GENERIC_SYCL: SYCL generic implementations (written in generic SYCL).
//
// The concat, sum and reorder primitives require specialized versions of the
// macros because their `pd_t::create` functions have unique signatures.

// Conditional macros for different vendors.
#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#define DNNL_GPU_INTEL_ONLY(...) __VA_ARGS__
#else
#define DNNL_GPU_INTEL_ONLY(...)
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#define DNNL_GPU_NVIDIA_ONLY(...) __VA_ARGS__
#else
#define DNNL_GPU_NVIDIA_ONLY(...)
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
#define DNNL_GPU_AMD_ONLY(...) __VA_ARGS__
#else
#define DNNL_GPU_AMD_ONLY(...)
#endif

#if defined(DNNL_WITH_SYCL) \
        && ((DNNL_GPU_VENDOR == DNNL_VENDOR_GENERIC) \
                || (DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA) \
                || (DNNL_AMD_ENABLE_SYCL_KERNELS == 1))
#define DNNL_GPU_GENERIC_SYCL_ONLY(...) __VA_ARGS__
#define GENERIC_SYCL_KERNELS_ENABLED
#else
#define DNNL_GPU_GENERIC_SYCL_ONLY(...)
#endif

// Primary instance macro for the GPU primitives.
#define GPU_INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::type_deduction_helper_t<__VA_ARGS__::pd_t>()),

// Specializations of the primary instance macro for concat, sum and reorder
// primitives.
#define GPU_CONCAT_INSTANCE(...) \
    impl_list_item_t(impl_list_item_t::concat_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>()),
#define GPU_SUM_INSTANCE(...) \
    impl_list_item_t(impl_list_item_t::sum_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>()),
#define GPU_REORDER_INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::reorder_type_deduction_helper_t<__VA_ARGS__>()),

// Vendor specific instance macros.
#define GPU_INSTANCE_INTEL(...) DNNL_GPU_INTEL_ONLY(GPU_INSTANCE(__VA_ARGS__))
#define GPU_INSTANCE_NVIDIA(...) DNNL_GPU_NVIDIA_ONLY(GPU_INSTANCE(__VA_ARGS__))
#define GPU_INSTANCE_AMD(...) DNNL_GPU_AMD_ONLY(GPU_INSTANCE(__VA_ARGS__))
#define GPU_INSTANCE_GENERIC_SYCL(...) \
    DNNL_GPU_GENERIC_SYCL_ONLY(GPU_INSTANCE(__VA_ARGS__))
#define GPU_INSTANCE_GENERIC(...) GPU_INSTANCE(__VA_ARGS__)

// Specializations of the vendor specific instance macros for concat, sum
// and reorder primitives.
#define GPU_CONCAT_INSTANCE_INTEL(...) \
    DNNL_GPU_INTEL_ONLY(GPU_CONCAT_INSTANCE(__VA_ARGS__))
#define GPU_CONCAT_INSTANCE_NVIDIA(...) \
    DNNL_GPU_NVIDIA_ONLY(GPU_CONCAT_INSTANCE(__VA_ARGS__))
#define GPU_CONCAT_INSTANCE_AMD(...) \
    DNNL_GPU_AMD_ONLY(GPU_CONCAT_INSTANCE(__VA_ARGS__))
#define GPU_CONCAT_INSTANCE_GENERIC_SYCL(...) \
    DNNL_GPU_GENERIC_SYCL_ONLY(GPU_CONCAT_INSTANCE(__VA_ARGS__))
#define GPU_CONCAT_INSTANCE_GENERIC(...) GPU_CONCAT_INSTANCE(__VA_ARGS__)

#define GPU_SUM_INSTANCE_INTEL(...) \
    DNNL_GPU_INTEL_ONLY(GPU_SUM_INSTANCE(__VA_ARGS__))
#define GPU_SUM_INSTANCE_NVIDIA(...) \
    DNNL_GPU_NVIDIA_ONLY(GPU_SUM_INSTANCE(__VA_ARGS__))
#define GPU_SUM_INSTANCE_AMD(...) \
    DNNL_GPU_AMD_ONLY(GPU_SUM_INSTANCE(__VA_ARGS__))
#define GPU_SUM_INSTANCE_GENERIC_SYCL(...) \
    DNNL_GPU_GENERIC_SYCL_ONLY(GPU_SUM_INSTANCE(__VA_ARGS__))
#define GPU_SUM_INSTANCE_GENERIC(...) GPU_SUM_INSTANCE(__VA_ARGS__)

#define GPU_REORDER_INSTANCE_INTEL(...) \
    DNNL_GPU_INTEL_ONLY(GPU_REORDER_INSTANCE(__VA_ARGS__))
#define GPU_REORDER_INSTANCE_NVIDIA(...) \
    DNNL_GPU_NVIDIA_ONLY(GPU_REORDER_INSTANCE(__VA_ARGS__))
#define GPU_REORDER_INSTANCE_AMD(...) \
    DNNL_GPU_AMD_ONLY(GPU_REORDER_INSTANCE(__VA_ARGS__))
#define GPU_REORDER_INSTANCE_GENERIC_SYCL(...) \
    DNNL_GPU_GENERIC_SYCL_ONLY(GPU_REORDER_INSTANCE(__VA_ARGS__))
#define GPU_REORDER_INSTANCE_GENERIC(...) GPU_REORDER_INSTANCE(__VA_ARGS__)

// Instance macros that are enabled only in the DEV_MODE.
#ifdef DNNL_DEV_MODE
#define GPU_INSTANCE_INTEL_DEVMODE(...) \
    DNNL_GPU_INTEL_ONLY(GPU_INSTANCE(__VA_ARGS__))
#else
#define GPU_INSTANCE_INTEL_DEVMODE(...)
#endif

// Instance macros that are enabled only with DNNL_EXPERIMENTAL.
#ifdef DNNL_EXPERIMENTAL
#define GPU_INSTANCE_INTEL_EXPERIMENTAL(...) \
    DNNL_GPU_INTEL_ONLY(GPU_INSTANCE(__VA_ARGS__))
#else
#define GPU_INSTANCE_INTEL_EXPERIMENTAL(...)
#endif

// Instance macros that are enabled only when REF is disabled
#ifdef DNNL_DISABLE_GPU_REF_KERNELS
#define GPU_INSTANCE_INTEL_REF(...)
#else
#define GPU_INSTANCE_INTEL_REF(...) \
    DNNL_GPU_INTEL_ONLY(GPU_INSTANCE(__VA_ARGS__))
#endif

#define DECLARE_IMPL_LIST(kind) \
    const impl_list_item_t *get_##kind##_impl_list(const kind##_desc_t *desc);

DECLARE_IMPL_LIST(batch_normalization);
DECLARE_IMPL_LIST(binary);
DECLARE_IMPL_LIST(convolution);
DECLARE_IMPL_LIST(deconvolution);
DECLARE_IMPL_LIST(eltwise);
DECLARE_IMPL_LIST(gemm);
DECLARE_IMPL_LIST(group_normalization);
DECLARE_IMPL_LIST(inner_product);
DECLARE_IMPL_LIST(layer_normalization);
DECLARE_IMPL_LIST(lrn);
DECLARE_IMPL_LIST(matmul);
DECLARE_IMPL_LIST(pooling);
DECLARE_IMPL_LIST(prelu);
DECLARE_IMPL_LIST(reduction);
DECLARE_IMPL_LIST(resampling);
DECLARE_IMPL_LIST(rnn);
DECLARE_IMPL_LIST(sdpa);
DECLARE_IMPL_LIST(shuffle);
DECLARE_IMPL_LIST(softmax);
DECLARE_IMPL_LIST(zero_pad);

#undef DECLARE_IMPL_LIST

const impl_list_item_t *get_concat_impl_list();
const impl_list_item_t *get_sum_impl_list();
const impl_list_item_t *get_reorder_impl_list(
        const memory_desc_t *, const memory_desc_t *);

class gpu_impl_list_t {
public:
    static const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc);
    static const impl_list_item_t *get_concat_implementation_list();
    static const impl_list_item_t *get_sum_implementation_list();
    static const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *, const memory_desc_t *);
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_GPU_IMPL_LIST_HPP
