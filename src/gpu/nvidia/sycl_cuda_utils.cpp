/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

bool compare_cuda_devices(
        const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_cuda_handle = compat::get_native<CUdevice>(lhs);
    auto rhs_cuda_handle = compat::get_native<CUdevice>(rhs);
    return lhs_cuda_handle == rhs_cuda_handle;
}

bool attr_post_ops_ok(const primitive_attr_t *attr) {
    using namespace primitive_kind;
    const auto &po = attr->post_ops_;
    const int eltwise_idx = po.find(eltwise);
    if (eltwise_idx != -1) {
        const auto &e = po.entry_[eltwise_idx].eltwise;

        using namespace alg_kind;
        const bool ok = utils::one_of(e.alg, eltwise_relu, eltwise_tanh,
                eltwise_elu, eltwise_logistic);
        if (!ok) return false;

        // No alpha or beta extension is supported.
        if (e.alpha != 0) return false;

        // Only a single eltwise post-op is supported.
        if (po.find(eltwise, eltwise_idx + 1) != -1) return false;
    }

    switch (po.len()) {
        case 0: return true;
        case 1: return po.contain(sum, 0) || po.contain(eltwise, 0);
        case 2: return po.contain(sum, 0) && po.contain(eltwise, 1);
        default: return false;
    }
}

bool has_bf16_support(const ::sycl::device &dev) {
    // This function checks compute capabilities of the given device.
    // BF16 is supported starting with compute capabilities 8.0.
    auto cuda_dev = compat::get_native<CUdevice>(dev);
    cudaDeviceProp prop {};
    cudaGetDeviceProperties(&prop, cuda_dev);
    return prop.major >= 8;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
