/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2024 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_BASE_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_BASE_HPP

#include "gpu/gpu_matmul_pd.hpp"

#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/nvidia/cudnn_matmul_executor.hpp"
#include "gpu/nvidia/cudnn_matmul_impl.hpp"
#include "gpu/nvidia/cudnn_matmul_lt_impl.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_base_t : public gpu::primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;
        virtual status_t init(impl::engine_t *engine) = 0;
    };
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
