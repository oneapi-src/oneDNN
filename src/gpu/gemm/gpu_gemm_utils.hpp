/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_GEMM_GPU_GEMM_UTILS_HPP
#define GPU_GEMM_GPU_GEMM_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/primitive_attr.hpp"
#include "gpu/gemm/gpu_gemm.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace gemm_utils {

inline const gpu_gemm_t *gpu_gemm(const std::shared_ptr<primitive_t> &p) {
    return utils::downcast<gpu_gemm_t *>(p.get());
}

} // namespace gemm_utils
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
