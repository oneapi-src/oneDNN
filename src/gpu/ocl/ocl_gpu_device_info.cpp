/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "gpu/ocl/ocl_gpu_device_info.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

size_t ocl_gpu_device_info_t::get_llc_cache_size() const {
    // Integrated GPUs share LLC with CPU which is L3 cache on CPU.
    size_t cache_size = cpu::platform::get_per_core_cache_size(3)
            * cpu::platform::get_num_cores();
    return cache_size;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
