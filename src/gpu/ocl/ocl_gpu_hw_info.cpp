/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "gpu/ocl/ocl_gpu_hw_info.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/utils/ngen_type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

void init_gpu_hw_info(cl_device_id device, cl_context context,
        compute::gpu_arch_t &gpu_arch, int &stepping_id) {
    using namespace ngen;

    HW hw = HW::Unknown;
    jit::jit_generator<HW::Unknown>::detectHWInfo(
            context, device, hw, stepping_id);

    gpu_arch = jit::convert_ngen_arch_to_dnnl(hw);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
