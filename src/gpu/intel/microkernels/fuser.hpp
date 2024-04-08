/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_MICROKERNELS_FUSER_HPP
#define GPU_MICROKERNELS_FUSER_HPP

#include <cstdint>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace micro {

// Markers for patch sections.
static constexpr uint32_t sigilStart = 0xCAFEFADE;
static constexpr uint32_t sigilEnd = 0xFADECAFE;
static constexpr const char *sigilBinary = "@_u_@";

// Fuse the microkernel machine code into the program binary of a compiled host kernel.
void fuseMicrokernel(std::vector<uint8_t> &binary,
        const std::vector<uint8_t> &microkernel, int id = 0);

// Fusing microkernels that were embedded directly in source code.
void fuseMicrokernels(std::vector<uint8_t> &binary, const char *source);
bool hasMicrokernels(const char *source);

} /* namespace micro */
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
