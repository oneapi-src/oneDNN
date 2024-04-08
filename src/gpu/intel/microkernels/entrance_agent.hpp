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

#ifndef GPU_MICROKERNELS_ENTRANCE_AGENT_HPP
#define GPU_MICROKERNELS_ENTRANCE_AGENT_HPP

#include "package.hpp"

// The entrance agent is a stateless class that analyzes an incoming package from the microkernel provider,
//   deducing information from the raw microkernel binary.

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace micro {

class EntranceAgent {
public:
    enum class Status {
        Success,
        UncertainClobbers,
        UnsupportedHW,
    };

    static Status scan(Package &package);
};

} /* namespace micro */
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
