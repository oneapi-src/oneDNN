/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_JIT_V2_CONV_IR_BUILDER_HPP
#define GPU_JIT_V2_CONV_IR_BUILDER_HPP

#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/v2/conv/kernel_desc.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

stmt_t build_ir(const kernel_desc_t &desc, const kernel_info_t &kernel_info,
        const grid_context_t &grid_ctx);

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
