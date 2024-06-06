/*******************************************************************************
 * Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_BINARY_FORMAT_HPP
#define GPU_INTEL_JIT_BINARY_FORMAT_HPP

#include "common/engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

status_t gpu_supports_binary_format(bool *ok, impl::engine_t *engine);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_JIT_BINARY_FORMAT_HPP
