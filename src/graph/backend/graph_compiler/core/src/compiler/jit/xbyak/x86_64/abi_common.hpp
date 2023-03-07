/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_ABI_COMMON_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_ABI_COMMON_HPP

#include <ostream>
#include <vector>
#include <compiler/jit/xbyak/x86_64/registers.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

/// See psABI 1.0 section 3.2.3.
///
/// This enumeration defines only the subset of value classes that we
/// currently support.
enum class abi_value_kind {
    INTEGER,
    SSE,

    /// A hack to let us continue to use an enum for this, even though the psABI
    /// wants to treat it as this sequence of 8-bytes: {SSEUP, ..., SSEUP, SSE}.
    /// (Going from highest-order on the left to lowest-order on the right.)
    SSEUPx15_SSE,

    // SSEUP,
    // X87,
    // X87UP,
    // COMPLEX_X87,
    // NO_CLASS,
    // MEMORY,
};

std::ostream &operator<<(std::ostream &, abi_value_kind v);

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
