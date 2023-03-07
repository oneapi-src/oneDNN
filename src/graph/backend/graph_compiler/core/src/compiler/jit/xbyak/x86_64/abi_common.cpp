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

#include <compiler/jit/xbyak/x86_64/abi_common.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

std::ostream &operator<<(std::ostream &os, abi_value_kind v) {
    switch (v) {
#define HANDLE_CASE(V) \
    case abi_value_kind::V: os << "abi_value_kind::" #V; break;

        HANDLE_CASE(INTEGER);
        HANDLE_CASE(SSE);
        HANDLE_CASE(SSEUPx15_SSE);

#undef HANDLE_CASE
    }

    return os;
}

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
