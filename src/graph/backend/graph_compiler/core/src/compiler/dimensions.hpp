/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_DIMENSIONS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_DIMENSIONS_HPP
#include <assert.h>
#include <stdint.h>
#include <vector>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
using sc_dim = int64_t;
using sc_dims = std::vector<sc_dim>;
namespace dimensions {
constexpr sc_dim dynamic_any = -1;
}

inline uint64_t dim2unsigned(sc_dim v) {
    assert(v >= 0);
    return v;
}

inline bool is_dynamic_dim(sc_dim v) {
    return v < 0;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
