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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_DATA_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_DATA_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph_map.hpp"
#include "tensor_slice.hpp"
#include <compiler/ir/sc_expr.hpp>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using slice_range_list = std::vector<slice_range>;
using slice_range_map = std::unordered_map<int, slice_range_list>;

enum class infer_status_code : int {
    OK = 0, // Successful
    RETRY, // Need retry another anchor
    FAIL, // Could not infer
    UNKNOWN, // Unknown
    END,
};

expr do_cast_and_fold(const expr &);

inline std::vector<expr> get_slice_idx(const slice_range &range) {
    std::vector<expr> ret;
    for (auto &r : range) {
        ret.emplace_back(do_cast_and_fold(r.first));
    }
    return ret;
}

inline std::vector<expr> get_slice_shape(const slice_range &range) {
    std::vector<expr> ret;
    for (auto &r : range) {
        ret.emplace_back(do_cast_and_fold(r.second));
    }
    return ret;
}

inline slice_range gen_slice_by_dims(const sc_dims &dims) {
    slice_range ret;
    for (auto &r : dims) {
        ret.emplace_back(std::make_pair(expr(0), dim2unsigned(r)));
    }
    return ret;
}

inline slice_range gen_slice_by_dims_expr(const std::vector<expr> &dims) {
    slice_range ret;
    for (auto &r : dims) {
        ret.emplace_back(std::make_pair(dim2unsigned(0), r));
    }
    return ret;
}

bool is_reshaped_tensor(const expr &tsr);

expr transform_tsr2stsr_with_range(const expr &tsr, const slice_range &range);

expr transform_tsl2stsr(const tensor_slice &tsl);

expr transform_tsr2tptr_with_range(const expr &tsr, const slice_range &range);

expr transform_tptr2stsr(const expr &tptr);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
