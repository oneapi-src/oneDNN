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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_PARALLEL_WORKLOAD_DISPATCH_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_PARALLEL_WORKLOAD_DISPATCH_HPP

#include "../function_pass.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// measured by a f32 FMA:
// for(i, 0, 2^16, 1) {
//    c[i] = c[i] + a[i] * b[i];
// }
// workload threshold = sigma(shape * sizeof(dtype) * read/write weight)
constexpr size_t memory_access_threshold_per_thread = 37440UL;
/**
 * According to workload marked in loop to calculate total
 * workloads(calculation/memory attachment) and decide whether to mark
 * `PARALLEL` in loops
 * */
class parallel_workload_dispatcher_t : public function_pass_t {
public:
    bool record_workload_;
    std::unordered_map<stmt_c, size_t> stmt_workload_map_;
    parallel_workload_dispatcher_t(bool record_workload = false)
        : record_workload_(record_workload) {}
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c f);
    SC_DECL_PASS_INFO_FUNC();
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
