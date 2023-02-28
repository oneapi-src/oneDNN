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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_DEPENDENCY_ANALYZER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_DEPENDENCY_ANALYZER_HPP

#include <memory>
#include "../function_pass.hpp"
#include <util/weakptr_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace dependency_analysis {
// if a tensor is directly accessed (take pointer of a tensor)
constexpr const char *attr_directly_accessed = "ir_analysis.directly_accessed";
// std::weak_ptr cannot be hashed. Use a trick to bypass it
using stmt_weak_set = utils::weakptr_hashset_t<stmt_base_t>;
struct dependency_t {
    stmt_weak_set depends_on_;
    stmt_weak_set depended_by_;
    std::weak_ptr<stmt_base_t> indexing_owner_;
    dependency_t() = default;
};

dependency_t &get_dep_info(const node_base *s);
} // namespace dependency_analysis

/**
 * Mark the dependency graph. Will attach a dependency_analyzer_t::dependency_t
 * on the temp_data
 * */
class dependency_analyzer_t : public function_pass_t {
public:
    func_c operator()(func_c f) override;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
