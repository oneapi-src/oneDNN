/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_STRUCTURAL_ANALYSIS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_STRUCTURAL_ANALYSIS_HPP

#include <memory>
#include <vector>
#include "passlet.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {
struct structural_result_t {
    using typed_addresser_t
            = typed_passlet<structural_result_t>::typed_addresser_t;
    std::weak_ptr<const stmt_base_t> parent_;
    std::weak_ptr<const stmt_base_t> cur_node_;

    structural_result_t() = default;

    structural_result_t(const stmt_c &parent, const stmt_c &cur_node)
        : parent_(parent.impl), cur_node_(cur_node.impl) {}

    bool is_parent_of(const structural_result_t &other,
            const typed_addresser_t &addresser, bool allow_across_for,
            bool allow_across_if,
            const structural_result_t **out_second_level_parent
            = nullptr) const;

    stmt get_parent_node() const;

    void reset_parent_node(const stmt_c &parent);

    const stmt_base_t *get_raw_parent() const;
    const stmt_base_t *get_raw_cur_node() const;

    const stmt_base_t *find_shared_parent(const structural_result_t &other,
            const typed_addresser_t &addresser, bool allow_across_for,
            bool allow_across_if) const;
};

/**
 * The passlet to analyze the parent stmt of each stmt
 * */
struct structural_analysis_t : public typed_passlet<structural_result_t> {
    using typed_addresser_t
            = typed_passlet<structural_result_t>::typed_addresser_t;
    std::vector<const stmt_base_t *> cur_parent_ {nullptr};
    structural_analysis_t(const typed_addresser_t &stmt_result_func)
        : typed_passlet<structural_result_t>(nullptr, stmt_result_func) {}
    void view(const stmt_c &v, pass_phase phase);
};
} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
