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
#include "scope_flatten.hpp"
#include <utility>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass/ir_copy.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static void do_scope_flatten(
        const std::vector<stmt> &seq, std::vector<stmt> &ret, int stmt_index) {
    if (seq[stmt_index].isa<stmts>()) {
        for (auto &v : seq[stmt_index].static_as<stmts>()->seq_) {
            ret.emplace_back(std::move(v));
        }
    } else {
        ret.emplace_back(seq[stmt_index]);
    }
}
void scope_flatten(std::vector<stmt> &seq, int stmt_index) {
    std::vector<stmt> ret;
    ret.reserve(seq.size());
    if (stmt_index < 0) {
        for (unsigned i = 0; i < seq.size(); i++) {
            do_scope_flatten(seq, ret, i);
        }
    } else {
        assert(seq.size() > unsigned(stmt_index));
        for (int i = 0; i < stmt_index; i++) {
            ret.emplace_back(seq[i]);
        }
        do_scope_flatten(seq, ret, stmt_index);
        for (unsigned i = stmt_index + 1; i < seq.size(); i++) {
            ret.emplace_back(seq[i]);
        }
    }
    seq = std::move(ret);
}

void scope_flatten(const stmt &seq, int stmt_index) {
    if (seq.isa<stmts>()) {
        scope_flatten(seq.static_as<stmts>()->seq_, stmt_index);
    } else {
        SC_WARN << "Flattening requires a stmts node";
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
