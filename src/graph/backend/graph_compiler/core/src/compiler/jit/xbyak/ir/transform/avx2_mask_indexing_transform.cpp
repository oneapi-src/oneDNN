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

#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/visitor.hpp>
#include <unordered_map>

#include "avx2_mask_indexing_transform.hpp"
#include "util/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class avx2_mask_indexing_transform_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    std::vector<stmt_c> idx_transform_stmt;
    std::unordered_map<expr_c, expr> idx_var_map;

    avx2_mask_indexing_transform_impl_t() = default;

    bool is_base_type_less_than_32bit(const sc_data_type_t &dtype) {
        return utils::get_sizeof_etype(dtype.type_code_) * 8 < 32;
    }

    stmt_c process_scope_indexing(
            const expr &value, const expr &var, const stmt_c &v) {
        auto is_mask_indexing = [&](const expr &inp) {
            return (inp.defined() && inp->node_type_ == sc_expr_type::indexing
                    && inp.dyn_as<indexing_c>()->mask_.defined()
                    && is_base_type_less_than_32bit(inp->dtype_));
        };
        // a = min(b, u16x8(A[i@M])) or A[i@M] = B[i@M]
        if ((value.defined() && value->node_type_ != sc_expr_type::indexing)
                || (is_mask_indexing(var) && is_mask_indexing(value))) {
            assert(idx_transform_stmt.empty());
            auto vv = dispatch(value);
            if (!idx_transform_stmt.empty()) {
                std::vector<stmt_c> cur_list(std::move(idx_transform_stmt));
                if (v.isa<define_c>()) {
                    cur_list.emplace_back(
                            builder::make_var_tensor_def_unattached(
                                    var, v.dyn_as<define_c>()->linkage_, vv));
                } else {
                    cur_list.emplace_back(
                            builder::make_assign_unattached(var, vv));
                }
                idx_var_map.clear();
                idx_transform_stmt.clear();
                return builder::make_stmts_unattached(cur_list);
            }
        }
        return v;
    }

    expr_c visit(tensor_c v) override {
        // avoid dispatch into for loop index dependent tensor
        return v;
    }

    expr_c visit(indexing_c v) override {
        auto vv = ir_visitor_t::visit(v).dyn_as<indexing_c>();
        if (vv.defined() && is_base_type_less_than_32bit(vv->dtype_)
                && vv->mask_.defined()) {
            if (idx_var_map.find(vv) != idx_var_map.end()) {
                return idx_var_map[vv];
            }
            auto nested_idx_var = builder::make_var(
                    vv->dtype_, "indexing_nested" + std::to_string(count_++));
            auto nested_var_define = builder::make_var_tensor_def_unattached(
                    nested_idx_var, linkage::local);
            auto nested_assign
                    = builder::make_assign_unattached(nested_idx_var, vv);
            std::vector<stmt_c> cur_list;
            cur_list.emplace_back(nested_var_define);
            cur_list.emplace_back(nested_assign);
            auto nested_stmts = builder::make_stmts_unattached(cur_list);
            idx_transform_stmt.emplace_back(nested_stmts);
            idx_var_map.insert(std::make_pair(vv, nested_idx_var));
            return nested_idx_var;
        }
        return vv;
    }

    stmt_c visit(define_c v) override {
        // multiple nested expr has indexing
        return process_scope_indexing(v->init_, v->var_, v);
    }

    stmt_c visit(assign_c v) override {
        // multiple nested expr has indexing
        return process_scope_indexing(v->value_, v->var_, v);
    }

private:
    size_t count_ = 0;
};

func_c avx2_mask_indexing_t::operator()(func_c v) {
    // No need for AVX2 legalization when AVX512 is available
    if (target_machine_.cpu_flags_.fAVX512F) { return v; }
    avx2_mask_indexing_transform_impl_t avx2_indexing_transform;
    return avx2_indexing_transform.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
