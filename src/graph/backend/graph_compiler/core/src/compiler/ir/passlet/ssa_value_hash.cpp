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

#include "ssa_value_hash.hpp"
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/ssa_data.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <util/hash_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {

static size_t hash_const(expr_base *v) {
    auto &values = static_cast<constant_node *>(v)->value_;
    auto type = get_etype_category_nothrow(v->dtype_.type_code_);
    size_t ret = 7;
    for (auto val : values) {
        switch (type) {
            case CATE_FLOAT:
                union {
                    float vf;
                    uint32_t vi;
                } caster;
                caster.vf = val.f32;
                hash_combine(ret, caster.vi);
                break;
            case CATE_OTHER:
                COMPILE_ASSERT(v->dtype_ == datatypes::pointer,
                        "Expecting pointer type for hashing constants");
            case CATE_INT:
            case CATE_UINT: hash_combine(ret, val.u64); break;
        }
    }
    return ret;
}

// if original value of v is 0, let v = hash. Else, call hash_combine(v,hash).
// this is for constant and copy propagation, to let contant value/copy of SSA
// value to propagate to their users without altering the hash value
// if is_comm_asso, use Add(+) rather than hash_combine for unordered hashing
static void propagated_hash_combine(bool is_comm_asso, size_t &v, size_t hash) {
    if (v == 0) {
        v = hash;
    } else {
        if (is_comm_asso) {
            v = v + hash;
        } else {
            hash_combine(v, hash);
        }
    }
}

static expr_base *get_single_value_ssa(const expr &v) {
    if (v.isa<ssa_phi>()) {
        auto p = v.static_as<ssa_phi>();
        if (p->values_.size() == 1UL) { return p->values_[0].get(); }
    }
    return nullptr;
}

void ssa_value_hash_t::view(const define_c &v, pass_phase phase) {
    if (phase != pass_phase::POST_VISIT) { return; }
    auto def = v.get();
    auto ths = this;
    size_t result = 0;
    // if the operator is commutative_and_associative. If true, use unordered
    // hashing. hash("a+b") == hash("b+a")
    bool is_comm_asso = false;
    if (def->var_.isa<var>() && def->init_.defined()
            && def->var_->ssa_data_->is_local()) {
        auto callback = [ths, &result, &is_comm_asso](array_ref<expr> v) {
            for (auto &val : v) {
                if (val->ssa_data_->is_global_ || val->ssa_data_->is_param_) {
                    // for global var/ parameter vars, hash the var addr
                    propagated_hash_combine(is_comm_asso, result,
                            reinterpret_cast<uintptr_t>(val.get()));
                    continue;
                }
                if (val.isa<constant>()) {
                    propagated_hash_combine(
                            is_comm_asso, result, hash_const(val.get()));
                    continue;
                }
                assert(val->ssa_data_->has_owner());
                auto owner = val->ssa_data_->get_owner();
                if (owner.isa<for_loop>()) {
                    propagated_hash_combine(is_comm_asso, result,
                            reinterpret_cast<uintptr_t>(val.get()));
                    continue;
                }
                size_t parent_hash = *ths->get_result(owner.get());
                propagated_hash_combine(is_comm_asso, result, parent_hash);
            }
        };
        if (def->init_.isa<var>() || def->init_.isa<constant>()) {
            // constant and copy propagation, we use result=0
            callback(&def->init_);
        } else if (auto single_ssa = get_single_value_ssa(def->init_)) {
            callback({single_ssa->node_ptr_from_this()});
        } else {
            result = static_cast<size_t>(def->init_->node_type_);
            is_comm_asso = constant_folding::is_op_commutative_and_associative(
                    def->init_);
            get_direct_dependency_of_expr(def->init_, callback);
        }
    } else if (def->var_.isa<tensor>()) {
        result = reinterpret_cast<uintptr_t>(def->var_.get());
    }
    *get_result(def) = result;
}

} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
