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

#include "volatility_analysis.hpp"
#include <functional>
#include <utility>
#include <vector>
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/ssa_data.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {
static volatility_result_t::state_t merge_state(
        volatility_result_t::state_t a, volatility_result_t::state_t b) {
    if (a == volatility_result_t::YES || b == volatility_result_t::YES) {
        return volatility_result_t::YES;
    }
    if (a == volatility_result_t::NO && b == volatility_result_t::NO) {
        return volatility_result_t::NO;
    }
    return volatility_result_t::UNDEF;
}

static bool is_tensor_read_only(const expr_base *s) {
    return (s->node_type_ == sc_expr_type::tensor)
            && any_map_t::fetch_or_else(
                    s->attr_.get(), attr_keys::read_only_tensor, false);
}

static bool expr_can_hoist(const expr_base *s) {
    return non_volatile_expr(s) || is_pure_func_call(s->node_ptr_from_this())
            || is_tensor_read_only(s);
}

void volatility_analysis_t::view(const define_c &v, pass_phase phase) {
    if (phase != pass_phase::POST_VISIT) { return; }
    auto def = v.get();
    auto ths = this;
    if (def->var_.isa<var>() && def->init_.defined()
            && def->var_->ssa_data_->is_local()) {
        volatility_result_t::state_t is_volatile = volatility_result_t::NO;
        bool is_loop_phi = check_loop_invarient_ && def->init_.isa<ssa_phi>()
                && def->init_.static_as<ssa_phi>()->is_loop_phi_;
        if (is_loop_phi || !expr_can_hoist(def->init_.get())) {
            is_volatile = volatility_result_t::YES;
        } else {
            auto callback = [ths, &is_volatile](array_ref<expr> v) {
                for (auto &val : v) {
                    if (is_volatile == volatility_result_t::YES) { break; }
                    if (val->ssa_data_->is_global_) {
                        is_volatile = volatility_result_t::YES;
                        break;
                    }
                    if (val->ssa_data_->is_param_) { continue; }
                    if (val.isa<constant>() || val.isa<tensor>()) { continue; }
                    assert(val->ssa_data_->has_owner());
                    auto owner = val->ssa_data_->get_owner();
                    if (owner.isa<for_loop>()) {
                        if (ths->check_loop_invarient_) {
                            is_volatile = volatility_result_t::YES;
                        }
                    } else {
                        auto parent_is_volatile
                                = ths->get_result(
                                             val->ssa_data_->get_owner().get())
                                          ->is_volatile_;
                        is_volatile
                                = merge_state(is_volatile, parent_is_volatile);
                    }
                }
            };
            if (def->init_.isa<var>() && def->init_->ssa_data_->is_global_) {
                callback({def->init_->node_ptr_from_this()});
            } else {
                get_direct_dependency_of_expr(
                        def->init_->node_ptr_from_this(), callback);
            }
        }
        if (is_volatile == volatility_result_t::UNDEF) {
            // if this var depends on an undefined var, it usually
            // mean it depends on a loop phi.
            if (ths->check_loop_invarient_) {
                // a loop var is not loop invarient
                is_volatile = volatility_result_t::YES;
            } else {
                // We need to remember these vars and try again
                ths->to_revisit_.push_back(def);
            }
        }
        ths->get_result(def)->is_volatile_ = is_volatile;
    } else {
        ths->get_result(def)->is_volatile_ = volatility_result_t::YES;
    }
}

void volatility_analysis_t::view(const func_c &v, pass_phase phase) {
    if (phase != pass_phase::POST_VISIT) { return; }
    if (!to_revisit_.empty()) {
        // finalize the results
        std::vector<const define_node_t *> to_revisit = std::move(to_revisit_);
        for (auto v : to_revisit) {
            view(v->node_ptr_from_this().static_as<define_c>(),
                    pass_phase::POST_VISIT);
            auto ret = get_result(v);
            // if a loop var's state is still undef, then it will not be
            // volatile
            if (ret->is_volatile_ == volatility_result_t::UNDEF) {
                ret->is_volatile_ = volatility_result_t::NO;
            }
        }
    }
}

volatility_analysis_t::volatility_analysis_t(
        bool check_loop_invarient, const typed_addresser_t &stmt_result_func)
    : typed_passlet {nullptr, stmt_result_func}
    , check_loop_invarient_(check_loop_invarient) {}

} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
