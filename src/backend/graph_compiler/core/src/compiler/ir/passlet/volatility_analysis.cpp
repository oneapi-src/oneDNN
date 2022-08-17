/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/ssa_data.hpp>

namespace sc {
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

static bool expr_can_hoist(const expr_base *s) {
    switch (s->node_type_) {
        case sc_expr_type::var:
        case sc_expr_type::cast:
        case sc_expr_type::select:
        case sc_expr_type::constant:
        case sc_expr_type::ssa_phi: return true; break;
        case sc_expr_type::intrin_call: {
            switch (static_cast<const intrin_call_node *>(s)->type_) {
                case intrin_type::min:
                case intrin_type::max:
                case intrin_type::abs:
                case intrin_type::round:
                case intrin_type::floor:
                case intrin_type::ceil:
                case intrin_type::exp:
                case intrin_type::sqrt:
                case intrin_type::rsqrt:
                case intrin_type::reduce_add:
                case intrin_type::reduce_mul:
                case intrin_type::reduce_max:
                case intrin_type::reduce_min:
                case intrin_type::fmadd:
                case intrin_type::unpack_low:
                case intrin_type::unpack_high:
                case intrin_type::shuffle:
                case intrin_type::permute:
                case intrin_type::int_and:
                case intrin_type::int_or:
                case intrin_type::int_xor:
                case intrin_type::reinterpret:
                case intrin_type::broadcast:
                case intrin_type::permutex2var:
                case intrin_type::isnan:
                case intrin_type::saturated_cast:
                case intrin_type::round_and_cast:
                case intrin_type::shl:
                case intrin_type::shr: return true; break;
                default: break;
            }
            return false;
            break;
        }
        default:
            if (dynamic_cast<const binary_node *>(s)) { return true; }
            if (dynamic_cast<const logic_node *>(s)) { return true; }
            if (dynamic_cast<const cmp_node *>(s)) { return true; }
            return false;
            break;
    }
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
                    if (val.isa<constant>()) { continue; }
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
} // namespace sc
