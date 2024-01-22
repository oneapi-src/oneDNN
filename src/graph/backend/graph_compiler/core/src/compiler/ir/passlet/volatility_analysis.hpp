/*******************************************************************************
 * Copyright 2022-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_VOLATILITY_ANALYSIS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_VOLATILITY_ANALYSIS_HPP

#include <vector>
#include "passlet.hpp"
#include <compiler/ir/attr_keys.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {

struct volatility_result_t {
    enum state_t {
        UNDEF,
        YES,
        NO,
    };
    state_t is_volatile_ = UNDEF;
};

inline bool is_tensor_read_only(const expr_base *s) {
    return (s->node_type_ == sc_expr_type::tensor)
            && any_map_t::fetch_or_else(
                    s->attr_.get(), attr_keys::read_only_tensor, false);
}

inline bool non_volatile_expr(const expr_base *s) {
    switch (s->node_type_) {
        case sc_expr_type::var:
        case sc_expr_type::cast:
        case sc_expr_type::select:
        case sc_expr_type::constant:
        case sc_expr_type::tensorptr:
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
                case intrin_type::log:
                case intrin_type::erf:
                case intrin_type::sqrt:
                case intrin_type::rsqrt:
                case intrin_type::reduce_add:
                case intrin_type::reduce_mul:
                case intrin_type::reduce_max:
                case intrin_type::reduce_min:
                case intrin_type::fnmadd:
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
                case intrin_type::permutexvar:
                case intrin_type::insert:
                case intrin_type::extract:
                case intrin_type::isnan:
                case intrin_type::get_group_thread_id:
                case intrin_type::get_group_id:
                case intrin_type::saturated_cast:
                case intrin_type::round_and_cast:
                case intrin_type::shl:
                case intrin_type::shr:
                case intrin_type::constant_load: return true; break;
                default: break;
            }
            return false;
            break;
        }
        case sc_expr_type::low_level_intrin: {
            auto intrin = static_cast<const low_level_intrin_node *>(s);
            switch (intrin->kind_) {
                case low_level_intrin_kind::x86_general:
                    switch (intrin->type_) {
                        case x86_intrin_type::avx_broadcast_idx: {
                            return is_tensor_read_only(intrin->args_[0].get());
                        } break;
                        case x86_intrin_type::avx_mask_cast:
                        case x86_intrin_type::avx_compare: return true; break;
                        default: break;
                    }
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

/**
 * The passlet to analyze whether an SSA value is related to any "side-effect".
 * If an SSA value produces/is affected by side-effects (it usually means
 * reads/writes memory, I/O, etc.), it will be marked "YES".
 * @param check_loop_invarient whether to consider loops as "side-effect"
 * @param stmt_result_func the addresser for stmt->volatility_result_t
 * */
struct volatility_analysis_t : public typed_passlet<volatility_result_t> {
    bool check_loop_invarient_;
    using typed_addresser_t
            = typed_passlet<volatility_result_t>::typed_addresser_t;
    // if there is a dependency loop, ususally it means the var depends on a
    // loop. Need to revisit again
    std::vector<const define_node_t *> to_revisit_;
    volatility_analysis_t(bool check_loop_invarient,
            const typed_addresser_t &stmt_result_func);
    void view(const define_c &v, pass_phase phase);
    void view(const func_c &v, pass_phase phase);
};
} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
