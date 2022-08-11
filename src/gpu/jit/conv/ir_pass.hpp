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

#ifndef GPU_JIT_CONV_IR_PASS_HPP
#define GPU_JIT_CONV_IR_PASS_HPP

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Transforms dpas to dpasw.
void inject_dpasw(ngen::HW hw, stmt_t &load_mul_stmt, const expr_t &c_buf,
        stmt_t &c_store_stmt, alloc_updater_t &alloc_updater,
        const expr_t &tg_idx0);

// Adds {Atomic} modifier to dpas/dpasw instructions when applicable.
stmt_t inject_atomic(const stmt_t &stmt);

stmt_t inject_external_var_let(const stmt_t &_stmt, ir_context_t &ir_ctx);

// Merges all SLM buffers into a single one.
stmt_t merge_slm_buffers(const stmt_t &_stmt, ir_context_t &ir_ctx);

stmt_t lift_buffer_offsets_in_send(const stmt_t &s, ir_context_t &ir_ctx);

stmt_t simplify_pass(const stmt_t &s, ir_context_t &ir_ctx);

// Replaces some heavy GRF reorders by reorder through SLM (store and load).
stmt_t inject_slm_reorder(const stmt_t &s, ir_context_t &ir_ctx,
        const conv_config_t &cfg, const grid_info_t &tg_grid);

stmt_t inject_send(const stmt_t &s, ir_context_t &ir_ctx);

// Lifts alloc statements out of loops.
stmt_t lift_alloc(
        const stmt_t &s, ir_context_t &ir_ctx, const conv_config_t &cfg);

// Lifts loop-invariant header assignments related to block 2D messages.
stmt_t lift_send_2d_header_store(const stmt_t &s, ir_context_t &ir_ctx);

// Moves invariant expressions out of loops.
stmt_t hoist_exprs(const stmt_t &s, ir_context_t &ir_ctx);

// Moves boolean mask computation from send calls to the top of the statement
// group corresponding to `label`. This is done to reduce GRF consumption and
// to reuse masks between calls. A vector boolean mask is stored as u16 type
// and converted to bool type right before the call. Transformation is limited
// to the statement group corresponding to `label`.
// If `split_by_and` is true then any ((A & B) & C) mask is split into A, B, C
// sub-masks which are initialized independently. This allows reusing those
// sub-masks for other masks.
stmt_t hoist_send_masks(const stmt_t &s, ir_context_t &ir_ctx,
        const stmt_label_t &label, bool split_by_and);

// Removes redundant u16 casts inside send masks which may appear after
// previous mask hoisting.
stmt_t remove_spurious_send_mask_cast(const stmt_t &s, ir_context_t &ir_ctx);

// Detects and converts expensive expression operations inside a loop to less
// expensive operations. Example:
// Before:
//     for (int j = 0; j < N; j++) {
//         int off = off_i + j * K;
//         a[off] = j;
//     }
// After:
//     int off = off_i;
//     for (int j = 0; j < N; j++) {
//         a[off] = j;
//         off += K;
//     }
stmt_t loop_strength_reduce(const stmt_t &s, ir_context_t &ir_ctx);

stmt_t optimize_alloc_let(const stmt_t &s, ir_context_t &ir_ctx);

// Eliminates let statements from the outer loops to be able to unroll loop
// nest for SLM buffering or prefetch injection. Example:
// Before:
//     for (int i = 0; i < I; i++) {
//         int tmp = TMP;
//         for (int j = 0; j < J; j++) {
//            ...
//         }
//     }
// After:
//     for (int i = 0; i < I; i++) {
//         for (int j = 0; j < J; j++) {
//             int tmp = TMP;
//             ...
//         }
//     }
stmt_t update_loops_for_unrolling(
        const stmt_t &s, ir_context_t &ir_ctx, const conv_config_t &cfg);

// Splits wide GRF stores otherwise unsupported in HW.
stmt_t split_wide_stores(const stmt_t &s, ir_context_t &ir_ctx);

// Detects and fixes overflows of operations with 32-bit integers.
// Before (a * b can overflow):
//     c.u64 = u64(c_ptr) + a.s32 * b.s32
// After:
//     c.u64 = u64(c_ptr) + s64(a.s32) * b.s32
stmt_t fix_int32_overflow(const stmt_t &s, ir_context_t &ir_ctx);

stmt_t optimize_peephole(const stmt_t &s, ir_context_t &ir_ctx);

stmt_t optimize_barrier(const stmt_t &s, ir_context_t &ir_ctx);

// Injects broadcasts for scalar if conditions. Example:
// Before:
//     if (cond) { ... }
// After (for SIMD8):
//     if (bcast8(cond)) { ... }
stmt_t fixup_if_conditions(const stmt_t &s, ir_context_t &ir_ctx);

// Unrolls loops according to their unroll attribute.
// Before:
//     for (int i = 0; i < 2; i++) [unroll: 2] {
//         body(i);
//     }
// After:
//     body(0);
//     body(1);
stmt_t unroll_loops(const stmt_t &s, ir_context_t &ir_ctx);

// Injects an allocation attribute to store information about buffer usages in
// instructions. This information is used during nGEN lowering to avoid bank
// conflicts in allocated buffers.
stmt_t inject_bank_conflict_attribute(const stmt_t &s, ir_context_t &ir_ctx);

// Converts dpas to dp4a.
stmt_t inject_dp4a(const stmt_t &s, ir_context_t &ir_ctx);

class expr_scalarizer_t : public ir_mutator_t {
public:
    expr_scalarizer_t(int elems, int idx,
            const object_map_t<expr_t, std::vector<expr_t>> &vec_vars)
        : elems_(elems), idx_(idx), vec_vars_(vec_vars) {}

    object_t _mutate(const cast_t &obj) override {
        if (obj.is_bool_vec_u16()) return obj;
        auto type = obj.type;
        auto expr = mutate(obj.expr);
        if (!type.is_scalar()) {
            ir_assert(type.elems() == elems_) << expr;
            type = type.scalar();
        }
        return cast_t::make(type, expr, obj.saturate);
    }

    object_t _mutate(const var_t &obj) override {
        if (obj.type.is_scalar()) return obj;

        auto it = vec_vars_.find(obj);
        ir_assert(it != vec_vars_.end()) << "Can't find variable: " << obj;
        ir_assert(int(it->second.size()) == elems_);
        return it->second[idx_];
    }

    object_t _mutate(const shuffle_t &obj) override {
        expr_t new_obj = ir_mutator_t::_mutate(obj);
        auto &shuffle = new_obj.as<shuffle_t>();
        ir_assert(shuffle.type.elems() == elems_) << new_obj;
        return new_obj[idx_];
    }

private:
    int elems_;
    int idx_;
    const object_map_t<expr_t, std::vector<expr_t>> &vec_vars_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
