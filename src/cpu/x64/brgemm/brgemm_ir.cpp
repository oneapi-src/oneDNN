/*******************************************************************************
* Copyright 2026 Intel Corporation
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

// The IR-based BRGEMM kernel converts a brgemm descriptor into IR, then runs
// the register allocator and code emitter.
//
// This kernel uses `jit_brgemm_kernel_t` as a reference and is comparable with
// the classic kernel instruction for instruction when it comes to compute
// structure.

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/brgemm/brgemm_ir.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/ir/codegen.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

#define VCONDCHECK_BRGEMM_IR(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, brgemm_ir, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

// The block of the output one N-block emission computes.
//
//   bd_block     - rows of the broadcast dimension the block covers
//   ld_block2    - vector registers of the load dimension it covers
//   ld_start     - load-dimension element the block starts at, counted from
//                  where the B and C pointers stand. B and C convert it with
//                  their own element size, so one count serves both.
//   ld_tail_mask - live elements of the last register, `none` when every
//                  register of the block is full
//
// `ld_start` is what lets the N tail blocks reach their columns without a
// pointer of their own. It is 0 for a block the N loop walks to.
struct out_block_t {
    int bd_block;
    int ld_block2;
    dim_t ld_start;
    ir::vreg_t ld_tail_mask;
};

// Fixed configuration used during IR generation for the BRGEMM builder.
//
// Every field is a build-time constant taken from the descriptor.
//
//   lda/ldb/ldc  - leading dimensions of A, B, and C, in elements
//   max_bs       - maximum batch size known at IR generation time
//   batch_kind   - how a batch element gives the address of its A and B
//   bd_block     - M rows per block
//   ld_block     - N elements per vector register
//   ld_block2    - vector registers of N per block
//   ldb_ld_elems - N elements one N block covers, `ld_block2 * ld_block`
//   rd_block     - K elements reduced per K block
//   dt_sz_a/b/c  - element size in bytes of A, B, and C
//   dt_a/b/c     - element data type of A, B, and C in memory
//   dt_acc       - accumulation data type
//   beta         - output scaling: 0 overwrites C, 1 accumulates into C
//   with_bias    - whether a bias is added to the output
//   dt_bias      - element data type of the bias
//   dt_sz_bias   - element size in bytes of the bias
//   bdb          - number of full M blocks
//   ldb2         - number of full N blocks
//   rdb          - number of full K blocks
//   bdb_tail     - M rows left over after the full M blocks
//   ldb2_tail    - full N registers left over after the full N blocks
//   ldb_tail     - N columns left over after every full N register
//   rdb_tail     - K elements left over after the full K blocks
//   rdb_a_off    - byte offset to advance A between K blocks
//   rdb_b_off    - byte offset to advance B between K blocks. This is also the
//                  distance a B load prefetches ahead, one K block of B rows
//   ldb_b_off    - byte offset to advance B between N blocks
//   ldb_c_off    - byte offset to advance C between N blocks
//   ldb_bias_off - byte offset to advance the bias between N blocks
//   bdb_a_off    - byte offset to advance A between M blocks
//   bdb_c_off    - byte offset to advance C between M blocks
//   with_injector_postops - whether attribute post-ops (eltwise, binary, sum)
//                  are requested
//   with_src_scales - whether a common source scale multiplies the output
//   with_wei_scales - whether a weights scale multiplies the output
//   per_n_wei_scales - whether the weights scale has one value per output
//                  column. Otherwise it is one common value
//   with_dst_scales - whether a common destination scale multiplies the output
//   dt_src_scales / dt_wei_scales - element data type of each scale
//   dt_sz_wei_scales - element size in bytes of a per-N weights scale, 0 for a
//                  common one, which is never indexed
//   ldb_wei_scale_off - byte offset to advance the weights scale between N
//                  blocks
struct brgemm_ir_conf_t {
    brgemm_ir_conf_t(const brgemm_desc_t &brg)
        : lda(brg.LDA)
        , ldb(brg.LDB)
        , ldc(brg.LDC)
        , max_bs(brg.brgattr.max_bs)
        , batch_kind(brg.type)
        , bd_block(brg.bd_block)
        , ld_block(brg.ld_block)
        , ld_block2(brg.ld_block2)
        , ldb_ld_elems(ld_block2 * ld_block)
        , rd_block(brg.rd_block)
        , dt_sz_a(brg.typesize_A)
        , dt_sz_b(brg.typesize_B)
        , dt_sz_c(brg.typesize_C)
        , dt_a(brg.dt_a)
        , dt_b(brg.dt_b)
        , dt_c(brg.dt_c)
        , dt_acc(data_type::f32)
        , beta(brg.beta)
        , with_bias(brg.with_bias)
        , dt_bias(brg.dt_bias)
        , dt_sz_bias(brg.typesize_bias)
        , bdb(brg.bdb)
        , ldb2(brg.ldb2)
        , rdb(brg.rdb)
        , bdb_tail(brg.bdb_tail)
        , ldb2_tail(brg.ldb2_tail)
        , ldb_tail(brg.ldb_tail)
        , rdb_tail(brg.rdb_tail)
        , rdb_a_off(dt_sz_a * rd_block)
        , rdb_b_off(dt_sz_b * rd_block * ldb)
        , ldb_b_off(dt_sz_b * ldb_ld_elems)
        , ldb_c_off(dt_sz_c * ldb_ld_elems)
        , ldb_bias_off(dt_sz_bias * ldb_ld_elems)
        , bdb_a_off(dt_sz_a * bd_block * lda)
        , bdb_c_off(dt_sz_c * bd_block * ldc)
        , with_injector_postops(
                  brg.with_eltwise || brg.with_binary || brg.with_sum)
        , with_src_scales(brg.with_src_scales)
        , with_wei_scales(brg.with_wei_scales)
        , per_n_wei_scales(brg.with_wei_scales && !brg.is_single_wei_scale)
        , with_dst_scales(brg.with_dst_scales)
        , dt_src_scales(brg.dt_src_scales)
        , dt_wei_scales(brg.dt_wei_scales)
        , dt_sz_wei_scales(per_n_wei_scales
                          ? (int)types::data_type_size(dt_wei_scales)
                          : 0)
        , ldb_wei_scale_off(dt_sz_wei_scales * ldb_ld_elems) {}

    const dim_t lda, ldb, ldc;
    const dim_t max_bs;
    const brgemm_batch_kind_t batch_kind;
    const int bd_block, ld_block, ld_block2, ldb_ld_elems, rd_block;
    const int dt_sz_a, dt_sz_b, dt_sz_c;
    const data_type_t dt_a, dt_b, dt_c, dt_acc;
    const float beta;
    const bool with_bias;
    const data_type_t dt_bias;
    const int dt_sz_bias;
    const dim_t bdb, ldb2, rdb;
    const int bdb_tail, ldb2_tail, ldb_tail, rdb_tail;
    const dim_t rdb_a_off, rdb_b_off;
    const dim_t ldb_b_off, ldb_c_off, ldb_bias_off;
    const dim_t bdb_a_off, bdb_c_off;
    const bool with_injector_postops;
    const bool with_src_scales, with_wei_scales, per_n_wei_scales;
    const bool with_dst_scales;
    const data_type_t dt_src_scales, dt_wei_scales;
    const int dt_sz_wei_scales;
    const dim_t ldb_wei_scale_off;

    // Displacements of one element of A, one vector of B, and one accumulator
    // of C from the start of the current block.
    //
    // These are the constants the builder encodes into the operation, the
    // `disp` of `ir::mem_t`. They are functions rather than expressions written
    // at each use because `brgemm_ir_supported()` range-checks the largest of
    // them against int32, and the check is only meaningful if there is
    // consistency between the check and the builder.

    // A is read one element at a time and broadcast, so `rd` is an element
    // index.
    dim_t a_off(int bd, int rd) const {
        return dt_sz_a * ((dim_t)bd * lda + rd);
    }

    // B is read one full vector per `ld`, so `rd` is a row index and `ld`
    // picks the vector within the N block.
    dim_t b_off(const out_block_t &blk, int ld, int rd) const {
        return dt_sz_b * ((dim_t)rd * ldb + ld_pos(blk, ld));
    }

    dim_t c_off(const out_block_t &blk, int bd, int ld) const {
        return dt_sz_c * ((dim_t)bd * ldc + ld_pos(blk, ld));
    }

    // One bias value per output column, so the offset has no row term.
    dim_t bias_off(const out_block_t &blk, int ld) const {
        return dt_sz_bias * ld_pos(blk, ld);
    }

    // A per-N weights scale is indexed like the bias.
    dim_t wei_scale_off(const out_block_t &blk, int ld) const {
        return dt_sz_wei_scales * ld_pos(blk, ld);
    }

    // Load-dimension element the vector `ld` of the block starts at, counted
    // from where the B and C pointers stand. B and C differ only in the
    // element size they scale it by.
    dim_t ld_pos(const out_block_t &blk, int ld) const {
        return blk.ld_start + (dim_t)ld * ld_block;
    }

    // Load-dimension elements the N loop leaves on the B offset and the C
    // pointer, which the M loop rewinds. `emit_loop_imm()` inlines a single
    // iteration without its step, so a one-block N loop advances nothing.
    dim_t ldb_loop_ld_adv() const { return ldb2 > 1 ? ldb2 * ldb_ld_elems : 0; }
    dim_t ldb_loop_c_adv() const { return dt_sz_c * ldb_loop_ld_adv(); }
    dim_t ldb_loop_bias_adv() const { return dt_sz_bias * ldb_loop_ld_adv(); }
    dim_t ldb_loop_wei_scale_adv() const {
        return dt_sz_wei_scales * ldb_loop_ld_adv();
    }

    // First load-dimension element of each N tail block, counted from where
    // the N loop left the pointers.
    //
    // Both blocks are emitted once, never in a loop, so they reach their
    // columns through the displacement of each access rather than through a
    // pointer of their own. `ldb2 * ldb_ld_elems` is where the tails start and
    // `ldb_loop_ld_adv()` is where the pointers stand.
    dim_t ldb2_tail_ld_start() const {
        return ldb2 * ldb_ld_elems - ldb_loop_ld_adv();
    }

    dim_t ldb_tail_ld_start() const {
        return ldb2_tail_ld_start() + (dim_t)ldb2_tail * ld_block;
    }

    // True if the kernel has anything to apply on a post-ops call: the bias, a
    // scale, or an attribute post-op.
    bool has_post_ops() const {
        return with_bias || with_injector_postops || with_src_scales
                || with_wei_scales || with_dst_scales;
    }
};

// M-loop input register classification
//
// The same split the GEMV IR kernel uses (see `brgemv_ir.cpp`), by whether a
// value advances across M-loop iterations. The one difference is that this
// loop nest has two levels over the output, M and N, so the level a register
// advances at is stated per field instead of shared by the struct.
//
// Design rule:
// - A field's vreg ID never changes after assignment.
// - Because of this, these structs are passed by const reference.
//
// Note:
// - Registers created inside a loop body (the accumulators, the B vectors, the
//   A broadcast, and the batch, A, and B pointers) are local temporaries and
//   are not included here.

// Registers that advance by a fixed byte offset as the nest walks the output.
struct advancing_regs_t {
    // Current C pointer. The N loop advances it by `ldb_c_off` per block, then
    // the M loop rewinds it and steps to the next M block with one constant,
    // `bdb_c_off - ldb_loop_c_adv()`. The classic kernel keeps a second base
    // pointer for this instead. One running pointer saves the general-purpose
    // register.
    ir::vreg_t c_ptr = ir::vreg_t::none;
    // Pointer the finished block is stored through. It holds the `ptr_D`
    // kernel argument on a post-ops call and `ptr_C` otherwise, and can be the
    // same register as `c_ptr` (see `init_m_loop_input_regs()`). C and D have
    // the same data type and leading dimension (`brgemm_ir_supported()`), so it
    // advances like `c_ptr`.
    ir::vreg_t store_ptr = ir::vreg_t::none;
    // Byte offset into A of the current M block. Starts at 0 and advances by
    // `bdb_a_off` per M block.
    ir::vreg_t a_off = ir::vreg_t::none;
    // Byte offset into B of the current N block. Advances by `ldb_b_off` per N
    // block and returns to 0 for the next M block.
    ir::vreg_t b_off = ir::vreg_t::none;
    // Current bias pointer. The bias is indexed by the load dimension only, so
    // it advances by `ldb_bias_off` per N block and rewinds for the next M
    // block.
    // `none` unless `with_bias`.
    ir::vreg_t bias_ptr = ir::vreg_t::none;
    // Current weights-scale pointer. A per-N scale advances and rewinds like
    // `bias_ptr`, a common one stays at its only element.
    // `none` unless `with_wei_scales`.
    ir::vreg_t wei_scale_ptr = ir::vreg_t::none;
};

// Registers that hold the same value for the entire M loop.
struct invariant_regs_t {
    // Base pointer of the batch-element array.
    ir::vreg_t batch = ir::vreg_t::none;
    // Batch size loop count. `none` when max_bs == 1 (single batch element).
    ir::vreg_t bs = ir::vreg_t::none;
    // A and B base pointers, the `ptr_A` and `ptr_B` kernel arguments, which
    // `brgemm_offs` adds the byte offsets of each batch element to. `none` for
    // `brgemm_addr`, where each element holds its own pointers.
    ir::vreg_t a_base = ir::vreg_t::none;
    ir::vreg_t b_base = ir::vreg_t::none;
    // The `do_post_ops` kernel argument. Non-zero applies the post-ops and
    // stores to D, zero stores to C. `none` when the kernel has no post-op.
    ir::vreg_t do_post_ops = ir::vreg_t::none;
    // Common source and destination scale pointers.
    // `none` unless `with_src_scales` and `with_dst_scales` respectively.
    ir::vreg_t src_scale_ptr = ir::vreg_t::none;
    ir::vreg_t dst_scale_ptr = ir::vreg_t::none;
};

// Complete input register set for the M loop, partitioned by whether values
// advance across iterations.
struct m_loop_input_regs_t {
    advancing_regs_t advancing;
    invariant_regs_t invariant;
};

// Sets up the M-loop input registers.
//
// Loads the kernel argument pointers and the batch count, and zeroes the
// running A and B offsets.
m_loop_input_regs_t init_m_loop_input_regs(
        ir::ir_t &ir, const brgemm_ir_conf_t &cfg) {
    m_loop_input_regs_t regs;

    regs.advancing.c_ptr = ir.new_gpr();
    ir.load_param(regs.advancing.c_ptr, GET_OFF(ptr_C));

    regs.invariant.batch = ir.new_gpr();
    ir.load_param(regs.invariant.batch, GET_OFF(batch));

    if (cfg.max_bs > 1) {
        regs.invariant.bs = ir.new_gpr();
        ir.load_param(regs.invariant.bs, GET_OFF(BS));
    }

    if (cfg.batch_kind == brgemm_offs) {
        regs.invariant.a_base = ir.new_gpr();
        ir.load_param(regs.invariant.a_base, GET_OFF(ptr_A));
        regs.invariant.b_base = ir.new_gpr();
        ir.load_param(regs.invariant.b_base, GET_OFF(ptr_B));
    }

    regs.advancing.a_off = ir.new_gpr();
    ir.mov_imm(regs.advancing.a_off, 0);

    regs.advancing.b_off = ir.new_gpr();
    ir.mov_imm(regs.advancing.b_off, 0);

    if (cfg.with_bias) {
        regs.advancing.bias_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.bias_ptr, GET_OFF(ptr_bias));
    }

    if (cfg.with_src_scales) {
        regs.invariant.src_scale_ptr = ir.new_gpr();
        ir.load_param(regs.invariant.src_scale_ptr, GET_OFF(ptr_src_scales));
    }

    if (cfg.with_wei_scales) {
        regs.advancing.wei_scale_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.wei_scale_ptr, GET_OFF(ptr_wei_scales));
    }

    if (cfg.with_dst_scales) {
        regs.invariant.dst_scale_ptr = ir.new_gpr();
        ir.load_param(regs.invariant.dst_scale_ptr, GET_OFF(ptr_dst_scales));
    }

    // The block goes to `ptr_D` when `do_post_ops` is set, and to `ptr_C`
    // otherwise. Resolve that once here so that the stores do not branch.
    //
    // A post-ops call with `beta == 1` reads C for the accumulators and writes
    // D, so the two pointers need separate registers. C is never read when
    // `beta == 0`, so one register is enough and it holds `ptr_D` on a post-ops
    // call.
    if (cfg.has_post_ops()) {
        regs.invariant.do_post_ops = ir.new_gpr();
        ir.load_param(regs.invariant.do_post_ops, GET_OFF(do_post_ops));

        if (cfg.beta == 0.0f) {
            regs.advancing.store_ptr = regs.advancing.c_ptr;
        } else {
            regs.advancing.store_ptr = ir.new_gpr();
            ir.mov_reg(regs.advancing.store_ptr, regs.advancing.c_ptr);
        }

        const int if_post_ops
                = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);
        ir.load_param(regs.advancing.store_ptr, GET_OFF(ptr_D));
        ir.if_end(if_post_ops);
    } else {
        regs.advancing.store_ptr = regs.advancing.c_ptr;
    }

    return regs;
}

// Innermost reduction step, over `rd_loop` elements of K.
//
// Loads one B vector per N register, then for each M row broadcasts one A
// element and multiply-adds it into that row's accumulators. This is the
// `!n_bcast_1_load` shape of `gemm_microkernel()` in `jit_brgemm_kernel.cpp`.
void emit_microkernel(ir::ir_t &ir, const brgemm_ir_conf_t &cfg,
        const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
        const std::vector<ir::vreg_t> &b, ir::vreg_t a, ir::vreg_t a_ptr,
        ir::vreg_t b_ptr, int rd_loop) {

    for (int rd = 0; rd < rd_loop; rd++) {
        // Only the last register of a block can be partial, and a block that
        // has one holds nothing else, so the mask covers every B load here.
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            if (blk.ld_tail_mask == ir::vreg_t::none)
                ir.vload(b[ld], b_ptr, cfg.b_off(blk, ld, rd), cfg.dt_b);
            else
                ir.vload_masked(b[ld], b_ptr, cfg.b_off(blk, ld, rd),
                        blk.ld_tail_mask, cfg.dt_b);
        }

        // One prefetch per B register, issued on the first `ld_block2` rows, so
        // each reduction step fetches the next K block of B exactly once.
        int n_pf_b = 0;
        for (int bd = 0; bd < blk.bd_block; bd++) {
            ir.vload_bcast(a, a_ptr, cfg.a_off(bd, rd), cfg.dt_a);
            if (n_pf_b < blk.ld_block2) {
                ir.prefetch(b_ptr, cfg.b_off(blk, n_pf_b, rd) + cfg.rdb_b_off);
                n_pf_b++;
            }

            for (int ld = 0; ld < blk.ld_block2; ld++)
                ir.vdot(acc[bd * blk.ld_block2 + ld], b[ld], a);
        }
    }
}

// One batch element.
//
// Derives the A and B pointers of the element, shifts them to the current M and
// N block, then runs the reduction loop over K.
void emit_bs_body(ir::ir_t &ir, const brgemm_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, const out_block_t &blk,
        const std::vector<ir::vreg_t> &acc, const std::vector<ir::vreg_t> &b,
        ir::vreg_t a, ir::vreg_t batch_ptr) {

    const ir::vreg_t a_ptr = ir.new_gpr();
    const ir::vreg_t b_ptr = ir.new_gpr();

    // Where this element's A and B start. Mirrors `set_A_B_matrices()` in
    // `jit_brgemm_kernel.cpp`.
    switch (cfg.batch_kind) {
        case brgemm_addr:
            ir.load(a_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.A));
            ir.load(b_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.B));
            break;
        case brgemm_offs:
            ir.load(a_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(offset.A));
            ir.add_reg(a_ptr, regs.invariant.a_base);
            ir.load(b_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(offset.B));
            ir.add_reg(b_ptr, regs.invariant.b_base);
            break;
        default: assert(!"unsupported batch kind"); break;
    }

    // Shift to the current M and N block.
    ir.add_reg(a_ptr, regs.advancing.a_off);
    ir.add_reg(b_ptr, regs.advancing.b_off);

    // Advance to the next batch element. A single batch element has no next
    // one, so nothing is emitted then.
    if (cfg.max_bs > 1) {
        ir.add_imm(batch_ptr, sizeof(brgemm_batch_element_t));
        // The classic kernel prefetches the next element for the address kind
        // only.
        if (cfg.batch_kind == brgemm_addr) ir.prefetch(batch_ptr, 0);
    }

    // Advance the A and B pointers by one K block.
    auto advance_ptrs = [&]() {
        ir.add_imm(a_ptr, cfg.rdb_a_off);
        ir.add_imm(b_ptr, cfg.rdb_b_off);
    };

    // Reduce the full K blocks, then the tail if any. Cases by `rdb`:
    //   *  == 0  the whole reduction is the tail
    //   *  == 1  one block, advance by hand only if a tail follows
    //   *  >= 2  loop, advancing per iteration
    if (cfg.rdb >= 2) {
        ir::emit_loop_imm(ir, cfg.rdb, [&]() {
            emit_microkernel(
                    ir, cfg, blk, acc, b, a, a_ptr, b_ptr, cfg.rd_block);
        }, advance_ptrs);
    } else if (cfg.rdb == 1) {
        emit_microkernel(ir, cfg, blk, acc, b, a, a_ptr, b_ptr, cfg.rd_block);
        if (cfg.rdb_tail > 0) advance_ptrs();
    }

    // K tail needs no mask. A reduction step reads a full B vector along N
    // and one element of A, so a short tail is just fewer steps.
    if (cfg.rdb_tail > 0)
        emit_microkernel(ir, cfg, blk, acc, b, a, a_ptr, b_ptr, cfg.rdb_tail);
}

// One N block.
//
// Holds one accumulator per (M row, N register), reduces them over the batch,
// and stores them to C, or to D on a post-ops call.
void emit_n_block(ir::ir_t &ir, const brgemm_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, const out_block_t &blk) {

    std::vector<ir::vreg_t> acc(blk.bd_block * blk.ld_block2, ir::vreg_t::none);

    for (int bd = 0; bd < blk.bd_block; bd++) {
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            const int i = bd * blk.ld_block2 + ld;
            acc[i] = ir.new_vec(cfg.dt_acc);

            // The kernel takes only 0 and 1 for beta.
            if (cfg.beta == 0.0f)
                ir.vzero(acc[i]);
            else if (blk.ld_tail_mask == ir::vreg_t::none)
                ir.vload(acc[i], regs.advancing.c_ptr, cfg.c_off(blk, bd, ld),
                        cfg.dt_c);
            else
                ir.vload_masked(acc[i], regs.advancing.c_ptr,
                        cfg.c_off(blk, bd, ld), blk.ld_tail_mask, cfg.dt_c);
        }
    }

    std::vector<ir::vreg_t> b(blk.ld_block2, ir::vreg_t::none);
    for (int ld = 0; ld < blk.ld_block2; ld++)
        b[ld] = ir.new_vec(cfg.dt_b);

    // Batch reduction over the bs dimension.
    const ir::vreg_t batch_ptr = ir.new_gpr();
    ir.mov_reg(batch_ptr, regs.invariant.batch);

    const ir::vreg_t a = ir.new_vec(cfg.dt_a);

    auto bs_body
            = [&]() { emit_bs_body(ir, cfg, regs, blk, acc, b, a, batch_ptr); };

    if (cfg.max_bs > 1)
        ir::emit_loop_reg(ir, regs.invariant.bs, bs_body);
    else
        ir::emit_loop_imm(ir, 1, bs_body);

    if (cfg.has_post_ops()) {
        // The order is the one oneDNN defines and the classic kernel follows:
        //   src scale -> weights scale -> bias -> post-ops -> dst scale.
        // The scales are applied one at a time, as in the classic kernel, so
        // the result rounds the same way.
        const int if_post_ops
                = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);

        if (cfg.with_src_scales) {
            const ir::vreg_t sc = ir.new_vec(cfg.dt_src_scales);
            ir.vload_bcast(
                    sc, regs.invariant.src_scale_ptr, 0, cfg.dt_src_scales);
            for (const ir::vreg_t &v : acc)
                ir.vmul(v, sc);
        }

        if (cfg.with_wei_scales && !cfg.per_n_wei_scales) {
            const ir::vreg_t sc = ir.new_vec(cfg.dt_wei_scales);
            ir.vload_bcast(
                    sc, regs.advancing.wei_scale_ptr, 0, cfg.dt_wei_scales);
            for (const ir::vreg_t &v : acc)
                ir.vmul(v, sc);
        }

        if (cfg.per_n_wei_scales) {
            // One scale vector per N register, shared by every M row of the
            // block.
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const ir::vreg_t sc = ir.new_vec(cfg.dt_wei_scales);
                if (blk.ld_tail_mask == ir::vreg_t::none)
                    ir.vload(sc, regs.advancing.wei_scale_ptr,
                            cfg.wei_scale_off(blk, ld), cfg.dt_wei_scales);
                else
                    ir.vload_masked(sc, regs.advancing.wei_scale_ptr,
                            cfg.wei_scale_off(blk, ld), blk.ld_tail_mask,
                            cfg.dt_wei_scales);

                for (int bd = 0; bd < blk.bd_block; bd++)
                    ir.vmul(acc[bd * blk.ld_block2 + ld], sc);
            }
        }

        if (cfg.with_bias) {
            // One bias vector per N register, shared by every M row of the
            // block.
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const ir::vreg_t bias = ir.new_vec(cfg.dt_bias);
                if (blk.ld_tail_mask == ir::vreg_t::none)
                    ir.vload(bias, regs.advancing.bias_ptr,
                            cfg.bias_off(blk, ld), cfg.dt_bias);
                else
                    ir.vload_masked(bias, regs.advancing.bias_ptr,
                            cfg.bias_off(blk, ld), blk.ld_tail_mask,
                            cfg.dt_bias);

                for (int bd = 0; bd < blk.bd_block; bd++)
                    ir.vadd(acc[bd * blk.ld_block2 + ld], bias);
            }
        }

        if (cfg.with_injector_postops) {
            // The output offset of each accumulator is its store displacement
            // below. A binary post-op finds its right-hand-side element through
            // it, and a sum post-op reads the previous value from where the
            // result is written. A block is either all full registers or the
            // one masked register of the N tail, which holds `ldb_tail`
            // elements.
            std::vector<dim_t> out_byte_off(acc.size());
            for (int bd = 0; bd < blk.bd_block; bd++)
                for (int ld = 0; ld < blk.ld_block2; ld++)
                    out_byte_off[bd * blk.ld_block2 + ld]
                            = cfg.c_off(blk, bd, ld);

            const int n_elems = blk.ld_tail_mask != ir::vreg_t::none
                    ? cfg.ldb_tail
                    : cfg.ld_block;
            ir.inject_postops(
                    acc, regs.advancing.store_ptr, out_byte_off, n_elems);
        }

        if (cfg.with_dst_scales) {
            // The caller passes the inverted f32 destination scale, so it is a
            // multiply.
            const ir::vreg_t sc = ir.new_vec(data_type::f32);
            ir.vload_bcast(sc, regs.invariant.dst_scale_ptr, 0, data_type::f32);
            for (const ir::vreg_t &v : acc)
                ir.vmul(v, sc);
        }

        ir.if_end(if_post_ops);
    }

    // `store_ptr` already points at C or D, so the stores do not branch. Their
    // memory data type is fixed when the IR is built, which is why C and D must
    // share one.
    for (int bd = 0; bd < blk.bd_block; bd++) {
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            const ir::vreg_t src = acc[bd * blk.ld_block2 + ld];

            if (blk.ld_tail_mask == ir::vreg_t::none) {
                ir.vstore(regs.advancing.store_ptr, cfg.c_off(blk, bd, ld), src,
                        cfg.dt_c);
            } else {
                ir.vstore_masked(regs.advancing.store_ptr,
                        cfg.c_off(blk, bd, ld), src, blk.ld_tail_mask,
                        cfg.dt_c);
            }
        }
    }
}

// One M block, which is the N loop over `ldb2` full blocks followed by the two
// N tails.
//
// The tails are emitted once each, so they address their columns through
// `ld_start`.
void emit_m_block(ir::ir_t &ir, const brgemm_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, int bd_block,
        ir::vreg_t ld_tail_mask) {

    auto advance_ptrs = [&]() {
        ir.add_imm(regs.advancing.b_off, cfg.ldb_b_off);
        ir.add_imm(regs.advancing.c_ptr, cfg.ldb_c_off);
        if (regs.advancing.store_ptr != regs.advancing.c_ptr)
            ir.add_imm(regs.advancing.store_ptr, cfg.ldb_c_off);
        if (cfg.with_bias)
            ir.add_imm(regs.advancing.bias_ptr, cfg.ldb_bias_off);
        if (cfg.per_n_wei_scales)
            ir.add_imm(regs.advancing.wei_scale_ptr, cfg.ldb_wei_scale_off);
    };
    const out_block_t full {bd_block, cfg.ld_block2, 0, ir::vreg_t::none};
    ir::emit_loop_imm(ir, cfg.ldb2,
            [&]() { emit_n_block(ir, cfg, regs, full); }, advance_ptrs);

    // Full registers left over after the N loop.
    if (cfg.ldb2_tail > 0) {
        const out_block_t blk {bd_block, cfg.ldb2_tail,
                cfg.ldb2_tail_ld_start(), ir::vreg_t::none};
        emit_n_block(ir, cfg, regs, blk);
    }

    // Columns left over after every full register, in one masked register.
    if (cfg.ldb_tail > 0) {
        const out_block_t blk {
                bd_block, 1, cfg.ldb_tail_ld_start(), ld_tail_mask};
        emit_n_block(ir, cfg, regs, blk);
    }
}

// Builds IR for BRGEMM.
//
// Computes:
//   C[i][j] = beta * C[i][j] + sum_bs sum_k A[i][k] * B[k][j]
//   (m = brg.bcast_dim, n = brg.load_dim, k = brg.reduce_dim)
//
// A call that asks for post-ops applies the scales, bias and attribute
// post-ops, and stores to D instead of C:
//   D[i][j] = post_ops(C[i][j] * src_scale * wei_scale[j] + bias[j])
//             * dst_scale
//
// The output is partitioned into M blocks of `bd_block` rows, each split into
// N blocks of `ld_block2 * ld_block` columns. Each block keeps its result in
// registers across the whole batch and K reduction, then stores it once.
void build_brgemm(const brgemm_desc_t &brg, ir::ir_t &ir) {
    const brgemm_ir_conf_t cfg(brg);
    const m_loop_input_regs_t regs = init_m_loop_input_regs(ir, cfg);

    // Every partial N block is the same width, so one mask serves the whole
    // kernel.
    ir::vreg_t ld_tail_mask = ir::vreg_t::none;

    if (cfg.ldb_tail > 0) {
        ld_tail_mask = ir.new_mask();
        ir.set_mask_imm(ld_tail_mask, cfg.ldb_tail);
    }

    // `store_ptr` is a separate register only when `beta == 1` and the kernel
    // has a post-op, so it needs its own advance only then. The bias and a
    // per-N weights scale do not depend on M, so they only rewind what the N
    // loop added.
    auto advance_ptrs = [&]() {
        ir.add_imm(regs.advancing.a_off, cfg.bdb_a_off);
        ir.add_imm(regs.advancing.c_ptr, cfg.bdb_c_off - cfg.ldb_loop_c_adv());
        if (regs.advancing.store_ptr != regs.advancing.c_ptr)
            ir.add_imm(regs.advancing.store_ptr,
                    cfg.bdb_c_off - cfg.ldb_loop_c_adv());
        if (cfg.with_bias && cfg.ldb_loop_bias_adv() != 0)
            ir.add_imm(regs.advancing.bias_ptr, -cfg.ldb_loop_bias_adv());
        if (cfg.per_n_wei_scales && cfg.ldb_loop_wei_scale_adv() != 0)
            ir.add_imm(regs.advancing.wei_scale_ptr,
                    -cfg.ldb_loop_wei_scale_adv());
        if (cfg.ldb_loop_ld_adv() != 0) ir.mov_imm(regs.advancing.b_off, 0);
    };
    ir::emit_loop_imm(ir, cfg.bdb, [&]() {
        emit_m_block(ir, cfg, regs, cfg.bd_block, ld_tail_mask);
    }, advance_ptrs);

    // Rows left over after the M loop. They form a block with fewer
    // accumulators and are otherwise a block like any other, so the pointers
    // have to arrive the way a loop iteration would leave them.
    if (cfg.bdb_tail > 0) {
        if (cfg.bdb == 1) advance_ptrs();
        emit_m_block(ir, cfg, regs, cfg.bdb_tail, ld_tail_mask);
    }
}

// IR-based BRGEMM kernel.
struct jit_brgemm_ir_kernel_t : public brgemm_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_ir_kernel_t)

    jit_brgemm_ir_kernel_t(const brgemm_desc_t &abrg)
        : brgemm_kernel_t(jit_name(), abrg.isa_impl), brg_(abrg) {}

    void generate() override {
        ir::ir_t ir;
        build_brgemm(brg_, ir);

        ir::postops_config_t postops;
        if (brg_.with_eltwise || brg_.with_binary || brg_.with_sum) {
            postops.post_ops = &brg_.attr()->post_ops_;
            postops.dst_md = brg_.dst_md();
            postops.rhs_arg_offset = GET_OFF(post_ops_binary_rhs_arg_vec);
            postops.dst_orig_off = GET_OFF(data_C_ptr_);
        }

        ir::generate_kernel(*this, ir, postops);
    }

private:
    brgemm_desc_t brg_;
};

} // namespace

// Returns `status::success` if the descriptor is supported by the GEMM IR
// kernel, otherwise `status::unimplemented`.
status_t brgemm_ir_supported(const brgemm_desc_t &brg) {
    using namespace data_type;
    using namespace utils;

    // This kernel supports only vector version of BRGEMM.
    VCONDCHECK_BRGEMM_IR(!brg.is_gemv, VERBOSE_UNSUPPORTED_FEATURE, "gemv");
    VCONDCHECK_BRGEMM_IR(!brg.is_dgmm, VERBOSE_UNSUPPORTED_FEATURE, "dgmm");
    VCONDCHECK_BRGEMM_IR(!brg.is_tmm && !brg.is_ace(),
            VERBOSE_UNSUPPORTED_FEATURE, "tile accumulation");

    // Allow the following ISAs. `avx2` and `avx512_core` are used to compute
    // f32 directly. For the other ISAs we receive already upconverted to f32
    // inputs.
    VCONDCHECK_BRGEMM_IR(
            one_of(brg.isa_impl, avx2, avx512_core, avx512_core_fp16, avx10_2),
            VERBOSE_UNSUPPORTED_ISA);
    VCONDCHECK_BRGEMM_IR(
            everyone_is(f32, brg.dt_a, brg.dt_b, brg.dt_c, brg.dt_d),
            VERBOSE_UNSUPPORTED_DT);

    VCONDCHECK_BRGEMM_IR(one_of(brg.type, brgemm_addr, brgemm_offs),
            VERBOSE_UNSUPPORTED_FEATURE, "batch kind");
    VCONDCHECK_BRGEMM_IR(brg.layout == brgemm_row_major,
            VERBOSE_UNSUPPORTED_FEATURE, "column-major layout");

    VCONDCHECK_BRGEMM_IR(everyone_is(0, brg.brgattr.max_top_vpad,
                                 brg.brgattr.max_bottom_vpad),
            VERBOSE_UNSUPPORTED_PAD_FEATURE, "virtual padding");

    VCONDCHECK_BRGEMM_IR(
            brg.alpha == 1.0f, VERBOSE_UNSUPPORTED_FEATURE, "alpha != 1");
    VCONDCHECK_BRGEMM_IR(brg.beta == 0.0f || brg.beta == 1.0f,
            VERBOSE_UNSUPPORTED_FEATURE, "beta != 0 && beta != 1");

    VCONDCHECK_BRGEMM_IR(!brg.req_s8s8_compensation,
            VERBOSE_UNSUPPORTED_FEATURE, "s8s8 compensation");
    VCONDCHECK_BRGEMM_IR(!brg.brgattr.use_intermediate_c_buffer,
            VERBOSE_UNSUPPORTED_FEATURE, "intermediate C buffer");
    VCONDCHECK_BRGEMM_IR(everyone_is(brgemm_broadcast_t::none, brg.zp_type_a,
                                 brg.zp_type_b, brg.zp_type_c),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    // Post-ops go through the JIT injector. Accept eltwise, binary and sum, and
    // only binary arguments the injector can handle on this ISA.
    if (brg.attr()) {
        const memory_desc_wrapper dst_d(brg.dst_md());
        const std::vector<injector::post_op_type> accepted
                = {injector::eltwise, injector::binary, injector::sum};
        VCONDCHECK_BRGEMM_IR(
                injector::post_ops_ok({brg.isa_impl, accepted,
                        brg.attr()->post_ops_, &dst_d,
                        /*sum_at_pos_0_only=*/false,
                        /*sum_requires_scale_one=*/false,
                        /*sum_requires_zp_zero=*/false,
                        /*sum_requires_same_params=*/false,
                        binary_injector::
                                get_all_strategies_supported_by_injector()}),
                VERBOSE_UNSUPPORTED_POSTOP);
    }

    // Scales are f32 values the kernel broadcasts or loads per N register:
    // src and dst scales are common, the weights scale common or per-N.
    VCONDCHECK_BRGEMM_IR(
            !brg.has_per_k_scales(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(brg.with_src_scales, brg.dt_src_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(brg.with_wei_scales, brg.dt_wei_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(!brg.with_weights_scale_adjust,
            VERBOSE_UNSUPPORTED_FEATURE, "weights scale adjust");

    VCONDCHECK_BRGEMM_IR(IMPLICATION(brg.with_bias, brg.dt_bias == f32),
            VERBOSE_UNSUPPORTED_BIAS_CFG);

    VCONDCHECK_BRGEMM_IR(brg.brgattr.hint_prefetchw == brgemm_prfw_default,
            VERBOSE_UNSUPPORTED_FEATURE, "store prefetch hint");

    VCONDCHECK_BRGEMM_IR(
            everyone_is(false, brg.is_runtime_lda, brg.is_runtime_ldb,
                    brg.is_runtime_ldc, brg.is_runtime_ldd),
            VERBOSE_UNSUPPORTED_FEATURE, "runtime leading dimension");

    VCONDCHECK_BRGEMM_IR(!brg.brgattr.generate_skip_accumulation,
            VERBOSE_UNSUPPORTED_FEATURE, "skip accumulation");

    VCONDCHECK_BRGEMM_IR(!brg.n_bcast_1_load, VERBOSE_UNSUPPORTED_FEATURE,
            "one-load microkernel");
    VCONDCHECK_BRGEMM_IR(!brg.embd_bcst, VERBOSE_UNSUPPORTED_FEATURE,
            "embedded broadcast microkernel");

    const brgemm_ir_conf_t cfg(brg);

    // A post-ops call stores to D through the C displacements.
    VCONDCHECK_BRGEMM_IR(IMPLICATION(cfg.has_post_ops(), brg.LDC == brg.LDD),
            VERBOSE_UNSUPPORTED_FEATURE, "LDC != LDD");

    // Every address the builder emits is a base register plus a build-time
    // displacement encoded in the instruction, so each displacement has to fit
    // in int32. These are the largest of them.
    auto fits = [](dim_t v) { return v <= INT32_MAX && v >= INT32_MIN; };
    const int rd_last = (cfg.rdb > 0 ? cfg.rd_block : cfg.rdb_tail) - 1;

    // Rows of the tallest M block. The M tail is shorter than a full block.
    const int max_bd_block = cfg.bdb > 0 ? cfg.bd_block : cfg.bdb_tail;

    // Upper bound for the B and C displacements.
    const out_block_t last_col {
            max_bd_block, 1, brg.load_dim - 1, ir::vreg_t::none};

    VCONDCHECK_BRGEMM_IR(fits(cfg.a_off(max_bd_block - 1, rd_last)),
            VERBOSE_UNSUPPORTED_FEATURE, "A displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.b_off(last_col, 0, rd_last) + cfg.rdb_b_off),
            VERBOSE_UNSUPPORTED_FEATURE, "B displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.c_off(last_col, max_bd_block - 1, 0)),
            VERBOSE_UNSUPPORTED_FEATURE, "C displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.bias_off(last_col, 0)),
            VERBOSE_UNSUPPORTED_FEATURE, "bias displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.wei_scale_off(last_col, 0)),
            VERBOSE_UNSUPPORTED_FEATURE,
            "weights scale displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.bdb_c_off - cfg.ldb_loop_c_adv()),
            VERBOSE_UNSUPPORTED_FEATURE, "C rewind overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.ldb_loop_bias_adv()),
            VERBOSE_UNSUPPORTED_FEATURE, "bias rewind overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.ldb_loop_wei_scale_adv()),
            VERBOSE_UNSUPPORTED_FEATURE,
            "weights scale rewind overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.bdb_a_off) && fits(cfg.rdb_a_off)
                    && fits(cfg.rdb_b_off) && fits(cfg.ldb_b_off)
                    && fits(cfg.ldb_c_off) && fits(cfg.ldb_bias_off)
                    && fits(cfg.ldb_wei_scale_off),
            VERBOSE_UNSUPPORTED_FEATURE, "pointer advance overflows int32");

    return status::success;
}

brgemm_kernel_t *create_brgemm_ir_kernel(const brgemm_desc_t &brg) {
    if (brgemm_ir_supported(brg) != status::success) return nullptr;
    return new jit_brgemm_ir_kernel_t(brg);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
