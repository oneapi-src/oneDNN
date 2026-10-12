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

// The IR-based GEMV kernel converts a brgemm descriptor into IR, then runs the
// register allocator and code emitter.
//
// This is the only file that knows about GEMV-specific math and data layout.
// Everything in IR is generic infrastructure that can be reused for other
// problems.
//
// The brgemv_ir_supported() function checks whether a problem matches the
// current requirements.

#include <cstdint>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/brgemm/brgemv_ir.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/ir/codegen.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

#define VCONDCHECK_BRGEMV_IR(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, brgemv_ir, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

// Fixed configuration used during IR generation for the GEMV builder.
//
//   m / k        - dimensions of matrix A (GEMV always has n = 1)
//   lda / incy   - leading dimensions of A and stride between elements of `y`
//   max_bs       - maximum batch size known at IR generation time
//   m_block      - M rows per full block
//   k_block      - K elements reduced per K block
//   dt_sz_a/x/y  - element size in bytes of A, x, and y
//   dt_a/x/y     - element data type of A, x, and y in memory
//   dt_a_reg / dt_x_reg - data type of the vec vreg holding A and x. It is the
//                  memory type when the reduction consumes it as it is, and
//                  f32 when the load converts. Paired with `k_block`, which is
//                  the element count that follows from the same choice.
//   dt_acc       - accumulation data type
//   beta         - output scaling: 0 overwrites y, 1 accumulates into y
//   m_blocks     - number of full M blocks
//   m_tail       - remaining M rows after the full blocks
//   k_blocks     - number of full K blocks
//   k_tail       - remaining K elements after the full blocks
//   mblk_*_off   - byte offset to advance A/y pointers between M blocks
//   kblk_*_off   - byte offset to advance A/x pointers between K blocks
//   with_bias    - whether a bias is added to the output
//   treat_y_as_row - whether the output y is a row. When true the bias is
//                    indexed per M row, otherwise broadcast from bias[0]
//   dt_bias      - element data type of the bias
//   dt_sz_bias   - element size in bytes of the bias
//   mblk_bias_off - byte offset to advance the bias pointer between M blocks
//   with_injector_postops - whether attribute post-ops (eltwise/binary/sum) are
//                    requested
//   with_src_scales - whether a source scale multiplies the output. Always a
//                    single common scalar for this kernel
//   with_wei_scales - whether a weights scale multiplies the output
//   single_wei_scale - whether the weights scale is one scalar for the whole
//                    output. False only for per-N scales with a row output,
//                    where each output element has its own scale
//   dt_src_scales / dt_wei_scales - element data type of each scale
//   dt_sz_wei_scales - element size in bytes of the weights scale
//   mblk_wei_scale_off - byte offset to advance the weights-scale pointer
//                    between M blocks
struct brgemv_ir_conf_t {
    brgemv_ir_conf_t(const brgemm_desc_t &brg)
        : m(brg.bcast_dim)
        , k(brg.reduce_dim)
        , lda(brg.LDA)
        , incy(brg.LDC)
        , max_bs(brg.brgattr.max_bs)
        , m_block(brg.gemv_bd_block())
        , k_block(brg.rd_block)
        , dt_sz_a(brg.typesize_A)
        , dt_sz_x(brg.typesize_B)
        , dt_sz_y(brg.typesize_C)
        , dt_a(brg.dt_a)
        , dt_x(brg.dt_b)
        , dt_y(brg.dt_c)
        , dt_a_reg(brg.gemv_use_vdpbf16ps() ? brg.dt_a : data_type::f32)
        , dt_x_reg(brg.gemv_use_vdpbf16ps() ? brg.dt_b : data_type::f32)
        , dt_acc(data_type::f32)
        , beta(brg.beta)
        , m_blocks(m / m_block)
        , m_tail(m % m_block)
        , k_blocks(k / k_block)
        , k_tail(k % k_block)
        , mblk_a_off(dt_sz_a * m_block * lda)
        , mblk_y_off(dt_sz_y * m_block * incy)
        , kblk_a_off(dt_sz_a * k_block)
        , kblk_x_off(dt_sz_x * k_block)
        , with_bias(brg.with_bias)
        , treat_y_as_row(brg.treat_y_as_row)
        , dt_bias(brg.dt_bias)
        , dt_sz_bias(brg.typesize_bias)
        , mblk_bias_off(dt_sz_bias * m_block)
        , with_injector_postops(
                  brg.with_eltwise || brg.with_binary || brg.with_sum)
        , with_src_scales(brg.with_src_scales)
        , with_wei_scales(brg.with_wei_scales)
        , single_wei_scale(brg.gemv_single_wei_scale())
        , dt_src_scales(brg.dt_src_scales)
        , dt_wei_scales(brg.dt_wei_scales)
        , dt_sz_wei_scales(with_wei_scales
                          ? (int)types::data_type_size(dt_wei_scales)
                          : 0)
        , mblk_wei_scale_off(dt_sz_wei_scales * m_block) {}

    const dim_t m, k, lda, incy;
    const dim_t max_bs;
    const int m_block, k_block;
    const int dt_sz_a, dt_sz_x, dt_sz_y;
    const data_type_t dt_a, dt_x, dt_y;
    const data_type_t dt_a_reg, dt_x_reg;
    const data_type_t dt_acc;
    const float beta;
    const dim_t m_blocks, m_tail, k_blocks, k_tail;
    const dim_t mblk_a_off, mblk_y_off;
    const dim_t kblk_a_off, kblk_x_off;
    const bool with_bias, treat_y_as_row;
    const data_type_t dt_bias;
    const int dt_sz_bias;
    const dim_t mblk_bias_off;
    const bool with_injector_postops;
    const bool with_src_scales, with_wei_scales, single_wei_scale;
    const data_type_t dt_src_scales, dt_wei_scales;
    const int dt_sz_wei_scales;
    const dim_t mblk_wei_scale_off;

    bool has_post_ops() const {
        return with_bias || with_injector_postops || with_src_scales
                || with_wei_scales;
    }
};

// M-loop input register classification
//
// These structs describe registers used by the M-block loop, split by
// whether their values advance each iteration or remain constant.
//
// Design rule:
// - A field's vreg ID never changes after assignment.
// - Because of this, these structs are passed by const reference.
//
// Note:
// - Registers created inside the M-block loop (acc, batch_ptr, ws, a_ptr,
//   x_ptr) are local temporaries and are not included here.

// Registers that advance each M-block iteration by a fixed byte offset.
struct advancing_regs_t {
    // Current `y` pointer, the vector `beta` accumulates into. Advances by
    // `mblk_y_off` per M-block.
    ir::vreg_t y_ptr = ir::vreg_t::none;
    // Pointer used to store the M-block result. Usually this is `y_ptr`, but
    // when post-ops run it holds `ptr_D`, which may be a separate output
    // buffer. The accumulation and output buffers share the same type and
    // stride (enforced by `brgemv_ir_supported()`), so `store_ptr` advances
    // together with `y_ptr`.
    ir::vreg_t store_ptr = ir::vreg_t::none;
    // Byte offset into A for the current M-block. Starts at 0 and advances by
    // `mblk_a_off` each iteration.
    ir::vreg_t a_off = ir::vreg_t::none;
    // Bias base pointer. Advances by `mblk_bias_off` per M-block when
    // `treat_y_as_row` is set, otherwise stays at the first element. `none`
    // unless `with_bias`.
    ir::vreg_t bias_ptr = ir::vreg_t::none;
    // Weights-scale base pointer. Advances by `mblk_wei_scale_off` per M-block
    // for per-N scales, otherwise stays at the first element. `none` unless
    // `with_wei_scales`.
    ir::vreg_t wei_scale_ptr = ir::vreg_t::none;
};

// Registers that remain constant for the entire M-loop.
struct invariant_regs_t {
    // Base pointer of the batch-element array.
    ir::vreg_t batch = ir::vreg_t::none;
    // Batch size loop count. `none` when max_bs == 1 (single batch element).
    ir::vreg_t bs = ir::vreg_t::none;
    // K-tail mask, shared by every masked tail load. `none` when `k_tail` is
    // zero, which is when there is no tail load to mask.
    ir::vreg_t k_tail_mask = ir::vreg_t::none;
    // Post-ops flag (params.do_post_ops). Non-zero applies the post-ops, zero
    // stores the raw accumulator. `none` unless the kernel has a post-op to
    // apply (bias, scales, or injector post-ops).
    ir::vreg_t do_post_ops = ir::vreg_t::none;
    // Source-scale pointer. A single common scalar, so it never advances.
    // `none` unless `with_src_scales`.
    ir::vreg_t src_scale_ptr = ir::vreg_t::none;
};

// Complete input register set for the M-loop, partitioned by whether values
// advance across iterations.
struct m_loop_input_regs_t {
    advancing_regs_t advancing;
    invariant_regs_t invariant;
};

// Sets up the M-loop input registers.
//
// Loads kernel argument pointers and the batch count, and initializes the
// running A offset to zero.
//
// Optional arguments are only initialized when the configuration specifies
// their presence. Otherwise, they remain `none`.
m_loop_input_regs_t init_m_loop_input_regs(
        ir::ir_t &ir, const brgemv_ir_conf_t &cfg) {
    m_loop_input_regs_t regs;

    regs.advancing.y_ptr = ir.new_gpr();
    ir.load_param(regs.advancing.y_ptr, GET_OFF(ptr_C));

    regs.invariant.batch = ir.new_gpr();
    ir.load_param(regs.invariant.batch, GET_OFF(batch));

    if (cfg.max_bs > 1) {
        regs.invariant.bs = ir.new_gpr();
        ir.load_param(regs.invariant.bs, GET_OFF(BS));
    }

    regs.advancing.a_off = ir.new_gpr();
    ir.mov_imm(regs.advancing.a_off, 0);

    if (cfg.k_tail > 0) {
        // We only need to set the mask once per kernel. It's lifetime is
        // managed automatically by the allocator.
        regs.invariant.k_tail_mask = ir.new_mask();
        ir.set_mask_imm(regs.invariant.k_tail_mask, (int)cfg.k_tail);
    }

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

    if (cfg.has_post_ops()) {
        regs.invariant.do_post_ops = ir.new_gpr();
        ir.load_param(regs.invariant.do_post_ops, GET_OFF(do_post_ops));

        regs.advancing.store_ptr = ir.new_gpr();
        ir.mov_reg(regs.advancing.store_ptr, regs.advancing.y_ptr);

        const int if_post_ops
                = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);
        ir.load_param(regs.advancing.store_ptr, GET_OFF(ptr_D));
        ir.if_end(if_post_ops);
    } else {
        regs.advancing.store_ptr = regs.advancing.y_ptr;
    }

    return regs;
}

} // namespace

namespace nontrans {

// Innermost reduction step.
//
// Loads one k_block-wide chunk of `x`, then performs a multiply-add into each
// accumulator in `acc` using the corresponding chunk of its A row.
//
// Each accumulator holds a partial dot product for one output. The number of
// accumulators is the number of rows in the current M block or M tail.
void emit_microkernel(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const std::vector<ir::vreg_t> &acc, ir::vreg_t a_ptr,
        ir::vreg_t x_ptr) {
    // GEMV is bandwidth-bound, so SW-prefetch ahead of each load: the x vector
    // once and every A row. The distance is empirically tuned.
    constexpr dim_t gemv_pf_dist = 512; // bytes = 8 cache lines

    const ir::vreg_t x = ir.new_vec(cfg.dt_x_reg);
    const ir::vreg_t a = ir.new_vec(cfg.dt_a_reg);
    ir.prefetch(x_ptr, gemv_pf_dist);
    ir.vload(x, x_ptr, 0, cfg.dt_x);
    for (int i = 0; i < (int)acc.size(); i++) {
        const dim_t a_off = cfg.dt_sz_a * (dim_t)i * cfg.lda;
        ir.prefetch(a_ptr, a_off + gemv_pf_dist);
        ir.vload(a, a_ptr, a_off, cfg.dt_a);
        ir.vdot(acc[i], a, x);
    }
}

// Innermost reduction step for K tail.
//
// Same shape as emit_microkernel, but uses masked loads.
void emit_microkernel_tail(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const std::vector<ir::vreg_t> &acc, ir::vreg_t a_ptr, ir::vreg_t x_ptr,
        ir::vreg_t mask) {
    const ir::vreg_t x = ir.new_vec(cfg.dt_x_reg);
    const ir::vreg_t a = ir.new_vec(cfg.dt_a_reg);
    ir.vload_masked(x, x_ptr, 0, mask, cfg.dt_x);
    for (int i = 0; i < (int)acc.size(); i++) {
        ir.vload_masked(
                a, a_ptr, cfg.dt_sz_a * (dim_t)i * cfg.lda, mask, cfg.dt_a);
        ir.vdot(acc[i], a, x);
    }
}

// One batch element.
//
// Loads A and x pointers, then runs the reduction loop over k.
// Each iteration processes one `k_block` chunk and advances the pointers.
void emit_bs_body(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const std::vector<ir::vreg_t> &acc, ir::vreg_t batch_ptr,
        ir::vreg_t a_off, ir::vreg_t k_tail_mask) {
    const ir::vreg_t a_ptr = ir.new_gpr();
    const ir::vreg_t x_ptr = ir.new_gpr();

    ir.load(a_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.A));
    ir.add_reg(a_ptr, a_off);
    ir.load(x_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.B));
    ir.add_imm(batch_ptr, sizeof(brgemm_batch_element_t));

    // Advance the A and x pointers by one K block.
    auto advance_ptrs = [&]() {
        ir.add_imm(a_ptr, cfg.kblk_a_off);
        ir.add_imm(x_ptr, cfg.kblk_x_off);
    };

    // Reduce the full K blocks, then the tail if any. Cases by `k_blocks`.
    // k > 0 guarantees k_blocks and k_tail are never both zero.
    //   *  == 0  whole reduction is the tail
    //   *  == 1  one block, advance by hand only if a tail follows
    //   *  >= 2  loop, advancing per iteration
    if (cfg.k_blocks >= 2) {
        ir::emit_loop_imm(ir, cfg.k_blocks, [&]() {
            emit_microkernel(ir, cfg, acc, a_ptr, x_ptr);
        }, advance_ptrs);
    } else if (cfg.k_blocks == 1) {
        emit_microkernel(ir, cfg, acc, a_ptr, x_ptr);
        if (cfg.k_tail > 0) advance_ptrs();
    }
    if (cfg.k_tail > 0)
        emit_microkernel_tail(ir, cfg, acc, a_ptr, x_ptr, k_tail_mask);
}

// One M block.
//
// Contains `m_block` independent accumulators, reduced across the batch.
// They are then horizontally reduced to a single scalar each and stored.
//
// Finally, advances the A and x pointers to the next M block.
void emit_m_block(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, int m_block) {
    std::vector<ir::vreg_t> acc(m_block, ir::vreg_t::none);
    for (int r = 0; r < m_block; r++) {
        acc[r] = ir.new_vec(cfg.dt_acc);

        // The current implementation supports only 0 and 1 for beta so for the
        // case where beta = 1 we load `y` into the accumulator registers and
        // then the microkernel adds the results of the multiplication to them.
        //
        // This reads `y_ptr` while a sum post-op reads `store_ptr`, so the two
        // never count the same value twice.
        if (cfg.beta == 0.0f)
            ir.vzero(acc[r]);
        else
            ir.vload_scalar(acc[r], regs.advancing.y_ptr,
                    cfg.dt_sz_y * (dim_t)r * cfg.incy, cfg.dt_y);
    }

    // Batch reduction over bs dimension
    const ir::vreg_t batch_ptr = ir.new_gpr();
    ir.mov_reg(batch_ptr, regs.invariant.batch);

    if (cfg.max_bs > 1) {
        ir::emit_loop_reg(ir, regs.invariant.bs, [&]() {
            emit_bs_body(ir, cfg, acc, batch_ptr, regs.advancing.a_off,
                    regs.invariant.k_tail_mask);
        });
    } else {
        ir::emit_loop_imm(ir, 1, [&]() {
            emit_bs_body(ir, cfg, acc, batch_ptr, regs.advancing.a_off,
                    regs.invariant.k_tail_mask);
        });
    }

    // Horizontal reduction + store
    const ir::vreg_t ws = ir.new_vec(cfg.dt_acc);
    for (int r = 0; r < m_block; r++)
        ir.vhreduce(acc[r], ws);

    if (cfg.has_post_ops()) {
        // The implemented order conforms to oneDNN's defined semantics for
        // applying scales, bias and post-ops:
        //   src scale -> weights scale -> bias -> post-ops.
        // Scales apply to the accumulated result before the bias add and
        // post-ops apply after bias. The two scales can swap because both are
        // multiplies but a scale must not move past the bias and the bias must
        // not move past the post-ops.
        const int if_post_ops
                = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);

        if (cfg.with_src_scales) {
            // Loaded once and applied to every output.
            const ir::vreg_t sc = ir.new_vec(cfg.dt_src_scales);
            ir.vload_scalar(
                    sc, regs.invariant.src_scale_ptr, 0, cfg.dt_src_scales);

            for (int r = 0; r < m_block; r++)
                ir.vmul(acc[r], sc);
        }

        if (cfg.with_wei_scales) {
            // The single scale is loop invariant, so load it once above the
            // loop. The per-N case loads a separate scale per output element.
            const ir::vreg_t sc = ir.new_vec(cfg.dt_wei_scales);
            if (cfg.single_wei_scale)
                ir.vload_scalar(
                        sc, regs.advancing.wei_scale_ptr, 0, cfg.dt_wei_scales);

            for (int r = 0; r < m_block; r++) {
                if (!cfg.single_wei_scale)
                    ir.vload_scalar(sc, regs.advancing.wei_scale_ptr,
                            cfg.dt_sz_wei_scales * (dim_t)r, cfg.dt_wei_scales);
                ir.vmul(acc[r], sc);
            }
        }

        if (cfg.with_bias) {
            // The broadcast bias is loop invariant, so load it once outside the
            // loop. A row output loads a separate bias per output element.
            const ir::vreg_t bias = ir.new_vec(cfg.dt_bias);
            if (!cfg.treat_y_as_row)
                ir.vload_scalar(bias, regs.advancing.bias_ptr, 0, cfg.dt_bias);

            for (int r = 0; r < m_block; r++) {
                if (cfg.treat_y_as_row)
                    ir.vload_scalar(bias, regs.advancing.bias_ptr,
                            cfg.dt_sz_bias * (dim_t)r, cfg.dt_bias);
                ir.vadd(acc[r], bias);
            }
        }

        if (cfg.with_injector_postops) {
            // Each accumulator is horizontally reduced to one scalar, which is
            // the active element count `generate()` gives the injector. The
            // output offset matches the store displacement below, so a sum
            // post-op reads each element from the address its result is
            // written to.
            std::vector<dim_t> out_byte_off(m_block);
            for (int r = 0; r < m_block; r++)
                out_byte_off[r] = cfg.dt_sz_y * (dim_t)r * cfg.incy;

            ir.inject_postops(acc, regs.advancing.store_ptr, out_byte_off);
        }

        ir.if_end(if_post_ops);
    }

    for (int r = 0; r < m_block; r++)
        ir.vstore_scalar(regs.advancing.store_ptr,
                cfg.dt_sz_y * (dim_t)r * cfg.incy, acc[r], cfg.dt_y);

    // Advance to next M block
    ir.add_imm(regs.advancing.a_off, cfg.mblk_a_off);
    ir.add_imm(regs.advancing.y_ptr, cfg.mblk_y_off);
    // When `!has_post_ops()`, `store_ptr` is the same IR vreg as `y_ptr` (see
    // `init_m_loop_input_regs`), so advancing `y_ptr` above covers both.
    if (cfg.has_post_ops())
        ir.add_imm(regs.advancing.store_ptr, cfg.mblk_y_off);

    if (cfg.with_bias && cfg.treat_y_as_row)
        ir.add_imm(regs.advancing.bias_ptr, cfg.mblk_bias_off);
    if (cfg.with_wei_scales && !cfg.single_wei_scale)
        ir.add_imm(regs.advancing.wei_scale_ptr, cfg.mblk_wei_scale_off);
}

// Builds IR for GEMV with a non-transposed A matrix.
//
// Computes:
//   y[i] = sum_k A[i][k] * x[k]
//   (m = brg.bcast_dim, k = brg.reduce_dim, n = 1)
//
// The output is partitioned into full M blocks of `m_block` rows each, with a
// final partial block of `m_tail` rows when m is not a multiple of m_block.
//
// Each M block:
// - Maintains one independent accumulator per row
// - Accumulates over k in k_block-wide chunks across the batch
// - Horizontally reduces each accumulator to a scalar output value
//   and stores it to memory
void build_gemv(const brgemm_desc_t &brg, ir::ir_t &ir) {
    const brgemv_ir_conf_t cfg(brg);
    const m_loop_input_regs_t regs = init_m_loop_input_regs(ir, cfg);

    if (cfg.m_blocks > 0)
        ir::emit_loop_imm(ir, cfg.m_blocks,
                [&]() { emit_m_block(ir, cfg, regs, cfg.m_block); });

    if (cfg.m_tail > 0) emit_m_block(ir, cfg, regs, (int)cfg.m_tail);
}

} // namespace nontrans

namespace {

// IR-based BRGEMV kernel.
struct jit_brgemv_ir_kernel_t : public brgemm_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemv_ir_kernel_t)

    jit_brgemv_ir_kernel_t(const brgemm_desc_t &abrg)
        : brgemm_kernel_t(jit_name(), abrg.isa_impl), brg_(abrg) {}

    void generate() override {
        ir::ir_t ir;
        nontrans::build_gemv(brg_, ir);

        ir::postops_config_t postops;
        if (brg_.with_eltwise || brg_.with_binary || brg_.with_sum) {
            postops.post_ops = &brg_.attr()->post_ops_;
            postops.dst_md = brg_.dst_md();
            postops.rhs_arg_offset = GET_OFF(post_ops_binary_rhs_arg_vec);
            postops.dst_orig_off = GET_OFF(data_C_ptr_);
            // A partial right-hand-side load reads the vector tail, or one
            // element for a scalar accumulator.
            postops.tail_elems = brg_.gemv_acc_is_vector() ? brg_.gemv_tail : 1;
        }

        ir::generate_kernel(*this, ir, postops);
    }

private:
    brgemm_desc_t brg_;
};

} // namespace

// Returns `status::success` if the descriptor is supported by the GEMV IR
// kernel, otherwise `status::unimplemented`.
status_t brgemv_ir_supported(const brgemm_desc_t &brg) {
    using namespace data_type;

    // Accepted input data type and the ISA it is built at:
    //   f32  - avx2
    //   bf16 - avx512_core_bf16
    //   f16  - avx512_core_fp16
    // GEMV blocking permits no other ISA (see `brgemm_blocking_vmm_gemv`).
    const bool dt_isa_ok = (brg.dt_a == f32 && brg.isa_impl == avx2)
            || (brg.dt_a == bf16 && brg.isa_impl == avx512_core_bf16)
            || (brg.dt_a == f16 && brg.isa_impl == avx512_core_fp16);
    VCONDCHECK_BRGEMV_IR(dt_isa_ok, VERBOSE_UNSUPPORTED_ISA);
    VCONDCHECK_BRGEMV_IR(brg.dt_b == brg.dt_a, VERBOSE_UNSUPPORTED_DT);
    // The kernel accumulates in f32 and has no conversion on the way out.
    VCONDCHECK_BRGEMV_IR(brg.dt_c == f32, VERBOSE_UNSUPPORTED_DT);
    VCONDCHECK_BRGEMV_IR(
            !brg.transA, VERBOSE_UNSUPPORTED_FEATURE, "transposed A");

    // With post-ops the kernel stores to D instead of C. It cannot convert the
    // accumulator on the way out, so it takes only `dt_d == dt_c`. The check is
    // unconditional because `dt_c != dt_d` alone already routes the descriptor
    // through the store-to-D path (see `are_post_ops_applicable()`), which the
    // `has_post_ops` below does not cover.
    VCONDCHECK_BRGEMV_IR(brg.dt_d == brg.dt_c, VERBOSE_UNSUPPORTED_DT);

    const bool has_post_ops = brg.with_bias || brg.with_eltwise
            || brg.with_binary || brg.with_sum || brg.with_src_scales
            || brg.with_wei_scales;
    VCONDCHECK_BRGEMV_IR(IMPLICATION(has_post_ops, brg.LDC == brg.LDD),
            VERBOSE_UNSUPPORTED_FEATURE, "LDC != LDD");

    // Post-ops go through the JIT injector. Accept eltwise, binary and sum, and
    // only binary arguments the injector can handle on this ISA. Other kinds
    // fall back.
    if (brg.attr()) {
        const memory_desc_wrapper dst_d(brg.dst_md());
        const std::vector<injector::post_op_type> accepted
                = {injector::eltwise, injector::binary, injector::sum};
        VCONDCHECK_BRGEMV_IR(
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
    VCONDCHECK_BRGEMV_IR(!brg.with_dst_scales, VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMV_IR(!brg.is_per_k_src_scales && !brg.is_per_k_wei_scales,
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMV_IR(
            IMPLICATION(brg.with_src_scales, brg.dt_src_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMV_IR(
            IMPLICATION(brg.with_wei_scales, brg.dt_wei_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMV_IR(!brg.req_s8s8_compensation,
            VERBOSE_UNSUPPORTED_FEATURE, "s8s8 compensation");
    VCONDCHECK_BRGEMV_IR(utils::everyone_is(brgemm_broadcast_t::none,
                                 brg.zp_type_a, brg.zp_type_b, brg.zp_type_c),
            VERBOSE_UNSUPPORTED_ZP_CFG);
    VCONDCHECK_BRGEMV_IR(
            !brg.with_bias || brg.dt_bias == f32, VERBOSE_UNSUPPORTED_BIAS_CFG);
    VCONDCHECK_BRGEMV_IR(
            brg.alpha == 1.0f, VERBOSE_UNSUPPORTED_FEATURE, "alpha != 1");
    VCONDCHECK_BRGEMV_IR(brg.beta == 0.0f || brg.beta == 1.0f,
            VERBOSE_UNSUPPORTED_FEATURE, "beta != 0 && beta != 1");

    VCONDCHECK_BRGEMV_IR(
            !brg.is_runtime_lda, VERBOSE_UNSUPPORTED_FEATURE, "runtime lda");
    VCONDCHECK_BRGEMV_IR(
            !brg.is_runtime_ldc, VERBOSE_UNSUPPORTED_FEATURE, "runtime ldc");

    const int m_block = brg.gemv_bd_block();
    const int dt_sz_a = brg.typesize_A;
    const int dt_sz_y = brg.typesize_C;
    const dim_t incy = brg.LDC;

    // Ensure indexed displacements fit in 32-bit
    auto fits = [](dim_t v) { return v <= INT32_MAX && v >= INT32_MIN; };

    VCONDCHECK_BRGEMV_IR(fits(dt_sz_a * (dim_t)m_block * brg.LDA),
            VERBOSE_UNSUPPORTED_FEATURE, "A block offset overflows int32");
    VCONDCHECK_BRGEMV_IR(fits(dt_sz_y * (dim_t)m_block * incy),
            VERBOSE_UNSUPPORTED_FEATURE, "y block offset overflows int32");

    return status::success;
}

brgemm_kernel_t *create_brgemv_ir_kernel(const brgemm_desc_t &brg) {
    if (brgemv_ir_supported(brg) != status::success) return nullptr;
    return new jit_brgemv_ir_kernel_t(brg);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
