/*******************************************************************************
* Copyright 2026 ZTE Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/pooling_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/jit_uni_pool_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;
using namespace alg_kind;

#define GET_OFF(field) static_cast<int>(offsetof(jit_uni_pooling_args_t, field))
// Parameter-struct offset helpers for the native kernels. Each references the
// local `using p_t/a_t/c_t = ...;` alias in scope at the call site, mirroring the
// x64/aarch64 GET_OFF convention instead of open-coding offsetof everywhere.
#define GET_OFF_P(field) static_cast<int>(offsetof(p_t, field))
#define GET_OFF_A(field) static_cast<int>(offsetof(a_t, field))
#define GET_OFF_C(field) static_cast<int>(offsetof(c_t, field))

// True when src1 is per-oc ([1,C,1,..] matching dst's channel dim, densely
// stored) — the rv64 binary injector reads it as a contiguous [c_block] run.
static bool binary_src1_is_per_oc(
        const memory_desc_wrapper &s1, const memory_desc_wrapper &dst_d) {
    if (dst_d.ndims() < 2 || s1.ndims() != dst_d.ndims()) return false;
    if (s1.dims()[1] != dst_d.dims()[1] || !s1.is_dense(true)) return false;
    for (int k = 0; k < dst_d.ndims(); k++)
        if (k != 1 && s1.dims()[k] != 1) return false;
    return true;
}

// Single source of truth for the fused-binary broadcast category. Both pd gates
// (baked post_ops_ok and native init_conf) classify with this; the result is
// stored in jpp.binary_bcast so downstream code (apply_postops, the native
// generate_* rhs load form, the driver's binary_off0) never re-derives it.
// Returns none for a src1 layout the kernels cannot fuse (the gate then rejects
// the chain to the reference).
static pool_binary_bcast_t classify_binary_src1(
        const memory_desc_wrapper &s1, const memory_desc_wrapper &dst_d) {
    if (s1.nelems(true) == 1) return pool_binary_bcast_t::scalar;
    if (binary_src1_is_per_oc(s1, dst_d)) return pool_binary_bcast_t::per_oc;
    if (s1.similar_to(dst_d, true, false)) return pool_binary_bcast_t::full_dst;
    return pool_binary_bcast_t::none;
}

// Map an effective channel-vector LMUL (1/2/4) to the xbyak_riscv LMUL enum.
static Xbyak_riscv::LMUL pool_lmul_to_enum(int lmul) {
    switch (lmul) {
        case 2: return Xbyak_riscv::LMUL::m2;
        case 4: return Xbyak_riscv::LMUL::m4;
        default: return Xbyak_riscv::LMUL::m1;
    }
}

// The strategies the pool kernels can position themselves (the injector runs
// in host-positioned byte-offset mode): scalar / per-oc(-spatial) / full-dst.
static bcast_set_t get_supported_postops_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::no_broadcast};
}

// Fill the shape/stride/kernel/padding fields shared verbatim by all three
// init_conf paths (baked, native forward, native backward). Channel bookkeeping
// (c / c_block / c_without_padding) differs per path and stays with the caller.
static void set_pool_spatial_dims(jit_pool_conf_t &jpp,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &dst_d,
        const pooling_desc_t &pd) {
    const int ndims = src_d.ndims();
    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];
    jpp.id = (ndims == 5) ? src_d.dims()[ndims - 3] : 1;
    jpp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[ndims - 3] : 1;
    jpp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jpp.ow = dst_d.dims()[ndims - 1];
    jpp.stride_d = (ndims == 5) ? pd.strides[0] : 1;
    jpp.stride_h = (ndims == 3) ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = (ndims == 3) ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];
    jpp.f_pad = (ndims == 5) ? pd.padding[0][0] : 0;
    jpp.t_pad = (ndims == 3) ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];
}

// Callee-saved GPRs in save order (s0..s11); their register numbers are not
// contiguous, so the sequence is spelled out for the preamble helpers below.
static const Reg pool_saved_gprs[]
        = {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11};

// Manual preamble/postamble: rv64 jit_generator_t emits no automatic frame, so
// every pool kernel saves/restores the s* registers it clobbers into consecutive
// 8-byte slots at the base of its frame. These fold the otherwise-repeated
// straight-line sd/ld runs; the frame allocation and any extra scratch slots
// (e.g. the baked kd-loop input/output spill) stay with each generate().
static void save_saved_gprs(jit_generator_t *h, int count) {
    for (int i = 0; i < count; i++)
        h->sd(pool_saved_gprs[i], sp, i * 8);
}
static void restore_saved_gprs(jit_generator_t *h, int count) {
    for (int i = 0; i < count; i++)
        h->ld(pool_saved_gprs[i], sp, i * 8);
}

// Extra frame scratch slots that sit above the s0..s11 save area (slots 0..88).
// The baked kernel spills reg_input/reg_output across the 3d kd-loop; the native
// forward kernels spill the n_pos loop counter.
static constexpr int kBakedSpillInput = 96;
static constexpr int kBakedSpillOutput = 104;
static constexpr int kNativePosSpill = 96;

template <cpu_isa_t isa>
jit_uni_pool_kernel_t<isa>::~jit_uni_pool_kernel_t() = default;

template <cpu_isa_t isa>
jit_uni_pool_kernel_t<isa>::jit_uni_pool_kernel_t(
        const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md)
    : jit_generator_t("jit_rvv_pool"), jpp(ajpp) {
    // dst_md is kept only for signature parity with the x64/aarch64 kernels,
    // which build the binary post-op injector in the ctor and need the dst
    // descriptor there. The rv64 kernel builds its injector in generate() from
    // jpp.post_ops instead, so it never reads dst_md.
    UNUSED(dst_md);
    // jpp.c_block is the base SIMD block (vlen/32) for blocked/f16 paths, or
    // base*LMUL for the widened f32 nspc forward-inference path (init_conf
    // scales it there). Recover the effective LMUL for tile register spacing.
    const int base = (int)(get_platform_vlen() / 32);
    lmul_ = (base > 0) ? jpp.c_block / base : 1;
    if (lmul_ != 1 && lmul_ != 2 && lmul_ != 4) lmul_ = 1;
}

static status_t set_binary_postops_formats(
        post_ops_t &post_ops, const memory_desc_t *dst_md) {
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        if (!post_ops.contain(primitive_kind::binary, idx)) continue;

        auto &src1_md = post_ops.entry_[idx].binary.src1_desc;
        const memory_desc_wrapper src1_mdw(src1_md);
        if (!src1_mdw.format_any()) {
            if (src1_mdw.is_blocking_desc())
                continue;
            else
                return status::unimplemented;
        }

        const memory_desc_wrapper dst_mdw(dst_md);
        assert(!dst_mdw.format_any());

        CHECK(memory_desc_init_by_blocking_desc(
                src1_md, dst_mdw.blocking_desc()));
    }
    return status::success;
}

template <cpu_isa_t isa>
bool jit_uni_pool_kernel_t<isa>::post_ops_ok(jit_pool_conf_t &jpp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;
    jpp.binary_bcast = pool_binary_bcast_t::none;

    if (jpp.is_backward) return post_ops.len() == 0;

    // Injector-level acceptance (x64 pattern): any number of forward eltwise
    // ops plus any number of binaries, no sum; the kernels supply 4 eltwise
    // aux groups.
    if (!injector::post_ops_ok(injector::post_ops_ok_args_t(isa,
                {injector::eltwise, injector::binary}, post_ops, &dst_d,
                false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
                true /*sum_requires_zp_zero*/,
                true /*sum_requires_same_params*/,
                get_supported_postops_bcast_strategies(), /*n_vaux=*/4)))
        return false;

    // Pool-specific residue. Binaries fuse at f32 (for f16 the accumulator is
    // already widened to f32 at the inject point, and the rhs is loaded as
    // f32 -- no narrow staging group is reserved). The kernels position every
    // binary in a chain at ONE shared byte offset, so all binaries must share
    // the same broadcast category (scalar / per-oc / full-dst); a mixed chain
    // would need per-entry offsets and is left to the reference.
    pool_binary_bcast_t bcast = pool_binary_bcast_t::none;
    for (const auto &e : post_ops.entry_) {
        if (e.is_eltwise()) {
            jpp.with_eltwise = true;
        } else if (e.is_binary()) {
            if (e.binary.src1_desc.data_type != data_type::f32) return false;
            const memory_desc_wrapper s1(e.binary.src1_desc);
            const pool_binary_bcast_t k = classify_binary_src1(s1, dst_d);
            if (k == pool_binary_bcast_t::none) return false;
            if (bcast != pool_binary_bcast_t::none && bcast != k)
                return false; // no mixing
            bcast = k;
            jpp.with_binary = true;
        } else
            return false;
    }

    jpp.binary_bcast = bcast;
    jpp.with_postops = jpp.with_eltwise || jpp.with_binary;
    return true;
}

template <cpu_isa_t isa>
status_t jit_uni_pool_kernel_t<isa>::init_conf(jit_pool_conf_t &jpp,
        memory_tracking::registrar_t &scratchpad, primitive_attr_t &attr,
        const pooling_pd_t *ppd) {
    using namespace format_tag;

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(
            ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
            ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());
    const int ndims = src_d.ndims();

    // One m1 register == c_block f32 lanes; the blocked tag matches this width.
    const uint32_t vlen = get_platform_vlen();
    if (vlen == 0) return status::unimplemented;
    int c_block = (int)(vlen / 32);
    if (c_block > 16) c_block = 16;
    if (c_block != 4 && c_block != 8 && c_block != 16)
        return status::unimplemented;

    dnnl_format_tag_t blocked_fmt_tag = dnnl_format_tag_undef;
    switch (c_block) {
        case 16:
            blocked_fmt_tag = utils::pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
            break;
        case 8:
            blocked_fmt_tag = utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
            break;
        case 4:
            blocked_fmt_tag = utils::pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
            break;
        default: return status::unimplemented;
    }
    const auto nspc_fmt_tag = utils::pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto fmt_tag
            = src_d.matches_one_of_tag(blocked_fmt_tag, nspc_fmt_tag);
    if (fmt_tag == format_tag::undef) return status::unimplemented;
    if (!dst_d.matches_tag(fmt_tag)) return status::unimplemented;

    jpp.tag_kind = (fmt_tag == nspc_fmt_tag) ? jit_pool_tag_kind_t::nspc
                                             : jit_pool_tag_kind_t::blocked;
    jpp.use_native = false; // x64/aarch64-style baked kernel

    jpp.nthr = dnnl_get_max_threads();
    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;
    set_pool_spatial_dims(jpp, src_d, dst_d, pd);
    jpp.c_without_padding = src_d.dims()[1];
    jpp.c_block = c_block;

    jpp.alg = pd.alg_kind;
    jpp.src_dt = ppd->is_fwd() ? src_d.data_type() : dst_d.data_type();
    jpp.dst_dt = ppd->is_fwd() ? dst_d.data_type() : src_d.data_type();
    jpp.is_f16 = (src_d.data_type() == data_type::f16);
    jpp.is_bf16 = (src_d.data_type() == data_type::bf16);
    jpp.dt_size = types::data_type_size(src_d.data_type());
    jpp.isa = isa;

    // This kernel accumulates diff_src in memory: one load-modify-store per
    // covering output window, so the running sum is rounded to the storage
    // dtype at every window. With bf16's 8-bit significand that drifts by 1 ulp
    // from the f32-accumulated reference, which pooling's exact-match check
    // rejects (seen on 3D blocked shapes, where the depth window forces one
    // round-trip per kd). Decline; the plain layouts already take the native
    // gather kernel, which keeps one f32 accumulator per input and stores once.
    if (jpp.is_bf16 && jpp.is_backward) return status::unimplemented;

    // Blocked with padded channels (is_c_padded) is left to the reference: the
    // RVV tail is a vl reduction, which cleanly covers nspc but not the padded
    // blocked store/postop lanes.
    jpp.is_c_padded = jpp.tag_kind == jit_pool_tag_kind_t::blocked
            && (jpp.c_without_padding % c_block != 0);
    if (jpp.is_c_padded) return status::unimplemented;

    jpp.c = jpp.c_without_padding;
    jpp.nb_c = utils::div_up(jpp.c, c_block);
    jpp.c_tail = jpp.c % c_block; // nonzero only for nspc

    const int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    const int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    const int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);
    if (jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
            || back_pad >= jpp.kd || bottom_pad >= jpp.kh
            || right_pad >= jpp.kw)
        return status::unimplemented;

    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;

    jpp.simple_alg = jpp.is_training
            || IMPLICATION(jpp.is_backward, jpp.kd <= jpp.stride_d);

    // Output-width unroll (ur_bc is always 1 on RVV). Capped so the per-shift
    // register tiles fit in the 24-register tile space (v8..v31): max forward
    // and both avg use acc+input (2 shifts) -> 12; max training/backward add an
    // index tile (3 shifts) -> 8 / 6.
    if (jpp.alg == pooling_max) {
        jpp.ur = jpp.is_training ? 8 : (jpp.is_backward ? 6 : 12);
    } else {
        jpp.ur = 12; // acc + input, both fwd and bwd
    }
    if (jpp.ur > jpp.ow) jpp.ur = jpp.ow;
    if (jpp.ur < 1) jpp.ur = 1;
    jpp.ur_bc = 1;
    jpp.ur_bc_tail = 0;

    if (!post_ops_ok(jpp, attr, dst_d)) return status::unimplemented;
    if (ppd->is_fwd() && jpp.with_binary)
        CHECK(set_binary_postops_formats(attr.post_ops_, dst_d.md_));
    jpp.post_ops = attr.post_ops_;

    // Widen the channel vector for f32 nspc forward inference to amortise the
    // per-channel-block call count, the cost that made the plain baked kernel
    // slower than the native full-C kernel.
    if (jpp.tag_kind == jit_pool_tag_kind_t::nspc && !jpp.is_f16 && !jpp.is_bf16
            && !jpp.is_backward && !jpp.is_training && !jpp.with_postops) {
        const int base = (int)(vlen / 32);
        int lmul = (base > 0) ? 16 / base : 1;
        if (lmul != 1 && lmul != 2 && lmul != 4) lmul = 1;
        if (lmul > 1) {
            jpp.c_block *= lmul;
            jpp.nb_c = utils::div_up(jpp.c, jpp.c_block);
            jpp.c_tail = jpp.c % jpp.c_block;
            const int max_ur
                    = tile_size / 2 / lmul; // 2*ur*lmul <= tile_size (24)
            if (jpp.ur > max_ur) jpp.ur = max_ur;
        }
    }

    UNUSED(scratchpad); // no plain<->blocked transpose on RVV
    return status::success;
}

// ---- small emit helpers --------------------------------------------------

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::load_f32_const(
        const FReg &f, float v, const Reg &gpr) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    li(gpr, (int64_t)(int32_t)bits);
    fmv_w_x(f, gpr);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::addr_off(
        const Reg &dst, const Reg &base, int off) {
    if (off >= -2048 && off <= 2047) {
        addi(dst, base, off);
    } else {
        // Materialize the offset in a distinct scratch (t2): several callers use
        // dst == base (pointer self-advance), where `li(dst, off)` would clobber
        // base before the add. t2 is never a base argument here.
        li(t2, off);
        add(dst, base, t2);
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::beqz_far(const Reg &r, Label &t) {
    Label cont;
    bnez(r, cont);
    j_(t);
    L(cont);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::bnez_far(const Reg &r, Label &t) {
    Label cont;
    beqz(r, cont);
    j_(t);
    L(cont);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::blt_far(const Reg &a, const Reg &b, Label &t) {
    Label cont;
    bge(a, b, cont);
    j_(t);
    L(cont);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::set_vl_e32() {
    // Tail-agnostic (VTA::ta): the tail (lanes >= vl) is never stored or read
    // back — every store here is vl-limited — so preserving it is only a false
    // dependency. Mask-undisturbed (VMA::mu) is REQUIRED, not optional here:
    // max_step_bwd() runs a masked vfadd_vv under this vtype and then stores the
    // whole vector, so the masked-off (non-argmax) lanes must keep their loaded
    // diff_src value. All other pool vsetvli sites use VMA::ma. LMUL follows the
    // effective channel-vector width (lmul_); the f32 nspc fwd path widens it.
    vsetvli(t0, reg_vl, SEW::e32, pool_lmul_to_enum(lmul_), VTA::ta, VMA::mu);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::load(int idx, const Reg &reg_ptr, int offset) {
    addr_off(t1, reg_ptr, offset);
    if (jpp.is_f16 || jpp.is_bf16) {
        // The widening convert takes the NARROW (source) vtype: run it under
        // e16/mf2 so it widens xf16->f32 (dest e32/m1), not f32->f64. Restore
        // e32 after. bf16 uses the Zvfbfmin convert, f16 the Zvfh one.
        vsetvli(t0, reg_vl, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
        vle16_v(v_tmp, t1);
        if (jpp.is_bf16)
            vfwcvtbf16_f_f_v(vreg(idx), v_tmp);
        else
            vfwcvt_f_f_v(vreg(idx), v_tmp);
        set_vl_e32();
    } else {
        vle32_v(vreg(idx), t1);
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::store(
        int idx, const Reg &reg_ptr, int offset) {
    addr_off(t1, reg_ptr, offset);
    if (jpp.is_f16 || jpp.is_bf16) {
        vsetvli(t0, reg_vl, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
        if (jpp.is_bf16)
            vfncvtbf16_f_f_w(v_tmp, vreg(idx));
        else
            vfncvt_f_f_w(v_tmp, vreg(idx));
        vse16_v(v_tmp, t1);
        set_vl_e32();
    } else {
        vse32_v(vreg(idx), t1);
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::load_indices(
        int idx, const Reg &reg_ptr, int offset) {
    addr_off(t1, reg_ptr, offset);
    if (jpp.ind_dt == data_type::u8) {
        vsetvli(t0, reg_vl, SEW::e8, LMUL::mf4, VTA::ta, VMA::ma);
        vle8_v(v_tmp, t1);
        set_vl_e32();
        vzext_vf4(vreg(idx), v_tmp);
    } else {
        vle32_v(vreg(idx), t1);
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::store_indices(
        int idx, const Reg &reg_ptr, int offset) {
    addr_off(t1, reg_ptr, offset);
    if (jpp.ind_dt == data_type::u8) {
        // values < 256: narrow e32 -> e16 -> e8 (truncation is exact).
        vsetvli(t0, reg_vl, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
        vnsrl_wi(v_tmp, vreg(idx), 0);
        vsetvli(t0, reg_vl, SEW::e8, LMUL::mf4, VTA::ta, VMA::ma);
        vnsrl_wi(v_tmp, v_tmp, 0);
        vse8_v(v_tmp, t1);
        set_vl_e32();
    } else {
        vse32_v(vreg(idx), t1);
    }
}

// ---- avg -----------------------------------------------------------------

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::maybe_recalculate_divisor(
        int jj, int ur_w, int pad_l, int pad_r) {
    if (jpp.alg != pooling_avg_exclude_padding) return;
    int non_zero_kw = jpp.kw;
    non_zero_kw -= nstl::max(0, pad_l - jj * jpp.stride_w);
    non_zero_kw -= nstl::max(0, pad_r - (ur_w - 1 - jj) * jpp.stride_w);
    if (non_zero_kw == prev_kw) return;
    prev_kw = non_zero_kw;
    // f_tmp = non_zero_kw * ker_area_h  (the exclude-padding divisor)
    li(t2, non_zero_kw);
    fcvt_s_w(f_tmp, t2);
    fmul_s(f_tmp, f_tmp, f_area);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::avg_step(int ur_w, int pad_l, int pad_r) {
    const int iw = jpp.iw;
    const int kw = jpp.kw;
    const int stride_w = jpp.stride_w;
    const int c_off
            = (jpp.tag_kind == jit_pool_tag_kind_t::nspc) ? jpp.c : jpp.c_block;
    const int dt = jpp.dt_size;

    // include-padding divisor is the constant window volume; exclude-padding
    // divisor is recomputed per output column in maybe_recalculate_divisor().
    if (jpp.alg == pooling_avg_include_padding)
        load_f32_const(f_tmp, (float)(jpp.kw * jpp.kh * jpp.kd), t2);

    // Initialise accumulators: backward loads output/divisor; forward zeroes.
    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward) maybe_recalculate_divisor(jj, ur_w, pad_l, pad_r);
        const int accr = reg_ind(0, jj, ur_w);
        if (jpp.is_backward) {
            load(accr, reg_output, dt * jj * c_off);
            vfdiv_vf(vreg(accr), vreg(accr), f_tmp); // acc = out / divisor
        } else {
            vmv_v_x(vreg(accr), x0); // 0.0f
        }
    }

    Label kd_label;
    const bool kd_loop = jpp.simple_alg && jpp.ndims == 5;
    if (kd_loop) {
        sd(reg_input, sp, kBakedSpillInput);
        sd(reg_output, sp, kBakedSpillOutput);
        mv(aux_reg_input_d, reg_input);
        ld(reg_kd, reg_param, GET_OFF(kd_padding));
        L(kd_label);
        mv(aux_reg_input, aux_reg_input_d);
    } else {
        mv(aux_reg_input, reg_input);
    }

    mv(t5, reg_kh); // kj = kh_padding
    Label kh_label;
    L(kh_label);
    for (int ki = 0; ki < kw; ki++) {
        const int jj_start = utils::div_up(nstl::max(0, pad_l - ki), stride_w);
        const int jj_end = ur_w
                - utils::div_up(nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
        for (int jj = jj_start; jj < jj_end; jj++) {
            const int aux_input_offset = (ki + jj * stride_w - pad_l) * c_off;
            if (aux_input_offset >= iw * c_off) continue;
            const int accr = reg_ind(0, jj, ur_w);
            const int inpr = reg_ind(1, jj, ur_w);
            if (jpp.is_backward) {
                load(inpr, aux_reg_input, dt * aux_input_offset);
                vfadd_vv(vreg(inpr), vreg(inpr), vreg(accr));
                store(inpr, aux_reg_input, dt * aux_input_offset);
            } else {
                load(inpr, aux_reg_input, dt * aux_input_offset);
                vfadd_vv(vreg(accr), vreg(accr), vreg(inpr));
            }
        }
    }
    addr_off(aux_reg_input, aux_reg_input, dt * iw * c_off);
    addi(t5, t5, -1);
    bnez_far(t5, kh_label);

    if (kd_loop) {
        addr_off(aux_reg_input_d, aux_reg_input_d, dt * jpp.ih * iw * c_off);
        addi(reg_kd, reg_kd, -1);
        bnez_far(reg_kd, kd_label);
        ld(reg_input, sp, kBakedSpillInput);
        ld(reg_output, sp, kBakedSpillOutput);
    }

    if (!jpp.is_backward) {
        for (int jj = 0; jj < ur_w; jj++) {
            maybe_recalculate_divisor(jj, ur_w, pad_l, pad_r);
            const int accr = reg_ind(0, jj, ur_w);
            vfdiv_vf(vreg(accr), vreg(accr), f_tmp); // acc /= divisor
        }
        if (jpp.with_postops) apply_postops(ur_w, c_off);
        for (int jj = 0; jj < ur_w; jj++)
            store(reg_ind(0, jj, ur_w), reg_output, dt * jj * c_off);
    }
}

// ---- max forward ---------------------------------------------------------

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::max_step_fwd(int ur_w, int pad_l, int pad_r) {
    const int iw = jpp.iw;
    const int kw = jpp.kw;
    const int stride_w = jpp.stride_w;
    const int c_off
            = (jpp.tag_kind == jit_pool_tag_kind_t::nspc) ? jpp.c : jpp.c_block;
    const int dt = jpp.dt_size;

    // Init accumulators to the dtype lowest (the xf16 lowest for f16/bf16 so an
    // all-pad window narrows exactly); zero the index accumulators for training.
    const float init_val = jpp.is_f16
            ? -65504.0f
            : (jpp.is_bf16 ? (float)nstl::numeric_limits<bfloat16_t>::lowest()
                           : nstl::numeric_limits<float>::lowest());
    load_f32_const(f_tmp, init_val, t2);
    for (int jj = 0; jj < ur_w; jj++) {
        vfmv_v_f(vreg(reg_ind(0, jj, ur_w)), f_tmp);
        if (jpp.is_training) vmv_v_x(vreg(reg_ind(2, jj, ur_w)), x0);
    }
    if (jpp.is_training) vmv_v_x(v_k_offset, reg_k_shift);

    Label kd_label;
    if (jpp.ndims == 5) {
        sd(reg_input, sp, kBakedSpillInput);
        sd(reg_output, sp, kBakedSpillOutput);
        mv(aux_reg_input_d, reg_input);
        ld(reg_kd, reg_param, GET_OFF(kd_padding));
        L(kd_label);
        mv(aux_reg_input, aux_reg_input_d);
    } else {
        mv(aux_reg_input, reg_input);
    }

    mv(t5, reg_kh);
    Label kh_label;
    L(kh_label);
    for (int ki = 0; ki < kw; ki++) {
        const int jj_start = utils::div_up(nstl::max(0, pad_l - ki), stride_w);
        const int jj_end = ur_w
                - utils::div_up(nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
        for (int jj = jj_start; jj < jj_end; jj++) {
            const int aux_input_offset = (ki + jj * stride_w - pad_l) * c_off;
            if (aux_input_offset >= iw * c_off) continue;
            const int accr = reg_ind(0, jj, ur_w);
            const int inpr = reg_ind(1, jj, ur_w);
            load(inpr, aux_reg_input, dt * aux_input_offset);
            vmflt_vv(v_mask, vreg(accr), vreg(inpr)); // acc < inp
            vmerge_vvm(vreg(accr), vreg(accr), vreg(inpr)); // acc = max
            if (jpp.is_training) {
                const int indr = reg_ind(2, jj, ur_w);
                vmerge_vvm(vreg(indr), vreg(indr), v_k_offset);
            }
        }
        if (jpp.is_training) vadd_vv(v_k_offset, v_k_offset, v_one);
    }
    addr_off(aux_reg_input, aux_reg_input, dt * iw * c_off);
    addi(t5, t5, -1);
    bnez_far(t5, kh_label);

    if (jpp.ndims == 5) {
        addr_off(aux_reg_input_d, aux_reg_input_d, dt * jpp.ih * iw * c_off);
        if (jpp.is_training) {
            ld(t2, reg_param, GET_OFF(kd_padding_shift));
            vadd_vx(v_k_offset, v_k_offset, t2);
        }
        addi(reg_kd, reg_kd, -1);
        bnez_far(reg_kd, kd_label);
        ld(reg_input, sp, kBakedSpillInput);
        ld(reg_output, sp, kBakedSpillOutput);
    }

    if (jpp.with_postops) apply_postops(ur_w, c_off);
    for (int jj = 0; jj < ur_w; jj++) {
        store(reg_ind(0, jj, ur_w), reg_output, dt * jj * c_off);
        if (jpp.is_training)
            store_indices(reg_ind(2, jj, ur_w), reg_index,
                    (jj * c_off) * (int)types::data_type_size(jpp.ind_dt));
    }
}

// ---- max backward --------------------------------------------------------

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::max_step_bwd(int ur_w, int pad_l, int pad_r) {
    const int iw = jpp.iw;
    const int kw = jpp.kw;
    const int stride_w = jpp.stride_w;
    const int c_off
            = (jpp.tag_kind == jit_pool_tag_kind_t::nspc) ? jpp.c : jpp.c_block;
    const int dt = jpp.dt_size;

    for (int jj = 0; jj < ur_w; jj++) {
        load(reg_ind(0, jj, ur_w), reg_output, dt * jj * c_off); // diff_dst
        load_indices(reg_ind(1, jj, ur_w), reg_index,
                (jj * c_off) * (int)types::data_type_size(jpp.ind_dt));
    }
    vmv_v_x(v_k_offset, reg_k_shift);

    Label kd_label;
    const bool kd_loop = jpp.simple_alg && jpp.ndims == 5;
    if (kd_loop) {
        sd(reg_input, sp, kBakedSpillInput);
        sd(reg_output, sp, kBakedSpillOutput);
        mv(aux_reg_input_d, reg_input);
        ld(reg_kd, reg_param, GET_OFF(kd_padding));
        ld(reg_kd_pad_shift, reg_param, GET_OFF(kd_padding_shift));
        L(kd_label);
        mv(aux_reg_input, aux_reg_input_d);
    } else {
        mv(aux_reg_input, reg_input);
    }

    mv(t5, reg_kh);
    Label kh_label;
    L(kh_label);
    for (int ki = 0; ki < kw; ki++) {
        const int jj_start = utils::div_up(nstl::max(0, pad_l - ki), stride_w);
        const int jj_end = ur_w
                - utils::div_up(nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
        for (int jj = jj_start; jj < jj_end; jj++) {
            const int aux_input_offset = (ki + jj * stride_w - pad_l) * c_off;
            if (aux_input_offset >= iw * c_off) continue;
            const int outr = reg_ind(0, jj, ur_w);
            const int indr = reg_ind(1, jj, ur_w);
            const int inpr = reg_ind(2, jj, ur_w);
            load(inpr, aux_reg_input, dt * aux_input_offset); // diff_src
            vmseq_vv(v_mask, vreg(indr), v_k_offset); // index == k_offset
            // diff_src += (mask ? diff_dst : 0); the VMA::mu set in set_vl_e32()
            // keeps the masked-off (non-argmax) lanes at their loaded diff_src.
            vfadd_vv(vreg(inpr), vreg(inpr), vreg(outr), VM::masked);
            store(inpr, aux_reg_input, dt * aux_input_offset);
        }
        vadd_vv(v_k_offset, v_k_offset, v_one);
    }
    addr_off(aux_reg_input, aux_reg_input, dt * iw * c_off);
    addi(t5, t5, -1);
    bnez_far(t5, kh_label);

    if (kd_loop) {
        addr_off(aux_reg_input_d, aux_reg_input_d, dt * jpp.ih * iw * c_off);
        vadd_vx(v_k_offset, v_k_offset, reg_kd_pad_shift);
        addi(reg_kd, reg_kd, -1);
        bnez_far(reg_kd, kd_label);
        ld(reg_input, sp, kBakedSpillInput);
        ld(reg_output, sp, kBakedSpillOutput);
    }
}

// ---- backward diff_src zeroing (simple_alg) ------------------------------

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::zero_diff_src() {
    const int c_off
            = (jpp.tag_kind == jit_pool_tag_kind_t::nspc) ? jpp.c : jpp.c_block;
    const int dt = jpp.dt_size;
    const int width = jpp.iw * c_off; // elements per input row for this block

    Label l_skip, id_label, ih_label;
    // reg registers reused locally: this runs before the window loop uses them.
    const Reg reg_zero_ptr = aux_reg_input; // s7
    const Reg reg_zero_id = reg_kd; // s9
    const Reg reg_zero_ih = t5;
    const Reg aux_ptr = t3;
    const Reg aux_ih = t4;

    vmv_v_x(v_tmp, x0); // zero vector (f32); an xf16 store narrows zero anyway

    ld(reg_zero_ptr, reg_param, GET_OFF(zero_ptr));
    ld(reg_zero_id, reg_param, GET_OFF(zero_id));
    beqz_far(reg_zero_id, l_skip);
    ld(reg_zero_ih, reg_param, GET_OFF(zero_ih));
    beqz_far(reg_zero_ih, l_skip);

    L(id_label);
    mv(aux_ptr, reg_zero_ptr);
    mv(aux_ih, reg_zero_ih);
    L(ih_label);
    // Step by the column stride (c_off): for nspc, consecutive iw positions are
    // C apart and each stores c_block (=vl) channels; for blocked c_off==c_block
    // (contiguous). Either way this zeroes iw columns * c_block channels.
    for (int i = 0; i < width; i += c_off) {
        if (jpp.is_f16 || jpp.is_bf16) {
            vsetvli(t0, reg_vl, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
            vmv_v_x(v_tmp, x0);
            addr_off(t1, aux_ptr, dt * i);
            vse16_v(v_tmp, t1);
            set_vl_e32();
        } else {
            addr_off(t1, aux_ptr, dt * i);
            vse32_v(v_tmp, t1);
        }
    }
    addr_off(aux_ptr, aux_ptr, dt * width);
    addi(aux_ih, aux_ih, -1);
    bnez_far(aux_ih, ih_label);

    addr_off(reg_zero_ptr, reg_zero_ptr, dt * width * jpp.ih);
    addi(reg_zero_id, reg_zero_id, -1);
    bnez_far(reg_zero_id, id_label);
    L(l_skip);
}

// ---- post-ops ------------------------------------------------------------

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::apply_postops(int ur_w, int c_off) {
    const int dt = jpp.dt_size;
    const int rhs_es = (int)sizeof(float); // binary rhs (src1) is always f32

    // Fused-binary broadcast (uniform across the chain), classified once in the
    // pd gate and read back here: 0 = eltwise-only, 1 = scalar, 2 = per-oc,
    // 3 = full-dst (see pool_binary_bcast_t).
    const int bcast = static_cast<int>(jpp.binary_bcast);

    // Position the shared rhs BYTE offset (reg t2) for the injector. src1 is f32,
    // so the offset is (element index) * sizeof(f32) irrespective of the dst
    // dtype — for f16 the accumulator is widened to f32 before this point, and
    // the binary is applied at f32.
    if (bcast == 1) {
        li(t2, 0);
    } else if (bcast == 2) { // per-oc: element (b_c * c_block) of the [C] rhs
        li(t3, jpp.c_block * rhs_es);
        mul(t2, reg_bc, t3);
    } else if (bcast == 3) {
        // full-dst: element index of this call's output base = dst_byte_off / dt
        // (dst element size); the per-column term is added below.
        ld(reg_kd_pad_shift, reg_param, GET_OFF(dst_orig));
        sub(t5, reg_output, reg_kd_pad_shift); // dst byte offset
        srli(t5, t5, (dt == 2) ? 1 : 2); // -> element index
    }

    for (int jj = 0; jj < ur_w; jj++) {
        const int acc = vreg(reg_ind(0, jj, ur_w)).getIdx();
        if (bcast == 3) {
            addr_off(t2, t5, jj * c_off); // element index of output column jj
            slli(t2, t2, 2); // * sizeof(f32)
        }
        // dynamic-params: t2 is the (byte) rhs offset for this column. The
        // baked kernel computes at e32/m1 (group_stride 1).
        binary_injector::rhs_arg_dynamic_params_t rd;
        rd.vmm_idx_to_out_reg.emplace(acc, t2);
        postops_injector_->compute_vector(acc, rd, 1 /*group_stride*/);
    }
}

// ---- generate ------------------------------------------------------------

template <cpu_isa_t isa>
void jit_uni_pool_kernel_t<isa>::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const int c_off
            = (jpp.tag_kind == jit_pool_tag_kind_t::nspc) ? jpp.c : jpp.c_block;
    const int dt = jpp.dt_size;
    const int ind_sz = jpp.ind_dt != data_type::undef
            ? (int)types::data_type_size(jpp.ind_dt)
            : 0;
    const bool has_index
            = jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward);

    // Preamble: save s0..s11 (+ two spill slots for the 3d kd-loop at 96/104).
    const int stack = 112;
    addi(sp, sp, -stack);
    save_saved_gprs(this, 12);

    // backward: src=diff_src (accumulated into), dst=diff_dst (read).
    ld(reg_input, reg_param, GET_OFF(src));
    ld(reg_output, reg_param, GET_OFF(dst));
    if (has_index) ld(reg_index, reg_param, GET_OFF(indices));
    ld(reg_kh, reg_param, GET_OFF(kh_padding));
    ld(reg_k_shift, reg_param, GET_OFF(kh_padding_shift));
    flw(f_area, reg_param, GET_OFF(ker_area_h));
    ld(reg_bc, reg_param, GET_OFF(b_c));

    // Channel AVL for this block: c_block, or c_tail for the nspc last block.
    li(reg_vl, jpp.c_block);
    if (jpp.c_tail != 0) {
        Label skip_tail;
        li(t2, jpp.nb_c - 1);
        bne(reg_bc, t2, skip_tail);
        li(reg_vl, jpp.c_tail);
        L(skip_tail);
    }
    set_vl_e32();

    if (has_index) {
        li(t2, 1);
        vmv_v_x(v_one, t2); // index increment
    }

    if (jpp.with_postops) {
        // v_tmp is dead after pooling accumulation and can serve as the fourth
        // eltwise aux at the post-op inject point.
        eltwise_injector::static_params_t esp(v_eltw0, v_eltw1, v_eltw2, v_tmp,
                v_tmp, f_eltw0, f_eltw1, t4, /*is_fwd=*/true);
        // Host-positioned injector mode (null dst_d): the kernel computes the
        // per-column byte offset itself, so the rhs classifies as scalar /
        // no_broadcast from the rhs element count. rhs is f32 (no gather/narrow).
        // The rhs pointer array is read through the args pointer (x64 model);
        // preserve_gpr shields the injector's fixed X_TMP scratch (t4 is this
        // kernel's aux_ih and t5 the bcast==3 base).
        binary_injector::rhs_arg_static_params_t rhs_arg_bsp(v_bin_rhs.getIdx(),
                t3, t2, t2, true /*preserve gpr*/, false /*preserve vmm*/,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(nullptr));
        rhs_arg_bsp.rhs_dt_helper_freg = f_bin;
        rhs_arg_bsp.off_is_bytes = true; // t2 is the per-column byte offset
        const binary_injector::static_params_t bsp(reg_param, rhs_arg_bsp);
        postops_injector_
                = utils::make_unique<injector::jit_uni_postops_injector_t<isa>>(
                        this, jpp.post_ops, bsp, esp);
    }

    prev_kw = 0;
    if (jpp.is_backward && jpp.simple_alg) zero_diff_src();

    const int ur_w = jpp.ur;
    const int ow = jpp.ow;
    const int iw = jpp.iw;
    const int kw = jpp.kw;
    const int stride_w = jpp.stride_w;
    const int l_pad = jpp.l_pad;

    auto process_oi = [&](int cur_ur_w, int lpad, int rpad) {
        step(cur_ur_w, lpad, rpad);
        addr_off(reg_input, reg_input,
                dt * nstl::max(0, cur_ur_w * stride_w - lpad) * c_off);
        addr_off(reg_output, reg_output, dt * cur_ur_w * c_off);
        if (has_index)
            addr_off(reg_index, reg_index, cur_ur_w * c_off * ind_sz);
    };

    const int n_oi = utils::div_up(ow, ur_w);
    const int ur_stride_w = ur_w * stride_w;
    const int l_pad_iters = nstl::min(n_oi, utils::div_up(l_pad, ur_stride_w));

    for (int i = 0; i < l_pad_iters; ++i) {
        const int ow_s = i * ur_w;
        const int ow_e = nstl::min(ow, ow_s + ur_w);
        const int cur_l_pad = l_pad - i * ur_stride_w;
        const int cur_r_pad = nstl::max(
                0, calculate_end_padding(l_pad, ow_e, iw, stride_w, kw));
        process_oi(ow_e - ow_s, cur_l_pad, cur_r_pad);
    }

    const int rem_n_oi = n_oi - l_pad_iters;
    const int cur_iw = l_pad_iters * ur_stride_w - l_pad;
    const int cur_iw_rightmost = cur_iw + kw - 1;
    const int no_pad_full = utils::saturate<int>(
            0, rem_n_oi, (iw - cur_iw_rightmost) / ur_stride_w);

    if (no_pad_full > 0) {
        Label ow_loop;
        if (no_pad_full > 1) li(t6, 0);
        L(ow_loop);
        process_oi(ur_w, 0, 0);
        if (no_pad_full > 1) {
            addi(t6, t6, 1);
            li(t5, no_pad_full);
            blt_far(t6, t5, ow_loop);
        }
    }

    for (int i = l_pad_iters + no_pad_full; i < n_oi; ++i) {
        const int ow_s = i * ur_w;
        const int ow_e = nstl::min(ow, ow_s + ur_w);
        const int cur_r_pad = nstl::max(
                0, calculate_end_padding(l_pad, ow_e, iw, stride_w, kw));
        process_oi(ow_e - ow_s, 0, cur_r_pad);
    }

    restore_saved_gprs(this, 12);
    addi(sp, sp, stack);
    ret();
#else
    ret();
#endif
}

template struct jit_uni_pool_kernel_t<v>;
template struct jit_uni_pool_kernel_t<zvfh>;
template struct jit_uni_pool_kernel_t<zvfbfwma>;

// ================== native VLA forward kernel (nspc/ncsp) ==================

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pool_ncsp_kernel_t<isa, d_type>::jit_uni_pool_ncsp_kernel_t(
        const jit_pool_conf_t &jpp)
    : jit_generator_t(jpp.alg == alg_kind::pooling_max ? "jit_rvv_pool_fwd_max"
                      : jpp.alg == alg_kind::pooling_avg_include_padding
                      ? "jit_rvv_pool_fwd_avg_inc"
                      : "jit_rvv_pool_fwd_avg_exc")
    , jpp_(jpp)
    , is_max_pool_(jpp.alg == alg_kind::pooling_max) {
    // create_kernel() is called (and CHECK'd) by the primitive's init(); a
    // codegen failure must propagate a status, not be swallowed here.
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_pool_ncsp_kernel_t<isa, d_type>::init_conf(
        jit_pool_conf_t &jpp, primitive_attr_t &attr, const pooling_pd_t *ppd) {
    using namespace alg_kind;
    using namespace format_tag;

    const memory_desc_wrapper src_d(ppd->src_md());
    const memory_desc_wrapper dst_d(ppd->dst_md());
    const int ndims = src_d.ndims();
    const auto &pd = *ppd->desc();

    set_pool_spatial_dims(jpp, src_d, dst_d, pd);
    jpp.c = src_d.dims()[1];
    jpp.c_without_padding = jpp.c;

    jpp.alg = pd.alg_kind;
    jpp.is_backward = false;
    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    // Max forward-training tracks the argmax into this workspace (u8/s32).
    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;
    jpp.src_dt = src_d.data_type();
    jpp.dst_dt = dst_d.data_type();
    jpp.dt_size = types::data_type_size(jpp.src_dt);
    jpp.is_f16 = d_type == data_type::f16;
    jpp.is_bf16 = d_type == data_type::bf16;
    jpp.isa = isa;
    jpp.nthr = dnnl_get_max_threads();
    jpp.ur_w = 4;

    // Native path: nspc or ncsp (plain layouts). blocked -> baked kernel.
    const auto nspc_tag = utils::pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto ncsp_tag = utils::pick(ndims - 3, ncw, nchw, ncdhw);
    if (src_d.matches_tag(nspc_tag) && dst_d.matches_tag(nspc_tag))
        jpp.tag_kind = jit_pool_tag_kind_t::nspc;
    else if (src_d.matches_tag(ncsp_tag) && dst_d.matches_tag(ncsp_tag))
        jpp.tag_kind = jit_pool_tag_kind_t::ncsp;
    else
        return status::unimplemented;
    jpp.use_native = true;

    // f32 nspc forward inference routes to the baked kernel. f16 nspc, ncsp,
    // and f32 nspc training stay on this native kernel.
    if (d_type == data_type::f32 && jpp.tag_kind == jit_pool_tag_kind_t::nspc
            && !jpp.is_training)
        return status::unimplemented;

    // f32 nspc builds the shape-baked interior kernel, which fully unrolls the
    // width sweep over max_p = (ur_w-1)*sw + kw input positions. Its per-channel
    // loop body must stay within a RISC-V B-type branch's ±4 KiB reach; a very
    // large window (e.g. a global-style kw) would overrun it and fail codegen.
    // Decline such shapes here so dispatch falls through to the baked kernel /
    // reference instead of building an interior kernel that cannot be generated.
    // Realistic pooling windows are far below this bound (f16/ncsp never build
    // the interior kernel, so they are unaffected).
    if (d_type == data_type::f32 && jpp.tag_kind == jit_pool_tag_kind_t::nspc) {
        constexpr int max_interior_positions = 64;
        const int max_p = (jpp.ur_w - 1) * jpp.stride_w + jpp.kw;
        if (max_p > max_interior_positions) return status::unimplemented;
    }

    // Post-ops fused in-kernel via the post-op injector: any eltwise chain plus
    // any number of binaries (indirect injector mode; they share one broadcast
    // category), for both f32 and f16 (f16 widens the accumulator to f32 before
    // the chain; the rhs is always f32). Anything else is rejected below to
    // ref_pooling. with_relu
    // (f32 single-ReLU) is separate — it only feeds the empty-window fill value
    // (empty_window_value()), not the fusion.
    const auto &po = attr.post_ops_;
    jpp.post_ops = po;
    jpp.with_postops = !po.has_default_values();
    // Max forward-training generates the argmax workspace. Native handles both
    // plain layouts for f32 (generate_f32, e32 index) and f16 (generate_f16, e16
    // index); the index store is contiguous for nspc and strided for ncsp via
    // ws_vec_byte_stride. A max-training chain may also fuse post-ops (the injector
    // runs after the argmax is computed). f16 tracks the argmax in e16, so guard
    // the (never-hit-by-real-pooling) window volume that would overflow a signed
    // 16-bit index.
    if (jpp.alg == alg_kind::pooling_max && jpp.is_training
            && utils::one_of(d_type, data_type::f16, data_type::bf16)
            && (int64_t)jpp.kd * jpp.kh * jpp.kw > 32768)
        return status::unimplemented;
    jpp.with_relu = false;
    jpp.binary_bcast = pool_binary_bcast_t::none;
    jpp.relu_alpha = 0.f;
    // fuse_eltwise / fuse_binary are assigned unconditionally below.
    if (po.len() == 1) {
        const auto &e = po.entry_[0];
        // with_relu is an f32-only flag feeding empty_window_value(); it is
        // not the fusion mechanism. f16 eltwise (incl ReLU) fuses via
        // fuse_eltwise in generate_f16.
        if (e.is_eltwise() && e.eltwise.alg == eltwise_relu
                && d_type == data_type::f32) {
            jpp.with_relu = true;
            jpp.relu_alpha = e.eltwise.alpha;
        }
    }
    // Post-op chains (any number of eltwise + any number of binaries, in
    // attribute order) are fused in-kernel via the post-op injector, for both f32
    // (generate_f32) and f16 (generate_f16, computed at f32). A chain that
    // contains the binary is routed by the driver through the channel-vectorized
    // single-position path so the rhs broadcast is uniform; the injector loads
    // the rhs (f32) per channel chunk. post_ops_ok already gated the chain.
    const bool inj_ok = injector::post_ops_ok(injector::post_ops_ok_args_t(isa,
            {injector::eltwise, injector::binary}, po, &dst_d,
            false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
            true /*sum_requires_zp_zero*/, true /*sum_requires_same_params*/,
            get_supported_postops_bcast_strategies(), /*n_vaux=*/4));
    bool po_has_binary = false;
    for (int i = 0; i < po.len(); i++)
        if (po.entry_[i].is_binary()) po_has_binary = true;
    jpp.fuse_eltwise = jpp.with_postops && inj_ok && !po_has_binary;
    // Binary fuses at f32 for both f32 and f16 dst (f16 widens the accumulator to
    // f32 before applying the chain). The rhs is always f32.
    jpp.fuse_binary = jpp.with_postops && inj_ok && po_has_binary;

    // Reject post-op chains this kernel cannot fully fuse so dispatch falls to
    // the reference instead of silently dropping the post-op. The injector runs
    // in indirect mode, so ANY number of binaries is allowed, but (like the baked
    // kernel) they must share ONE broadcast category — the kernel positions a
    // single shared rhs offset. src1 must be f32; broadcast in {scalar, per-oc,
    // full-dst}. A mixed-broadcast chain goes to the reference.
    if (jpp.with_postops) {
        if (!inj_ok) return status::unimplemented;
        pool_binary_bcast_t bcast = pool_binary_bcast_t::none;
        for (int i = 0; i < po.len(); i++) {
            if (!po.entry_[i].is_binary()) continue;
            const auto &b = po.entry_[i].binary;
            if (b.src1_desc.data_type != data_type::f32)
                return status::unimplemented;
            const memory_desc_wrapper s1(b.src1_desc);
            const pool_binary_bcast_t k = classify_binary_src1(s1, dst_d);
            if (k == pool_binary_bcast_t::none)
                return status::unimplemented; // unsupported src1 layout
            if (bcast != pool_binary_bcast_t::none && bcast != k)
                return status::unimplemented; // mixed broadcast -> reference
            bcast = k;
        }
        jpp.binary_bcast = bcast;
    }
    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_ncsp_kernel_t<isa, d_type>::generate() {
    if (utils::one_of(d_type, data_type::f16, data_type::bf16))
        generate_xf16();
    else
        generate_f32();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_ncsp_kernel_t<isa, d_type>::generate_f32() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const VReg v_mask(0);
    const VReg v_acc(4), v_tmp(8);
    // Max forward-training (f32): track the per-channel argmax in v_ind while the
    // window is swept, and store it to the workspace. t3 holds the running
    // window index (reset to pos_base per channel chunk), advanced +1 per iw and
    // by pos_ih_step/pos_id_step (from args, reloaded via a0) at row/plane ends.
    const bool max_train = is_max_pool_ && jpp_.is_training;
    // Max-training may also fuse post-ops (the injector runs on v_acc after the
    // argmax is computed). When a binary is fused the ws pointer keeps s10 and the
    // rhs origin array moves to a4, s11 carries the shared binary offset, and
    // pos_base is reloaded per chunk (v_ind stays v28, clear of the injector).
    const bool mt_bin = max_train && jpp_.fuse_binary;
    const VReg v_ind(28);
    const bool ind_u8 = jpp_.ind_dt == data_type::u8;
    const int ind_sz = ind_u8 ? 1 : 4; // workspace index element size (u8/s32)

    // Fused-binary rhs load form, from the broadcast category classified once in
    // init_conf (the driver routes binary through the channel-vectorized
    // single-position path, so the lanes are channels): scalar -> flw; per-oc
    // [1,C,1,..] -> [C]-contiguous vle; full-dst -> per-channel strided vlse
    // (src1 follows the dst layout).
    const bool bin_scalar = jpp_.binary_bcast == pool_binary_bcast_t::scalar;
    const bool bin_strided = jpp_.binary_bcast == pool_binary_bcast_t::full_dst;

    // Save callee-saved regs (s0-s11) holding loop invariants across the nested
    // id/ih/iw/channel loops. Without a fused binary or max-training, an outer
    // loop processes n_pos output positions per call (the ncsp driver batches
    // interior output rows, which share the window shape, so only the base
    // pointers advance): s10/s11 hold the src/dst position bases and the
    // counter is spilled. Otherwise a single position is processed and s10/s11
    // carry the binary/workspace state.
    const bool use_pos_loop = !jpp_.fuse_binary && !max_train;
    const int stack_size = use_pos_loop ? 112 : 96;
    addi(sp, sp, -stack_size);
    save_saved_gprs(this, 12);

    // Load params from jit_uni_pool_ncsp_args_t
    using p_t = jit_uni_pool_ncsp_args_t;
    ld(s0, reg_param, GET_OFF_P(src));
    ld(s1, reg_param, GET_OFF_P(dst));
    ld(s2, reg_param, GET_OFF_P(channels));
    ld(s6, reg_param, GET_OFF_P(id_start));
    ld(s7, reg_param, GET_OFF_P(ih_start));
    ld(a1, reg_param, GET_OFF_P(iw_start));
    ld(t5, reg_param, GET_OFF_P(id_end));
    ld(t6, reg_param, GET_OFF_P(ih_end));
    ld(a2, reg_param, GET_OFF_P(iw_end));
    ld(t2, reg_param, GET_OFF_P(inW_stride));
    ld(t3, reg_param, GET_OFF_P(inD_stride));
    ld(s3, reg_param, GET_OFF_P(w_spatial_byte_stride));
    flw(fa0, reg_param, GET_OFF_P(init_val));
    flw(ft1, reg_param, GET_OFF_P(scale_val));

    fmv_w_x(fa1, x0); // f_zero = 0.0

    // Compute byte strides
    slli(s4, t2, 2); // inW_stride_bytes = inW_stride * 4
    slli(s5, t3, 2); // inD_stride_bytes = inD_stride * 4

    ld(s8, reg_param, GET_OFF_P(src_vec_byte_stride));
    ld(s9, reg_param, GET_OFF_P(dst_vec_byte_stride));
    if (jpp_.fuse_binary && !max_train) {
        // s11 = shared byte offset (advanced per channel chunk); the injector
        // reads the rhs pointer array through the args pointer (x64 model).
        ld(s11, reg_param, GET_OFF_P(post_op_off0));
    }
    if (max_train) {
        ld(s10, reg_param, GET_OFF_P(indices));
        // With a fused binary, s11 carries the shared rhs offset and pos_base is
        // reloaded per chunk; otherwise s11 holds pos_base directly.
        if (mt_bin)
            ld(s11, reg_param, GET_OFF_P(post_op_off0));
        else
            ld(s11, reg_param, GET_OFF_P(pos_base));
    }
    if (use_pos_loop) {
        mv(s10, s0);
        mv(s11, s1);
        ld(t0, reg_param, GET_OFF_P(n_pos));
        sd(t0, sp, kNativePosSpill);
    }

    li(t4, 4); // constant for unit-stride comparison

    Label pos_loop, pos_done;
    if (use_pos_loop) {
        L(pos_loop);
        ld(t0, sp, kNativePosSpill);
        beqz(t0, pos_done);

        mv(s0, s10);
        mv(s1, s11);
        ld(s2, reg_param, GET_OFF_P(channels));
    }

    // Channel loop: process channels in vector chunks (one output position).
    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(s2, ch_done);

    // Set vector length for remaining channels
    vsetvli(t0, s2, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);

    // Initialize accumulator
    if (is_max_pool_) {
        // init_val = -FLT_MAX for max pool
        vfmv_v_f(v_acc, fa0);
    } else {
        vfmv_v_f(v_acc, fa1); // zero
    }
    if (max_train) {
        vmv_v_x(v_ind, x0); // argmax index accumulator = 0
        if (mt_bin) {
            // s11 is the binary offset; reload pos_base (a5 is free before the
            // window sweep, which then reuses it as the iw counter).
            ld(a5, reg_param, GET_OFF_P(pos_base));
            mv(t3, a5);
        } else
            mv(t3, s11); // running window index = pos_base (reset per chunk)
    }

    // Depth (id) loop
    mv(a3, s6); // reg_id = id_start
    mul(t1, a3, s5);
    add(t1, s0, t1); // t1 = depth_ptr = src + id * inD_stride_bytes

    Label id_loop, id_done;
    L(id_loop);
    bge(a3, t5, id_done); // if id >= id_end, done

    // Height (ih) loop
    mv(a4, s7); // reg_ih = ih_start
    mul(t2, a4, s4);
    add(a6, t1, t2); // a6 = row_ptr

    Label ih_loop, ih_done;
    L(ih_loop);
    bge(a4, t6, ih_done); // if ih >= ih_end, done

    // Width (iw) loop
    mv(a5, a1); // reg_iw = iw_start
    mul(t2, a5, s3);
    add(a7, a6, t2); // a7 = src_ptr

    Label iw_loop, iw_done;
    L(iw_loop);
    bge(a5, a2, iw_done); // if iw >= iw_end, done

    // Load — use vle32_v for unit stride to avoid potential uarch penalty
    {
        Label strided_src, src_ld_done;
        bne(s8, t4, strided_src);
        vle32_v(v_tmp, a7);
        j_(src_ld_done);
        L(strided_src);
        vlse32_v(v_tmp, a7, s8);
        L(src_ld_done);
    }
    if (is_max_pool_) {
        vmflt_vv(v_mask, v_acc, v_tmp);
        vmerge_vvm(v_acc, v_acc, v_tmp);
        if (max_train) {
            vmerge_vxm(v_ind, v_ind, t3); // ind = (acc < tmp) ? pos : ind
            addi(t3, t3, 1); // advance window position (every iw)
        }
    } else {
        vfadd_vv(v_acc, v_acc, v_tmp);
    }

    addi(a5, a5, 1); // iw++
    add(a7, a7, s3); // advance src_ptr by w_spatial_byte_stride
    j_(iw_loop);
    L(iw_done);
    // Skip the clamped-off kw positions so the window index stays full-kernel
    // relative (pos_ih_step = KW - kw_count; reloaded via a0).
    if (max_train) {
        ld(t2, reg_param, GET_OFF_P(pos_ih_step));
        add(t3, t3, t2);
    }

    addi(a4, a4, 1); // ih++
    add(a6, a6, s4); // advance row_ptr by inW_stride_bytes
    j_(ih_loop);
    L(ih_done);
    if (max_train) { // skip clamped-off kh rows (pos_id_step = KH*KW - kh_cnt*KW)
        ld(t2, reg_param, GET_OFF_P(pos_id_step));
        add(t3, t3, t2);
    }

    addi(a3, a3, 1); // id++
    add(t1, t1, s5); // advance depth_ptr by inD_stride_bytes
    j_(id_loop);
    L(id_done);

    // Apply avg pooling divide
    if (!is_max_pool_) { vfmul_vf(v_acc, v_acc, ft1); }

    // Apply the fused post-op chain (any number of eltwise + any number of
    // binaries, in attribute order) via the in-kernel injector. Eltwise covers
    // ReLU and the other supported algs; binary loads each rhs base from the
    // pointer array and applies the shared offset in s11.
    if (jpp_.fuse_eltwise || jpp_.fuse_binary) {
        // v24 is also the binary rhs scratch, but the entries execute serially,
        // so it is free as the fourth eltwise aux during an eltwise entry.
        eltwise_injector::static_params_t esp(VReg(12), VReg(16), VReg(20),
                VReg(24), VReg(24), fa3, fa4, t2, /*is_fwd=*/true);
        // Indirect injector mode (any number of binaries): the rhs pointer
        // array is read through the args pointer (x64 model; max-training
        // needs no dedicated array-base register anymore), s11 = shared byte
        // offset, a3 = per-binary scratch. Binary rhs scratch v24/fa5 (clear
        // of v_ind = v28). full-dst uses a strided (vlse) load keyed on the
        // dst channel stride (s9 == the f32 rhs channel stride);
        // per-oc/scalar the contiguous form.
        // Host-positioned injector mode (null dst_d): the kernel maintains
        // the shared byte offset itself; preserve_gpr shields the injector's
        // fixed X_TMP scratch (t4 holds the live unit-stride comparison
        // constant here).
        binary_injector::rhs_arg_static_params_t rhs_arg_bsp(24, a3, s11, s11,
                true /*preserve gpr*/, false /*preserve vmm*/,
                GET_OFF_P(post_op_rhs), memory_desc_wrapper(nullptr));
        rhs_arg_bsp.rhs_dt_helper_freg = fa5;
        rhs_arg_bsp.off_is_bytes = true;
        if (bin_strided) {
            rhs_arg_bsp.is_strided = true;
            rhs_arg_bsp.rhs_stride = s9;
        }
        const binary_injector::static_params_t bsp(reg_param, rhs_arg_bsp);
        injector::jit_uni_postops_injector_t<isa> po_inj(
                this, jpp_.post_ops, bsp, esp);
        // dynamic-params: s11 is the per-chunk byte offset (off_is_bytes).
        binary_injector::rhs_arg_dynamic_params_t rhs_dyn;
        rhs_dyn.vmm_idx_to_out_reg.emplace(v_acc.getIdx(), s11);
        // ncsp channel-vec computes at e32/m1 (group_stride 1).
        po_inj.compute_vector(v_acc.getIdx(), rhs_dyn, 1 /*group_stride*/);
    }
    // Store result — use vse32_v for unit stride
    {
        Label strided_dst, dst_st_done;
        bne(s9, t4, strided_dst);
        vse32_v(v_acc, s1);
        j_(dst_st_done);
        L(strided_dst);
        vsse32_v(v_acc, s1, s9);
        L(dst_st_done);
    }
    if (max_train) {
        // Store the per-channel argmax to the workspace at s10. nspc: contiguous
        // over channels (ws_vec_byte_stride == ind_sz) -> unit store. ncsp:
        // channels are dst_spatial apart -> strided store (vsse). u8: narrow
        // e32->e16->e8 (indices < 256); s32: direct. Uses t3 as the vsetvli
        // scratch (free after the window sweep) so t0 (the channel vl) survives
        // for the pointer advance below.
        ld(t2, reg_param, GET_OFF_P(ws_vec_byte_stride));
        li(t1, static_cast<int>(ind_sz));
        if (ind_u8) {
            vsetvli(t3, s2, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
            vnsrl_wi(v_tmp, v_ind, 0);
            vsetvli(t3, s2, SEW::e8, LMUL::mf4, VTA::ta, VMA::ma);
            vnsrl_wi(v_tmp, v_tmp, 0);
            Label u8_unit, u8_done;
            beq(t2, t1, u8_unit);
            vsse8_v(v_tmp, s10, t2);
            j_(u8_done);
            L(u8_unit);
            vse8_v(v_tmp, s10);
            L(u8_done);
            vsetvli(t3, s2, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        } else {
            Label s32_unit, s32_done;
            beq(t2, t1, s32_unit);
            vsse32_v(v_ind, s10, t2);
            j_(s32_done);
            L(s32_unit);
            vse32_v(v_ind, s10);
            L(s32_done);
        }
    }

    // Advance src/dst pointers by vl * stride
    {
        Label strided_src_adv, src_adv_done;
        bne(s8, t4, strided_src_adv);
        slli(t1, t0, 2);
        j_(src_adv_done);
        L(strided_src_adv);
        mul(t1, t0, s8);
        L(src_adv_done);
    }
    add(s0, s0, t1);
    {
        Label strided_dst_adv, dst_adv_done;
        bne(s9, t4, strided_dst_adv);
        slli(t1, t0, 2);
        j_(dst_adv_done);
        L(strided_dst_adv);
        mul(t1, t0, s9);
        L(dst_adv_done);
    }
    add(s1, s1, t1);
    if (jpp_.fuse_binary && !bin_scalar) {
        // advance the shared rhs offset by vl * per-channel byte stride (full-dst:
        // dst channel stride s9; per-oc: contiguous f32 element size). The origin
        // array (s10) is fixed; the injector reads array[arg_idx] + s11.
        if (bin_strided)
            mul(t1, t0, s9);
        else
            slli(t1, t0, 2);
        add(s11, s11, t1);
    }
    if (max_train) { // advance the workspace ptr by vl * ws_vec_byte_stride
        ld(t2, reg_param, GET_OFF_P(ws_vec_byte_stride));
        mul(t1, t0, t2);
        add(s10, s10, t1);
    }
    sub(s2, s2, t0); // channels -= vl

    j_(ch_loop);
    L(ch_done);

    if (use_pos_loop) {
        ld(t0, reg_param, GET_OFF_P(pos_src_byte_stride));
        add(s10, s10, t0);
        ld(t0, reg_param, GET_OFF_P(pos_dst_byte_stride));
        add(s11, s11, t0);
        ld(t0, sp, kNativePosSpill);
        addi(t0, t0, -1);
        sd(t0, sp, kNativePosSpill);
        j_(pos_loop);
        L(pos_done);
    }

    // Restore callee-saved regs
    restore_saved_gprs(this, 12);
    addi(sp, sp, stack_size);

    ret();
#else
    ret();
#endif
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_ncsp_kernel_t<isa, d_type>::generate_xf16() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const VReg v_mask(0);
    // Nspc f16 max uses e16/m2 at VLEN=256 to process twice as many contiguous
    // channels. Other layouts and VLENs retain e16/m1: the wider group regressed
    // on tested VLEN=128 and VLEN=1024 hardware.
    // Avg and bf16 retain f32/m2 + e16/m1.
    const VReg v_acc(4), v_tmp(8);
    // bf16 has no e16 arithmetic, so its max compares at f32 like avg already
    // does: every load is widened into v_wide and the accumulator stays f32/m2.
    // v24 is otherwise only the f16-max post-op widen buffer, which the bf16
    // path does not need (its accumulator is already f32), so it is free here.
    constexpr bool is_bf16 = d_type == data_type::bf16;
    const bool use_wide_f16 = !is_bf16 && is_max_pool_
            && jpp_.tag_kind == jit_pool_tag_kind_t::nspc
            && get_platform_vlen() == 256;
    // Keep the channel count unchanged across widening and narrowing.
    const int f32_group_stride = use_wide_f16 ? 4 : 2;
    const auto f32_lmul = pool_lmul_to_enum(f32_group_stride);
    const auto f16_lmul = pool_lmul_to_enum(f32_group_stride / 2);
    const auto u8_lmul = use_wide_f16 ? LMUL::m1 : LMUL::mf2;
    const VReg v_wide(24);
    const bool acc_is_f32 = !is_max_pool_ || is_bf16;
    const VReg v_res = (is_max_pool_ && !is_bf16) ? v_acc : v_tmp;
    // Max forward-training tracks the per-channel argmax. The index fits e16 for
    // any realistic pooling window (< 32768), so v_ind uses the same e16 LMUL
    // as the data and avoids a per-window-element vtype switch. It is narrowed
    // to e8 (u8 ws) or widened to e32 in v28 (s32 ws) only at store.
    const bool max_train = is_max_pool_ && jpp_.is_training;
    // Max-training may also fuse post-ops (applied to the widened f32 max before
    // the narrow/store). When a binary is fused the ws pointer keeps s10, the rhs
    // origin array moves to a4, the f32 rhs stride to a6, and pos_base is reloaded
    // per chunk; s11 carries the shared binary offset.
    const bool mt_bin = max_train && jpp_.fuse_binary;
    // v_ind stays clear of the max-postop widen buffer (v24) and the binary rhs
    // scratch (v28) so the argmax survives a fused post-op chain.
    const VReg v_ind(10);
    const bool ind_u8 = jpp_.ind_dt == data_type::u8;
    const int ind_sz = ind_u8 ? 1 : 4;

    const int stack_size = 112;
    addi(sp, sp, -stack_size);
    save_saved_gprs(this, 12);

    using p_t = jit_uni_pool_ncsp_args_t;
    ld(s0, reg_param, GET_OFF_P(src));
    ld(s1, reg_param, GET_OFF_P(dst));
    ld(s2, reg_param, GET_OFF_P(channels));
    ld(s6, reg_param, GET_OFF_P(id_start));
    ld(s7, reg_param, GET_OFF_P(ih_start));
    ld(a1, reg_param, GET_OFF_P(iw_start));
    ld(t5, reg_param, GET_OFF_P(id_end));
    ld(t6, reg_param, GET_OFF_P(ih_end));
    ld(a2, reg_param, GET_OFF_P(iw_end));
    ld(t2, reg_param, GET_OFF_P(inW_stride));
    ld(t3, reg_param, GET_OFF_P(inD_stride));
    ld(s3, reg_param, GET_OFF_P(w_spatial_byte_stride));
    flw(ft1, reg_param, GET_OFF_P(scale_val));

    // f16 element strides: inW_stride / inD_stride are element counts -> * 2.
    slli(s4, t2, 1);
    slli(s5, t3, 1);

    ld(s8, reg_param, GET_OFF_P(src_vec_byte_stride));
    ld(s9, reg_param, GET_OFF_P(dst_vec_byte_stride));

    li(t4, 2); // unit-stride comparison constant (f16 = 2 bytes)

    // Fused-binary rhs load form, from the category classified once in init_conf
    // (f16 computes the chain at f32; the rhs is always f32): scalar -> flw;
    // per-oc [1,C,1,..] -> [C]-contiguous vle; full-dst -> per-channel strided
    // vlse. The full-dst rhs channel stride is in f32 units = dst f16 channel
    // stride (s9) * 2.
    const bool bin_scalar = jpp_.binary_bcast == pool_binary_bcast_t::scalar;
    const bool bin_strided = jpp_.binary_bcast == pool_binary_bcast_t::full_dst;

    if (max_train) {
        // s10 = argmax workspace ptr. n_pos == 1, so no position loop. With a
        // fused binary, s11 carries the shared rhs offset and pos_base is reloaded
        // per chunk (the rhs origin array a4 and f32 rhs stride a6 are set at the
        // inject point); otherwise s11 holds pos_base directly.
        ld(s10, reg_param, GET_OFF_P(indices));
        if (mt_bin)
            ld(s11, reg_param, GET_OFF_P(post_op_off0));
        else
            ld(s11, reg_param, GET_OFF_P(pos_base));
    } else if (jpp_.fuse_binary) {
        // fuse_binary => the driver routes each output column through the
        // single-position path (n_pos == 1), so no position loop is needed.
        // Indirect injector mode: the rhs pointer array is read through the
        // args pointer (x64 model), s11 = shared byte offset, t3 = f32 rhs
        // channel stride for full-dst (= 2 * the f16 dst channel stride s9;
        // the rhs is f32 while the dst is f16). t3 held inD_stride only until
        // s5 was derived above, so it is free here.
        ld(s11, reg_param, GET_OFF_P(post_op_off0));
        slli(t3, s9, 1);
    } else {
        mv(s10, s0);
        mv(s11, s1);
        ld(t0, reg_param, GET_OFF_P(n_pos));
        sd(t0, sp, kNativePosSpill);
    }

    // In-kernel post-op injector (f16 computes the chain at f32). t1 is a scratch
    // GPR reused across window iterations, free at the inject point. The binary
    // rhs scratch is v28/fa5 (f32); v24 is reserved for the max widen buffer.
    post_ops_t no_po;
    // v28 is the binary rhs scratch and is temporally free for eltwise entries.
    eltwise_injector::static_params_t esp(VReg(12), VReg(16), VReg(20),
            VReg(28), VReg(28), fa3, fa4, t1, /*is_fwd=*/true);
    // Indirect mode (any number of binaries): the rhs pointer array is read
    // through the args pointer (x64 model; max-training needs no dedicated
    // array-base register anymore), s11 = shared byte offset, a3 = per-binary
    // scratch. full-dst uses a strided (vlse) f32 load keyed on the f32 rhs
    // channel stride (t3, or a6 when max-training); per-oc/scalar vle.
    const Reg reg_stride = mt_bin ? a6 : t3;
    // Host-positioned injector mode (null dst_d): the kernel maintains the
    // shared byte offset itself; preserve_gpr shields the injector's fixed
    // X_TMP scratch (t4 holds the live unit-stride comparison constant here).
    binary_injector::rhs_arg_static_params_t rhs_arg_bsp(28, a3, s11, s11,
            true /*preserve gpr*/, false /*preserve vmm*/,
            GET_OFF_P(post_op_rhs), memory_desc_wrapper(nullptr));
    rhs_arg_bsp.rhs_dt_helper_freg = fa5;
    rhs_arg_bsp.off_is_bytes = true;
    if (bin_strided) {
        rhs_arg_bsp.is_strided = true;
        rhs_arg_bsp.rhs_stride = reg_stride;
    }
    const binary_injector::static_params_t bsp(reg_param, rhs_arg_bsp);
    injector::jit_uni_postops_injector_t<isa> po_inj(this,
            (jpp_.fuse_eltwise || jpp_.fuse_binary) ? jpp_.post_ops : no_po,
            bsp, esp);
    // dynamic-params: s11 is the per-chunk byte offset (off_is_bytes). Both the
    // avg (v_acc) and max (v24) post-op registers use the same offset.
    binary_injector::rhs_arg_dynamic_params_t rhs_dyn;
    rhs_dyn.vmm_idx_to_out_reg.emplace(v_acc.getIdx(), s11);
    rhs_dyn.vmm_idx_to_out_reg.emplace(24, s11);

    Label pos_loop, pos_done;
    if (!jpp_.fuse_binary && !max_train) {
        L(pos_loop);
        ld(t0, sp, kNativePosSpill);
        beqz(t0, pos_done);

        mv(s0, s10);
        mv(s1, s11);
    }
    ld(s2, reg_param, GET_OFF_P(channels));

    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(s2, ch_done);

    // Initialize accumulator. Use vmv.v.x with the raw bit pattern to avoid
    // f16 scalar NaN-boxing concerns.
    if (is_max_pool_) {
        if (is_bf16) {
            // f32 accumulator seeded with the lowest *bf16* value, so an
            // all-padding window narrows back exactly.
            vsetvli(t0, s2, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
            li(t1, 0xFF7F0000); // (float)bf16 lowest (-3.3895314e38)
            vmv_v_x(v_acc, t1);
            vsetvli(t0, s2, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        } else {
            vsetvli(t0, s2, SEW::e16, f16_lmul, VTA::ta, VMA::ma);
            li(t1, 0xFBFF); // f16 lowest (-65504.0)
            vmv_v_x(v_acc, t1);
        }
        if (max_train) {
            // Argmax index accumulator: e16, same LMUL as v_acc.
            vmv_v_x(v_ind, x0);
            if (mt_bin) {
                // s11 is the binary offset; reload pos_base (a5 is free before the
                // window sweep, which then reuses it as the iw counter).
                ld(a5, reg_param, GET_OFF_P(pos_base));
                mv(t3, a5);
            } else
                mv(t3, s11); // running window index = pos_base (reset per chunk)
        }
    } else {
        // avg: f32m2 accumulator; window runs under e16/m1 (same vl).
        vsetvli(t0, s2, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        vmv_v_x(v_acc, x0);
        vsetvli(t0, s2, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
    }

    mv(a3, s6);
    mul(t1, a3, s5);
    add(t1, s0, t1);
    Label id_loop, id_done;
    L(id_loop);
    bge(a3, t5, id_done);

    mv(a4, s7);
    mul(t2, a4, s4);
    add(a6, t1, t2);
    Label ih_loop, ih_done;
    L(ih_loop);
    bge(a4, t6, ih_done);

    mv(a5, a1);
    mul(t2, a5, s3);
    add(a7, a6, t2);
    Label iw_loop, iw_done;
    L(iw_loop);
    bge(a5, a2, iw_done);

    {
        Label strided_src, src_ld_done;
        bne(s8, t4, strided_src);
        vle16_v(v_tmp, a7);
        j_(src_ld_done);
        L(strided_src);
        vlse16_v(v_tmp, a7, s8);
        L(src_ld_done);
    }
    if (is_max_pool_ && !is_bf16) {
        vmflt_vv(v_mask, v_acc, v_tmp);
        vmerge_vvm(v_acc, v_acc, v_tmp);
        if (max_train) {
            vmerge_vxm(v_ind, v_ind, t3); // ind = (acc < tmp) ? cur_idx : ind
            addi(t3, t3, 1); // advance window position (every iw)
        }
    } else if (is_max_pool_) {
        // bf16 max: widen then compare at f32. e16/m1 and e32/m2 share a vl, so
        // the mask produced at e32 selects the same lanes when v_ind is merged
        // back under the e16 vtype.
        vfwcvtbf16_f_f_v(v_wide, v_tmp); // reads the e16 narrow vtype
        vsetvli(t0, s2, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        vmflt_vv(v_mask, v_acc, v_wide);
        vmerge_vvm(v_acc, v_acc, v_wide);
        vsetvli(t0, s2, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        if (max_train) {
            vmerge_vxm(v_ind, v_ind, t3);
            addi(t3, t3, 1);
        }
    } else if (is_bf16) {
        // bf16 avg: no widening add exists, so widen then add at f32.
        vfwcvtbf16_f_f_v(v_wide, v_tmp); // reads the e16 narrow vtype
        vsetvli(t0, s2, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        vfadd_vv(v_acc, v_acc, v_wide);
        vsetvli(t0, s2, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
    } else {
        // f32m2 += widen(f16m1), evaluated under the e16 vtype.
        vfwadd_wv(v_acc, v_acc, v_tmp);
    }

    addi(a5, a5, 1);
    add(a7, a7, s3);
    j_(iw_loop);
    L(iw_done);
    // Skip clamped-off kw positions so the window index stays full-kernel
    // relative (pos_ih_step = KW - kw_count).
    if (max_train) {
        ld(t2, reg_param, GET_OFF_P(pos_ih_step));
        add(t3, t3, t2);
    }

    addi(a4, a4, 1);
    add(a6, a6, s4);
    j_(ih_loop);
    L(ih_done);
    if (max_train) { // skip clamped-off kh rows (pos_id_step = KH*KW - kh*KW)
        ld(t2, reg_param, GET_OFF_P(pos_id_step));
        add(t3, t3, t2);
    }

    addi(a3, a3, 1);
    add(t1, t1, s5);
    j_(id_loop);
    L(id_done);

    // f32 accumulator (avg for both dtypes, and bf16 max): scale for avg,
    // apply the fused post-op chain at f32, then narrow to xf16. bf16 max needs
    // no widen/narrow round trip here because the compare already ran at f32.
    if (acc_is_f32) {
        vsetvli(t0, s2, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        if (!is_max_pool_) vfmul_vf(v_acc, v_acc, ft1);
        if (jpp_.fuse_eltwise || jpp_.fuse_binary) {
            if (is_max_pool_ && mt_bin) {
                // Max-training positions the rhs origin array / f32 rhs stride
                // at the inject point (s10 and t3 are the ws pointer and the
                // argmax scratch until the window sweep ends).
                ld(a4, reg_param, GET_OFF_P(post_op_rhs));
                slli(a6, s9, 1);
            }
            po_inj.compute_vector(v_acc.getIdx(), rhs_dyn, 2 /*group_stride*/);
        }
        vsetvli(t0, s2, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        if (is_bf16)
            vfncvtbf16_f_f_w(v_tmp, v_acc);
        else
            vfncvt_f_f_w(v_tmp, v_acc);
    } else if (jpp_.fuse_eltwise || jpp_.fuse_binary) {
        // max: widen the f16 accumulator to f32, apply the post-op
        // chain (eltwise and/or binary, rhs in v28), then narrow back in place
        // so the store path is unchanged. For max-training the f32 rhs stride
        // (a6) is positioned here (t3 holds the argmax scratch), free after
        // the window sweep.
        if (mt_bin) {
            slli(a6, s9, 1); // f32 rhs channel stride (= 2 * f16 dst stride)
        }
        vsetvli(t0, s2, SEW::e16, f16_lmul, VTA::ta, VMA::ma);
        vfwcvt_f_f_v(VReg(24), v_acc);
        vsetvli(t0, s2, SEW::e32, f32_lmul, VTA::ta, VMA::ma);
        po_inj.compute_vector(24, rhs_dyn, f32_group_stride);
        vsetvli(t0, s2, SEW::e16, f16_lmul, VTA::ta, VMA::ma);
        vfncvt_f_f_w(v_acc, VReg(24));
    }

    {
        Label strided_dst, dst_st_done;
        bne(s9, t4, strided_dst);
        vse16_v(v_res, s1);
        j_(dst_st_done);
        L(strided_dst);
        vsse16_v(v_res, s1, s9);
        L(dst_st_done);
    }
    if (max_train) {
        // Store the per-channel argmax to the workspace at s10. u8: narrow
        // e16->e8; s32: widen e16->e32 in v28. Unit (nspc,
        // ws_vec_byte_stride == ind_sz) vs strided (ncsp). t3 is the vsetvli
        // scratch (free after the window sweep) so t0 (the channel vl) survives.
        ld(t2, reg_param, GET_OFF_P(ws_vec_byte_stride));
        li(t1, static_cast<int>(ind_sz));
        if (ind_u8) {
            vsetvli(t3, s2, SEW::e8, u8_lmul, VTA::ta, VMA::ma);
            vnsrl_wi(v_tmp, v_ind, 0);
            Label u8_unit, u8_done;
            beq(t2, t1, u8_unit);
            vsse8_v(v_tmp, s10, t2);
            j_(u8_done);
            L(u8_unit);
            vse8_v(v_tmp, s10);
            L(u8_done);
        } else {
            vsetvli(t3, s2, SEW::e32, f32_lmul, VTA::ta, VMA::ma);
            vzext_vf2(v28, v_ind);
            Label s32_unit, s32_done;
            beq(t2, t1, s32_unit);
            vsse32_v(v28, s10, t2);
            j_(s32_done);
            L(s32_unit);
            vse32_v(v28, s10);
            L(s32_done);
        }
        vsetvli(t0, s2, SEW::e16, f16_lmul, VTA::ta, VMA::ma);
    }

    // Advance src/dst by vl * stride (f16 unit stride = vl * 2).
    {
        Label strided_src_adv, src_adv_done;
        bne(s8, t4, strided_src_adv);
        slli(t1, t0, 1);
        j_(src_adv_done);
        L(strided_src_adv);
        mul(t1, t0, s8);
        L(src_adv_done);
    }
    add(s0, s0, t1);
    {
        Label strided_dst_adv, dst_adv_done;
        bne(s9, t4, strided_dst_adv);
        slli(t1, t0, 1);
        j_(dst_adv_done);
        L(strided_dst_adv);
        mul(t1, t0, s9);
        L(dst_adv_done);
    }
    add(s1, s1, t1);
    if (jpp_.fuse_binary && !bin_scalar) {
        // advance the shared offset; the origin array is fixed. full-dst: vl *
        // f32 channel stride (2 * s9); per-oc: vl * sizeof(f32). Derived from s9
        // (t3 may hold the argmax scratch during max-training).
        if (bin_strided) {
            mul(t1, t0, s9);
            slli(t1, t1, 1);
        } else
            slli(t1, t0, 2);
        add(s11, s11, t1);
    }
    if (max_train) { // advance the workspace ptr by vl * ws_vec_byte_stride
        ld(t2, reg_param, GET_OFF_P(ws_vec_byte_stride));
        mul(t1, t0, t2);
        add(s10, s10, t1);
    }
    sub(s2, s2, t0);

    j_(ch_loop);
    L(ch_done);

    if (!jpp_.fuse_binary && !max_train) {
        ld(t0, reg_param, GET_OFF_P(pos_src_byte_stride));
        add(s10, s10, t0);
        ld(t0, reg_param, GET_OFF_P(pos_dst_byte_stride));
        add(s11, s11, t0);
        ld(t0, sp, kNativePosSpill);
        addi(t0, t0, -1);
        sd(t0, sp, kNativePosSpill);
        j_(pos_loop);
        L(pos_done);
    }

    restore_saved_gprs(this, 12);
    addi(sp, sp, stack_size);
    ret();
#else
    ret();
#endif
}

template struct jit_uni_pool_ncsp_kernel_t<v, data_type::f32>;
template struct jit_uni_pool_ncsp_kernel_t<zvfh, data_type::f16>;
template struct jit_uni_pool_ncsp_kernel_t<zvfbfwma, data_type::bf16>;

// === Shape-baked interior kernel (nspc, ur_w input reuse) ===

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pool_interior_kernel_t<isa, d_type>::jit_uni_pool_interior_kernel_t(
        const jit_pool_conf_t &ajpp)
    : jit_generator_t("jit_rvv_pool_interior"), jpp_(ajpp) {
    // create_kernel() is called (and CHECK'd) by the primitive's init(); a
    // codegen failure must propagate a status, not be swallowed here.
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_interior_kernel_t<isa, d_type>::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    generate_nspc();
#else
    ret();
#endif
}

#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_interior_kernel_t<isa, d_type>::generate_nspc() {
    using p_t = jit_uni_pool_interior_args_t;
    auto fbits = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };

    const bool is_max = jpp_.alg == alg_kind::pooling_max;
    const VReg v_mask(0);
    const bool is_avg_exclude
            = jpp_.alg == alg_kind::pooling_avg_exclude_padding;
    const int kw = jpp_.kw;
    const int sw = jpp_.stride_w;
    const int ur_w = jpp_.ur_w;
    const int max_p = (ur_w - 1) * sw + kw; // W positions in the unrolled sweep
    const float init_val = is_max ? -FLT_MAX : 0.0f;
    const float avg_inc_scale = 1.0f / (float)(jpp_.kd * jpp_.kh * jpp_.kw);

    const Reg reg_param = a0;
    const VReg v_tmp(4 + ur_w);
    auto acc = [&](int j) { return VReg(4 + j); };

    // Prologue: save s0-s11.
    const int stack = 96;
    addi(sp, sp, -stack);
    save_saved_gprs(this, 12);

    // Persistent state (survives the block/channel/window loops).
    ld(s0, reg_param, GET_OFF_P(src)); // block_base_src
    ld(s1, reg_param, GET_OFF_P(dst)); // block_base_dst
    ld(s2, reg_param, GET_OFF_P(channels)); // channels_orig
    ld(s3, reg_param, GET_OFF_P(kh_count));
    ld(s4, reg_param, GET_OFF_P(kd_count));
    ld(s5, reg_param, GET_OFF_P(w_stride));
    ld(s6, reg_param, GET_OFF_P(inW_stride));
    ld(s7, reg_param, GET_OFF_P(inD_stride));
    ld(s10, reg_param, GET_OFF_P(n_blocks));
    // Per-block base strides derived from w_stride: src advances ur_w*sw W
    // positions, dst advances ur_w output columns (dst column stride == w_stride
    // for nspc).
    li(t0, (uint32_t)(ur_w * sw));
    mul(s8, s5, t0);
    li(t0, (uint32_t)ur_w);
    mul(s9, s5, t0);

    // FP constants.
    li(t0, fbits(init_val));
    fmv_w_x(fa0, t0);
    fmv_w_x(fa3, x0); // zero
    if (is_avg_exclude) {
        flw(fa1, reg_param, GET_OFF_P(scale_val));
    } else if (!is_max) { // avg_include: scale is baked
        li(t0, fbits(avg_inc_scale));
        fmv_w_x(fa1, t0);
    }
    // In-kernel eltwise post-op injector for the shape-baked f32 nspc interior
    // kernel: the eltwise chain is applied when fuse_eltwise, else no_po keeps
    // the injector empty. A fused binary makes the driver skip this interior
    // kernel (n_blocks=0; every column goes through the channel-vec path), so
    // only eltwise chains reach here.
    post_ops_t no_po;
    eltwise_injector::static_params_t esp(VReg(12), VReg(16), VReg(20),
            VReg(24), VReg(24), fa4, fa5, t6, /*is_fwd=*/true);
    // The chain here is eltwise-only, so the (mandatory) binary static params
    // are never consumed; pass placeholder scratch.
    binary_injector::rhs_arg_static_params_t rhs_arg_bsp(24, x0, x0, x0,
            false /*preserve gpr*/, false /*preserve vmm*/,
            0 /*abi_param_offset*/, memory_desc_wrapper(nullptr));
    rhs_arg_bsp.rhs_dt_helper_freg = fa5;
    const binary_injector::static_params_t bsp(x0, rhs_arg_bsp);
    injector::jit_uni_postops_injector_t<isa> po_inj(
            this, jpp_.fuse_eltwise ? jpp_.post_ops : no_po, bsp, esp);
    binary_injector::rhs_arg_dynamic_params_t rhs_dyn; // eltwise-only here

    Label block_loop, block_done;
    L(block_loop);
    beqz(s10, block_done);

    mv(a4, s0); // working_src_base
    mv(a5, s1); // working_dst_base
    mv(a6, s2); // working channels

    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(a6, ch_done);
    vsetvli(t0, a6, SEW::e32, LMUL::m1, VTA::ta, VMA::ma); // vl in t0

    for (int j = 0; j < ur_w; j++)
        vfmv_v_f(acc(j), fa0);

    // Depth (kd) loop — runtime count.
    mv(t1, a4); // id_ptr
    mv(t2, s4); // id_cnt
    Label id_loop, id_done;
    L(id_loop);
    beqz(t2, id_done);

    // Height (kh) loop — runtime count.
    mv(t3, t1); // ih_ptr
    mv(t4, s3); // ih_cnt
    Label ih_loop, ih_done;
    L(ih_loop);
    beqz(t4, ih_done);

    // Width sweep — fully unrolled; each loaded input feeds every output column
    // whose window covers it (input reuse).
    mv(t5, t3); // w_ptr
    for (int p = 0; p < max_p; p++) {
        vle32_v(v_tmp, t5);
        for (int j = 0; j < ur_w; j++) {
            if (j * sw <= p && p < j * sw + kw) {
                if (is_max) {
                    vmflt_vv(v_mask, acc(j), v_tmp);
                    vmerge_vvm(acc(j), acc(j), v_tmp);
                } else
                    vfadd_vv(acc(j), acc(j), v_tmp);
            }
        }
        if (p + 1 < max_p) add(t5, t5, s5);
    }

    addi(t4, t4, -1);
    add(t3, t3, s6); // ih_ptr += inW_stride
    j_(ih_loop);
    L(ih_done);

    addi(t2, t2, -1);
    add(t1, t1, s7); // id_ptr += inD_stride
    j_(id_loop);
    L(id_done);

    // Scale (avg) + fused eltwise post-op chain + store for each of the ur_w
    // output columns.
    mv(a1, a5); // dst_ptr
    for (int j = 0; j < ur_w; j++) {
        if (!is_max) vfmul_vf(acc(j), acc(j), fa1);
        // nspc interior computes at e32/m1 (group_stride 1); eltwise-only here.
        if (jpp_.fuse_eltwise)
            po_inj.compute_vector(acc(j).getIdx(), rhs_dyn, 1 /*group_stride*/);
        vse32_v(acc(j), a1);
        if (j + 1 < ur_w) add(a1, a1, s5); // dst column stride == w_stride
    }

    // Advance to the next channel chunk.
    slli(a2, t0, 2); // vl * sizeof(float)
    add(a4, a4, a2);
    add(a5, a5, a2);
    sub(a6, a6, t0);
    j_(ch_loop);
    L(ch_done);

    // Advance to the next ur_w output block.
    add(s0, s0, s8);
    add(s1, s1, s9);
    addi(s10, s10, -1);
    j_(block_loop);
    L(block_done);

    // Epilogue.
    restore_saved_gprs(this, 12);
    addi(sp, sp, stack);
    ret();
}
#endif

template struct jit_uni_pool_interior_kernel_t<v, data_type::f32>;

// ===================== native gather backward kernel =======================
// One kernel call handles one input position: accumulate the diff_dst channel
// rows of the covering outputs (listed in jit_uni_pool_bwd_contrib_t) into the
// input's diff_src channel row and store it once. Channels are the vector dim
// (VLA); unit-stride for nspc, strided for ncsp. max adds diff_dst only where the
// stored argmax matches the contribution index; avg adds diff_dst * scale.

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pool_bwd_kernel_t<isa, d_type>::jit_uni_pool_bwd_kernel_t(
        const jit_pool_conf_t &jpp)
    : jit_generator_t(jpp.alg == alg_kind::pooling_max ? "jit_rvv_pool_bwd_max"
                                                       : "jit_rvv_pool_bwd_avg")
    , jpp_(jpp)
    , is_max_pool_(jpp.alg == alg_kind::pooling_max) {
    // create_kernel() is called (and CHECK'd) by the primitive's init(); a
    // codegen failure must propagate a status, not be swallowed here.
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_pool_bwd_kernel_t<isa, d_type>::init_conf(
        jit_pool_conf_t &jpp, const pooling_pd_t *ppd) {
    using namespace alg_kind;
    using namespace format_tag;

    const memory_desc_wrapper src_d(ppd->diff_src_md());
    const memory_desc_wrapper dst_d(ppd->diff_dst_md());
    const int ndims = src_d.ndims();
    const auto &pd = *ppd->desc();

    set_pool_spatial_dims(jpp, src_d, dst_d, pd);
    jpp.c = src_d.dims()[1];
    jpp.c_without_padding = jpp.c;

    jpp.alg = pd.alg_kind;
    jpp.is_backward = true;
    jpp.is_training = false;
    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;
    jpp.src_dt = src_d.data_type();
    jpp.dst_dt = dst_d.data_type();
    jpp.dt_size = types::data_type_size(jpp.src_dt);
    jpp.is_f16 = d_type == data_type::f16;
    jpp.is_bf16 = d_type == data_type::bf16;
    jpp.isa = isa;
    jpp.nthr = dnnl_get_max_threads();

    // Native path: nspc or ncsp (plain layouts). blocked -> baked kernel.
    const auto nspc_tag = utils::pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto ncsp_tag = utils::pick(ndims - 3, ncw, nchw, ncdhw);
    if (src_d.matches_tag(nspc_tag) && dst_d.matches_tag(nspc_tag))
        jpp.tag_kind = jit_pool_tag_kind_t::nspc;
    else if (src_d.matches_tag(ncsp_tag) && dst_d.matches_tag(ncsp_tag))
        jpp.tag_kind = jit_pool_tag_kind_t::ncsp;
    else
        return status::unimplemented;
    jpp.use_native = true;

    // The per-input covering-output count is bounded by ceil(K/S)+1 per spatial
    // dim (and by the output extent). Decline windows that could exceed the cap
    // so the driver's fixed stack contribution array can never overflow.
    auto cov = [](int k, int s, int o) {
        return nstl::min(o, utils::div_up(k, s) + 1);
    };
    const int64_t bound = (int64_t)cov(jpp.kd, jpp.stride_d, jpp.od)
            * cov(jpp.kh, jpp.stride_h, jpp.oh)
            * cov(jpp.kw, jpp.stride_w, jpp.ow);
    if (bound > max_contrib) return status::unimplemented;

    jpp.post_ops = post_ops_t();
    jpp.with_postops = false;
    jpp.fuse_eltwise = jpp.fuse_binary = false;
    jpp.with_relu = false;
    jpp.binary_bcast = pool_binary_bcast_t::none; // backward has default attrs
    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_bwd_kernel_t<isa, d_type>::generate() {
    if (utils::one_of(d_type, data_type::f16, data_type::bf16))
        generate_xf16();
    else
        generate_f32();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_bwd_kernel_t<isa, d_type>::generate_f32() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    using a_t = jit_uni_pool_bwd_args_t;
    using c_t = jit_uni_pool_bwd_contrib_t;
    const Reg reg_param = a0;
    const VReg v_mask(0), v_acc(4), v_tmp(8), v_ws(12), v_ws_raw(16),
            v_zero(20);
    const bool ind_u8 = jpp_.ind_dt == data_type::u8;
    const int csize = static_cast<int>(sizeof(c_t));

    const int stack_size = 80; // save s0-s8
    addi(sp, sp, -stack_size);
    save_saved_gprs(this, 9);

    ld(s0, reg_param, GET_OFF_A(diff_src));
    ld(s1, reg_param, GET_OFF_A(contribs));
    ld(s2, reg_param, GET_OFF_A(count));
    ld(s3, reg_param, GET_OFF_A(channels));
    ld(s4, reg_param, GET_OFF_A(src_vec_byte_stride));
    ld(s5, reg_param, GET_OFF_A(dst_vec_byte_stride));
    if (is_max_pool_) ld(s6, reg_param, GET_OFF_A(ws_vec_byte_stride));
    mv(s7, x0); // diff_dst running channel byte offset
    if (is_max_pool_) mv(s8, x0); // ws running channel byte offset
    li(t4, 4); // f32 unit-stride comparison constant

    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(s3, ch_done);
    vsetvli(t0, s3, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    vmv_v_x(v_acc, x0); // acc = 0
    if (is_max_pool_) vmv_v_x(v_zero, x0);

    // Contribution loop: accumulate each covering output's diff_dst channel row.
    mv(a1, s1); // contrib ptr
    mv(a2, s2); // count remaining
    Label c_loop, c_done;
    L(c_loop);
    beqz(a2, c_done);
    ld(a3, a1, GET_OFF_C(diff_dst));
    add(a3, a3, s7);
    {
        Label str, done;
        bne(s5, t4, str);
        vle32_v(v_tmp, a3);
        j_(done);
        L(str);
        vlse32_v(v_tmp, a3, s5);
        L(done);
    }
    if (is_max_pool_) {
        ld(a4, a1, GET_OFF_C(ws));
        add(a4, a4, s8);
        lw(a5, a1, GET_OFF_C(index));
        if (ind_u8) {
            Label str, done;
            li(t1, 1);
            vsetvli(t2, s3, SEW::e8, LMUL::mf4, VTA::ta, VMA::ma);
            bne(s6, t1, str);
            vle8_v(v_ws_raw, a4);
            j_(done);
            L(str);
            vlse8_v(v_ws_raw, a4, s6);
            L(done);
            vsetvli(t2, s3, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
            vzext_vf4(v_ws, v_ws_raw); // u8 index -> e32
        } else {
            Label str, done;
            li(t1, 4);
            bne(s6, t1, str);
            vle32_v(v_ws, a4);
            j_(done);
            L(str);
            vlse32_v(v_ws, a4, s6);
            L(done);
        }
        vmseq_vx(v_mask, v_ws, a5); // mask = (ws == index)
        vmerge_vvm(v_tmp, v_zero, v_tmp); // v_tmp = mask ? diff_dst : 0
        vfadd_vv(v_acc, v_acc, v_tmp);
    } else {
        flw(fa0, a1, GET_OFF_C(scale));
        vfmacc_vf(v_acc, fa0, v_tmp); // acc += scale * diff_dst
    }
    addi(a1, a1, csize);
    addi(a2, a2, -1);
    j_(c_loop);
    L(c_done);

    // Store the accumulated diff_src channel row (0 if no covering output).
    {
        Label str, done;
        bne(s4, t4, str);
        vse32_v(v_acc, s0);
        j_(done);
        L(str);
        vsse32_v(v_acc, s0, s4);
        L(done);
    }
    // Advance to the next channel chunk (uniform vl * channel byte stride).
    mul(t1, t0, s4);
    add(s0, s0, t1);
    mul(t1, t0, s5);
    add(s7, s7, t1);
    if (is_max_pool_) {
        mul(t1, t0, s6);
        add(s8, s8, t1);
    }
    sub(s3, s3, t0);
    j_(ch_loop);
    L(ch_done);

    restore_saved_gprs(this, 9);
    addi(sp, sp, stack_size);
    ret();
#else
    ret();
#endif
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_bwd_kernel_t<isa, d_type>::generate_xf16() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    using a_t = jit_uni_pool_bwd_args_t;
    using c_t = jit_uni_pool_bwd_contrib_t;
    const Reg reg_param = a0;
    // acc/wide/ws/zero are f32/e32m2 (v4,v12,v16,v24); the xf16 load and the
    // narrowed store result share v8 (e16m1); v_ws_raw (v20) is the raw u8 load.
    // Everything is accumulated at f32, so f16 and bf16 differ only in the
    // widen/narrow pair (Zvfh vfwcvt/vfncvt vs Zvfbfmin vfwcvtbf16/vfncvtbf16).
    constexpr bool is_bf16 = d_type == data_type::bf16;
    const VReg v_mask(0), v_acc(4), v_ddst(8), v_wide(12), v_ws(16),
            v_ws_raw(20), v_zero(24);
    const bool ind_u8 = jpp_.ind_dt == data_type::u8;
    const int csize = static_cast<int>(sizeof(c_t));

    const int stack_size = 80;
    addi(sp, sp, -stack_size);
    save_saved_gprs(this, 9);

    ld(s0, reg_param, GET_OFF_A(diff_src));
    ld(s1, reg_param, GET_OFF_A(contribs));
    ld(s2, reg_param, GET_OFF_A(count));
    ld(s3, reg_param, GET_OFF_A(channels));
    ld(s4, reg_param, GET_OFF_A(src_vec_byte_stride));
    ld(s5, reg_param, GET_OFF_A(dst_vec_byte_stride));
    if (is_max_pool_) ld(s6, reg_param, GET_OFF_A(ws_vec_byte_stride));
    mv(s7, x0);
    if (is_max_pool_) mv(s8, x0);
    li(t4, 2); // xf16 unit-stride comparison constant

    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(s3, ch_done);
    // f32m2 accumulator; e32m2 and e16m1 share the same vl (equal VLMAX), so t0
    // (set here) is the channel vl reused by every vsetvli below.
    vsetvli(t0, s3, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
    vmv_v_x(v_acc, x0);
    if (is_max_pool_) vmv_v_x(v_zero, x0);

    mv(a1, s1);
    mv(a2, s2);
    Label c_loop, c_done;
    L(c_loop);
    beqz(a2, c_done);
    ld(a3, a1, GET_OFF_C(diff_dst));
    add(a3, a3, s7);
    vsetvli(t2, s3, SEW::e16, LMUL::m1, VTA::ta,
            VMA::ma); // xf16 load vtype (vl == t0)
    {
        Label str, done;
        bne(s5, t4, str);
        vle16_v(v_ddst, a3);
        j_(done);
        L(str);
        vlse16_v(v_ddst, a3, s5);
        L(done);
    }
    // xf16 -> f32m2 (reads the e16 narrow vtype)
    if (is_bf16)
        vfwcvtbf16_f_f_v(v_wide, v_ddst);
    else
        vfwcvt_f_f_v(v_wide, v_ddst);
    if (is_max_pool_) {
        ld(a4, a1, GET_OFF_C(ws));
        add(a4, a4, s8);
        lw(a5, a1, GET_OFF_C(index));
        if (ind_u8) {
            Label str, done;
            li(t1, 1);
            vsetvli(t2, s3, SEW::e8, LMUL::mf2, VTA::ta, VMA::ma);
            bne(s6, t1, str);
            vle8_v(v_ws_raw, a4);
            j_(done);
            L(str);
            vlse8_v(v_ws_raw, a4, s6);
            L(done);
            vsetvli(t2, s3, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
            vzext_vf4(v_ws, v_ws_raw);
        } else {
            Label str, done;
            li(t1, 4);
            vsetvli(t2, s3, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
            bne(s6, t1, str);
            vle32_v(v_ws, a4);
            j_(done);
            L(str);
            vlse32_v(v_ws, a4, s6);
            L(done);
        }
        vmseq_vx(v_mask, v_ws, a5);
        vmerge_vvm(v_wide, v_zero, v_wide); // mask ? wide : 0
        vfadd_vv(v_acc, v_acc, v_wide);
    } else {
        vsetvli(t2, s3, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        flw(fa0, a1, GET_OFF_C(scale));
        vfmacc_vf(v_acc, fa0, v_wide);
    }
    addi(a1, a1, csize);
    addi(a2, a2, -1);
    j_(c_loop);
    L(c_done);

    // Narrow the f32 accumulator back to xf16 and store.
    vsetvli(t2, s3, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
    if (is_bf16)
        vfncvtbf16_f_f_w(v_ddst, v_acc);
    else
        vfncvt_f_f_w(v_ddst, v_acc);
    {
        Label str, done;
        bne(s4, t4, str);
        vse16_v(v_ddst, s0);
        j_(done);
        L(str);
        vsse16_v(v_ddst, s0, s4);
        L(done);
    }
    mul(t1, t0, s4);
    add(s0, s0, t1);
    mul(t1, t0, s5);
    add(s7, s7, t1);
    if (is_max_pool_) {
        mul(t1, t0, s6);
        add(s8, s8, t1);
    }
    sub(s3, s3, t0);
    j_(ch_loop);
    L(ch_done);

    restore_saved_gprs(this, 9);
    addi(sp, sp, stack_size);
    ret();
#else
    ret();
#endif
}

template struct jit_uni_pool_bwd_kernel_t<v, data_type::f32>;
template struct jit_uni_pool_bwd_kernel_t<zvfh, data_type::f16>;
template struct jit_uni_pool_bwd_kernel_t<zvfbfwma, data_type::bf16>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
