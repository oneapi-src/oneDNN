/*******************************************************************************
* Copyright 2026 openKylin community
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

#include <cstddef>

#include "common/bit_cast.hpp"
#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/rv64/jit_uni_resampling_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

// The broadcasts the resampling kernel can position itself (the injector runs
// in host-positioned byte-offset mode). Keep in sync with the pd's set in
// jit_uni_resampling.hpp.
static bcast_set_t get_supported_postops_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::no_broadcast};
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_resampling_kernel_t<isa, d_type>::jit_uni_resampling_kernel_t(
        const jit_resampling_conf_t &conf)
    : jit_generator_t(conf.alg == alg_kind::resampling_nearest
                      ? "jit_rvv_resampling_nearest"
                      : "jit_rvv_resampling_linear")
    , conf_(conf) {
    create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_resampling_kernel_t<isa, d_type>::init_conf(
        jit_resampling_conf_t &conf, const resampling_pd_t *pd) {
    const memory_desc_wrapper src_d(pd->src_md());
    const memory_desc_wrapper dst_d(pd->dst_md());
    const int ndims = pd->ndims();

    // The template dtype (f32 for isa v, f16 for isa zvfh, bf16 for isa
    // zvfbfwma) must match src/dst.
    if (src_d.data_type() != d_type || dst_d.data_type() != d_type)
        return status::unimplemented;
    const alg_kind_t alg = pd->desc()->alg_kind;
    if (!utils::one_of(
                alg, alg_kind::resampling_nearest, alg_kind::resampling_linear))
        return status::unimplemented;

    // Layout comes from the blocking descriptors, not from format tags: tags
    // are a user-facing label and can share a name across differing strides.
    if (!src_d.is_blocking_desc() || !dst_d.is_blocking_desc())
        return status::unimplemented;

    // The vector runs along C, so only a single innermost block on the channel
    // dimension keeps channels contiguous. Returns 0 to reject.
    auto channel_block = [](const memory_desc_wrapper &md) -> int {
        const auto &bd = md.blocking_desc();
        if (bd.inner_nblks == 0) return 1;
        if (bd.inner_nblks != 1 || bd.inner_idxs[0] != 1) return 0; // reject
        return (int)bd.inner_blks[0];
    };
    const int src_blk = channel_block(src_d), dst_blk = channel_block(dst_d);
    if (src_blk == 0 || dst_blk == 0) return status::unimplemented;
    // One grouping drives both tensors, so the blocks must agree.
    if (src_blk != dst_blk) return status::unimplemented;
    conf.block = src_blk;

    const auto &sbd = src_d.blocking_desc();
    const auto &dbd = dst_d.blocking_desc();
    // Blocked: strides[1] steps between blocks. Plain: it is the channel step.
    conf.src_c_stride = conf.block > 1 ? 1 : sbd.strides[1];
    conf.dst_c_stride = conf.block > 1 ? 1 : dbd.strides[1];
    conf.src_cb_stride = conf.block > 1 ? sbd.strides[1] : 0;
    conf.dst_cb_stride = conf.block > 1 ? dbd.strides[1] : 0;
    conf.src_mb_stride = sbd.strides[0];
    conf.dst_mb_stride = dbd.strides[0];
    conf.src_w_stride = sbd.strides[ndims - 1];
    conf.dst_w_stride = dbd.strides[ndims - 1];
    conf.src_h_stride = ndims >= 4 ? sbd.strides[ndims - 2] : 0;
    conf.dst_h_stride = ndims >= 4 ? dbd.strides[ndims - 2] : 0;
    conf.src_d_stride = ndims >= 5 ? sbd.strides[ndims - 3] : 0;
    conf.dst_d_stride = ndims >= 5 ? dbd.strides[ndims - 3] : 0;

    conf.ndims = ndims;
    conf.mb = pd->MB();
    conf.c = pd->C();
    conf.id = pd->ID();
    conf.ih = pd->IH();
    conf.iw = pd->IW();
    conf.od = pd->OD();
    conf.oh = pd->OH();
    conf.ow = pd->OW();
    conf.alg = alg;
    conf.data_type = d_type;
    conf.dt_size = (int)types::data_type_size(d_type);
    conf.isa = isa;
    conf.num_corners
            = (alg == alg_kind::resampling_nearest) ? 1 : (1 << (ndims - 2));

    // The pd's post_ops_ok() already restricted the chain; set the fusion flags
    // here, and bail to ref if the gate ever lets through something unfusable.
    const auto &po = pd->attr()->post_ops_;
    conf.post_ops = po;
    conf.with_postops = !po.has_default_values();
    conf.fuse_eltwise = conf.fuse_binary = conf.fuse_sum = false;
    conf.sum_idx = -1;
    conf.sum_scale = 1.f;
    if (conf.with_postops) {
        // The injector cannot carry sum; gate the rest on a sum-free copy.
        post_ops_t po_no_sum;
        for (int i = 0; i < po.len(); i++) {
            const auto &e = po.entry_[i];
            if (e.kind == primitive_kind::sum) {
                conf.fuse_sum = true;
                conf.sum_idx = i;
                conf.sum_scale = e.sum.scale;
            } else
                po_no_sum.entry_.push_back(e);
        }
        const bool inj_ok = injector::post_ops_ok(injector::post_ops_ok_args_t(
                isa, {injector::eltwise, injector::binary}, po_no_sum, &dst_d,
                false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
                true /*sum_requires_zp_zero*/,
                true /*sum_requires_same_params*/,
                get_supported_postops_bcast_strategies(), /*n_vaux=*/3));
        bool has_binary = false, has_eltwise = false;
        for (int i = 0; i < po_no_sum.len(); i++) {
            if (po_no_sum.entry_[i].is_binary()) has_binary = true;
            if (po_no_sum.entry_[i].is_eltwise()) has_eltwise = true;
        }
        conf.fuse_eltwise = inj_ok && has_eltwise && !has_binary;
        conf.fuse_binary = inj_ok && has_binary && (d_type == data_type::f32);
        if (!inj_ok) return status::unimplemented;
        if (!conf.fuse_eltwise && !conf.fuse_binary && !conf.fuse_sum)
            return status::unimplemented;
    }

    // Plain ncsp keeps channels a whole plane apart, so a vector along C turns
    // every lane into its own cache line. Vectorize along OW instead when W is
    // contiguous in both tensors (f32, no binary: its rhs is positioned along
    // C by the host).
    conf.w_vec = d_type == data_type::f32 && conf.block == 1
            && conf.src_w_stride == 1 && conf.dst_w_stride == 1
            && conf.src_c_stride != 1 && conf.dst_c_stride != 1
            && !conf.fuse_binary;
    conf.w_rows = alg == alg_kind::resampling_nearest ? 1 : (1 << (ndims - 3));
    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_resampling_kernel_t<isa, d_type>::generate() {
    if (utils::one_of(d_type, data_type::f16, data_type::bf16))
        generate_xf16();
    else if (conf_.w_vec)
        generate_f32_w();
    else
        generate_f32();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_resampling_kernel_t<isa, d_type>::generate_f32() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const VReg v_acc(4), v_tmp(8);
    const int n = conf_.num_corners;
    const bool po = conf_.fuse_eltwise || conf_.fuse_binary;

    // Pick the rhs load form. Lanes are channels: per-tensor -> scalar; per-oc
    // and contiguous full-dst -> unit stride; full-dst over a channel-strided
    // dst -> strided (a full-dst rhs matches the dst layout).
    bool bin_scalar = false, bin_strided = false;
    if (conf_.fuse_binary)
        for (int i = 0; i < conf_.post_ops.len(); i++) {
            const auto &e = conf_.post_ops.entry_[i];
            if (!e.is_binary()) continue;
            const memory_desc_wrapper s1(e.binary.src1_desc);
            bin_scalar = s1.nelems() == 1;
            const bool per_oc = !bin_scalar && s1.nelems() == (dim_t)conf_.c;
            const bool full = !bin_scalar && !per_oc;
            bin_strided = full && conf_.dst_c_stride != 1;
        }

    const Reg corner_reg[8] = {s1, s2, s3, s4, s5, s6, s7, s8};
    const FReg wei_reg[8] = {fa0, fa1, fa2, fa3, fa4, fa5, fa6, fa7};

    const int stack_size = 96;
    addi(sp, sp, -stack_size);
    sd(s1, sp, 0);
    sd(s2, sp, 8);
    sd(s3, sp, 16);
    sd(s4, sp, 24);
    sd(s5, sp, 32);
    sd(s6, sp, 40);
    sd(s7, sp, 48);
    sd(s8, sp, 56);
    sd(s9, sp, 64);
    sd(s10, sp, 72);
    sd(s11, sp, 80);

    using p_t = jit_resampling_args_t;
    for (int i = 0; i < n; i++) {
        ld(corner_reg[i], reg_param,
                static_cast<int>(offsetof(p_t, src) + i * sizeof(void *)));
        if (n > 1)
            flw(wei_reg[i], reg_param,
                    static_cast<int>(
                            offsetof(p_t, weights) + i * sizeof(float)));
    }
    ld(s9, reg_param, static_cast<int>(offsetof(p_t, dst)));
    ld(s10, reg_param, static_cast<int>(offsetof(p_t, channels)));
    ld(s11, reg_param, static_cast<int>(offsetof(p_t, src_vec_byte_stride)));
    ld(a1, reg_param, static_cast<int>(offsetof(p_t, dst_vec_byte_stride)));
    if (conf_.fuse_binary) {
        // a5 = shared byte offset (advanced per channel chunk); the injector
        // reads the rhs pointer array through the args pointer (x64 model).
        ld(a5, reg_param, static_cast<int>(offsetof(p_t, post_op_off0)));
    }

    addi(t2, x0, 4); // element size, for the unit-stride fast path

    // Post-op injector (built once; the binary rhs base is read from the
    // pointer array through the args pointer, at the offset in a5, advanced
    // per chunk).
    injector::jit_uni_postops_injector_t<isa> *po_inj = nullptr;
    // v24 doubles as the binary rhs scratch; entries execute serially, and
    // post_ops_ok(n_vaux = 3) rejects the algs that would read v_aux3/v_aux4.
    eltwise_injector::static_params_t esp(VReg(12), VReg(16), VReg(20),
            VReg(24), VReg(24), ft0, ft1, t3, /*is_fwd=*/true);
    // Host-positioned injector mode (null dst_d): the kernel maintains the
    // shared byte offset in a5 itself, so the rhs classifies as scalar /
    // no_broadcast from its element count. The rhs pointer array is read
    // through the args pointer (x64 model), so a4 is now only the injector's
    // address scratch. The rhs is f32 (pd-gated), so neither the narrow-dtype
    // staging group nor the gather index group is needed. preserve_gpr shields
    // a5 -- the live per-chunk offset -- and the injector's fixed X_TMP
    // scratch (t4, which this kernel uses for the sum scale).
    binary_injector::rhs_arg_static_params_t rhs_arg_bsp(VReg(24).getIdx(), a4,
            a5, a3, true /*preserve gpr*/, false /*preserve vmm*/,
            static_cast<std::size_t>(offsetof(p_t, post_op_rhs)),
            memory_desc_wrapper(nullptr));
    rhs_arg_bsp.rhs_dt_helper_freg = ft2;
    rhs_arg_bsp.off_is_bytes = true;
    if (bin_strided) {
        rhs_arg_bsp.is_strided = true;
        rhs_arg_bsp.rhs_stride = a1;
    }
    const binary_injector::static_params_t bsp(reg_param, rhs_arg_bsp);
    // The in-kernel sum runs as a host lambda at its exact chain position (the
    // x64/aarch64 scheme), so a single compute_vector covers the whole chain --
    // replacing the old split into two entry sub-ranges around the sum.
    injector::lambda_jit_injectors_t lambda_jit_injectors;
    if (conf_.fuse_sum)
        lambda_jit_injectors.emplace(primitive_kind::sum, [&]() {
            // dst is read back with the same access form as the store.
            Label sum_strided, sum_done;
            bne(a1, t2, sum_strided);
            vle32_v(v_tmp, s9);
            j_(sum_done);
            L(sum_strided);
            vlse32_v(v_tmp, s9, a1);
            L(sum_done);
            if (conf_.sum_scale == 1.f)
                vfadd_vv(v_acc, v_acc, v_tmp);
            else {
                li(t4, (int32_t)utils::bit_cast<uint32_t>(conf_.sum_scale));
                fmv_w_x(ft3, t4);
                vfmacc_vf(v_acc, ft3, v_tmp);
            }
        });
    injector::jit_uni_postops_injector_t<isa> po_inj_obj(
            this, conf_.post_ops, bsp, esp, lambda_jit_injectors);
    if (po || conf_.fuse_sum) po_inj = &po_inj_obj;

    auto load_vec = [&](const VReg &vd, const Reg &ptr) {
        Label strided, done;
        bne(s11, t2, strided);
        vle32_v(vd, ptr);
        j_(done);
        L(strided);
        vlse32_v(vd, ptr, s11);
        L(done);
    };

    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(s10, ch_done);
    vsetvli(t0, s10, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);

    if (n == 1) {
        load_vec(v_acc, corner_reg[0]);
    } else {
        load_vec(v_tmp, corner_reg[0]);
        vfmul_vf(v_acc, v_tmp, wei_reg[0]);
        for (int i = 1; i < n; i++) {
            load_vec(v_tmp, corner_reg[i]);
            vfmacc_vf(v_acc, wei_reg[i], v_tmp);
        }
    }

    if (po || conf_.fuse_sum) {
        // a5 is the per-chunk byte offset of the first active lane
        // (off_is_bytes); the injector derives each rhs address from it. The
        // sum lands in attribute order via its lambda injector.
        binary_injector::rhs_arg_dynamic_params_t rhs_dyn;
        rhs_dyn.vmm_idx_to_out_reg.emplace(v_acc.getIdx(), a5);
        // The f32 path computes at e32/m1 (group_stride 1).
        po_inj->compute_vector(v_acc.getIdx(), rhs_dyn, 1 /*group_stride*/);
    }

    {
        Label strided_dst, dst_done;
        bne(a1, t2, strided_dst);
        vse32_v(v_acc, s9);
        j_(dst_done);
        L(strided_dst);
        vsse32_v(v_acc, s9, a1);
        L(dst_done);
    }

    // Advance corner pointers by vl * src_vec_byte_stride (shared stride).
    {
        Label strided_adv, adv_done;
        bne(s11, t2, strided_adv);
        slli(t1, t0, 2);
        j_(adv_done);
        L(strided_adv);
        mul(t1, t0, s11);
        L(adv_done);
    }
    for (int i = 0; i < n; i++)
        add(corner_reg[i], corner_reg[i], t1);
    // Advance dst by vl * dst_vec_byte_stride.
    {
        Label strided_adv, adv_done;
        bne(a1, t2, strided_adv);
        slli(t1, t0, 2);
        j_(adv_done);
        L(strided_adv);
        mul(t1, t0, a1);
        L(adv_done);
    }
    add(s9, s9, t1);
    // Advance the shared rhs offset by vl * per-channel byte stride (full-dst on
    // ncsp: the dst channel stride a1; per-oc: contiguous element size). The
    // origin array (a4) is fixed; the injector reads array[arg_idx] + a5. Scalar
    // rhs is broadcast, so it does not advance.
    if (conf_.fuse_binary && !bin_scalar) {
        if (bin_strided)
            mul(t1, t0, a1);
        else
            slli(t1, t0, 2);
        add(a5, a5, t1);
    }
    sub(s10, s10, t0); // channels -= vl

    j_(ch_loop);
    L(ch_done);

    ld(s1, sp, 0);
    ld(s2, sp, 8);
    ld(s3, sp, 16);
    ld(s4, sp, 24);
    ld(s5, sp, 32);
    ld(s6, sp, 40);
    ld(s7, sp, 48);
    ld(s8, sp, 56);
    ld(s9, sp, 64);
    ld(s10, sp, 72);
    ld(s11, sp, 80);
    addi(sp, sp, stack_size);
    ret();
#else
    ret();
#endif
}

// Plain ncsp f32, vectorized along OW. One call produces one output row: the
// lanes are output columns, gathered (vluxei32) from conf_.w_rows source rows
// through per-OW byte-offset tables built once by the primitive. For linear,
// each row contributes left/right columns weighted by row_weight * w_k, the
// same products and the same accumulation order as the per-point kernel, so
// both paths give identical results.
template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_resampling_kernel_t<isa, d_type>::generate_f32_w() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const VReg v_idx0(1), v_idx1(2), v_wei0(3), v_acc(4), v_wei1(5), v_lane(6),
            v_src(8);
    const bool linear = conf_.alg == alg_kind::resampling_linear;
    const int nr = conf_.w_rows;
    // A binary post-op never reaches this path (see init_conf).
    const bool po = conf_.fuse_eltwise;

    const Reg row_reg[4] = {s1, s2, s3, s4};
    const FReg row_wei[4] = {fa0, fa1, fa2, fa3};

    const int stack_size = 80;
    addi(sp, sp, -stack_size);
    sd(s1, sp, 0);
    sd(s2, sp, 8);
    sd(s3, sp, 16);
    sd(s4, sp, 24);
    sd(s5, sp, 32);
    sd(s6, sp, 40);
    sd(s7, sp, 48);
    sd(s8, sp, 56);
    sd(s9, sp, 64);
    sd(s10, sp, 72);

    using p_t = jit_resampling_args_t;
    for (int r = 0; r < nr; r++) {
        ld(row_reg[r], reg_param,
                static_cast<int>(offsetof(p_t, src) + r * sizeof(void *)));
        if (linear)
            flw(row_wei[r], reg_param,
                    static_cast<int>(
                            offsetof(p_t, weights) + r * sizeof(float)));
    }
    ld(s5, reg_param, static_cast<int>(offsetof(p_t, w_idx)));
    if (linear) {
        ld(s6, reg_param,
                static_cast<int>(offsetof(p_t, w_idx) + sizeof(void *)));
        ld(s7, reg_param, static_cast<int>(offsetof(p_t, w_wei)));
        ld(s8, reg_param,
                static_cast<int>(offsetof(p_t, w_wei) + sizeof(void *)));
    }
    ld(s9, reg_param, static_cast<int>(offsetof(p_t, dst)));
    ld(s10, reg_param, static_cast<int>(offsetof(p_t, channels))); // OW

    // Eltwise chain and sum only, as in the xf16 kernel: the binary static
    // params are never consumed.
    injector::jit_uni_postops_injector_t<isa> *po_inj = nullptr;
    eltwise_injector::static_params_t esp(VReg(12), VReg(16), VReg(20),
            VReg(24), VReg(24), ft0, ft1, t3, /*is_fwd=*/true);
    binary_injector::rhs_arg_static_params_t rhs_arg_bsp(VReg(24).getIdx(), x0,
            x0, x0, false /*preserve gpr*/, false /*preserve vmm*/,
            0 /*abi_param_offset*/, memory_desc_wrapper(nullptr));
    rhs_arg_bsp.rhs_dt_helper_freg = ft2;
    const binary_injector::static_params_t bsp(x0, rhs_arg_bsp);
    injector::lambda_jit_injectors_t lambda_jit_injectors;
    if (conf_.fuse_sum)
        lambda_jit_injectors.emplace(primitive_kind::sum, [&]() {
            vle32_v(v_src, s9); // dst row is contiguous
            if (conf_.sum_scale == 1.f)
                vfadd_vv(v_acc, v_acc, v_src);
            else {
                li(t4, (int32_t)utils::bit_cast<uint32_t>(conf_.sum_scale));
                fmv_w_x(ft3, t4);
                vfmacc_vf(v_acc, ft3, v_src);
            }
        });
    injector::jit_uni_postops_injector_t<isa> po_inj_obj(
            this, conf_.post_ops, bsp, esp, lambda_jit_injectors);
    if (po || conf_.fuse_sum) po_inj = &po_inj_obj;
    binary_injector::rhs_arg_dynamic_params_t rhs_dyn;

    Label ow_loop, ow_done;
    L(ow_loop);
    beqz(s10, ow_done);
    vsetvli(t0, s10, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);

    vle32_v(v_idx0, s5);
    if (!linear) {
        vluxei32_v(v_acc, s1, v_idx0);
    } else {
        vle32_v(v_idx1, s6);
        vle32_v(v_wei0, s7);
        vle32_v(v_wei1, s8);
        for (int r = 0; r < nr; r++) {
            // Corner weight = (row weight) * (column weight), as on the host.
            vfmul_vf(v_lane, v_wei0, row_wei[r]);
            vluxei32_v(v_src, row_reg[r], v_idx0);
            if (r == 0)
                vfmul_vv(v_acc, v_src, v_lane);
            else
                vfmacc_vv(v_acc, v_lane, v_src);
            vfmul_vf(v_lane, v_wei1, row_wei[r]);
            vluxei32_v(v_src, row_reg[r], v_idx1);
            vfmacc_vv(v_acc, v_lane, v_src);
        }
    }

    if (po || conf_.fuse_sum)
        po_inj->compute_vector(v_acc.getIdx(), rhs_dyn, 1 /*group_stride*/);

    vse32_v(v_acc, s9);

    slli(t1, t0, 2); // vl * sizeof(f32), also the int32 table step
    add(s5, s5, t1);
    if (linear) {
        add(s6, s6, t1);
        add(s7, s7, t1);
        add(s8, s8, t1);
    }
    add(s9, s9, t1);
    sub(s10, s10, t0);
    j_(ow_loop);
    L(ow_done);

    ld(s1, sp, 0);
    ld(s2, sp, 8);
    ld(s3, sp, 16);
    ld(s4, sp, 24);
    ld(s5, sp, 32);
    ld(s6, sp, 40);
    ld(s7, sp, 48);
    ld(s8, sp, 56);
    ld(s9, sp, 64);
    ld(s10, sp, 72);
    addi(sp, sp, stack_size);
    ret();
#else
    ret();
#endif
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_resampling_kernel_t<isa, d_type>::generate_xf16() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    // xf16 loads (m1) widened to an f32 accumulator (m2); the weighted sum and
    // any eltwise post-op run at f32; the result is narrowed back to xf16.
    const VReg v_xf16(2); // xf16 load buffer (m1)
    const VReg v_acc(4); // f32 accumulator (m2: v4-v5)
    const VReg v_wide(8); // f32 widened corner (m2: v8-v9)
    const int n = conf_.num_corners;
    // The binary post-op is f32-only, so it is rejected for xf16.
    const bool po = conf_.fuse_eltwise;

    // bf16 converts through Zvfbfmin, f16 through Zvfh. Both are read under the
    // narrow e16 vtype, so the surrounding vsetvli sequence is identical.
    constexpr bool is_bf16 = d_type == data_type::bf16;
    auto widen = [&](const VReg &vd, const VReg &vs) {
        if (is_bf16)
            vfwcvtbf16_f_f_v(vd, vs); // e16m1 -> e32m2
        else
            vfwcvt_f_f_v(vd, vs);
    };
    auto narrow = [&](const VReg &vd, const VReg &vs) {
        if (is_bf16)
            vfncvtbf16_f_f_w(vd, vs); // e32m2 -> e16m1
        else
            vfncvt_f_f_w(vd, vs);
    };

    const Reg corner_reg[8] = {s1, s2, s3, s4, s5, s6, s7, s8};
    const FReg wei_reg[8] = {fa0, fa1, fa2, fa3, fa4, fa5, fa6, fa7};

    const int stack_size = 96;
    addi(sp, sp, -stack_size);
    sd(s1, sp, 0);
    sd(s2, sp, 8);
    sd(s3, sp, 16);
    sd(s4, sp, 24);
    sd(s5, sp, 32);
    sd(s6, sp, 40);
    sd(s7, sp, 48);
    sd(s8, sp, 56);
    sd(s9, sp, 64);
    sd(s10, sp, 72);
    sd(s11, sp, 80);

    using p_t = jit_resampling_args_t;
    for (int i = 0; i < n; i++) {
        ld(corner_reg[i], reg_param,
                static_cast<int>(offsetof(p_t, src) + i * sizeof(void *)));
        if (n > 1)
            flw(wei_reg[i], reg_param,
                    static_cast<int>(
                            offsetof(p_t, weights) + i * sizeof(float)));
    }
    ld(s9, reg_param, static_cast<int>(offsetof(p_t, dst)));
    ld(s10, reg_param, static_cast<int>(offsetof(p_t, channels)));
    ld(s11, reg_param, static_cast<int>(offsetof(p_t, src_vec_byte_stride)));
    ld(a1, reg_param, static_cast<int>(offsetof(p_t, dst_vec_byte_stride)));

    addi(t2, x0, 2); // xf16 element size (2 bytes) for the unit-stride path

    // Eltwise-only injector for xf16 (computed at f32 on the m2 accumulator).
    injector::jit_uni_postops_injector_t<isa> *po_inj = nullptr;
    eltwise_injector::static_params_t esp(VReg(12), VReg(16), VReg(20),
            VReg(24), VReg(24), ft0, ft1, t3, /*is_fwd=*/true);
    // The pd rejects a binary for xf16, so the (mandatory) binary static params
    // are never consumed; pass placeholder scratch.
    binary_injector::rhs_arg_static_params_t rhs_arg_bsp(VReg(24).getIdx(), x0,
            x0, x0, false /*preserve gpr*/, false /*preserve vmm*/,
            0 /*abi_param_offset*/, memory_desc_wrapper(nullptr));
    rhs_arg_bsp.rhs_dt_helper_freg = ft2;
    const binary_injector::static_params_t bsp(x0, rhs_arg_bsp);
    // The in-kernel sum runs as a host lambda at its exact chain position, so a
    // single compute_vector covers the whole chain. It is entered and left at
    // e32/m2; dst is read back with the same access form as the store.
    injector::lambda_jit_injectors_t lambda_jit_injectors;
    if (conf_.fuse_sum)
        lambda_jit_injectors.emplace(primitive_kind::sum, [&]() {
            vsetvli(t0, s10, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
            Label sum_strided, sum_done;
            bne(a1, t2, sum_strided);
            vle16_v(v_xf16, s9);
            j_(sum_done);
            L(sum_strided);
            vlse16_v(v_xf16, s9, a1);
            L(sum_done);
            widen(v_wide, v_xf16); // xf16 m1 -> f32 m2
            vsetvli(t0, s10, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
            if (conf_.sum_scale == 1.f)
                vfadd_vv(v_acc, v_acc, v_wide);
            else {
                li(t4, (int32_t)utils::bit_cast<uint32_t>(conf_.sum_scale));
                fmv_w_x(ft3, t4);
                vfmacc_vf(v_acc, ft3, v_wide);
            }
        });
    injector::jit_uni_postops_injector_t<isa> po_inj_obj(
            this, conf_.post_ops, bsp, esp, lambda_jit_injectors);
    if (po || conf_.fuse_sum) po_inj = &po_inj_obj;
    // Eltwise-only here: no binary rhs to address.
    binary_injector::rhs_arg_dynamic_params_t rhs_dyn;

    // Load an xf16 channel vector (unit-stride vle16 for nspc/blocked, strided
    // vlse16 for ncsp) into v_xf16 under the current e16 vtype.
    auto load_xf16 = [&](const Reg &ptr) {
        Label strided, done;
        bne(s11, t2, strided);
        vle16_v(v_xf16, ptr);
        j_(done);
        L(strided);
        vlse16_v(v_xf16, ptr, s11);
        L(done);
    };

    // Apply the chain to the f32 accumulator; entered and left at e32/m2. The
    // sum reads dst back as xf16 and widens it (same access form as the store).
    const bool need_f32 = po || conf_.fuse_sum;
    auto apply_chain = [&]() {
        // The xf16 path computes at e32/m2 (group_stride 2); the sum lands in
        // attribute order via its lambda injector.
        po_inj->compute_vector(v_acc.getIdx(), rhs_dyn, 2 /*group_stride*/);
    };

    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(s10, ch_done);

    if (n == 1) {
        // Nearest: copy the single xf16 corner. With a fused eltwise post-op,
        // widen to f32, apply the chain, narrow back.
        vsetvli(t0, s10, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        load_xf16(corner_reg[0]);
        if (need_f32) {
            widen(v_acc, v_xf16); // xf16 m1 -> f32 m2
            vsetvli(t0, s10, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
            apply_chain();
            vsetvli(t0, s10, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
            narrow(v_xf16, v_acc); // f32 m2 -> xf16 m1
        }
    } else {
        // Linear: widen each xf16 corner to f32, weighted-accumulate at f32.
        vsetvli(t0, s10, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        load_xf16(corner_reg[0]);
        widen(v_acc, v_xf16);
        vsetvli(t0, s10, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        vfmul_vf(v_acc, v_acc, wei_reg[0]);
        for (int i = 1; i < n; i++) {
            vsetvli(t0, s10, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
            load_xf16(corner_reg[i]);
            widen(v_wide, v_xf16);
            vsetvli(t0, s10, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
            vfmacc_vf(v_acc, wei_reg[i], v_wide);
        }
        if (need_f32) apply_chain();
        vsetvli(t0, s10, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        narrow(v_xf16, v_acc); // narrow result to xf16
    }

    // Store the xf16 result (unit-stride for nspc/blocked, strided for ncsp).
    {
        Label strided_dst, dst_done;
        bne(a1, t2, strided_dst);
        vse16_v(v_xf16, s9);
        j_(dst_done);
        L(strided_dst);
        vsse16_v(v_xf16, s9, a1);
        L(dst_done);
    }

    // Advance corner pointers by vl * src_vec_byte_stride.
    {
        Label strided_adv, adv_done;
        bne(s11, t2, strided_adv);
        slli(t1, t0, 1); // vl * 2 (xf16)
        j_(adv_done);
        L(strided_adv);
        mul(t1, t0, s11);
        L(adv_done);
    }
    for (int i = 0; i < n; i++)
        add(corner_reg[i], corner_reg[i], t1);
    {
        Label strided_adv, adv_done;
        bne(a1, t2, strided_adv);
        slli(t1, t0, 1);
        j_(adv_done);
        L(strided_adv);
        mul(t1, t0, a1);
        L(adv_done);
    }
    add(s9, s9, t1);
    sub(s10, s10, t0);

    j_(ch_loop);
    L(ch_done);

    ld(s1, sp, 0);
    ld(s2, sp, 8);
    ld(s3, sp, 16);
    ld(s4, sp, 24);
    ld(s5, sp, 32);
    ld(s6, sp, 40);
    ld(s7, sp, 48);
    ld(s8, sp, 56);
    ld(s9, sp, 64);
    ld(s10, sp, 72);
    ld(s11, sp, 80);
    addi(sp, sp, stack_size);
    ret();
#else
    ret();
#endif
}

template struct jit_uni_resampling_kernel_t<v, data_type::f32>;
template struct jit_uni_resampling_kernel_t<zvfh, data_type::f16>;
template struct jit_uni_resampling_kernel_t<zvfbfwma, data_type::bf16>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
