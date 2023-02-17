/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
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

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/jit_uni_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

template <cpu_isa_t isa>
struct jit_softmax_base_t : public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const void *src, *dst, *diff_dst; // src dubs as diff_src
        const void *interim; // scratch memory for intermediate storage
        const void *src_scales; // src_scales defined for all data type cases
        const void *dst_scales; // dst_scales defined for all data type cases
        size_t process_n_elems;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_softmax_t)

    // cpu specific part
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;
    const int vlen = cpu_isa_traits<isa>::vlen;

    const softmax_pd_t *pd_;
    const memory_desc_wrapper src_d_, dst_d_, diff_dst_d_;

    virtual void operator()(const call_params_t *p) = 0;
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector_;
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> log_injector_;

    XReg reg_param = abi_param1;

    XReg reg_exp_injector_table = x1;
    XReg reg_log_injector_table = x3;
    XReg reg_src = x8;
    XReg reg_diff_src = reg_src;
    XReg reg_dst = x9;
    XReg reg_diff_dst = x14;
    XReg reg_src_spat_offt = x10;
    XReg reg_process_n_elems = x11;
    XReg reg_reverse_n_elems = x12;
    XReg reg_tmp = x13;
    XReg reg_dst_spat_offt = x16;
    XReg reg_diff_dst_spat_offt = reg_log_injector_table;
    XReg reg_interim = reg_diff_dst;
    XReg reg_interim_spat_offt = abi_not_param1;
    XReg reg_src_scales = x6;
    XReg reg_dst_scales = x7;

    const PReg p_shuff0 = p11;
    const PReg p_shuff1 = p5;
    const PReg injector_mask = p1;
    const PReg injector_tmp = p6;

    TReg vtmp = TReg(27);
    TReg tail_vmask = TReg(0);
    TReg vneg_flt_max = TReg(28);
    TReg vone = TReg(29);
    TReg vsum = TReg(30);
    TReg vmax = TReg(31);
    TReg vsbr = vsum; // must be not equal to vmax
    TReg vzero = TReg(21);
    TReg vsaturation_ubound = vneg_flt_max;
    TReg v_tmp0 = TReg(23);

    bool is_softmax_ = pd_->is_softmax();
    bool is_logsoftmax_ = pd_->is_logsoftmax();
    bool axis_is_blocked_;
    bool need_scratchpad_;

    size_t simd_w_ = 0;
    size_t unroll_regs_ = 4;

    size_t axis_simd_full_;
    size_t axis_simd_tail_;
    size_t n_loops_;
    size_t loop_tail_;
    size_t process_n_elems_;
    size_t src_axis_stride_;
    size_t interim_axis_stride_;
    size_t dst_axis_stride_;
    size_t diff_dst_axis_stride_;

    void compute_predefined_variables() {
        axis_simd_full_ = pd_->axis_size() / simd_w_;
        axis_simd_tail_ = pd_->axis_size() % simd_w_;
        n_loops_ = axis_simd_full_ / unroll_regs_;
        loop_tail_ = axis_simd_full_ - n_loops_ * unroll_regs_;
        process_n_elems_ = compute_process_n_elems(dst_d_);
        src_axis_stride_ = compute_axis_stride(src_d_);
        interim_axis_stride_ = simd_w_ * sizeof(float);
        dst_axis_stride_ = compute_axis_stride(dst_d_);
        if (!pd_->is_fwd())
            diff_dst_axis_stride_ = compute_axis_stride(diff_dst_d_);
        axis_is_blocked_ = pd_->axis_size(true) != pd_->axis_size();
    }

    size_t compute_process_n_elems(const memory_desc_wrapper &mdw) {
        const auto &bd = mdw.blocking_desc();
        if (bd.inner_nblks) return bd.strides[pd_->axis()];
        return simd_w_;
    }

    size_t compute_axis_stride(const memory_desc_wrapper &mdw) {
        return compute_process_n_elems(mdw) * mdw.data_type_size();
    }

    void load_common_params() {
        fmov(vone.s, 1.0);
        mov(W_TMP_0, float2int(-FLT_MAX));
        dup(vneg_flt_max.s, W_TMP_0);

#define PARAM_OFF(x) offsetof(call_params_t, x)
#define PARAM_LOAD(reg, var) \
    add_imm(X_DEFAULT_ADDR, reg_param, PARAM_OFF(var), X_TMP_0); \
    ldr(reg, ptr(X_DEFAULT_ADDR));

        /* Address offset must be less than 256. */
        PARAM_LOAD(reg_process_n_elems, process_n_elems);
        PARAM_LOAD(reg_dst, dst);
        if (pd_->is_fwd()) {
            PARAM_LOAD(reg_src, src);
        } else {
            PARAM_LOAD(reg_diff_src, src);
            PARAM_LOAD(reg_diff_dst, diff_dst);
        }
        if (need_scratchpad_) { PARAM_LOAD(reg_interim, interim); }
        PARAM_LOAD(reg_src_scales, src_scales);
        PARAM_LOAD(reg_dst_scales, dst_scales);
#undef PARAM_OFF
#undef PARAM_LOAD
    }

    void uni_fmax(const ZReg &dst, const ZReg &src, const ZReg &src2,
            const PReg &mask = PReg(DUMMY_IDX)) {
        const uint32_t idxDst = dst.getIdx();
        const uint32_t idxSrc = src.getIdx();
        const uint32_t idxSrc2 = src2.getIdx();
        uint32_t pattern = 0;
        PReg mask_reg(DUMMY_IDX);

        pattern += (idxDst == idxSrc) ? (1 << 2) : 0;
        pattern += (idxDst == idxSrc2) ? (1 << 1) : 0;
        pattern += (idxSrc == idxSrc2) ? 1 : 0;

        if (mask.getIdx() == DUMMY_IDX)
            mask_reg = P_ALL_ONE;
        else
            mask_reg = mask;

        switch (pattern) {
            case 0x4: /* dst = src && dst != src2 && src != src2
                   This is the most popular case. */
                fmax(dst.s, mask_reg / T_m, src2.s);
                break;
            default: assert(!"Unreachable!"); break;
        }
    }

    XReg xreg_addr(const XReg &base, const XReg &off = XReg(DUMMY_IDX),
            const int disp = 0) {
        XReg x_addr = base;
        uint32_t offIdx = off.getIdx();

        if (offIdx <= SP_IDX) {
            add(X_DEFAULT_ADDR, base, off);
            x_addr = X_DEFAULT_ADDR;
        }
        if (disp) {
            add_imm(X_DEFAULT_ADDR, x_addr, disp, X_TMP_0);
            x_addr = X_DEFAULT_ADDR;
        }

        return x_addr;
    }

    XReg diff_src_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_src, reg_src_spat_offt, offt);
    }

    XReg src_ptr(size_t offt = 0) {
        return xreg_addr(reg_src, reg_src_spat_offt, offt);
    }

    XReg interim_ptr(size_t offt = 0) {
        return xreg_addr(reg_interim, reg_interim_spat_offt, offt);
    }

    XReg dst_ptr(size_t offt = 0) {
        return xreg_addr(reg_dst, reg_dst_spat_offt, offt);
    }

    XReg diff_dst_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_dst, reg_diff_dst_spat_offt, offt);
    }

    enum class op_t : unsigned { max, sum };

    void perform_op(TReg v, TReg vtmp, op_t op) {
        if (op == op_t::max)
            uni_fmax(v, v, vtmp);
        else if (op == op_t::sum)
            fadd(v.s, v.s, vtmp.s);
    }

    template <typename body_t>
    void axis_loop(body_t body) {
        Label main_loop, tail_loop, tail_axis;

        // reverse_spat_offt to dispatch between labels
        mov(reg_reverse_n_elems, reg_process_n_elems);
        mov_imm(reg_src_spat_offt, 0); // src/diff_src addr
        mov_imm(reg_dst_spat_offt, 0); // dst addr
        if (need_scratchpad_) mov_imm(reg_interim_spat_offt, 0); // scratch addr
        if (!pd_->is_fwd()) mov_imm(reg_diff_dst_spat_offt, 0); // d_dst addr
        L(main_loop);
        {
            if (n_loops_) {
                cmp(reg_reverse_n_elems, unroll_regs_ * process_n_elems_);
                b(LT, tail_loop);

                body(unroll_regs_, false);
                sub_imm(reg_reverse_n_elems, reg_reverse_n_elems,
                        unroll_regs_ * process_n_elems_, X_TMP_0);
                add_imm(reg_src_spat_offt, reg_src_spat_offt,
                        unroll_regs_ * src_axis_stride_, X_TMP_0);
                add_imm(reg_dst_spat_offt, reg_dst_spat_offt,
                        unroll_regs_ * dst_axis_stride_, X_TMP_0);
                if (need_scratchpad_)
                    add_imm(reg_interim_spat_offt, reg_interim_spat_offt,
                            unroll_regs_ * interim_axis_stride_, X_TMP_0);
                if (!pd_->is_fwd())
                    add_imm(reg_diff_dst_spat_offt, reg_diff_dst_spat_offt,
                            unroll_regs_ * diff_dst_axis_stride_, X_TMP_0);
                b(main_loop);
            }
        }

        L(tail_loop);
        {
            if (loop_tail_) {
                body(loop_tail_, false);
                add_imm(reg_src_spat_offt, reg_src_spat_offt,
                        loop_tail_ * src_axis_stride_, X_TMP_0);
                add_imm(reg_dst_spat_offt, reg_dst_spat_offt,
                        loop_tail_ * dst_axis_stride_, X_TMP_0);
                if (need_scratchpad_)
                    add_imm(reg_interim_spat_offt, reg_interim_spat_offt,
                            loop_tail_ * interim_axis_stride_, X_TMP_0);
                if (!pd_->is_fwd())
                    add_imm(reg_diff_dst_spat_offt, reg_diff_dst_spat_offt,
                            loop_tail_ * diff_dst_axis_stride_, X_TMP_0);
            }
        }

        L(tail_axis);
        {
            if (axis_simd_tail_) { body(1, true); }
        }
    }

    virtual void prepare_tail_mask() = 0;
    virtual void get_horizontal_op(const TReg &v, const TReg &vtmp, op_t op)
            = 0;
    virtual void accumulate_vmax() = 0;
    virtual void accumulate_vsum() = 0;
    virtual void compute_dst() = 0;
    virtual void initialization_hook() {}
    virtual void accumulate_vsbr() {}
    virtual void compute_diff_src() {}

    void forward() {
        accumulate_vmax();
        accumulate_vsum();
        compute_dst();
    }

    void backward() {
        accumulate_vsbr();
        compute_diff_src();
    }

    void prepare_mask() {
        if (isa == sve_512) {
            sub_imm(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 64 * 2, X_TMP_0);
            str(p_shuff0, ptr(X_TRANSLATOR_STACK, 0, MUL_VL));
            str(p_shuff1, ptr(X_TRANSLATOR_STACK, 1, MUL_VL));
            not_(P_TMP_1.b, P_ALL_ONE, P_ALL_ONE.b);
            trn1(p_shuff0.d, P_ALL_ONE.d, P_TMP_1.d);
            trn1(p_shuff0.d, p_shuff0.d, p_shuff0.d);
            trn1(p_shuff1.s, P_ALL_ONE.s, P_TMP_1.s);
        }
    }

    void restore_mask() {
        assert(isa == sve_512);

        ldr(p_shuff0, ptr(X_TRANSLATOR_STACK, 0, MUL_VL));
        ldr(p_shuff1, ptr(X_TRANSLATOR_STACK, 1, MUL_VL));
        add_imm(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 64 * 2, X_TMP_0);
    }

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void generate() override {
        if (pd_->is_fwd() || is_logsoftmax_)
            exp_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                    alg_kind::eltwise_exp, 0.0f, 0.0f, 1.0f, true,
                    reg_exp_injector_table, injector_mask, injector_tmp));
        if (pd_->is_fwd() && is_logsoftmax_) {
            log_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                    alg_kind::eltwise_log, 0.0f, 0.0f, 1.0f, true,
                    reg_log_injector_table, injector_mask, injector_tmp));
        }

        compute_predefined_variables();
        preamble();
        initialization_hook();

        prepare_mask();

        if (exp_injector_) exp_injector_->load_table_addr();
        if (log_injector_) log_injector_->load_table_addr();
        if (axis_simd_tail_) prepare_tail_mask();
        load_common_params();
        if (pd_->is_fwd())
            forward();
        else
            backward();

        restore_mask();
        postamble();
        if (exp_injector_) exp_injector_->prepare_table();
        if (log_injector_) log_injector_->prepare_table();
    }

    jit_softmax_base_t(const softmax_pd_t *pd)
        : jit_generator(nullptr, MAX_CODE_SIZE, true)
        , pd_(pd)
        , src_d_(pd_->is_fwd() ? pd_->src_md() : pd_->diff_src_md())
        , dst_d_(pd_->dst_md())
        , diff_dst_d_(pd_->diff_dst_md()) {
        simd_w_ = vlen / sizeof(float); // bf16 works on ymms
        need_scratchpad_ = utils::one_of(
                dst_d_.data_type(), data_type::u8, data_type::s8);
    }
};

template <cpu_isa_t isa>
struct jit_softmax_t;

template <>
struct jit_softmax_t<sve_512> : public jit_softmax_base_t<sve_512> {
    PReg tail_opmask = p2;

    void store(const XReg &addr, const ZReg &vmm, data_type_t dt,
            bool tail = false) {
        PReg opmask = P_ALL_ONE;
        bool tail_mask_valid = false;
        auto effective_addr = addr;
        TReg src_vmm = vmm;

        if (tail) {
            if (dt == data_type::f32) {
                if (axis_is_blocked_) {
                    src_vmm = vzero;
                    eor(vzero.d, vzero.d, vzero.d);
                    mov(src_vmm.s, tail_opmask / T_m, vmm.s);
                    effective_addr = addr;
                } else {
                    effective_addr = addr;
                    tail_mask_valid = true;
                }
            } else { // int8 store instructions assume mask on register
                tail_mask_valid = true;
            }
        }

        if (tail_mask_valid) opmask = tail_opmask;

        switch (dt) {
            case data_type::f32:
                st1w(src_vmm.s, opmask, ptr(effective_addr));
                break;
            case data_type::u8:
                eor(vzero.d, vzero.d, vzero.d); // since vzero might be spoiled
                saturate_f32(vmm, vzero, vsaturation_ubound, data_type::u8,
                        P_ALL_ONE);
                frinti(vmm.s, P_ALL_ONE / T_m, vmm.s);
                fcvtzu(vmm.s, P_ALL_ONE / T_m, vmm.s);
                smin(vmm.s, 127);
                st1b(vmm.s, opmask, ptr(effective_addr));
                // Need to restore data back to fp32 since we apply exp after
                // storing and data should be fp32.
                if (is_logsoftmax_) scvtf(vmm.s, opmask / T_m, vmm.s);
                break;
            case data_type::s8:
                saturate_f32(vmm, vzero, vsaturation_ubound, data_type::s8,
                        P_ALL_ONE);
                frinti(vmm.s, opmask / T_m, vmm.s);
                fcvtzs(vmm.s, opmask / T_m, vmm.s);
                smin(vmm.s, 127);
                smax(vmm.s, -128);
                st1b(vmm.s, opmask, ptr(effective_addr));
                // Need to restore data back to fp32 since we apply exp after
                // storing and data should be fp32.
                if (is_logsoftmax_) scvtf(vmm.s, opmask / T_m, vmm.s);
                break;
            default: assert(!"unsupported"); break;
        }
    };

    void load(const TReg &vmm, const XReg &addr, data_type_t dt,
            bool tail = false) {
        PReg tmp_mask = P_ALL_ONE;
        ZRegS effective_vmm = vmm.s;

        if (tail) tmp_mask = tail_opmask;

        switch (dt) {
            case data_type::f32:
                ld1w(effective_vmm, tmp_mask, ptr(addr));
                break;
            case data_type::u8:
                ld1b(effective_vmm, tmp_mask / T_z, ptr(addr));
                scvtf(effective_vmm, P_ALL_ONE / T_m, effective_vmm);
                break;
            case data_type::s8:
                ld1sb(effective_vmm, tmp_mask / T_z, ptr(addr));
                scvtf(effective_vmm, P_ALL_ONE / T_m, effective_vmm);
                break;
            default: assert(!"unsupported"); break;
        }
    };

    void prepare_tail_mask() override {
        const int sw_tail = axis_simd_tail_;
        PRegS p = tail_opmask.s;
        switch (sw_tail) {
            case 16: ptrue(p, VL16); break;
            case 8: ptrue(p, VL8); break;
            case 7: ptrue(p, VL7); break;
            case 6: ptrue(p, VL6); break;
            case 5: ptrue(p, VL5); break;
            case 4: ptrue(p, VL4); break;
            case 3: ptrue(p, VL3); break;
            case 2: ptrue(p, VL2); break;
            case 1: ptrue(p, VL1); break;
            default:
                index(vtmp.s, 1, 1);
                cmple(p, P_ALL_ONE / T_z, vtmp.s, sw_tail);
                break;
        }
    }

    void get_horizontal_op(const ZReg &v, const ZReg &vtmp, op_t op) override {
        mov(vtmp.d, v.d);
        ext(vtmp.b, v.b, 32);
        perform_op(v, vtmp, op);
        mov(vtmp.s, P_ALL_ONE, v.s);
        mov(v_tmp0.s, P_ALL_ONE, v.s);
        ext(v_tmp0.b, v.b, 48);
        ext(vtmp.b, v.b, 16);
        mov(vtmp.d, p_shuff0 / T_m, v_tmp0.d);
        perform_op(v, vtmp, op);
        uzp2(v_tmp0.d, v.d, v.d);
        trn1(vtmp.d, v_tmp0.d, v.d);
        perform_op(v, vtmp, op);
        trn1(vtmp.s, v.s, v.s);
        trn2(v_tmp0.s, v.s, v.s);
        mov(vtmp.s, p_shuff1 / T_m, v_tmp0.s);
        perform_op(v, vtmp, op);
    }

    void accumulate_vmax() override {
        // flush to -FLT_MAX before accumulation
        mov(vmax.d, vneg_flt_max.d);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                TReg vreg_tmp_src = TReg(i + 1);
                load(vreg_tmp_src, src_ptr(src_axis_stride_ * i),
                        src_d_.data_type(), tail);
                if (tail) {
                    uni_fmax(vmax, vmax, vreg_tmp_src, tail_opmask);
                } else {
                    uni_fmax(vmax, vmax, vreg_tmp_src);
                }
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vsum() override {
        // Initialize saturation vector register
        if (utils::one_of(dst_d_.data_type(), data_type::u8, data_type::s8)) {
            init_saturate_f32(vzero, vsaturation_ubound, reg_tmp,
                    data_type::f32, dst_d_.data_type());
        }

        eor(vsum.d, vsum.d, vsum.d); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                TReg vreg_tmp_src = TReg(i + 1);
                load(vreg_tmp_src, src_ptr(src_axis_stride_ * i),
                        src_d_.data_type(), tail);
                fsub(vreg_tmp_src.s, vreg_tmp_src.s, vmax.s);
                if (is_logsoftmax_) { // store before applying exp
                    if (need_scratchpad_) {
                        store(interim_ptr(interim_axis_stride_ * i),
                                vreg_tmp_src, data_type::f32, tail);
                    } else {
                        store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                                dst_d_.data_type(), tail);
                    }
                }
                exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                if (tail)
                    fadd(vsum.s, tail_opmask / T_m, vreg_tmp_src.s);
                else
                    fadd(vsum.s, vsum.s, vreg_tmp_src.s);
                if (is_softmax_) { // store after applying exp
                    if (need_scratchpad_) {
                        store(interim_ptr(interim_axis_stride_ * i),
                                vreg_tmp_src, data_type::f32, tail);
                    } else {
                        store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                                dst_d_.data_type(), tail);
                    }
                }
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_) {
            mov(v_tmp0.d, vsum.d);
            mov(vsum.d, P_ALL_ONE, vone.d);
            fdiv(vsum.s, P_ALL_ONE / T_m, v_tmp0.s);
        }
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void compute_dst() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                ZReg vreg_tmp_src = ZReg(i + 1);
                if (need_scratchpad_) {
                    load(vreg_tmp_src, interim_ptr(interim_axis_stride_ * i),
                            data_type::f32, tail);
                } else {
                    load(vreg_tmp_src, dst_ptr(dst_axis_stride_ * i),
                            dst_d_.data_type(), tail);
                }

                if (is_softmax_) {
                    fmul(vreg_tmp_src.s, vreg_tmp_src.s, vsum.s);
                }
                if (is_logsoftmax_) {
                    fsub(vreg_tmp_src.s, vreg_tmp_src.s, vsum.s);
                }

                TReg vscale = vmax;
                ldr(vscale, ptr(reg_src_scales));
                fmul(vreg_tmp_src.s, vreg_tmp_src.s, vscale.s);
                // Reserved spot for post-ops injector.
                ldr(vscale, ptr(reg_dst_scales));
                fmul(vreg_tmp_src.s, vreg_tmp_src.s, vscale.s);
                store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                        dst_d_.data_type(), tail);
            }
        });
    }

    void accumulate_vsbr() override {
        eor(vsbr.d, vsbr.d, vsbr.d); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                ZReg vreg_tmp_dst = ZReg(i * 2 + 1);
                ZReg vreg_tmp_diff_dst = ZReg(i * 2 + 2);
                load(vreg_tmp_diff_dst, diff_dst_ptr(diff_dst_axis_stride_ * i),
                        diff_dst_d_.data_type(), tail);
                if (is_softmax_) {
                    load(vreg_tmp_dst, dst_ptr(dst_axis_stride_ * i),
                            dst_d_.data_type(), tail);
                    fmul(vreg_tmp_diff_dst.s, vreg_tmp_diff_dst.s,
                            vreg_tmp_dst.s);
                }
                fadd(vsbr.s, vsbr.s, vreg_tmp_diff_dst.s);
            }
        });

        get_horizontal_op(vsbr, vtmp = vmax, op_t::sum);
    }

    void compute_diff_src() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                ZReg vreg_tmp_dst = ZReg(i * 2 + 1);
                ZReg vreg_tmp_diff_dst = ZReg(i * 2 + 2);
                load(vreg_tmp_dst, dst_ptr(dst_axis_stride_ * i),
                        dst_d_.data_type(), tail);
                load(vreg_tmp_diff_dst, diff_dst_ptr(diff_dst_axis_stride_ * i),
                        diff_dst_d_.data_type(), tail);
                if (is_softmax_) {
                    fsub(vreg_tmp_diff_dst.s, vreg_tmp_diff_dst.s, vsbr.s);
                    fmul(vreg_tmp_diff_dst.s, vreg_tmp_dst.s,
                            vreg_tmp_diff_dst.s);
                }
                if (is_logsoftmax_) {
                    exp_injector_->compute_vector(vreg_tmp_dst.getIdx());
                    fmls(vreg_tmp_diff_dst.s, P_ALL_ONE / T_m, vreg_tmp_dst.s,
                            vsbr.s);
                }
                store(diff_src_ptr(src_axis_stride_ * i), vreg_tmp_diff_dst,
                        src_d_.data_type(), tail);
            }
        });
    }

    void initialization_hook() override {}

    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_base_t(pd) {}

    void operator()(const call_params_t *p) override {
        return jit_generator::operator()(p);
    }
}; // namespace aarch64

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::jit_uni_softmax_fwd_t(const pd_t *apd)
    : primitive_t(apd)
    , softmax_driver_(new softmax_impl::driver_t<isa>(pd())) {}

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::~jit_uni_softmax_fwd_t() {
    delete softmax_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_fwd_t<isa>::init(engine_t *engine) {
    return softmax_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    auto scratchpad_ptr = ctx.get_scratchpad_grantor().template get<char>(
            memory_tracking::names::key_softmax_interim_store);

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const auto src_data_type_size = src_d.data_type_size();
    const auto dst_data_type_size = dst_d.data_type_size();
    const auto &bd = src_d.blocking_desc();
    const auto axis = pd()->axis();

    const auto axis_size_padded = pd()->axis_size(true);
    const auto inner_stride
            = bd.inner_nblks ? bd.inner_blks[bd.inner_nblks - 1] : (dim_t)1;
    const auto inner_size = bd.strides[axis] / inner_stride;
    const auto process_n_elems = pd()->axis_size() * inner_size;
    const auto outer_stride = axis_size_padded * inner_size;
    const auto outer_size = src_d.nelems(true) / outer_stride;

    const int nthr = pd()->nthr_;

    parallel_nd_ext(nthr, outer_size, inner_size,
            [&](int ithr, int, dim_t ou, dim_t in) {
                dim_t offset = (ou * outer_stride + in * inner_stride);
                const char *src_ptr = src + offset * src_data_type_size;
                char *dst_ptr = dst + offset * dst_data_type_size;
                char *interim_ptr = scratchpad_ptr ? scratchpad_ptr
                                + ithr * axis_size_padded * sizeof(float)
                                                   : nullptr;
                softmax_driver_->exec(src_ptr, dst_ptr, interim_ptr, src_scales,
                        dst_scales, process_n_elems);
            });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_softmax_bwd_t<isa>::jit_uni_softmax_bwd_t(const pd_t *apd)
    : primitive_t(apd)
    , softmax_driver_(new softmax_impl::driver_t<isa>(pd())) {}

template <cpu_isa_t isa>
jit_uni_softmax_bwd_t<isa>::~jit_uni_softmax_bwd_t() {
    delete softmax_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_bwd_t<isa>::init(engine_t *engine) {
    return softmax_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_bwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    auto dst = CTX_IN_MEM(const char *, DNNL_ARG_DST);
    auto diff_dst = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const auto dst_data_type_size = dst_d.data_type_size();
    const auto diff_dst_data_type_size = diff_dst_d.data_type_size();
    const auto diff_src_data_type_size = diff_src_d.data_type_size();
    const auto &bd = dst_d.blocking_desc();
    const auto axis = pd()->axis();

    const auto inner_stride
            = bd.inner_nblks ? bd.inner_blks[bd.inner_nblks - 1] : (dim_t)1;
    const auto inner_size = bd.strides[axis] / inner_stride;
    const auto process_n_elems = pd()->axis_size() * inner_size;
    const auto outer_stride = pd()->axis_size(true) * inner_size;
    const auto outer_size = dst_d.nelems(true) / outer_stride;

    parallel_nd(outer_size, inner_size, [&](dim_t ou, dim_t in) {
        dim_t offset = (ou * outer_stride + in * inner_stride);
        char *diff_src_ptr = diff_src + offset * diff_src_data_type_size;
        const char *dst_ptr = dst + offset * dst_data_type_size;
        const char *diff_dst_ptr = diff_dst + offset * diff_dst_data_type_size;
        softmax_driver_->exec(
                diff_src_ptr, dst_ptr, diff_dst_ptr, process_n_elems);
    });

    return status::success;
}

namespace softmax_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {

    driver_t(const softmax_pd_t *pd) : pd_(pd), ker_(pd_) {}

    void exec(const void *src, void *dst, void *interim, const void *src_scales,
            const void *dst_scales, const dim_t process_n_elems) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.process_n_elems = process_n_elems;
        p.src = src;
        p.dst = dst;
        p.interim = interim;
        p.src_scales = src_scales;
        p.dst_scales = dst_scales;
        ker_(&p);
    }

    void exec(void *diff_src, const void *dst, const void *diff_dst,
            const dim_t process_n_elems) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.process_n_elems = process_n_elems;
        p.src = diff_src;
        p.dst = dst;
        p.diff_dst = diff_dst;
        ker_(&p);
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    const softmax_pd_t *pd_;
    jit_softmax_t<isa> ker_;
};

} // namespace softmax_impl

/* struct instantiation */
template struct jit_uni_softmax_fwd_t<sve_512>;
template struct jit_uni_softmax_bwd_t<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
