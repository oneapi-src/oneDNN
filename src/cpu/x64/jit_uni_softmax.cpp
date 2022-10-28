/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_uni_softmax.hpp"

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1900)
// Intel Compilers 17.x and 18.x do not like that diff_src_ptr() is only used
// in a single descendant class and marks it as unused. This breaks builds
// with DNNL_WERROR=on. Disabling the warning for this file seems to be less
// ugly than all the fixes that I came up with.
#pragma warning disable : 177
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

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
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword
            = (isa == sse41) ? xword : (isa == avx2) ? yword : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    const softmax_pd_t *pd_;
    const memory_desc_wrapper src_d_, dst_d_, diff_dst_d_;

    virtual void operator()(const call_params_t *p) = 0;
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector_;
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> log_injector_;

    Reg64 reg_param = abi_param1;

    Reg64 reg_exp_injector_table = rax;
    Reg64 reg_log_injector_table = rbx;
    Reg64 reg_src = r8;
    Reg64 reg_diff_src = reg_src;
    Reg64 reg_dst = r9;
    Reg64 reg_diff_dst = r14;
    Reg64 reg_src_spat_offt = r10;
    Reg64 reg_process_n_elems = r11;
    Reg64 reg_reverse_n_elems = r12;
    Reg64 reg_tmp = r13;
    Reg64 reg_dst_spat_offt = r15;
    Reg64 reg_diff_dst_spat_offt = reg_log_injector_table;
    Reg64 reg_interim = reg_diff_dst;
    Reg64 reg_interim_spat_offt = abi_not_param1;
    Reg64 reg_src_scales = rsi;
    Reg64 reg_dst_scales = rdx;

    Opmask injector_mask = Opmask(1);

    Vmm vtmp; // assigned at placed where used
    Vmm tail_vmask = Vmm(0);
    Xmm xneg_flt_max = Xmm(12);
    Vmm vneg_flt_max = Vmm(isa == avx512_core ? 28 : 12);
    Xmm xone = Xmm(13);
    Vmm vone = Vmm(isa == avx512_core ? 29 : 13);
    Vmm vsum = Vmm(isa == avx512_core ? 30 : 14);
    Vmm vmax = Vmm(isa == avx512_core ? 31 : 15);
    Vmm vsbr = vsum; // must be not equal to vmax
    Vmm vzero = Vmm(isa == avx512_core ? 21 : 11);
    Vmm vsaturation_ubound = vneg_flt_max;

    bool is_bf16_ = false;
    bool is_f16_ = false;
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
        mov(reg_tmp, float2int(1.0f));
        uni_vmovq(xone, reg_tmp);
        uni_vbroadcastss(vone, xone);
        mov(reg_tmp, float2int(-FLT_MAX));
        uni_vmovq(xneg_flt_max, reg_tmp);
        uni_vbroadcastss(vneg_flt_max, xneg_flt_max);

#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_process_n_elems, ptr[reg_param + PARAM_OFF(process_n_elems)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        if (pd_->is_fwd())
            mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        else {
            mov(reg_diff_src, ptr[reg_param + PARAM_OFF(src)]); // src is reused
            mov(reg_diff_dst, ptr[reg_param + PARAM_OFF(diff_dst)]);
        }
        if (need_scratchpad_) {
            mov(reg_interim, ptr[reg_param + PARAM_OFF(interim)]);
        }
        mov(reg_src_scales, ptr[reg_param + PARAM_OFF(src_scales)]);
        mov(reg_dst_scales, ptr[reg_param + PARAM_OFF(dst_scales)]);
#undef PARAM_OFF
    }

    Address diff_src_ptr(size_t offt = 0) {
        return vmmword[reg_diff_src + reg_src_spat_offt + offt];
    }

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + reg_src_spat_offt + offt];
    }

    Address interim_ptr(size_t offt = 0) {
        return vmmword[reg_interim + reg_interim_spat_offt + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_dst_spat_offt + offt];
    }

    Address diff_dst_ptr(size_t offt = 0) {
        return vmmword[reg_diff_dst + reg_diff_dst_spat_offt + offt];
    }

    enum class op_t : unsigned { max, sum };

    void perform_op(Vmm v, Vmm vtmp, op_t op) {
        if (op == op_t::max)
            uni_vmaxps(v, v, vtmp);
        else if (op == op_t::sum)
            uni_vaddps(v, v, vtmp);
    }

    template <typename body_t>
    void axis_loop(body_t body) {
        Label main_loop, tail_loop, tail_axis;

        // reverse_spat_offt to dispatch between labels
        mov(reg_reverse_n_elems, reg_process_n_elems);
        xor_(reg_src_spat_offt, reg_src_spat_offt); // src/diff_src addr
        xor_(reg_dst_spat_offt, reg_dst_spat_offt); // dst addr
        if (need_scratchpad_)
            xor_(reg_interim_spat_offt, reg_interim_spat_offt); // scratch addr
        if (!pd_->is_fwd())
            xor_(reg_diff_dst_spat_offt, reg_diff_dst_spat_offt); // d_dst addr
        L(main_loop);
        {
            if (n_loops_) {
                cmp(reg_reverse_n_elems, unroll_regs_ * process_n_elems_);
                jl(tail_loop, T_NEAR);

                body(unroll_regs_, false);
                sub(reg_reverse_n_elems, unroll_regs_ * process_n_elems_);
                add(reg_src_spat_offt, unroll_regs_ * src_axis_stride_);
                add(reg_dst_spat_offt, unroll_regs_ * dst_axis_stride_);
                if (need_scratchpad_)
                    add(reg_interim_spat_offt,
                            unroll_regs_ * interim_axis_stride_);
                if (!pd_->is_fwd())
                    add(reg_diff_dst_spat_offt,
                            unroll_regs_ * diff_dst_axis_stride_);
                jmp(main_loop);
            }
        }

        L(tail_loop);
        {
            if (loop_tail_) {
                body(loop_tail_, false);
                add(reg_src_spat_offt, loop_tail_ * src_axis_stride_);
                add(reg_dst_spat_offt, loop_tail_ * dst_axis_stride_);
                if (need_scratchpad_)
                    add(reg_interim_spat_offt,
                            loop_tail_ * interim_axis_stride_);
                if (!pd_->is_fwd())
                    add(reg_diff_dst_spat_offt,
                            loop_tail_ * diff_dst_axis_stride_);
            }
        }

        L(tail_axis);
        {
            if (axis_simd_tail_) { body(1, true); }
        }
    }

    virtual void prepare_tail_mask() = 0;
    virtual void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) = 0;
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

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void generate() override {
        if (pd_->is_fwd() || is_logsoftmax_)
            exp_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                    alg_kind::eltwise_exp, 0.0f, 0.0f, 1.0f, true,
                    reg_exp_injector_table, injector_mask));
        if (pd_->is_fwd() && is_logsoftmax_) {
            log_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                    alg_kind::eltwise_log, 0.0f, 0.0f, 1.0f, true,
                    reg_log_injector_table, injector_mask));
        }

        compute_predefined_variables();
        preamble();
        initialization_hook();
        if (exp_injector_) exp_injector_->load_table_addr();
        if (log_injector_) log_injector_->load_table_addr();
        if (axis_simd_tail_) prepare_tail_mask();
        load_common_params();
        if (pd_->is_fwd())
            forward();
        else
            backward();
        postamble();
        if (exp_injector_) exp_injector_->prepare_table();
        if (log_injector_) log_injector_->prepare_table();
    }

    jit_softmax_base_t(const softmax_pd_t *pd)
        : jit_generator(jit_name(), nullptr, MAX_CODE_SIZE, true, isa)
        , pd_(pd)
        , src_d_(pd_->is_fwd() ? pd_->src_md() : pd_->diff_src_md())
        , dst_d_(pd_->dst_md())
        , diff_dst_d_(pd_->diff_dst_md()) {
        is_bf16_ = utils::one_of(
                data_type::bf16, src_d_.data_type(), dst_d_.data_type());
        is_f16_ = utils::one_of(
                data_type::f16, src_d_.data_type(), dst_d_.data_type());
        simd_w_ = vlen / sizeof(float); // bf16 works on ymms
        need_scratchpad_ = utils::one_of(
                dst_d_.data_type(), data_type::u8, data_type::s8);
    }
};

template <cpu_isa_t isa>
struct jit_softmax_t;

template <>
struct jit_softmax_t<avx512_core> : public jit_softmax_base_t<avx512_core> {
    std::unique_ptr<bf16_emulation_t> bf16_emu_ = nullptr;
    Ymm bf16_cvt_ymm = Ymm(22);
    Zmm bf16_emu_zmm_1 = Zmm(23);
    Zmm bf16_emu_zmm_2 = Zmm(24);
    Zmm bf16_emu_zmm_3 = Zmm(25);
    Zmm bf16_emu_zmm_4 = Zmm(26);
    Zmm bf16_emu_zmm_5 = Zmm(27);
    Reg64 bf16_emu_gpr = reg_tmp;

    Opmask tail_opmask = Opmask(2);

    void store(const Address &addr, const Vmm &vmm, data_type_t dt,
            bool tail = false) {
        auto effective_addr = addr;
        Vmm src_vmm = vmm;

        if (tail) {
            if (utils::one_of(
                        dt, data_type::f32, data_type::bf16, data_type::f16)) {
                if (axis_is_blocked_) {
                    src_vmm = vzero | tail_opmask;
                    uni_vxorps(vzero, vzero, vzero);
                    uni_vmovups(src_vmm, vmm);
                    src_vmm = vzero;
                    effective_addr = addr;
                } else {
                    effective_addr = addr | tail_opmask;
                }
            } else { // int8 store instructions assume mask on register
                src_vmm = src_vmm | tail_opmask;
            }
        }

        switch (dt) {
            case data_type::f32: uni_vmovups(effective_addr, src_vmm); break;
            case data_type::bf16:
                if (bf16_emu_)
                    bf16_emu_->vcvtneps2bf16(bf16_cvt_ymm, src_vmm);
                else
                    vcvtneps2bf16(bf16_cvt_ymm, src_vmm);
                vmovdqu16(effective_addr, bf16_cvt_ymm);
                break;
            case data_type::f16:
                vcvtps2ph(effective_addr, src_vmm, _op_mxcsr);
                break;
            case data_type::u8:
                uni_vxorps(vzero, vzero, vzero); // since vzero might be spoiled
                saturate_f32(vmm, vzero, vsaturation_ubound, data_type::u8);
                vcvtps2dq(vmm, vmm);
                vpmovusdb(effective_addr, src_vmm);
                // Need to restore data back to fp32 since we apply exp after
                // storing and data should be fp32.
                if (is_logsoftmax_) vcvtdq2ps(vmm, vmm);
                break;
            case data_type::s8:
                saturate_f32(vmm, vzero, vsaturation_ubound, data_type::s8);
                vcvtps2dq(vmm, vmm);
                vpmovsdb(effective_addr, src_vmm);
                // Need to restore data back to fp32 since we apply exp after
                // storing and data should be fp32.
                if (is_logsoftmax_) vcvtdq2ps(vmm, vmm);
                break;
            default: assert(!"unsupported"); break;
        }
    };

    void load(const Vmm &vmm, const Address &addr, data_type_t dt,
            bool tail = false) {
        auto effective_vmm = vmm;
        if (tail) effective_vmm = vmm | tail_opmask | T_z;

        switch (dt) {
            case data_type::f32: uni_vmovups(effective_vmm, addr); break;
            case data_type::bf16:
                vpmovzxwd(effective_vmm, addr);
                vpslld(effective_vmm, effective_vmm, 0x10);
                break;
            case data_type::f16: vcvtph2psx(effective_vmm, addr); break;
            case data_type::u8:
                vpmovzxbd(effective_vmm, addr);
                vcvtdq2ps(effective_vmm, effective_vmm);
                break;
            case data_type::s8:
                vpmovsxbd(effective_vmm, addr);
                vcvtdq2ps(effective_vmm, effective_vmm);
                break;
            default: assert(!"unsupported"); break;
        }
    };

    void prepare_tail_mask() override {
        const int mask_f32 = (1 << axis_simd_tail_) - 1;
        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask_f32);
        kmovw(tail_opmask, regw_tmp);
    }

    void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) override {
        vshuff32x4(vtmp, v, v, 0x4E); // 256-bit shuffle
        perform_op(v, vtmp, op);
        vshuff32x4(vtmp, v, v, 0xB1); // 128/256-bit shuffle
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0x4E); // 64/128-bit shuffle
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0xB1); // 32/64-bit shuffle
        perform_op(v, vtmp, op);
    }

    void accumulate_vmax() override {
        // flush to -FLT_MAX before accumulation
        uni_vmovups(vmax, vneg_flt_max);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                load(vreg_tmp_src, src_ptr(src_axis_stride_ * i),
                        src_d_.data_type(), tail);
                if (tail)
                    uni_vmaxps(vmax | tail_opmask, vmax, vreg_tmp_src);
                else
                    uni_vmaxps(vmax, vmax, vreg_tmp_src);
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

        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                load(vreg_tmp_src, src_ptr(src_axis_stride_ * i),
                        src_d_.data_type(), tail);
                uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
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
                    uni_vaddps(vsum | tail_opmask, vsum, vreg_tmp_src);
                else
                    uni_vaddps(vsum, vsum, vreg_tmp_src);
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
        if (is_softmax_) uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void compute_dst() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (need_scratchpad_) {
                    load(vreg_tmp_src, interim_ptr(interim_axis_stride_ * i),
                            data_type::f32, tail);
                } else {
                    load(vreg_tmp_src, dst_ptr(dst_axis_stride_ * i),
                            dst_d_.data_type(), tail);
                }

                if (is_softmax_) {
                    uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                }
                if (is_logsoftmax_) {
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                }

                Vmm vscale = vmax;
                uni_vmovups(vscale, ptr[reg_src_scales]);
                uni_vmulps(vreg_tmp_src, vreg_tmp_src, vscale);
                // Reserved spot for post-ops injector.
                uni_vmovups(vscale, ptr[reg_dst_scales]);
                uni_vmulps(vreg_tmp_src, vreg_tmp_src, vscale);
                store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                        dst_d_.data_type(), tail);
            }
        });
    }

    void accumulate_vsbr() override {
        uni_vpxor(vsbr, vsbr, vsbr); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_dst = Vmm(i * 2 + 1);
                Vmm vreg_tmp_diff_dst = Vmm(i * 2 + 2);
                load(vreg_tmp_diff_dst, diff_dst_ptr(diff_dst_axis_stride_ * i),
                        diff_dst_d_.data_type(), tail);
                if (is_softmax_) {
                    load(vreg_tmp_dst, dst_ptr(dst_axis_stride_ * i),
                            dst_d_.data_type(), tail);
                    uni_vmulps(
                            vreg_tmp_diff_dst, vreg_tmp_diff_dst, vreg_tmp_dst);
                }
                uni_vaddps(vsbr, vsbr, vreg_tmp_diff_dst);
            }
        });

        get_horizontal_op(vsbr, vtmp = vmax, op_t::sum);
    }

    void compute_diff_src() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_dst = Vmm(i * 2 + 1);
                Vmm vreg_tmp_diff_dst = Vmm(i * 2 + 2);
                load(vreg_tmp_dst, dst_ptr(dst_axis_stride_ * i),
                        dst_d_.data_type(), tail);
                load(vreg_tmp_diff_dst, diff_dst_ptr(diff_dst_axis_stride_ * i),
                        diff_dst_d_.data_type(), tail);
                if (is_softmax_) {
                    vsubps(vreg_tmp_diff_dst, vreg_tmp_diff_dst, vsbr);
                    vmulps(vreg_tmp_diff_dst, vreg_tmp_dst, vreg_tmp_diff_dst);
                }
                if (is_logsoftmax_) {
                    exp_injector_->compute_vector(vreg_tmp_dst.getIdx());
                    uni_vfnmadd231ps(vreg_tmp_diff_dst, vreg_tmp_dst, vsbr);
                }
                store(diff_src_ptr(src_axis_stride_ * i), vreg_tmp_diff_dst,
                        src_d_.data_type(), tail);
            }
        });
    }

    void initialization_hook() override {
        if (bf16_emu_) bf16_emu_->init_vcvtneps2bf16();
    }

    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_base_t(pd) {
        if (is_bf16_ && !mayiuse(avx512_core_bf16))
            bf16_emu_.reset(new bf16_emulation_t(this, bf16_emu_zmm_1,
                    bf16_emu_zmm_2, bf16_emu_zmm_3, bf16_emu_gpr,
                    bf16_emu_zmm_4, bf16_emu_zmm_5));
    }

    void operator()(const call_params_t *p) override {
        return jit_generator::operator()(p);
    }
};

template <>
struct jit_softmax_t<avx2> : public jit_softmax_base_t<avx2> {
    Vmm tail_vmask = Vmm(0);

    void prepare_tail_mask() override {
        static const uint32_t mask_f32[14]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};
        mov(reg_tmp, reinterpret_cast<size_t>(&mask_f32[7 - axis_simd_tail_]));
        vmovups(tail_vmask, ptr[reg_tmp]);
    }

    void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) override {
        vperm2f128(vtmp, v, v, 0x1); // 128/256-bit shuffle
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0x4E); // 64/128-bit shuffle
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0xB1); // 32/64-bit shuffle
        perform_op(v, vtmp, op);
    }

    void accumulate_vmax() override {
        // flush to -FLT_MAX before accumulation
        uni_vmovups(vmax, vneg_flt_max);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                if (!tail)
                    uni_vmaxps(vmax, vmax, src_ptr(src_axis_stride_ * i));
                else {
                    vtmp = Vmm(i + 1);
                    uni_vmovups_tail(
                            vtmp, tail_vmask, src_ptr(src_axis_stride_ * i));
                    uni_vblendvps(vtmp, vneg_flt_max, vtmp, tail_vmask);
                    uni_vmaxps(vmax, vmax, vtmp);
                }
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vsum() override {
        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    uni_vmovups(vreg_tmp_src, src_ptr(src_axis_stride_ * i));
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) // store before applying exp
                        uni_vmovups(
                                dst_ptr(dst_axis_stride_ * i), vreg_tmp_src);
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    uni_vaddps(vsum, vsum, vreg_tmp_src);
                    if (is_softmax_) // store after applying exp
                        uni_vmovups(
                                dst_ptr(dst_axis_stride_ * i), vreg_tmp_src);
                } else {
                    uni_vmovups_tail(vreg_tmp_src, tail_vmask,
                            src_ptr(src_axis_stride_ * i));
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) // store before applying exp
                        uni_vmovups_tail(dst_ptr(dst_axis_stride_ * i),
                                tail_vmask, vreg_tmp_src);
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    vtmp = Vmm(vreg_tmp_src.getIdx() + 1);
                    uni_vpxor(vtmp, vtmp, vtmp);
                    uni_vblendvps(vtmp, vtmp, vreg_tmp_src, tail_vmask);
                    uni_vaddps(vsum, vsum, vtmp);
                    if (is_softmax_) // store after applying exp
                        uni_vmovups_tail(dst_ptr(dst_axis_stride_ * i),
                                tail_vmask, vreg_tmp_src);
                }
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_) uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void compute_dst() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    if (is_softmax_)
                        uni_vmulps(vreg_tmp_src, vsum,
                                dst_ptr(dst_axis_stride_ * i));
                    if (is_logsoftmax_) {
                        uni_vmovups(
                                vreg_tmp_src, dst_ptr(dst_axis_stride_ * i));
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                    }
                    uni_vmovups(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src);
                } else {
                    uni_vmovups_tail(vreg_tmp_src, tail_vmask,
                            dst_ptr(dst_axis_stride_ * i));
                    if (is_softmax_)
                        uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                    if (is_logsoftmax_)
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);

                    if (axis_is_blocked_) {
                        uni_vxorps(vzero, vzero, vzero);
                        uni_vblendvps(vzero, vzero, vreg_tmp_src, tail_vmask);
                        uni_vmovups(dst_ptr(dst_axis_stride_ * i), vzero);
                    } else {
                        uni_vmovups_tail(dst_ptr(dst_axis_stride_ * i),
                                tail_vmask, vreg_tmp_src);
                    }
                }
            }
        });
    }

    void operator()(const call_params_t *p) override {
        return jit_generator::operator()(p);
    }

    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_base_t(pd) {}
};

template <>
struct jit_softmax_t<sse41> : public jit_softmax_base_t<sse41> {
    Vmm tail_vmask = Vmm(0);

    void prepare_tail_mask() override {
        static const uint32_t mask_f32[4] = {0xffffffff, 0, 0, 0};
        mov(reg_tmp, reinterpret_cast<size_t>(mask_f32));
        movups(tail_vmask, ptr[reg_tmp]);
    }

    void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) override {
        uni_vmovups(vtmp, v);
        shufps(vtmp, vtmp, 0x4E); // 64/128-bit shuffle
        perform_op(v, vtmp, op);
        uni_vmovups(vtmp, v);
        shufps(vtmp, vtmp, 0xB1); // 32/64-bit shuffle
        perform_op(v, vtmp, op);
    }

    void accumulate_vmax() override {
        // flush to -FLT_MAX before accumulation
        uni_vmovups(vmax, vneg_flt_max);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    // SIGSEGV on unaligned addr if do maxps directly on memory
                    uni_vmovups(vreg_tmp_src, src_ptr(src_axis_stride_ * i));
                    uni_vmaxps(vmax, vmax, vreg_tmp_src);
                } else {
                    vtmp = Vmm(vreg_tmp_src.getIdx()
                            + 1); // next after vreg_tmp_src

                    for (size_t j = 0; j < axis_simd_tail_; j++) {
                        uni_vmovups(vreg_tmp_src, vneg_flt_max);
                        uni_vmovss(vtmp,
                                src_ptr(src_axis_stride_ * i
                                        + src_d_.data_type_size() * j));
                        uni_vblendvps(
                                vreg_tmp_src, vreg_tmp_src, vtmp, tail_vmask);
                        uni_vmaxps(vmax, vmax, vreg_tmp_src);
                    }
                }
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vsum() override {
        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    uni_vmovups(vreg_tmp_src, src_ptr(src_axis_stride_ * i));
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) // store before applying exp
                        uni_vmovups(
                                dst_ptr(dst_axis_stride_ * i), vreg_tmp_src);
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    uni_vaddps(vsum, vsum, vreg_tmp_src);
                    if (is_softmax_) // store after applying exp
                        uni_vmovups(
                                dst_ptr(dst_axis_stride_ * i), vreg_tmp_src);
                } else {
                    vtmp = Vmm(vreg_tmp_src.getIdx() + 1);
                    for (size_t j = 0; j < axis_simd_tail_; j++) {
                        uni_vmovss(vreg_tmp_src,
                                src_ptr(src_axis_stride_ * i
                                        + src_d_.data_type_size() * j));
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                        if (is_logsoftmax_) // store before applying exp
                            uni_vmovss(dst_ptr(dst_axis_stride_ * i
                                               + dst_d_.data_type_size() * j),
                                    vreg_tmp_src);
                        exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                        uni_vpxor(vtmp, vtmp, vtmp);
                        uni_vblendvps(vtmp, vtmp, vreg_tmp_src, tail_vmask);
                        uni_vaddps(vsum, vsum, vtmp);
                        if (is_softmax_) // store after applying exp
                            uni_vmovss(dst_ptr(dst_axis_stride_ * i
                                               + dst_d_.data_type_size() * j),
                                    vreg_tmp_src);
                    }
                }
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_) uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void compute_dst() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    uni_vmovups(vreg_tmp_src, dst_ptr(dst_axis_stride_ * i));
                    if (is_softmax_)
                        uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                    if (is_logsoftmax_)
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                    uni_vmovups(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src);
                } else {
                    for (size_t j = 0; j < axis_simd_tail_; j++) {
                        uni_vmovss(vreg_tmp_src,
                                dst_ptr(dst_axis_stride_ * i
                                        + dst_d_.data_type_size() * j));
                        if (is_softmax_)
                            uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                        if (is_logsoftmax_)
                            uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                        uni_vmovss(dst_ptr(dst_axis_stride_ * i
                                           + dst_d_.data_type_size() * j),
                                vreg_tmp_src);
                    }
                }
            }
        });
    }

    void operator()(const call_params_t *p) override {
        return jit_generator::operator()(p);
    }

    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_base_t(pd) {}
};

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
template struct jit_uni_softmax_fwd_t<sse41>;
template struct jit_uni_softmax_fwd_t<avx2>;
template struct jit_uni_softmax_fwd_t<avx512_core>;
template struct jit_uni_softmax_bwd_t<avx512_core>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
