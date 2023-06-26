/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_uni_softmax.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

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

namespace softmax_impl {
using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_softmax_kernel_t : jit_softmax_kernel_base_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_softmax_kernel_t)

    // cpu specific part
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = is_superset(isa, avx512_core) ? zword
            : is_superset(isa, avx)                             ? yword
                                                                : xword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    const memory_desc_wrapper src_d_, dst_d_, diff_dst_d_;
    io::jit_io_multi_dt_helper_t<Vmm> io_;

    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector_;
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> log_injector_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;

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
    Vmm vneg_flt_max = Vmm(is_superset(isa, avx512_core) ? 28 : 12);
    Xmm xone = Xmm(13);
    Vmm vone = Vmm(is_superset(isa, avx512_core) ? 29 : 13);
    Vmm vsum = Vmm(is_superset(isa, avx512_core) ? 30 : 14);
    Vmm vmax = Vmm(is_superset(isa, avx512_core) ? 31 : 15);
    Vmm vsbr = vsum; // must be not equal to vmax
    Vmm vzero = Vmm(is_superset(isa, avx512_core) ? 21 : 11);
    Vmm vcvt_vmm = Vmm(is_superset(isa, avx512_core) ? 22 : 10);
    Vmm vsaturation_ubound = vneg_flt_max;

    bool is_bf16_ = false;
    bool is_f16_ = false;
    bool is_avx2_ne_xf16_ = false;
    bool is_softmax_ = pd_->is_softmax();
    bool is_logsoftmax_ = pd_->is_logsoftmax();
    bool axis_is_blocked_;
    bool need_scratchpad_;
    bool with_postops_ = false;
    bool with_binary_ = false;
    bool with_eltwise_ = false;

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

    const int bf16_emu_zmm_1_idx_ = 23;
    const int bf16_emu_zmm_2_idx_ = 24;
    const int bf16_emu_zmm_3_idx_ = 25;
    const int bf16_emu_zmm_4_idx_ = 26;
    const int tail_opmask_idx_ = 2;

    Opmask tail_opmask = Opmask(tail_opmask_idx_);

    void operator()(const call_params_t *p) const override {
        return jit_generator::operator()(p);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    bool is_data_type_xf16(data_type_t dt) {
        return utils::one_of(dt, data_type::bf16, data_type::f16);
    }

    void compute_predefined_variables() {
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

    void get_horizontal_op(const Vmm &vsrc, const Vmm &vtmp, op_t op) {
        const Zmm &zsrc = Zmm(vsrc.getIdx());
        const Zmm &ztmp = Zmm(vtmp.getIdx());
        const Ymm &ysrc = Ymm(vsrc.getIdx());
        const Ymm &ytmp = Ymm(vtmp.getIdx());

        if (is_superset(isa, avx512_core)) {
            vshuff32x4(ztmp, zsrc, zsrc, 0x4E); // 256-bit shuffle
            perform_op(vsrc, vtmp, op);
            vshuff32x4(ztmp, zsrc, zsrc, 0xB1); // 128/256-bit shuffle
            perform_op(vsrc, vtmp, op);
        } else if (is_superset(isa, avx2)) {
            vperm2f128(ytmp, ysrc, ysrc, 0x1); // 128/256-bit shuffle
            perform_op(vsrc, vtmp, op);
        }
        uni_vshufps(vtmp, vsrc, vsrc, 0x4E); // 64/128-bit shuffle
        perform_op(vsrc, vtmp, op);
        uni_vshufps(vtmp, vsrc, vsrc, 0xB1); // 32/64-bit shuffle
        perform_op(vsrc, vtmp, op);
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

    void uni_vaddps_maybe_tail(
            const Vmm &v1, const Vmm &v2, const Vmm &vtmp, const bool tail) {
        if (tail) {
            if (is_superset(isa, avx512_core)) {
                uni_vaddps(v1 | tail_opmask, v1, v2);
            } else {
                uni_vpxor(vtmp, vtmp, vtmp);
                uni_vblendvps(vtmp, vtmp, v2, tail_vmask);
                uni_vaddps(v1, v1, vtmp);
            }
        } else
            uni_vaddps(v1, v1, v2);
    }

    void uni_vmaxps_maybe_tail(
            const Vmm &v1, const Vmm &v2, const Vmm &vtmp, const bool tail) {
        if (tail) {
            if (is_superset(isa, avx512_core)) {
                uni_vmaxps(v1 | tail_opmask, v1, v2);
            } else if (is_superset(isa, avx)) {
                uni_vblendvps(v2, vneg_flt_max, v2, tail_vmask);
                uni_vmaxps(v1, v1, v2);
            } else {
                uni_vmovups(vtmp, v2);
                uni_vmovups(v2, vneg_flt_max);
                uni_vblendvps(v2, v2, vtmp, tail_vmask);
                uni_vmaxps(v1, v1, v2);
            }
        } else
            uni_vmaxps(v1, v1, v2);
    }

    void store(const Address &addr, const Vmm &vmm, data_type_t dt,
            bool tail = false) {
        // Use temporary register in storing when convertion is needed
        // Or we need to restore data back to fp32 since we apply exp after
        // storing and data should be fp32
        const bool need_restore = is_logsoftmax_ && dt != data_type::f32;
        Vmm src_vmm = vmm;

        if (tail && axis_is_blocked_) {
            if (is_superset(isa, avx512_core)
                    && utils::one_of(dt, data_type::f32, data_type::bf16,
                            data_type::f16)) {
                src_vmm = vzero | tail_opmask;
                uni_vxorps(vzero, vzero, vzero);
                uni_vmovups(src_vmm, vmm);
                src_vmm = vzero;
            } else {
                uni_vpxor(vzero, vzero, vzero);
                uni_vblendvps(vzero, vzero, src_vmm, tail_vmask);
                src_vmm = vzero;
            }
        } else if (need_restore) {
            uni_vmovups(vcvt_vmm, vmm);
            src_vmm = vcvt_vmm;
        }

        io_[dt]->store(src_vmm, addr, tail && !axis_is_blocked_);
    }

    // Use ne_convert instruction to load xf16 even/odd elements from memory
    void accumulate_avx2_ne_xf16_vmax() {
        // flush to -FLT_MAX before accumulation
        uni_vmovups(vmax, vneg_flt_max);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i += 2) {
                const bool can_load_two_simdw = unroll - i >= 2;
                Vmm vreg_tmp_src_even = Vmm(i + 1);
                Vmm vreg_tmp_src_odd = Vmm(i + 2);
                vtmp = Vmm(i + 3);
                if (can_load_two_simdw) {
                    io_[src_d_.data_type()]->load_two_simdw_xf16(
                            src_ptr(src_axis_stride_ * i), vreg_tmp_src_even,
                            vreg_tmp_src_odd);
                } else
                    io_[src_d_.data_type()]->load(src_ptr(src_axis_stride_ * i),
                            vreg_tmp_src_even, tail);
                uni_vmaxps_maybe_tail(vmax, vreg_tmp_src_even, vtmp, tail);
                if (can_load_two_simdw)
                    uni_vmaxps_maybe_tail(vmax, vreg_tmp_src_odd, vtmp, tail);
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vmax() {
        if (is_avx2_ne_xf16_ && is_data_type_xf16(src_d_.data_type())) {
            accumulate_avx2_ne_xf16_vmax();
            return;
        }

        // flush to -FLT_MAX before accumulation
        uni_vmovups(vmax, vneg_flt_max);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                vtmp = Vmm(i + 2);
                // do maxps directly from memory on f32 avx2 for performance purpose
                if (!tail && is_superset(isa, avx2)
                        && !is_superset(isa, avx512_core)
                        && src_d_.data_type() == data_type::f32) {
                    uni_vmaxps(vmax, vmax, src_ptr(src_axis_stride_ * i));
                } else {
                    io_[src_d_.data_type()]->load(
                            src_ptr(src_axis_stride_ * i), vreg_tmp_src, tail);
                    uni_vmaxps_maybe_tail(vmax, vreg_tmp_src, vtmp, tail);
                }
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    // Use ne_convert instruction to load xf16 even/odd elements from memory
    void accumulate_avx2_ne_xf16_vsum() {
        // Initialize saturation vector register
        io_.init_saturate_f32({dst_d_.data_type()});

        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i += 2) {
                const bool can_load_two_simdw = unroll - i >= 2;
                Vmm vreg_tmp_src_even = Vmm(i + 1);
                Vmm vreg_tmp_src_odd = Vmm(i + 2);
                vtmp = Vmm(i + 3);
                if (can_load_two_simdw) {
                    io_[src_d_.data_type()]->load_two_simdw_xf16(
                            src_ptr(src_axis_stride_ * i), vreg_tmp_src_even,
                            vreg_tmp_src_odd);
                    io_[src_d_.data_type()]->merge_interleaved_to_plain(
                            vreg_tmp_src_even, vreg_tmp_src_odd, vtmp);
                } else
                    io_[src_d_.data_type()]->load(src_ptr(src_axis_stride_ * i),
                            vreg_tmp_src_even, tail);
                for (int i_odd = 0; i_odd < 2 && i_odd + i < unroll; i_odd++) {
                    const auto vreg_tmp_src
                            = i_odd ? vreg_tmp_src_odd : vreg_tmp_src_even;
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) // store before applying exp
                        store(dst_ptr(dst_axis_stride_ * (i + i_odd)),
                                vreg_tmp_src, dst_d_.data_type(), tail);
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    uni_vaddps_maybe_tail(vsum, vreg_tmp_src, vtmp, tail);
                    if (is_softmax_) // store after applying exp
                        store(dst_ptr(dst_axis_stride_ * (i + i_odd)),
                                vreg_tmp_src, dst_d_.data_type(), tail);
                }
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_) uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void accumulate_vsum() {
        if (is_avx2_ne_xf16_ && is_data_type_xf16(src_d_.data_type())) {
            accumulate_avx2_ne_xf16_vsum();
            return;
        }

        // Initialize saturation vector register
        io_.init_saturate_f32({dst_d_.data_type()});

        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                vtmp = Vmm(i + 2);
                io_[src_d_.data_type()]->load(
                        src_ptr(src_axis_stride_ * i), vreg_tmp_src, tail);
                uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                if (is_logsoftmax_) { // store before applying exp
                    if (need_scratchpad_)
                        store(interim_ptr(interim_axis_stride_ * i),
                                vreg_tmp_src, data_type::f32, tail);
                    else
                        store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                                dst_d_.data_type(), tail);
                }
                exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                uni_vaddps_maybe_tail(vsum, vreg_tmp_src, vtmp, tail);
                if (is_softmax_) { // store after applying exp
                    if (need_scratchpad_)
                        store(interim_ptr(interim_axis_stride_ * i),
                                vreg_tmp_src, data_type::f32, tail);
                    else
                        store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                                dst_d_.data_type(), tail);
                }
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_) uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    // Use ne_convert instruction to load xf16 even/odd elements from memory
    void compute_avx2_ne_xf16_dst() {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i += 2) {
                const bool can_load_two_simdw = unroll - i >= 2;
                Vmm vreg_tmp_src_even = Vmm(i + 1);
                Vmm vreg_tmp_src_odd = Vmm(i + 2);
                vtmp = Vmm(i + 3);
                if (can_load_two_simdw) {
                    io_[dst_d_.data_type()]->load_two_simdw_xf16(
                            dst_ptr(dst_axis_stride_ * i), vreg_tmp_src_even,
                            vreg_tmp_src_odd);
                    io_[dst_d_.data_type()]->merge_interleaved_to_plain(
                            vreg_tmp_src_even, vreg_tmp_src_odd, vtmp);
                } else
                    io_[dst_d_.data_type()]->load(dst_ptr(dst_axis_stride_ * i),
                            vreg_tmp_src_even, tail);
                for (int i_odd = 0; i_odd < 2 && i_odd + i < unroll; i_odd++) {
                    const auto vreg_tmp_src
                            = i_odd ? vreg_tmp_src_odd : vreg_tmp_src_even;
                    if (is_softmax_)
                        uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                    if (is_logsoftmax_)
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);

                    if (with_postops_) {
                        binary_injector::rhs_arg_dynamic_params_t
                                rhs_arg_params;
                        if (with_binary_) {
                            rhs_arg_params.vmm_idx_to_out_addr.emplace(
                                    vreg_tmp_src.getIdx(), dst_ptr());
                            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                                    vreg_tmp_src.getIdx(),
                                    dst_axis_stride_ * i);
                            if (tail)
                                rhs_arg_params.vmm_tail_idx_.emplace(
                                        vreg_tmp_src.getIdx());
                        }
                        postops_injector_->compute_vector(
                                vreg_tmp_src.getIdx(), rhs_arg_params);
                    }

                    store(dst_ptr(dst_axis_stride_ * (i + i_odd)), vreg_tmp_src,
                            dst_d_.data_type(), tail);
                }
            }
        });
    }

    void compute_dst() {
        if (is_avx2_ne_xf16_ && is_data_type_xf16(dst_d_.data_type())) {
            compute_avx2_ne_xf16_dst();
            return;
        }

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (need_scratchpad_)
                    io_[data_type::f32]->load(
                            interim_ptr(interim_axis_stride_ * i), vreg_tmp_src,
                            tail);
                else
                    io_[dst_d_.data_type()]->load(
                            dst_ptr(dst_axis_stride_ * i), vreg_tmp_src, tail);

                if (is_softmax_) uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                if (is_logsoftmax_)
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);

                if (is_superset(isa, avx512_core)) {
                    Vmm vscale = vmax;
                    uni_vmovups(vscale, ptr[reg_src_scales]);
                    uni_vmulps(vreg_tmp_src, vreg_tmp_src, vscale);
                }
                if (with_postops_) {
                    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
                    if (with_binary_) {
                        rhs_arg_params.vmm_idx_to_out_addr.emplace(
                                vreg_tmp_src.getIdx(), dst_ptr());
                        rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                                vreg_tmp_src.getIdx(), dst_axis_stride_ * i);
                        if (tail)
                            rhs_arg_params.vmm_tail_idx_.emplace(
                                    vreg_tmp_src.getIdx());
                    }
                    postops_injector_->compute_vector(
                            vreg_tmp_src.getIdx(), rhs_arg_params);
                }
                if (is_superset(isa, avx512_core)) {
                    Vmm vscale = vmax;
                    uni_vmovups(vscale, ptr[reg_dst_scales]);
                    uni_vmulps(vreg_tmp_src, vreg_tmp_src, vscale);
                }
                store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                        dst_d_.data_type(), tail);
            }
        });
    }

    void accumulate_vsbr() {
        uni_vpxor(vsbr, vsbr, vsbr); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_dst = Vmm(i * 2 + 1);
                Vmm vreg_tmp_diff_dst = Vmm(i * 2 + 2);
                io_[diff_dst_d_.data_type()]->load(
                        diff_dst_ptr(diff_dst_axis_stride_ * i),
                        vreg_tmp_diff_dst, tail);
                if (is_softmax_) {
                    io_[dst_d_.data_type()]->load(
                            dst_ptr(dst_axis_stride_ * i), vreg_tmp_dst, tail);
                    uni_vmulps(
                            vreg_tmp_diff_dst, vreg_tmp_diff_dst, vreg_tmp_dst);
                }
                uni_vaddps(vsbr, vsbr, vreg_tmp_diff_dst);
            }
        });

        get_horizontal_op(vsbr, vtmp = vmax, op_t::sum);
    }

    void compute_diff_src() {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_dst = Vmm(i * 2 + 1);
                Vmm vreg_tmp_diff_dst = Vmm(i * 2 + 2);
                io_[dst_d_.data_type()]->load(
                        dst_ptr(dst_axis_stride_ * i), vreg_tmp_dst, tail);
                io_[diff_dst_d_.data_type()]->load(
                        diff_dst_ptr(diff_dst_axis_stride_ * i),
                        vreg_tmp_diff_dst, tail);
                if (is_softmax_) {
                    uni_vsubps(vreg_tmp_diff_dst, vreg_tmp_diff_dst, vsbr);
                    uni_vmulps(
                            vreg_tmp_diff_dst, vreg_tmp_dst, vreg_tmp_diff_dst);
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
        if (with_postops_) {
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = true;
            static constexpr std::size_t tmp_vmm_injector = 0u;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    tmp_vmm_injector, this->r14, this->r15, this->r13,
                    preserve_gpr, preserve_vmm,
                    PARAM_OFF(post_ops_binary_rhs_arg_vec), PARAM_OFF(dst_orig),
                    dst_d_, axis_simd_tail_, tail_opmask,
                    use_exact_tail_scalar_bcast};

            const binary_injector::static_params_t bsp {
                    reg_param, get_supported_bcast_strategies(), rhs_sp};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<isa>>(
                    this, pd_->attr()->post_ops_, bsp);
        }
#undef PARAM_OFF

        compute_predefined_variables();
        preamble();
        io_.init_bf16();
        if (exp_injector_) exp_injector_->load_table_addr();
        if (log_injector_) log_injector_->load_table_addr();
        if (axis_simd_tail_) io_.prepare_tail_mask();
        load_common_params();
        if (pd_->is_fwd())
            forward();
        else
            backward();
        postamble();
        if (exp_injector_) exp_injector_->prepare_table();
        if (log_injector_) log_injector_->prepare_table();
        if (with_eltwise_ && postops_injector_)
            postops_injector_->prepare_table();
    }

    jit_softmax_kernel_t(const softmax_pd_t *pd)
        : jit_softmax_kernel_base_t(pd)
        , jit_generator(jit_name(), nullptr, MAX_CODE_SIZE, true, isa)
        , src_d_(pd_->is_fwd() ? pd_->src_md() : pd_->diff_src_md())
        , dst_d_(pd_->dst_md())
        , diff_dst_d_(pd_->diff_dst_md()) {
        is_bf16_ = utils::one_of(
                data_type::bf16, src_d_.data_type(), dst_d_.data_type());
        is_f16_ = utils::one_of(
                data_type::f16, src_d_.data_type(), dst_d_.data_type());
        simd_w_ = vlen / sizeof(float); // bf16 works on ymms
        is_avx2_ne_xf16_ = mayiuse(avx2_vnni_2) && !mayiuse(avx512_core)
                && (is_bf16_ || is_f16_);
        axis_simd_full_ = pd_->axis_size() / simd_w_;
        axis_simd_tail_ = pd_->axis_size() % simd_w_;
        need_scratchpad_ = utils::one_of(
                dst_d_.data_type(), data_type::u8, data_type::s8);

        const auto &post_ops = pd_->attr()->post_ops_;
        with_postops_ = post_ops.len() != 0;
        with_binary_ = post_ops.find(primitive_kind::binary) != -1;
        with_eltwise_ = post_ops.find(primitive_kind::eltwise) != -1;

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, axis_simd_tail_,
                tail_opmask_idx_, tail_vmask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(bf16_emu_zmm_1_idx_,
                bf16_emu_zmm_2_idx_, bf16_emu_zmm_3_idx_, reg_tmp,
                bf16_emu_zmm_4_idx_);
        io::io_saturation_conf_t io_saturation_conf(
                vzero.getIdx(), vsaturation_ubound.getIdx(), reg_tmp);
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, isa,
                {src_d_.data_type(), dst_d_.data_type(),
                        data_type::f32 /* stats */},
                io_conf, io_tail_conf, io_bf16_conf,
                {{dst_d_.data_type(), io_saturation_conf}});
    }
};

jit_softmax_kernel_base_t *jit_softmax_kernel_base_t::create(
        const softmax_pd_t *pd, const cpu_isa_t isa) {
#define HANDLE_ISA(isa_) \
    if ((isa_) == isa) return new jit_softmax_kernel_t<isa_>(pd)
    REG_AVX512_ISA(HANDLE_ISA(avx512_core_fp16));
    REG_AVX512_ISA(HANDLE_ISA(avx512_core_bf16));
    REG_AVX512_ISA(HANDLE_ISA(avx512_core));
    REG_AVX2_ISA(HANDLE_ISA(avx2_vnni_2));
    REG_AVX2_ISA(HANDLE_ISA(avx2));
    REG_SSE41_ISA(HANDLE_ISA(sse41));
#undef HANDLE_ISA
    assert(!"kernel is empty.");
    return nullptr;
}

std::vector<cpu_isa_t> get_supported_isa(bool is_fwd) {
    if (is_fwd)
        return {avx512_core_fp16, avx512_core_bf16, avx512_core, avx2_vnni_2,
                avx2, sse41};
    else
        return {avx512_core_fp16, avx512_core_bf16, avx512_core};
}

bcast_set_t get_supported_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::no_broadcast};
}
} // namespace softmax_impl

jit_uni_softmax_fwd_t::jit_uni_softmax_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

status_t jit_uni_softmax_fwd_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(ker_,
            softmax_impl::jit_softmax_kernel_base_t::create(pd(), pd()->isa_)));
    if (ker_) CHECK(ker_->create_kernel());
    return status::success;
}

status_t jit_uni_softmax_fwd_t::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    auto scratchpad_ptr = ctx.get_scratchpad_grantor().template get<char>(
            memory_tracking::names::key_softmax_interim_store);

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->attr()->post_ops_, ctx);

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

    const char *dst_orig_ptr = dst;
    parallel_nd_ext(nthr, outer_size, inner_size,
            [&](int ithr, int, dim_t ou, dim_t in) {
                dim_t offset = (ou * outer_stride + in * inner_stride);
                const char *src_ptr = src + offset * src_data_type_size;
                char *dst_ptr = dst + offset * dst_data_type_size;
                char *interim_ptr = scratchpad_ptr ? scratchpad_ptr
                                + ithr * axis_size_padded * sizeof(float)
                                                   : nullptr;
                softmax_impl::jit_softmax_kernel_base_t::call_params_t p;
                p.process_n_elems = process_n_elems;
                p.src = src_ptr;
                p.dst = dst_ptr;
                p.interim = interim_ptr;
                p.src_scales = src_scales;
                p.dst_scales = dst_scales;
                // post-ops
                p.dst_orig = dst_orig_ptr;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                (*ker_)(&p);
            });

    return status::success;
}

jit_uni_softmax_bwd_t::jit_uni_softmax_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

status_t jit_uni_softmax_bwd_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(ker_,
            softmax_impl::jit_softmax_kernel_base_t::create(pd(), pd()->isa_)));
    if (ker_) CHECK(ker_->create_kernel());
    return status::success;
}

status_t jit_uni_softmax_bwd_t::execute(const exec_ctx_t &ctx) const {
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
        softmax_impl::jit_softmax_kernel_base_t::call_params_t p;
        p.process_n_elems = process_n_elems;
        p.src = diff_src_ptr;
        p.dst = dst_ptr;
        p.diff_dst = diff_dst_ptr;
        (*ker_)(&p);
    });

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
