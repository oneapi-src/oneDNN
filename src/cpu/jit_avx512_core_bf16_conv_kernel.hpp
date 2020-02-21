/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef JIT_AVX512_CORE_BF16_CONV_KERNEL_HPP
#define JIT_AVX512_CORE_BF16_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "jit_avx512_core_bf16cvt.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <typename Vmm>
struct _jit_avx512_core_bf16_fwd_kernel : public jit_generator {

    _jit_avx512_core_bf16_fwd_kernel(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jit_generator(nullptr, ker_code_size)
        , jcp(ajcp)
        , attr_(attr)
        , eltwise_injector_(nullptr)
        , bf16_emu_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_core>(
                    this, jcp.eltwise);
        if (!isa_has_bf16(jcp.isa))
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                    bf16_emu_reserv_4, bf16_emu_reserv_5);

        generate();
        jit_ker_ = (decltype(jit_ker_))getCode();
    }

    ~_jit_avx512_core_bf16_fwd_kernel() {
        delete bf16_emu_;
        delete eltwise_injector_;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_core_bf16_fwd_kernel)

    const jit_conv_conf_t &jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker_)(jit_conv_call_s *);

private:
    using Vmm_down_t =
            typename utils::conditional<std::is_same<Vmm, Xbyak::Zmm>::value,
                    Xbyak::Ymm, Xbyak::Xmm>::type;
    using reg64_t = const Xbyak::Reg64;
    enum {
        ker_reg_base_idx = 28,
        ker_code_size = 1024 * 1024,
    };

    reg64_t param = abi_param1; //L: RDI, W: RCX

    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;
    reg64_t reg_owb = r11;

    reg64_t aux_reg_inp = r12;
    reg64_t aux_reg_ker = r13;

    reg64_t aux_reg_ker_d = r14;
    reg64_t aux_reg_inp_d = r15;

    reg64_t reg_icb = rax;
    reg64_t reg_bias = rbx;

    reg64_t reg_kj = abi_not_param1;
    reg64_t reg_ki = reg_bias;
    reg64_t reg_oi = rdx;
    reg64_t reg_kh = rsi;

    reg64_t reg_out_long_offt = r14;

    Vmm vmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Vmm(idx);
    }

    Vmm vmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Vmm(idx);
    }

    Vmm_down_t vmm_inp_down(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Vmm_down_t(idx);
    }

    Vmm vmm_wei = Vmm(31);
    Vmm vmm_prev_dst = Vmm(31);
    Vmm vmm_bias = Vmm(31);

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(28);
    reg64_t bf16_emu_scratch = reg_icb;
    Xbyak::Zmm bf16_emu_reserv_4 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(30);

    Xbyak::Opmask odd_load_mask = Xbyak::Opmask(1);
    Xbyak::Opmask even_load_mask = Xbyak::Opmask(2);

    jit_uni_eltwise_injector_f32<avx512_core> *eltwise_injector_;
    bf16_emulation_t *bf16_emu_;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop(int ur_w, int pad_l, int pad_r);

    void generate();

    size_t get_output_offset(int oi, int n_oc_block) {
        return (size_t)jcp.typesize_out
                * ((size_t)n_oc_block * jcp.oh * jcp.ow * jcp.od + oi)
                * jcp.oc_block;
    }

    size_t get_input_offset(int ki, int ic, int oi, int pad_l) {
        size_t scale = 2; //bf16 vnni is used
        size_t iw_str = jcp.is_1stconv ? 1 : jcp.ic_block;
        size_t ic_str = jcp.is_1stconv ? (size_t)jcp.iw * jcp.ih * jcp.id : 1;
        return (size_t)jcp.typesize_in
                * ((size_t)(ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l)
                                * iw_str
                        + scale * ic * ic_str);
    }

    size_t get_kernel_offset(int ki, int ic, int n_oc_block, int ker_number) {
        int scale = 2; //bf16 vnni is used
        int rnd_ic_block = utils::rnd_up(jcp.ic_block, scale);

        size_t oc_block_stride
                = (size_t)jcp.nb_ic * rnd_ic_block * jcp.kh * jcp.kw * jcp.kd;
        return jcp.typesize_in * jcp.oc_block
                * (n_oc_block * oc_block_stride + (ic + ker_number) * scale
                        + ki * rnd_ic_block);
    }

    int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }

    int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w
                - nstl::max(0,
                        utils::div_up(
                                pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1),
                                jcp.stride_w));
    }
};

struct jit_avx512_core_bf16_fwd_kernel {
    jit_avx512_core_bf16_fwd_kernel(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jit_ker(nullptr)
        , zmm_kernel_(nullptr)
        , ymm_kernel_(nullptr)
        , xmm_kernel_(nullptr) {
        switch (ajcp.oc_block) {
            case 16:
                zmm_kernel_ = new _jit_avx512_core_bf16_fwd_kernel<Xbyak::Zmm>(
                        ajcp, attr);
                jit_ker = zmm_kernel_->jit_ker_;
                return;
            case 8:
                ymm_kernel_ = new _jit_avx512_core_bf16_fwd_kernel<Xbyak::Ymm>(
                        ajcp, attr);
                jit_ker = ymm_kernel_->jit_ker_;
                return;
            case 4:
                xmm_kernel_ = new _jit_avx512_core_bf16_fwd_kernel<Xbyak::Xmm>(
                        ajcp, attr);
                jit_ker = xmm_kernel_->jit_ker_;
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    ~jit_avx512_core_bf16_fwd_kernel() {
        delete zmm_kernel_;
        delete ymm_kernel_;
        delete xmm_kernel_;
    }

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, const primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void (*jit_ker)(jit_conv_call_s *);
    _jit_avx512_core_bf16_fwd_kernel<Xbyak::Zmm> *zmm_kernel_;
    _jit_avx512_core_bf16_fwd_kernel<Xbyak::Ymm> *ymm_kernel_;
    _jit_avx512_core_bf16_fwd_kernel<Xbyak::Xmm> *xmm_kernel_;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_avx512_core_bf16_fwd_kernel);
};

template <typename Vmm>
struct _jit_avx512_core_bf16_bwd_data_kernel : public jit_generator {

    _jit_avx512_core_bf16_bwd_data_kernel(const jit_conv_conf_t &ajcp)
        : jit_generator(nullptr, ker_code_size), jcp(ajcp), bf16_emu_(nullptr) {
        if (!isa_has_bf16(jcp.isa))
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                    bf16_emu_reserv_4, bf16_emu_reserv_5);
        generate();
        jit_ker_ = (decltype(jit_ker_))getCode();
    }

    ~_jit_avx512_core_bf16_bwd_data_kernel() { delete bf16_emu_; }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_core_bf16_bwd_data_kernel_f32)

    const jit_conv_conf_t &jcp;
    void (*jit_ker_)(jit_conv_call_s *);

private:
    using Vmm_down_t =
            typename utils::conditional<std::is_same<Vmm, Xbyak::Zmm>::value,
                    Xbyak::Ymm, Xbyak::Xmm>::type;
    using reg64_t = const Xbyak::Reg64;
    enum {
        ker_reg_base_idx = 31,
        ker_code_size = 1024 * 1024,
    };

    reg64_t param = abi_param1;
    reg64_t reg_dst = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_src = r10;

    reg64_t reg_iwb = rdx;

    reg64_t aux_reg_dst = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_dst_d = r12;
    reg64_t aux_reg_ker_d = r13;
    reg64_t reg_ki = rsi;

    reg64_t reg_kj = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_ocb = r11;

    Vmm vmm_inp(int i_ic) {
        int idx = i_ic + jcp.nb_ic_blocking * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Vmm(idx);
    }

    Vmm_down_t vmm_inp_down(int i_ic) {
        int idx = i_ic + jcp.nb_ic_blocking * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Vmm_down_t(idx);
    }

    Vmm vmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Vmm(idx);
    }

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(28);
    reg64_t bf16_emu_scratch = reg_kj;
    Xbyak::Zmm bf16_emu_reserv_4 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(30);

    Vmm vmm_wei = Vmm(31);
    bf16_emulation_t *bf16_emu_;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop(int ur_w, int l_overflow, int r_overflow);
    void generate();

    int get_iw_start(int ki, int l_overflow) {
        int res = (jcp.iw - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return res;
    }

    int get_iw_end(int ur_w, int ki, int r_overflow) {
        if (utils::one_of(ur_w, jcp.iw, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }
};

struct jit_avx512_core_bf16_bwd_data_kernel {

    jit_avx512_core_bf16_bwd_data_kernel(const jit_conv_conf_t &ajcp)
        : jit_ker(nullptr)
        , zmm_kernel_(nullptr)
        , ymm_kernel_(nullptr)
        , xmm_kernel_(nullptr) {
        switch (ajcp.ic_block) {
            case 16:
                zmm_kernel_
                        = new _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Zmm>(
                                ajcp);
                jit_ker = zmm_kernel_->jit_ker_;
                return;
            case 8:
                ymm_kernel_
                        = new _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Ymm>(
                                ajcp);
                jit_ker = ymm_kernel_->jit_ker_;
                return;
            case 4:
                xmm_kernel_
                        = new _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Xmm>(
                                ajcp);
                jit_ker = xmm_kernel_->jit_ker_;
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    ~jit_avx512_core_bf16_bwd_data_kernel() {
        delete zmm_kernel_;
        delete ymm_kernel_;
        delete xmm_kernel_;
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &diff_src_md,
            memory_desc_t &weights_md, memory_desc_t &diff_dst_md,
            int nthreads);

    void (*jit_ker)(jit_conv_call_s *);
    _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Zmm> *zmm_kernel_;
    _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Ymm> *ymm_kernel_;
    _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Xmm> *xmm_kernel_;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_avx512_core_bf16_bwd_data_kernel);
};

struct jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 : public jit_generator {

    jit_avx512_core_bf16_conv_bwd_weights_kernel_f32(
            const jit_conv_conf_t &ajcp)
        : jit_generator(nullptr, ker_code_size), jcp(ajcp), bf16_emu_(nullptr) {
        if (!isa_has_bf16(jcp.isa)) {
            bf16_emu_ = new bf16_emulation_t(
                    this, one, even, selector, scratch, tmp0, tmp1);
        }
        generate();
        jit_ker = (decltype(jit_ker))getCode();
    }

    ~jit_avx512_core_bf16_conv_bwd_weights_kernel_f32() { delete bf16_emu_; }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_avx512_core_bf16_conv_bwd_weights_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
            memory_desc_t &diff_dst_md, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    const jit_conv_conf_t &jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    Xbyak::Label dst_prm_table;
    // Used by compute_ic_block_step_{vpermw, interleave}
    Xbyak::Opmask m_ffffffff = Xbyak::Opmask(1);
    // Used by compute_ic_block_step_vpermw
    Xbyak::Opmask m_0000ffff = Xbyak::Opmask(2);
    Xbyak::Opmask m_ffff0000 = Xbyak::Opmask(3);
    // Used by compute_ic_block_step_extern (1st_conv only)
    Xbyak::Opmask everyother_mask = Xbyak::Opmask(6);
    Xbyak::Opmask everyother_shift_mask = Xbyak::Opmask(7);
    // Used by compute_ic_block_step_interleave (1st_conv only)
    Xbyak::Opmask underflow_mask = Xbyak::Opmask(4);
    Xbyak::Opmask overflow_mask = Xbyak::Opmask(5);
    Xbyak::Opmask underflow_stride_mask = Xbyak::Opmask(6);
    Xbyak::Opmask overflow_stride_mask = Xbyak::Opmask(7);

    using reg64_t = const Xbyak::Reg64;
    enum {
        sizeof_cacheline = 64,
        full_spat_opt_working_set_size = 48 * 1024,
        full_spat_max_working_set_size = 128 * 1024,
        ker_code_size = 1024 * 1024,
    };
    static const int max_ur_w;

    reg64_t param = abi_param1;
    reg64_t reg_input = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_output = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh = r9;
    reg64_t reg_ur_w_trips = r10;
    reg64_t reg_oj = r15;
    reg64_t reg_tmp = r14;
    reg64_t reg_ih_shift = reg_tmp;
    reg64_t reg_long_offt = r14;

    reg64_t ki = r11;
    reg64_t reg_oj_setup = r11;
    reg64_t reg_kd_count = r12;
    reg64_t reg_oi = r12;
    reg64_t reg_d_index = r13;
    reg64_t reg_input_d = r15;
    reg64_t reg_output_d = rbx;
    reg64_t aux_reg_input = r12;
    reg64_t aux_reg_kernel = r13;
    reg64_t reg_bias = rbx;

    Xbyak::Zmm one = Xbyak::Zmm(27);
    Xbyak::Zmm even = Xbyak::Zmm(28);
    Xbyak::Zmm selector = Xbyak::Zmm(29);
    Xbyak::Zmm tmp0 = Xbyak::Zmm(30);
    Xbyak::Zmm tmp1 = Xbyak::Zmm(31);
    reg64_t scratch = r11;

    inline void maybe_zero_kernel();
    inline void get_ur_w(int &ur_w, int &ur_w_tail, int &ur_w_trips);
    inline void compute_oh_step_unroll_ow_icblock(int ic_block_step);
    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_oh_step_unroll_ow(int ic_block_step);
    inline void compute_ic_block_step(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset, bool is_tail = false);
    inline void compute_ic_block_step_extern(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset, bool is_tail = false);
    inline void compute_ic_block_step_interleave(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset, bool is_tail = false);
    inline void compute_ic_block_step_vpermw(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset, bool is_tail = false);
    inline void compute_oh_step_common(int ic_block_step);
    inline void compute_oh_step_disp();
    inline void compute_loop();
    inline void compute_oh_loop_common(bool partial = false);
    inline void compute_od_loop_common(bool partial = false);
    void compute_full_spat_loop();
    void convert_src_to_vnni_format(
            int ur_w, int pad_l, int pad_r, int input_offset);
    inline void compute_ic_block_step_vpermw_expl(int ur_w, int pad_l,
            int pad_r, int ic_block_step, int input_offset, int kernel_offset,
            int output_offset, bool is_tail = false);

    void generate();

    static void balance(const jit_conv_conf_t &j, int &nthr, int &nthr_mb,
            int &nthr_g, int &nthr_oc_b, int &nthr_ic_b);

    void get_w_positions(int ur_w, int pad_l, int pad_r, int i_ur, int i_kw,
            int &iw_0, int &iw_1) {
        auto get_w_position = [=](int idx) {
            int iw = i_ur + idx;
            if (iw >= ur_w) return -1;
            iw += i_kw;
            if (iw - pad_l < 0 || iw > (ur_w - 1) + (jcp.kw - 1) - pad_r)
                return -1;
            return iw - pad_l;
        };
        iw_0 = get_w_position(0);
        iw_1 = get_w_position(1);
    };
    bool check_borders(int ur_w, int pad_l, int pad_r, int i_ur, int i_kw) {
        int iw_1, iw_2;
        get_w_positions(ur_w, pad_l, pad_r, i_ur, i_kw, iw_1, iw_2);

        return (iw_1 == -1 && iw_2 == -1) ? false : true;
    };
    bool get_load_mask(int ur_w, int pad_l, int pad_r, int i_ur, int i_kw,
            Xbyak::Opmask &load_mask) {
        int iw_1, iw_2;
        get_w_positions(ur_w, pad_l, pad_r, i_ur, i_kw, iw_1, iw_2);

        bool rt = true;
        if (iw_1 != -1 && iw_2 != -1)
            load_mask = m_ffffffff;
        else if (iw_1 != -1 && iw_2 == -1)
            load_mask = m_0000ffff;
        else if (iw_1 == -1 && iw_2 != -1)
            load_mask = m_ffff0000;
        else
            rt = false;

        return rt;
    };

    ptrdiff_t get_inp_offset(
            int pad_l, int i_ur, int i_kw, ptrdiff_t base_offset_bytes) {
        ptrdiff_t local_offset_bytes
                = jcp.typesize_in * (i_ur + i_kw - pad_l) * jcp.ic_block;
        return base_offset_bytes + local_offset_bytes;
    };

    Xbyak::Zmm get_perm_reg() {
        int idx = !(jcp.uses_permw_transposition
                          && jcp.kernel_kind == expl_bcast)
                ? 24
                : ((!isa_has_bf16(jcp.isa)) ? 26 : 31);
        return Xbyak::Zmm(idx);
    }
    bf16_emulation_t *bf16_emu_;

    inline int interleave_w_reorder_size(int ur_w);
    inline int interleave_w_reorder_bytes(int ur_w);
    inline int interleave_stack_size(int ur_w, int ic_block_step);
    inline int permw_stack_size(int ur_w) {
        return (ur_w + jcp.kw - 1) * sizeof_cacheline;
    }

    inline void setup_stack_space();
    static const int extern_ic_block_step_stack_size = 0;
    int ic_block_step_stack_size;
    int stack_space_needed;
    int permw_buffer_start;
    int kd_count_offset;
    int input_d_offset;
    int output_d_offset;
    int d_index_offset;
    int trans_tmp_offset;
    int ih_dilate_shift;
};
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
