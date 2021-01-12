/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_AMX_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_AMX_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_amx_copy_to_wbuffer_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_copy_to_wbuffer_t)

    using reg64_t = Xbyak::Reg64;

    jit_avx512_core_amx_copy_to_wbuffer_t(const jit_conv_conf_t &ajcp)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx)
        , jcp(ajcp) {}

private:
    jit_conv_conf_t jcp;

    const reg64_t reg_src = rax;
    const reg64_t reg_dst = rbx;
    const reg64_t reg_tmp = rdx;

    const Xbyak::Opmask kmask_load = Xbyak::Opmask(2);

    const Xbyak::Zmm zmm_src = Xbyak::Zmm(0);
    const Xbyak::Zmm zmm_dst = Xbyak::Zmm(1);
    const Xbyak::Zmm zmm_idx = Xbyak::Zmm(2);
    const Xbyak::Zmm zmm_zero = Xbyak::Zmm(3);

    void generate() override;
};

struct jit_avx512_core_amx_copy_to_pbuffer_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_copy_to_pbuffer_t)

    using reg64_t = Xbyak::Reg64;

    jit_avx512_core_amx_copy_to_pbuffer_t(const jit_conv_conf_t &ajcp)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx)
        , jcp(ajcp) {}

private:
    jit_conv_conf_t jcp;

    const reg64_t reg_inp_ptr = r15;
    const reg64_t reg_out_ptr = r14;

    const reg64_t reg_aux_inp_ptr = r13;
    const reg64_t reg_aux_out_ptr = r12;

    const reg64_t reg_khp = r10;

    /* relow stuff */
    const reg64_t reg_kht = r11;
    const reg64_t reg_tov = r9;
    const reg64_t reg_bov = r8;
    const reg64_t reg_kwp = rax;
    const reg64_t reg_lov = reg_aux_inp_ptr;
    const reg64_t reg_rov = rbx;
    const reg64_t reg_save_out_ptr = rdx;
    const reg64_t reg_cnt = rbp;
    /* relow stuff */

    /* non-relow stuff */
    const reg64_t reg_kdp = abi_not_param1;
    const reg64_t reg_kdc = rbp;
    const reg64_t reg_khc = r11;

    const reg64_t reg_icb = r9;

    const reg64_t reg_kh_over = r8;
    const reg64_t reg_tover = rax;
    const reg64_t reg_bover = rbx;

    const reg64_t reg_owb = rdx;
    /* non-relow stuff */

    const reg64_t reg_tmp = rsi;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    const Xbyak::Opmask ktail_load = Xbyak::Opmask(3);
    const Xbyak::Opmask ktail_stor = Xbyak::Opmask(4);

    const Xbyak::Ymm ymm_tmp = Xbyak::Ymm(0);
    const Xbyak::Zmm zmm_tmp = Xbyak::Zmm(0);
    const Xbyak::Zmm zmm_zero = Xbyak::Zmm(1);

    void generate() override;
    void copy_row(int icb);
    void copy_row_body(int lpad, int iw_len, int icb);
    void copy_row_reduced_lowering();
};

struct jit_avx512_core_amx_fwd_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_fwd_kernel_t)

    jit_avx512_core_amx_fwd_kernel_t(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx)
        , jcp(ajcp)
        , attr_(attr)
        , eltwise_injector_(nullptr)
        , copy_to_wbuffer_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);
        copy_to_pbuffer_ = new jit_avx512_core_amx_copy_to_pbuffer_t(jcp);
        if (jcp.is_relo)
            copy_to_wbuffer_ = new jit_avx512_core_amx_copy_to_wbuffer_t(jcp);
    }

    status_t create_kernel() override {
        CHECK(jit_generator::create_kernel());
        CHECK(copy_to_pbuffer_->create_kernel());
        if (jcp.is_relo) CHECK(copy_to_wbuffer_->create_kernel());
        return status::success;
    }
    ~jit_avx512_core_amx_fwd_kernel_t() {
        delete eltwise_injector_;
        delete copy_to_pbuffer_;
        delete copy_to_wbuffer_;
    }

    static bool post_ops_ok(
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, const primitive_attr_t &attr, int nthreads);
    static status_t init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    void tile_configure(char *tcfg_buff);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

    const jit_avx512_core_amx_copy_to_pbuffer_t &copy_to_pbuffer() const {
        return *copy_to_pbuffer_;
    }
    const jit_avx512_core_amx_copy_to_wbuffer_t &copy_to_wbuffer() const {
        return *copy_to_wbuffer_;
    }

private:
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;
    jit_avx512_core_amx_copy_to_pbuffer_t *copy_to_pbuffer_;
    jit_avx512_core_amx_copy_to_wbuffer_t *copy_to_wbuffer_;

    enum {
        zmm_idx_limit_bf16 = 29,
        zmm_idx_limit_int8 = 27,
    };

    int prv_width_;
    int row_count_;
    bool is_store_done_;
    bool is_buffer_empty_;

    /* data regs */
    const Xbyak::Reg64 reg_inp_ptr = r15;
    const Xbyak::Reg64 reg_wei_ptr = r14;
    const Xbyak::Reg64 reg_out_ptr = r13;
    const Xbyak::Reg64 reg_wsp_ptr = r12;

    const Xbyak::Reg64 reg_kd = rsi;

    const Xbyak::Reg64 reg_bias = r11;
    const Xbyak::Reg64 reg_ptr_scales = r10;
    const Xbyak::Reg64 reg_ptr_sum_scale = r9;
    const Xbyak::Reg64 reg_aux_saturation = reg_ptr_sum_scale;

    const Xbyak::Reg64 reg_inp_stride = rbx;
    const Xbyak::Reg64 reg_wei_stride = rdx;
    // zero-point computation
    const Xbyak::Reg64 reg_zp_compensation = rax;
    const Xbyak::Reg64 reg_src_zero_point = r8;
    const Xbyak::Reg64 reg_dst_zero_point = abi_not_param1;

    // rbp - reserved for EVEX compression
    const Xbyak::Reg64 reg_last_h = abi_not_param1;

    // temporary, used in generate() function only
    const Xbyak::Reg64 reg_oc_blocks = rax;
    const Xbyak::Reg64 reg_tmp = r8;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    const Xbyak::Zmm zmm_bias = Xbyak::Zmm(31);
    const Xbyak::Zmm zmm_saturation = zmm_bias;
    const Xbyak::Zmm zmm_zero = Xbyak::Zmm(30);
    const Xbyak::Zmm zmm_prev_dst = Xbyak::Zmm(29);
    /* zero-point */
    const Xbyak::Zmm zmm_zp = Xbyak::Zmm(29);
    const Xbyak::Zmm zmm_src_zp = Xbyak::Zmm(28);
    const Xbyak::Zmm zmm_dst_zp = Xbyak::Zmm(27);

    // AUX: Steps, shifts and offsets
    size_t get_inp_icb_step() const;
    size_t get_wei_icb_step() const;
    size_t get_inp_d_step() const;
    size_t get_inp_h_step() const;
    size_t get_wei_d_step() const;
    size_t get_wei_h_step() const;
    size_t get_out_ocb_offset(int ohb, int ocb) const;
    size_t get_out_row_offset(int ohb, int ocb, int j) const;
    size_t get_out_shift(int width) const;
    size_t get_wsp_ocb_offset(int ohb, int ocb) const;
    size_t get_wsp_row_offset(int ohb, int ocb, int j) const;
    size_t get_wsp_shift() const;
    size_t get_wei_offset(int ocb, int kw) const;
    size_t get_inp_shift() const;
    size_t get_inp_offset(int ohb, int kw) const;

    int get_out_tensor(int h, int i, bool is_h_tail = false) const;
    int get_inp_tensor(int h, bool is_h_tail = false) const;
    int get_wei_tensor(int i) const;

    void prepare_output(int tail);
    void init_runtime_counters(bool start_with_last_tile_block);

    bool maybe_eltwise(int position);
    void cvt2ps(data_type_t type_in, Xbyak::Zmm ymm_in,
            const Xbyak::Operand &op, bool mask_flag);
    Xbyak::Zmm zmm_out(const int idx) {
        const int upper_limit = jcp.src_dt == data_type::bf16
                ? zmm_idx_limit_bf16
                : zmm_idx_limit_int8;
        assert(upper_limit > idx);
        MAYBE_UNUSED(upper_limit);
        return Xbyak::Zmm(idx);
    }
    Xbyak::Ymm ymm_mask(
            const Xbyak::Ymm zmm_in, bool mask_flag, bool store = false);
    Xbyak::Zmm zmm_mask(
            const Xbyak::Zmm zmm_in, bool mask_flag, bool store = false);

    void store_output_vector_bf16(
            const Xbyak::Zmm zmm_out, int ocb, int h, int w);
    void store_output_vector_int8(
            const Xbyak::Zmm zmm_out, int ocb, int h, int w);
    void store_output_vector(const Xbyak::Zmm zmm_out, int ocb, int h, int w);
    void store_output(int width, int tail, bool do_store);
    void interleave_store(int width);
    void compute_icb_loop(int width, bool do_store);
    void compute_ow_loop();

    void generate() override;
};

struct jit_avx512_core_amx_bwd_data_copy_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_bwd_data_copy_kernel_t)

    using reg64_t = Xbyak::Reg64;

    jit_avx512_core_amx_bwd_data_copy_kernel_t(jit_conv_conf_t ajcp)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx)
        , jcp(ajcp) {}

private:
    jit_conv_conf_t jcp;

    // pointers
    const reg64_t reg_ptr_inp = r15;
    const reg64_t reg_ptr_out = r14;

    // auxiliary pointers
    const reg64_t reg_ptr_aux_inp_h = r13;
    const reg64_t reg_ptr_aux_inp_w = r12;
    const reg64_t reg_ptr_aux_out = r11;

    // variables
    const reg64_t reg_khp = r10; // kh padding
    const reg64_t reg_tov = r9; // top overflow
    const reg64_t reg_bov = r8; // bottom overflow
    const reg64_t reg_kwp = rax; // kw padding
    const reg64_t reg_lov = rbx; // left overflow
    const reg64_t reg_rov = abi_not_param1; // right overflow

    // counters
    const reg64_t reg_cnt_khp = rdx;
    const reg64_t reg_cnt_tmp = rbp;
    const reg64_t reg_cnt_ocb = rsi;

    const reg64_t reg_tmp = reg_cnt_tmp;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    const Xbyak::Zmm zmm_tmp = Xbyak::Zmm(1);
    const Xbyak::Zmm zmm_zero = Xbyak::Zmm(0);

    void generate() override;
    void copy_row(bool is_masked);
};

struct jit_avx512_core_amx_bwd_data_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_bwd_data_kernel_t)

    jit_avx512_core_amx_bwd_data_kernel_t(
            const jit_conv_conf_t ajcp, const primitive_attr_t &attr)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx)
        , jcp(ajcp)
        , attr_(attr)
        , eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);
        bwd_data_copy_kernel_
                = new jit_avx512_core_amx_bwd_data_copy_kernel_t(jcp);
    }
    status_t create_kernel() override {
        CHECK(jit_generator::create_kernel());
        CHECK(bwd_data_copy_kernel_->create_kernel());
        return status::success;
    }
    ~jit_avx512_core_amx_bwd_data_kernel_t() {
        delete eltwise_injector_;
        delete bwd_data_copy_kernel_;
    }

    static bool post_ops_ok(
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &diff_src_pd,
            memory_desc_t &weights_pd, memory_desc_t &diff_dst_pd,
            memory_desc_t *bias_pd, const primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    void tile_configure(char *tcfg_buff);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

    const jit_avx512_core_amx_bwd_data_copy_kernel_t &
    bwd_data_copy_kernel() const {
        return *bwd_data_copy_kernel_;
    }

private:
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;
    jit_avx512_core_amx_bwd_data_copy_kernel_t *bwd_data_copy_kernel_;

    int prv_width_;
    int row_count_;
    bool is_store_done_;
    bool is_buffer_empty_;

    /* data regs */
    const Xbyak::Reg64 reg_inp_ptr = r15;
    const Xbyak::Reg64 reg_wei_ptr = r14;
    const Xbyak::Reg64 reg_out_ptr = r13;
    const Xbyak::Reg64 reg_wsp_ptr = r12;

    const Xbyak::Reg64 reg_bias = r11;
    const Xbyak::Reg64 reg_ptr_scales = r10;
    const Xbyak::Reg64 reg_ptr_sum_scale = r9;
    const Xbyak::Reg64 reg_aux_saturation = reg_ptr_sum_scale;

    const Xbyak::Reg64 reg_aux_inp_ptr = r8;
    const Xbyak::Reg64 reg_inp_stride = rbx;
    const Xbyak::Reg64 reg_wei_stride = rdx;

    // rsi - free and available
    // rbp - reserved for EVEX compression
    const Xbyak::Reg64 reg_last_h = abi_not_param1;

    // temporary, used in generate() function only
    const Xbyak::Reg64 reg_ic_blocks = rax;
    const Xbyak::Reg64 reg_tmp = reg_aux_inp_ptr;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    const Xbyak::Zmm zmm_bias = Xbyak::Zmm(31);
    const Xbyak::Zmm zmm_saturation = zmm_bias;
    const Xbyak::Zmm zmm_zero = Xbyak::Zmm(30);
    const Xbyak::Zmm zmm_prev_dst = Xbyak::Zmm(29);

    // AUX: Steps, shifts and offsets
    size_t get_inp_kh_step() const;
    size_t get_inp_ocb_step() const;
    size_t get_inp_offset(int ihb, int kh, int kw) const;
    size_t get_inp_shift() const;
    size_t get_out_icb_offset(int ihb, int icb) const;
    size_t get_out_row_offset(int ihb, int icb, int j) const;
    size_t get_out_shift(int width) const;
    size_t get_wei_kh_step() const;
    size_t get_wei_ocb_step() const;
    size_t get_wei_offset(int icb, int kh, int kw) const;
    size_t get_wsp_icb_offset(int ihb, int icb) const;
    size_t get_wsp_row_offset(int ihb, int icb, int j) const;
    size_t get_wsp_shift() const;

    int get_out_tensor(int h, int i) const;
    int get_inp_tensor(int h) const;
    int get_wei_tensor(int i) const;

    void prepare_output();
    void init_runtime_counters(bool start_with_last_tile_block);

    bool maybe_eltwise(int position);
    void cvt2ps(data_type_t type_in, Xbyak::Zmm ymm_in,
            const Xbyak::Operand &op, bool mask_flag);
    Xbyak::Ymm ymm_mask(
            const Xbyak::Ymm zmm_in, bool mask_flag, bool store = false);
    Xbyak::Zmm zmm_mask(
            const Xbyak::Zmm zmm_in, bool mask_flag, bool store = false);

    void store_output_vector_bf16(
            const Xbyak::Zmm zmm_out, int icb, int ihb, int iw);
    void store_output_vector_int8(
            const Xbyak::Zmm zmm_out, int icb, int ihb, int iw);
    void store_output_vector(const Xbyak::Zmm zmm_out, int icb, int ih, int iw);
    void store_output(int width, bool do_store);
    void interleave_store(int width);
    void compute_ocb_loop(int width, bool do_store);
    void compute_iw_loop();

    void generate() override;
};

struct jit_avx512_core_amx_bwd_weights_kernel_t : public jit_generator {

    jit_avx512_core_amx_bwd_weights_kernel_t(const jit_conv_conf_t &ajcp)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx)
        , jcp(ajcp) {}

    ~jit_avx512_core_amx_bwd_weights_kernel_t() {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_bwd_weights_kernel_t)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
            memory_desc_t &diff_dst_md, int nthreads);
    static status_t init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, memory_desc_t &src_md,
            memory_desc_t &diff_weights_md, memory_desc_t &diff_dst_md);

    void tile_configure(char *tcfg_buff);

    const jit_conv_conf_t &jcp;

private:
    int get_wei_tensor(int ocb, int icb) const;
    int get_src_tensor(int icb) const;
    int get_ddst_tensor(int ocb) const;

    using reg64_t = const Xbyak::Reg64;
    static const int max_ur_w;

    reg64_t param = abi_param1;
    reg64_t reg_src = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_ddst = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh = r9;
    reg64_t reg_oj = r15;
    reg64_t reg_tmp = r14;
    reg64_t reg_ih_shift = reg_tmp;
    reg64_t reg_long_offt = r14;
    reg64_t reg_icb = rbx;

    reg64_t ki = r11;
    reg64_t reg_oj_setup = r11;
    reg64_t reg_kd_count = r12;
    reg64_t reg_oi = r12;
    reg64_t reg_d_index = r13;
    reg64_t reg_src_d = r15;
    reg64_t reg_ddst_d = rbx;
    reg64_t aux_reg_src = r12;
    reg64_t aux_reg_kernel = r13;

    reg64_t reg_b_stride = reg_icb;
    reg64_t reg_a_stride = r10;

    Xbyak::Zmm vreg_bias_acc = Xbyak::Zmm(0);
    Xbyak::Zmm vreg_bias_unit = Xbyak::Zmm(1);
    Xbyak::Zmm vreg_bias_ddst = Xbyak::Zmm(2);

    enum {
        full_spat_opt_working_set_size = 48 * 1024,
        full_spat_max_working_set_size = 128 * 1024,
    };

    inline void maybe_zero_kernel(int nb_ic_blocking, int nb_oc_blocking);
    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_ic_loop(
            int ic_block, int nb_ic_blocking, int nb_oc_blocking);
    inline void compute_full_spat_loop(int nb_ic_blocking, int nb_oc_blocking);
    inline void compute_oh_step_common(int nb_ic_blocking, int nb_oc_blocking);
    inline void compute_loop(int nb_ic_blocking, int nb_oc_blocking);
    inline void compute_oh_loop_common(
            int nb_ic_blocking, int nb_oc_blocking, bool partial = false);
    inline void compute_od_loop_common(
            int nb_ic_blocking, int nb_oc_blocking, bool partial = false);
    void compute_diff_bias_init(int ocb = 0);
    void compute_diff_bias_row(bool is_partial, int ocb);
    void maybe_compute_diff_bias(int nb_oc_blocking);
    void may_be_set_oc_tail_mask();
    void may_be_reset_oc_tail_mask();

    void generate() override;

    static void balance(const jit_conv_conf_t &j, int &nthr, int &nthr_mb,
            int &nthr_g, int &nthr_oc_b, int &nthr_ic_b);

    inline dim_t filter_w_to_src(int kw, int ow = 0, int pad_l = 0) {
        return kw * (jcp.dilate_w + 1) + ow - pad_l;
    }
    inline dim_t filter_h_to_src(int kh) { return kh * (jcp.dilate_h + 1); }
    inline dim_t filter_d_to_src(int kd) {
        return kd * (jcp.dilate_d + 1) * jcp.ih;
    }

    inline dim_t get_src_offset(dim_t ic_idx, dim_t w_idx, dim_t hd_idx = 0) {
        return jcp.typesize_in
                * (hd_idx * jcp.tr_iw * jcp.ic_block + jcp.tr_iw * ic_idx
                        + w_idx);
    }

    inline dim_t get_ddst_offset(dim_t w_idx, dim_t hd_idx = 0) {
        int ow_per_oc = 2;
        dim_t w_off = w_idx / ow_per_oc * ow_per_oc * jcp.oc_block
                + w_idx % ow_per_oc;
        return jcp.typesize_in * (w_off + jcp.tr_ow * jcp.oc_block * hd_idx);
    }

    inline dim_t get_kernel_offset(int ic_idx, dim_t ksp_idx) {
        return jcp.typesize_out * jcp.oc_block
                * (ksp_idx * jcp.ic_block + ic_idx);
    }
    inline dim_t get_full_kernel_offset(int ocb, int icb, int kh, int kw) {
        return jcp.typesize_out
                * (ocb * jcp.nb_ic * jcp.kd * jcp.kh * jcp.kw * jcp.ic_block
                                * jcp.oc_block
                        + icb * jcp.kd * jcp.kh * jcp.kw * jcp.ic_block
                                * jcp.oc_block
                        + kh * jcp.kw * jcp.ic_block * jcp.oc_block
                        + kw * jcp.ic_block * jcp.oc_block);
    };

    inline void setup_stack_space();
    int ic_block_step_stack_size = 0;
    int stack_space_needed = 0;
    int kd_count_offset = 0;
    int src_d_offset = 0;
    int ddst_d_offset = 0;
    int d_index_offset = 0;
    int ih_dilate_offset = 0;
    int src_save_offset = 0;
    int ddst_save_offset = 0;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
