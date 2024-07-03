/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_BRGEMM_TRANSPOSE_UTILS_HPP
#define CPU_AARCH64_JIT_BRGEMM_TRANSPOSE_UTILS_HPP

#include "cpu/aarch64/jit_brgemm_primitive_conf.hpp"
#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct jit_brgemm_trans_src_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;

        dim_t current_gemm_batch;
        dim_t current_M, current_K;
    };

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_trans_src_t(const jit_brgemm_primitive_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_trans_src_t() {}

    const jit_brgemm_primitive_conf_t *conf_;
};

struct jit_brgemm_copy_to_coarse_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_copy_to_coarse_t)

    struct ctx_t {
        const void *data;
        const void *tr_data;

        dim_t os_work;
        dim_t last_row_blk;
    };

    void operator()(ctx_t *ctx) { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

    jit_brgemm_copy_to_coarse_t(const jit_brgemm_primitive_conf_t *conf)
        : conf_(conf)
        , typesize_(sizeof(bfloat16_t))
        , is_fwd_dir_(utils::one_of(conf_->prop_kind,
                  prop_kind::forward_training, prop_kind::forward_inference))
        , row_block_size_(is_fwd_dir_ ? conf_->ic_block : conf_->oc_block)
        , row_size_(is_fwd_dir_ ? conf_->ic_without_padding
                                : conf_->oc_without_padding)
        , tr_row_size_(conf_->LDA)
        , row_granularity_(granularity_in_bytes / typesize_)
        , row_step_(zmm_size_in_bytes / typesize_)
        , data_stride_(row_size_ * typesize_)
        , tr_data_stride_(tr_row_size_ * typesize_) {

        // Kernel is supposed to be called under the following constraints
        assert(row_size_ % row_granularity_ != 0);

        MAYBE_UNUSED(row_granularity_);
    }
    ~jit_brgemm_copy_to_coarse_t() {}

private:
    enum {
        zmm_size_in_bytes = 64,
        row_loop_unroll = 16,
        granularity_in_bytes = 4,
    };

    const jit_brgemm_primitive_conf_t *conf_;
    const int typesize_;
    const bool is_fwd_dir_;
    const int row_block_size_, row_size_, tr_row_size_, row_granularity_,
            row_step_;
    const dim_t data_stride_, tr_data_stride_;

    inline size_t addr_offset(int row_idx) {
        return row_idx * row_step_ * typesize_;
    }
    inline Xbyak_aarch64::ZReg get_zmm_copy(int row_idx) const {
        assert(row_idx >= 0 && row_idx < row_loop_unroll);
        return Xbyak_aarch64::ZReg(row_idx);
    }

    const Xbyak_aarch64::ZReg zmm_zero = Xbyak_aarch64::ZReg(row_loop_unroll);
    const Xbyak_aarch64::ZReg zmm_row_tail
            = Xbyak_aarch64::ZReg(row_loop_unroll + 1);

    const Xbyak_aarch64::PReg reg_m_full_row_tail_load = p7;
    const Xbyak_aarch64::PReg reg_m_full_row_tail_store = p6;
    const Xbyak_aarch64::PReg reg_m_last_row_tail_load = p5;
    const Xbyak_aarch64::PReg reg_m_last_row_tail_store = p4;

    const Xbyak_aarch64::XReg reg_data = x0;
    const Xbyak_aarch64::XReg reg_tr_data = x3;

    const Xbyak_aarch64::XReg reg_os_work = x11;
    const Xbyak_aarch64::XReg reg_last_row_blk = x12;
    const Xbyak_aarch64::XReg reg_tail_mask = x13;

    void copy_os_loop();
    void copy_row_loop();

    void copy_row_blks(int num_row_blks);
    void copy_row_tail(bool is_last_iteration, int row_offset);
    void zero_out_rows();

    void set_full_row_tail_masks();
    void set_last_row_tail_masks();

    void generate() override;
};

struct jit_brgemm_trans_to_vnni_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;

        dim_t current_gemm_batch;
        dim_t current_col_size, current_row_size;
    };

    typedef enum matrix_to_transform {
        matrix_B,
        matrix_C
    } matrix_to_transform_t;

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_trans_to_vnni_t(const jit_brgemm_primitive_conf_t *conf,
            matrix_to_transform_t matrix_to_transform)
        : conf_(conf), matrix_to_transform_(matrix_to_transform) {}
    virtual ~jit_brgemm_trans_to_vnni_t() {}

    const jit_brgemm_primitive_conf_t *conf_;
    matrix_to_transform_t matrix_to_transform_;
};

struct jit_brgemm_trans_wei_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;

        dim_t current_gemm_batch;
        dim_t current_N, current_K;
    };

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_trans_wei_t(const jit_brgemm_primitive_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_trans_wei_t() {}

    const jit_brgemm_primitive_conf_t *conf_;
};

status_t create_brgemm_trans_src(
        std::unique_ptr<jit_brgemm_trans_src_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf);
status_t create_brgemm_copy_to_coarse(
        std::unique_ptr<jit_brgemm_copy_to_coarse_t> &copy_ker,
        const jit_brgemm_primitive_conf_t *conf);
status_t create_brgemm_trans_to_vnni(
        std::unique_ptr<jit_brgemm_trans_to_vnni_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf,
        jit_brgemm_trans_to_vnni_t::matrix_to_transform_t matrix_to_transform);
status_t create_brgemm_trans_wei(
        std::unique_ptr<jit_brgemm_trans_wei_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf);
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
