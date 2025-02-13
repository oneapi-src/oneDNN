/*******************************************************************************
* Copyright 2025 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_jit_int8_matmul_HPP
#define CPU_AARCH64_jit_int8_matmul_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/aarch64/jit_int8_matmul_utils.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

typedef enum {
    none = 0,
    per_tensor = 1,
    per_m = 2,
    per_n = 3,
    per_k = 4,
} jit_int8_broadcast_t;

struct brg_int8_t_ {
    int M, K, N;
    const int m_blk = 8, n_blk = 4, k_blk = 8;
    const int ld_block = 6, rd_block = 4, bd_block = 8;
    int na, nb;
    int m_tail, n_tail, k_tail;
    int is_m_tail, is_k_tail, is_n_tail, is_zp_cal;
    int dst_dt_sz;
    bool is_s8;
    bool is_bias;
    bool with_scales;
    bool with_dst_scales;
    bool is_oc_scales;
    jit_int8_broadcast_t zp_type_a = jit_int8_broadcast_t::none;
    jit_int8_broadcast_t zp_type_b = jit_int8_broadcast_t::none;
    jit_int8_broadcast_t zp_type_c = jit_int8_broadcast_t::none;
    bool is_zp_b_int8 = false;
    bool b_reo = true;
    data_type_t zp_b_dt;
    dim_t B;
};

struct jit_int8_matmul_kernel_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_matmul_kernel_t)

    struct call_params_t {
        const uint8_t *src, *wei;
        float *dst;
        const float *bias, *scales, *dst_scales;
        dim_t M, K, N;
        char *buf_B_ptr_;
        int *na, *nb;
        int32_t *src_zero_point, *wei_zero_point, *dst_zero_point;
        const int8_t *wei_zero_point_buf;
        float *zp_a_ptr, *zp_b_ptr;
    };

    XReg reg_param = abi_param1;
    XReg reg_a = x3;
    XReg reg_b = x4;
    XReg reg_c = x5;
    XReg reg_aux_a = x6;
    XReg reg_aux_b = x7;
    XReg reg_aux_c = x8;
    XReg reg_aux_a1 = x9;
    XReg reg_zp_aux_b_buf = x10;
    XReg reg_aux_c1 = x11;
    XReg reg_ld_loop = x12;
    XReg reg_rd_loop = x13;
    XReg reg_bd_loop = x14;
    XReg reg_tmp = x15;
    XReg reg_tmp_1 = x16;
    XReg reg_bias = x17;
    XReg reg_zp_a = x18;

    XReg reg_scales = x20;
    XReg reg_aux_scales = x24; //used X_TMP_1
    XReg reg_na = x25; //used X_TMP_2
    XReg reg_zp_b = x26; //used X_TMP_3
    XReg reg_zp_aux_b = x27; //used X_TMP_4
    PReg prd_ld = p1;
    PReg prd_st = p2;
    PReg prd_b = p3;
    PReg prd_8 = p4;
    PReg prd_zp_b_tl = p5;
    XReg reg_zp_val_c = x2;

    XReg reg_zp_val_a = reg_scales;
    XReg reg_zp_val_b = reg_aux_scales;

    call_params_t inp;

    void operator()(const call_params_t *p) {
        return jit_generator::operator()(p);
    }

    jit_int8_matmul_kernel_t(const brg_int8_t_ &k) : brg(k) {}
    ~jit_int8_matmul_kernel_t() override = default;

private:
    ZReg loadb(int ld);
    ZReg acc(int bd, int ld);
    void zero_regs();
    void store_regs(int bdb, int ldb, int tail);
    void microkernel(int rdb, int bdb, int ldb, int tail);
    void loop_k(int bdb, int ldb, int tail);
    void loop_k_zp(int bdb, int ldb, int is_a, int is_b);
    void han_blk();
    void han_blk_zp();
    void zp_comp(int rdb, int bdb, int ldb, int is_a, int is_b);
    void config();
    void generate() override;

    brg_int8_t_ brg;

    int ldb;
    int bdb;
    int rdb;
    int k_full_blks;
    int k_tail_blk;
    int k_residual_blk;
    int n_blks;
};

struct jit_int8_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("jit:int8", jit_int8_matmul_t);

        status_t init(engine_t *engine);

        bool formats_ok() const {

            const memory_desc_wrapper src_d(src_md_);
            const memory_desc_wrapper weights_d(weights_md_);
            const memory_desc_wrapper dst_d(dst_md_);
            const bool is_dst = dst_d.matches_one_of_tag(format_tag::ab,
                                        format_tag::abc, format_tag::abcd)
                    || dst_d.format_kind() == format_kind::any;
            const bool is_wei
                    = weights_d.matches_one_of_tag(format_tag::ab,
                              format_tag::abc, format_tag::abcd,
                              format_tag::BA24b8a, format_tag::aCB24c8b,
                              format_tag::abDC24d8c)
                    || weights_d.format_kind() == format_kind::any;
            const bool is_src = src_d.matches_one_of_tag(format_tag::ab,
                                        format_tag::abc, format_tag::abcd)
                    || src_d.format_kind() == format_kind::any;
            return is_dst && is_wei && is_src;
        }
        const brg_int8_t_ &get_b() const { return brg; }

        const dyn_vals_t_ &get_d() const { return dyn; }

        int get_idx(int z, int m, int k, int n, const brg_int8_t_ b) const {

            if (b.zp_type_a == jit_int8_broadcast_t::none
                    && b.zp_type_b == jit_int8_broadcast_t::none && z == 1)
                return -1;
            int mt = b.M % b.m_blk;
            int nt = b.N % (b.n_blk * b.ld_block);
            int kt = b.K % (b.k_blk * 4);
            if ((m == 1 && mt == 0) || (k == 1 && kt == 0)
                    || (n == 1 && nt == 0) || (k == 0 && kt == 1))
                return -1;
            return k + n * 2 + m * 2 * 2 + z * 2 * 2 * 2;
        }

    private:
        brg_int8_t_ brg;
        dyn_vals_t_ dyn;
    };

    jit_int8_matmul_t(const pd_t *apd);
    ~jit_int8_matmul_t() override;
    int get_idx(int z, int m, int k, int n, int M, int K, int N);
    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_int8_matmul_kernel_t> int8_kernels_[16];
    std::unique_ptr<jit_int8_matmul_utils_kernel_t> reo_ker_a_;
    std::unique_ptr<jit_int8_matmul_utils_kernel_t> reo_ker_b_;
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif