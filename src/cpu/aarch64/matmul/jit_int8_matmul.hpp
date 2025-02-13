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

#ifndef CPU_AARCH64_JIT_INT8_MATMUL_HPP
#define CPU_AARCH64_JIT_INT8_MATMUL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/aarch64/matmul/jit_int8_kernel_types.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct jit_int8_matmul_kernel_t;
struct jit_int8_matmul_utils_kernel_t;

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
                            != format_tag::undef
                    || dst_d.format_kind() == format_kind::any;
            const bool is_wei
                    = weights_d.matches_one_of_tag(format_tag::ab,
                              format_tag::abc, format_tag::abcd,
                              format_tag::BA24b8a, format_tag::aCB24c8b,
                              format_tag::abDC24d8c)
                            != format_tag::undef
                    || weights_d.format_kind() == format_kind::any;
            const bool is_src = src_d.matches_one_of_tag(format_tag::ab,
                                        format_tag::abc, format_tag::abcd)
                            != format_tag::undef
                    || src_d.format_kind() == format_kind::any;
            return is_dst && is_wei && is_src;
        }
        const brg_int8_t &get_b() const { return brg_; }

        const dyn_vals_t &get_d() const { return dyn_; }

        int get_idx(int z, int m, int k, int n, const brg_int8_t b) const {

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
        brg_int8_t brg_;
        dyn_vals_t dyn_;
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