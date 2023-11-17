/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_JIT_XE_HP_SYSTOLIC_GEMM_HPP
#define GPU_JIT_XE_HP_SYSTOLIC_GEMM_HPP

#include <assert.h>
#include <memory>
#include <tuple>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel.hpp"
#include "gpu/jit/gemm/jit_gemm_pd.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct xe_hp_systolic_gemm_t : public gpu_gemm_t {
    struct pd_t : public jit_gemm_pd_t {
        using jit_gemm_pd_t::jit_gemm_pd_t;

        DECLARE_COMMON_PD_T("jit:xe_hp:gemm:any", xe_hp_systolic_gemm_t);

        status_t init(engine_t *engine);
        void init_scratchpad();

        bool use_nocopy();
        bool use_nocopy_xehpg(data_type_t dt, unsigned ld_align);
        status_t set_default_formats(data_type_t dt);

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;

        data_type_t impl_co_type() const {
            using namespace data_type;
            return with_bias() ? desc()->bias_type()
                               : (utils::one_of(desc()->a_type(), s8, u8)
                                               ? s32
                                               : desc()->c_type());
        }

        data_type_t impl_acc_type() const {
            using namespace data_type;
            return utils::one_of(desc()->c_type(), s8, u8, f16, bf16, f32)
                    ? (utils::one_of(desc()->a_type(), s8, u8) ? s32 : f32)
                    : s32;
        }

        float alpha() const { return 1.0f; }
        float beta() const { return beta_; }

        bool with_bias() const {
            return (desc()->bias_type() != data_type::undef)
                    && !bias_via_binary_;
        }

        int bias_cmask() const {
            unsigned char to_cmask[8] = {0, 4, 2, 6, 1, 5, 3, 7};
            assert(unsigned(desc()->bias_mask()) < 8);
            return with_bias() ? to_cmask[desc()->bias_mask() & 7] : -1;
        }

        bool packed_a() const { return packed_a_; }
        bool packed_b() const { return packed_b_; }
        bool packed_c() const { return packed_c_; }

        static int64_t nice_ld(int64_t ld, int sz, bool get_max = false) {
            const auto align = 32;
            const auto no_align = 64;

            auto new_ld = (ld * sz + align - 1) & ~(align - 1);
            if (get_max || (new_ld & (no_align - 1)) == 0) new_ld += align;

            return new_ld / sz;
        }

        int64_t get_ld_packed(int64_t k, bool get_max = false) const {
            auto a_sz = types::data_type_size(desc()->a_type());

            int unroll_k = int(32 / a_sz);
            auto ld = utils::rnd_up(k, unroll_k);
            if (with_ab_zero_points()) ld += unroll_k;

            return nice_ld(ld, int(a_sz), get_max);
        }

        int64_t max_ld_packed(int64_t k) const {
            return get_ld_packed(k, true);
        }

        dim_t lda_packed(int64_t k) const {
            return packed_a() ? desc()->b_desc.format_desc.blocking
                                        .strides[with_batch() ? 2 : 1]
                            / unroll_m()
                              : get_ld_packed(k);
        }
        dim_t ldb_packed(int64_t k) const {
            return packed_b() ? desc()->a_desc.format_desc.blocking
                                        .strides[with_batch() ? 1 : 0]
                            / unroll_n()
                              : get_ld_packed(k);
        }
        dim_t ldc_packed() const {
            return packed_c() ? desc()->c_desc.format_desc.blocking
                                        .strides[with_batch() ? 1 : 0]
                            / unroll_n()
                              : 0;
        }

        int batch_dims() const {
            return nstl::max(desc()->c_desc.ndims - 2, 0);
        }

        bool with_batch() const { return desc()->is_batched(); }
        bool with_a_zero_points() const { return a_zp_; }
        bool with_b_zero_points() const { return b_zp_; }
        bool with_ab_zero_points() const { return a_zp_ || b_zp_; }
        bool with_c_zero_points() const { return c_zp_; }

        bool allow_k_blocking() const {
            return (desc()->acc_type == desc()->c_type())
                    && IMPLICATION(post_ops()->len() > 0,
                            post_ops()->entry_[0].kind == primitive_kind::sum);
        }

        int unroll_m() const { return unroll_m_; }
        int unroll_n() const { return unroll_n_; }
        bool alt() const { return alt_; }

        status_t query(query_t what, int idx, void *result) const override {
            switch ((int)what) {
                case (int)query::preferred_gpu_threads_per_eu: {
                    *(int *)result = 4;
                    break;
                }
                default: return gpu_gemm_pd_t::query(what, idx, result);
            }
            return status::success;
        }

        const compute::device_info_t *dev_info_ = nullptr;

    private:
        bool any_prepacked_ = false;
        bool packed_a_ = false, packed_b_ = false, packed_c_ = false;
        bool a_zp_ = false, b_zp_ = false, c_zp_ = false;
        int unroll_m_ = 0;
        int unroll_n_ = 0;
        bool alt_ = false;
    };

    status_t init(engine_t *engine) override;

public:
    xe_hp_systolic_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    virtual status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    status_t init_compute(engine_t *engine);

    bool enable_mn_blocking() const;
    std::tuple<int64_t, int64_t, int64_t> get_blocking() const;

    status_t launch_clear_sum(const gemm_exec_ctx_t &ctx, int64_t r, int64_t c,
            const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
            bool copyb) const;
    status_t launch_copy(const gemm_exec_ctx_t &ctx, int64_t r, int64_t c,
            const memory_storage_t &src, int64_t offset_src, int64_t ld_src,
            const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
            bool copyb) const;
    status_t launch_compute(const gemm_exec_ctx_t &ctx, int32_t m, int32_t n,
            int32_t k, const memory_storage_t &ap, int64_t offset_a,
            int32_t lda, const memory_storage_t &bp, int64_t offset_b,
            int32_t ldb, const memory_storage_t &c, int64_t offset_c,
            int32_t ldc, float alpha, float beta, const memory_storage_t *ao,
            const memory_storage_t *bo, const memory_storage_t &co,
            int32_t offset_co, int po_count, const memory_storage_t **po_src,
            int32_t *offset_po_src, bool first_k_block, bool last_k_block,
            int32_t batch, int32_t stride_a, int32_t stride_b,
            int32_t stride_c) const;

    static const int A_PACKED_ = 0;
    static const int B_PACKED_ = 1;

    compute::kernel_t kernel_[2][2]; // [first_k_block][last_k_block]
    compute::kernel_t copy_kernel_[2][2]; // [trans][clear_sum]

    CommonDriverInfo compute_info_;

    compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;
    int eu_count_ = 0;

    char co_kind_ = 'N';
    bool walk_n_first_ = false;

    GEMMProblem problem_;

    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
