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

#ifndef GPU_JIT_GEMM_GEN_GEMM_HPP
#define GPU_JIT_GEMM_GEN_GEMM_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/compute/kernel.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel.hpp"
#include "gpu/jit/gemm/jit_gemm_pd.hpp"
#include "gpu/jit/gemm/zero_pool.hpp"
#include "gpu/jit/jit_post_op_injector.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct gen_gemm_t : public gpu_gemm_t {
    struct pd_t : public jit_gemm_pd_t {
        using jit_gemm_pd_t::jit_gemm_pd_t;
        using kernel_desc_t = gen_gemm_nocopy_kernel_desc_t;

        DECLARE_COMMON_PD_T("jit:gemm:any", gen_gemm_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            using namespace primitive_kind;
            using namespace alg_kind;
            using smask_t = primitive_attr_t::skip_mask_t;
            using arch_t = compute::gpu_arch_t;

            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            // LIMITATIONS:
            // - runtime dims are not supported
            bool ok = true;

            auto attr_skip_mask = smask_t::scales_runtime | smask_t::post_ops;

            dev_info_ = compute_engine->device_info();
            arch_ = dev_info_->gpu_arch();
            int stepping = dev_info_->stepping_id();

            ok = set_default_formats();
            if (!ok) return status::unimplemented;

            const auto d = desc();

            // If m = 1, swap A/B to use more efficient n = 1 kernels if possible.
            eff_lda_ = desc()->lda();
            eff_ldb_ = desc()->ldb();

            bool check_lda
                    = ((desc()->transa() == dnnl_notrans && desc()->lda() == 1)
                            || (desc()->transa() == dnnl_trans));
            swap_ab_ = (desc()->m() == 1 && desc()->ldc() == 1 && check_lda);

            if (swap_ab_) {
                std::swap(eff_lda_, eff_ldb_);
                if (desc()->transa() == dnnl_notrans) eff_ldb_ = desc()->k();
            }

            // Pad leading dimensions in case of a single row/column.
            if ((d->k() == 1 && eff_transa() == dnnl_notrans)
                    || (eff_m() == 1 && eff_transa() == dnnl_trans)) {
                eff_lda_ = utils::rnd_up(eff_lda_, 16);
            }

            if ((eff_n() == 1 && eff_transb() == dnnl_notrans)
                    || (d->k() == 1 && eff_transb() == dnnl_trans)) {
                eff_ldb_ = utils::rnd_up(eff_ldb_, 16);
            }

            // Check parameters.
            if (utils::one_of(d->c_type(), s32, f16, f32, u8, s8)
                    && utils::one_of(d->a_type(), u8, s8)) {
                ok = ok && utils::one_of(d->b_type(), u8, s8);

                a_zp_ = !attr()->zero_points_.has_default_values(DNNL_ARG_SRC);
                b_zp_ = !attr()->zero_points_.has_default_values(
                        DNNL_ARG_WEIGHTS);
                if (swap_ab_) std::swap(a_zp_, b_zp_);

                int cmask_a = 0, cmask_b = 0, cmask_c = 0;
                CHECK(attr()->zero_points_.get(DNNL_ARG_WEIGHTS, &cmask_b));
                CHECK(attr()->zero_points_.get(DNNL_ARG_SRC, &cmask_a));
                CHECK(attr()->zero_points_.get(DNNL_ARG_DST, &cmask_c));
                ok &= (cmask_a == 0) && (cmask_b == 0)
                        && utils::one_of(cmask_c, 0, 1 << 0, 1 << 1);

                attr_skip_mask |= smask_t::zero_points_runtime;

                ok = ok
                        && IMPLICATION(
                                utils::one_of(d->c_type(), f32, s8, u8, f16),
                                arch_ >= arch_t::xe_hp);
            } else if (d->a_type() == bf16) {
                ok = ok && d->b_type() == bf16
                        && utils::one_of(d->c_type(), bf16, f32)
                        && utils::one_of(d->acc_type, bf16, f32);
            } else {
                ok = ok && utils::one_of(d->a_type(), f32, f16)
                        && d->b_type() == d->a_type()
                        && utils::one_of(d->acc_type, d->a_type(), f32);
            }

            ok = ok && !has_blocks() && batch_dims() <= 2
                    && !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m(), d->n(),
                            d->k(), d->lda(), d->ldb(), d->ldc(), d->batch())
                    && IMPLICATION(with_bias(),
                            utils::one_of(d->bias_type(), f32, bf16, f16)
                                    && (d->bias_desc.ndims <= 3)
                                    && utils::one_of(bias_cmask(), 0, 1, 2, 3))
                    && compute_engine->mayiuse_ngen_kernels()
                    && attr()->has_default_values(attr_skip_mask)
                    && attr()->output_scales_.mask_ == 0
                    && IMPLICATION(with_sum_ab(),
                            !with_bias()
                                    && (attr()->zero_points_.has_default_values(
                                            DNNL_ARG_DST)))
                    && attr()->post_ops_.check_sum_consistency(
                            d->c_type(), utils::one_of(d->a_type(), s8, u8));

            auto status = init_post_ops();
            if (status != status::success) return status;

            bool with_binary = (post_ops_.find(binary) != -1);

            // check GPU architecture
            ok &= utils::one_of(arch_, arch_t::gen9, arch_t::gen11,
                    arch_t::xe_lp, arch_t::xe_hp, arch_t::xe_hpg,
                    arch_t::xe_hpc);
            ok &= IMPLICATION(with_binary, arch_ >= arch_t::xe_hp);

            if (!ok) return status::unimplemented;

            bool has_systolic
                    = compute_engine->mayiuse(compute::device_ext_t::
                                      intel_subgroup_matrix_multiply_accumulate)
                    || compute_engine->mayiuse(compute::device_ext_t::
                                    intel_subgroup_split_matrix_multiply_accumulate);

            // size checks for fused reduction kernels
            if (with_sum_ab()) {
                auto mnk = d->m() * d->n() * d->k();
                if (arch_ == arch_t::xe_hpc && d->a_type() == f32)
                    ok &= (mnk <= 256 * 1024 * 1024);

                if (!ok) return status::unimplemented;
            }

            // choose kernel
            auto co_type = with_bias() ? d->bias_type()
                    : with_sum_ab()
                    ? d->sum_ab_type
                    : (utils::one_of(eff_a_type(), s8, u8) ? s32 : d->c_type());

            auto acc_type = utils::one_of(eff_a_type(), s8, u8) ? s32 : f32;

            if (d->c_type() == f16 && !has_systolic) acc_type = data_type::f16;

            if (types::data_type_size(acc_type) < 4) {
                // Limited post-op support for low-precision accumulation.
                ok &= !with_binary && IMPLICATION(with_sum_, sum_at_begin_);
            }

            kernel_desc_t::compute_mode mode = kernel_desc_t::mode_default;

            if (attr()->mayidownconvert(f32, tf32))
                mode = static_cast<decltype(mode)>(
                        mode | kernel_desc_t::mode_tf32);

            if (attr()->mayidownconvert(f32, bf16))
                mode = static_cast<decltype(mode)>(
                        mode | kernel_desc_t::mode_bf16x1);

            status = kernel_desc_.select_kernel(arch_, stepping,
                    dev_info_->eu_count(), has_systolic, mode, batch_dims(),
                    eff_transa(), eff_transb(), eff_trans_bias(), swap_ab(),
                    with_a_zero_points(), with_b_zero_points(),
                    with_c_zero_points(), with_bias(), eff_sum_ab(), alpha(),
                    beta(), post_ops_, eff_a_type(), eff_b_type(),
                    desc()->c_type(), co_type, acc_type, eff_align_a(),
                    eff_align_b(), align_c(), eff_m(), eff_n(), d->k(),
                    eff_lda(), eff_ldb(), d->ldc(), d->batch());

            if (status != status::success) return status;

            // Global k-parallel kernels don't support post-ops or non-f32/s32
            //   accumulation unless fusion is enabled.
            if (kernel_desc_.driver_info()->kParallel()
                    && !kernel_desc_.driver_info()->fusedPostOps()) {
                bool with_eltwise = (post_ops_.find(eltwise) != -1);

                ok &= !with_eltwise && !with_binary
                        && utils::one_of(d->c_type(), f32, s32);
            }

            if (!ok) return status::unimplemented;

            init_scratchpad();

            return status::success;
        }

        status_t query(query_t what, int idx, void *result) const override {
            switch ((int)what) {
                case (int)query::preferred_gpu_threads_per_eu: {
                    int grfs = kernel_desc_.driver_info()->grfCount;
                    *(int *)result = (grfs > 128) ? 4 : 8;
                    break;
                }
                default: return gpu_gemm_pd_t::query(what, idx, result);
            }
            return status::success;
        }

        bool set_default_formats() {
            using namespace data_type;
            using namespace format_tag;
            using arch_t = compute::gpu_arch_t;

            auto d = desc();

            auto m = d->m();
            auto n = d->n();
            auto k = d->k();
            auto a_t = d->a_type();
            auto b_t = d->b_type();
            auto c_t = d->c_type();
            auto a_t_sz = types::data_type_size(a_t);
            auto b_t_sz = types::data_type_size(b_t);

            bool is_f16 = utils::everyone_is(f16, a_t, b_t, c_t);
            bool is_xe_hp_plus = arch_ >= arch_t::xe_hp;

            // Rename memory descriptors following column major format.
            auto &a_desc = desc_.b_desc;
            auto &b_desc = desc_.a_desc;
            auto &c_desc = desc_.c_desc;

            memory_desc_wrapper a_mdw(&a_desc);
            memory_desc_wrapper b_mdw(&b_desc);
            memory_desc_wrapper c_mdw(&c_desc);

            bool a_any = a_mdw.format_any();
            bool b_any = b_mdw.format_any();
            bool c_any = c_mdw.format_any();

            if (!a_any && !is_md_gemm_compatible_plain_format(&a_desc))
                return false;
            if (!b_any && !is_md_gemm_compatible_plain_format(&b_desc))
                return false;
            if (!c_any && !is_md_gemm_compatible_plain_format(&c_desc, true))
                return false;

            bool is_a_trans = (desc()->transa() == dnnl_trans);
            bool is_b_trans = (desc()->transb() == dnnl_trans);

            auto lda = is_a_trans ? m : k;
            auto ldb = is_b_trans ? k : n;

            auto is_aligned = [](dim_t ld, size_t sz, int byte) {
                return ld * sz % byte == 0;
            };

            bool a_4B_aligned = is_aligned(lda, a_t_sz, 4);
            bool b_4B_aligned = is_aligned(ldb, b_t_sz, 4);
            bool ab_4B_aligned = a_4B_aligned && b_4B_aligned;

            bool a_tn_4B_aligned = is_aligned(k, a_t_sz, 4);
            bool b_tn_4B_aligned = is_aligned(k, b_t_sz, 4);
            bool ab_tn_4B_aligned = a_tn_4B_aligned && b_tn_4B_aligned;

            bool use_tn = (m <= 32 || n <= 32) && !ab_4B_aligned
                    && ab_tn_4B_aligned;

            bool batch = d->is_batched();

            auto dotrans = batch ? acb : ba;
            auto notrans = batch ? abc : ab;

            if (is_f16 && is_xe_hp_plus && use_tn) {
                if (a_any && b_any) {
                    CHECK(memory_desc_init_by_tag(a_desc, dotrans));
                    CHECK(memory_desc_init_by_tag(b_desc, notrans));
                } else if (a_any && !is_b_trans) {
                    CHECK(memory_desc_init_by_tag(a_desc, dotrans));
                } else if (b_any && is_a_trans) {
                    CHECK(memory_desc_init_by_tag(b_desc, notrans));
                }
            }

            return gpu_gemm_pd_t::set_default_formats();
        }

        void init_scratchpad() {
            const auto *info = kernel_desc()->driver_info();
            if (info->needsTempC()) {
                auto scratchpad = scratchpad_registry().registrar();

                int temp_c_sz = nstl::max(
                        (int)types::data_type_size(desc()->c_type()), 4);
                int temp_c_elems = info->wgTile(LoopM) * info->wgTile(LoopN);
                if (with_sum_ab())
                    temp_c_elems += nstl::max(
                            info->wgTile(LoopM), info->wgTile(LoopN));
                temp_c_elems = utils::rnd_up(temp_c_elems, 64);
                temp_c_elems *= max_k_sliced_groups();

                scratchpad.book(memory_tracking::names::key_gemm_accumulator,
                        temp_c_elems, temp_c_sz, 64, 65536);
            }
        }

        float alpha() const { return 1.0f; }

        float beta() const { return beta_; }

        bool with_bias() const {
            return desc()->bias_type() != data_type::undef && !bias_via_binary_;
        }

        int bias_cmask() const {
            unsigned char to_cmask[8] = {0, 4, 2, 6, 1, 5, 3, 7};
            assert(unsigned(desc()->bias_mask()) < 8);
            return with_bias() ? to_cmask[desc()->bias_mask() & 7] : -1;
        }

        sum_ab_t sum_ab() const { return desc()->sum_ab; }
        sum_ab_t eff_sum_ab() const {
            if (swap_ab() && sum_ab() == sum_ab::sum_a_row)
                return sum_ab::sum_b_col;
            if (swap_ab() && sum_ab() == sum_ab::sum_b_col)
                return sum_ab::sum_a_row;
            return sum_ab();
        }

        bool with_sum_ab() const { return sum_ab() != sum_ab::sum_none; }

        int sum_ab_cmask() const {
            switch (eff_sum_ab()) {
                default:
                case sum_ab::sum_none: return 0;
                case sum_ab::sum_a_row: return 1;
                case sum_ab::sum_b_col: return 2;
            }
        }

        bool with_a_zero_points() const { return a_zp_; }
        bool with_b_zero_points() const { return b_zp_; }

        bool with_c_zero_points() const {
            return !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
        }

        bool swap_ab() const { return swap_ab_; }

        int batch_dims() const {
            return nstl::max(desc()->c_desc.ndims - 2, 0);
        }
        bool eff_transa() const {
            return !swap_ab() ? (desc()->transa() == dnnl_trans)
                              : (desc()->transb() == dnnl_notrans);
        }
        bool eff_transb() const {
            return !swap_ab() ? (desc()->transb() == dnnl_trans) : false;
        }
        bool eff_trans_bias() const {
            return swap_ab() ? (desc()->trans_bias() == dnnl_notrans)
                             : (desc()->trans_bias() == dnnl_trans);
        }
        dim_t eff_m() const { return !swap_ab() ? desc()->m() : desc()->n(); }
        dim_t eff_n() const { return !swap_ab() ? desc()->n() : desc()->m(); }
        dim_t eff_lda() const { return eff_lda_; }
        dim_t eff_ldb() const { return eff_ldb_; }
        dim_t eff_stride_a(int dim) const {
            return !swap_ab() ? desc()->stride_a(dim) : desc()->stride_b(dim);
        }
        dim_t eff_stride_b(int dim) const {
            return !swap_ab() ? desc()->stride_b(dim) : desc()->stride_a(dim);
        }
        data_type_t eff_a_type() const {
            return !swap_ab() ? desc()->a_type() : desc()->b_type();
        }
        data_type_t eff_b_type() const {
            return !swap_ab() ? desc()->b_type() : desc()->a_type();
        }
        int eff_align_a() const {
            auto sz = types::data_type_size(eff_a_type());
            auto align = utils::max_pow2_div(eff_lda() * sz);
            for (int b = 0; b < batch_dims(); b++)
                align = nstl::min(
                        align, utils::max_pow2_div(eff_stride_a(b) * sz));
            return int(align);
        }
        int eff_align_b() const {
            auto sz = types::data_type_size(eff_b_type());
            auto align = utils::max_pow2_div(eff_ldb() * sz);
            for (int b = 0; b < batch_dims(); b++)
                align = nstl::min(
                        align, utils::max_pow2_div(eff_stride_b(b) * sz));
            return int(align);
        }
        int align_c() const {
            auto sz = types::data_type_size(desc()->c_type());
            auto align = utils::max_pow2_div(desc()->ldc() * sz);
            for (int b = 0; b < batch_dims(); b++)
                align = nstl::min(
                        align, utils::max_pow2_div(desc()->stride_c(b) * sz));
            return int(align);
        }

        const gen_gemm_nocopy_kernel_desc_t *kernel_desc() const {
            return &kernel_desc_;
        }

        int max_k_sliced_groups() const {
            const auto *info = kernel_desc()->driver_info();
            bool large_grf_mode = (info->grfCount > 128);

            auto groups = dev_info_->hw_threads(large_grf_mode)
                    / (info->wg[LoopM] * info->wg[LoopN]);
            if (info->kParallelVariable()) groups *= 2;

            return groups;
        }

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;
        size_t dyn_offset_co = 0;

        bool swap_ab_ = false;
        bool a_zp_ = false, b_zp_ = false;
        dim_t eff_lda_ = 0, eff_ldb_ = 0;

        const compute::device_info_t *dev_info_ = nullptr;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

        kernel_desc_t kernel_desc_;
    };

    gen_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    ~gen_gemm_t() {
        if (zero_pool_) release_zero_pool(zero_pool_);
    }

    status_t init(engine_t *engine) override { return init_nocopy(engine); }

    status_t init_nocopy(engine_t *engine) {
        using namespace data_type;

        auto kd = pd()->kernel_desc();
        CHECK(create_kernel(engine, nocopy_kernel_, "gemm_kernel", *kd));

        scalar_type_ = kd->scalar_type();
        const auto *info = nocopy_info();

        if (get_verbose(verbose_t::debuginfo) >= 2) {
            printf("onednn_verbose,info,gpu,gemm,kernel:%dx%d,%dx%dx%d\n",
                    info->unroll[LoopM], info->unroll[LoopN], info->wg[LoopM],
                    info->wg[LoopN], info->wg[LoopK]);
        }

        if (info->fusedBeta() || info->fusedPostOps()) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            int zg_cl = 0;
            if (info->fusedBeta()) zg_cl++;
            if (info->fusedPostOps()) zg_cl++;

            zero_pool_bytes_ = pd()->max_k_sliced_groups() * 64 * zg_cl;

            auto zg_max = pd()->dev_info_->hw_threads(false);
            auto zg_bytes_max = zg_max * 2 * 2 * 64;

            CHECK(lookup_zero_pool(compute_engine, zg_bytes_max, &zero_pool_));

            nocopy_kernel_.save_output_events();
        }

        return status::success;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    status_t launch_nocopy(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, const memory_storage_t &a,
            const memory_storage_t &b, const memory_storage_t &c,
            const memory_storage_t *ao, const memory_storage_t *bo,
            const memory_storage_t &co, const memory_storage_t *c_temp,
            int po_count, const memory_storage_t **po_src, int64_t offset_a,
            int64_t offset_b, int64_t offset_c, int32_t offset_co,
            int32_t *offset_po_src, int32_t lda, int32_t ldb, int32_t ldc,
            int32_t m, int32_t n, int32_t k, int32_t k0, float alpha,
            float beta, int32_t cmask, bool last_k_block, bool swapab,
            bool disable_hilbert) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    const CommonDriverInfo *nocopy_info() const {
        return pd()->kernel_desc()->driver_info();
    }

    compute::kernel_t nocopy_kernel_;
    compute::scalar_type_t scalar_type_;
    zero_pool_t *zero_pool_ = nullptr;
    size_t zero_pool_bytes_ = 0;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
