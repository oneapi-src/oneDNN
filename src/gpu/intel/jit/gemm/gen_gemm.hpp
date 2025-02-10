/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_GEMM_GEN_GEMM_HPP
#define GPU_INTEL_JIT_GEMM_GEN_GEMM_HPP

#include <assert.h>
#include <limits>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/compute/zero_pool.hpp"
#include "gpu/intel/gemm/gpu_gemm.hpp"
#include "gpu/intel/jit/gemm/gen_gemm_kernel.hpp"
#include "gpu/intel/jit/gemm/jit_gemm_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct gen_gemm_t : public gpu_gemm_t {
    struct pd_t : public jit_gemm_pd_t {
        using jit_gemm_pd_t::jit_gemm_pd_t;
        using kernel_desc_t = gen_gemm_nocopy_kernel_desc_t;

        DECLARE_COMMON_PD_T("jit:gemm:any", gen_gemm_t);

        status_t init(impl::engine_t *engine) {
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
            auto attr_skip_mask = smask_t::scales_runtime | smask_t::post_ops
                    | smask_t::fpmath_mode | smask_t::accumulation_mode
                    | smask_t::rounding_mode;
            auto &attr_zps = attr()->zero_points_;

            dev_info_ = compute_engine->device_info();
            arch_ = dev_info_->gpu_arch();
            int stepping = dev_info_->stepping_id();

            const auto d = desc();
            bool all_f8 = (utils::one_of(d->a_type(), f8_e5m2, f8_e4m3)
                    && utils::one_of(d->b_type(), f8_e5m2, f8_e4m3)
                    && utils::one_of(
                            d->c_type(), f8_e5m2, f8_e4m3, f16, bf16, f32));
            wei_decomp_ = (utils::one_of(d->c_type(), f32, f16, bf16, f8_e5m2,
                                   f8_e4m3)
                                  && utils::one_of(d->a_type(), u8, s8, s4, u4)
                                  && utils::one_of(d->b_type(), f16, f32, bf16,
                                          f8_e5m2, f8_e4m3))
                    && attr()->mayiconvert(d->a_type(), f32);
            dy_quant_enabled_
                    = (utils::one_of(d->c_type(), f32, f16, bf16)
                              && utils::one_of(d->a_type(), u8, s8, s4, u4)
                              && utils::one_of(d->b_type(), u8, s8))
                    || all_f8;
            quant_enabled_ = wei_decomp_ || dy_quant_enabled_;
            CHECK(set_default_formats(false));

            with_sround_ = attr()->rounding_mode_.get(DNNL_ARG_DST)
                    == rounding_mode::stochastic;

            // If m = 1, swap A/B to use more efficient n = 1 kernels if possible.
            eff_lda_ = d->lda();
            eff_ldb_ = d->ldb();
            eff_transa_ = d->transa() == dnnl_trans;
            eff_transb_ = d->transb() == dnnl_trans;

            bool check_lda = ((d->transa() == dnnl_notrans && d->lda() == 1)
                    || (d->transa() == dnnl_trans));
            swap_ab_ = (d->m() == 1 && d->ldc() == 1 && check_lda)
                    || d->transc() == dnnl_trans;

            if (swap_ab_) {
                std::swap(eff_lda_, eff_ldb_);
                std::swap(eff_transa_, eff_transb_);
                eff_transa_ = !eff_transa_;
                eff_transb_ = !eff_transb_;

                // Do not use transposed B when it is unnecessary
                if (eff_transb_ && eff_n() == 1) {
                    eff_transb_ = false;
                    eff_ldb_ = d->k();
                }
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

            if (quant_enabled_) {
                attr_skip_mask |= smask_t::fpmath_mode
                        | smask_t::scales_runtime_data_type
                        | smask_t::scales_runtime_groups
                        | smask_t::zero_points_runtime_data_type
                        | smask_t::zero_points_runtime_groups;
            }

            const int mask_scalar = 1 << 0;
            const int mask_per_oc = 1 << 1;
            const int mask_per_ic = 1 << 2;

            bool src_zp_2d = false;
            bool wei_zp_2d = false;
            auto wei_scales_type = data_type::undef;
            auto src_scales_type = data_type::undef;
            int wei_q2d_group_k = 0;
            int src_q2d_group_k = 0;
            int a_ndims = desc()->a_desc.ndims;
            int b_ndims = desc()->b_desc.ndims;

            // Check parameters.
            if (utils::one_of(d->c_type(), s32, f16, f32, u8, s8)
                    && utils::one_of(d->a_type(), u8, s8, u4, s4)) {
                VDISPATCH_GEMM(
                        (utils::one_of(d->b_type(), u8, s8) || wei_decomp_),
                        VERBOSE_UNSUPPORTED_DT);
                attr_skip_mask |= smask_t::zero_points_runtime;

                VDISPATCH_GEMM(IMPLICATION(utils::one_of(d->c_type(), f32, s8,
                                                   u8, f16),
                                       arch_ >= arch_t::xe_hp),
                        VERBOSE_ISA_DT_MISMATCH);
            } else if (d->a_type() == bf16) {
                VDISPATCH_GEMM(
                        d->b_type() == bf16, VERBOSE_INCONSISTENT_DT, "a", "b");
                VDISPATCH_GEMM(
                        utils::one_of(d->c_type(), bf16, f32, f8_e5m2, f8_e4m3),
                        VERBOSE_INCONSISTENT_DT, "a", "c");
                VDISPATCH_GEMM(utils::one_of(d->acc_type, bf16, f32),
                        VERBOSE_INCONSISTENT_DT, "a", "acc");
            } else if (!wei_decomp_) {
                VDISPATCH_GEMM(utils::one_of(d->a_type(), f64, f32, f16,
                                       f8_e5m2, f8_e4m3, f4_e2m1, f4_e3m0),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_GEMM(
                        (d->b_type() == d->a_type()
                                || (utils::one_of(d->a_type(), f8_e5m2, f8_e4m3)
                                        && utils::one_of(
                                                d->b_type(), f8_e5m2, f8_e4m3))
                                || (utils::one_of(d->a_type(), f4_e2m1, f4_e3m0)
                                        && utils::one_of(
                                                d->b_type(), f4_e2m1, f4_e3m0)))
                                && d->c_type() != f4_e3m0,
                        VERBOSE_INCONSISTENT_DT, "a", "b");
                VDISPATCH_GEMM(utils::one_of(d->acc_type, d->a_type(), f32),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_GEMM(IMPLICATION(utils::one_of(f64, d->a_type(),
                                                   d->b_type()),
                                       dev_info_->has_native(f64)),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_GEMM(
                        IMPLICATION(utils::one_of(f8_e5m2, f8_e4m3, d->a_type(),
                                            d->b_type(), d->c_type()),
                                arch_ >= arch_t::xe_hpc),
                        VERBOSE_ISA_DT_MISMATCH);
            }

            VDISPATCH_GEMM(!has_blocks(), VERBOSE_BLOCKING_FAIL, "");
            VDISPATCH_GEMM(
                    batch_dims() <= 4, VERBOSE_BAD_DIM, "batch", batch_dims());
            VDISPATCH_GEMM(
                    !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m(), d->n(), d->k(),
                            d->lda(), d->ldb(), d->ldc(), d->batch()),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);
            VDISPATCH_GEMM(IMPLICATION(with_bias(),
                                   utils::one_of(d->bias_type(), f64, f32, bf16,
                                           f16, f8_e5m2, f8_e4m3)
                                           && (d->bias_desc.ndims <= 6)
                                           && d->bias_mask() < 8),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_GEMM(
                    IMPLICATION(with_bias(),
                            (d->c_type() != f64 || d->bias_type() == f64)),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_GEMM(compute_engine->mayiuse_ngen_kernels(),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "ngen_kernels");
            VDISPATCH_GEMM(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(IMPLICATION(with_sum_ab(),
                                   !with_bias()
                                           && (attr_zps.has_default_values(
                                                   DNNL_ARG_DST))),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(attr()->post_ops_.check_sum_consistency(d->c_type(),
                                   utils::one_of(d->a_type(), s8, u8)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            auto valid_2d_mask = [](int mask, int ndims) {
                return utils::one_of(mask, (1 << (ndims - 1)),
                        (1 << (ndims - 1)) + (1 << (ndims - 2)));
            };

            if (!attr()->zero_points_.has_default_values()) {
                bool a_zp = !attr_zps.has_default_values(DNNL_ARG_A);
                bool b_zp = !attr_zps.has_default_values(DNNL_ARG_B);

                int cmask_a = 0, cmask_b = 0, cmask_c = 0;
                CHECK(attr_zps.get(DNNL_ARG_A, &cmask_a));
                CHECK(attr_zps.get(DNNL_ARG_B, &cmask_b));
                CHECK(attr_zps.get(DNNL_ARG_C, &cmask_c));

                src_zp_2d = attr_zps.get_groups_ndims(DNNL_ARG_B) > 1;
                wei_zp_2d = attr_zps.get_groups_ndims(DNNL_ARG_A) > 1;
                VDISPATCH_GEMM(
                        (utils::one_of(cmask_a, 0, mask_per_oc, mask_per_ic)
                                || (wei_zp_2d
                                        && valid_2d_mask(cmask_a, a_ndims)))
                                && utils::one_of(
                                        cmask_c, 0, mask_scalar, mask_per_oc),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                VDISPATCH_GEMM(utils::one_of(cmask_b, 0, mask_scalar,
                                       mask_per_oc | mask_per_ic)
                                || (src_zp_2d
                                        && valid_2d_mask(cmask_b, b_ndims)),
                        VERBOSE_UNSUPPORTED_ZP_CFG);

                ao_dims_ = a_zp ? (cmask_a != 0 ? 1 : 0) : -1;
                bo_dims_ = b_zp ? (cmask_b != 0 ? 1 : 0) : -1;
                if (wei_zp_2d) ao_dims_ = 2;
                if (src_zp_2d) bo_dims_ = 2;
                if (swap_ab_) std::swap(ao_dims_, bo_dims_);

                if (wei_zp_2d) {
                    const auto idx = DNNL_ARG_WEIGHTS;
                    auto zp_group_k = attr_zps.get_groups_ndims(idx) > 0
                            ? attr_zps.get_groups(idx)[0]
                            : 1;
                    if (zp_group_k >= d->k()) {
                        wei_zp_2d = false;
                        ao_dims_ = 1;
                    } else {
                        wei_q2d_group_k = zp_group_k;
                    }
                    const auto wei_q2d_group_n
                            = attr_zps.get_groups_ndims(idx) > 0
                            ? attr_zps.get_groups(idx)[1]
                            : 1;
                    // Non-trivial N group unsupported.
                    VDISPATCH_GEMM(
                            wei_q2d_group_n == 1, VERBOSE_UNSUPPORTED_ZP_CFG);
                }
                if (src_zp_2d) {
                    const auto idx = DNNL_ARG_SRC;
                    auto zp_group_k = attr_zps.get_groups_ndims(idx) > 0
                            ? attr_zps.get_groups(idx)[1]
                            : 1;
                    if (zp_group_k >= d->k()) {
                        src_zp_2d = false;
                        bo_dims_ = 1;
                    } else {
                        src_q2d_group_k = zp_group_k;
                    }
                    const auto src_q2d_group_m
                            = attr_zps.get_groups_ndims(idx) > 0
                            ? attr_zps.get_groups(idx)[0]
                            : 1;
                    // Non-trivial M group unsupported.
                    VDISPATCH_GEMM(
                            src_q2d_group_m == 1, VERBOSE_UNSUPPORTED_ZP_CFG);
                }
                // Zero points with non-trivial groups only supported when
                // target tensor is being dequantized.
                VDISPATCH_GEMM(
                        IMPLICATION(wei_zp_2d,
                                !dy_quant_enabled_
                                        || utils::one_of(d->a_type(), s4, u4)
                                        || wei_q2d_group_k == desc()->k()),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                VDISPATCH_GEMM(
                        IMPLICATION(src_zp_2d,
                                !dy_quant_enabled_
                                        || utils::one_of(d->b_type(), s4, u4)
                                        || src_q2d_group_k == desc()->k()),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
            }

            auto &wei_scales = attr()->scales_.get(DNNL_ARG_WEIGHTS);
            auto &src_scales = attr()->scales_.get(DNNL_ARG_SRC);

            if (quant_enabled_ && !wei_scales.has_default_groups())
                wei_scales_2d_ = true;
            if (quant_enabled_ && !src_scales.has_default_groups())
                src_scales_2d_ = true;

            for (auto s : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
                if (attr()->scales_.has_default_values(s)) continue;

                auto mask = attr()->scales_.get_mask(s);
                VDISPATCH_GEMM(utils::one_of(mask, 0, mask_scalar, mask_per_oc,
                                       mask_per_ic)
                                || (s == DNNL_ARG_WEIGHTS && wei_scales_2d_
                                        && valid_2d_mask(mask, a_ndims))
                                || (s == DNNL_ARG_SRC && src_scales_2d_
                                        && valid_2d_mask(mask, a_ndims)),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }

            if (wei_scales_2d_) {
                auto scales_group_k = wei_scales.get_group(0);
                if (scales_group_k >= d->k()) {
                    wei_scales_2d_ = false;
                } else {
                    wei_scales_type = wei_scales.get_data_type();
                    if (!wei_zp_2d)
                        wei_q2d_group_k = scales_group_k;
                    else {
                        VDISPATCH_GEMM((wei_q2d_group_k == scales_group_k),
                                VERBOSE_UNSUPPORTED_SCALES_CFG);
                    }
                }
                // Non-trivial N group unsupported.
                VDISPATCH_GEMM(wei_scales.get_group(1) == 1,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }
            if (src_scales_2d_) {
                src_scales_type = src_scales.get_data_type();
                src_po_sc_ = src_scales.get_mask() == 2;
                auto scales_group_k = src_scales.get_group(1);
                if (scales_group_k >= d->k())
                    src_scales_2d_ = false;
                else {
                    src_q2d_group_k = scales_group_k;
                    VDISPATCH_GEMM(dy_quant_enabled_
                                    && utils::one_of(eff_a_type(), s4, u4),
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                }
            }

            VDISPATCH_GEMM_SC(init_post_ops(), VERBOSE_UNSUPPORTED_POSTOP);

            bool with_binary = (post_ops_.find(binary) != -1)
                    || (post_ops_.find(prelu) != -1);
            bool with_eltwise = (post_ops_.find(eltwise) != -1);

            // Check GPU architecture.
            bool arch_ok = utils::one_of(arch_, arch_t::gen9, arch_t::gen11,
                    arch_t::xe_lp, arch_t::xe_hp, arch_t::xe_hpg,
                    arch_t::xe_hpc, arch_t::xe2, arch_t::xe3);

            VDISPATCH_GEMM(arch_ok, VERBOSE_UNSUPPORTED_ARCH, "gpu");
            VDISPATCH_GEMM(IMPLICATION(with_binary, arch_ >= arch_t::xe_hp),
                    VERBOSE_UNSUPPORTED_ARCH, "gpu");

            bool has_systolic
                    = compute_engine->mayiuse(compute::device_ext_t::
                                      intel_subgroup_matrix_multiply_accumulate)
                    || compute_engine->mayiuse(compute::device_ext_t::
                                    intel_subgroup_split_matrix_multiply_accumulate);

            bool is_integrated = compute_engine->device_info()->is_integrated();

            // Size checks for fused reduction kernels.
            if (with_sum_ab()) {
                auto mnk = d->m() * d->n() * d->k();
                if (arch_ == arch_t::xe_hpc && d->a_type() == f32)
                    VDISPATCH_GEMM(
                            (mnk <= 256 * 1024 * 1024), VERBOSE_LARGE_SHAPES);
            }

            // Wrangle data types.
            auto ao_type = with_a_zero_points()
                    ? attr_zps.get_data_type(DNNL_ARG_A)
                    : data_type::s32;
            auto bo_type = with_b_zero_points()
                    ? attr_zps.get_data_type(DNNL_ARG_B)
                    : data_type::s32;
            if (swap_ab_) std::swap(ao_type, bo_type);
            bool int_acc = utils::one_of(eff_a_type(), s8, u8);
            auto co_type = with_bias() ? d->bias_type()
                    : with_sum_ab()    ? d->sum_ab_type
                    : int_acc          ? s32
                                       : d->c_type();

            // Choose accumulation data type.
            auto acc_type = int_acc
                    ? s32
                    : (utils::one_of(f64, eff_a_type(), eff_b_type()) ? f64
                                                                      : f32);
            VDISPATCH_GEMM(
                    IMPLICATION(acc_type == f64, !with_eltwise && !with_binary),
                    VERBOSE_UNSUPPORTED_POSTOP);

            bool need_x32_acc
                    = with_binary || !IMPLICATION(with_sum_, sum_at_begin_);

            switch (attr()->acc_mode_) {
                case accumulation_mode::any:
                    if (!need_x32_acc) acc_type = data_type::undef;
                    break;
                case accumulation_mode::f16: acc_type = data_type::f16; break;
                case accumulation_mode::f32: acc_type = data_type::f32; break;
                case accumulation_mode::s32: acc_type = data_type::s32; break;
                default: break;
            }

            // Handle special compute modes.
            kernel_desc_t::compute_mode mode = kernel_desc_t::mode_default;

            if (attr()->mayiconvert(f32, tf32))
                set_mode(mode, kernel_desc_t::mode_tf32);
            if (attr()->mayiconvert(f32, bf16))
                set_mode(mode, kernel_desc_t::mode_bf16x1);
            if (attr()->mayiconvert(f32, f16))
                set_mode(mode, kernel_desc_t::mode_f16x1);
            if (attr()->mayiconvert(f32, f32))
                set_mode(mode, kernel_desc_t::mode_strict);
            if (attr()->deterministic_)
                set_mode(mode, kernel_desc_t::mode_deterministic);
            if (attr()->acc_mode_ == accumulation_mode::relaxed)
                set_mode(mode, kernel_desc_t::mode_relaxed_acc);

            if (wei_decomp_) {
                acc_type = data_type::f32;
                set_mode(mode, kernel_desc_t::mode_w_decomp);
            }

            // GEMM kernels down convert the following parameters to
            // int/uint32_t
            VDISPATCH_GEMM(std::max({eff_m(), eff_n(), d->k(), d->batch()})
                            <= std::numeric_limits<int32_t>::max(),
                    VERBOSE_SHAPE_RESTRICTION);
            VDISPATCH_GEMM(std::max({eff_lda(), eff_ldb(), d->ldc()})
                            <= std::numeric_limits<uint32_t>::max(),
                    VERBOSE_SHAPE_RESTRICTION);

            // Call kernel selector to choose a kernel.
            gpu_post_ops_t gpu_post_ops;
            CHECK(gpu_post_ops_t::make(gpu_post_ops, post_ops_, dst_md(),
                    get_post_op_specializations()));

            CHECK(kernel_desc_.select_kernel(arch_, stepping,
                    dev_info_->eu_count(), has_systolic, is_integrated, mode,
                    batch_dims(), eff_transa(), eff_transb(), eff_trans_bias(),
                    swap_ab(), ao_dims_, bo_dims_, wei_scales_2d_,
                    src_scales_2d_, with_sround_, wei_q2d_group_k,
                    src_q2d_group_k, with_c_zero_points(), with_bias(),
                    eff_sum_ab(), alpha(), beta(), eff_a_type(), eff_b_type(),
                    desc()->c_type(), ao_type, bo_type, wei_scales_type,
                    src_scales_type, co_type, acc_type, eff_align_a(),
                    eff_align_b(), align_c(), eff_m(), eff_n(), d->k(),
                    eff_lda(), eff_ldb(), d->ldc(), d->batch(),
                    std::move(gpu_post_ops)));

            // Global k-parallel kernels don't support post-ops or non-f32/s32
            //   accumulation unless fusion is enabled.
            if (kernel_desc_.driver_info()->kParallel()
                    && !kernel_desc_.driver_info()->fusedPostOps()) {
                VDISPATCH_GEMM(!with_eltwise && !with_binary
                                && utils::one_of(d->c_type(), f32, s32),
                        VERBOSE_UNSUPPORTED_POSTOP);
            }

            // Limited post-op support for low-precision accumulation.
            if (kernel_desc_.problem()->Tc.size() < 4) {
                VDISPATCH_GEMM(!need_x32_acc, VERBOSE_UNSUPPORTED_POSTOP);
            }

            // Ensure kernel can be run deterministically if required.
            if (attr()->deterministic_)
                VDISPATCH_GEMM(!kernel_desc_.driver_info()->nondeterministic(),
                        VERBOSE_DETERMINISTIC_FAIL);

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

        status_t set_default_formats(bool no_transpose_c) {
            using namespace data_type;
            using namespace format_tag;
            using arch_t = compute::gpu_arch_t;

            auto d = desc();

            auto m = d->m();
            auto n = d->n();
            auto k = d->k();
            auto a_t = (utils::one_of(d->a_type(), s4, u4)) ? s8 : d->a_type();
            auto b_t = (utils::one_of(d->b_type(), s4, u4)) ? s8 : d->b_type();
            auto c_t = d->c_type();

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
                return status::unimplemented;
            if (!b_any && !is_md_gemm_compatible_plain_format(&b_desc))
                return status::unimplemented;
            if (!c_any
                    && !is_md_gemm_compatible_plain_format(
                            &c_desc, no_transpose_c))
                return status::unimplemented;

            bool is_a_trans = (desc()->transa() == dnnl_trans);
            bool is_b_trans = (desc()->transb() == dnnl_trans);

            auto lda = is_a_trans ? m : k;
            auto ldb = is_b_trans ? k : n;

            auto is_aligned = [](dim_t ld, data_type_t dt, int byte) {
                return types::elements_to_bytes(dt, ld) % byte == 0;
            };

            bool a_4B_aligned = is_aligned(lda, a_t, 4);
            bool b_4B_aligned = is_aligned(ldb, b_t, 4);
            bool ab_4B_aligned = a_4B_aligned && b_4B_aligned;

            bool a_tn_4B_aligned = is_aligned(k, a_t, 4);
            bool b_tn_4B_aligned = is_aligned(k, b_t, 4);
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

            return gpu_gemm_pd_t::set_default_formats() ? status::success
                                                        : status::unimplemented;
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

        bool with_a_zero_points() const { return (ao_dims_ >= 0); }
        bool with_b_zero_points() const { return (bo_dims_ >= 0); }
        bool with_c_zero_points() const {
            return !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
        }
        bool with_sround() const { return with_sround_; }

        bool wei_scales_2d() const { return wei_scales_2d_; }
        bool src_scales_2d() const { return src_scales_2d_; }

        bool swap_ab() const { return swap_ab_; }

        int batch_dims() const {
            return nstl::max(desc()->c_desc.ndims - 2, 0);
        }
        bool eff_transa() const { return eff_transa_; }
        bool eff_transb() const { return eff_transb_; }
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
            auto dt = eff_a_type();
            auto align = utils::max_pow2_div(
                    types::elements_to_bytes(dt, eff_lda()));
            for (int b = 0; b < batch_dims(); b++)
                align = nstl::min(align,
                        utils::max_pow2_div(
                                types::elements_to_bytes(dt, eff_stride_a(b))));
            return int(align);
        }
        int eff_align_b() const {
            auto dt = eff_b_type();
            auto align = utils::max_pow2_div(
                    types::elements_to_bytes(dt, eff_ldb()));
            for (int b = 0; b < batch_dims(); b++)
                align = nstl::min(align,
                        utils::max_pow2_div(
                                types::elements_to_bytes(dt, eff_stride_b(b))));
            return int(align);
        }
        int align_c() const {
            auto dt = desc()->c_type();
            auto align = utils::max_pow2_div(
                    types::elements_to_bytes(dt, desc()->ldc()));
            for (int b = 0; b < batch_dims(); b++)
                align = nstl::min(align,
                        utils::max_pow2_div(types::elements_to_bytes(
                                dt, desc()->stride_c(b))));
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
        int ao_dims_ = -1, bo_dims_ = -1;
        bool a_zp_ = false, b_zp_ = false;
        bool wei_decomp_ = false;
        bool quant_enabled_ = false;
        bool dy_quant_enabled_ = false;
        bool wei_scales_2d_ = false;
        bool src_scales_2d_ = false;
        bool src_po_sc_ = false;
        dim_t eff_lda_ = 0, eff_ldb_ = 0;
        bool eff_transa_ = false, eff_transb_ = false;
        bool with_sround_ = false;

        const compute::device_info_t *dev_info_ = nullptr;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

        kernel_desc_t kernel_desc_;
    };

    gen_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    ~gen_gemm_t() {
        if (zero_pool_) release_zero_pool(zero_pool_);
    }

    status_t init(impl::engine_t *engine) override {
        return init_nocopy(engine);
    }

    status_t init_nocopy(impl::engine_t *engine) {
        using namespace data_type;

        auto kd = pd()->kernel_desc();

        CHECK(create_kernel(engine, nocopy_kernel_, "gemm_kernel", *kd));

        scalar_type_ = kd->scalar_type();
        const auto *info = nocopy_info();

        if (info->fusedBeta() || info->fusedPostOps()) {
            int zg_cl = 0;
            if (info->fusedBeta()) zg_cl++;
            if (info->fusedPostOps()) zg_cl++;

            zero_pool_bytes_ = pd()->max_k_sliced_groups() * 64 * zg_cl;

            auto zg_max = pd()->dev_info_->hw_threads(false);
            zero_pool_chunk_size_ = zg_max * 2 * 2 * 64;

#ifndef DNNL_WITH_SYCL
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            CHECK(lookup_zero_pool(compute_engine, nullptr,
                    zero_pool_chunk_size_, &zero_pool_));
#endif

            nocopy_kernel_.save_output_events();
        }

        return status::success;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    status_t launch_nocopy(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, zero_pool_t *zero_pool,
            const memory_storage_t &a, const memory_storage_t &b,
            const memory_storage_t &c, const memory_storage_t *ao,
            const memory_storage_t *bo, const memory_storage_t *a_scales,
            const memory_storage_t *b_scales, const memory_storage_t &co,
            const memory_storage_t *c_temp, const memory_storage_t *sround_seed,
            int po_count, const memory_storage_t **po_src, int64_t offset_a,
            int64_t offset_b, int64_t offset_c, int64_t offset_aq,
            int64_t offset_bq, int64_t offset_co, int64_t *offset_po_src,
            int32_t lda, int32_t ldb, int32_t ldc, int32_t m, int32_t n,
            int32_t k, int32_t k0, float alpha, float beta, int32_t cmask,
            bool last_k_block, bool swapab, bool disable_hilbert) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    const CommonDriverInfo *nocopy_info() const {
        return pd()->kernel_desc()->driver_info();
    }

    compute::kernel_t nocopy_kernel_;
    compute::scalar_type_t scalar_type_;
    zero_pool_t *zero_pool_ = nullptr;
    size_t zero_pool_bytes_ = 0;
    size_t zero_pool_chunk_size_ = 0;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
