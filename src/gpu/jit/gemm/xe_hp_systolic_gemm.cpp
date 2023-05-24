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

#include "gpu/jit/gemm/xe_hp_systolic_gemm.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/float16.hpp"
#include "common/impl_registration.hpp"
#include "common/type_helpers.hpp"
#include "gpu/jit/gemm/gemm_walk_orders.hpp"
#include "gpu/jit/utils/ngen_type_bridge.hpp"
#include "gpu/ocl/gemm/xe_systolic_gemm_copy_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t xe_hp_systolic_gemm_t::pd_t::init(engine_t *engine) {
    using namespace prop_kind;
    using namespace data_type;
    using namespace primitive_kind;
    using smask_t = primitive_attr_t::skip_mask_t;
    using arch_t = compute::gpu_arch_t;

    assert(engine->kind() == engine_kind::gpu);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    if (!compute_engine->mayiuse_ngen_kernels()) return status::unimplemented;
    if (!compute_engine->mayiuse_large_grf_mode()) return status::unimplemented;

    dev_info_ = compute_engine->device_info();
    auto arch = dev_info_->gpu_arch();

    const auto &d = desc();

    bool dt_float_ok = (d->a_type() == d->b_type()
            && utils::one_of(d->a_type(), bf16, f16)
            && utils::one_of(d->c_type(), f32, d->a_type()));

    bool dt_int_ok = (utils::one_of(d->a_type(), u8, s8)
            && utils::one_of(d->b_type(), u8, s8)
            && utils::one_of(d->c_type(), s32, f32, s8, u8, f16));

    if (dt_int_ok) {
        a_zp_ = !attr()->zero_points_.has_default_values(DNNL_ARG_SRC);
        b_zp_ = !attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS);
        c_zp_ = !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
    }

    bool ok = set_default_formats(d->a_type());
    if (!ok) return status::unimplemented;

    CHECK(attr_.set_default_formats(dst_md(0)));

    if (use_nocopy()) return status::unimplemented;

    // LIMITATIONS:
    // - batch is not supported for unpacked inputs.
    // - runtime dims are not supported
    bool limits_ok
            = !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m(), d->n(), d->k());
    if (!packed_a())
        limits_ok = limits_ok && (d->lda() != DNNL_RUNTIME_DIM_VAL)
                && (d->batch() == 1);
    if (!packed_b())
        limits_ok = limits_ok && (d->ldb() != DNNL_RUNTIME_DIM_VAL)
                && (d->batch() == 1);
    if (!packed_c())
        limits_ok = limits_ok && (d->ldc() != DNNL_RUNTIME_DIM_VAL);

    auto attr_skip_mask = smask_t::scales_runtime | smask_t::post_ops;

    if (dt_int_ok) attr_skip_mask |= smask_t::zero_points_runtime;

    bool arch_ok = utils::one_of(
            arch, arch_t::xe_hp, arch_t::xe_hpg, arch_t::xe_hpc);

    ok = ok && limits_ok && (dt_float_ok || dt_int_ok) && arch_ok
            && compute_engine->mayiuse(compute::device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate)
            && attr()->has_default_values(attr_skip_mask)
            && desc()->sum_ab == sum_ab::sum_none
            && IMPLICATION(with_bias(),
                    utils::one_of(d->bias_type(), d->a_type(), f32)
                            && utils::one_of(bias_cmask(), 0, 1, 2, 3));

    auto status = init_post_ops();
    if (status != status::success) return status;

    if (dt_int_ok) {
        ok &= IMPLICATION(a_zp_, !packed_b())
                && IMPLICATION(b_zp_, !packed_a());

        int cmask_a = 0, cmask_b = 0, cmask_c = 0;
        CHECK(attr()->zero_points_.get(DNNL_ARG_WEIGHTS, &cmask_b));
        CHECK(attr()->zero_points_.get(DNNL_ARG_SRC, &cmask_a));
        CHECK(attr()->zero_points_.get(DNNL_ARG_DST, &cmask_c));
        ok &= (cmask_a == 0) && (cmask_b == 0)
                && utils::one_of(cmask_c, 0, 1 << 0, 1 << 1);
    }

    if (!ok) return status::unimplemented;

    init_scratchpad();

    return status::success;
}

namespace {
// Use no-copy if m*n < mn_limit * mn_limit and k < k_limit.
// Zero means no limit.
struct nocopy_table_t {
    int mn_limit[2][2];
    int k_limit[2][2];
};

const nocopy_table_t xe_hp_f16_nocopy_table[] = {
        // NN    NT     TN    TT
        {{{2880, 512}, {4096, 1024}}, {{0, 0}, {0, 0}}}};

const nocopy_table_t xe_hp_x8x8s32_nocopy_table[] = {
        // NN    NT     TN    TT
        {{{1344, 576}, {4800, 384}}, {{0, 0}, {0, 0}}}};

const nocopy_table_t xe_hp_f16_nocopy_bad_ld_table[] = {
        // NN   NT     TN   TT
        {{{288, 320}, {288, 288}}, {{288, 320}, {288, 288}}}};

const nocopy_table_t xe_hp_x8x8s32_nocopy_bad_ld_table[] = {
        // NN   NT     TN   TT
        {{{656, 528}, {352, 384}}, {{656, 528}, {352, 384}}}};

const nocopy_table_t xe_hpc_f16_nocopy_table[] = {
        // NN    NT      TN    TT
        {{{8192, 8192}, {8192, 8192}}, {{0, 0}, {0, 0}}}};

const nocopy_table_t xe_hpc_x8x8s32_nocopy_table[] = {
        // NN    NT      TN    TT
        {{{2049, 3000}, {1088, 1024}}, {{0, 0}, {0, 0}}}};

const nocopy_table_t xe_hpc_f16_nocopy_bad_ld_table[] = {
        // NN    NT      TN    TT
        {{{1024, 1024}, {1024, 1024}}, {{0, 0}, {0, 0}}}};

const nocopy_table_t xe_hpc_x8x8s32_nocopy_bad_ld_table[] = {
        // NN    NT    TN   TT
        {{{624, 624}, {480, 624}}, {{0, 0}, {0, 0}}}};
} // namespace

bool xe_hp_systolic_gemm_t::pd_t::use_nocopy() {
    using namespace data_type;

    const auto &d = desc();
    bool xehpc = (dev_info_->gpu_arch() == compute::gpu_arch_t::xe_hpc);

    if (any_prepacked_ || (packed_a_ && packed_b_)) return false;

    // Use no-copy for gemv/ger cases.
    if (d->m() <= 1 || d->n() <= 1 || d->k() <= 1) return true;

    // Use no-copy implementation if one matrix is very small.
    if (d->m() < 32 && d->n() < 32) return true;
    if (d->m() < 32 && d->k() < 32) return true;
    if (d->n() < 32 && d->k() < 32) return true;

    // Use no-copy for small/medium sizes.
    if (utils::one_of(d->a_type(), bf16, f16, s8, u8)) {
        // clang-format off
        const nocopy_table_t *all_tables[2][2][3] = {
            {{xe_hp_f16_nocopy_table, xe_hp_f16_nocopy_table, xe_hp_x8x8s32_nocopy_table},
             {xe_hp_f16_nocopy_bad_ld_table, xe_hp_f16_nocopy_bad_ld_table, xe_hp_x8x8s32_nocopy_bad_ld_table}},
            {{xe_hpc_f16_nocopy_table, xe_hpc_f16_nocopy_table, xe_hpc_x8x8s32_nocopy_table},
             {xe_hpc_f16_nocopy_bad_ld_table, xe_hpc_f16_nocopy_bad_ld_table, xe_hpc_x8x8s32_nocopy_bad_ld_table}}
        };
        // clang-format on
        int type_idx = (d->a_type() == f16) ? 0 : (d->a_type() == bf16) ? 1 : 2;
        int arch_idx = xehpc ? 1 : 0;
        bool bad_ld = false;

        if (!packed_a_) {
            auto lda_bytes = d->lda() * types::data_type_size(d->a_type());
            bad_ld |= ((lda_bytes & 0x3) != 0);
        }
        if (!packed_b_) {
            auto ldb_bytes = d->ldb() * types::data_type_size(d->b_type());
            bad_ld |= ((ldb_bytes & 0x3) != 0);
        }

        auto table = all_tables[arch_idx][int(bad_ld)][type_idx];
        long mnl = table->mn_limit[d->transa()][d->transb()];
        long kl = table->k_limit[d->transa()][d->transb()];

        if ((mnl == 0 || d->m() * d->n() < mnl * mnl)
                && (kl == 0 || d->k() < kl))
            return true;
    }

    return false;
}

bool xe_hp_systolic_gemm_t::pd_t::set_default_formats(data_type_t dt) {
    using namespace format_tag;
    using new_kd_t = gen_gemm_xe_systolic_kernel_desc_t;

    auto sz = types::data_type_size(dt);
    const auto &d = desc();
    auto arch = dev_info_->gpu_arch();

    auto &a_desc = desc_.b_desc;
    auto &b_desc = desc_.a_desc;
    auto &c_desc = desc_.c_desc;

    memory_desc_wrapper a_mdw(&a_desc);
    memory_desc_wrapper b_mdw(&b_desc);
    memory_desc_wrapper c_mdw(&c_desc);

    bool a_any = a_mdw.format_any();
    bool b_any = b_mdw.format_any();
    bool c_any = c_mdw.format_any();
    bool batch = d->is_batched();

    if (batch_dims() > 1) return false;

    format_tag_t a_packed_tag_16 = undef;
    format_tag_t a_packed_tag_32 = undef;
    format_tag_t a_packed_tag_64 = undef;
    format_tag_t b_packed_tag_16 = undef;
    format_tag_t b_packed_tag_32 = undef;
    format_tag_t b_packed_tag_48 = undef;
    format_tag_t unpacked_tag = batch ? abc : ab;

    if (arch == compute::gpu_arch_t::xe_hpc) {
        a_packed_tag_64 = batch ? ((sz == 2) ? aCB4c8b16c2b : aCB4c8b16c4b)
                                : ((sz == 2) ? BA4b8a16b2a : BA4b8a16b4a);
        a_packed_tag_16 = batch ? ((sz == 2) ? aCB16c2b : aCB16c4b)
                                : ((sz == 2) ? BA16b2a : BA16b4a);
        b_packed_tag_16 = batch ? ((sz == 2) ? aBC16b16c : aBC16b32c)
                                : ((sz == 2) ? AB16a16b : AB16a32b);
    } else {
        a_packed_tag_32 = batch ? ((sz == 2) ? aCB4c8b8c2b : aCB4c8b8c4b)
                                : ((sz == 2) ? BA4b8a8b2a : BA4b8a8b4a);
        b_packed_tag_48 = batch ? ((sz == 2) ? aBC48b16c : aBC48b32c)
                                : ((sz == 2) ? AB48a16b : AB48a32b);
    }
    b_packed_tag_32 = batch ? ((sz == 2) ? aBC32b16c : aBC32b32c)
                            : ((sz == 2) ? AB32a16b : AB32a32b);

    bool a_prepacked_16 = a_mdw.matches_tag(a_packed_tag_16);
    bool a_prepacked_32 = a_mdw.matches_tag(a_packed_tag_32);
    bool a_prepacked_64 = a_mdw.matches_tag(a_packed_tag_64);
    bool bc_prepacked_16 = b_mdw.matches_tag(b_packed_tag_16)
            || c_mdw.matches_tag(b_packed_tag_16);
    bool bc_prepacked_32 = b_mdw.matches_tag(b_packed_tag_32)
            || c_mdw.matches_tag(b_packed_tag_32);
    bool bc_prepacked_48 = b_mdw.matches_tag(b_packed_tag_48)
            || c_mdw.matches_tag(b_packed_tag_48);

    any_prepacked_ = a_prepacked_16 || a_prepacked_32 || a_prepacked_64
            || bc_prepacked_16 || bc_prepacked_32 || bc_prepacked_48;

    unroll_m_ = 0;
    unroll_n_ = 0;
    alt_ = false;
    if (a_prepacked_16) unroll_m_ = 16;
    if (a_prepacked_32) unroll_m_ = 32;
    if (a_prepacked_64) unroll_m_ = 64;
    if (bc_prepacked_16) unroll_n_ = 16;
    if (bc_prepacked_32) unroll_n_ = 32;
    if (bc_prepacked_48) unroll_n_ = 48;

    new_kd_t::choose_unrolls(arch, dev_info_->eu_count(), d->a_type(),
            d->b_type(), d->c_type(), d->m(), d->n(), d->k(), d->batch(),
            unroll_m_, unroll_n_, alt_);

    format_tag_t a_packed_tag = (unroll_m_ == 64) ? a_packed_tag_64
            : (unroll_m_ == 32)                   ? a_packed_tag_32
                                                  : a_packed_tag_16;
    format_tag_t b_packed_tag = (unroll_n_ == 48) ? b_packed_tag_48
            : (unroll_n_ == 32)                   ? b_packed_tag_32
                                                  : b_packed_tag_16;
    format_tag_t c_packed_tag = b_packed_tag;

    packed_a_ = packed_b_ = packed_c_ = false;

    if (a_any) {
        if (b_zp_) {
            CHECK(memory_desc_init_by_tag(a_desc, unpacked_tag));
        } else {
            CHECK(memory_desc_init_by_tag(a_desc, a_packed_tag));
            auto ld = a_desc.padded_dims[batch ? 1 : 0];
            ld = nice_ld(ld, int(sz));
            auto &ostride = a_desc.format_desc.blocking.strides[batch ? 2 : 1];
            if (batch) {
                auto &bstride = a_desc.format_desc.blocking.strides[0];
                bstride = (bstride / ostride) * unroll_m_ * ld;
            }
            ostride = unroll_m_ * ld;
            packed_a_ = true;
        }
    } else if (!a_mdw.matches_tag(a_packed_tag)
            && !is_md_gemm_compatible_plain_format(&a_desc))
        return false;

    if (b_any) {
        if (a_zp_) {
            CHECK(memory_desc_init_by_tag(b_desc, unpacked_tag));
        } else {
            CHECK(memory_desc_init_by_tag(b_desc, b_packed_tag));
            if (unroll_n_ > 16) { // Bug in zero-padding when unroll_n_ == 16
                auto ld = b_desc.padded_dims[batch ? 2 : 1];
                ld = nice_ld(ld, int(sz));
                auto &ostride
                        = b_desc.format_desc.blocking.strides[batch ? 1 : 0];
                if (batch) {
                    auto &bstride = b_desc.format_desc.blocking.strides[0];
                    bstride = (bstride / ostride) * unroll_n_ * ld;
                }
                ostride = unroll_n_ * ld;
            }
            packed_b_ = true;
        }
    } else if (!b_mdw.matches_tag(b_packed_tag)
            && !is_md_gemm_compatible_plain_format(&b_desc))
        return false;

    if (c_any) {
        CHECK(memory_desc_init_by_tag(c_desc, c_packed_tag));
        if (unroll_n_ > 16) { // Bug in zero-padding when unroll_n_ == 16
            auto ld = c_desc.padded_dims[batch ? 2 : 1];
            ld = nice_ld(ld, int(sz));
            auto &ostride = c_desc.format_desc.blocking.strides[batch ? 1 : 0];
            if (batch) {
                auto &bstride = c_desc.format_desc.blocking.strides[0];
                bstride = (bstride / ostride) * unroll_n_ * ld;
            }
            ostride = unroll_n_ * ld;
        }
        packed_c_ = true;
    } else if (!c_mdw.matches_tag(c_packed_tag)
            && !is_md_gemm_compatible_plain_format(&c_desc, true))
        return false;

    packed_a_ = packed_a_ || a_mdw.matches_tag(a_packed_tag);
    packed_b_ = packed_b_ || b_mdw.matches_tag(b_packed_tag);
    packed_c_ = packed_c_ || c_mdw.matches_tag(b_packed_tag);

    // No 16x16 copy kernels currently.
    if ((!packed_a_ && unroll_m_ == 16) || (!packed_b_ && unroll_n_ == 16))
        return false;

    return gpu_gemm_pd_t::set_default_formats();
}

void xe_hp_systolic_gemm_t::pd_t::init_scratchpad() {
    if (packed_a() && packed_b()) return;

    auto a_type = desc()->a_type();
    auto b_type = desc()->b_type();

    auto m = desc()->m();
    auto n = desc()->n();
    auto k = desc()->k();

    int64_t align_m = unroll_m_ * 8; // TODO: this should not be hardcoded
    int64_t align_n = unroll_n_ * 8; // instead read from DriverInfo

    auto m_aligned = utils::rnd_up(m, align_m);
    auto n_aligned = utils::rnd_up(n, align_n);

    auto max_ldab_packed = max_ld_packed(k);

    auto scratchpad = scratchpad_registry().registrar();

    if (!packed_a()) {
        scratchpad.book(memory_tracking::names::key_gemm_blocked_a,
                m_aligned * max_ldab_packed, types::data_type_size(a_type), 64,
                65536);
    }

    if (!packed_b()) {
        scratchpad.book(memory_tracking::names::key_gemm_blocked_b,
                n_aligned * max_ldab_packed, types::data_type_size(b_type), 64,
                65536);
    }
}

status_t xe_hp_systolic_gemm_t::init(engine_t *engine) {
    arch_ = pd()->dev_info_->gpu_arch();
    eu_count_ = pd()->dev_info_->eu_count();

    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();

    int cmask = -1;

    if (pd()->with_c_zero_points())
        CHECK(pd()->attr()->zero_points_.get(DNNL_ARG_DST, &cmask));
    else if (pd()->with_bias())
        cmask = pd()->bias_cmask();

    switch (cmask) {
        case 0: co_kind_ = 'F'; break;
        case (1 << 1): co_kind_ = 'R'; break;
        case (1 << 0): co_kind_ = 'C'; break;
        case 3: co_kind_ = 'M'; break;
        case -1:
        default: co_kind_ = 'N'; break;
    }

    // Initialize compute kernels (assembly)
    {
        auto status = init_compute(engine);
        if (status != status::success) return status;
    }

    // Initialize copy kernels (OpenCL)
    for (bool copy_b : {false, true}) {
        for (bool clear_sum : {false, true}) {
            if (clear_sum && !pd()->with_ab_zero_points()) continue;
            if (!copy_b ? pd()->packed_a() : pd()->packed_b()) continue;

            using copy_kernel_params_t
                    = trivial_key_t<ocl::xe_systolic_gemm_copy_kernel_t>;
            compute::kernel_ctx_t kernel_ctx;

            auto trans
                    = !copy_b ? pd()->desc()->transa() : pd()->desc()->transb();
            copy_kernel_params_t params;
            CHECK(params.init(arch_, !copy_b ? a_type : b_type,
                    pd()->unroll_n(), copy_b, trans,
                    pd()->with_ab_zero_points(), clear_sum));

            // TODO: Refactor so this can be switched to 1 batch compilation.
            // Having up to 4 calls to the OpenCL compiler is sub-optimal.
            CHECK(create_kernel(engine, copy_kernel_[copy_b][clear_sum],
                    params.name(), params));
            if (!copy_kernel_[copy_b][clear_sum]) return status::runtime_error;
        }
    }

    if (verbose_debuginfo() >= 2) {
        printf("onednn_verbose,info,gpu,gemm,kernel:%dx%d,%dx%dx%d\n",
                pd()->unroll_m(), pd()->unroll_n(), compute_info_.wg[LoopM],
                compute_info_.wg[LoopN], compute_info_.wg[LoopK]);
    }

    return status::success;
}

status_t xe_hp_systolic_gemm_t::init_compute(engine_t *engine) {
    using kd_t = gen_gemm_xe_systolic_kernel_desc_t;

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    int stepping = compute_engine->device_info()->stepping_id();

    const auto d = pd()->desc();

    auto a_type = d->a_type();
    auto b_type = d->b_type();
    auto c_type = d->c_type();
    auto co_type = pd()->impl_co_type();
    auto acc_type = pd()->impl_acc_type();
    bool trans_co = pd()->with_bias() && (d->trans_bias() == dnnl_trans);

    bool may_k_block
            = (d->k() > kd_t::min_block_k(a_type)) && pd()->allow_k_blocking();
    bool got_info = false;

    auto post_ops = pd()->post_ops();
    bool with_post_ops = (post_ops->find(primitive_kind::eltwise) != -1)
            || (post_ops->find(primitive_kind::binary) != -1);

    kd_t kd_full;

    auto status = kd_full.select_kernel(arch_, stepping, eu_count_,
            pd()->with_batch(), pd()->packed_c(), trans_co,
            pd()->with_a_zero_points(), pd()->with_b_zero_points(),
            pd()->with_c_zero_points(), pd()->with_bias(), pd()->alpha(),
            pd()->beta(), *post_ops, a_type, b_type, c_type, co_type, acc_type,
            d->m(), d->n(), d->k(), d->batch(), pd()->unroll_m(),
            pd()->unroll_n(), pd()->alt());

    if (status != status::success) return status;

    problem_ = std::move(*kd_full.problem());

    for (bool first_k_block : {false, true}) {
        for (bool last_k_block : {false, true}) {
            if ((!first_k_block || !last_k_block) && !may_k_block) continue;
            if (may_k_block && last_k_block && !pd()->with_c_zero_points()
                    && !with_post_ops)
                kernel_[first_k_block][last_k_block]
                        = kernel_[first_k_block][false];
            else if (may_k_block && first_k_block && pd()->beta() == 1.0f)
                kernel_[first_k_block][last_k_block]
                        = kernel_[false][last_k_block];
            else {
                auto this_beta = pd()->beta();
                bool this_c_offset = pd()->with_c_zero_points();
                auto *this_post_ops = pd()->post_ops();
                post_ops_t no_post_ops;

                if (!first_k_block) this_beta = 1.0f;
                if (!last_k_block) {
                    this_c_offset = false;
                    this_post_ops = &no_post_ops;
                }

                kd_t kd;

                auto status = kd.select_kernel(arch_, stepping, eu_count_,
                        pd()->with_batch(), pd()->packed_c(), trans_co,
                        pd()->with_a_zero_points(), pd()->with_b_zero_points(),
                        this_c_offset, pd()->with_bias(), pd()->alpha(),
                        this_beta, *this_post_ops, a_type, b_type, c_type,
                        co_type, acc_type, d->m(), d->n(), d->k(), d->batch(),
                        pd()->unroll_m(), pd()->unroll_n(), pd()->alt());

                if (status != status::success) return status;

                if (!got_info) {
                    compute_info_ = *kd.driver_info();
                    got_info = true;
                }

                CHECK(create_kernel(engine,
                        kernel_[first_k_block][last_k_block], "gemm_kernel",
                        kd));

                if (!kernel_[first_k_block][last_k_block])
                    return status::runtime_error;
            }
        }
    }

    return status::success;
}

bool xe_hp_systolic_gemm_t::enable_mn_blocking() const {
    return (pd()->desc()->m() >= 8192) && (pd()->desc()->n() >= 8192);
}

std::tuple<int64_t, int64_t, int64_t>
xe_hp_systolic_gemm_t::get_blocking() const {
    int64_t m = pd()->desc()->m();
    int64_t n = pd()->desc()->n();
    int64_t k = pd()->desc()->k();

    int64_t unroll_k = compute_info_.unroll[LoopK];

    int64_t align_m = compute_info_.wgTile(LoopM);
    int64_t align_n = compute_info_.wgTile(LoopN);

    m = utils::rnd_up(m, align_m);
    n = utils::rnd_up(n, align_n);

    // Decide on m/n blocking.
    int64_t block_m = compute_info_.blocking[LoopM];
    int64_t block_n = compute_info_.blocking[LoopN];
    int64_t max_block_m = utils::rnd_up(m, align_m);
    int64_t max_block_n = utils::rnd_up(n, align_n);

    if (enable_mn_blocking()) {
        if (n <= block_n)
            block_m = (block_m * block_n) / n;
        else if (m <= block_m)
            block_n = (block_m * block_n) / m;
        else if (n < 2 * block_n) {
            block_n = utils::rnd_up(n / 2, align_n);
            block_m = (2 * block_m * block_n) / n;
        } else if (m < 2 * block_m) {
            block_m = utils::rnd_up(m / 2, align_m);
            block_n = (2 * block_m * block_n) / m;
        }

        block_m = utils::rnd_dn(nstl::min(block_m, max_block_m), align_m);
        block_n = utils::rnd_dn(nstl::min(block_n, max_block_n), align_n);
    } else {
        block_m = m;
        block_n = n;
    }

    // Decide on k blocking.
    int64_t block_k = compute_info_.blocking[LoopK];
    int64_t nblock_k = utils::div_up(k, block_k);
    nblock_k = nstl::max<int64_t>(nblock_k, 1);
    block_k = utils::div_up(k, nblock_k);
    block_k = nstl::max<dim_t>(block_k, 1);
    block_k = utils::rnd_up(pd()->allow_k_blocking() ? block_k : k, unroll_k);
    block_k = nstl::max<dim_t>(block_k, 1);

    return std::make_tuple(block_m, block_n, block_k);
}

status_t xe_hp_systolic_gemm_t::launch_copy(const gemm_exec_ctx_t &ctx,
        int64_t r, int64_t c, const memory_storage_t &src, int64_t offset_src,
        int64_t ld_src, const memory_storage_t &dst, int32_t offset_dst,
        int32_t ld_dst, bool copyb) const {

    using copy_kernel_t = ocl::xe_systolic_gemm_copy_kernel_t;

    if (pd()->with_ab_zero_points()) {
        auto status
                = launch_clear_sum(ctx, r, c, dst, offset_dst, ld_dst, copyb);
        if (status) return status;
    }

    int64_t unroll_k = compute_info_.unroll[LoopK];

    int64_t align_r = 0, align_c = 0;

    if (!copyb) {
        align_r = compute_info_.wgTile(LoopM);
        align_c = unroll_k;
    } else {
        align_r = unroll_k;
        align_c = compute_info_.wgTile(LoopN);
    }

    bool transa = (pd()->desc()->transa() == dnnl_trans);
    bool transb = (pd()->desc()->transb() == dnnl_trans);
    bool trans = !copyb ? transa : transb;

    auto &kernel = copy_kernel_[copyb][false];

    assert(kernel);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, r);
    arg_list.set(1, c);
    arg_list.set(2, src);
    arg_list.set(3, offset_src);
    arg_list.set(4, ld_src);
    arg_list.set(5, dst);
    arg_list.set(6, offset_dst);
    arg_list.set(7, ld_dst);

    auto elt_size = types::data_type_size(pd()->desc()->a_type());
    size_t r_threads = utils::div_up(utils::rnd_up(r, align_r),
            copy_kernel_t::unroll_r(
                    arch_, elt_size, pd()->unroll_n(), copyb, trans));
    size_t c_threads = utils::div_up(utils::rnd_up(c, align_c),
            copy_kernel_t::unroll_c(
                    arch_, elt_size, pd()->unroll_n(), copyb, trans));
    size_t sg = copy_kernel_t::subgroup_size(arch_, elt_size, copyb, trans);

    size_t r_lsz = trans ? 1 : 16;
    size_t c_lsz = trans ? 16 : 1;

    if (r_threads > r_lsz)
        r_threads = utils::rnd_up(r_threads, r_lsz);
    else
        r_lsz = r_threads;

    if (c_threads > c_lsz)
        c_threads = utils::rnd_up(c_threads, c_lsz);
    else
        c_lsz = c_threads;

    size_t gws[3] = {r_threads * sg, c_threads, 1};
    size_t lws[3] = {r_lsz * sg, c_lsz, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t xe_hp_systolic_gemm_t::launch_clear_sum(const gemm_exec_ctx_t &ctx,
        int64_t r, int64_t c, const memory_storage_t &dst, int32_t offset_dst,
        int32_t ld_dst, bool copyb) const {

    auto &kernel = copy_kernel_[copyb][true];

    assert(kernel);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, r);
    arg_list.set(1, c);
    arg_list.set(2, dst);
    arg_list.set(3, offset_dst);
    arg_list.set(4, ld_dst);

    auto elt_size = types::data_type_size(pd()->desc()->a_type());
    size_t threads = !copyb ? utils::div_up(r, pd()->unroll_m())
                            : utils::div_up(c, pd()->unroll_n());
    size_t sg = ocl::xe_systolic_gemm_copy_kernel_t::subgroup_size_clear_sum(
            arch_, elt_size, copyb);

    size_t gws[3] = {threads * sg, 1, 1};
    size_t lws[3] = {sg, 1, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t xe_hp_systolic_gemm_t::launch_compute(const gemm_exec_ctx_t &ctx,
        int32_t m, int32_t n, int32_t k, const memory_storage_t &ap,
        int64_t offset_a, int32_t lda, const memory_storage_t &bp,
        int64_t offset_b, int32_t ldb, const memory_storage_t &c,
        int64_t offset_c, int32_t ldc, float alpha, float beta,
        const memory_storage_t *ao, const memory_storage_t *bo,
        const memory_storage_t &co, int32_t offset_co, int po_count,
        const memory_storage_t **po_srcs, int32_t *offset_po_src,
        bool first_k_block, bool last_k_block, int32_t batch, int32_t stride_a,
        int32_t stride_b, int32_t stride_c) const {

    auto tg_m = compute_info_.wg[LoopM];
    auto tg_n = compute_info_.wg[LoopN];

    auto &kernel = kernel_[first_k_block][last_k_block];

    //   kernel void gemm_kernel(global char *Ap, global uchar *Bp, global int *C,
    //                           int k, int ldc,
    //                           long offsetA, long offsetB, long offsetC,
    //                           int m, int n,
    //                           float alpha, float beta,
    //                           int lda, int ldb)

    assert(kernel);

    compute::kernel_arg_list_t arg_list;
    int argn = 0;
    arg_list.set(argn++, ap);
    arg_list.set(argn++, bp);
    arg_list.set(argn++, c);
    arg_list.set(argn++, offset_a);
    arg_list.set(argn++, offset_b);
    arg_list.set(argn++, offset_c);
    arg_list.set(argn++, lda);
    arg_list.set(argn++, ldb);
    arg_list.set(argn++, ldc);
    arg_list.set(argn++, m);
    arg_list.set(argn++, n);
    arg_list.set(argn++, k);
    arg_list.set(argn++, alpha);
    arg_list.set(argn++, beta);

    if (pd()->with_a_zero_points()) arg_list.set(argn++, *ao);
    if (pd()->with_b_zero_points()) arg_list.set(argn++, *bo);
    if ((pd()->with_bias() || pd()->with_c_zero_points())) {
        arg_list.set(argn++, co);
        arg_list.set(argn++, offset_co);
        if (pd()->with_bias()) {
            int32_t ldco = pd()->desc()->ld_bias();
            arg_list.set(argn++, ldco);
        }
    }
    for (int i = 0; i < po_count; i++) {
        if (!po_srcs[i]) continue;
        arg_list.set(argn++, *po_srcs[i]);
        arg_list.set(argn++, offset_po_src[i]);

        if (problem_.binaryRow[i] && problem_.binaryCol[i])
            arg_list.set(argn++, int32_t(pd()->ld_binary(i)));
    }

    uint32_t flags = 0;
    if (co_kind_ == 'R') flags |= FlagCORow;
    if (co_kind_ == 'C') flags |= FlagCOColumn;
    if (co_kind_ == 'M') flags |= FlagCORow | FlagCOColumn;
    if (!first_k_block) flags |= FlagNoninitialKBlock;
    if (!last_k_block) flags |= FlagNonfinalKBlock;
    arg_list.set(argn++, flags);

    if (pd()->with_batch()) {
        arg_list.set(argn++, stride_a);
        arg_list.set(argn++, stride_b);
        arg_list.set(argn++, stride_c);
        for (int i = 0; i < po_count; i++)
            if (problem_.binaryBatch[i])
                arg_list.set(argn++, int32_t(pd()->stride_binary(i, 0)));
    }

    auto thread_m = utils::div_up(m, pd()->unroll_m() * tg_m) * tg_m;
    auto thread_n = utils::div_up(n, pd()->unroll_n() * tg_n) * tg_n;

    if (walk_n_first_) std::swap(thread_m, thread_n);

    size_t gws[3] = {size_t(thread_m), size_t(thread_n), 1};
    size_t lws[3] = {size_t(tg_m), size_t(tg_n), 1};
    if (pd()->with_batch()) gws[2] = batch;

    lws[1] *= compute_info_.wgExpand;
    gws[1] *= compute_info_.wgExpand;

    gemm_linear_order_args(arg_list, argn, lws, gws, m, n, false, compute_info_,
            pd()->dev_info_);

    lws[0] *= compute_info_.subgroupSize;
    gws[0] *= compute_info_.subgroupSize;

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t xe_hp_systolic_gemm_t::execute(const gemm_exec_ctx_t &ctx) const {
    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();
    auto c_type = pd()->desc()->c_type();
    auto bias_type = pd()->desc()->bias_type();

    auto m = pd()->desc()->m();
    auto n = pd()->desc()->n();
    auto k = pd()->desc()->k();
    auto batch = pd()->desc()->batch();

    bool packed_a = pd()->packed_a();
    bool packed_b = pd()->packed_b();
    bool packed_c = pd()->packed_c();

    auto lda = packed_a ? 0 : pd()->desc()->lda();
    auto ldb = packed_b ? 0 : pd()->desc()->ldb();
    auto ldc = packed_c ? pd()->ldc_packed() : pd()->desc()->ldc();
    auto ldco = pd()->with_bias() ? pd()->desc()->ld_bias() : 0;

    auto stride_a = pd()->desc()->stride_a();
    auto stride_b = pd()->desc()->stride_b();
    auto stride_c = pd()->desc()->stride_c();

    auto alpha = pd()->alpha();
    auto beta = pd()->beta();

    auto &a = GEMM_CTX_ARG_STORAGE(b);
    auto &b = GEMM_CTX_ARG_STORAGE(a);
    auto &c = GEMM_CTX_ARG_STORAGE(c);
    auto &c_zp = GEMM_CTX_ARG_STORAGE(c_zero_point);
    auto &bias = GEMM_CTX_ARG_STORAGE(bias);
    auto *co = &c_zp;
    memory_storage_t *ao = nullptr, *bo = nullptr;

    std::unique_ptr<memory_storage_t> a_packed_temp, b_packed_temp;

    if (!packed_a)
        a_packed_temp = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_blocked_a);
    if (!packed_b)
        b_packed_temp = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_blocked_b);

    auto &a_packed = packed_a ? a : *a_packed_temp;
    auto &b_packed = packed_b ? b : *b_packed_temp;

    const memory_storage_t *po_srcs[GEMM_MAX_PO];

    int po_count = pd()->post_ops()->len();
    assert(po_count <= GEMM_MAX_PO);

    for (int i = 0; i < po_count; i++) {
        auto &src = pd()->binary_srcs()[i];
        switch (src.type) {
            case pd_t::binary_src_t::binary:
                po_srcs[i]
                        = ctx.args()
                                  .exec_args
                                  .at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(src.index)
                                          | DNNL_ARG_SRC_1)
                                  .mem->memory_storage();
                break;
            case pd_t::binary_src_t::bias: po_srcs[i] = &bias; break;
            case pd_t::binary_src_t::scales:
                switch (src.index) {
                    case DNNL_ARG_WEIGHTS:
                        po_srcs[i] = &GEMM_CTX_ARG_STORAGE(a_scales);
                        break;
                    case DNNL_ARG_SRC:
                        po_srcs[i] = &GEMM_CTX_ARG_STORAGE(b_scales);
                        break;
                    case DNNL_ARG_DST:
                        po_srcs[i] = &GEMM_CTX_ARG_STORAGE(c_scales);
                        break;
                    default:
                        po_srcs[i] = nullptr;
                        assert(!"invalid scale type");
                        break;
                }
                break;
            default: po_srcs[i] = nullptr; break;
        }
    }

    size_t off_a0
            = a.offset() / types::data_type_size(a_type) + pd()->dyn_offset_a;
    size_t off_b0
            = b.offset() / types::data_type_size(b_type) + pd()->dyn_offset_b;
    size_t off_c0
            = c.offset() / types::data_type_size(c_type) + pd()->dyn_offset_c;
    size_t off_co0 = 0;

    int32_t po_offsets0[GEMM_MAX_PO] = {0}, po_offsets[GEMM_MAX_PO] = {0};
    for (int i = 0; i < po_count; i++)
        if (po_srcs[i])
            po_offsets0[i] = po_srcs[i]->offset() / problem_.Tbinary[i];

    if (pd()->with_ab_zero_points()) {
        ao = &GEMM_CTX_ARG_STORAGE(a_zero_point);
        bo = &GEMM_CTX_ARG_STORAGE(b_zero_point);
    }

    if (pd()->with_bias()) {
        off_co0 = bias.offset() / types::data_type_size(bias_type);
        co = &bias;
    }

    int64_t block_m = 0, block_n = 0, block_k = 0;
    std::tie(block_m, block_n, block_k) = get_blocking();

    auto lda_packed = pd()->lda_packed(k);
    auto ldb_packed = pd()->ldb_packed(k);

    status_t status;

    if (!packed_a) {
        assert(batch == 1);
        status = launch_copy(
                ctx, m, k, a, off_a0, lda, a_packed, 0, lda_packed, false);
        if (status) return status;
    }

    if (!packed_b) {
        assert(batch == 1);
        status = launch_copy(
                ctx, k, n, b, off_b0, ldb, b_packed, 0, ldb_packed, true);
        if (status) return status;
    }

    for (int64_t Bk = 0; Bk < nstl::max<dim_t>(k, 1); Bk += block_k) {
        int64_t size_k = k - Bk;
        bool first_k_block = (Bk == 0);
        bool last_k_block = (size_k <= block_k);
        if (!last_k_block) size_k = block_k;

        for (int64_t Bm = 0; Bm < m; Bm += block_m) {
            int64_t size_m = m - Bm;
            if (size_m > block_m) size_m = block_m;

            auto off_a_packed = Bm * lda_packed + Bk * pd()->unroll_m();
            if (packed_a) off_a_packed += off_a0;

            for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                int64_t size_n = n - Bn;
                if (size_n > block_n) size_n = block_n;

                auto off_b_packed = Bn * ldb_packed + Bk * pd()->unroll_n();
                if (packed_b) off_b_packed += off_b0;

                auto off_c = off_c0 + Bm + Bn * ldc;
                auto off_co = int32_t(off_co0);
                switch (co_kind_) {
                    case 'R': off_co += Bm; break;
                    case 'C': off_co += Bn; break;
                    case 'M':
                        off_co += isColMajor(problem_.CO.layout)
                                ? (Bn * ldco + Bm)
                                : (Bm * ldco + Bn);
                        break;
                    default: break;
                }

                for (int i = 0; i < po_count; i++) {
                    po_offsets[i] = po_offsets0[i];
                    bool row = problem_.binaryRow[i],
                         col = problem_.binaryCol[i];
                    if (row && col) {
                        auto ld = pd()->ld_binary(i);
                        po_offsets[i] += isColMajor(problem_.binary[i].layout)
                                ? (Bn * ld + Bm)
                                : (Bm * ld + Bn);
                    } else if (row)
                        po_offsets[i] += Bm;
                    else if (col)
                        po_offsets[i] += Bn;
                }

                float this_beta = first_k_block ? beta : 1.0f;
                status = launch_compute(ctx, size_m, size_n, size_k, a_packed,
                        off_a_packed, lda_packed, b_packed, off_b_packed,
                        ldb_packed, c, off_c, ldc, alpha, this_beta, ao, bo,
                        *co, off_co, po_count, po_srcs, po_offsets,
                        first_k_block, last_k_block, batch, stride_a, stride_b,
                        stride_c);
                if (status) return status;
            }
        }
    }

    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
