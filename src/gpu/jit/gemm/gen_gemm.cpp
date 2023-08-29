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

#include "gpu/jit/gemm/gen_gemm.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/float16.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/jit/gemm/gemm_walk_orders.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel_common.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t gen_gemm_t::launch_nocopy(const gemm_exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, const memory_storage_t &a,
        const memory_storage_t &b, const memory_storage_t &c,
        const memory_storage_t *ao, const memory_storage_t *bo,
        const memory_storage_t &co, const memory_storage_t *c_temp,
        int po_count, const memory_storage_t **po_srcs, int64_t offset_a,
        int64_t offset_b, int64_t offset_c, int32_t offset_co,
        int32_t *offset_po_src, int32_t lda, int32_t ldb, int32_t ldc,
        int32_t m, int32_t n, int32_t k, int32_t k0, float alpha, float beta,
        int32_t cmask, bool last_k_block, bool swapab,
        bool disable_hilbert) const {

    uint32_t flags = 0;
    bool k_parallel_fixed
            = (nocopy_info()->kParallel() || nocopy_info()->kParallelLocal())
            && !nocopy_info()->kParallelVariable();

    auto problem = pd()->kernel_desc()->problem();

    auto stride_a0 = int32_t(pd()->eff_stride_a(0));
    auto stride_b0 = int32_t(pd()->eff_stride_b(0));
    auto stride_c0 = int32_t(pd()->desc()->stride_c(0));

    auto stride_a1 = int32_t(pd()->eff_stride_a(1));
    auto stride_b1 = int32_t(pd()->eff_stride_b(1));
    auto stride_c1 = int32_t(pd()->desc()->stride_c(1));

    if (!last_k_block) flags |= FlagNonfinalKBlock;
    if (cmask & 1) flags |= FlagCOColumn;
    if (cmask & 2) flags |= FlagCORow;

    compute::kernel_arg_list_t arg_list;
    int argn = 0;

    arg_list.set(argn++, a);
    arg_list.set(argn++, b);
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

    set_scalar_arg_cvt(arg_list, argn++, alpha, scalar_type_);
    set_scalar_arg_cvt(arg_list, argn++, beta, scalar_type_);

    if (pd()->with_a_zero_points()) arg_list.set(argn++, *ao);
    if (pd()->with_b_zero_points()) arg_list.set(argn++, *bo);
    if (pd()->with_c_zero_points() || pd()->with_bias()
            || pd()->with_sum_ab()) {
        arg_list.set(argn++, co);
        arg_list.set(argn++, offset_co);
        if (pd()->with_bias()) {
            int32_t ldco = pd()->desc()->ld_bias();
            arg_list.set(argn++, ldco);
        }
    }
    if (nocopy_info()->needsTempC()) arg_list.set(argn++, *c_temp);
    for (int i = 0; i < po_count; i++) {
        if (!po_srcs[i]) continue;
        arg_list.set(argn++, *po_srcs[i]);
        arg_list.set(argn++, offset_po_src[i]);

        if (problem->binaryRow[i] && problem->binaryCol[i])
            arg_list.set(argn++, int32_t(pd()->ld_binary(i)));
    }
    arg_list.set(argn++, flags);
    if (k_parallel_fixed) arg_list.set(argn++, k0);

    std::unique_ptr<memory_storage_t> zeros;
    int zp_token = 0;
    if (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps()) {
        CHECK(zero_pool_->claim(
                compute_stream, zero_pool_bytes_, zeros, &zp_token));
        arg_list.set(argn++, *zeros);
    }

    if (pd()->batch_dims() >= 1) {
        arg_list.set(argn++, stride_a0);
        arg_list.set(argn++, stride_b0);
        arg_list.set(argn++, stride_c0);
        for (int i = 0; i < po_count; i++)
            if (problem->binaryBatch[i])
                arg_list.set(argn++, int32_t(pd()->stride_binary(i, 0)));
    }

    if (pd()->batch_dims() >= 2) {
        auto batchSize1 = uint32_t(pd()->desc()->c_desc.dims[1]);
        uint32_t recipBatchSize1 = uint32_reciprocal(batchSize1);
        arg_list.set(argn++, stride_a1);
        arg_list.set(argn++, stride_b1);
        arg_list.set(argn++, stride_c1);
        for (int i = 0; i < po_count; i++)
            if (problem->binaryBatch[i])
                arg_list.set(argn++, int32_t(pd()->stride_binary(i, 1)));
        arg_list.set(argn++, batchSize1);
        arg_list.set(argn++, recipBatchSize1);
    }

    auto lws_k = pd()->kernel_desc()->aux_params()->wgK;

    size_t gws[3] = {0, 0, 1};

    gws[0] = utils::div_up(m, nocopy_info()->unroll[LoopM]);
    gws[1] = utils::div_up(n, nocopy_info()->unroll[LoopN]);
    gws[2] = nocopy_info()->kParallel() ? nstl::max(1, utils::div_up(k, k0))
                                        : lws_k;

    size_t lws[3] = {size_t(nocopy_info()->wg[LoopM]),
            size_t(nocopy_info()->wg[LoopN]), size_t(lws_k)};

    if (nocopy_info()->isNMK()) {
        std::swap(lws[0], lws[1]);
        std::swap(gws[0], gws[1]);
    }

    if (nocopy_info()->fusedEUs() && (lws[0] > 1))
        gws[0] = utils::rnd_up(gws[0], 2);

    lws[2] = nstl::min(lws[2], gws[2]);

    if (nocopy_info()->kParallel() && nocopy_info()->kPadding())
        gws[2] += lws[2];

    int last_non_1 = 2;
    for (; last_non_1 >= 0 && (gws[last_non_1] == 1 || lws[last_non_1] == 1);
            last_non_1--)
        ;

    for (int d = 0; d < 3; d++) {
        if (nocopy_info()->fixedWG() || (gws[d] > lws[d]))
            gws[d] = utils::rnd_up(gws[d], lws[d]);
        else {
            // Workaround to avoid local ID reordering until reqd_walk_group_order implemented in UMD.
            if (pd()->arch_ >= compute::gpu_arch_t::xe_hp && d < last_non_1)
                gws[d] = utils::rnd_up_pow2(gws[d]);
            lws[d] = gws[d];
        }
    }

    lws[1] *= nocopy_info()->wgExpand;
    gws[1] *= nocopy_info()->wgExpand;

    gws[2] *= pd()->desc()->batch();

    gemm_linear_order_args(arg_list, argn, lws, gws, m, n, k, disable_hilbert,
            *nocopy_info(), pd()->kernel_desc()->aux_params(), pd()->dev_info_);

    if (nocopy_info()->perKSLM > 0) {
        size_t slm = nocopy_info()->slm;
        if (lws[2] > 1) slm = nstl::max(slm, nocopy_info()->perKSLM * lws[2]);
        arg_list.set(argn++, slm, nullptr);
    }

    lws[0] *= nocopy_info()->subgroupSize;
    gws[0] *= nocopy_info()->subgroupSize;

    auto nd_range = compute::nd_range_t(gws, lws);
    auto status = parallel_for(ctx, nd_range, nocopy_kernel_, arg_list);

    if (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps())
        zero_pool_->async_release(zp_token, compute_stream->ctx().get_deps());

    return status;
}

status_t gen_gemm_t::execute(const gemm_exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    const auto d = pd()->desc();
    const auto &problem = *pd()->kernel_desc()->problem();

    const bool swapab = pd()->swap_ab();

    auto a_type = pd()->eff_a_type();
    auto b_type = pd()->eff_b_type();
    auto c_type = d->c_type();

    const auto m = pd()->eff_m();
    const auto n = pd()->eff_n();
    auto k = d->k();

    const bool transa = pd()->eff_transa();
    const bool transb = pd()->eff_transb();

    const auto lda = pd()->eff_lda();
    const auto ldb = pd()->eff_ldb();
    auto ldc = d->ldc();
    auto ldco = pd()->with_bias() ? d->ld_bias() : 0;

    auto alpha = pd()->alpha();
    auto beta = pd()->beta();

    bool k_parallel_global = nocopy_info()->kParallel();
    bool k_parallel_fixed
            = (nocopy_info()->kParallel() || nocopy_info()->kParallelLocal())
            && !nocopy_info()->kParallelVariable();

    auto &a = swapab ? GEMM_CTX_ARG_STORAGE(a) : GEMM_CTX_ARG_STORAGE(b);
    auto &b = swapab ? GEMM_CTX_ARG_STORAGE(b) : GEMM_CTX_ARG_STORAGE(a);
    auto &c = GEMM_CTX_ARG_STORAGE(c);
    auto &c_zp = GEMM_CTX_ARG_STORAGE(c_zero_point);
    auto &bias = GEMM_CTX_ARG_STORAGE(bias);
    auto &sum_ab = GEMM_CTX_ARG_STORAGE(sum_ab);
    auto *co = &c_zp;
    memory_storage_t *ao = nullptr, *bo = nullptr;

    std::unique_ptr<memory_storage_t> c_temp;
    if (nocopy_info()->needsTempC()) {
        c_temp = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_accumulator);
    }

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
            po_offsets0[i] = po_srcs[i]->offset() / problem.Tbinary[i];

    int cmask = 0;

    if (pd()->with_c_zero_points()) {
        off_co0 = co->offset() / types::data_type_size(c_type)
                + pd()->dyn_offset_co;
        CHECK(pd()->attr()->zero_points_.get(DNNL_ARG_DST, &cmask));
    } else if (pd()->with_bias()) {
        off_co0 = bias.offset() / types::data_type_size(c_type);
        co = &bias;
        cmask = pd()->bias_cmask();
    } else if (pd()->with_sum_ab()) {
        off_co0 = sum_ab.offset() / types::data_type_size(c_type);
        co = &sum_ab;
        cmask = pd()->sum_ab_cmask();
    }

    if (pd()->with_a_zero_points() || pd()->with_b_zero_points()) {
        ao = &GEMM_CTX_ARG_STORAGE(a_zero_point);
        bo = &GEMM_CTX_ARG_STORAGE(b_zero_point);
        if (swapab) std::swap(ao, bo);
    }

    if (swapab) {
        uint8_t swap_table[4] = {0, 2, 1, 3};
        cmask = (cmask & ~3) | swap_table[cmask & 3];
    }

    status_t status;

    auto block_m = nocopy_info()->blocking[0];
    auto block_n = nocopy_info()->blocking[1];
    auto block_k = nocopy_info()->blocking[2];

    bool disable_hilbert = (k <= 64) && nocopy_info()->isHilbert();
    if (disable_hilbert) {
        block_m = nocopy_info()->blockingAlt[0];
        block_n = nocopy_info()->blockingAlt[1];
    }

    if (!utils::one_of(pd()->desc()->c_type(), data_type::f32, data_type::f16))
        block_k = k;
    if (pd()->post_ops()->len() > 0
            && pd()->post_ops()->entry_[0].kind != primitive_kind::sum)
        block_k = k;

    if (k_parallel_fixed) block_k = pd()->kernel_desc()->aux_params()->k0;

    block_m = utils::rnd_up(block_m, nocopy_info()->wgTile(LoopM));
    block_n = utils::rnd_up(block_n, nocopy_info()->wgTile(LoopN));

    int32_t k0 = 1;
    if (k_parallel_fixed) {
        k0 = block_k;
        block_k = nstl::max<dim_t>(k, 1);

        if (k_parallel_global && !nocopy_info()->fusedBeta() && beta != 1.0f
                && (k > dim_t(k0) * pd()->kernel_desc()->aux_params()->wgK)) {
            status = launch_nocopy(ctx, compute_stream, a, b, c, ao, bo, *co,
                    nullptr, po_count, po_srcs, off_a0, off_b0, off_c0,
                    int32_t(off_co0), po_offsets0, lda, ldb, ldc, m, n, 0, 1,
                    1.0f, beta, 0, false, swapab, true);
            if (status) return status;
            beta = 1.0f;
        }
    }

    for (int64_t Bk = 0; Bk < nstl::max<dim_t>(k, 1); Bk += block_k) {
        int64_t size_k = k - Bk;
        bool last_k_block = (size_k <= block_k);
        if (!last_k_block) size_k = block_k;

        for (int64_t Bm = 0; Bm < m; Bm += block_m) {
            int64_t size_m = m - Bm;
            if (size_m > block_m) size_m = block_m;

            auto off_a_src
                    = off_a0 + (!transa ? (Bm + Bk * lda) : (Bk + Bm * lda));

            for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                int64_t size_n = n - Bn;
                if (size_n > block_n) size_n = block_n;

                auto off_b_src = off_b0
                        + (!transb ? (Bk + Bn * ldb) : (Bn + Bk * ldb));

                auto off_c = off_c0 + Bm + Bn * ldc;
                auto off_co = int32_t(off_co0);
                switch (cmask & 3) {
                    case 1: off_co += Bn; break;
                    case 2: off_co += Bm; break;
                    case 3:
                        off_co += isColMajor(problem.CO.layout)
                                ? (Bn * ldco + Bm)
                                : (Bm * ldco + Bn);
                        break;
                }

                for (int i = 0; i < po_count; i++) {
                    po_offsets[i] = po_offsets0[i];
                    bool row = problem.binaryRow[i], col = problem.binaryCol[i];
                    if (row && col) {
                        auto ld = pd()->ld_binary(i);
                        po_offsets[i] += isColMajor(problem.binary[i].layout)
                                ? (Bn * ld + Bm)
                                : (Bm * ld + Bn);
                    } else if (row)
                        po_offsets[i] += Bm;
                    else if (col)
                        po_offsets[i] += Bn;
                }

                float eff_beta = (Bk == 0) ? beta : 1.0f;
                status = launch_nocopy(ctx, compute_stream, a, b, c, ao, bo,
                        *co, c_temp.get(), po_count, po_srcs, off_a_src,
                        off_b_src, off_c, off_co, po_offsets, lda, ldb, ldc,
                        size_m, size_n, size_k, k0, alpha, eff_beta, cmask,
                        last_k_block, swapab, disable_hilbert);

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
