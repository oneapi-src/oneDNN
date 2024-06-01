/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/ocl/micro_sdpa.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/gemm/gen_gemm_kernel.hpp"
#include "gpu/intel/jit/gemm/microkernel_provider.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

namespace {

struct sdpa_config_t {
    int unroll_m_kq, unroll_n_kq; // Subgroup tile sizes for K*Q GEMM
    int unroll_m_vs, unroll_n_vs; // Subgroup tile sizes for V*S GEMM
    int wg_m_kq, wg_n_kq; // Workgroup configuration for K*Q GEMM
    int wg_m_vs, wg_n_vs; // Workgroup configuration for V*S GEMM
};

// Kernel configurations:
//  h<N> -- maximum head size = N
//  s<M> -- target sequence length = M
//   2nd -- second token (thin Q)
sdpa_config_t xehpg_h64 = {16, 32, 16, 16, 8, 4, 4, 8};

sdpa_config_t xehpc_h32 = {16, 64, 32, 16, 4, 2, 1, 8};
sdpa_config_t xehpc_h32_s32 = {16, 16, 16, 16, 2, 4, 2, 4};
sdpa_config_t xehpc_h32_2nd = {16, 64, 16, 16, 8, 1, 2, 4};

sdpa_config_t xehpc_h64 = {16, 64, 32, 16, 8, 2, 2, 8};
sdpa_config_t xehpc_h64_s64 = {32, 32, 32, 16, 4, 2, 2, 4};
sdpa_config_t xehpc_h64_s32 = {16, 16, 16, 16, 4, 2, 4, 2};
sdpa_config_t xehpc_h64_2nd = {32, 32, 32, 16, 4, 1, 2, 2};
sdpa_config_t xehpc_h64_s64_2nd = {16, 16, 16, 16, 4, 1, 4, 1};

sdpa_config_t xehpc_h128 = {16, 64, 32, 16, 16, 2, 4, 8};
sdpa_config_t xehpc_h128_s64 = {16, 32, 32, 32, 4, 2, 4, 2};
sdpa_config_t xehpc_h128_s32 = {16, 16, 16, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_h128_2nd = {32, 32, 32, 16, 8, 1, 4, 2};

sdpa_config_t xehpc_h256 = {16, 32, 32, 32, 8, 4, 8, 4};
sdpa_config_t xehpc_h256_s64 = {16, 32, 32, 32, 8, 1, 8, 1};
sdpa_config_t xehpc_h256_2nd = {16, 16, 16, 16, 16, 1, 16, 1};

sdpa_config_t *choose_config_xehpg(int head_size, int seq, bool thin_q) {
    if (head_size <= 64) return &xehpg_h64;
    return nullptr;
}

sdpa_config_t *choose_config_xehpc(int head_size, int seq, bool thin_q) {
    if (head_size <= 32) {
        if (thin_q) return &xehpc_h32_2nd;
        if (seq <= 32) return &xehpc_h32_s32;
        return &xehpc_h32;
    } else if (head_size <= 64) {
        if (thin_q) {
            if (seq <= 64) return &xehpc_h64_s64_2nd;
            return &xehpc_h64_2nd;
        }
        if (seq <= 32) return &xehpc_h64_s32;
        if (seq <= 64) return &xehpc_h64_s64;
        return &xehpc_h64;
    } else if (head_size <= 128) {
        if (thin_q) return &xehpc_h128_2nd;
        if (seq <= 32) return &xehpc_h128_s32;
        if (seq <= 64) return &xehpc_h128_s64;
        return &xehpc_h128;
    } else if (head_size <= 256) {
        if (thin_q) return &xehpc_h256_2nd;
        if (seq <= 64) return &xehpc_h256_s64;
        return &xehpc_h256;
    }
    return nullptr;
}

} /* anonymous namespace */

status_t micro_sdpa_t::pd_t::init_microkernels(impl::engine_t *engine) {
    using namespace jit;
    using arch_t = compute::gpu_arch_t;

    assert(engine->kind() == engine_kind::gpu);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto *dev_info = compute_engine->device_info();
    arch_ = dev_info->gpu_arch();
    auto *d = desc();

    /* Retrieve pre-tuned kernel configuration */
    sdpa_config_t *config = nullptr;
    bool thin_q = (d->queries() <= 16);

    switch (arch_) {
        case arch_t::xe_hpg:
            config = choose_config_xehpg(d->head_size(), d->keys(), thin_q);
            break;
        case arch_t::xe_hpc:
        case arch_t::xe2:
            config = choose_config_xehpc(d->head_size(), d->keys(), thin_q);
        default: break;
    }

    if (!config) return status::unimplemented;

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = compute_engine->mayiuse(
            compute::device_ext_t::intel_subgroup_matrix_multiply_accumulate);

    if (hw_info.gmdid == 0) return status::unimplemented;

    sg_size_ = dev_info->min_subgroup_size();

    auto convert_dnnl_to_kernel_layout = [](const memory_desc_t *md) {
        return (gemm_desc_t::get_trans(*md) == dnnl_trans) ? MatrixLayout::T
                                                           : MatrixLayout::N;
    };

    /* Set up GEMMProblem structure for first GEMM: K^T * Q */
    GEMMProblem problem;
    problem.Ta = problem.Ta_ext
            = jit::convert_dnnl_to_kernel_type(key_md()->data_type);
    problem.Tb = problem.Tb_ext
            = jit::convert_dnnl_to_kernel_type(qry_md()->data_type);
    problem.Tc = problem.Tc_ext = Type::f32;
    problem.Ts = problem.Tc;

    auto problem_kq = problem;
    problem_kq.A.layout = convert_dnnl_to_kernel_layout(key_md());
    problem_kq.B.layout = MatrixLayout::Pr;
    problem_kq.C.layout = MatrixLayout::T;
    problem_kq.A.setAlignment(alignmentForLD(d->head_size() * problem.Ta));
    problem_kq.B.setAlignment(64); // Q is packed in VNNI format in SLM
    problem_kq.B.crosspack = 2;
    problem_kq.B.tileR = d_max();
    problem_kq.B.tileC = sg_size_;

    /* Set up problem size information */
    SizeParams sizes;
    sizes.m = d->keys();
    sizes.n = d->queries();
    sizes.k = d->head_size();
    sizes.batch = d->batch_size();

    /* Set up microkernel strategy */
    std::vector<StrategyRequirement> reqs_kq;
    reqs_kq.push_back(StrategyRequirement::UnrollM == config->unroll_m_kq);
    reqs_kq.push_back(StrategyRequirement::UnrollN == config->unroll_n_kq);
    reqs_kq.push_back(StrategyRequirement::WGM == config->wg_m_kq);
    reqs_kq.push_back(StrategyRequirement::WGN == config->wg_n_kq);

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_kq;
    opts_kq.localB = true;
    opts_kq.slmPtr = true;

    /* Ask microkernel provider for microkernel */
    try {
        gemm_kq_ = selectGEMMMicrokernel(
                opts_kq, hw_info, sizes, problem_kq, reqs_kq);
    } catch (...) { return status::unimplemented; }

    /* Update for second GEMM: V*S */
    auto problem_vs = problem;
    problem_vs.Ta = problem_vs.Ta_ext
            = jit::convert_dnnl_to_kernel_type(val_md()->data_type);
    problem_vs.A.layout = convert_dnnl_to_kernel_layout(val_md());
    problem_vs.B.layout = MatrixLayout::Pr;
    problem_vs.C.layout = MatrixLayout::N;
    problem_vs.A.setAlignment(alignmentForLD(d->head_size() * problem.Ta));
    problem_vs.B.setAlignment(64); // S is packed in SLM
    problem_vs.B.crosspack = 16;
    sizes.m = d->values();
    sizes.n = gemm_kq_.getSetting("wg_tile_n");
    sizes.k = gemm_kq_.getSetting("wg_tile_m");

    /* Set up microkernel strategy */
    std::vector<StrategyRequirement> reqs_vs;
    reqs_vs.push_back(StrategyRequirement::UnrollM == config->unroll_m_vs);
    reqs_vs.push_back(StrategyRequirement::UnrollN == config->unroll_n_vs);
    reqs_vs.push_back(StrategyRequirement::WGM == config->wg_m_vs);
    reqs_vs.push_back(StrategyRequirement::WGN == config->wg_n_vs);

    micro::GEMMProtocol::Options opts_vs;
    opts_vs.localB = true;
    opts_vs.slmPtr = true;

    /* Ask microkernel provider for microkernel */
    try {
        auto adjust_vs = [](GEMMStrategy &strategy) {
            /* Enable dpasw */
            strategy.dpasw |= strategy.fused;
        };
        gemm_vs_ = selectGEMMMicrokernel(
                opts_vs, hw_info, sizes, problem_vs, reqs_vs, adjust_vs);
    } catch (...) { return status::unimplemented; }

    return status::success;
}

status_t micro_sdpa_t::init(impl::engine_t *engine) {
    using namespace micro;

    compute::kernel_ctx_t kernel_ctx;

    auto *d = pd()->desc();

    kernel_ctx.set_data_type(pd()->dst_md()->data_type);

    int ndims = 4;
    const memory_desc_wrapper qry_mdw(pd()->qry_md());
    const memory_desc_wrapper key_mdw(pd()->key_md());
    const memory_desc_wrapper val_mdw(pd()->val_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());
    const memory_desc_wrapper msk_mdw(pd()->attn_mask_md());
    using offset_t = decltype(offsets_t().src_off);
    offset_t qry_off, key_off, val_off, dst_off, msk_off;
    set_offsets(qry_mdw, qry_off);
    set_offsets(key_mdw, key_off);
    set_offsets(val_mdw, val_off);
    set_offsets(dst_mdw, dst_off);
    set_offsets(msk_mdw, msk_off);
    def_offsets(qry_off, kernel_ctx, "QRY", ndims);
    def_offsets(key_off, kernel_ctx, "KEY", ndims);
    def_offsets(val_off, kernel_ctx, "VAL", ndims);
    def_offsets(dst_off, kernel_ctx, "DST", ndims);
    def_offsets(msk_off, kernel_ctx, "MSK", ndims);
    kernel_ctx.define_int("NDIMS", ndims);

    auto ldq = gemm_desc_t::get_ld(*pd()->qry_md()) * qry_mdw.data_type_size();
    auto ldk = gemm_desc_t::get_ld(*pd()->key_md()) * key_mdw.data_type_size();
    auto ldv = gemm_desc_t::get_ld(*pd()->val_md()) * val_mdw.data_type_size();
    auto lda = gemm_desc_t::get_ld(*pd()->dst_md()) * dst_mdw.data_type_size();
    kernel_ctx.define_int("Q_ALIGN", jit::alignmentForLD(int(ldq)));
    kernel_ctx.define_int("K_ALIGN", jit::alignmentForLD(int(ldk)));
    kernel_ctx.define_int("V_ALIGN", jit::alignmentForLD(int(ldv)));
    kernel_ctx.define_int("A_ALIGN", jit::alignmentForLD(int(lda)));

    kernel_ctx.define_int("TRANSPOSE_K",
            gemm_desc_t::get_trans(*pd()->key_md()) == dnnl_trans);

    def_data_type(kernel_ctx, d->scale_dt, "SCALE");
    kernel_ctx.define_int("INVERT_SCALE", d->invert_scale);

    kernel_ctx.define_int("WITH_ATTN_MASK", pd()->with_attn_mask());

    kernel_ctx.define_int("SUBGROUP_SIZE", pd()->sg_size());
    kernel_ctx.define_int("D_MAX", pd()->d_max());

    int tile_k = pd()->gemm_kq().getSetting("wg_tile_m");
    int tile_q = pd()->gemm_kq().getSetting("wg_tile_n");
    int tile_v = pd()->gemm_vs().getSetting("wg_tile_m");

    bool d_full = (d->head_size() == pd()->d_max());
    bool v_full = (d->head_size() == tile_v);
    bool k_full = ((d->keys() % tile_k) == 0);

    kernel_ctx.define_int("REMAINDER_K", !k_full);

    if (d_full) {
        if (ldq % 4 == 0) kernel_ctx.define_int("BLOCK_Q", 1);
        if (lda % 4 == 0 && v_full) kernel_ctx.define_int("BLOCK_A", 1);
        kernel_ctx.define_int("REMAINDER_Q", (d->queries() % tile_q) != 0);
    } else if (pd()->arch() >= compute::gpu_arch_t::xe_hpc) {
        auto vbytes = d->values() * val_mdw.data_type_size();
        if (lda % 16 == 0 && vbytes % 4 == 0)
            kernel_ctx.define_int("BLOCK_2D_A", 1);
    }

    if (pd()->arch() >= compute::gpu_arch_t::xe_hpc) {
        kernel_ctx.define_int("PREFETCH_MASK", 1);
        kernel_ctx.define_int("PREFETCH_K0", 1);
        kernel_ctx.define_int("PREFETCH_K", 1);
        kernel_ctx.define_int("PREFETCH_V", 1);
        bool no_rem = d_full && v_full && (d->keys() % tile_k == 0);
        kernel_ctx.define_int("PREFETCH_REMAINDER", !no_rem);
        kernel_ctx.define_int("PREFETCH_D_MAX", nstl::min(pd()->d_max(), 64));
    }

    /* Generate microkernel shims */
    ShimOptions shimOptions;
    shimOptions.subgroupSize = pd()->sg_size();
    shimOptions.useTileOps = true;
    shimOptions.decorator = "kq";

    kernel_ctx.add_custom_header("gemm_kq.h",
            micro::generateShim(
                    pd()->gemm_kq(), HostLanguage::OpenCL_C, shimOptions));

    shimOptions.microkernelID++;
    shimOptions.decorator = "vs";

    kernel_ctx.add_custom_header("gemm_vs.h",
            micro::generateShim(
                    pd()->gemm_vs(), HostLanguage::OpenCL_C, shimOptions));

    if (pd()->gemm_kq().grfMin > 128 || pd()->gemm_vs().grfMin > 128)
        kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");

    CHECK(create_kernel(engine, &kernel_, "micro_sdpa", kernel_ctx));
    if (!kernel_) return status::runtime_error;
    return status::success;
}

status_t micro_sdpa_t::execute(const exec_ctx_t &ctx) const {
    const auto &qry = CTX_IN_STORAGE(DNNL_ARG_QUERIES);
    const auto &key = CTX_IN_STORAGE(DNNL_ARG_KEYS);
    const auto &val = CTX_IN_STORAGE(DNNL_ARG_VALUES);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    const auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    const auto &attn_mask = CTX_IN_STORAGE(DNNL_ARG_ATTN_MASK);

    const dim_t Q = pd()->desc()->queries();
    const dim_t K = pd()->desc()->keys();
    const dim_t D = pd()->desc()->head_size();

    auto &gemm_kq = pd()->gemm_kq();
    auto wg_tile_q = gemm_kq.getSetting("wg_tile_n");
    auto sg_per_wg = gemm_kq.getSetting("sg_per_wg_m")
            * gemm_kq.getSetting("sg_per_wg_n");

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, key);
    arg_list.set(1, qry);
    arg_list.set(2, val);
    arg_list.set(3, dst);
    arg_list.set(4, scale);
    arg_list.set(5, attn_mask);
    arg_list.set(6, (int)D);
    arg_list.set(7, (int)K);
    arg_list.set(8, (int)Q);

    compute::range_t lws = {(size_t)pd()->sg_size(), (size_t)sg_per_wg, 1};
    compute::range_t gws = lws;

    gws[0] *= utils::div_up(Q, wg_tile_q);
    gws[1] *= pd()->dst_md()->dims[1];
    gws[2] *= pd()->dst_md()->dims[0];

    auto nd_range = compute::nd_range_t(gws, lws);
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
