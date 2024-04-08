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
#include "gpu/intel/jit/gemm/microkernel_provider.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t micro_sdpa_t::pd_t::init_microkernels(engine_t *engine) {
    using namespace jit;

    assert(engine->kind() == engine_kind::gpu);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto *dev_info = compute_engine->device_info();
    auto arch = dev_info->gpu_arch();
    auto *d = desc();

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = compute_engine->mayiuse(
            compute::device_ext_t::intel_subgroup_matrix_multiply_accumulate);

    if (hw_info.gmdid == 0) return status::unimplemented;

    auto max_wg_slm = dev_info->max_slm_size_per_tg(arch);

    /* Set up GEMMProblem structure for first GEMM: K^T * Q */
    GEMMProblem problem;
    problem.Ta = problem.Ta_ext = Type::f16;
    problem.Tb = problem.Tb_ext = Type::f16;
    problem.Tc = problem.Tc_ext = Type::f32;
    problem.Ts = problem.Tc;
    problem.A.layout = MatrixLayout::T;
    problem.B.layout = MatrixLayout::N;
    problem.C.layout = MatrixLayout::N;
    problem.A.setAlignment(alignmentForLD(d->head_size() * problem.Ta));
    problem.B.setAlignment(alignmentForLD(d->head_size() * problem.Tb));
    problem.C.setAlignment(problem.Tc.size());

    /* Set up problem size information */
    SizeParams sizes;
    sizes.m = d->keys();
    sizes.n = d->queries();
    sizes.k = d->head_size();
    sizes.batch = d->batch_size();

    /* Set up special kernel requirements */
    std::vector<StrategyRequirement> reqs_kq;
    reqs_kq.push_back(StrategyRequirement::WGTileMN <= max_wg_slm / 4);
    reqs_kq.push_back(StrategyRequirement::WGTileM >= sizes.m);

    /* Ask microkernel provider for microkernel */
    try {
        gemm_kq_ = selectGEMMMicrokernel(
                micro::GEMMProtocol(), hw_info, sizes, problem, reqs_kq);
    } catch (...) { return status::unimplemented; }

    /* Update for second GEMM: V*S */
    problem.A.layout = MatrixLayout::N;
    problem.B.setAlignment(64);
    sizes.m = d->head_size();
    sizes.n = gemm_kq_.getSetting("wg_tile_n");
    sizes.k = d->keys();

    /* Set up special kernel requirements */
    int sg_per_wg = gemm_kq_.getSetting("sg_per_wg_m")
            * gemm_kq_.getSetting("sg_per_wg_n");

    std::vector<StrategyRequirement> reqs_vs;
    reqs_vs.push_back(StrategyRequirement::WGTileM
            >= sizes.m); /* could relax with loop over d */
    reqs_vs.push_back(StrategyRequirement::WGTileN >= sizes.n);
    reqs_vs.push_back(StrategyRequirement::WG == sg_per_wg);

    micro::GEMMProtocol::Options opts_vs;
    opts_vs.localB = true;

    /* Ask microkernel provider for microkernel */
    try {
        gemm_vs_ = selectGEMMMicrokernel(
                opts_vs, hw_info, sizes, problem, reqs_vs);
    } catch (...) { return status::unimplemented; }

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
    auto wg_tile_k = gemm_kq.getSetting("wg_tile_m");
    auto wg_tile_q = gemm_kq.getSetting("wg_tile_n");
    auto sg_per_wg = gemm_kq.getSetting("sg_per_wg_m")
            * gemm_kq.getSetting("sg_per_wg_n");
    auto slm_stride = std::max(wg_tile_k, 4 * sg_size_);
    auto slm = std::max<size_t>(gemm_kq.getSetting("slm_size"),
            slm_stride * wg_tile_q * sizeof(float));

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
    arg_list.set(9, slm, nullptr);

    compute::range_t lws = {(size_t)sg_size_, (size_t)sg_per_wg, 1};
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
