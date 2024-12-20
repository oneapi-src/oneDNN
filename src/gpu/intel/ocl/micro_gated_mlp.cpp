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

#include "gpu/intel/ocl/micro_gated_mlp.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/scratchpad.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/gemm/gen_gemm_kernel.hpp"
#include "gpu/intel/jit/gemm/include/microkernel_provider.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

namespace {

// TODO: figure out tiling for specific gated_mlp cases
struct gated_mlp_config_t {
    //int unroll_m_gwu, unroll_n_gwu; // Subgroup tile sizes for src*gate and src*wts_up GEMM
    int unroll_m_gwu, unroll_n_gwu;
    int wg_m_gwu, wg_n_gwu; // #sg per m,n Workgroup configuration for src*gate and src*wts_up GEMM
};

// TODO: pre-tuned kernel configs
// Kernel configurations:
//gated_mlp_config_t xehpg_h32 = {32, 16, 16, 16, 2, 16, 2, 16};
//gated_mlp_config_t xehpg_h32 = {32, 16, 2, 8};
//gated_mlp_config_t xehpg_h32 = {16, 8, 4, 8};
//gated_mlp_config_t xehpg_h32 = {8, 8, 4, 4};
//gated_mlp_config_t xehpg_h32 = {8, 8, 2, 4};
gated_mlp_config_t xehpg_h32 = {8, 8, 8, 8};
//gated_mlp_config_t xehpg_h32 = {8, 32, 4, 1};

    //SUCCESS 16 8 4 8
    //SUCCESS 32 8 2 4
    //SUCCESS 32 8 2 16
    //SUCCESS 32 16 2 2
    //SUCCESS 32 16 2 4
    //SUCCESS 32 16 2 8

//gated_mlp_config_t xehpc_h32 = {16, 64, 32, 16, 4, 2, 1, 8};

//TODO: determine strategy selection, primarily based on B? where B = {1,8,1024}
// OC > IC? IC > OC?
// ensure for BMG and LNL, hpc optional?
gated_mlp_config_t *choose_config_xehpg(int B, int IC, int OC) {
    if (B <= 32) return &xehpg_h32;
    return nullptr;
}

gated_mlp_config_t *choose_config_xehpc(int B, int IC, int OC) {
    return nullptr;
}

} /* anonymous namespace */

status_t micro_gated_mlp_t::pd_t::init_microkernels(impl::engine_t *engine) {
    using namespace jit;
    using arch_t = compute::gpu_arch_t;

    assert(engine->kind() == engine_kind::gpu);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto *dev_info = compute_engine->device_info();
    arch_ = dev_info->gpu_arch();
    auto *d = desc();

    VCONDCHECK(primitive, create, check, gated_mlp, mayiuse_microkernels(engine),
            status::unimplemented,
            "Microkernels not supported by the OpenCL driver.");

    /* Retrieve pre-tuned kernel configuration */
    gated_mlp_config_t *config = nullptr;

    switch (arch_) {
        case arch_t::xe_hpg:
            //config = choose_config_xehpg(d->head_size(), d->keys(), thin_q); // TODO
            break;
        case arch_t::xe_hpc:
        case arch_t::xe2:
            //config = choose_config_xehpc(d->head_size(), d->keys(), thin_q);
        default: break;
    }
    config = &xehpg_h32; // TODO

    if (!config) return status::unimplemented;

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = compute_engine->mayiuse(
            compute::device_ext_t::intel_subgroup_matrix_multiply_accumulate);

    if (hw_info.gmdid == 0) return status::unimplemented;

    sg_size_ = dev_info->min_subgroup_size();

    /* Set up GEMMProblem structure for first GEMM sizes: src * W_gate, src * W_up */
    GEMMProblem problem;
    problem.Ta = problem.Ta_ext
            = jit::convert_dnnl_to_kernel_type(src_md()->data_type);
    problem.Tb = problem.Tb_ext
            = jit::convert_dnnl_to_kernel_type(W_gate_md()->data_type);
    problem.Tc = problem.Tc_ext = Type::f32; //TODO: return fp16? or is Tc_ext in SLM
    problem.Ts = problem.Tc;

    auto problem_wgu = problem;
    // Transpose A and B for C^T = B^T * A^T, allowing slm B to be shared for both matrix multiplies
    problem_wgu.A.layout = MatrixLayout::N;
    problem_wgu.B.layout = MatrixLayout::Pr;
    problem_wgu.C.layout = MatrixLayout::T;

    problem_wgu.A.setAlignment(alignmentForLD(d->oc_sz() * problem.Ta)); // OC lda? for wgu?
    //problem_wgu.A.setAlignment(alignmentForLD(d->ic_sz() * problem.Ta));
    problem_wgu.B.setAlignment(64); // STF?? Q is packed in VNNI format in SLM
    problem_wgu.B.crosspack = 2;

    //problem_wgu.B.tileR = into<uint16_t>(b_max()); //TODO: determine programatically? exhaust MB dimension? possible for 1024?
    problem_wgu.B.tileR = into<uint16_t>(64); //TODO: will this be right ? //tileR,tileC ->transposed internally if layout.C == T???
    problem_wgu.B.tileC = into<uint16_t>(sg_size_);

    /* Set up transposed problem size */
    SizeParams sizes;
    sizes.m = d->oc_sz();
    sizes.n = d->mb_sz();
    sizes.k = d->ic_sz();
    sizes.batch = 1;

    std::vector<StrategyRequirement> reqs_wgu;
    reqs_wgu.push_back(StrategyRequirement::UnrollM == config->unroll_m_gwu);
    reqs_wgu.push_back(StrategyRequirement::UnrollN == config->unroll_n_gwu);
    reqs_wgu.push_back(StrategyRequirement::WGM == config->wg_m_gwu);
    reqs_wgu.push_back(StrategyRequirement::WGN == config->wg_n_gwu);

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_wgu;
    opts_wgu.localA = false; //TODO: local A? since reused?
    opts_wgu.localB = true;
    opts_wgu.slmPtr = true;

    //std::cout << "problemStr: " << problem.toString() << std::endl;
    /* Ask microkernel provider for microkernel */
    //printf("SUCCESS config->sg_unroll_mn %d %d config->wg_m,n %d %d\n", config->unroll_m_gwu, config->unroll_n_gwu, config->wg_m_gwu, config->wg_n_gwu);

    try {
        gemm_gateup_ = selectGEMMMicrokernel(
                opts_wgu, hw_info, sizes, problem_wgu, reqs_wgu);
    } catch (std::exception &e) { std::cout << "MICROKERNEL EXCEPTION" << e.what() << std::endl; return status::unimplemented; }

    return status::success;
}

status_t micro_gated_mlp_t::init(impl::engine_t *engine) {
    using namespace micro;

    compute::kernel_ctx_t kernel_ctx;

    kernel_ctx.set_data_type(pd()->dst_md()->data_type);

    int ndims = 2;
    const memory_desc_wrapper src_mdw(pd()->src0_md());
    const memory_desc_wrapper W_gate_mdw(pd()->W_gate_md());
    const memory_desc_wrapper W_up_mdw(pd()->W_up_md());
    const memory_desc_wrapper W_down_mdw(pd()->W_down_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());

    using offset_t = decltype(offsets_t().src_off);
    offset_t src_off, W_gate_off, W_up_off, W_down_off, dst_off;
    set_offsets(src_mdw, src_off);
    set_offsets(W_gate_mdw, W_gate_off);
    set_offsets(W_up_mdw, W_up_off);
    set_offsets(W_down_mdw, W_down_off);
    set_offsets(dst_mdw, dst_off);

    def_offsets(src_off, kernel_ctx, "SRC", ndims);
    def_offsets(W_gate_off, kernel_ctx, "W_GATE", ndims);
    def_offsets(W_up_off, kernel_ctx, "W_UP", ndims);
    def_offsets(W_down_off, kernel_ctx, "W_DOWN", ndims);
    def_offsets(dst_off, kernel_ctx, "DST", ndims);
    kernel_ctx.define_int("NDIMS", ndims);

    kernel_ctx.define_int("SIZE_MB", pd()->desc()->mb_sz());

    auto lds   = gemm_desc_t::get_ld(*pd()->src_md()) * src_mdw.data_type_size();
    auto ldwgu = gemm_desc_t::get_ld(*pd()->W_gate_md()) * W_gate_mdw.data_type_size();
    auto ldwd  = gemm_desc_t::get_ld(*pd()->W_down_md()) * W_down_mdw.data_type_size();
    auto lda   = gemm_desc_t::get_ld(*pd()->dst_md()) * dst_mdw.data_type_size();

    kernel_ctx.define_int("SRC_ALIGN", jit::alignmentForLD(int(lds)));
    kernel_ctx.define_int("WGU_ALIGN", jit::alignmentForLD(int(ldwgu)));
    kernel_ctx.define_int("WD_ALIGN", jit::alignmentForLD(int(ldwd)));
    kernel_ctx.define_int("A_ALIGN", jit::alignmentForLD(int(lda)));

    //TODO: add zp + scale + activation??
    //kernel_ctx.define_int("WITH_ATTN_SCALE", pd()->with_attn_scale());

    kernel_ctx.define_int("SUBGROUP_SIZE", pd()->sg_size());
    kernel_ctx.define_int("B_MAX", 64); // 8*8
    //kernel_ctx.define_int("B_MAX", 16);

    //TODO: this needs to change? tiling should happen accross ic dimension
    int tile_wgu_m = pd()->gemm_gateup().getSetting("wg_tile_m"); // partition tiles primarily across inner dim - k
    int tile_wgu_n = pd()->gemm_gateup().getSetting("wg_tile_n");

    auto *d = pd()->desc();
    //TODO:  mb,oc,ic full
    bool s_full = ((d->ic_sz() % tile_wgu_m) == 0); //todo: check m?
    bool b_full = (d->mb_sz() == pd()->b_max()); //todo: exhaust MB??
    //bool b_full = (d->mb_sz() % tile_ic_m == 0);
    //bool k_full = ((d->keys() % tile_k) == 0);

    kernel_ctx.define_int("REMAINDER_SRC", !b_full); //todo: %n instead??
                                                     //
    if (b_full) {
        //todo: determine tile load types
        if (lds % 4 == 0) kernel_ctx.define_int("BLOCK_SRC", 1);
        //if (ldwgu % 4 == 0) kernel_ctx.define_int("BLOCK_WGU", 1);
        if (lda % 4 == 0 && b_full) kernel_ctx.define_int("BLOCK_A", 1);
        kernel_ctx.define_int("REMAINDER_WGU", (d->ic_sz() % tile_wgu_n) != 0);
    } else if (pd()->arch() >= compute::gpu_arch_t::xe_hpc) {
        auto vbytes = d->mb_sz() * W_down_mdw.data_type_size();
        if (lda % 16 == 0 && vbytes % 4 == 0)
            kernel_ctx.define_int("BLOCK_2D_A", 1);
    }

    /* Generate microkernel shims */
    ShimOptions shimOptions;
    shimOptions.subgroupSize = pd()->sg_size();
    shimOptions.useTileOps = true;
    shimOptions.decorator = "wgu";

    //TODO: enable shim header
    kernel_ctx.add_custom_header("gemm_gateup.h",
            micro::generateShim(
                    pd()->gemm_gateup(), HostLanguage::OpenCL_C, shimOptions));

    // large grf needed?
    if (pd()->gemm_gateup().grfMin > 128 || pd()->gemm_down().grfMin > 128) {
        kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");
    }

    CHECK(create_kernel(engine, &fused_mlp_kernel_, "micro_gated_mlp", kernel_ctx));
    if (!fused_mlp_kernel_) return status::runtime_error;

    //TODO: rnn style mm here? or allow graph api to handle?
    //TODO: or second kernel?? fc_down_mm_kernel_

    return status::success;
}

void micro_gated_mlp_t::pd_t::init_scratchpad() {
    auto *d = desc();
    const size_t size = d->mb_sz() * d->oc_sz(); // TODO: just intermediate mm or reduction instead? much greater memory demands
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_gated_mlp_reduction, size,
            types::data_type_size(data_type::f16), OCL_BUFFER_ALIGNMENT);
}

status_t micro_gated_mlp_t::execute(const exec_ctx_t &ctx) const {
    const auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    const auto &W_gate = CTX_IN_STORAGE(DNNL_ARG_WTS_GATE);
    const auto &W_up = CTX_IN_STORAGE(DNNL_ARG_WTS_UP);
    const auto &W_down = CTX_IN_STORAGE(DNNL_ARG_WTS_DOWN);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const dim_t MB = pd()->desc()->mb_sz();
    const dim_t IC = pd()->desc()->ic_sz();
    const dim_t OC = pd()->desc()->oc_sz();

    auto &gemm_gateup = pd()->gemm_gateup();

    auto wg_tile_OC = gemm_gateup.getSetting("wg_tile_m"); //TODO: which? OC along lda should get largest tile dim?
    auto wg_tile_MB = gemm_gateup.getSetting("wg_tile_n");

    auto sg_per_wg  = gemm_gateup.getSetting("sg_per_wg_m")
                    * gemm_gateup.getSetting("sg_per_wg_n");

//   printf("wg_tile_m %d wg_tile_n %d sg_per_wg_m %d sg_per_wg_n %d \n", 
//       gemm_gateup.getSetting("wg_tile_m"), //64
//       gemm_gateup.getSetting("wg_tile_n"), //64
//       gemm_gateup.getSetting("sg_per_wg_m"), //4
//       gemm_gateup.getSetting("sg_per_wg_n")); //8

    std::unique_ptr<memory_storage_t> tmp_reduce;
    tmp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_gated_mlp_reduction);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, W_gate);
    arg_list.set(2, W_up);
    arg_list.set(3, W_down);
    arg_list.set(4, dst);
    arg_list.set(5, MB);
    arg_list.set(6, IC);
    arg_list.set(7, OC);
    arg_list.set(8, *tmp_reduce);

    compute::range_t lws = {(size_t)pd()->sg_size(), (size_t)sg_per_wg, 1};
    compute::range_t gws = lws;

    gws[0] *= utils::div_up(OC, wg_tile_OC);
    gws[2] *= utils::div_up(MB, wg_tile_MB);

    auto nd_range = compute::nd_range_t(gws, lws);
    //printf("gws[%zu %zu %zu] lws[%zu %zu %zu] \n",gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
    auto s = parallel_for(ctx, nd_range, fused_mlp_kernel_, arg_list);
    return s;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
