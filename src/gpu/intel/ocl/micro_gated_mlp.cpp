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
#include "common/scratchpad.hpp"
#include "common/type_helpers.hpp"
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
    int unroll_m_gwu, unroll_n_gwu; // Subgroup tile sizes for src*gate and src*wts_up GEMM
    int wg_m_gwu, wg_n_gwu; // #sg per m,n Workgroup configuration for src*gate and src*wts_up GEMM
};

// TODO: pre-tuned kernel configs
// Kernel configurations:
//gated_mlp_config_t xehpg_h32 = {32, 16, 16, 16, 2, 16, 2, 16};
//gated_mlp_config_t xehpg_h32 = {32, 16, 2, 8};
//gated_mlp_config_t xehpg_h32 = {16, 8, 4, 8};
//gated_mlp_config_t xehpg_h32 = {8, 8, 4, 4};
//gated_mlp_config_t xehpg_h32 = {8, 8, 2, 4};
//gated_mlp_config_t xehpg_h32 = {8, 8, 2, 2};

//gated_mlp_config_t xehpg_h32 = {16, 8, 2, 4};

//gated_mlp_config_t xehpg_h32 = {8, 8, 4, 4};

// gated_mlp_config_t xehpg_h32 = {8, 32, 4, 1};

//gated_mlp_config_t xehpg_h32 = {16, 16, 8, 1};
//gated_mlp_config_t xehpg_h32 = {16, 16, 16, 1};
//gated_mlp_config_t xehpg_h32 = {16, 16, 8, 8};
//gated_mlp_config_t xehpg_h32 = {16, 16, 16, 1};

//gated_mlp_config_t xehpg_h32 = {8, 8, 4, 4};
gated_mlp_config_t xehpg_h32 = { 16, 16, 2, 2 };
//gated_mlp_config_t xehpg_h32 = {16, 16, 16, 1}; //big K
//gated_mlp_config_t xehpg_h32 = {32, 32, 1, 1};
//gated_mlp_config_t xehpg_h32 = {16, 16, 32, 2};
//gated_mlp_config_t xehpg_h32 = {8, 8, 8, 2}; // dg2
//gated_mlp_config_t xehpg_h32 = {32, 32, 1, 1};
//gated_mlp_config_t xehpg_h32 = {32, 32, 8, 1};
//gated_mlp_config_t xehpg_h32 = {32, 32, 16, 1};


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

/// Returns true if a common scales value is used for each slice of the
/// tensor operation
bool with_quantize_common(const quant_entry_t &scales) {
    return !scales.has_default_values() && (scales.get_mask() == 0);
}

/// Returns true if a common zero points value is used for each slice of the
/// tensor operation
bool with_quantize_common(const zero_points_t &zp) {
    int mask = zp.get_mask(DNNL_ARG_WEIGHTS);
    return !zp.has_default_values() && (mask == 0);
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
        case arch_t::xe3:
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
    problem.Ta_ext = jit::convert_dnnl_to_kernel_type(W_gate_md()->data_type);
    problem.Tb_ext = jit::convert_dnnl_to_kernel_type(src_md()->data_type);
    problem.Ta = problem.Tb = Type::f16;
    problem.Tc = problem.Tc_ext = Type::f32; //TODO: return fp16? or is Tc_ext in SLM
    problem.Ts = problem.Tc;

    auto problem_wgu = problem;
    // Transpose A and B for C^T = B^T * A^T, allowing slm B to be shared for both matrix multiplies
    problem_wgu.A.layout = MatrixLayout::N;
    problem_wgu.B.layout = MatrixLayout::Pr;
    problem_wgu.C.layout = MatrixLayout::T;

    const memory_desc_wrapper W_gate_mdw(W_gate_md());
    auto ldgu = static_cast<int>(
            gemm_desc_t::get_ld(*W_gate_md()) * W_gate_mdw.data_type_size()); // todo: / elems_per_byte??
    problem_wgu.A.setAlignment(alignmentForLD(ldgu));
    problem_wgu.B.setAlignment(64);
    problem_wgu.B.crosspack = 2;

    int tileM = config->unroll_m_gwu * config->wg_m_gwu;
    problem_wgu.B.tileR = into<uint16_t>(tileM);

    /* Quantization configuration */
    // group configuration common for gate/up matrices
    bool wgu_common_scales = with_quantize_common(d->wts_gate_scales);
    bool wgu_common_zp     = with_quantize_common(d->wts_gate_zero_points);

    if(with_wts_gate_scales() && !wgu_common_scales) {
        auto scale_dt = wts_gate_scales_dt();
        problem_wgu.Ta_scale = jit::convert_dnnl_to_kernel_type(scale_dt); //TODO: * whatever is lda
        //problem_wgu.A_scale.alignment = uint8_t(d->ic_sz() * types::data_type_size(scale_dt));
        problem_wgu.A_scale.alignment = uint8_t(types::data_type_size(scale_dt));
        problem_wgu.A_scale.layout = MatrixLayout::N; //TODO: T? or N?, data is row major //umar says always n
        problem_wgu.aScale2D = true;
    }
    if(with_wts_gate_zp()) {
       auto zp_dt = wts_gate_zp_dt();
       problem_wgu.Tao = jit::convert_dnnl_to_kernel_type(zp_dt);
       //problem_wgu.AO.alignment = uint8_t(d->ic_sz() * types::data_type_size(zp_dt));//TODO: * whatever is lda
       problem_wgu.AO.alignment = uint8_t(types::data_type_size(zp_dt));//TODO: * whatever is lda
       problem_wgu.AO.layout = MatrixLayout::N; //N?
       problem_wgu.aoPtrDims = wgu_common_zp ? 0 : 2;
       problem_wgu.aOffset = ABOffset::Calc;
    }

    if (with_wts_gate_scales() || with_wts_gate_zp()) {
        problem_wgu.aqGroupM = 1;
        problem_wgu.aqGroupK = (wgu_common_scales || wgu_common_zp) ? 1 : wts_gate_group_size();
    }

    /* Set up transposed problem size */
    SizeParams sizes;
    sizes.m = d->oc_sz();
    sizes.n = d->mb_sz();
    sizes.k = d->ic_sz();
    sizes.batch = 1;

    std::vector<StrategyRequirement> reqs_wgu;

    reqs_wgu.push_back(StrategyRequirement::UnrollM == config->unroll_m_gwu); //affects A
    reqs_wgu.push_back(StrategyRequirement::UnrollN == config->unroll_n_gwu); //affects B

    reqs_wgu.push_back(StrategyRequirement::WGM == config->wg_m_gwu);
    reqs_wgu.push_back(StrategyRequirement::WGN == config->wg_n_gwu);

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_wgu;
    opts_wgu.localB = true;
    opts_wgu.slmPtr = true;

    /* Enable quantization options */
    opts_wgu.scaleA = with_wts_gate_scales() && !wgu_common_scales;
    opts_wgu.offsetA = with_wts_gate_zp();

    // opts_wgu.addToC = true; //addToC broken w/wg tile 32?

    //std::cout << "problemStr: " << problem.toString() << std::endl;
    /* Ask microkernel provider for microkernel */
    try {
        gemm_gateup_ = selectGEMMMicrokernel(
                opts_wgu, hw_info, sizes, problem_wgu, reqs_wgu);
    } catch (std::exception &e) {
        VDISPATCH_GATED_MLP(false,
                            "gemm_gateup microkernel generation failed with message: %s",
                            e.what());
    }

    return status::success;
}

status_t micro_gated_mlp_t::init(impl::engine_t *engine) {
    using namespace micro;

    compute::kernel_ctx_t kernel_ctx;

    kernel_ctx.set_data_type(pd()->dst_md()->data_type);

    int ndims = 2;
    const memory_desc_wrapper src_mdw(pd()->src0_md());

    const memory_desc_wrapper W_up_mdw(pd()->W_up_md());
    const memory_desc_wrapper W_gate_mdw(pd()->W_gate_md());
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

    def_data_type(kernel_ctx, src_mdw.data_type(), "SRC");
    def_data_type(kernel_ctx, W_gate_mdw.data_type(), "WTS_GATE");
    def_data_type(kernel_ctx, W_up_mdw.data_type(), "WTS_UP");
    def_data_type(kernel_ctx, W_down_mdw.data_type(), "WTS_DOWN");

    def_data_type(kernel_ctx, pd()->wts_gate_scales_dt(), "WTS_GATE_ATTR_SCALES");
    def_data_type(kernel_ctx, pd()->wts_up_scales_dt(), "WTS_UP_ATTR_SCALES");
    def_data_type(kernel_ctx, pd()->wts_down_scales_dt(), "WTS_DOWN_ATTR_SCALES");

    def_data_type(kernel_ctx, pd()->wts_gate_zp_dt(), "WTS_GATE_ATTR_ZP");
    def_data_type(kernel_ctx, pd()->wts_up_zp_dt(), "WTS_UP_ATTR_ZP");
    def_data_type(kernel_ctx, pd()->wts_down_zp_dt(), "WTS_DOWN_ATTR_ZP");

    auto lds   = gemm_desc_t::get_ld(*pd()->src_md()) * src_mdw.data_type_size();
    auto ldwgu = gemm_desc_t::get_ld(*pd()->W_gate_md()) * W_gate_mdw.data_type_size();
    auto lda   = gemm_desc_t::get_ld(*pd()->dst_md()) * dst_mdw.data_type_size(); //TODO: replace w/tmp mem?

    kernel_ctx.define_int("SRC_ALIGN", jit::alignmentForLD(int(lds)));
    kernel_ctx.define_int("WGU_ALIGN", jit::alignmentForLD(int(ldwgu)));
    kernel_ctx.define_int("DST_ALIGN", jit::alignmentForLD(int(lda)));

    auto *d = pd()->desc();

    int wts_gate_scales_mask = (static_cast<int>(pd()->with_wts_gate_scales()) << 1)
        | static_cast<int>(with_quantize_common(d->wts_gate_scales));
    int wts_up_scales_mask = (static_cast<int>(pd()->with_wts_up_scales()) << 1)
        | static_cast<int>(with_quantize_common(d->wts_up_scales));
    int wts_down_scales_mask = (static_cast<int>(pd()->with_wts_down_scales()) << 1)
        | static_cast<int>(with_quantize_common(d->wts_down_scales));

    kernel_ctx.define_int("WTS_GATE_SCALES", wts_gate_scales_mask);
    kernel_ctx.define_int("WTS_UP_SCALES", wts_up_scales_mask);
    kernel_ctx.define_int("WTS_DOWN_SCALES", wts_down_scales_mask);

    int wts_gate_zp_mask = (static_cast<int>(pd()->with_wts_gate_zp()) << 1)
        | static_cast<int>(with_quantize_common(d->wts_gate_zero_points));
    int wts_up_zp_mask   = (static_cast<int>(pd()->with_wts_up_zp()) << 1)
        | static_cast<int>(with_quantize_common(d->wts_up_zero_points));
    int wts_down_zp_mask = (static_cast<int>(pd()->with_wts_down_zp()) << 1)
        | static_cast<int>(with_quantize_common(d->wts_down_zero_points));

    kernel_ctx.define_int("WTS_GATE_ZERO_POINTS", wts_gate_zp_mask);
    kernel_ctx.define_int("WTS_UP_ZERO_POINTS", wts_up_zp_mask);
    kernel_ctx.define_int("WTS_DOWN_ZERO_POINTS", wts_down_zp_mask);

    using namespace data_type;
    auto elems_per_byte = [](data_type_t dt) {
        switch (dt) {
            case u4:
            case s4: return 2;
            default: return 1;
        }
    };
    kernel_ctx.define_int(
            "WTS_GATE_ELEMENTS_PER_BYTE", elems_per_byte(W_gate_mdw.data_type()));
    kernel_ctx.define_int(
            "WTS_UP_ELEMENTS_PER_BYTE", elems_per_byte(W_up_mdw.data_type()));
    kernel_ctx.define_int(
            "WTS_DOWN_ELEMENTS_PER_BYTE", elems_per_byte(W_down_mdw.data_type()));

    kernel_ctx.define_int(
            "WTS_GATE_ZP_ELEMENTS_PER_BYTE", elems_per_byte(pd()->wts_gate_zp_dt()));
    kernel_ctx.define_int(
            "WTS_UP_ZP_ELEMENTS_PER_BYTE", elems_per_byte(pd()->wts_up_zp_dt()));
    kernel_ctx.define_int(
            "WTS_DOWN_ZP_ELEMENTS_PER_BYTE", elems_per_byte(pd()->wts_down_zp_dt()));

    //TODO: add zp + scale + activation??
    //TODO: ensure zp and scales have identical group size
    if (pd()->with_wts_gate_scales() || pd()->with_wts_gate_zp())
        kernel_ctx.define_int("WTS_GATE_GROUP_SIZE", pd()->wts_gate_group_size());
    if (pd()->with_wts_up_scales() || pd()->with_wts_gate_zp())
        kernel_ctx.define_int("WTS_UP_GROUP_SIZE", pd()->wts_up_group_size());
    if (pd()->with_wts_down_scales() || pd()->with_wts_down_zp())
        kernel_ctx.define_int("WTS_DOWN_GROUP_SIZE", pd()->wts_down_group_size());

    kernel_ctx.define_int("SUBGROUP_SIZE", pd()->sg_size());

    int tile_wgu_m = pd()->gemm_gateup().getSetting("wg_tile_m"); // partition tiles primarily across inner dim - k
    int tile_wgu_n = pd()->gemm_gateup().getSetting("wg_tile_n");

    //TODO:  mb,oc,ic full
    kernel_ctx.define_int("REMAINDER_SRC", d->mb_sz() % tile_wgu_n); //todo: %n instead??
    if (lds % 4 == 0) kernel_ctx.define_int("BLOCK_SRC", 1);
    if (lda % 4 == 0 && (d->oc_sz() % tile_wgu_m) == 0) kernel_ctx.define_int("BLOCK_DST", 1);

    /* Generate microkernel shims */
    ShimOptions shimOptions;
    shimOptions.subgroupSize = pd()->sg_size();
    shimOptions.useTileOps = true;
    shimOptions.decorator = "wgu";

    //TODO: enable shim header
    //std::string header = micro::generateShim(
                    //pd()->gemm_gateup(), HostLanguage::OpenCL_C, shimOptions);
    //std::cout << "shim HEADER" << std::endl;
    //std::cout << header << std::endl;
    kernel_ctx.add_custom_header("gemm_gateup.h", micro::generateShim(
                    pd()->gemm_gateup(), HostLanguage::OpenCL_C, shimOptions));

    if (pd()->gemm_gateup().grfMin > 128) {
        kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");
    }

    CHECK(create_kernel(engine, &fused_mlp_kernel_, "micro_gated_mlp", kernel_ctx));
    if (!fused_mlp_kernel_) return status::runtime_error;

    /*
    bool gemm_ok =
        create_nested_primitive(
                gemm_fc_down_, pd()->gemm_fc_down_pd_, engine) == status::success;
    if (!gemm_ok) return status::runtime_error;
    */

    return status::success;
}

void micro_gated_mlp_t::pd_t::init_scratchpad() {
    auto *d = desc();
    const size_t size = d->mb_sz() * d->oc_sz();
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

    const auto &wts_gate_scales
            = CTX_IN_STORAGE(DNNL_ARG_WTS_GATE | DNNL_ARG_ATTR_SCALES);
    const auto &wts_up_scales
            = CTX_IN_STORAGE(DNNL_ARG_WTS_UP | DNNL_ARG_ATTR_SCALES);
    const auto &wts_down_scales
            = CTX_IN_STORAGE(DNNL_ARG_WTS_DOWN | DNNL_ARG_ATTR_SCALES);

    const auto &wts_gate_zp
            = CTX_IN_STORAGE(DNNL_ARG_WTS_GATE | DNNL_ARG_ATTR_ZERO_POINTS);
    const auto &wts_up_zp
            = CTX_IN_STORAGE(DNNL_ARG_WTS_UP | DNNL_ARG_ATTR_ZERO_POINTS);
    const auto &wts_down_zp
            = CTX_IN_STORAGE(DNNL_ARG_WTS_DOWN | DNNL_ARG_ATTR_ZERO_POINTS);

    const dim_t MB = pd()->desc()->mb_sz();
    const dim_t IC = pd()->desc()->ic_sz();
    const dim_t OC = pd()->desc()->oc_sz();

    auto &gemm_gateup = pd()->gemm_gateup();

    auto wg_tile_OC = gemm_gateup.getSetting("wg_tile_m"); //TODO: which? OC along lda should get largest tile dim?
    //auto wg_tile_OC = gemm_gateup.getSetting("wg_tile_n"); //TODO: which? OC along lda should get largest tile dim?
    auto wg_tile_MB = gemm_gateup.getSetting("wg_tile_n");

    auto sg_per_wg  = gemm_gateup.getSetting("sg_per_wg_m")
                    * gemm_gateup.getSetting("sg_per_wg_n");

    std::unique_ptr<memory_storage_t> tmp_reduce;
    tmp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_gated_mlp_reduction);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0,  src);
    arg_list.set(1,  W_gate);
    arg_list.set(2,  W_up);
    arg_list.set(3,  W_down);
    arg_list.set(4,  dst);
    arg_list.set(5,  MB);
    arg_list.set(6,  IC);
    arg_list.set(7,  OC);
    arg_list.set(8,  *tmp_reduce);
    arg_list.set(9,  wts_gate_scales);
    arg_list.set(10, wts_gate_zp);
    arg_list.set(11, wts_up_scales);
    arg_list.set(12, wts_up_zp);
    arg_list.set(13, wts_down_scales);
    arg_list.set(14, wts_down_zp);

    compute::range_t lws = {(size_t)pd()->sg_size(), (size_t)sg_per_wg, 1};
    compute::range_t gws = lws;

    gws[0] *= utils::div_up(OC, wg_tile_OC);
    gws[2] *= utils::div_up(MB, wg_tile_MB);

    //size_t k_split = IC / gemm_gateup.getSetting("wg_tile_m");
    //gws[1] *= k_split;

    auto nd_range = compute::nd_range_t(gws, lws);
    //printf("gws[%zu %zu %zu] lws[%zu %zu %zu] \n",gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
    status_t gateup_status = parallel_for(ctx, nd_range, fused_mlp_kernel_, arg_list);

    if (gateup_status != status::success) return gateup_status;

    // perform fc_down gemm
    using namespace memory_tracking::names;

    gemm_exec_args_t gemm_args;
    //gemm_args.a = tmp_reduce.get();
    gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_SRC);
    //gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_WTS_SRC);
    gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_WTS_GATE);
    gemm_args.c = &CTX_OUT_STORAGE(DNNL_ARG_DST);

    /*
     * TODO: zp + scales for fc_down
    memory_storage_t *a0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    memory_storage_t *b0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    memory_storage_t *c0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    gemm_args.a_zero_point = b0;
    gemm_args.b_zero_point = a0;
    gemm_args.c_zero_point = c0;
    gemm_args.a_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    gemm_args.b_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    gemm_args.c_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    *
    */

    //TODO: reenable mm
    //gemm_args.exec_args = ctx.args();
    //gemm_exec_ctx_t gemm_ctx(ctx, gemm_args);

    //TODO: reenable mm
    //nested_scratchpad_t ns(ctx, memory_tracking::names::key_nested, gemm_fc_down_);
    //gemm_ctx.set_scratchpad_grantor(ns.grantor());

    //TODO: reenable mm
    //status_t gemm_exec_status = gpu_gemm(gemm_fc_down_)->execute(gemm_ctx);
    //if (gemm_exec_status != status::success) { return gemm_exec_status; }

    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
