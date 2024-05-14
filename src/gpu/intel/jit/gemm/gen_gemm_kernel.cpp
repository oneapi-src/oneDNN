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

#include "gpu/intel/jit/gemm/gen_gemm_kernel.hpp"
#include "common/impl_registration.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/jit/gemm/kernel_catalog.hpp"
#include "gpu/intel/jit/gemm/kernel_selector.hpp"
#include "gpu/intel/jit/gemm/strategy_parser.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

#define _CATALOG_ gemm_catalog
#include "gpu/intel/jit/gemm/kernel.db"
;
#undef _CATALOG_

status_t gen_gemm_kernel_desc_t::create_generator(
        const compute::compute_engine_t &engine,
        compute::kernel_t &kernel) const {
    gen_gemm_kernel_t kd(*this);
    return engine.create_kernel(&kernel, &kd, cache_blob_t());
}

status_t gen_gemm_kernel_desc_t::finalize(const char *tags) {
    // Update problem alignments to match catalog entry.
    if (!isPacked(problem_.A.layout)) {
        problem_.A.setAlignment(
                std::max(problem_.Ta.isInt4() ? 1 : problem_.Ta.size(),
                        entry_->driverInfo.alignment[0]));
    }

    if (!isPacked(problem_.B.layout)) {
        problem_.B.setAlignment(
                std::max(problem_.Ta.isInt4() ? 1 : problem_.Ta.size(),
                        entry_->driverInfo.alignment[1]));
    }

    if (!isPacked(problem_.C.layout)) {
        problem_.C.setAlignment(std::max(
                problem_.Tc_ext.size(), entry_->restrictions.alignment[2]));
    }

    problem_.CO.setAlignment(problem_.Tco.size());

    // Parse strategy string.
    strategy_ = GEMMStrategy(hw_, stepping_);
    strategy_.unroll[LoopM] = entry_->driverInfo.unroll[LoopM];
    strategy_.unroll[LoopN] = entry_->driverInfo.unroll[LoopN];
    parseStrategy(entry_->strategy, hw_, problem_, strategy_);
    strategy_.panelCheck
            |= (isPacked(problem_.A.layout) || isPacked(problem_.B.layout));
    adjustStrategy(hw_, problem_, strategy_, tags);
    modifyStrategy(strategy_, aux_params_);

    if (hw_ == ngen::HW::Xe2) {
        // Temporary hack to use XeHPC register banking on Xe2, in order
        //   to successfully reuse XeHPC strategies.
        strategy_.raHW = ngen::HW::XeHPC;

        // Bump up alignments to 16 bytes for block 2D if available.
        bool block_2d_a = false, block_2d_b = false;
        for (auto c = tags; *c; c++) {
            block_2d_a |= (*c == kcatalog::ReqBlock2DA);
            block_2d_b |= (*c == kcatalog::ReqBlock2DB);
        }
        if (block_2d_a && strategy_.legalAAlignment(problem_, 16))
            problem_.A.setAlignment(nstl::max<int>(problem_.A.alignment, 16));
        if (block_2d_b && strategy_.legalBAlignment(problem_, 16))
            problem_.B.setAlignment(nstl::max<int>(problem_.B.alignment, 16));
    }

    // Disable global k parallelization if it wouldn't be used.
    if (strategy_.kParallel && k_ >= 0) {
        auto k_min = aux_params_.k0 * aux_params_.wgK;
        if (k_ <= k_min) {
            strategy_.kParallel = false;
            strategy_.C.atomic = false;
            strategy_.CO.atomic = false;
        }
    }

    // Always use variable beta for global k-parallel kernels.
    if (strategy_.kParallel && !strategy_.fuseBeta) problem_.beta = Scalar();

    // Omit periodic barriers when k is small.
    if (strategy_.barrierFreq > 0 && k_ >= 0 && k_ < 2 * strategy_.barrierFreq)
        strategy_.barrierFreq = 0;

    // Disable linear ordering and persistent threads if the GEMM doesn't fill the GPU.
    if (m_ >= 0 && n_ >= 0 && eu_count_ >= 0) {
        int wg_tile_m = strategy_.wg[LoopM] * strategy_.unroll[LoopM];
        int wg_tile_n = strategy_.wg[LoopN] * strategy_.unroll[LoopN];
        if (wg_tile_m > 0 && wg_tile_n > 0) {
            dim_t m_tiles = dim_t(utils::div_up(m_, wg_tile_m));
            dim_t n_tiles = dim_t(utils::div_up(n_, wg_tile_n));
            dim_t thread_per_tg = strategy_.wg[LoopM] * strategy_.wg[LoopN];
            if (!strategy_.kParallelVariable)
                thread_per_tg *= std::max(strategy_.wg[LoopK], 1);
            dim_t thread_gpu = eu_count_
                    * compute::device_info_t::threads_per_eu(
                            arch_, strategy_.GRFs > 128);
            dim_t tiles_gpu = thread_gpu / thread_per_tg;

            bool use_linear = (m_tiles * n_tiles <= tiles_gpu);
            bool use_linear_m = (m_tiles * m_tiles <= 2 * tiles_gpu);
            bool use_linear_n = (n_tiles * n_tiles <= 2 * tiles_gpu);

            if (strategy_.fused)
                if (strategy_.wg[LoopM] % 2 || strategy_.wg[LoopN] % 2)
                    use_linear_m = use_linear_n = false; /* cannot swap */

            if (use_linear) {
                if (strategy_.kParallelVariable)
                    strategy_.cWalkOrder = WalkOrder::SimpleLinear;
                else if (strategy_.kParallel
                        && (strategy_.fuseBeta || strategy_.fusePostOps)) {
                    strategy_.persistent = false;
                    strategy_.cWalkOrder = WalkOrder::SimpleLinear;
                } else {
                    strategy_.persistent = false;
                    strategy_.cWalkOrder = WalkOrder::HW2D;
                    strategy_.blocking[LoopM] = 16777216;
                    strategy_.blocking[LoopN] = 16777216;
                }
            } else if (use_linear_m || use_linear_n) {
                if (use_linear_n && !use_linear_m) {
                    strategy_.loopOrder[0] = LoopN;
                    strategy_.loopOrder[1] = LoopM;
                } else if (use_linear_m && !use_linear_n) {
                    strategy_.loopOrder[0] = LoopM;
                    strategy_.loopOrder[1] = LoopN;
                }
                strategy_.cWalkOrder = WalkOrder::SimpleLinear;
            }
        }
    }

    strategy_.systolicAvailable &= !disable_systolic_;
    strategy_.preflight(hw_, problem_);

    // Check for legal 2D quantization group size.
    if (problem_.aoPtrDims == 2 || problem_.aScale2D)
        if (problem_.aqGroupK % strategy_.aqGroupKGranularity())
            return status::unimplemented;
    if (problem_.boPtrDims == 2 || problem_.bScale2D)
        if (problem_.bqGroupK % strategy_.bqGroupKGranularity())
            return status::unimplemented;

    update_driver_info();

    return status::success;
}

void gen_gemm_kernel_desc_t::update_driver_info() {
#define ARCH_DISPATCH(arch) \
    case ngen::HW::arch: \
        driver_info_ = gemm_kernel_generator_t<ngen::HW::arch>::driverInfo( \
                problem_, strategy_); \
        break;

    switch (hw_) {
        REG_GEN9_ISA(ARCH_DISPATCH(Gen9))
        REG_GEN11_ISA(ARCH_DISPATCH(Gen11))
        REG_XELP_ISA(ARCH_DISPATCH(XeLP))
        REG_XEHP_ISA(ARCH_DISPATCH(XeHP))
        REG_XEHPG_ISA(ARCH_DISPATCH(XeHPG))
        REG_XEHPC_ISA(ARCH_DISPATCH(XeHPC))
        REG_XE2_ISA(ARCH_DISPATCH(Xe2))
        default:
            assert(!"Unsupported architecture");
            driver_info_ = entry_->driverInfo;
            break;
    }
#undef ARCH_DISPATCH
}

status_t gen_gemm_kernel_desc_t::transfer_post_ops(
        gpu_post_ops_t &&post_ops_, bool swap_ab) {
    problem_.postOps = std::move(post_ops_);
    const auto &post_ops = problem_.postOps;

    if (post_ops.len() > 0) {

        size_t po_count = post_ops.len();
        problem_.Tbinary.reserve(po_count);
        problem_.binary.reserve(po_count);
        problem_.binaryRow = {};
        problem_.binaryCol = {};
        problem_.binaryBatch = {};
        problem_.binaryTrans = {};

        if (problem_.Ta == Type::f16) problem_.Ts = Type::f32;
        if (problem_.Ta.isF8() || problem_.Tb.isF8()) problem_.Ts = Type::f32;

        for (size_t i = 0; i < po_count; i++) {
            const auto &entry = post_ops[i];
            if (!entry.is_binary()) {
                problem_.Tbinary.push_back(Type::invalid);
                problem_.binary.push_back(MatrixAddressing {});
                continue;
            }

            auto &src_rmd = entry.as_binary().src1_desc;

            auto T = convert_dnnl_to_kernel_type(src_rmd.dt);
            bool is_multi_row = (src_rmd.broadcast_mask & 1) == 0;
            bool is_multi_col = (src_rmd.broadcast_mask & 2) == 0;

            bool is_compatible = src_rmd.inner_layout.empty();
            if (!is_compatible) return status::unimplemented;

            bool trans = is_multi_row && !src_rmd.inner_dim.is_innermost();

            if (swap_ab) {
                trans = !trans;
                std::swap(is_multi_row, is_multi_col);
            }

            problem_.Tbinary.push_back(T);
            problem_.binaryRow[i] = is_multi_row;
            problem_.binaryCol[i] = is_multi_col;
            problem_.binaryBatch[i] = src_rmd.ndims() >= 3;
            problem_.binaryTrans[i] = trans;

            MatrixAddressing atype;
            atype.layout = trans ? MatrixLayout::T : MatrixLayout::N;
            atype.crosspack = 1;
            atype.packSize = 0;
            atype.setAlignment(T.size());

            problem_.binary.push_back(atype);
        }
    }

    return status::success;
}

status_t gen_gemm_nocopy_kernel_desc_t::select_kernel(compute::gpu_arch_t arch,
        int stepping, int eu_count, bool has_systolic, compute_mode mode,
        int batch_dims, bool trans_a, bool trans_b, bool trans_co, bool swap_ab,
        int ao_dims, int bo_dims, bool wei_scale_2d, int wei_q2d_group_k,
        bool c_offset, bool bias, sum_ab_t reduce_ab, float alpha, float beta,
        data_type_t a_type, data_type_t b_type, data_type_t c_type,
        data_type_t ao_type, data_type_t bo_type, data_type_t wei_scales_type,
        data_type_t co_type, data_type_t acc_type, int align_a, int align_b,
        int align_c, dim_t m, dim_t n, dim_t k, dim_t lda, dim_t ldb, dim_t ldc,
        dim_t batch, gpu_post_ops_t &&post_ops) {
    using namespace ngen;
    using namespace kcatalog;

    arch_ = arch;
    hw_ = convert_dnnl_arch_to_ngen(arch);
    stepping_ = stepping;
    m_ = m;
    n_ = n;
    k_ = k;
    eu_count_ = eu_count;
    disable_systolic_ = !has_systolic;

    align_a = nstl::max(align_a, int(types::data_type_size(a_type)));
    align_b = nstl::max(align_b, int(types::data_type_size(b_type)));
    align_c = nstl::max(align_c, int(types::data_type_size(c_type)));

    bool can_2d_a = (lda * problem_.Ta <= 16777216);
    bool can_2d_b = (ldb * problem_.Tb <= 16777216);
    bool can_2d_c = (ldc * problem_.Tc <= 16777216);

    // Xe2 requires stronger alignment for block 2D.
    if (arch == compute::gpu_arch_t::xe2) {
        can_2d_a &= (align_a % 16 == 0);
        can_2d_b &= (align_b % 16 == 0);
        can_2d_c &= (align_c % 16 == 0);
    }

    // Set up problem structure.
    problem_.Ta = problem_.Ta_ext = convert_dnnl_to_kernel_type(a_type);
    problem_.Tb = problem_.Tb_ext = convert_dnnl_to_kernel_type(b_type);
    problem_.Tc = convert_dnnl_to_kernel_type(acc_type);
    problem_.Tc_ext = convert_dnnl_to_kernel_type(c_type);
    problem_.Ts = problem_.Tc;
    problem_.Tao = convert_dnnl_to_kernel_type(ao_type);
    problem_.Tbo = convert_dnnl_to_kernel_type(bo_type);
    problem_.Tco = convert_dnnl_to_kernel_type(co_type);
    problem_.A.layout = trans_a ? MatrixLayout::T : MatrixLayout::N;
    problem_.B.layout = trans_b ? MatrixLayout::T : MatrixLayout::N;
    problem_.C.layout = MatrixLayout::N;
    problem_.A.crosspack = problem_.B.crosspack = problem_.C.crosspack = 1;
    problem_.A.packSize = problem_.B.packSize = problem_.C.packSize = 0;
    problem_.A.setAlignment(align_a);
    problem_.B.setAlignment(align_b);
    problem_.C.setAlignment(align_c);
    if (batch_dims > 0) {
        problem_.batch = BatchMode::Strided;
        problem_.batchDims = batch_dims;
    }
    if (ao_dims >= 0) problem_.aOffset = ABOffset::Calc;
    if (bo_dims >= 0) problem_.bOffset = ABOffset::Calc;
    problem_.aoPtrDims = ao_dims;
    problem_.boPtrDims = bo_dims;
    problem_.AO.layout = MatrixLayout::N;
    problem_.BO.layout = MatrixLayout::T;
    problem_.AO.crosspack = problem_.BO.crosspack = 1;
    problem_.AO.packSize = problem_.BO.packSize = 0;
    problem_.A_scale = problem_.AO;
    problem_.B_scale = problem_.BO;
    if (ao_type != data_type::undef)
        problem_.AO.setAlignment(int(types::data_type_size(ao_type)));
    if (bo_type != data_type::undef)
        problem_.BO.setAlignment(int(types::data_type_size(bo_type)));
    if (!swap_ab) {
        problem_.aScale2D = wei_scale_2d;
        problem_.aqGroupK = wei_q2d_group_k;
        if (wei_scales_type != data_type::undef) {
            problem_.Ta_scale = convert_dnnl_to_kernel_type(wei_scales_type);
            problem_.A_scale.setAlignment(
                    int(types::data_type_size(wei_scales_type)));
        }
    } else {
        problem_.bScale2D = wei_scale_2d;
        problem_.bqGroupK = wei_q2d_group_k;
        if (wei_scales_type != data_type::undef) {
            problem_.Tb_scale = convert_dnnl_to_kernel_type(wei_scales_type);
            problem_.B_scale.setAlignment(
                    int(types::data_type_size(wei_scales_type)));
        }
    }

    if (problem_.Ta.isInteger()) problem_.Ts = Type::f32;

    if (alpha == 1.0f) problem_.alpha = alpha;
    if (beta == 0.0f || beta == 1.0f) problem_.beta = beta;

    auto status = transfer_post_ops(std::move(post_ops), swap_ab);
    if (status != status::success) return status;

    if (c_offset || bias || reduce_ab != sum_ab::sum_none) {
        assert(!(c_offset && bias));
        if (bias) problem_.cOffset = COffset::Pre;
        if (c_offset) problem_.cOffset = COffset::Post;
        problem_.CO.crosspack = 1;
        problem_.CO.alignment = problem_.C.alignment;
        problem_.CO.layout = trans_co ? MatrixLayout::T : MatrixLayout::N;
    }

    problem_.sumA = (reduce_ab == sum_ab::sum_b_col);
    problem_.sumB = (reduce_ab == sum_ab::sum_a_row);

    // Select a kernel from the catalog.
    MatchParams match_params[3];
    int npatterns = 1;

    match_params[0] = MatchParams(hw_, problem_);

    match_params[0].sizes.m = m;
    match_params[0].sizes.n = n;
    match_params[0].sizes.k = k;
    match_params[0].sizes.batch = batch;
    match_params[0].stepping = stepping;

    auto tags = const_cast<char *>(match_params[0].tags);
    while (*tags)
        tags++;
    if (can_2d_a) *tags++ = kcatalog::ReqBlock2DA;
    if (can_2d_b) *tags++ = kcatalog::ReqBlock2DB;
    if (can_2d_c) *tags++ = kcatalog::ReqBlock2DC;
    if (has_systolic) *tags++ = kcatalog::ReqSystolic;

    if ((mode & mode_tf32)
            && utils::everyone_is(Type::f32, problem_.Ta, problem_.Tb)) {
        match_params[npatterns] = match_params[0];
        match_params[npatterns].selector.precisions[0] = "T";
        match_params[npatterns].selector.precisions[1] = "T";
        npatterns++;
    }

    if ((mode & mode_bf16x1)
            && utils::everyone_is(Type::f32, problem_.Ta, problem_.Tb)) {
        match_params[npatterns] = match_params[0];
        match_params[npatterns].selector.precisions[0] = "[SB]";
        match_params[npatterns].selector.precisions[1] = "[SB]";
        npatterns++;
    }

    EvaluateParams eval_params;

    eval_params.sizes = match_params[0].sizes;
    eval_params.alpha = alpha;
    eval_params.beta = beta;
    eval_params.postOps = !problem_.postOps.empty();
    eval_params.cConvert = (acc_type != c_type);
    eval_params.euCount = eu_count;
    eval_params.batch = (batch_dims > 0);
    eval_params.deterministic = (mode & mode_deterministic);

    entry_ = select(
            gemm_catalog, npatterns, match_params, eval_params, aux_params_);

    if (!entry_) return status::unimplemented;

    if (mode & mode_tf32) {
        if (entry_->selector.precisions[0][0] == 'T')
            problem_.Ta = problem_.Ta_ext = Type::tf32;
        if (entry_->selector.precisions[1][0] == 'T')
            problem_.Tb = problem_.Tb_ext = Type::tf32;
    }

    if (mode & mode_bf16x1) {
        if (entry_->selector.precisions[0][0] == '[') problem_.Ta = Type::bf16;
        if (entry_->selector.precisions[1][0] == '[') problem_.Tb = Type::bf16;
    }

    auto block_k = entry_->driverInfo.blocking[LoopK];
    if (block_k > 0 && k > block_k && beta != 1.0f) problem_.beta = Scalar();

    return finalize(match_params[0].tags);
}

status_t gen_gemm_xe_systolic_kernel_desc_t::select_kernel(
        compute::gpu_arch_t arch, int stepping, int eu_count, int batch_dims,
        bool packed_c, bool trans_co, bool a_offset, bool b_offset,
        bool c_offset, bool bias, float alpha, float beta, data_type_t a_type,
        data_type_t b_type, data_type_t c_type, data_type_t ao_type,
        data_type_t bo_type, data_type_t co_type, data_type_t acc_type, dim_t m,
        dim_t n, dim_t k, dim_t batch, int unroll_m, int unroll_n, bool alt,
        gpu_post_ops_t &&post_ops) {
    using namespace ngen;
    using namespace kcatalog;

    arch_ = arch;
    hw_ = convert_dnnl_arch_to_ngen(arch);
    stepping_ = stepping;
    m_ = m;
    n_ = n;
    k_ = k;
    eu_count_ = eu_count;

    if (hw_ != HW::Xe2)
        if (!utils::one_of(hw_, HW::XeHP, HW::XeHPG, HW::XeHPC))
            return status::unimplemented;

    bool xehpc = (hw_ >= HW::XeHPC);

    auto osys = xehpc ? 16 : 8;
    auto ksys = int(32 / types::data_type_size(a_type));
    auto csys = int(4 / types::data_type_size(a_type));

    problem_.Ta = problem_.Ta_ext = convert_dnnl_to_kernel_type(a_type);
    problem_.Tb = problem_.Tb_ext = convert_dnnl_to_kernel_type(b_type);
    problem_.Tc = convert_dnnl_to_kernel_type(acc_type);
    problem_.Tc_ext = convert_dnnl_to_kernel_type(c_type);
    problem_.Ts = Type::f32;
    problem_.Tao = convert_dnnl_to_kernel_type(ao_type);
    problem_.Tbo = convert_dnnl_to_kernel_type(bo_type);
    problem_.Tco = convert_dnnl_to_kernel_type(co_type);
    problem_.A.layout = MatrixLayout::PackedColumns;
    problem_.B.layout = MatrixLayout::PackedRows;
    problem_.C.layout = MatrixLayout::N;
    problem_.A.crosspack = csys;
    problem_.B.crosspack = ksys;
    problem_.C.crosspack = 1;
    problem_.A.packSize = unroll_m;
    problem_.B.packSize = unroll_n;
    problem_.C.packSize = 0;
    if (osys < unroll_m) {
        problem_.A.tileR = osys;
        problem_.A.tileC = ksys;
    }
    problem_.A.setAlignment(32);
    problem_.B.setAlignment(32);
    problem_.C.setAlignment(int(types::data_type_size(c_type)));
    if (packed_c) problem_.C = problem_.B;
    if (batch_dims > 0) {
        problem_.batch = BatchMode::Strided;
        problem_.batchDims = batch_dims;
    }
    if (a_offset) {
        problem_.aOffset = ABOffset::Load;
        problem_.aoPtrDims = 0;
    }
    if (b_offset) {
        problem_.bOffset = ABOffset::Load;
        problem_.boPtrDims = 0;
    }
    if (alpha == 1.0f) problem_.alpha = alpha;
    if (beta == 0.0f || beta == 1.0f) problem_.beta = beta;

    auto status = transfer_post_ops(std::move(post_ops), false);
    if (status != status::success) return status;

    if (c_offset) problem_.cOffset = COffset::Post;

    if (bias) {
        if (problem_.cOffset != COffset::None) return status::unimplemented;
        problem_.cOffset = COffset::Pre;
        problem_.CO.layout = trans_co ? MatrixLayout::T : MatrixLayout::N;
    }

    if (problem_.cOffset != COffset::None) {
        problem_.CO.crosspack = 1;
        problem_.CO.alignment = problem_.C.alignment;
    }

    // Find it in the catalog.
    MatchParams match_params(hw_, problem_);

    match_params.sizes.m = m;
    match_params.sizes.n = n;
    match_params.sizes.k = k;
    match_params.sizes.batch = batch;
    match_params.unroll[LoopM] = unroll_m;
    match_params.unroll[LoopN] = unroll_n;

    auto tags = const_cast<char *>(match_params.tags);
    while (*tags)
        tags++;

    *tags++ = kcatalog::ReqSystolic;
    if (alt) *tags++ = kcatalog::ReqCustom1;

    EvaluateParams eval_params;

    eval_params.sizes = match_params.sizes;
    eval_params.alpha = alpha;
    eval_params.beta = beta;
    eval_params.euCount = eu_count;
    eval_params.postOps = !problem_.postOps.empty();
    eval_params.cConvert = (acc_type != c_type);
    eval_params.batch = (batch_dims > 0);

    entry_ = select(gemm_catalog, match_params, eval_params, aux_params_);

    if (!entry_) return status::unimplemented;

    return finalize(match_params.tags);
}

void gen_gemm_xe_systolic_kernel_desc_t::choose_unrolls(
        compute::gpu_arch_t arch, int eu_count, data_type_t a_type,
        data_type_t b_type, data_type_t c_type, dim_t m, dim_t n, dim_t k,
        dim_t batch, int &unroll_m, int &unroll_n, bool &alt) {

    using namespace data_type;

    alt = false;

    switch (arch) {
        case compute::gpu_arch_t::xe_hp:
        case compute::gpu_arch_t::xe_hpg:
            if (unroll_m == 0) unroll_m = 32;
            if (unroll_n == 0) unroll_n = (m * n >= 6144 * eu_count) ? 48 : 32;

            if (unroll_n == 48) alt = (m * n >= 13824 * eu_count);
            break;
        case compute::gpu_arch_t::xe_hpc:
        case compute::gpu_arch_t::xe2:
            if (utils::one_of(a_type, f16, bf16)) {
                if (unroll_m != 0)
                    unroll_n = (unroll_m > 16) ? 32 : 16;
                else if (unroll_n != 0)
                    unroll_m = (unroll_n > 16) ? 64 : 16;
                else if (m * n < 4096 * eu_count)
                    unroll_m = unroll_n = 16;
                else {
                    unroll_m = 64;
                    unroll_n = 32;
                }
            } else {
                unroll_m = 64;
                unroll_n = 32;
            }
            break;
        default: assert(!"Unsupported architecture.");
    }
}

void gen_gemm_kernel_t::init_interface() {
    using namespace ngen;

    auto &problem = *desc()->problem();
    auto &strategy = *desc()->strategy();

    interface_ = NEOInterfaceHandler {desc()->hw_};
    auto s_type_ngen = problem.Ts.ngen();

    interface_.newArgument("A", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("B", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("C", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("offset_A", DataType::q);
    interface_.newArgument("offset_B", DataType::q);
    interface_.newArgument("offset_C", DataType::q);
    interface_.newArgument("lda", DataType::d);
    interface_.newArgument("ldb", DataType::d);
    interface_.newArgument("ldc", DataType::d);
    interface_.newArgument("m", DataType::d);
    interface_.newArgument("n", DataType::d);
    interface_.newArgument("k", DataType::d);
    interface_.newArgument("alpha_real", s_type_ngen);
    interface_.newArgument("beta_real", s_type_ngen);
    if (problem.aoPtrDims >= 0)
        interface_.newArgument("ao_ptr", ExternalArgumentType::GlobalPtr);
    if (problem.boPtrDims >= 0)
        interface_.newArgument("bo_ptr", ExternalArgumentType::GlobalPtr);
    if (problem.aScale2D)
        interface_.newArgument("a_scale_ptr", ExternalArgumentType::GlobalPtr);
    if (problem.bScale2D)
        interface_.newArgument("b_scale_ptr", ExternalArgumentType::GlobalPtr);
    if (problem.aoPtrDims == 2 || problem.aScale2D)
        interface_.newArgument("ldaq", DataType::d);
    if (problem.boPtrDims == 2 || problem.bScale2D)
        interface_.newArgument("ldbq", DataType::d);
    if (problem.cOffset != COffset::None || problem.sumA || problem.sumB) {
        interface_.newArgument("CO", ExternalArgumentType::GlobalPtr);
        interface_.newArgument("offset_CO", DataType::d);
        if (problem.cOffset == COffset::Pre)
            interface_.newArgument("ldco", DataType::d);
    }

    if (strategy.needsTempC(problem))
        interface_.newArgument("temp_C", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("flags", DataType::ud);
    if ((strategy.kParallel || strategy.kParallelLocal)
            && !strategy.kParallelVariable)
        interface_.newArgument("k0", DataType::d);
    for (size_t i = 0; i < problem.postOps.len(); i++) {
        if (!problem.postOps[i].is_binary()) continue;
        auto bname = "binary" + std::to_string(i);
        interface_.newArgument(bname, ExternalArgumentType::GlobalPtr);
        interface_.newArgument("offset_" + bname, DataType::d);
        if (problem.binaryRow[i] && problem.binaryCol[i])
            interface_.newArgument("ld" + bname, DataType::d);
    }
    if (problem.batch == BatchMode::Strided) {
        if (problem.batchDims > 1) {
            interface_.newArgument("stride_A1", DataType::d);
            interface_.newArgument("stride_B1", DataType::d);
            interface_.newArgument("stride_C1", DataType::d);
            for (size_t i = 0; i < problem.postOps.len(); i++)
                if (problem.postOps[i].is_binary() && problem.binaryBatch[i])
                    interface_.newArgument(
                            "stride1_binary" + std::to_string(i), DataType::d);
        }
        interface_.newArgument("stride_A", DataType::d);
        interface_.newArgument("stride_B", DataType::d);
        interface_.newArgument("stride_C", DataType::d);
        for (size_t i = 0; i < problem.postOps.len(); i++)
            if (problem.postOps[i].is_binary() && problem.binaryBatch[i])
                interface_.newArgument(
                        "stride_binary" + std::to_string(i), DataType::d);
        if (problem.batchDims > 1) {
            interface_.newArgument("batch_size1", DataType::ud);
            interface_.newArgument("recip_batch_size1", DataType::ud);
        }
    }
    if (strategy.fuseBeta || strategy.fusePostOps)
        interface_.newArgument("status", ExternalArgumentType::GlobalPtr,
                GlobalAccessType::Stateless);
    if (strategy.fuseBeta && strategy.kParallel)
        interface_.newArgument("group_count_k", DataType::ud);
    if (strategy.linearOrder()) {
        interface_.newArgument("group_count_m", DataType::ud);
        interface_.newArgument("group_count_n", DataType::ud);
    }
    if (strategy.cWalkOrder == WalkOrder::SimpleLinear)
        interface_.newArgument("group_count_recip", DataType::ud);
    else if (strategy.cWalkOrder == WalkOrder::Hilbertlike) {
        interface_.newArgument("hilbert_vd", DataType::ud);
        interface_.newArgument("hilbert_uvd_recip", DataType::ud);
        interface_.newArgument("hilbert_bail", DataType::ud);
    } else if (strategy.cWalkOrder == WalkOrder::Boustrophedon) {
        interface_.newArgument("bslice", DataType::d);
        interface_.newArgument("bthresh", DataType::d);
    }
    if (strategy.kParallelVariable) {
        interface_.newArgument("k0", DataType::ud);
        interface_.newArgument("k_parallel_start", DataType::ud);
        interface_.newArgument("k_recip", DataType::ud);
        if (strategy.fuseBeta) interface_.newArgument("k0_recip", DataType::ud);
    }
    if (strategy.persistent)
        interface_.newArgument("group_stride", DataType::ud);
    if (strategy.variableSLM())
        interface_.newArgument("local_mem", ExternalArgumentType::LocalPtr);
    if (problem.aoPtrDims >= 1 || problem.aScale2D)
        interface_.newArgument("offset_Aq", DataType::d);
    if (problem.boPtrDims >= 1 || problem.bScale2D)
        interface_.newArgument("offset_Bq", DataType::d);

    if (desc()->hw_ >= HW::XeHPG) interface_.allowArgumentRearrangement(false);
    interface_.externalName(kernel_name());
}

xpu::binary_t gen_gemm_kernel_t::get_binary(
        cl_context context, cl_device_id device) {
    init_interface();

#define ARCH_DISPATCH(arch) \
    case ngen::HW::arch: { \
        gemm_kernel_generator_t<ngen::HW::arch> generator; \
        generator.setStepping(desc()->stepping_); \
        generator.gemm(*desc()->problem(), *desc()->strategy(), interface_); \
        return generator.getBinary(context, device); \
        break; \
    }

    try {
        switch (desc()->hw_) {
            REG_GEN9_ISA(ARCH_DISPATCH(Gen9))
            REG_GEN11_ISA(ARCH_DISPATCH(Gen11))
            REG_XELP_ISA(ARCH_DISPATCH(XeLP))
            REG_XEHP_ISA(ARCH_DISPATCH(XeHP))
            REG_XEHPG_ISA(ARCH_DISPATCH(XeHPG))
            REG_XEHPC_ISA(ARCH_DISPATCH(XeHPC))
            REG_XE2_ISA(ARCH_DISPATCH(Xe2))
            default: assert(!"Unsupported architecture"); break;
        }
    } catch (...) {}

    return {};

#undef ARCH_DISPATCH
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
