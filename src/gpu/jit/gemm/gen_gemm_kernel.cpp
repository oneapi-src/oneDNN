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

#include <cctype>

#include "common/gemm_utils.hpp"
#include "common/impl_registration.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel.hpp"
#include "gpu/jit/gemm/kernel_catalog.hpp"
#include "gpu/jit/gemm/kernel_selector.hpp"
#include "gpu/jit/gemm/strategy_parser.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

#define _CATALOG_ gemm_catalog
#include "gpu/jit/gemm/kernel.db"
;
#undef _CATALOG_

status_t gen_gemm_kernel_desc_t::create_generator(
        engine_t *engine, compute::compiled_kernel_t &generator) const {
    gen_gemm_kernel_t kd(*this);
    return compute::compiled_kernel_t::create(generator, engine, kd);
}

status_t gen_gemm_kernel_desc_t::finalize() {
    // Update problem alignments to match catalog entry.
    if (!isPacked(problem_.A.layout)) {
        problem_.A.setAlignment(
                std::max(problem_.Ta.size(), entry_->driverInfo.alignment[0]));
    }

    if (!isPacked(problem_.B.layout)) {
        problem_.B.setAlignment(
                std::max(problem_.Tb.size(), entry_->driverInfo.alignment[1]));
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
    adjustStrategy(hw_, problem_, strategy_);
    modifyStrategy(strategy_, aux_params_);

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
            dim_t thread_count = dim_t(utils::div_up(m_, wg_tile_m))
                    * utils::div_up(n_, wg_tile_n) * strategy_.wg[LoopM]
                    * strategy_.wg[LoopN];
            if (!strategy_.kParallelVariable)
                thread_count *= std::max(strategy_.wg[LoopK], 1);
            dim_t thread_gpu = eu_count_
                    * compute::device_info_t::threads_per_eu(
                            arch_, strategy_.GRFs > 128);
            if (thread_count <= thread_gpu) {
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
            }
        }
    }

    strategy_.systolicAvailable &= !disable_systolic_;
    strategy_.preflight(hw_, problem_);

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
        default:
            assert(!"Unsupported architecture");
            driver_info_ = entry_->driverInfo;
            break;
    }
#undef ARCH_DISPATCH
}

status_t gen_gemm_kernel_desc_t::transfer_post_ops(
        const post_ops_t &post_ops, bool swap_ab) {
    if (post_ops.len() > 0) {
        problem_.postOps = post_ops;
        int po_count = post_ops.len();
        problem_.Tbinary.reserve(po_count);
        problem_.binary.reserve(po_count);
        problem_.binaryRow.reserve(po_count);
        problem_.binaryCol.reserve(po_count);
        problem_.binaryBatch.reserve(po_count);

        if (problem_.Ta == Type::f16) problem_.Ts = Type::f32;

        for (int i = 0; i < po_count; i++) {
            const auto &entry = post_ops.entry_[i];
            if (entry.kind != primitive_kind::binary) {
                problem_.Tbinary.push_back(Type::invalid);
                problem_.binaryRow.push_back(false);
                problem_.binaryCol.push_back(false);
                problem_.binaryBatch.push_back(false);
                problem_.binary.push_back(MatrixAddressing {});
                continue;
            }

            const auto &src_md = entry.binary.src1_desc;
            memory_desc_wrapper src_mdw(src_md);

            int ndims = src_mdw.ndims();
            auto T = convert_dnnl_to_kernel_type(src_mdw.data_type());
            int nr = (ndims >= 1) ? src_mdw.dims()[ndims - 1] : 1;
            int nc = (ndims >= 2) ? src_mdw.dims()[ndims - 2] : 1;
            bool trans = false;

            if (src_mdw.ndims() >= 2) {
                if (src_md.format_kind != format_kind::blocked
                        || !is_md_gemm_compatible_plain_format(&src_md, false))
                    return status::unimplemented;
                trans = (src_md.format_desc.blocking.strides[ndims - 1] > 1);
            }

            if (swap_ab) {
                trans = !trans;
                std::swap(nr, nc);
            }

            problem_.Tbinary.push_back(T);
            problem_.binaryRow.push_back(nr > 1);
            problem_.binaryCol.push_back(nc > 1);
            problem_.binaryBatch.push_back(ndims >= 3);

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
        bool a_offset, bool b_offset, bool c_offset, bool bias,
        sum_ab_t reduce_ab, float alpha, float beta, const post_ops_t &post_ops,
        data_type_t a_type, data_type_t b_type, data_type_t c_type,
        data_type_t co_type, data_type_t acc_type, int align_a, int align_b,
        int align_c, dim_t m, dim_t n, dim_t k, dim_t lda, dim_t ldb, dim_t ldc,
        dim_t batch) {
    using namespace ngen;
    using namespace kcatalog;

    arch_ = arch;
    hw_ = convert_dnnl_arch_to_hw(arch);
    stepping_ = stepping;
    m_ = m;
    n_ = n;
    k_ = k;
    eu_count_ = eu_count;
    disable_systolic_ = !has_systolic;

    align_a = nstl::max(align_a, int(types::data_type_size(a_type)));
    align_b = nstl::max(align_b, int(types::data_type_size(b_type)));
    align_c = nstl::max(align_c, int(types::data_type_size(c_type)));

    // Set up problem structure.
    problem_.Ta = problem_.Ta_ext = convert_dnnl_to_kernel_type(a_type);
    problem_.Tb = problem_.Tb_ext = convert_dnnl_to_kernel_type(b_type);
    problem_.Tc = convert_dnnl_to_kernel_type(acc_type);
    problem_.Tco = convert_dnnl_to_kernel_type(co_type);
    problem_.Tc_ext = convert_dnnl_to_kernel_type(c_type);
    problem_.Ts = problem_.Tc;
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
    if (a_offset || b_offset) problem_.abOffset = ABOffset::Calc;
    if (a_offset) problem_.aoPtrDims = 0;
    if (b_offset) problem_.boPtrDims = 0;

    if (problem_.Ta.isInteger()) problem_.Ts = Type::f32;

    if (alpha == 1.0f) problem_.alpha = alpha;
    if (beta == 0.0f || beta == 1.0f) problem_.beta = beta;

    auto status = transfer_post_ops(post_ops, swap_ab);
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
    if (lda * problem_.Ta >= 64) *tags++ = kcatalog::ReqBlock2DA;
    if (ldb * problem_.Tb >= 64) *tags++ = kcatalog::ReqBlock2DB;
    if (ldc * problem_.Tc >= 64) *tags++ = kcatalog::ReqBlock2DC;
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
    eval_params.postOps = (post_ops.len() > 0);
    eval_params.cConvert = (acc_type != c_type);
    eval_params.euCount = eu_count;
    eval_params.batch = (batch_dims > 0);

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

    return finalize();
}

status_t gen_gemm_xe_systolic_kernel_desc_t::select_kernel(
        compute::gpu_arch_t arch, int stepping, int eu_count, int batch_dims,
        bool packed_c, bool trans_co, bool a_offset, bool b_offset,
        bool c_offset, bool bias, float alpha, float beta,
        const post_ops_t &post_ops, data_type_t a_type, data_type_t b_type,
        data_type_t c_type, data_type_t co_type, data_type_t acc_type, dim_t m,
        dim_t n, dim_t k, dim_t batch, int unroll_m, int unroll_n, bool alt) {
    using namespace ngen;
    using namespace kcatalog;

    arch_ = arch;
    hw_ = convert_dnnl_arch_to_hw(arch);
    stepping_ = stepping;
    m_ = m;
    n_ = n;
    k_ = k;
    eu_count_ = eu_count;

    if (!utils::one_of(hw_, HW::XeHP, HW::XeHPG, HW::XeHPC))
        return status::unimplemented;

    bool xehpc = (hw_ == HW::XeHPC);

    auto osys = xehpc ? 16 : 8;
    auto ksys = int(32 / types::data_type_size(a_type));
    auto csys = int(4 / types::data_type_size(a_type));

    problem_.Ta = problem_.Ta_ext = convert_dnnl_to_kernel_type(a_type);
    problem_.Tb = problem_.Tb_ext = convert_dnnl_to_kernel_type(b_type);
    problem_.Tc = convert_dnnl_to_kernel_type(acc_type);
    problem_.Tco = convert_dnnl_to_kernel_type(co_type);
    problem_.Tc_ext = convert_dnnl_to_kernel_type(c_type);
    problem_.Ts = Type::f32;
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
    if (a_offset || b_offset) problem_.abOffset = ABOffset::Load;
    if (a_offset) problem_.aoPtrDims = 0;
    if (b_offset) problem_.boPtrDims = 0;
    if (alpha == 1.0f) problem_.alpha = alpha;
    if (beta == 0.0f || beta == 1.0f) problem_.beta = beta;

    auto status = transfer_post_ops(post_ops);
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
    eval_params.postOps = (post_ops.len() > 0);
    eval_params.cConvert = (acc_type != c_type);
    eval_params.batch = (batch_dims > 0);

    entry_ = select(gemm_catalog, match_params, eval_params, aux_params_);

    if (!entry_) return status::unimplemented;

    return finalize();
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
    if (problem.abOffset != ABOffset::None) {
        if (problem.aoPtrDims < 0 && problem.boPtrDims < 0)
            interface_.newArgument("abo", DataType::ud);
        else {
            if (problem.aoPtrDims >= 0)
                interface_.newArgument(
                        "ao_ptr", ExternalArgumentType::GlobalPtr);
            if (problem.boPtrDims >= 0)
                interface_.newArgument(
                        "bo_ptr", ExternalArgumentType::GlobalPtr);
        }
    }
    if (problem.cOffset != COffset::None || problem.sumA || problem.sumB) {
        interface_.newArgument("CO", ExternalArgumentType::GlobalPtr);
        interface_.newArgument("offset_CO", DataType::d);
        if (problem.cOffset == COffset::Pre)
            interface_.newArgument("ldco", DataType::d);
    }

    if (strategy.needsTempC(problem))
        interface_.newArgument("temp_C", ExternalArgumentType::GlobalPtr);
    for (int i = 0; i < problem.postOps.len(); i++) {
        if (problem.postOps.entry_[i].kind != primitive_kind::binary) continue;
        auto bname = "binary" + std::to_string(i);
        interface_.newArgument(bname, ExternalArgumentType::GlobalPtr);
        interface_.newArgument("offset_" + bname, DataType::d);
        if (problem.binaryRow[i] && problem.binaryCol[i])
            interface_.newArgument("ld" + bname, DataType::d);
    }
    interface_.newArgument("flags", DataType::ud);
    if ((strategy.kParallel || strategy.kParallelLocal)
            && !strategy.kParallelVariable)
        interface_.newArgument("k0", DataType::d);
    if (problem.batch == BatchMode::Strided) {
        if (problem.batchDims > 1) {
            interface_.newArgument("stride_A1", DataType::d);
            interface_.newArgument("stride_B1", DataType::d);
            interface_.newArgument("stride_C1", DataType::d);
            for (int i = 0; i < problem.postOps.len(); i++)
                if (problem.postOps.entry_[i].kind == primitive_kind::binary
                        && problem.binaryBatch[i])
                    interface_.newArgument(
                            "stride1_binary" + std::to_string(i), DataType::d);
        }
        interface_.newArgument("stride_A", DataType::d);
        interface_.newArgument("stride_B", DataType::d);
        interface_.newArgument("stride_C", DataType::d);
        for (int i = 0; i < problem.postOps.len(); i++)
            if (problem.postOps.entry_[i].kind == primitive_kind::binary
                    && problem.binaryBatch[i])
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

    interface_.externalName(kernel_name());
}

gpu::compute::binary_t gen_gemm_kernel_t::get_binary(
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

    switch (desc()->hw_) {
        REG_GEN9_ISA(ARCH_DISPATCH(Gen9))
        REG_GEN11_ISA(ARCH_DISPATCH(Gen11))
        REG_XELP_ISA(ARCH_DISPATCH(XeLP))
        REG_XEHP_ISA(ARCH_DISPATCH(XeHP))
        REG_XEHPG_ISA(ARCH_DISPATCH(XeHPG))
        REG_XEHPC_ISA(ARCH_DISPATCH(XeHPC))
        default: assert(!"Unsupported architecture"); break;
    }

    return {};

#undef ARCH_DISPATCH
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
