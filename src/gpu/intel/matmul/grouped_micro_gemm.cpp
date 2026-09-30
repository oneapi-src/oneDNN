/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/matmul/grouped_micro_gemm.hpp"
#include "gpu/intel/matmul/grouped_post_ops_gen.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "gemmstone/microkernel/shim.hpp"
#include "gemmstone/microkernel_selector.hpp"
#include "gemmstone/strategy_parser.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/jit/ir/block_2d_utils.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

namespace {

// grouped_micro_gemm cross-thread argument bytes, plus headroom.
constexpr int host_argument_bytes = 256;

} // namespace

status_t grouped_micro_gemm_t::pd_t::init_microkernels(
        const impl::engine_t *engine) {
    using namespace jit;
    using namespace gemmstone;
    using namespace gemmstone::microkernel;
    using gemm::jit::convert_dnnl_to_kernel_type;

    assert(engine->kind() == engine_kind::gpu);
    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    bool use_systolic_ukernel = dev_info->mayiuse_systolic();

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = use_systolic_ukernel;
    hw_info.isEfficient64Bit = dev_info->is_efficient_64bit();
    auto product = dev_info->product();
    auto hw = getCore(product.family);
    auto stepping = hw_info.gmdid & 0xFF;

    if (hw_info.gmdid == 0) return status::unimplemented;

    memory_desc_wrapper src_mdw(src_md(0));
    memory_desc_wrapper wei_mdw(weights_md());
    memory_desc_wrapper dst_mdw(dst_md());

    auto convert_dnnl_to_kernel_layout = [](const memory_desc_t *md) {
        return (gemm_desc_t::get_trans(*md) == dnnl_trans) ? MatrixLayout::T
                                                           : MatrixLayout::N;
    };

    GEMMProblem problem;
    problem.product = product;
    problem.Ta_ext = convert_dnnl_to_kernel_type(wei_mdw.data_type());
    problem.Tb_ext = convert_dnnl_to_kernel_type(src_mdw.data_type());
    problem.Tc_ext = problem.Ts = problem.Tc = Type::f32;

    problem.Ta = problem.Ta_ext;
    problem.Tb = problem.Tb_ext;

    dim_t lda, ldb;
    data_type_t adt, bdt;
    SizeParams sizes;

    switch (grouped_axis_) {
        case grouped_axis_t::m_axis:
            // src is the B operand: [total_M, K] row-major -> ldb = K, contiguous
            // along k (layout N). gemm_desc_t helpers read the blocking desc,
            // which is only populated for the dense 3D weights of this workload.
            sizes.m = static_cast<uint16_t>(N());
            sizes.n = is_gemv_ ? 1 : 32;
            sizes.k = static_cast<uint16_t>(K());
            lda = gemm_desc_t::get_ld(*wei_mdw.md_);
            ldb = K();
            problem.A.layout = convert_dnnl_to_kernel_layout(wei_mdw.md_);
            problem.B.layout = MatrixLayout::N;
            adt = wei_mdw.data_type();
            bdt = src_mdw.data_type();
            break;
        case grouped_axis_t::k_axis: {
            // The C tile is col-major, so the operand order follows the
            // dst transposition:
            //   notrans (N contiguous): A = wei [total_K, N] -> tile is N x M
            //   trans   (M contiguous): A = src [M, total_K] -> tile is M x N
            // Both operands are k-major, hence A layout N and B layout T either
            // way; only the leading dimensions and extents swap.
            const bool trans_c = transc();
            if (trans_c) {
                lda = sizes.m = static_cast<uint16_t>(M());
                ldb = sizes.n = static_cast<uint16_t>(N());
                adt = src_mdw.data_type();
                bdt = wei_mdw.data_type();
            } else {
                lda = sizes.m = static_cast<uint16_t>(N());
                ldb = sizes.n = static_cast<uint16_t>(M());
                adt = wei_mdw.data_type();
                bdt = src_mdw.data_type();
            }
            // use the average k size to avoid unnecessary unrolls
            sizes.k = static_cast<uint16_t>(utils::div_up(K(), ngroups_));
            problem.A.layout = MatrixLayout::N;
            problem.B.layout = MatrixLayout::T;
        } break;
        default:
            assert(!"grouped_axis::n_axis is not implemented");
            return status::unimplemented;
    }

    problem.C.layout = MatrixLayout::N;
    problem.A.setAlignment(alignmentForLD(
            static_cast<int>(types::elements_to_bytes(adt, lda))));

    auto ldb_bytes = types::elements_to_bytes(bdt, ldb);
    if (dev_info->gpu_arch() >= compute::gpu_arch_t::xe_hpc
            && ldb_bytes % block_2d_base_alignment(hw) == 0) {
        problem.B.setAlignment(static_cast<int>(ldb_bytes));
    } else {
        problem.B.setAlignment(alignmentForLD(static_cast<int>(ldb)));
    }

    problem.C.setAlignment(problem.Tc.size());

    GEMMOptions opts;
    opts.scaleA = wei_quant_.with_scale() && wei_group_sizes_[1] < K();
    opts.offsetA = wei_quant_.with_zp();
    opts.scaleB = src_quant_.with_scale() && src_group_sizes_[1] < K();
    opts.offsetB = src_quant_.with_zp();
    opts.slmPtr = true;
    opts.kParallelLocal = is_gemv_;
    const HostPayload host {sg_size_, host_argument_bytes};

    if (opts.scaleA) {
        data_type_t wei_scale_dt = wei_quant_.scale_dt();
        problem.Ta_scale = convert_dnnl_to_kernel_type(wei_scale_dt);
        problem.A_scale.setAlignment(alignmentForLD(
                static_cast<int>(types::data_type_size(wei_scale_dt))));
        problem.A_scale.layout = MatrixLayout::N;
        problem.asPtrDims = 2;
    }

    if (opts.offsetA) {
        data_type_t wei_zp_dt = wei_quant_.zp_dt();
        problem.Tao = convert_dnnl_to_kernel_type(wei_zp_dt);
        problem.AO.setAlignment(
                static_cast<int>(types::data_type_size(wei_zp_dt)));
        problem.AO.layout = MatrixLayout::N;
        problem.aoPtrDims = (wei_quant_.zp_mask() == 5
                                    && attr()->zero_points_.has_default_groups(
                                            DNNL_ARG_WEIGHTS))
                ? 1
                : 2;
        problem.aOffset = ABOffset::Calc;
    }

    if (opts.scaleB) {
        data_type_t src_scale_dt = src_quant_.scale_dt();
        problem.Tb_scale = convert_dnnl_to_kernel_type(src_scale_dt);
        problem.B_scale.setAlignment(
                static_cast<int>(types::data_type_size(src_scale_dt)));
        problem.bsPtrDims
                = (src_quant_.scale_mask() == (src_qmask_M() | src_qmask_K()))
                ? 2
                : 1;
        problem.B_scale.layout
                = problem.bsPtrDims > 1 ? MatrixLayout::N : MatrixLayout::T;
    }
    if (opts.offsetB) {
        data_type_t src_zp_dt = src_quant_.zp_dt();
        problem.Tbo = convert_dnnl_to_kernel_type(src_zp_dt);
        problem.BO.setAlignment(
                static_cast<int>(types::data_type_size(src_zp_dt)));
        problem.boPtrDims
                = (src_quant_.zp_mask() == (src_qmask_M() | src_qmask_K())) ? 2
                                                                            : 1;
        problem.bOffset = ABOffset::Calc;
        problem.BO.layout
                = problem.boPtrDims > 1 ? MatrixLayout::N : MatrixLayout::T;
    }

    if (opts.scaleA || opts.offsetA) {
        problem.aqGroupM = wei_group_sizes_[2];
        problem.aqGroupK = utils::rnd_up_pow2(wei_group_sizes_[1]);
    }

    if (opts.scaleB || opts.offsetB) {
        problem.bqGroupN = src_group_sizes_[0];
        problem.bqGroupK
                = static_cast<int>(utils::rnd_up_pow2(src_group_sizes_[1]));
    }

    // Mixed s8/s4 DPAS support:
    // - Xe3p: Not supported, require s4->s8 upconversion
    // - pre-Xe3p: supported, but only when s4 matrix doesn't have zero points
    bool has_s8s4_dpas = dev_info->gpu_arch() != compute::gpu_arch_t::xe3p;
    if (problem.Ta_ext.isSubByteInt() && problem.Tb_ext.isInt8()) {
        bool s8s4_dpas_ok = (problem.Ta_ext.bits() == 4) && has_s8s4_dpas
                && !opts.offsetA;
        if (!s8s4_dpas_ok) problem.Ta = Type::s8;
    }
    if (problem.Tb_ext.isSubByteInt() && problem.Ta_ext.isInt8()) {
        bool s8s4_dpas_ok = (problem.Tb_ext.bits() == 4) && has_s8s4_dpas
                && !opts.offsetB;
        if (!s8s4_dpas_ok) problem.Tb = Type::s8;
    }

    // When both A and B are integers and group sums are needed, we
    // can avoid using group sums by converting one of the inputs to
    // f16/bf16.
    if (problem.Ta.isInteger() && problem.Tb.isInteger()
            && ((problem.needsAGroupSums() || problem.needsBGroupSums())
                    || ((opts.scaleA || opts.scaleB)
                            && dev_info->gpu_arch()
                                    < compute::gpu_arch_t::xe_hpc))) {
        Type ctype = Type::f16;
        if (utils::one_of(Type::bf16, problem.Ta_scale, problem.Tb_scale,
                    convert_dnnl_to_kernel_type(dst_mdw.data_type())))
            ctype = Type::bf16;
        if (problem.Ta_ext.bits() < problem.Tb_ext.bits()) {
            problem.Ta = ctype;
        } else {
            problem.Tb = ctype;
        }
    } else if (problem.Ta.isInteger() && problem.Tb.isInteger() && !opts.scaleA
            && !opts.scaleB) {
        problem.Tc = problem.Ts = problem.Tc_ext = Type::s32;
        ugemm_result_dt_ = data_type::s32;
    }

    auto strat_override = [&](gemmstone::GEMMStrategy &strat) {
        std::string newStrat;
        using namespace gemmstone;
        strat.dpasw |= strat.fused;
        newStrat = gpu_utils::dev_getenv("GRPGEMM_USTRATEGY", newStrat);
        if (!newStrat.empty()) {
            // Example: 16 16 1 0 aT32 aM32 aB wg 2x4 sys
            printf("GRPGEMM_USTRATEGY: %s\n", newStrat.c_str());
            strat = GEMMStrategy(hw, stepping);
            std::stringstream ss(newStrat);
            ss >> strat.unroll[0];
            ss >> strat.unroll[1];
            float a, b;
            ss >> a;
            ss >> b;
            Scalar alpha((int)a), beta((int)b);
            std::string strategyString;
            std::getline(ss >> std::ws, strategyString);
            parseStrategy(strategyString, problem, strat);
            adjustStrategy(hw, problem, strat);
        }
        strategyGRFs_ = strat.GRFs;
    };

    try {
        gemm_ = selectGEMM(
                opts, host, hw_info, sizes, problem, {}, strat_override);
    } catch (const std::runtime_error &) {
        std::vector<StrategyRequirement> reqs;

        // TODO: These values should be based on the eu_count
        dim_t m_unroll = sg_size_;

        // Extent of the microkernel's n dimension: tokens per group for the
        // m_axis workload; for k_axis it follows the operand order above.
        float avg_m = float(M()) / ngroups_;
        if (grouped_axis_ == grouped_axis_t::k_axis) {
            avg_m = float(transc() ? N() : M());
        }

        dim_t n_unroll = std::max<dim_t>(2, utils::rnd_up_pow2(dim_t(avg_m)));
        dim_t min_n_unroll = 1;
        dim_t max_n_unroll = 0;
        dim_t max_wg_n = 4;
        dim_t min_wg_n = 1;

        switch (dev_info->gpu_arch()) {
            case compute::gpu_arch_t::xe_lp:
            case compute::gpu_arch_t::xe_hp:
                max_n_unroll = (problem.aqGroupK > 64
                                       && problem.A.layout == MatrixLayout::T)
                                || (problem.Ta_ext.isF8() && opts.scaleA
                                        && opts.scaleB)
                                || (problem.Ta_scale == Type::f8_e8m0
                                        || problem.Tb_scale == Type::f8_e8m0)
                        ? 8
                        : 16;
                break;
            case compute::gpu_arch_t::xe_hpg: {
                bool is_xelpg = (product.family == ngen::ProductFamily::ARL
                        || product.family == ngen::ProductFamily::MTL);
                max_n_unroll = (problem.Ta_ext.bits() <= 8
                                       && problem.Ta_ext.isInteger())
                        ? sg_size_ * problem.Ta_ext
                        : 16;
                if (is_xelpg && !hw_info.systolicAvailable
                        && problem.Ta_ext.bits() <= 8) {
                    min_n_unroll = (opts.scaleA || opts.scaleB) ? sg_size_ : 4;
                }
                if (!dev_info->mayiuse_systolic()) max_wg_n = 2;
                if (problem.Ta_ext.bits() <= 8) min_wg_n = 2;
            } break;
            case compute::gpu_arch_t::xe_hpc: max_n_unroll = 32; break;
            default:
                if (problem.Tb != Type::f32) {
                    m_unroll = std::max<dim_t>(
                            sg_size_, sg_size_ / problem.Ta_ext);
                }
                max_n_unroll = 32;
        }

        reqs.push_back(StrategyRequirement::UnrollM == m_unroll);
        reqs.push_back(StrategyRequirement::UnrollN
                == std::max(min_n_unroll, std::min(n_unroll, max_n_unroll)));
        reqs.push_back(StrategyRequirement::WGM == 2);
        reqs.push_back(StrategyRequirement::WGN
                == utils::rnd_up_pow2(std::max(min_wg_n,
                        std::min((dim_t)(avg_m / reqs[1].value), max_wg_n))));
        try {
            gemm_ = selectGEMM(
                    opts, host, hw_info, sizes, problem, reqs, strat_override);
        } catch (const std::runtime_error &ex) {
            VDISPATCH_MATMUL_IC(false,
                    "gemm microkernel generation failure with message: %s",
                    ex.what());
        }
    }

    CHECK(compute::validate_microkernel(gemm_, "grouped_gemm"));

    /* Generate microkernel shims */
    kernel_ctx_.define_int("SUBGROUP_SIZE", sg_size_);

    compute::microkernel_shims_t shims(
            kernel_ctx_, sg_size_, dev_info->gpu_arch());
    shims.add("gemm_grouped.h", "grouped", gemm_);
    shims.require_grfs(strategyGRFs_);
    shims.finalize();

    return status::success;
}

template <size_t N>
void calc_group_sizes(std::array<int, N> &dims, const quant_entry_t &entry,
        const memory_desc_wrapper &desc) {
    memory_desc_t md;
    entry.get_md(md, *desc.md_);
    std::transform(desc.dims(), desc.dims() + dims.size(), md.dims, begin(dims),
            [](dim_t d, dim_t d2) -> int {
        return static_cast<int>(d2 == 0 ? 1 : d / d2);
    });
}

status_t grouped_micro_gemm_t::pd_t::init_k_axis(const impl::engine_t *engine) {
    using namespace data_type;

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper wei_d(weights_md(0));
    memory_desc_wrapper dst_d(dst_md());

    const data_type_t src_dt = src_d.data_type();

    VDISPATCH_MATMUL(!with_reduce(), VERBOSE_UNSUPPORTED_FEATURE, "reduce");
    VDISPATCH_MATMUL(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    // src is col-major, wei is row-major
    const sparse_desc_t::grouped_desc_t &src_grouped
            = src_d.sparse_desc().grouped_desc;
    const sparse_desc_t::grouped_desc_t &wei_grouped
            = wei_d.sparse_desc().grouped_desc;

    VDISPATCH_MATMUL(src_grouped.variable_dim_idx == 1
                    && wei_grouped.variable_dim_idx == 0,
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_MATMUL(src_grouped.group_count == wei_grouped.group_count,
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src ngroups", "wei ngroups",
            (int)src_grouped.group_count, (int)wei_grouped.group_count);
    VDISPATCH_MATMUL(utils::everyone_is(s32, src_d.metadata_type(0),
                             wei_d.metadata_type(0)),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    const dims_t &src_strides = src_d.strides();
    const dims_t &wei_strides = wei_d.strides();
    VDISPATCH_MATMUL(src_strides[0] == 1 && src_strides[1] == src_d.dims()[0],
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_MATMUL(wei_strides[1] == 1 && wei_strides[0] == wei_d.dims()[1],
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    VDISPATCH_MATMUL(!dst_d.is_sparse_desc() && !dst_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    if (dst_d.format_any())
        CHECK(memory_desc_init_by_strides(dst_md_, nullptr));

    VDISPATCH_MATMUL(dst_d.matches_one_of_tag(format_tag::abc, format_tag::acb),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");

    VDISPATCH_MATMUL(
            utils::everyone_is(src_dt, wei_d.data_type(), dst_d.data_type()),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            utils::one_of(src_dt, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);

    ngroups_ = src_grouped.group_count;
    is_gemv_ = false;
    with_post_op_ = false;

    return status::success;
}

status_t grouped_micro_gemm_t::pd_t::init_m_axis(const impl::engine_t *engine) {
    using namespace data_type;

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper wei_d(weights_md(0));
    memory_desc_wrapper dst_d(dst_md());

    data_type_t src_dt = src_d.data_type();
    data_type_t wei_dt = wei_d.data_type();
    data_type_t dst_dt = dst_d.data_type();

    // Check for grouped encoding on dst
    VDISPATCH_MATMUL(dst_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Weights should be dense
    VDISPATCH_MATMUL(!wei_d.is_sparse_desc() && !wei_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    VDISPATCH_MATMUL_SC(attr_.post_ops_.set_default_formats(dst_md(0)),
            VERBOSE_UNSUPPORTED_POSTOP);

    // Extract grouped encoding
    const sparse_desc_t::grouped_desc_t &src_grouped
            = src_d.sparse_desc().grouped_desc;
    const sparse_desc_t::grouped_desc_t &dst_grouped
            = dst_d.sparse_desc().grouped_desc;

    // Resolve format_any to plain dense
    if (wei_d.format_any())
        CHECK(memory_desc_init_by_strides(weights_md_, nullptr));

    VDISPATCH_MATMUL(wei_d.matches_one_of_tag(format_tag::abc, format_tag::acb),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    // Validate matching number of groups
    VDISPATCH_MATMUL(src_grouped.group_count == dst_grouped.group_count,
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src ngroups", "dst ngroups",
            (int)src_grouped.group_count, (int)dst_grouped.group_count);

    ngroups_ = src_grouped.group_count;
    is_gemv_ = M() < ngroups_;
    with_post_op_ = !attr()->post_ops_.has_default_values();
    if (with_post_op_) {
        CHECK(check_post_op_chain(
                *attr(), dst_d, ngroups_, po_chain_, binary_scale_dts_));
    }

    VDISPATCH_MATMUL(utils::one_of(src_dt, f32, f16, bf16, u8, s8, f8_e5m2,
                             f8_e4m3, f4_e2m1),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(utils::one_of(wei_dt, f32, f16, bf16, u8, s8, s4, u4,
                             f8_e5m2, f8_e4m3, f4_e2m1),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            utils::one_of(dst_dt, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);

    // WOQ (fp src + int wei) requires weight scales and fpmath apply_to_int
    VDISPATCH_MATMUL(
            IMPLICATION(!types::is_integral_dt(src_dt)
                            && types::is_integral_dt(wei_dt),
                    wei_quant_.with_scale() && attr()->fpmath_.apply_to_int_),
            VERBOSE_UNSUPPORTED_DT_CFG);

    const bool wei_subbyte = utils::one_of(wei_dt, s4, u4);
    VDISPATCH_MATMUL(IMPLICATION(wei_subbyte, (K() % 2) == 0), VERBOSE_BAD_DIM,
            "weights", 1);
    VDISPATCH_MATMUL(IMPLICATION(wei_subbyte, (N() % 2) == 0), VERBOSE_BAD_DIM,
            "weights", 2);

    // Check offsets are int32
    VDISPATCH_MATMUL(utils::everyone_is(s32, src_d.metadata_type(0),
                             dst_d.metadata_type(0)),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Check for limited Bias support
    if (with_bias()) {
        memory_desc_wrapper bia_d(weights_md(1));
        VDISPATCH_MATMUL(!bia_d.is_sparse_desc() && !bia_d.is_grouped_desc(),
                VERBOSE_UNSUPPORTED_BIAS_CFG);
        VDISPATCH_MATMUL(bia_d.ndims() == 2, VERBOSE_UNSUPPORTED_BIAS_CFG);
        // Bias shape should be [num_experts, N]
        VDISPATCH_MATMUL(bia_d.dims()[0] == src_grouped.group_count,
                VERBOSE_INCONSISTENT_DIM, "bia_d", 0, "src_grouped.group_count",
                -1);
        VDISPATCH_MATMUL(bia_d.dims()[1] == wei_d.dims()[2],
                VERBOSE_INCONSISTENT_DIM, "bia_d", 1, "wei_d", 2);
    }

    // Check for supported quantization schemes
    if (src_quant_.with_scale()) {
        VDISPATCH_MATMUL(utils::one_of(src_quant_.scale_dt(), f32, f16, bf16,
                                 f8_e5m2, f8_e4m3, e8m0, f4_e2m1),
                VERBOSE_UNSUPPORTED_SCALES_CFG ": src scales dt(%s)",
                dnnl_dt2str(src_quant_.scale_dt()));
    }

    if (src_quant_.with_zp()) {
        const int src_zp_mask = src_quant_.zp_mask();
        // Only per-row or per-column zero points supported for src
        VDISPATCH_MATMUL(utils::one_of(src_zp_mask, src_qmask_M(),
                                 src_qmask_M() | src_qmask_K(), 0),
                VERBOSE_UNSUPPORTED_ZP_CFG ": src zero points mask(%d)",
                src_zp_mask);
        VDISPATCH_MATMUL(utils::one_of(src_quant_.zp_dt(), u8, s8),
                VERBOSE_UNSUPPORTED_ZP_CFG ": src zero points dt(%s)",
                dnnl_dt2str(src_quant_.zp_dt()));
    }

    if (src_quant_.with_scale() && src_quant_.with_zp()) {
        const int src_scale_mask = src_quant_.scale_mask();
        const int src_zp_mask = src_quant_.zp_mask();
        VDISPATCH_MATMUL(src_scale_mask == src_zp_mask,
                VERBOSE_UNSUPPORTED_SCALES_CFG
                ": src scale(%d) and zp(%d) mask must match",
                src_scale_mask, src_zp_mask);
    }

    if (wei_quant_.with_scale()) {
        const int wei_mask = wei_quant_.scale_mask();
        VDISPATCH_MATMUL(
                utils::one_of(wei_mask, 7, 5), VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_MATMUL(utils::one_of(wei_quant_.scale_dt(), f32, f16, bf16,
                                 f8_e5m2, f8_e4m3, e8m0, f4_e2m1),
                VERBOSE_UNSUPPORTED_SCALES_CFG ": wei scales dt(%s)",
                dnnl_dt2str(wei_quant_.scale_dt()));
    }

    if (wei_quant_.with_zp()) {
        const int wei_zp_mask = wei_quant_.zp_mask();
        // Only per-column zero points supported for weights
        VDISPATCH_MATMUL(utils::one_of(wei_zp_mask, 7, 5),
                VERBOSE_UNSUPPORTED_ZP_CFG ": wei zero points mask(%d)",
                wei_zp_mask);
        if (wei_zp_mask == 5) {
            VDISPATCH_MATMUL(
                    utils::one_of(wei_quant_.zp_dt(), s32, u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_ZP_CFG ": wei zero points dt(%s)",
                    dnnl_dt2str(wei_quant_.zp_dt()));
        } else {
            VDISPATCH_MATMUL(utils::one_of(wei_quant_.zp_dt(), u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_ZP_CFG ": wei zero points dt(%s)",
                    dnnl_dt2str(wei_quant_.zp_dt()));
        }
    }

    if (wei_quant_.with_scale() && wei_quant_.with_zp()) {
        const int wei_scale_mask = wei_quant_.scale_mask();
        const int wei_zp_mask = wei_quant_.zp_mask();
        VDISPATCH_MATMUL(wei_scale_mask == wei_zp_mask,
                VERBOSE_UNSUPPORTED_SCALES_CFG
                ": wei scale(%d) and zp(%d) mask must match",
                wei_scale_mask, wei_zp_mask);
    }

    if (src_quant_.with_scale()) {
        calc_group_sizes(
                src_group_sizes_, attr()->scales_.get(DNNL_ARG_SRC), src_d);
    } else if (src_quant_.with_zp()) {
        calc_group_sizes(src_group_sizes_,
                attr()->zero_points_.get(DNNL_ARG_SRC), src_d);
    }
    if (wei_quant_.with_scale()) {
        calc_group_sizes(
                wei_group_sizes_, attr()->scales_.get(DNNL_ARG_WEIGHTS), wei_d);
    } else if (wei_quant_.with_zp()) {
        calc_group_sizes(wei_group_sizes_,
                attr()->zero_points_.get(DNNL_ARG_WEIGHTS), wei_d);
    }

    return status::success;
}

// The k_axis kernel names its inputs after the microkernel operand roles,
// since transc() decides which tensor is which. Define the types by role too,
// so the kernel never has to assume that src and wei share one.
status_t grouped_micro_gemm_t::pd_t::init_kernel_ctx_k_axis() {
    const data_type_t src_dt = src_md()->data_type;
    const data_type_t wei_dt = weights_md(0)->data_type;

    const data_type_t a_dt = transc() ? src_dt : wei_dt;
    const data_type_t b_dt = transc() ? wei_dt : src_dt;

    def_data_type(kernel_ctx_, a_dt, "A");
    def_data_type(kernel_ctx_, b_dt, "B");
    kernel_ctx_.define_int(
            "A_ELEMS_PER_BYTE", types::bytes_to_elements(a_dt, 1));
    kernel_ctx_.define_int(
            "B_ELEMS_PER_BYTE", types::bytes_to_elements(b_dt, 1));

    return status::success;
}

// The m_axis kernel additionally carries quantization, bias, GEMV k-slicing
// and post-ops.
status_t grouped_micro_gemm_t::pd_t::init_kernel_ctx_m_axis() {
    const data_type_t src_dt = src_md()->data_type;
    const data_type_t wei_dt = weights_md(0)->data_type;

    def_data_type(kernel_ctx_, src_dt, "SRC");
    def_data_type(kernel_ctx_, wei_dt, "WEI");
    def_data_type(kernel_ctx_, ugemm_result_dt_, "UGEMM_RESULT");

    kernel_ctx_.define_int(
            "SRC_ELEMS_PER_BYTE", types::bytes_to_elements(src_dt, 1));
    kernel_ctx_.define_int(
            "WEI_ELEMS_PER_BYTE", types::bytes_to_elements(wei_dt, 1));

    src_quant_.define_macros(kernel_ctx_, "SRC");
    wei_quant_.define_macros(kernel_ctx_, "WEI");

    kernel_ctx_.define_int("WITH_SRC_SCALES", src_quant_.with_scale());
    kernel_ctx_.define_int("WITH_WEI_SCALES", wei_quant_.with_scale());
    kernel_ctx_.define_int("WITH_SRC_ZP", src_quant_.with_zp());
    kernel_ctx_.define_int("WITH_WEI_ZP", wei_quant_.with_zp());
    if (src_quant_.with_scale() || src_quant_.with_zp()) {
        kernel_ctx_.define_int("SRC_GROUP_SIZE", src_group_sizes_[1]);
    }
    if (wei_quant_.with_scale() || wei_quant_.with_zp()) {
        kernel_ctx_.define_int("WEI_K_GROUP_SIZE", wei_group_sizes_[1]);
        kernel_ctx_.define_int("WEI_N_GROUP_SIZE", wei_group_sizes_[2]);
    }

    kernel_ctx_.define_int("SRC_SCALES_GROUPED",
            src_quant_.with_scale() && src_group_sizes_[1] < K());
    kernel_ctx_.define_int("WEI_SCALES_GROUPED",
            wei_quant_.with_scale() && wei_group_sizes_[1] < K());

    if (src_quant_.with_zp()) {
        kernel_ctx_.define_int("SRC_ZP_ELEMS_PER_BYTE",
                types::bytes_to_elements(src_quant_.zp_dt(), 1));
    }
    if (wei_quant_.with_zp()) {
        kernel_ctx_.define_int("WEI_ZP_ELEMS_PER_BYTE",
                types::bytes_to_elements(wei_quant_.zp_dt(), 1));
    }

    auto bia_dt = weights_md(1)->data_type;
    bool with_binary_grouped_scale
            = (find_po_in_chain(po_chain_, po_kind_t::binary_grouped_scale)
                    != -1);
    bool with_binary_dense_scale
            = (find_po_in_chain(po_chain_, po_kind_t::binary_dense_scale)
                    != -1);
    bool with_binary_nvfp4_scale
            = (find_po_in_chain(po_chain_, po_kind_t::binary_nvfp4_scale)
                    != -1);
    def_data_type(kernel_ctx_, bia_dt, "BIA");
    kernel_ctx_.define_int("WITH_BIAS", with_bias());
    kernel_ctx_.define_int("K_PARALLEL_LOCAL", is_gemv_);
    kernel_ctx_.define_int("WITH_SPARSE_GROUPS", is_gemv_);
    kernel_ctx_.define_int("NUM_GROUPS", ngroups_);
    kernel_ctx_.define_int("WITH_POST_OP", with_post_op_);
    kernel_ctx_.define_int(
            "WITH_BINARY_GROUPED_SCALE", with_binary_grouped_scale);
    kernel_ctx_.define_int("WITH_BINARY_DENSE_SCALE", with_binary_dense_scale);
    kernel_ctx_.define_int("WITH_BINARY_NVFP4_SCALE", with_binary_nvfp4_scale);
    kernel_ctx_.add_custom_header("grouped_post_ops.h",
            generate_post_ops_microgemm_header(*attr(), po_chain_));
    def_data_type(kernel_ctx_, binary_scale_dts_[0], "BINARY_SCALE_GROUPED");
    def_data_type(kernel_ctx_, binary_scale_dts_[1], "BINARY_SCALE_DENSE");

    return status::success;
}

status_t grouped_micro_gemm_t::pd_t::init(const impl::engine_t *engine) {
    assert(engine->kind() == engine_kind::gpu);
    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    VDISPATCH_MATMUL(compute::mayiuse_microkernels(intel_engine),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "microkernels");
    sg_size_ = dev_info->min_subgroup_size();

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper wei_d(weights_md(0));
    memory_desc_wrapper dst_d(dst_md());

    const data_type_t dst_dt = dst_d.data_type();
    src_quant_ = quantization_t(attr(), src_d, DNNL_ARG_SRC);
    wei_quant_ = quantization_t(attr(), wei_d, DNNL_ARG_WEIGHTS);

    // Grouped src is common to both workloads; the tensor grouped alongside it
    // selects which axis is partitioned.
    VDISPATCH_MATMUL(src_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    auto get_var_dim_axis = [](const memory_desc_wrapper &md) {
        if (md.is_grouped_desc()) {
            return md.sparse_desc().grouped_desc.variable_dim_idx;
        } else {
            return -1;
        }
    };

    int src_var_dim_axis = get_var_dim_axis(src_d);
    int wei_var_dim_axis = get_var_dim_axis(wei_d);
    int dst_var_dim_axis = get_var_dim_axis(dst_d);

    VDISPATCH_MATMUL(src_var_dim_axis != -1, VERBOSE_UNSUPPORTED_SPARSE_CFG);

    if (wei_var_dim_axis == -1) {
        VDISPATCH_MATMUL(src_var_dim_axis == dst_var_dim_axis,
                VERBOSE_UNSUPPORTED_SPARSE_CFG
                "src axis(%d) and dst axis(%d) must vary along the same axis "
                "when weights are dense",
                src_var_dim_axis, dst_var_dim_axis);
    } else {
        VDISPATCH_MATMUL(src_var_dim_axis == 1 && wei_var_dim_axis == 0,
                VERBOSE_UNSUPPORTED_SPARSE_CFG
                "src axis(%d) and weights axis(%d) must vary along the k "
                "axis when weights are grouped",
                src_var_dim_axis, wei_var_dim_axis);
    }

    switch (src_var_dim_axis) {
        case 0:
            grouped_axis_ = grouped_axis_t::m_axis;
            kernel_name_ = "grouped_gemm:micro:m_axis";
            CHECK(init_m_axis(engine));
            break;
        case 1:
            grouped_axis_ = grouped_axis_t::k_axis;
            kernel_name_ = "grouped_gemm:micro:k_axis";
            CHECK(init_k_axis(engine));
            break;
        default: return status::unimplemented;
    }

    CHECK(init_microkernels(engine));

    // Definitions shared by the m_axis and k_axis kernels
    kernel_ctx_.set_data_type(dst_dt);
    def_data_type(kernel_ctx_, dst_dt, "DST");
    kernel_ctx_.define_int("WITH_SLM", gemm_.getSetting("slm_size") > 0);
    kernel_ctx_.add_option("-cl-std=CL3.0");

    switch (grouped_axis_) {
        case grouped_axis_t::m_axis: return init_kernel_ctx_m_axis();
        case grouped_axis_t::k_axis: return init_kernel_ctx_k_axis();
        default: return status::unimplemented;
    }
}

status_t grouped_micro_gemm_t::init(impl::engine_t *engine) {
    const char *kernel_name;
    switch (pd()->grouped_axis_) {
        case grouped_axis_t::m_axis:
            kernel_name = "grouped_micro_gemm_m_axis";
            break;
        case grouped_axis_t::k_axis:
            kernel_name = "grouped_micro_gemm_k_axis";
            break;
        default: return status::unimplemented;
    }
    return create_kernel(engine, &kernel_, kernel_name, pd()->kernel_ctx_);
}

status_t grouped_micro_gemm_t::execute_k_axis(const exec_ctx_t &ctx) const {
    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 0);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);

    const memory_desc_t *src_md = ctx.input(DNNL_ARG_SRC)->md();
    const memory_desc_t *wei_md = ctx.input(DNNL_ARG_WEIGHTS)->md();
    const memory_desc_t *dst_md = ctx.output(DNNL_ARG_DST)->md();

    const dim_t M = dst_md->dims[1];
    const dim_t N = dst_md->dims[2];

    const dim_t ldsrc = memory_desc_wrapper(src_md).strides()[1];
    const dim_t ldwei = memory_desc_wrapper(wei_md).strides()[0];
    const dim_t ldc = gemm_desc_t::get_ld(*dst_md);

    // Assign the microkernel operands so that the C tile stores contiguously:
    // its m dimension has to be the contiguous dst dimension. This must match
    // the layouts and extents init_microkernels() described to the generator.
    //   notrans: a = wei, tile is N x M, ldc = N
    //   trans:   a = src, tile is M x N, ldc = M
    const bool trans_c = pd()->transc();
    const memory_storage_t &a_data = trans_c ? src_data : wei_data;
    const memory_storage_t &b_data = trans_c ? wei_data : src_data;
    const dim_t lda = trans_c ? ldsrc : ldwei;
    const dim_t ldb = trans_c ? ldwei : ldsrc;
    const dim_t m = trans_c ? M : N;
    const dim_t n = trans_c ? N : M;

    compute::kernel_arg_list_t arg_list;
    arg_list.append(a_data);
    arg_list.append(lda);
    arg_list.append(b_data);
    arg_list.append(ldb);
    arg_list.append(dst_data);
    arg_list.append(ldc);
    arg_list.append(src_offsets);
    arg_list.append(m);
    arg_list.append(n);

    const size_t wg_tile_m = pd()->gemm_.getSetting("wg_tile_m");
    const size_t wg_tile_n = pd()->gemm_.getSetting("wg_tile_n");

    compute::range_t lws = compute::range_t::one(3);
    lws[0] *= pd()->sg_size_ * pd()->gemm_.getSetting("sg_per_wg_m");
    lws[1] *= pd()->gemm_.getSetting("sg_per_wg_n");
    lws[2] *= pd()->gemm_.getSetting("sg_per_wg_k");

    compute::range_t gws = lws;
    gws[0] *= utils::div_up(m, wg_tile_m);
    gws[1] *= utils::div_up(n, wg_tile_n);
    gws[2] *= pd()->ngroups_;

    return parallel_for(ctx, compute::nd_range_t(gws, lws), kernel_, arg_list);
}

status_t grouped_micro_gemm_t::execute(const exec_ctx_t &ctx) const {
    switch (pd()->grouped_axis_) {
        case grouped_axis_t::m_axis: return execute_m_axis(ctx);
        case grouped_axis_t::k_axis: return execute_k_axis(ctx);
        default:
            assert(!"unknown grouped workload");
            return status::invalid_arguments;
    }
}

status_t grouped_micro_gemm_t::execute_m_axis(const exec_ctx_t &ctx) const {
    // buffer 0: values, buffer 1: offsets
    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 0);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);
    const auto &dst_offsets = CTX_OUT_STORAGE(DNNL_ARG_DST, 1);

    const auto &src_scales = pd()->src_quant_.scales(ctx);
    const auto &src_zero_points = pd()->src_quant_.zero_points(ctx);
    const auto &wei_scales = pd()->wei_quant_.scales(ctx);
    const auto &wei_zero_points = pd()->wei_quant_.zero_points(ctx);

    const auto &bias_data = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    const memory_desc_t *src_md = ctx.input(DNNL_ARG_SRC)->md();
    const memory_desc_t *wei_md = pd()->weights_md();
    const memory_desc_t *dst_md = ctx.output(DNNL_ARG_DST)->md();

    const bool with_src_scales = pd()->src_quant_.with_scale();
    const bool with_src_zero_points = pd()->src_quant_.with_zp();
    const bool with_wei_scales = pd()->wei_quant_.with_scale();
    const bool with_wei_zero_points = pd()->wei_quant_.with_zp();

    dim_t ldsrcq = 0;
    dim_t ldweiq = 0;

    if (with_src_scales || with_src_zero_points) {
        const memory_desc_t *src_quant_md = with_src_scales
                ? ctx.input(DNNL_ARG_SRC | DNNL_ARG_ATTR_SCALES)->md()
                : ctx.input(DNNL_ARG_SRC | DNNL_ARG_ATTR_ZERO_POINTS)->md();
        ldsrcq = static_cast<int>(
                src_quant_md->format_desc.blocking.strides[0]);
    }
    if (with_wei_scales || with_wei_zero_points) {
        const memory_desc_t *wei_quant_md = with_wei_scales
                ? ctx.input(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_SCALES)->md()
                : ctx.input(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_ZERO_POINTS)->md();
        ldweiq = static_cast<int>(
                wei_quant_md->format_desc.blocking.strides[1]);
    }
    dim_t m_all = dst_md->dims[dst_md->ndims - 2];
    dim_t n = dst_md->dims[dst_md->ndims - 1];
    dim_t k = src_md->dims[src_md->ndims - 1];

    dim_t ldsrc = src_md->dims[src_md->ndims - 1];
    dim_t lddst = dst_md->dims[dst_md->ndims - 1];
    const dims_t &wei_strides_ = wei_md->format_desc.blocking.strides;
    compute::int64x4_t wei_strides
            = {static_cast<int64_t>(wei_strides_[wei_md->ndims - 3]),
                    static_cast<int64_t>(wei_strides_[wei_md->ndims - 2]),
                    static_cast<int64_t>(wei_strides_[wei_md->ndims - 1]),
                    static_cast<int64_t>(wei_strides_[wei_md->ndims - 0])};

    compute::kernel_arg_list_t arg_list;

    arg_list.append(src_data);
    arg_list.append(ldsrc);
    arg_list.append(wei_data);
    arg_list.append(wei_strides);
    arg_list.append(dst_data);
    arg_list.append(lddst);
    arg_list.append(src_offsets);
    arg_list.append(dst_offsets);
    arg_list.append(src_scales);
    arg_list.append(src_zero_points);
    arg_list.append(ldsrcq);
    arg_list.append(wei_scales);
    arg_list.append(wei_zero_points);
    arg_list.append(ldweiq);
    arg_list.append(n);
    arg_list.append(k);
    arg_list.append(bias_data);

    const memory_storage_t *grouped_scale = &memory_storage_t::empty_storage();
    const memory_storage_t *dense_scale = &memory_storage_t::empty_storage();
    const memory_storage_t *nvfp4_scale = &memory_storage_t::empty_storage();
    if (pd()->with_post_op_) {
        const auto &po_chain = pd()->po_chain_;
        for (int i = 0; i < pd()->attr()->post_ops_.len(); ++i) {
            auto &e = pd()->attr()->post_ops_.entry_[i];
            if (!e.is_binary()) continue;
            const int po_arg
                    = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1;
            if (po_chain[i] == po_kind_t::binary_grouped_scale) {
                grouped_scale = &CTX_IN_STORAGE(po_arg, 0);
            } else if (po_chain[i] == po_kind_t::binary_dense_scale) {
                dense_scale = &CTX_IN_STORAGE(po_arg, 0);
            } else if (po_chain[i] == po_kind_t::binary_nvfp4_scale) {
                nvfp4_scale = &CTX_IN_STORAGE(po_arg, 0);
            }
        }
    }

    arg_list.append(*grouped_scale);
    arg_list.append(*dense_scale);
    arg_list.append(*nvfp4_scale);

    size_t sg_per_wg_m = pd()->gemm_.getSetting("sg_per_wg_m");
    size_t sg_per_wg_n = pd()->gemm_.getSetting("sg_per_wg_n");
    size_t sg_per_wg_k = pd()->gemm_.getSetting("sg_per_wg_k");
    size_t wg_tile_m = pd()->gemm_.getSetting("wg_tile_m");
    size_t wg_tile_n = pd()->gemm_.getSetting("wg_tile_n");

    // Use total_tokens as upper bound for M dimension
    compute::range_t lws = compute::range_t::one(3);
    lws[0] *= pd()->sg_size_;

    lws[0] *= sg_per_wg_m;
    lws[1] *= sg_per_wg_n;
    lws[2] *= sg_per_wg_k;

    dim_t m_dispatch = m_all;
    const int32_t *max_var_dim
            = CTX_IN_MEM(const int32_t *, DNNL_ARG_HINT_MAX_GROUP_SIZE);
    if (max_var_dim && *max_var_dim > 0 && *max_var_dim <= m_all)
        m_dispatch = *max_var_dim;

    compute::range_t gws = lws;
    // Swap wg_tile_[mn]_ for col-major vs row-major representations
    gws[0] *= utils::div_up(n, wg_tile_m);
    // Sparse (gemv) dispatch derives the M offset from the flat token index
    // in dim 2, so dim 1 only needs a single work-group.
    gws[1] *= pd()->is_gemv_ ? 1 : utils::div_up(m_dispatch, wg_tile_n);
    gws[2] *= pd()->is_gemv_ ? m_all : pd()->ngroups_;

    return parallel_for(ctx, compute::nd_range_t(gws, lws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
