/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#include "cpu/x64/zen64/matmul/zen_lowp_matmul.hpp"

#include <assert.h>
#include <limits>

#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/ref_io_helper.hpp"

#include "cpu/matmul/gemm_based_common.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/zen64/common/zen_format_tag.hpp"

#if DNNL_X64_USE_ZEN
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "zendnnl.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace matmul {

using namespace data_type;
using namespace dnnl::impl::cpu::matmul;

#if DNNL_X64_USE_ZEN
using namespace zen_lowp;

namespace {

struct zen_lowp_bmm_params_t {
    int batch_a = 1;
    int batch_b = 1;
    size_t stride_src = static_cast<size_t>(-1);
    size_t stride_wei = static_cast<size_t>(-1);
    size_t stride_dst = static_cast<size_t>(-1);
};

inline zen_lowp_bmm_params_t compute_zen_lowp_bmm_params(
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const matmul_helper_t &helper, bool wei_is_zen_packed) {
    zen_lowp_bmm_params_t p;
    if (src_d.ndims() != 3) return p;

    p.batch_a = static_cast<int>(src_d.dims()[0]);
    p.batch_b = static_cast<int>(weights_d.dims()[0]);
    p.stride_src = static_cast<size_t>(helper.get_a_stride(0));
    p.stride_dst = static_cast<size_t>(helper.get_c_stride(0));
    if (wei_is_zen_packed) {
        const size_t wei_storage_size = weights_d.data_type_size();
        assert(wei_storage_size > 0
                && weights_d.zen_packed_desc().per_slice_size % wei_storage_size
                        == 0);
        // ZenDNN multiplies this stride by size_of(weight dtype). Its s4/u4
        // size is one byte (the packed storage unit), so this also advances
        // correctly between packed 4-bit slices.
        p.stride_wei
                = weights_d.zen_packed_desc().per_slice_size / wei_storage_size;
    } else {
        p.stride_wei = static_cast<size_t>(helper.get_b_stride(0));
    }
    return p;
}

} // namespace
#endif

status_t zen_lowp_matmul_t::pd_t::init(const engine_t *engine) {
    using smask_t = primitive_attr_t::skip_mask_t;

#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else
    // CPU engine only.
    VDISPATCH_MATMUL(
            engine->kind() == engine_kind::cpu, VERBOSE_BAD_ENGINE_KIND);

    // Dense format only (no sparse).
    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // AMD-only vendor gate via xbyak (portable across GCC/Clang/MSVC).
    VDISPATCH_MATMUL(::dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tAMD),
            "This implementation only supports AMD CPUs");

    // Base ISA gate: AVX-512 core (bf16 is a superset on AMD Zen4+).
    VDISPATCH_MATMUL(mayiuse(avx512_core), VERBOSE_UNSUPPORTED_ISA);

    // ---- Memory descriptor data types ----
    const auto src_dt = src_md(0)->data_type;
    const auto wei_dt = weights_md(0)->data_type;
    const auto dst_dt = dst_md(0)->data_type;

    // Support a single leading batch dimension. Higher-rank batching cannot be
    // represented by ZenDNN's single Batch_A/Batch_B stride pair.
    VDISPATCH_MATMUL(
            utils::one_of(ndims(), 2, 3), VERBOSE_BAD_NDIMS, "dst", ndims());
    const bool is_batched = ndims() == 3;

    // No zero-dim tensors.
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    dim_t src_batch = 1, wei_batch = 1;
    if (is_batched) {
        src_batch = src_md(0)->dims[0];
        wei_batch = weights_md(0)->dims[0];
        VDISPATCH_MATMUL(
                src_batch == wei_batch || src_batch == 1 || wei_batch == 1,
                VERBOSE_INCONSISTENT_DIM, "src", 0, "weights", 0);
    }

    // ---- Datatype validation ----
    // Two low-precision configurations are handled by this impl:
    //  1. int8 static quant: u8/s8 src, s8 wei, u8/s8/s32/f32/bf16 dst; s32
    //     accum.
    //  2. WOQ (weight decompression): bf16 src, s4/u4 wei, bf16/f32 dst; f32
    //     accum; weights dequantized via weight scales (+ a u4 weight zp,
    //     which is required for u4 and rejected for s4).
    const bool is_int8 = utils::one_of(src_dt, u8, s8) && wei_dt == s8
            && utils::one_of(dst_dt, u8, s8, s32, f32, bf16);
    const bool is_woq = src_dt == bf16 && utils::one_of(wei_dt, s4, u4)
            && utils::one_of(dst_dt, bf16, f32);
    VDISPATCH_MATMUL(
            utils::one_of(true, is_int8, is_woq), VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(desc()->accum_data_type == (is_woq ? f32 : s32),
            VERBOSE_UNSUPPORTED_DT_CFG);

    // int8 dot-product needs AVX-512 VNNI (vpdpbusd); WOQ decompresses weights
    // to bf16 and only needs avx512_core.
    VDISPATCH_MATMUL(IMPLICATION(is_int8, mayiuse(avx512_core_vnni)),
            VERBOSE_UNSUPPORTED_ISA);

    // WOQ is opt-in via fpmath_mode with apply_to_int (weight decompression).
    // Weights are decompressed to bf16, so the mode must permit bf16 math.
    const bool woq_fpmath_ok = attr()->fpmath_.apply_to_int_
            && utils::one_of(
                    attr()->fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);
    VDISPATCH_MATMUL(
            IMPLICATION(is_woq, woq_fpmath_ok), VERBOSE_UNSUPPORTED_ATTR);

    // ---- Bias validation ----
    // Bias, when present, must be f32 or bf16 and follow the 1xN broadcast.
    if (with_bias()) {
        const auto bia_dt = weights_md(1)->data_type;
        VDISPATCH_MATMUL(utils::one_of(bia_dt, f32, bf16) && is_bias_1xN(),
                VERBOSE_UNSUPPORTED_BIAS_CFG);
    }

    // ---- Attribute validation ----
    // Allow post-ops, sum dtype, scales (including a non-default scale data
    // type), and zero-points. For WOQ also allow grouped scales/zero-points and
    // a non-default fpmath mode. For the int8 path scales_groups /
    // zero_points_groups stay out of the mask, so any grouped scale/zp is
    // rejected. Path-specific scale/zp constraints are enforced below.
    auto attr_mask = smask_t::post_ops | smask_t::sum_dt | smask_t::scales
            | smask_t::scales_data_type | smask_t::zero_points;
    if (is_woq)
        attr_mask = attr_mask | smask_t::scales_groups
                | smask_t::zero_points_groups | smask_t::zero_points_data_type
                | smask_t::fpmath_mode;
    VDISPATCH_MATMUL(attr()->has_default_values(attr_mask, dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    // ZenDNN consumes all quantization buffers as precomputed inputs. In
    // particular, it does not compute dynamic destination scales, which oneDNN
    // exposes as an output argument. Keep this implementation restricted to
    // static scales; path-specific masks, groups, and data types are checked
    // below.
    CHECK(attr_scales_ok(engine));

    // ZenDNN reuses one quantization buffer for every BMM slice. Reject any
    // scale or zero-point whose mask varies along the leading batch dimension,
    // even if a future path-specific mask check becomes more permissive.
    if (is_batched) {
        constexpr int batch_mask = 1 << 0;
        const auto has_batch_quant = [&](const auto &q) {
            for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
                if (!q.has_default_values(arg)
                        && (q.get_mask(arg) & batch_mask) != 0)
                    return true;
            }
            return false;
        };
        VDISPATCH_MATMUL(!has_batch_quant(attr()->scales_),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_MATMUL(!has_batch_quant(attr()->zero_points_),
                VERBOSE_UNSUPPORTED_ZP_CFG);
    }

    // Sum-consistency check (catches sum.dt != dst_dt precision bugs).
    VDISPATCH_MATMUL(attr()->post_ops_.check_sum_consistency(dst_dt, is_int8),
            VERBOSE_UNSUPPORTED_POSTOP);

    // ---- Post-ops validation ----
    // Zen supports: sum, eltwise (relu, gelu_tanh, gelu_erf, tanh,
    // sigmoid/logistic, swish), binary (add, mul).
    auto check_postops = [&]() -> bool {
        const auto &po = attr()->post_ops_;
        for (int i = 0; i < po.len(); i++) {
            const auto &entry = po.entry_[i];
            if (entry.is_sum(/*require_scale_one=*/true,
                        /*require_zp_zero=*/true)) {
                // Sum maps to Zen beta = 1 (plain accumulate); a non-unit sum
                // scale is not detected as sum here and is rejected below as an
                // unsupported post-op.
                if (i != 0) return false;
                if (!utils::one_of(entry.sum.dt, data_type::undef, dst_dt))
                    return false;
                continue;
            } else if (entry.is_eltwise()) {
                if (entry.eltwise.scale != 1.f) return false;
                using namespace alg_kind;
                if (!utils::one_of(entry.eltwise.alg, eltwise_relu,
                            eltwise_gelu_tanh, eltwise_gelu_erf, eltwise_tanh,
                            eltwise_logistic, eltwise_swish))
                    return false;
                if (entry.eltwise.alg == eltwise_relu
                        && entry.eltwise.alpha != 0.f)
                    return false;
            } else if (entry.is_binary()) {
                using namespace alg_kind;
                if (!utils::one_of(entry.binary.alg, binary_add, binary_mul))
                    return false;
                const auto src1_dt = entry.binary.src1_desc.data_type;
                if (!utils::one_of(src1_dt, f32, bf16)) return false;
            } else {
                return false;
            }
        }
        return true;
    };
    VDISPATCH_MATMUL(check_postops(), VERBOSE_UNSUPPORTED_POSTOP);

    // ---- Scale / zero-point validation (path-specific) ----
    if (is_woq) {
        // WOQ weight-scale / weight-zero-point validation lives in init_woq().
        CHECK(init_woq(engine));
    } else {
        // int8 static quant.
        // Scales: f32/bf16 dtype; no groups; masks limited to
        //   src : per-tensor only (mask 0)
        //   wei : per-tensor (0) or per-channel along N/OC (wei_qmask_N())
        //   dst : per-tensor only (mask 0)
        // ZenDNN's AOCL-DLP int8 path supports a per-channel weights scale but
        // only a per-tensor dst scale, so per-channel dst scales are rejected.
        const auto &scales = attr()->scales_;
        auto scale_arg_ok
                = [&](int arg, std::initializer_list<int> allowed_masks) {
            if (scales.has_default_values(arg)) return true;
            if (!utils::one_of(scales.get_data_type(arg), f32, bf16))
                return false;
            if (!scales.get(arg).has_default_groups()) return false;
            const int mask = scales.get_mask(arg);
            for (int m : allowed_masks)
                if (mask == m) return true;
            return false;
        };
        const bool scales_ok = scale_arg_ok(DNNL_ARG_SRC, {0})
                && scale_arg_ok(DNNL_ARG_WEIGHTS, {0, wei_qmask_N()})
                && scale_arg_ok(DNNL_ARG_DST, {0});
        VDISPATCH_MATMUL(scales_ok, VERBOSE_UNSUPPORTED_SCALES_CFG);

        // ZenDNN applies source/weight scales to beta*C as well as A*B, while
        // oneDNN sum semantics apply those scales only to A*B before adding C.
        // A destination scale is safe because oneDNN applies it after sum too.
        const bool with_sum = attr()->post_ops_.find(primitive_kind::sum) != -1;
        VDISPATCH_MATMUL(
                IMPLICATION(with_sum,
                        scales.has_default_values(DNNL_ARG_SRC)
                                && scales.has_default_values(DNNL_ARG_WEIGHTS)),
                VERBOSE_UNSUPPORTED_POSTOP);

        // Zero-points: a per-tensor source zero-point with a u8 source, and a
        // per-tensor destination zero-point with a u8 destination (ZenDNN's
        // AOCL-DLP static-quant zero-point compensation is u8-only). Weight
        // zero-points, s8 src/dst zero-points, and any non-per-tensor / grouped
        // zero-point are rejected.
        const auto &zero_points = attr()->zero_points_;
        VDISPATCH_MATMUL(zero_points.has_default_values(DNNL_ARG_WEIGHTS),
                VERBOSE_UNSUPPORTED_ZP_CFG);
        if (!zero_points.has_default_values(DNNL_ARG_SRC)) {
            VDISPATCH_MATMUL(src_dt == u8
                            && zero_points.get_mask(DNNL_ARG_SRC) == 0
                            && zero_points.get(DNNL_ARG_SRC)
                                       .has_default_groups(),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }
        if (!zero_points.has_default_values(DNNL_ARG_DST)) {
            VDISPATCH_MATMUL(dst_dt == u8
                            && zero_points.get_mask(DNNL_ARG_DST) == 0
                            && zero_points.get(DNNL_ARG_DST)
                                       .has_default_groups(),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    // oneDNN applies the dst scale as a division after bias/post-ops, whereas
    // ZenDNN's AOCL-DLP applies it as a multiply SCALE post-op. We therefore
    // pass ZenDNN the reciprocal of the dst scale, computed at execute() into
    // this scratchpad (per-tensor: 1 element). The reciprocal is always stored
    // as f32 -- computing 1/scale in f32 avoids the extra rounding error a
    // bf16 reciprocal buffer would introduce versus oneDNN's f32 division.
    // (WOQ has no dst scale, so this books nothing for that path.)
    if (!attr()->scales_.has_default_values(DNNL_ARG_DST)) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_matmul_dst_scales,
                /*nelems=*/1, types::data_type_size(f32),
                /*alignment=*/64);
    }

    // Zen matmul_direct uses int for M/N/K; reject runtime dims/strides
    // before set_default_formats() to avoid undefined behavior.
    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Quantized weights must be passed in a plain ab/ba layout (abc/acb with a
    // batch). Prepacked (zen_packed) weights are disabled: when this
    // implementation declines, the next one cannot read the opaque layout.
    // format_any is declined as well, so the implementation that runs picks
    // the weights layout. WOQ consumes only prepacked s4/u4 weights, so it
    // declines.
    VDISPATCH_MATMUL(!zen::is_zen_packed(*weights_md(0)),
            VERBOSE_UNSUPPORTED_FORMAT_KIND);
    VDISPATCH_MATMUL(!memory_desc_wrapper(weights_md(0)).format_any(),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");
    VDISPATCH_MATMUL(!is_woq, VERBOSE_UNSUPPORTED_FEATURE,
            "weight-only quantization requires prepacked weights");

    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    VDISPATCH_MATMUL(gemm_based::check_gemm_compatible_formats(*this),
            VERBOSE_INCOMPATIBLE_GEMM_FMT);

    // Source and destination may have padded inner leading dimensions, but
    // must expose layouts representable by the Zen GEMM/BMM API.
    const memory_desc_wrapper dst_d(dst_md(0));
    VDISPATCH_MATMUL(gemm_based::check_gemm_output_format(*dst_md(0)),
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(gemm_based::check_gemm_input_format(*src_md(0)),
            VERBOSE_UNSUPPORTED_TAG);

    const auto &dst_strides = dst_d.blocking_desc().strides;
    bool dst_no_zero_stride = true;
    for (int i = 0; i < dst_d.ndims(); i++)
        dst_no_zero_stride = dst_no_zero_stride && dst_strides[i] != 0;
    VDISPATCH_MATMUL(dst_no_zero_stride, VERBOSE_UNSUPPORTED_TAG);

    if (is_batched) {
        auto batch_is_outermost = [](const memory_desc_t *md) {
            const memory_desc_wrapper mdw(md);
            const auto &s = mdw.blocking_desc().strides;
            return s[0] >= s[1] && s[0] >= s[2];
        };
        VDISPATCH_MATMUL(
                batch_is_outermost(src_md(0)), VERBOSE_UNSUPPORTED_TAG);
        VDISPATCH_MATMUL(
                batch_is_outermost(dst_md(0)), VERBOSE_UNSUPPORTED_TAG);
        VDISPATCH_MATMUL(
                batch_is_outermost(weights_md(0)), VERBOSE_UNSUPPORTED_TAG);
    }

    VDISPATCH_MATMUL(!::dnnl::impl::cpu::x64::binary_injector::
                             any_binary_postop_rhs_with_ternary_scalar_bcast(
                                     attr()->post_ops_, dst_d),
            VERBOSE_UNSUPPORTED_POSTOP);

    // Resolve format_tag::any on binary post-op src1 memory descriptors.
    VDISPATCH_MATMUL(attr_.set_default_formats(dst_md(0)) == status::success,
            VERBOSE_UNSUPPORTED_POSTOP);

    // ---- Binary post-op shape/format validation ----
    auto check_binary_postop_formats = [&]() -> bool {
        const auto &po = attr()->post_ops_;
        for (int i = 0; i < po.len(); i++) {
            const auto &entry = po.entry_[i];
            if (!entry.is_binary()) continue;

            const auto &src1_desc = entry.binary.src1_desc;
            const auto *dst = dst_md(0);

            if (src1_desc.ndims != dst->ndims) return false;
            const int nd = src1_desc.ndims;
            const int channel_dim = nd - 1;
            for (int d = 0; d < nd; d++) {
                const bool full = src1_desc.dims[d] == dst->dims[d];
                const bool bcast = src1_desc.dims[d] == 1;
                if (!(full || bcast)) return false;
                if (d == channel_dim && !full) return false;
            }
            // ZenDNN's BMM post-op offset advances by m_start*N, so an
            // M-broadcast binary operand cannot be represented.
            if (nd == 3 && src1_desc.dims[nd - 2] != dst->dims[nd - 2])
                return false;

            const memory_desc_wrapper src1_mdw(src1_desc);
            if (!src1_mdw.is_plain()) return false;
            const auto &strides = src1_mdw.blocking_desc().strides;
            if (strides[nd - 1] != 1
                    || strides[nd - 2] != src1_desc.dims[nd - 1])
                return false;
            if (nd == 3 && src1_desc.dims[0] != 1) {
                const size_t m = static_cast<size_t>(src1_desc.dims[1]);
                const size_t n = static_cast<size_t>(src1_desc.dims[2]);
                if (n != 0 && m > std::numeric_limits<size_t>::max() / n)
                    return false;
                if (static_cast<size_t>(strides[0]) != m * n) return false;
            }
        }
        return true;
    };
    VDISPATCH_MATMUL(check_binary_postop_formats(), VERBOSE_UNSUPPORTED_POSTOP);

    // Zen matmul_direct uses int for M/N/K and leading dimensions.
    const matmul_helper_t helper(memory_desc_wrapper(src_md(0)),
            memory_desc_wrapper(weights_md(0)), memory_desc_wrapper(dst_md(0)));
    const dim_t int_max = std::numeric_limits<int>::max();
    const dim_t batch_count = src_batch > wei_batch ? src_batch : wei_batch;
    bool fits_zen_int_api = helper.M() <= int_max && helper.N() <= int_max
            && helper.K() <= int_max && helper.lda() <= int_max
            && helper.ldb() <= int_max && helper.ldc() <= int_max
            && src_batch <= int_max && wei_batch <= int_max
            && batch_count <= int_max;

    if (is_batched && fits_zen_int_api) {
        const auto bmm
                = compute_zen_lowp_bmm_params(memory_desc_wrapper(src_md(0)),
                        memory_desc_wrapper(weights_md(0)), helper,
                        /*wei_is_zen_packed=*/false);
        const size_t int_max_sz = static_cast<size_t>(int_max);
        fits_zen_int_api = bmm.stride_src <= int_max_sz
                && bmm.stride_wei <= int_max_sz && bmm.stride_dst <= int_max_sz;
    }
    VDISPATCH_MATMUL(fits_zen_int_api, VERBOSE_UNSUPPORTED_FEATURE,
            "dimension/stride > INT_MAX is not supported");

    return status::success;
#endif // DNNL_X64_USE_ZEN
}

status_t zen_lowp_matmul_t::pd_t::init_woq(const engine_t *engine) {
    // Weight-only quantization: bf16 src, s4/u4 wei dequantized via a weight
    // scale (per-channel along N, or per-group along K), with a u4 weight
    // zero-point matching the scale granularity -- required for u4 and
    // rejected for s4. No src/dst scales or src/dst zero-points.
    const auto wei_dt = weights_md(0)->data_type;
    const auto &scales = attr()->scales_;
    const auto &zero_points = attr()->zero_points_;

    // No source / destination scales in WOQ; a weight scale is required.
    VDISPATCH_MATMUL(scales.has_default_values(DNNL_ARG_SRC)
                    && scales.has_default_values(DNNL_ARG_DST),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_MATMUL(!scales.has_default_values(DNNL_ARG_WEIGHTS),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    // Weight scale: f32/bf16 dtype; granularity is per-channel (mask N, no
    // groups) or per-group along K (mask K+N, group {G,1}, K % G == 0, and
    // G % 16 == 0 when 1 < G < K to satisfy the hardware group alignment).
    VDISPATCH_MATMUL(
            utils::one_of(scales.get_data_type(DNNL_ARG_WEIGHTS), f32, bf16),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    const int wei_scale_mask = scales.get_mask(DNNL_ARG_WEIGHTS);
    const bool wei_scale_grouped
            = !scales.get(DNNL_ARG_WEIGHTS).has_default_groups();
    bool wei_scale_ok = false;
    if (!wei_scale_grouped) {
        wei_scale_ok = wei_scale_mask == wei_qmask_N();
    } else {
        const dim_t gK = scales.get_group(DNNL_ARG_WEIGHTS, 0);
        const dim_t gN = scales.get_group(DNNL_ARG_WEIGHTS, 1);
        wei_scale_ok = wei_scale_mask == (wei_qmask_K() | wei_qmask_N())
                && gN == 1 && gK > 1 && (K() % gK == 0)
                && IMPLICATION(gK < K(), gK % 16 == 0);
    }
    VDISPATCH_MATMUL(wei_scale_ok, VERBOSE_UNSUPPORTED_SCALES_CFG);

    // Zero-points: none on src/dst. A weight zero-point is supported only for
    // u4 weights, with s8 dtype, and must match the weight-scale granularity
    // (same mask and, if grouped, the same K group size). ZenDNN mandates a
    // weight zero-point for u4 weights and forbids one for (symmetric) s4.
    VDISPATCH_MATMUL(zero_points.has_default_values(DNNL_ARG_SRC)
                    && zero_points.has_default_values(DNNL_ARG_DST),
            VERBOSE_UNSUPPORTED_ZP_CFG);
    const bool has_wei_zp = !zero_points.has_default_values(DNNL_ARG_WEIGHTS);
    VDISPATCH_MATMUL(
            IMPLICATION(wei_dt == u4, has_wei_zp), VERBOSE_UNSUPPORTED_ZP_CFG);
    VDISPATCH_MATMUL(
            IMPLICATION(wei_dt == s4, !has_wei_zp), VERBOSE_UNSUPPORTED_ZP_CFG);
    if (has_wei_zp) {
        VDISPATCH_MATMUL(wei_dt == u4, VERBOSE_UNSUPPORTED_ZP_CFG);
        VDISPATCH_MATMUL(zero_points.get_data_type(DNNL_ARG_WEIGHTS) == s8,
                VERBOSE_UNSUPPORTED_ZP_CFG);
        const bool wei_zp_grouped
                = !zero_points.get(DNNL_ARG_WEIGHTS).has_default_groups();
        VDISPATCH_MATMUL(
                zero_points.get_mask(DNNL_ARG_WEIGHTS) == wei_scale_mask
                        && wei_zp_grouped == wei_scale_grouped,
                VERBOSE_UNSUPPORTED_ZP_CFG);
        if (wei_zp_grouped) {
            VDISPATCH_MATMUL(zero_points.get_group(DNNL_ARG_WEIGHTS, 0)
                                    == scales.get_group(DNNL_ARG_WEIGHTS, 0)
                            && zero_points.get_group(DNNL_ARG_WEIGHTS, 1)
                                    == scales.get_group(DNNL_ARG_WEIGHTS, 1),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    return status::success;
}

status_t zen_lowp_matmul_t::init(engine_t *engine) {
    MAYBE_UNUSED(engine);
#if DNNL_X64_USE_ZEN
    // Build Zen matmul_post_op chain directly from oneDNN attributes.
    const auto &po = pd()->attr()->post_ops_;
    zen_postop_.clear();
    zen_postop_.reserve(po.len());
    postop_indices_.clear();
    postop_indices_.reserve(po.len());
    beta_ = 0.f;

    using pot = zendnnl::ops::post_op_type_t;
    using zd = zendnnl::common::data_type_t;

    for (int i = 0; i < po.len(); i++) {
        const auto &entry = po.entry_[i];
        if (entry.is_sum(/*require_scale_one=*/true,
                    /*require_zp_zero=*/true)) {
            // Sum maps to Zen beta = 1 (C = alpha*A*B + C). pd_t::init() only
            // accepts a unit-scale sum, so beta is always 1 here.
            beta_ = 1.f;
            continue; // not a Zen post-op entry
        }
        matmul_post_op lpo {};
        if (entry.is_eltwise()) {
            switch (entry.eltwise.alg) {
                case alg_kind::eltwise_relu: lpo.po_type = pot::relu; break;
                case alg_kind::eltwise_gelu_tanh:
                    lpo.po_type = pot::gelu_tanh;
                    break;
                case alg_kind::eltwise_gelu_erf:
                    lpo.po_type = pot::gelu_erf;
                    break;
                case alg_kind::eltwise_tanh: lpo.po_type = pot::tanh; break;
                case alg_kind::eltwise_logistic:
                    lpo.po_type = pot::sigmoid;
                    break;
                case alg_kind::eltwise_swish: lpo.po_type = pot::swish; break;
                default: return status::runtime_error;
            }
            lpo.alpha = entry.eltwise.alpha;
            lpo.beta = entry.eltwise.beta;
        } else if (entry.is_binary()) {
            lpo.po_type = (entry.binary.alg == alg_kind::binary_add)
                    ? pot::binary_add
                    : pot::binary_mul;
            switch (entry.binary.src1_desc.data_type) {
                case f32: lpo.dtype = zd::f32; break;
                case bf16: lpo.dtype = zd::bf16; break;
                default: return status::runtime_error;
            }
            const auto &src1_desc = entry.binary.src1_desc;
            lpo.dims.assign(src1_desc.dims, src1_desc.dims + src1_desc.ndims);
        }
        zen_postop_.push_back(lpo);
        postop_indices_.push_back(i);
    }

    // ---- Quantization metadata (scales + zero-points) ----
    // Resolve per-argument dtype and granularity dims once here; the buffer
    // pointers are patched from the exec context at execute().
    const auto &scales = pd()->attr()->scales_;
    const auto &zero_points = pd()->attr()->zero_points_;
    const dim_t N = pd()->N();
    const dim_t K = pd()->K();

    // Weight granularity dims for ZenDNN quant_params:
    //   per-tensor (mask 0)          -> {1}
    //   per-channel along N          -> {1, N}
    //   per-group along K (WOQ)      -> {K / gK, N}  (gK = group size along K)
    // (ZenDNN treats empty dims as 0 elements, so per-tensor uses {1}.)
    auto wei_quant_dims = [&](const auto &q, int arg) -> std::vector<int64_t> {
        const int mask = q.get_mask(arg);
        if (mask == 0) return {1};
        if (!q.get(arg).has_default_groups()) {
            const dim_t gK = q.get_group(arg, 0);
            return {static_cast<int64_t>(K / gK), static_cast<int64_t>(N)};
        }
        return {1, static_cast<int64_t>(N)};
    };

    // Per-tensor src/dst scales (int8 path only).
    auto setup_scalar = [&](int arg, zen_lowp_scale_info_t &s) {
        if (scales.has_default_values(arg)) return;
        s.present = true;
        s.dt = to_zen_dt(scales.get_data_type(arg));
        s.dims = {1};
    };
    setup_scalar(DNNL_ARG_SRC, src_scale_);
    setup_scalar(DNNL_ARG_DST, dst_scale_);
    // The dst scale is passed to ZenDNN as an f32 reciprocal (computed at
    // execute), so advertise f32 regardless of the incoming scale dtype.
    if (dst_scale_.present) dst_scale_.dt = zd::f32;

    // Weight scale: per-tensor/per-channel (int8) or per-channel/per-group
    // along K (WOQ).
    if (!scales.has_default_values(DNNL_ARG_WEIGHTS)) {
        wei_scale_.present = true;
        wei_scale_.dt = to_zen_dt(scales.get_data_type(DNNL_ARG_WEIGHTS));
        wei_scale_.dims = wei_quant_dims(scales, DNNL_ARG_WEIGHTS);
    }

    // Per-tensor src/dst zero-points (int8 path, s32).
    if (!zero_points.has_default_values(DNNL_ARG_SRC)) {
        src_zp_.present = true;
        src_zp_.dt = zd::s32;
        src_zp_.dims = {1};
    }
    if (!zero_points.has_default_values(DNNL_ARG_DST)) {
        dst_zp_.present = true;
        dst_zp_.dt = zd::s32;
        dst_zp_.dims = {1};
    }
    // Weight zero-point (WOQ: u4 weights, s8 dtype, matching the weight-scale
    // granularity).
    if (!zero_points.has_default_values(DNNL_ARG_WEIGHTS)) {
        wei_zp_.present = true;
        wei_zp_.dt = to_zen_dt(zero_points.get_data_type(DNNL_ARG_WEIGHTS));
        wei_zp_.dims = wei_quant_dims(zero_points, DNNL_ARG_WEIGHTS);
    }
#endif // DNNL_X64_USE_ZEN
    return status::success;
}

// ================================================================
// Zen helpers and wrapper (translation-unit local).
// ================================================================
#if DNNL_X64_USE_ZEN
namespace {

status_t zen_lowp_matmul_direct(data_type_t src_dt, data_type_t wei_dt,
        data_type_t dst_dt, data_type_t bia_dt, const void *A, const void *B,
        void *C, const void *bias, dim_t M, dim_t N, dim_t K, dim_t lda,
        dim_t ldb, dim_t ldc, char transA, char transB, char mem_format_b,
        int Batch_A, int Batch_B, size_t batch_stride_src,
        size_t batch_stride_wei, size_t batch_stride_dst,
        const std::vector<matmul_post_op> &cached_postops,
        const std::vector<int> &cached_postop_po_indices, float cached_beta,
        const zen_lowp_scale_info_t &src_scale,
        const zen_lowp_scale_info_t &wei_scale,
        const zen_lowp_scale_info_t &dst_scale, const void *dst_scale_recip,
        const zen_lowp_scale_info_t &src_zp,
        const zen_lowp_scale_info_t &dst_zp,
        const zen_lowp_scale_info_t &wei_zp, const exec_ctx_t &ctx) {
    using zd = zendnnl::common::data_type_t;

    matmul_batch_params_t batch {};
    batch.Batch_A = Batch_A;
    batch.Batch_B = Batch_B;
    batch.batch_stride_src = batch_stride_src;
    batch.batch_stride_wei = batch_stride_wei;
    batch.batch_stride_dst = batch_stride_dst;

    matmul_params params {};
    params.dtypes.src = to_zen_dt(src_dt);
    params.dtypes.wei = to_zen_dt(wei_dt);
    params.dtypes.dst = to_zen_dt(dst_dt);
    params.dtypes.bias = (bias ? to_zen_dt(bia_dt) : zd::none);
    // Static int8: leave compute unset (none) so ZenDNN infers s32 accumulation
    // for the AOCL-DLP int8 path (a non-none value selects dynamic-quant).
    params.dtypes.compute = zd::none;

    // 'r' = pre-packed, 'n' = plain weights.
    params.mem_format_b = mem_format_b;

    const char layout = 'r'; // row-major
    const bool trans_a = (transA != 'N');
    const bool trans_b = (transB != 'N');
    const float alpha = 1.f;
    // Prepacked weights are constant. WOQ (s4/u4) always arrives prepacked
    // (the pd rejects non-zen_packed WOQ weights), so mem_format_b=='r' already
    // covers it; ZenDNN's WOQ path relies on is_weights_const to cache the
    // dequantized/reordered weights.
    const bool is_weights_const = (mem_format_b == 'r');

    // Copy pre-built post-ops and patch binary buffer pointers.
    params.postop_ = cached_postops;
    for (size_t j = 0; j < params.postop_.size(); j++) {
        auto &lpo = params.postop_[j];
        if (lpo.po_type == zendnnl::ops::post_op_type_t::binary_add
                || lpo.po_type == zendnnl::ops::post_op_type_t::binary_mul) {
            lpo.buff = const_cast<void *>(CTX_IN_MEM(const void *,
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(cached_postop_po_indices[j])
                            | DNNL_ARG_SRC_1));
        }
    }

    // Static-quantization scales and (u8) source/destination zero-points.
    // Metadata (dtype, granularity dims) is resolved at primitive init(); only
    // the buffer pointers come from the execution context.
    // Weight zero-points: rejected for int8 static quant; populated for WOQ u4
    // (validated in pd_t::init_woq()).
    auto set_quant = [&](matmul_quantization_params_t::matmul_quant_t &q,
                             const zen_lowp_scale_info_t &s, const void *buff) {
        if (!s.present) return;
        q.buff = buff;
        q.dt = s.dt;
        q.dims = s.dims;
    };
    // src/wei scales pass through directly (ZenDNN multiplies, matching oneDNN);
    // dst scale uses the precomputed reciprocal buffer.
    set_quant(params.quant_params.src_scale, src_scale,
            CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC));
    set_quant(params.quant_params.wei_scale, wei_scale,
            CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS));
    if (dst_zp.present && !dst_scale.present) {
        // LOWOHA wires dst_zp through its destination SCALE post-op.
        static constexpr float unit_dst_scale = 1.f;
        params.quant_params.dst_scale.buff = &unit_dst_scale;
        params.quant_params.dst_scale.dt = zd::f32;
        params.quant_params.dst_scale.dims = {1};
    } else {
        set_quant(params.quant_params.dst_scale, dst_scale, dst_scale_recip);
    }
    set_quant(params.quant_params.src_zp, src_zp,
            CTX_IN_MEM(const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC));
    set_quant(params.quant_params.dst_zp, dst_zp,
            CTX_IN_MEM(const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST));
    // WOQ weight zero-point (u4 weights).
    set_quant(params.quant_params.wei_zp, wei_zp,
            CTX_IN_MEM(const void *,
                    DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS));

    const auto st = matmul_direct(layout, trans_a, trans_b, (int)M, (int)N,
            (int)K, alpha, A, (int)lda, B, (int)ldb, bias, cached_beta, C,
            (int)ldc, is_weights_const, batch, params);

    // Defensive: scrub buffer pointers (owned by the exec_ctx) before the
    // local params destructs.
    for (auto &lpo : params.postop_)
        lpo.buff = nullptr;
    params.quant_params.src_scale.buff = nullptr;
    params.quant_params.wei_scale.buff = nullptr;
    params.quant_params.dst_scale.buff = nullptr;
    params.quant_params.src_zp.buff = nullptr;
    params.quant_params.dst_zp.buff = nullptr;
    params.quant_params.wei_zp.buff = nullptr;

    return to_dnnl_status(st);
}

} // anonymous namespace
#endif // DNNL_X64_USE_ZEN

status_t zen_lowp_matmul_t::execute_body(const exec_ctx_t &ctx) const {
#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else
    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    if (src_d.has_zero_dim() || weights_d.has_zero_dim()
            || dst_d.has_zero_dim())
        return status::success;

    matmul_helper_t helper(src_d, weights_d, dst_d);

    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const char transA = helper.transA();
    const dim_t lda = helper.lda();
    // ZenDNN uses row-major dst, so ldc must be at least N.
    const dim_t ldc = nstl::max(N, helper.ldc());

    const bool wei_is_zen_packed = is_zen_packed(*pd()->weights_md(0));
    assert(!wei_is_zen_packed
            || weights_d.zen_packed_desc().gemm_src_dt == src_d.data_type());
    char mem_format_b = 'n';
    char transB;
    dim_t ldb;
    if (wei_is_zen_packed) {
        mem_format_b = 'r';
        transB = 'N';
        ldb = N;
    } else {
        transB = helper.transB();
        ldb = helper.ldb();
    }

    const auto bmm = compute_zen_lowp_bmm_params(
            src_d, weights_d, helper, wei_is_zen_packed);
    const int Batch_A = bmm.batch_a;
    const int Batch_B = bmm.batch_b;
    const size_t batch_stride_src = bmm.stride_src;
    const size_t batch_stride_wei = bmm.stride_wei;
    const size_t batch_stride_dst = bmm.stride_dst;

    assert(M <= std::numeric_limits<int>::max()
            && N <= std::numeric_limits<int>::max()
            && K <= std::numeric_limits<int>::max()
            && lda <= std::numeric_limits<int>::max()
            && ldb <= std::numeric_limits<int>::max()
            && ldc <= std::numeric_limits<int>::max());
    const size_t int_max_sz
            = static_cast<size_t>(std::numeric_limits<int>::max());
    assert(IMPLICATION(pd()->ndims() == 3,
            batch_stride_src <= int_max_sz && batch_stride_wei <= int_max_sz
                    && batch_stride_dst <= int_max_sz));
    MAYBE_UNUSED(int_max_sz);

    const auto src_dt = src_d.data_type();
    const auto wei_dt = weights_d.data_type();
    const auto dst_dt = dst_d.data_type();
    const auto bia_dt = pd()->with_bias() ? pd()->weights_md(1)->data_type
                                          : data_type::undef;

    VDEBUGINFO(2, primitive, matmul,
            "zen lowp matmul: M=%ld N=%ld K=%ld transA=%c transB=%c lda=%ld "
            "ldb=%ld ldc=%ld src_dt=%d wei_dt=%d dst_dt=%d Batch_A=%d "
            "Batch_B=%d",
            (long)M, (long)N, (long)K, transA, transB, (long)lda, (long)ldb,
            (long)ldc, (int)src_dt, (int)wei_dt, (int)dst_dt, Batch_A, Batch_B);

    const void *A = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const void *B = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *C = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *bias = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;

    // Precompute the reciprocal of the dst scale (oneDNN divides by the dst
    // scale; ZenDNN multiplies), storing it in the scratchpad booked in init().
    const void *dst_scale_recip = nullptr;
    if (dst_scale_.present) {
        const auto dsc_dt = pd()->attr()->scales_.get_data_type(DNNL_ARG_DST);
        const void *dsc
                = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
        void *recip = ctx.get_scratchpad_grantor().get<void>(
                memory_tracking::names::key_matmul_dst_scales);
        if (recip == nullptr) return status::out_of_memory;
        const float v = io::load_float_value(dsc_dt, dsc, 0);
        io::store_float_value(
                data_type::f32, v != 0.f ? 1.f / v : 0.f, recip, 0);
        dst_scale_recip = recip;
    }

    return zen_lowp_matmul_direct(src_dt, wei_dt, dst_dt, bia_dt, A, B, C, bias,
            M, N, K, lda, ldb, ldc, transA, transB, mem_format_b, Batch_A,
            Batch_B, batch_stride_src, batch_stride_wei, batch_stride_dst,
            zen_postop_, postop_indices_, beta_, src_scale_, wei_scale_,
            dst_scale_, dst_scale_recip, src_zp_, dst_zp_, wei_zp_, ctx);
#endif // DNNL_X64_USE_ZEN
}

} // namespace matmul
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
