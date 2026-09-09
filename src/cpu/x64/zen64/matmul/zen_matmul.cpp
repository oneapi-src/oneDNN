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

#include "cpu/x64/zen64/matmul/zen_matmul.hpp"

#include <assert.h>
#include <limits>

#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

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
using namespace zen_matmul;

namespace {

// Resolved Zen BMM batch parameters. For the non-batched (2D) path the batch
// counts stay 1 and the strides keep the (size_t)-1 sentinel (the backend then
// derives dense strides). For 3D the leading dim is the single batch dim and
// the per-batch strides are expressed in elements (matching the Zen looper,
// which scales them by the per-tensor element size).
struct zen_bmm_params_t {
    int batch_a = 1;
    int batch_b = 1;
    size_t stride_src = static_cast<size_t>(-1);
    size_t stride_wei = static_cast<size_t>(-1);
    size_t stride_dst = static_cast<size_t>(-1);
};

// Compute the BMM batch parameters from resolved (post set_default_formats)
// descriptors. Shared by pd_t::init (for INT_MAX validation) and execute_body
// (for the values actually handed to the backend) so the two never diverge.
inline zen_bmm_params_t compute_zen_bmm_params(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const matmul_helper_t &helper,
        bool wei_is_zen_packed) {
    zen_bmm_params_t p;
    if (src_d.ndims() != 3) return p; // 2D: defaults (batch 1, derive strides)

    p.batch_a = static_cast<int>(src_d.dims()[0]);
    p.batch_b = static_cast<int>(weights_d.dims()[0]);
    // oneDNN strides are already in elements.
    p.stride_src = static_cast<size_t>(helper.get_a_stride(0));
    p.stride_dst = static_cast<size_t>(helper.get_c_stride(0));
    if (wei_is_zen_packed) {
        // The packed weights md is opaque (no blocking_desc); each batch
        // occupies one fixed-size packed slot. Convert the per-slice byte size
        // to elements for the Zen looper. pd_t::init() has already verified that
        // per_slice_size is an exact multiple of the (non-zero) element size,
        // so this division is exact.
        const size_t wei_elem = weights_d.data_type_size();
        assert(wei_elem > 0
                && weights_d.zen_packed_desc().per_slice_size % wei_elem == 0);
        p.stride_wei = weights_d.zen_packed_desc().per_slice_size / wei_elem;
    } else {
        p.stride_wei = static_cast<size_t>(helper.get_b_stride(0));
    }
    return p;
}

} // namespace
#endif

status_t zen_matmul_t::pd_t::init(const engine_t *engine) {
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

    // Zen matmul requires AVX-512 core support regardless of data type.
    // Note: no separate avx512_core_bf16 check is needed here. On AMD CPUs,
    // AVX-512 first appeared with Zen 4, which shipped avx512_core and
    // avx512_core_bf16 together (and every later Zen generation does too).
    // Since avx512_core_bf16 is a superset of avx512_core, any AMD CPU that
    // satisfies avx512_core also supports avx512_core_bf16, so gating on
    // avx512_core alone is sufficient for the bf16 paths as well.
    VDISPATCH_MATMUL(mayiuse(avx512_core), VERBOSE_UNSUPPORTED_ISA);

    // ---- Memory descriptor data types ----
    const auto src_dt = src_md(0)->data_type;
    const auto wei_dt = weights_md(0)->data_type;
    const auto dst_dt = dst_md(0)->data_type;

    // 2D plain matmul and 3D batched matmul (a single leading batch dim) are
    // supported. Higher-rank batched matmul (ndims > 3, multiple batch dims)
    // maps to several batch strides which the Zen BMM looper -- a single
    // (Batch_A, Batch_B) batch index -- cannot express, so reject it here.
    VDISPATCH_MATMUL(
            utils::one_of(ndims(), 2, 3), VERBOSE_BAD_NDIMS, "dst", ndims());

    const bool is_batched = ndims() == 3;

    // No zero-dim tensors.
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    // ---- Batch (BMM) validation ----
    // For 3D the leading dim is the single batch dim. The Zen BMM looper
    // broadcasts an operand whose batch count is 1 (get_batch_index returns
    // b % batch, i.e. always 0 when batch == 1) and otherwise expects the
    // batch index to map 1:1. oneDNN only allows a batch dim to broadcast
    // when its extent is 1, so require the src and weights batch counts to be
    // equal or for one of them to broadcast (== 1). The output batch count is
    // max(src_batch, wei_batch), enforced by the framework on the dst desc.
    dim_t src_batch = 1, wei_batch = 1;
    if (is_batched) {
        src_batch = src_md(0)->dims[0];
        wei_batch = weights_md(0)->dims[0];
        VDISPATCH_MATMUL(
                src_batch == wei_batch || src_batch == 1 || wei_batch == 1,
                VERBOSE_INCONSISTENT_DIM, "src", 0, "weights", 0);
    }

    // ---- Datatype validation (aligned with Zen support) ----
    // Supported configurations:
    //  1. Uniform f32:  f32 src, f32 wei, f32 dst
    //  2. Uniform bf16: bf16 src, bf16 wei, bf16 dst
    //  3. bf16 mixed:   bf16 src, bf16 wei, f32 dst
    //  4. Uniform f16:  f16 src, f16 wei, f16 dst  (AVX512-FP16, acc-mode f16/any)
    //  5. f16 mixed:    f16 src, f16 wei, f32 dst   (AVX512-FP16, acc-mode f16/any)
    // Explicitly unsupported: f32 src with bf16/f16 dst.
    // int8 static quant is handled by the dedicated zen_lowp_matmul_t impl.
    const bool all_f32 = utils::everyone_is(f32, src_dt, wei_dt, dst_dt);
    const bool all_bf16 = utils::everyone_is(bf16, src_dt, wei_dt, dst_dt);
    const bool bf16_mixed
            = utils::everyone_is(bf16, src_dt, wei_dt) && dst_dt == f32;
    const bool all_f16 = utils::everyone_is(f16, src_dt, wei_dt, dst_dt);
    const bool f16_mixed
            = utils::everyone_is(f16, src_dt, wei_dt) && dst_dt == f32;
    VDISPATCH_MATMUL(utils::one_of(true, all_f32, all_bf16, bf16_mixed, all_f16,
                             f16_mixed),
            VERBOSE_UNSUPPORTED_DT_CFG);
    // Accumulation mode must match what ZenDNN/AOCL-DLP can execute for each
    // config. oneDNN keeps desc()->accum_data_type at the library default (f32
    // for f16 tensors); the user's choice is attr()->acc_mode_ (e.g. benchdnn
    // --attr-acc-mode=f16). Both uniform f16 (f16f16f16of16) and mixed
    // f16->f32 (f16f16f16of32) use native FP16 matmul accumulation on DLP;
    // accept f16 configs when acc-mode is f16 or any (fastest/low-precision
    // accumulation allowed). Default strict/f32 accumulation is left to brgemm.
    const auto acc_mode = attr()->acc_mode_;
    const bool is_f16_gemm = all_f16 || f16_mixed;
    const bool f16_acc_mode_ok = utils::one_of(
            acc_mode, accumulation_mode::f16, accumulation_mode::any);
    const bool acc_mode_ok = IMPLICATION(is_f16_gemm, f16_acc_mode_ok)
            && IMPLICATION(all_f32 || all_bf16 || bf16_mixed,
                    acc_mode != accumulation_mode::f16
                            && acc_mode != accumulation_mode::s32);
    VDISPATCH_MATMUL(acc_mode_ok, VERBOSE_UNSUPPORTED_ATTR);
    // F16 requires AVX512-FP16 (Zen5+). Gate here so f32/bf16 configs on
    // Zen4 still dispatch to this impl without needing FP16 ISA.
    VDISPATCH_MATMUL(
            IMPLICATION(all_f16 || f16_mixed, mayiuse(avx512_core_fp16)),
            VERBOSE_UNSUPPORTED_ISA);

    // ---- Bias validation ----
    // Zen supports bias with matching/compatible dtypes;
    // bias must follow 1xN broadcast pattern.
    auto check_bias = [&]() -> bool {
        if (!with_bias()) return true;
        const auto bia_dt = weights_md(1)->data_type;
        const bool bia_dt_ok = IMPLICATION(all_f32, bia_dt == f32)
                && IMPLICATION(all_bf16 || bf16_mixed,
                        utils::one_of(bia_dt, bf16, f32))
                && IMPLICATION(
                        all_f16 || f16_mixed, utils::one_of(bia_dt, f16, f32));
        return bia_dt_ok && is_bias_1xN();
    };
    VDISPATCH_MATMUL(check_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    // ---- Attribute validation ----
    // For f32/bf16/f16: post-ops (eltwise + binary + sum) are supported;
    // fpmath_mode, scales and zero-points must be default. accumulation_mode
    // is validated explicitly above (f16 configs require acc-mode f16 or any).
    VDISPATCH_MATMUL(
            attr()->has_default_values(smask_t::post_ops | smask_t::sum_dt
                            | smask_t::accumulation_mode,
                    dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    // Sum-consistency check (catches sum.dt != dst_dt precision bugs).
    VDISPATCH_MATMUL(
            attr()->post_ops_.check_sum_consistency(dst_dt, /*is_int8=*/false),
            VERBOSE_UNSUPPORTED_POSTOP);

    // ---- Post-ops validation ----
    // Zen supports: sum, eltwise (relu, gelu_tanh, gelu_erf, tanh,
    // sigmoid/logistic, swish), binary (add, mul).
    auto check_postops = [&]() -> bool {
        const auto &po = attr()->post_ops_;
        for (int i = 0; i < po.len(); i++) {
            const auto &entry = po.entry_[i];
            if (entry.is_sum(/*require_scale_one=*/false,
                        /*require_zp_zero=*/true)) {
                // Sum maps to plain beta accumulation and must be the very
                // first post-op; at any later position Zen has already
                // consumed the destination, so it cannot be honored.
                if (i != 0) return false;
                // Sum maps to plain beta accumulation, which reads the
                // destination in its native dtype. A sum.dt that differs
                // from dst_dt asks the destination bytes to be
                // reinterpreted as that dtype before accumulation (e.g.
                // f32 dst read as s32); Zen cannot honor that, so only
                // accept a default sum.dt or one matching dst_dt.
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
                // Zen maps eltwise_relu to a plain ReLU (slope 0); it cannot
                // honor a leaky-relu negative slope, so reject alpha != 0.
                if (entry.eltwise.alg == eltwise_relu
                        && entry.eltwise.alpha != 0.f)
                    return false;
            } else if (entry.is_binary()) {
                using namespace alg_kind;
                // Binary post-ops are supported on both the 2D and 3D (BMM)
                // paths. On BMM the Zen backend advances the src1 buffer per
                // batch by a dense M*N stride and per row by N, so the operand
                // must be a dense row-major tensor that is full in M and N;
                // the shape/stride contract is enforced in
                // check_binary_postop_formats() below.
                if (!utils::one_of(entry.binary.alg, binary_add, binary_mul))
                    return false;
                const auto src1_dt = entry.binary.src1_desc.data_type;
                if (!utils::one_of(src1_dt, f32, bf16, f16)) return false;
                // F16 binary operands require AVX512-FP16 (Zen5+); without it
                // the runtime cannot allocate/process f16 post-op buffers even
                // when the matmul itself is f32/bf16.
                if (src1_dt == f16 && !mayiuse(avx512_core_fp16)) return false;
            } else {
                // Unsupported post-op kind.
                return false;
            }
        }
        return true;
    };
    VDISPATCH_MATMUL(check_postops(), VERBOSE_UNSUPPORTED_POSTOP);

    // ---- Scales / zero-points validation ----
    // f32/bf16/f16 path does not support scales or zero-points.
    VDISPATCH_MATMUL(attr()->scales_.has_default_values(),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_MATMUL(attr()->zero_points_.has_default_values(),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    // Zen matmul_direct uses int for M/N/K; reject runtime dims/strides
    // before set_default_formats() to avoid undefined behavior.
    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Zen f32/bf16/f16 matmul: prepack path. When the framework leaves the
    // weights layout open (format_any) we advertise the dedicated opaque
    // `format_kind::zen_packed` weights format. The bytes are produced by
    // zen_reorder_t (the Zen backend packer) and consumed directly by the
    // backend (mem_format_b='r'), so no oneDNN blocked layout is involved.
    const bool wei_format_any = memory_desc_wrapper(weights_md(0)).format_any();

    // The caller may pass back a weights descriptor that is already in the
    // opaque zen_packed format (e.g. one obtained from a previous query),
    // in which case it must be accepted as-is: no re-packing is needed and
    // the plain-weights paths below (GEMM-format check, matmul_helper_t::ldb)
    // must be skipped since the descriptor has no blocking_desc.
    const bool wei_already_packed = zen::is_zen_packed(*weights_md(0));

    // Resolve format_any memory descriptors to concrete (dense) formats first.
    // For the prepack path this gives the weights a plain blocked layout
    // (dims/padded_dims set), which we then convert in-place to the opaque
    // packed format.
    // An already-packed (opaque) descriptor is not format_any, so
    // set_default_formats() leaves it untouched.
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    // Advertise the opaque zen_packed weights format whenever the framework
    // leaves the weights layout open. For BMM (3D) the packed buffer holds one
    // packed (K, N) slice per weight batch (wei_batch back-to-back slots),
    // produced by zen_reorder_t and consumed here with mem_format_b='r' plus a
    // per-slice batch_stride_wei.
    bool wei_zen_packed = wei_already_packed;
    if (wei_format_any && utils::one_of(wei_dt, f32, bf16, f16)) {
        VDISPATCH_MATMUL_SC(zen::init_zen_packed_md(weights_md_, src_dt, K(),
                                    N(), is_batched ? wei_batch : 1),
                VERBOSE_UNSUPPORTED_TAG);
        wei_zen_packed = true;
    }

    // The Zen prepacked path stores weights in the opaque zen_packed
    // format that gemm_based's plain-weights compatibility check rejects; the
    // backend consumes the packed buffer directly (mem_format_b='r'), so the
    // gemm-format check only applies to the plain weights path.
    VDISPATCH_MATMUL(
            wei_zen_packed || gemm_based::check_gemm_compatible_formats(*this),
            VERBOSE_INCOMPATIBLE_GEMM_FMT);

    // Source / destination layout validation.
    //
    // The Zen backend drives the GEMM via explicit leading dims and per-batch
    // strides, so it accepts a padded leading dim: src plain with one inner
    // axis contiguous, dst plain with the last (N) axis contiguous.
    // check_gemm_{input,output}_format() capture exactly this contract, so an
    // exact matches_one_of_tag() check would needlessly reject supported
    // padded layouts. Applied unconditionally (including the packed-weights
    // path) so an unsupported layout declines at creation, not at runtime.
    const memory_desc_wrapper dst_d(dst_md(0));
    VDISPATCH_MATMUL(gemm_based::check_gemm_output_format(*dst_md(0)),
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(gemm_based::check_gemm_input_format(*src_md(0)),
            VERBOSE_UNSUPPORTED_TAG);

    // check_gemm_output_format() only enforces plain + contiguous N; unlike
    // check_gemm_input_format() it does not reject zero strides. A zero stride
    // on dst (an M- or batch-broadcast output) would make the Zen kernel write
    // repeatedly to the same location and produce wrong results, so reject it
    // here explicitly (is_plain() is guaranteed by the check above).
    const auto &dst_strides = dst_d.blocking_desc().strides;
    bool dst_no_zero_stride = true;
    for (int i = 0; i < dst_d.ndims(); i++)
        dst_no_zero_stride = dst_no_zero_stride && dst_strides[i] != 0;
    VDISPATCH_MATMUL(dst_no_zero_stride, VERBOSE_UNSUPPORTED_TAG);

    // For BMM the Zen looper treats dim 0 as the single batch dimension and
    // advances each operand by its dim-0 (batch) stride. check_gemm_*_format()
    // above only constrains the inner M/K axes, so it would also accept a
    // permuted layout where the batch dim is not outermost. To
    // keep the accepted layouts matched to the abc/acb (batch-outermost)
    // contract, require dim 0 to carry the largest stride for every batched
    // operand that exposes strides (packed weights are opaque and validated via
    // their per-slice size instead).
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
        if (!wei_zen_packed)
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
    // The src1 operand must be a plain row-major dense tensor (ab for 2D, abc
    // for 3D); the Zen backend does not support arbitrary strides. The channel
    // (last/N) dim must always be full-size. Supported broadcast patterns:
    //   2D:  * per_tensor  : dims == dst (M, N)
    //        * per_channel : M broadcast (1, N) -- applied via the non-batched
    //                        path, which honors a 1-row operand.
    //   3D:  * per_tensor  : dims == dst (batch, M, N)
    //        * batch bcast : (1, M, N) -- reused across batches.
    //        M must be full on the BMM path: the Zen looper applies a fixed
    //        per-row (m_start*N) and per-batch (M*N) offset, so a row-broadcast
    //        operand would be read out of bounds. The batch dim may be full or
    //        broadcast (1).
    auto check_binary_postop_formats = [&]() -> bool {
        const auto &po = attr()->post_ops_;
        for (int i = 0; i < po.len(); i++) {
            const auto &entry = po.entry_[i];
            if (!entry.is_binary()) continue;

            const auto &src1_desc = entry.binary.src1_desc;
            const auto *dst = dst_md(0);

            if (src1_desc.ndims != dst->ndims) return false;
            const int nd = src1_desc.ndims;
            const int channel_dim = nd - 1; // N
            for (int d = 0; d < nd; d++) {
                const bool full = src1_desc.dims[d] == dst->dims[d];
                const bool bcast = src1_desc.dims[d] == 1;
                if (!(full || bcast)) return false;
                if (d == channel_dim && !full) return false;
            }
            // On the BMM (3D) path the M dim (nd-2) must be full; the Zen looper
            // offsets the operand by m_start*N rows, so a 1-row (M-broadcast)
            // operand cannot be supported there.
            if (nd == 3 && src1_desc.dims[nd - 2] != dst->dims[nd - 2])
                return false;

            // Plain row-major dense: last dim contiguous, row stride == N, and
            // (3D, batch full) batch stride == M*N.
            const memory_desc_wrapper src1_mdw(src1_desc);
            if (!src1_mdw.is_plain()) return false;
            const auto &strides = src1_mdw.blocking_desc().strides;
            if (strides[nd - 1] != 1) return false;
            if (strides[nd - 2] != src1_desc.dims[nd - 1]) return false;
            if (nd == 3 && src1_desc.dims[0] != 1) {
                // Compute M*N in size_t with an overflow guard: an oversized
                // product would wrap in signed dim_t and invoke UB even though
                // such a case is rejected later by the INT_MAX guard.
                const size_t m = static_cast<size_t>(src1_desc.dims[1]);
                const size_t n = static_cast<size_t>(src1_desc.dims[2]);
                if (n != 0 && m > std::numeric_limits<size_t>::max() / n)
                    return false;
                const size_t batch_stride = m * n;
                if (static_cast<size_t>(strides[0]) != batch_stride)
                    return false;
            }
        }
        return true;
    };
    VDISPATCH_MATMUL(check_binary_postop_formats(), VERBOSE_UNSUPPORTED_POSTOP);

    // Zen matmul_direct uses int for M/N/K and leading dimensions.
    // Reject descriptors whose dimensions or strides exceed INT_MAX.
    const matmul_helper_t helper(memory_desc_wrapper(src_md(0)),
            memory_desc_wrapper(weights_md(0)), memory_desc_wrapper(dst_md(0)));
    const dim_t int_max = std::numeric_limits<int>::max();
    // For the packed path the weights md is opaque (no blocking_desc), and the
    // backend uses ldb = N; otherwise read ldb from the plain weights strides.
    const dim_t wei_ldb = wei_zen_packed ? N() : helper.ldb();
    // The Zen BMM looper takes Batch_A/Batch_B as int; the batch count is
    // max(src_batch, wei_batch). Guard all three against the int API.
    const dim_t batch_count = src_batch > wei_batch ? src_batch : wei_batch;
    bool fits_zen_int_api = helper.M() <= int_max && helper.N() <= int_max
            && helper.K() <= int_max && helper.lda() <= int_max
            && wei_ldb <= int_max && helper.ldc() <= int_max
            && src_batch <= int_max && wei_batch <= int_max
            && batch_count <= int_max;

    // For batched pre-packed weights the per-batch weight stride is derived as
    // per_slice_size / wei_elem (the opaque md has no blocking_desc). Reject a
    // descriptor whose per-slice byte size is not an exact multiple of the
    // weight element size: the integer division would otherwise truncate and
    // point the Zen looper at the wrong packed slot (silently wrong results).
    if (is_batched && wei_zen_packed) {
        const memory_desc_wrapper wei_d(weights_md(0));
        const size_t wei_elem = wei_d.data_type_size();
        const size_t per_slice = wei_d.zen_packed_desc().per_slice_size;
        VDISPATCH_MATMUL(wei_elem > 0 && per_slice % wei_elem == 0,
                VERBOSE_INCONSISTENT_MDS, "weights", "packed-slice-size");
    }

    // The per-batch strides handed to the Zen looper must also fit the int API
    // (defense in depth: they are size_t in the backend but derived from dims).
    // Only meaningful for the batched path; 2D keeps the (size_t)-1 sentinel.
    // Guard the call behind fits_zen_int_api: compute_zen_bmm_params() casts the
    // batch dims to int, so it must not run when the batch/dim validation above
    // has already failed (batch dims > INT_MAX would be an out-of-range,
    // implementation-defined integral conversion). When fits_zen_int_api is
    // already false the VDISPATCH below rejects the primitive regardless.
    if (is_batched && fits_zen_int_api) {
        const auto bmm = compute_zen_bmm_params(memory_desc_wrapper(src_md(0)),
                memory_desc_wrapper(weights_md(0)), helper, wei_zen_packed);
        const size_t int_max_sz = static_cast<size_t>(int_max);
        fits_zen_int_api = fits_zen_int_api && bmm.stride_src <= int_max_sz
                && bmm.stride_wei <= int_max_sz && bmm.stride_dst <= int_max_sz;
    }
    VDISPATCH_MATMUL(fits_zen_int_api, VERBOSE_UNSUPPORTED_FEATURE,
            "dimension/stride > INT_MAX is not supported");

    return status::success;
#endif // DNNL_X64_USE_ZEN
}

status_t zen_matmul_t::init(engine_t *engine) {
    MAYBE_UNUSED(engine);
#if DNNL_X64_USE_ZEN
    // Build Zen matmul_post_op chain directly from oneDNN attributes.
    // Static parts (type, alpha, beta, dtype, dims) are set here; only
    // binary buffer pointers are patched at execute() time.
    //
    // The chain is owned by the primitive (not pd_t) so that pd_t remains
    // cheaply-copyable by the framework's primitive cache, matching the
    // brgemm_matmul_t convention.
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
        if (entry.is_sum(/*require_scale_one=*/false,
                    /*require_zp_zero=*/true)) {
            // Sum maps to Zen beta (C = alpha*A*B + beta*C).
            beta_ = entry.sum.scale;
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
                case f16: lpo.dtype = zd::f16; break;
                default: return status::runtime_error;
            }
            const auto &src1_desc = entry.binary.src1_desc;
            lpo.dims.assign(src1_desc.dims, src1_desc.dims + src1_desc.ndims);
            // buff pointer will be patched at execute() time
        }
        zen_postop_.push_back(lpo);
        postop_indices_.push_back(i);
    }
#endif // DNNL_X64_USE_ZEN
    return status::success;
}

// ================================================================
// Zen helpers and wrapper (translation-unit local).
// ================================================================
#if DNNL_X64_USE_ZEN
namespace {

// Unified Zen direct MatMul launcher.
// - Each tensor (src, wei, dst, bias) gets its own data type from the
//   oneDNN memory descriptors, supporting mixed-precision configs
//   (e.g. bf16 src/wei -> f32 dst, or f32 bias with bf16 compute).
// - transA/transB, lda/ldb/ldc are derived from matmul_helper_t.
// - Post-ops are pre-built at zen_matmul_t::init(engine_t*); only
//   binary buffer pointers are patched here from the execution context.
status_t zen_matmul_direct(data_type_t src_dt, data_type_t wei_dt,
        data_type_t dst_dt, data_type_t bia_dt, const void *A, const void *B,
        void *C, const void *bias, dim_t M, dim_t N, dim_t K, dim_t lda,
        dim_t ldb, dim_t ldc, char transA, char transB, char mem_format_b,
        int Batch_A, int Batch_B, size_t batch_stride_src,
        size_t batch_stride_wei, size_t batch_stride_dst,
        const std::vector<matmul_post_op> &cached_postops,
        const std::vector<int> &cached_postop_po_indices, float cached_beta,
        const exec_ctx_t &ctx) {
    using zd = zendnnl::common::data_type_t;

    // Batch_A/Batch_B default to 1 (single GEMM) for the non-batched path.
    // For BMM the strides are expressed in elements (the Zen looper scales
    // them by the per-tensor type size); a sentinel of (size_t)-1 lets the
    // backend derive the dense stride from the dimensions.
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
    // pd_t::init accepts f16 src/wei only with acc-mode f16 or any (uniform and
    // mixed); GEMM backend accumulates the matmul product in native FP16 only.
    params.dtypes.compute
            = utils::everyone_is(f16, src_dt, wei_dt) ? zd::f16 : zd::f32;

    // 'r' = pre-packed, 'n' = plain weights;
    params.mem_format_b = mem_format_b;

    const char layout = 'r'; // row-major
    const bool trans_a = (transA != 'N');
    const bool trans_b = (transB != 'N');
    const float alpha = 1.f;
    const bool is_weights_const = (mem_format_b == 'r');

    // Copy pre-built post-ops and patch binary buffer pointers (only
    // available at execute time from the execution context).
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

    const auto st = matmul_direct(layout, trans_a, trans_b, (int)M, (int)N,
            (int)K, alpha, A, (int)lda, B, (int)ldb, bias, cached_beta, C,
            (int)ldc, is_weights_const, batch, params);

    // Defensive: scrub binary buffer pointers before the local `params`
    // (and its `params.postop_` vector) destructs. The buffers belong to
    // the oneDNN exec_ctx, not to matmul_post_op; leaving them set would
    // be a double-free hazard if matmul_post_op ever gained an owning dtor.
    for (auto &lpo : params.postop_)
        lpo.buff = nullptr;

    return to_dnnl_status(st);
}

} // anonymous namespace
#endif // DNNL_X64_USE_ZEN

status_t zen_matmul_t::execute_body(const exec_ctx_t &ctx) const {
#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else
    // Build memory_desc_wrappers (needed by matmul_helper_t).
    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    if (src_d.has_zero_dim() || weights_d.has_zero_dim()
            || dst_d.has_zero_dim())
        return status::success;

    // Use matmul_helper_t to derive M, N, K, transA/B, lda/b/c (handles
    // arbitrary layouts and transpositions correctly).
    matmul_helper_t helper(src_d, weights_d, dst_d);

    // M, N, K and the src/dst leading dims come from src/dst descriptors and
    // are always valid. transB/ldb depend on the weights layout and are
    // resolved below (the packed weights md is opaque, so it has no
    // blocking_desc for matmul_helper_t to read).
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const char transA = helper.transA();
    const dim_t lda = helper.lda();
    // ZenDNN uses row-major dst, so ldc must be at least N.
    const dim_t ldc = nstl::max(N, helper.ldc());

    // If weights are Zen pre-packed (opaque format_kind::zen_packed,
    // produced by zen_reorder_t), the backend expects:
    //   - mem_format_b = 'r'  (pre-packed)
    //   - transB       = 'N'  (logical orientation is K×N, matching trans='n')
    //   - ldb          = N    (plain K×N leading dim)
    // For plain `ab`/`ba` weights we use the helper-derived values and let
    // the backend pack them itself with mem_format_b='n'.
    const bool wei_is_zen_packed = is_zen_packed(*pd()->weights_md(0));
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

    // Batch (BMM) parameters (shared with pd_t::init via compute_zen_bmm_params
    // so validation and execution agree). The non-batched path keeps Batch == 1
    // and the (size_t)-1 stride sentinel (the backend derives dense strides).
    const auto bmm = compute_zen_bmm_params(
            src_d, weights_d, helper, wei_is_zen_packed);
    const int Batch_A = bmm.batch_a;
    const int Batch_B = bmm.batch_b;
    const size_t batch_stride_src = bmm.stride_src;
    const size_t batch_stride_wei = bmm.stride_wei;
    const size_t batch_stride_dst = bmm.stride_dst;

    // pd_t::init() rejects dimensions/strides above INT_MAX; assert here as a
    // defensive invariant before casting to the int-based Zen API. The batch
    // strides keep their (size_t)-1 sentinel on the 2D path, so only assert
    // them for the batched path.
    const size_t int_max_sz
            = static_cast<size_t>(std::numeric_limits<int>::max());
    assert(M <= std::numeric_limits<int>::max()
            && N <= std::numeric_limits<int>::max()
            && K <= std::numeric_limits<int>::max()
            && lda <= std::numeric_limits<int>::max()
            && ldb <= std::numeric_limits<int>::max()
            && ldc <= std::numeric_limits<int>::max());
    assert(IMPLICATION(pd()->ndims() == 3,
            batch_stride_src <= int_max_sz && batch_stride_wei <= int_max_sz
                    && batch_stride_dst <= int_max_sz));
    MAYBE_UNUSED(int_max_sz);

    // Per-tensor data types (may differ for mixed-precision configs).
    const auto src_dt = src_d.data_type();
    const auto wei_dt = weights_d.data_type();
    const auto dst_dt = dst_d.data_type();
    const auto bia_dt = pd()->with_bias() ? pd()->weights_md(1)->data_type
                                          : data_type::undef;

    VDEBUGINFO(2, primitive, matmul,
            "zen matmul: M=%ld N=%ld K=%ld transA=%c transB=%c lda=%ld "
            "ldb=%ld ldc=%ld src_dt=%d wei_dt=%d dst_dt=%d Batch_A=%d "
            "Batch_B=%d",
            (long)M, (long)N, (long)K, transA, transB, (long)lda, (long)ldb,
            (long)ldc, (int)src_dt, (int)wei_dt, (int)dst_dt, Batch_A, Batch_B);

    // Extract raw pointers.
    const void *A = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const void *B = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *C = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *bias = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;

    // Dispatch to Zen with per-tensor data types and layout info.
    // Post-ops were pre-built at primitive init(); only binary buffer
    // pointers are patched here from the execution context.
    return zen_matmul_direct(src_dt, wei_dt, dst_dt, bia_dt, A, B, C, bias, M,
            N, K, lda, ldb, ldc, transA, transB, mem_format_b, Batch_A, Batch_B,
            batch_stride_src, batch_stride_wei, batch_stride_dst, zen_postop_,
            postop_indices_, beta_, ctx);
#endif // DNNL_X64_USE_ZEN
}

} // namespace matmul
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
