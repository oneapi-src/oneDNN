/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#include "cpu/x64/zen64/reorder/zen_reorder.hpp"

#include "common/c_types_map.hpp"
#include "common/int4.hpp"
#include "common/memory_desc.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/nibble.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include <cstdint>
#include <cstring>
#include <limits>

#include "cpu/ref_io_helper.hpp"
#include "cpu/x64/cpu_isa_traits.hpp" // cpu().has(tAMD)
#include "cpu/x64/zen64/common/zen_format_tag.hpp" // is_zen_packed

#if DNNL_X64_USE_ZEN
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include "zendnnl.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace reorder {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::status;

#if DNNL_X64_USE_ZEN
namespace {

using zd = zendnnl::common::data_type_t;
using zendnnl::ops::matmul_algo_t;

// One (K, N) slice through reorder_direct. `unpack` selects unreorder;
// otherwise this is the prepack. src_dt is the matmul source dtype: it equals
// wei_dt for f32/bf16/f16, and may differ for int8 (u8 source, s8 weights).
status_t zen_weight_reorder(const void *src, void *dst, zd wei_dt, zd src_dt,
        int64_t K, int64_t N, int64_t ldb, bool transposed,
        bool unpack = false) {
    zendnnl::lowoha::reorder::reorder_params_t rp;
    rp.is_prepack = !unpack;
    rp.is_unreorder = unpack;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = wei_dt;
    rp.prepack.src_dtype = src_dt;
    rp.prepack.K = K;
    rp.prepack.N = N;
    rp.prepack.ldb = ldb;
    rp.prepack.transposed = transposed;

    return to_dnnl_status(
            zendnnl::lowoha::reorder::reorder_direct(src, dst, rp));
}

// Dense plain destination of exactly K*N elements per slice: row-major
// (`ab`/`abc`) or column-major (`ba`/`acb`).
struct unpack_dst_layout_t {
    bool transposed = false;
    int64_t ldb = 0;
    int64_t batch_stride = 0;
};

bool get_unpack_dst_layout(const memory_desc_wrapper &od, int kdim, int ndim,
        unpack_dst_layout_t &l) {
    if (!od.is_plain() || od.extra().flags != memory_extra_flags::none)
        return false;
    for (int d = 0; d < od.ndims(); d++)
        if (od.padded_dims()[d] != od.dims()[d]) return false;

    const auto &strides = od.blocking_desc().strides;
    const dim_t K = od.dims()[kdim];
    const dim_t N = od.dims()[ndim];
    if (strides[ndim] == 1 && (K == 1 || strides[kdim] == N)) {
        l.transposed = false;
        l.ldb = N;
    } else if (strides[kdim] == 1 && (N == 1 || strides[ndim] == K)) {
        l.transposed = true;
        l.ldb = K;
    } else {
        return false;
    }

    l.batch_stride = 0;
    if (od.ndims() == 3 && od.dims()[0] > 1) {
        if (strides[0] != K * N) return false;
        l.batch_stride = strides[0];
    }
    return true;
}

// Plain element conversion between f32 and bf16/f16 (standard reorder_direct
// path). Writes K rows of N elements contiguously into `dst` (row-major).
//
// `ldb` is the source leading dim (in elements), read from the actual src
// strides so a padded leading dim is honored:
//   * ab (row-major): src_strides = {ldb, 1}   (ldb == N for dense, N+pad if padded)
//   * ba (col-major): src_strides = {1, ldb}    (ldb == K for dense, K+pad if padded)
// The destination is always written contiguous (lowoha reorder ignores
// dst_strides), so dst ends up in `ab` (K-major) regardless of src.
status_t convert_plain(const void *src, void *dst, zd src_dt, zd dst_dt,
        int64_t K, int64_t N, int64_t ldb, bool src_is_ab) {
    zendnnl::lowoha::reorder::reorder_params_t rp;
    rp.src_dtype = src_dt;
    rp.dst_dtype = dst_dt;
    rp.src_shape = {K, N};
    rp.dst_shape = {K, N};
    if (src_is_ab)
        rp.src_strides = {ldb, 1};
    else
        rp.src_strides = {1, ldb};

    return to_dnnl_status(
            zendnnl::lowoha::reorder::reorder_direct(src, dst, rp));
}

// Plain f32 -> s8 element conversion (saturating round-to-nearest). Writes K
// rows of N elements contiguously into `dst` (row-major `ab`). `ldb` is the
// source leading dim (in elements), read from the actual src strides so a
// padded leading dim is honored:
//   * ab (row-major): src_strides = {ldb, 1}  -> element (k,n) at k*ldb + n
//   * ba (col-major): src_strides = {1, ldb}  -> element (k,n) at k + n*ldb
// Used by the int8 prepack path when the caller supplies an f32 weight
// reference (e.g. benchdnn keeps integer-valued weights in f32); the values
// are exact int8s so the cast is lossless. No quantization scale is applied
// here -- the matmul applies the weight scale.
status_t f32_to_s8_plain(const void *src, void *dst, int64_t K, int64_t N,
        int64_t ldb, bool src_is_ab) {
    const float *s = static_cast<const float *>(src);
    int8_t *d = static_cast<int8_t *>(dst);
    for (int64_t k = 0; k < K; k++) {
        for (int64_t n = 0; n < N; n++) {
            const int64_t soff = src_is_ab ? (k * ldb + n) : (k + n * ldb);
            const float v = io::load_float_value(
                    data_type::f32, s, static_cast<dim_t>(soff));
            io::store_float_value(
                    data_type::s8, v, d, static_cast<dim_t>(k * N + n));
        }
    }
    return status::success;
}

// Plain f32 -> packed s4/u4 element conversion (saturating round-to-nearest),
// writing a contiguous row-major (`ab`) K*N nibble stream (2 values per byte,
// low nibble = even element index -- oneDNN's nibble packing). Used by the WOQ
// prepack path when the caller supplies an f32 weight reference (e.g. benchdnn
// keeps the integer-valued 4-bit codes in f32). No scale is applied -- the
// matmul applies the weight scale. `out_dt` is s4 or u4.
status_t f32_to_int4_plain(const void *src, void *dst, int64_t K, int64_t N,
        int64_t ldb, bool src_is_ab, data_type_t out_dt) {
    // NOTE: io::store_float_value() has no s4/u4 case (it hits default:
    // assert(!"bad data_type") in debug builds), so convert with the
    // reference reorder's saturate-and-round semantics, then pack the
    // resulting 4-bit value directly via nibble2_t.
    const float *s = static_cast<const float *>(src);
    auto *d = reinterpret_cast<nibble2_t *>(dst);
    for (int64_t k = 0; k < K; k++) {
        for (int64_t n = 0; n < N; n++) {
            const int64_t soff = src_is_ab ? (k * ldb + n) : (k + n * ldb);
            const float v = io::load_float_value(
                    data_type::f32, s, static_cast<dim_t>(soff));
            const uint8_t raw = out_dt == data_type::u4
                    ? q10n::saturate_and_round<uint4_t>(v).raw_bits_
                    : q10n::saturate_and_round<int4_t>(v).raw_bits_;
            const int64_t oidx = k * N + n;
            const int nibble = static_cast<int>(oidx % 2);
            nibble2_t pair(0);
            // An even element starts a new byte. Preserve the existing low
            // nibble only when writing the following odd element.
            if (nibble != 0) pair = d[oidx / 2];
            pair.set(raw, nibble);
            d[oidx / 2] = pair;
        }
    }
    return status::success;
}

// Conversions between f32 and bf16/f16, done by convert_plain().
bool is_float_conversion(data_type_t src_dt, data_type_t dst_dt) {
    return (src_dt == data_type::f32
                   && utils::one_of(dst_dt, data_type::bf16, data_type::f16))
            || (dst_dt == data_type::f32
                    && utils::one_of(src_dt, data_type::bf16, data_type::f16));
}

// Prepack paths that convert each (K, N) slice into a plain buffer of the
// packed dtype before packing it.
bool prepack_converts(data_type_t src_dt, data_type_t dst_dt) {
    return is_float_conversion(src_dt, dst_dt)
            || (src_dt == data_type::f32
                    && utils::one_of(dst_dt, data_type::s8, data_type::s4,
                            data_type::u4));
}

} // namespace
#endif

status_t zen_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else

    VDISPATCH_REORDER(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    VDISPATCH_REORDER(src_engine->kind() == engine_kind::cpu
                    && dst_engine->kind() == engine_kind::cpu,
            VERBOSE_UNSUPPORTED_FEATURE, "non-CPU engine");

    VDISPATCH_REORDER(::dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tAMD),
            "This implementation only supports AMD CPUs");

    // Zen weight prepack requires AVX-512 core support for f32/bf16. F16
    // prepack additionally requires avx512_core_fp16 (AVX512-FP16, Zen5+).
    VDISPATCH_REORDER(mayiuse(avx512_core), VERBOSE_UNSUPPORTED_ISA);

    // Packed to an identical packed descriptor (a clone) is a byte copy.
    // Descriptor equality includes the recorded packing parameters, so
    // incompatible packs are declined.
    if (is_zen_packed(src_md_) && is_zen_packed(dst_md_)) {
        VDISPATCH_REORDER(
                *src_md() == *dst_md(), VERBOSE_INCONSISTENT_MDS, "src", "dst");
        VDISPATCH_REORDER(
                attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
        return status::success;
    }

    // zen_packed src is the unpack direction; below is prepack into zen_packed.
    if (is_zen_packed(src_md_)) return init_unpack(engine);

    const memory_desc_wrapper id(src_md_), od(dst_md_);

    // 2D weight slice, or 3D batched weights (one (K, N) slice per batch).
    VDISPATCH_REORDER(
            utils::one_of(id.ndims(), 2, 3) && id.ndims() == od.ndims(),
            VERBOSE_BAD_NDIMS, "src/dst", id.ndims());

    const int ndims = id.ndims();
    const bool batched = ndims == 3;

    const auto type_i = id.data_type();
    const auto type_o = od.data_type();
    // Supported dtype combos:
    //   bf16 -> bf16  : reorder_direct prepack (Zen blocked algo)
    //   f16  -> f16   : reorder_direct prepack (Zen blocked algo)
    //   f32  -> f32   : reorder_direct prepack (Zen blocked algo)
    //   f32  -> bf16  : f32->bf16 plain reorder_direct, then bf16 prepack
    //                   (avoids the backend f32->bf16 fringe-N bug; see execute)
    //   f32  -> f16   : f32->f16 plain reorder_direct, then f16 prepack
    //                   (same two-step pattern as f32->bf16)
    //   bf16 -> f32   : widen to plain f32, then f32 prepack
    //   f16  -> f32   : widen to plain f32, then f32 prepack
    //   s8   -> s8    : int8 static-quant weight prepack (Zen blocked algo)
    //   f32  -> s8    : f32->s8 plain cast, then s8 prepack (for callers that
    //                   keep integer-valued weights in f32, e.g. benchdnn)
    //   s4   -> s4    : WOQ 4-bit weight prepack (bf16 gemm source)
    //   u4   -> u4    : WOQ 4-bit weight prepack (bf16 gemm source)
    //   f32  -> s4/u4 : f32->4-bit plain cast, then WOQ prepack (for callers
    //                   that keep integer-valued 4-bit codes in f32, e.g.
    //                   benchdnn)
    const bool dt_ok = (type_i == data_type::bf16 && type_o == data_type::bf16)
            || (type_i == data_type::f16 && type_o == data_type::f16)
            || (type_i == data_type::f32 && type_o == data_type::f32)
            || (type_i == data_type::f32 && type_o == data_type::bf16)
            || (type_i == data_type::f32 && type_o == data_type::f16)
            || (utils::one_of(type_i, data_type::bf16, data_type::f16)
                    && type_o == data_type::f32)
            || (type_i == data_type::s8 && type_o == data_type::s8)
            || (type_i == data_type::f32 && type_o == data_type::s8)
            || (type_i == data_type::s4 && type_o == data_type::s4)
            || (type_i == data_type::u4 && type_o == data_type::u4)
            || (type_i == data_type::f32
                    && utils::one_of(type_o, data_type::s4, data_type::u4));
    VDISPATCH_REORDER(dt_ok, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(
            IMPLICATION(type_o == data_type::f16, mayiuse(avx512_core_fp16)),
            VERBOSE_UNSUPPORTED_ISA);

    // Dispatch trigger: only fire when the dst uses the dedicated opaque
    // Zen packed format; otherwise let the regular reorder list handle it.
    VDISPATCH_REORDER(is_zen_packed(dst_md_), VERBOSE_UNSUPPORTED_FORMAT_KIND);

    // The dst is the opaque packed format (no oneDNN blocked layout), so there
    // is no blocked-layout / K-alignment / zero-padding requirement to validate
    // here. The recorded buffer size is cross-checked against the backend's
    // packed size further below.

    VDISPATCH_REORDER(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    // The src is a plain blocked layout; the dst is the opaque packed format
    // (not a blocking_desc).
    VDISPATCH_REORDER(
            id.is_blocking_desc(), VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    VDISPATCH_REORDER(!id.has_runtime_dims_or_strides()
                    && !od.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    VDISPATCH_REORDER(!id.has_zero_dim() && !od.has_zero_dim(), VERBOSE_BAD_DIM,
            "src/dst", 0);

    // Each (K, N) slice of src must be plain row-major (`ab`/`abc`) or
    // col-major (`ba`/`acb`). The packer takes an explicit leading dim (derived
    // from the actual strides in execute()), so a padded leading dim on one
    // axis is supported -- an exact tag check would reject those layouts. Match
    // the matmul contract: require a plain layout with one of the inner (K, N)
    // axes contiguous and no zero strides. For 3D also require the batch dim to
    // be outermost (largest stride) so the per-batch stride is the true slice
    // span (execute() advances src by strides[0] per batch).
    const auto &src_strides = id.blocking_desc().strides;
    const bool inner_contig
            = src_strides[ndims - 1] == 1 || src_strides[ndims - 2] == 1;
    bool no_zero_stride = true;
    for (int i = 0; i < ndims; i++)
        no_zero_stride = no_zero_stride && src_strides[i] != 0;
    VDISPATCH_REORDER(id.is_plain() && inner_contig && no_zero_stride,
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_REORDER(!batched
                    || (src_strides[0] >= src_strides[1]
                            && src_strides[0] >= src_strides[2]),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    // A batched sub-byte source can be advanced with a raw byte pointer only
    // when every slice starts on a byte boundary. An odd logical-element batch
    // stride starts the next slice in the high nibble, which this direct
    // prepack interface cannot represent.
    const size_t src_elems_per_byte
            = types::bytes_to_elements(id.data_type(), 1);
    VDISPATCH_REORDER(!batched
                    || static_cast<size_t>(src_strides[0]) % src_elems_per_byte
                            == 0,
            VERBOSE_UNSUPPORTED_TAG_S, "src");

    // src and dst logical dims must agree (oneDNN reorder API contract).
    for (int i = 0; i < ndims; i++)
        VDISPATCH_REORDER(id.dims()[i] == od.dims()[i],
                VERBOSE_INCONSISTENT_DIM, "src", i, "dst", i);

    // Logical (K, N) of one slice and the batch count. The packed dst records
    // the weights orientation: the last two dims are [.., K, N] for matmul and
    // [.., N, K] for inner product (weights_transposed), so pick K/N accordingly.
    const bool nk = od.zen_packed_desc().weights_transposed;
    const int kdim = nk ? ndims - 1 : ndims - 2;
    const int ndim = nk ? ndims - 2 : ndims - 1;
    const dim_t K = od.dims()[kdim];
    const dim_t N = od.dims()[ndim];
    const dim_t batch = batched ? od.dims()[0] : 1;

    // execute() collapses a K==1 slice to contiguous row-major since a dense
    // single row is layout-agnostic; that assumption needs the N axis
    // contiguous. A padded (non-contiguous) K==1 row is declined here.
    VDISPATCH_REORDER(
            K != 1 || src_strides[ndim] == 1, VERBOSE_UNSUPPORTED_TAG_S, "src");

    // zen_weight_reorder takes int64_t K/N/ldb but the matmul that consumes the
    // packed buffer drives them through the int Zen API; reject oversized slices
    // up front so packing and matmul agree on what is representable.
    const dim_t int_max = std::numeric_limits<int>::max();
    VDISPATCH_REORDER(K <= int_max && N <= int_max && batch <= int_max,
            VERBOSE_UNSUPPORTED_FEATURE,
            "dimension > INT_MAX is not supported");

    // Cross-check the recorded packed sizes against the backend-reported
    // per-slice size to turn a silent overrun into a clean dispatch failure.
    // The opaque dst carries per_slice_size and size == per_slice_size * batch.
    const auto &zpd = od.zen_packed_desc();
    // The per-slice packed size can depend on the matmul source dtype (recorded
    // in the packed descriptor as gemm_src_dt), so query with
    // (wei=type_o, src=gemm_src_dt) -- for f32/bf16/f16 gemm_src_dt == type_o.
    const dim_t expected_per_slice
            = zen_prepack_size(type_o, zpd.gemm_src_dt, K, N);
    VDISPATCH_REORDER(expected_per_slice > 0
                    && zpd.per_slice_size
                            == static_cast<size_t>(expected_per_slice),
            VERBOSE_INCONSISTENT_MDS, "dst", "packed-slice-size");

    // batch is validated >= 1 (no zero dims) and <= INT_MAX above. Guard the
    // per_slice_size * batch product against size_t overflow before comparing:
    // a wrap could otherwise make an undersized buffer pass this check and lead
    // to out-of-bounds writes during packing. Mirrors init_zen_packed_md().
    const size_t batch_sz = static_cast<size_t>(batch);
    VDISPATCH_REORDER(zpd.per_slice_size <= SIZE_MAX / batch_sz,
            VERBOSE_INCONSISTENT_MDS, "dst", "packed-size-overflow");
    VDISPATCH_REORDER(
            zpd.size == zpd.per_slice_size * batch_sz && od.size() == zpd.size,
            VERBOSE_INCONSISTENT_MDS, "dst", "packed-size");

    // The f32 -> {bf16, f16, s8, s4, u4} and {bf16, f16} -> f32 prepack paths
    // need a per-slice K*N conversion buffer in the packed dtype (f32 = 4
    // bytes/elem, bf16/f16 = 2 bytes/elem, s8 = 1 byte/elem, s4/u4 = 2
    // elems/byte). Book it on the primitive scratchpad (declared here,
    // consumed in execute() via the grantor), reused across batches so
    // execution stays allocation-free.
    if (prepack_converts(type_i, type_o)) {
        const size_t nelems = static_cast<size_t>(K) * static_cast<size_t>(N);
        size_t convert_buf_bytes;
        if (type_o == data_type::f32)
            convert_buf_bytes = nelems * sizeof(float);
        else if (utils::one_of(type_o, data_type::bf16, data_type::f16))
            convert_buf_bytes = nelems * sizeof(int16_t);
        else if (type_o == data_type::s8)
            convert_buf_bytes = nelems * sizeof(int8_t);
        else // s4 / u4: two 4-bit values per byte
            convert_buf_bytes = (nelems + 1) / 2;
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_reorder_space,
                convert_buf_bytes, /*data_size=*/1, /*alignment=*/64);
    }

    return status::success;
#endif
}

status_t zen_reorder_t::pd_t::init_unpack(const engine_t *engine) {
#if !DNNL_X64_USE_ZEN
    MAYBE_UNUSED(engine);
    return status::unimplemented;
#else
    const memory_desc_wrapper id(src_md_), od(dst_md_);

    // Dense plain destination only (`ab`/`ba`, `abc`/`acb`, no padding).
    // Blocked tags such as BA16a64b, a padded leading dim, and s4/u4 stay
    // unimplemented.
    VDISPATCH_REORDER(
            od.is_blocking_desc(), VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    VDISPATCH_REORDER(
            utils::one_of(id.ndims(), 2, 3) && id.ndims() == od.ndims(),
            VERBOSE_BAD_NDIMS, "src/dst", id.ndims());

    const int ndims = id.ndims();
    const bool batched = ndims == 3;

    // Same dtype, or a conversion between f32 and bf16/f16 applied after
    // unpacking in the packed dtype. s8 covers both int8 packs; gemm_src_dt
    // selects s8s8 vs u8s8.
    const auto type_i = id.data_type();
    const auto type_o = od.data_type();
    const bool convert = is_float_conversion(type_i, type_o);
    VDISPATCH_REORDER(convert
                    || (type_i == type_o
                            && utils::one_of(type_i, data_type::f32,
                                    data_type::bf16, data_type::f16,
                                    data_type::s8)),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(
            IMPLICATION(type_i == data_type::f16, mayiuse(avx512_core_fp16)),
            VERBOSE_UNSUPPORTED_ISA);

    VDISPATCH_REORDER(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    VDISPATCH_REORDER(!id.has_runtime_dims_or_strides()
                    && !od.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    VDISPATCH_REORDER(!id.has_zero_dim() && !od.has_zero_dim(), VERBOSE_BAD_DIM,
            "src/dst", 0);

    for (int i = 0; i < ndims; i++)
        VDISPATCH_REORDER(id.dims()[i] == od.dims()[i],
                VERBOSE_INCONSISTENT_DIM, "src", i, "dst", i);

    const auto &zpd = id.zen_packed_desc();
    const bool nk = zpd.weights_transposed;
    const int kdim = nk ? ndims - 1 : ndims - 2;
    const int ndim = nk ? ndims - 2 : ndims - 1;
    const dim_t K = id.dims()[kdim];
    const dim_t N = id.dims()[ndim];
    const dim_t batch = batched ? id.dims()[0] : 1;

    const dim_t int_max = std::numeric_limits<int>::max();
    VDISPATCH_REORDER(K <= int_max && N <= int_max && batch <= int_max,
            VERBOSE_UNSUPPORTED_FEATURE,
            "dimension > INT_MAX is not supported");

    const dim_t expected_per_slice
            = zen_prepack_size(type_i, zpd.gemm_src_dt, K, N);
    VDISPATCH_REORDER(expected_per_slice > 0
                    && zpd.per_slice_size
                            == static_cast<size_t>(expected_per_slice),
            VERBOSE_INCONSISTENT_MDS, "src", "packed-slice-size");
    const size_t batch_sz = static_cast<size_t>(batch);
    VDISPATCH_REORDER(zpd.per_slice_size <= SIZE_MAX / batch_sz,
            VERBOSE_INCONSISTENT_MDS, "src", "packed-size-overflow");
    VDISPATCH_REORDER(
            zpd.size == zpd.per_slice_size * batch_sz && id.size() == zpd.size,
            VERBOSE_INCONSISTENT_MDS, "src", "packed-size");

    unpack_dst_layout_t layout;
    VDISPATCH_REORDER(get_unpack_dst_layout(od, kdim, ndim, layout),
            VERBOSE_UNSUPPORTED_TAG);

    // A conversion unpacks each slice into a dense K*N buffer of the packed
    // dtype first.
    if (convert) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_reorder_space,
                types::data_type_size(type_i) * static_cast<size_t>(K)
                        * static_cast<size_t>(N),
                /*data_size=*/1, /*alignment=*/64);
    }

    return status::success;
#endif
}

status_t zen_reorder_t::execute(const exec_ctx_t &ctx) const {
#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else
    // Packed to an identical packed descriptor: a byte copy.
    if (is_zen_packed(*pd()->src_md()) && is_zen_packed(*pd()->dst_md())) {
        const auto *src = CTX_IN_MEM(const void *, DNNL_ARG_FROM);
        auto *dst = CTX_OUT_MEM(void *, DNNL_ARG_TO);
        if (dst != src)
            std::memcpy(dst, src, memory_desc_wrapper(pd()->src_md()).size());
        return status::success;
    }
    if (is_zen_packed(*pd()->src_md())) return execute_unpack(ctx);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    // src is a plain 2D (K, N) slice or a 3D batched [B, K, N] tensor, each
    // slice plain `ab`/`ba` (validated by pd_t::init). Logical per-slice dims
    // are always (K, N); the batch dim (if any) is outermost.
    const int ndims = src_d.ndims();
    const bool batched = ndims == 3;
    // The packed dst records the weights orientation: the last two dims are
    // [.., K, N] for matmul and [.., N, K] for inner product (weights_transposed).
    // Map K/N onto the right physical axes.
    const bool nk = dst_d.zen_packed_desc().weights_transposed;
    const int kdim = nk ? ndims - 1 : ndims - 2;
    const int ndim = nk ? ndims - 2 : ndims - 1;
    const int64_t K = src_d.dims()[kdim];
    const int64_t N = src_d.dims()[ndim];
    const int64_t batch = batched ? src_d.dims()[0] : 1;

    // Describe the logical (K, N) matrix's physical layout to the packer via
    // transposed + ldb, read from the actual strides so a padded leading dim is
    // honored (pd_t::init guarantees one of the last two axes is unit-stride):
    //   N axis contiguous -> row-major [K,N] -> trans='n', ldb = K-axis stride
    //   K axis contiguous -> col-major [K,N] -> trans='t', ldb = N-axis stride
    //   K==1 degenerate row -> both coincide -> row-major, ldb = N.
    // For matmul (nk=false) this reduces to the plain ab/ba test.
    const auto &src_strides = src_d.blocking_desc().strides;
    const int64_t k_stride = src_strides[kdim];
    const int64_t n_stride = src_strides[ndim];
    bool transposed;
    int64_t ldb;
    if (K == 1) {
        transposed = false;
        ldb = N;
    } else if (n_stride == 1) {
        transposed = false;
        ldb = k_stride;
    } else {
        transposed = true;
        ldb = n_stride;
    }
    const bool src_is_ab = !transposed;

    const auto src_dt = src_d.data_type();
    const auto dst_dt = dst_d.data_type();

    // Per-batch advance: source strides are expressed in logical elements.
    // Convert to bytes explicitly because s4/u4 store two elements per byte.
    // pd_t::init() rejects a sub-byte stride that starts on a high nibble.
    const size_t src_slice_bytes = batched
            ? types::elements_to_bytes(
                      src_dt, static_cast<size_t>(src_strides[0]))
            : 0;
    const size_t dst_slice_bytes = dst_d.zen_packed_desc().per_slice_size;

    const auto *src_base = CTX_IN_MEM(const uint8_t *, DNNL_ARG_FROM);
    auto *dst_base = CTX_OUT_MEM(uint8_t *, DNNL_ARG_TO);

    // f32 -> {bf16, f16, s8, s4, u4} and {bf16, f16} -> f32 prepack need a
    // per-slice conversion buffer (booked in pd_t::init), reused across
    // batches so execute() stays allocation-free.
    void *convert_buf = nullptr;
    if (prepack_converts(src_dt, dst_dt)) {
        convert_buf = ctx.get_scratchpad_grantor().get<void>(
                memory_tracking::names::key_reorder_space);
        if (convert_buf == nullptr) return status::out_of_memory;
    }

    // The matmul source dtype (u8/s8 for int8; f32/bf16/f16 otherwise) is recorded
    // in the packed descriptor; the AOCL-DLP blocked layout can depend on it,
    // so forward it explicitly to the packer.
    const zd gemm_src = to_zen_dt(dst_d.zen_packed_desc().gemm_src_dt);

    // Pack one (K, N) slice from `src` into `dst`.
    auto prepack_slice = [&](const void *src, void *dst) -> status_t {
        if (src_dt == data_type::bf16 && dst_dt == data_type::bf16)
            return zen_weight_reorder(
                    src, dst, zd::bf16, zd::bf16, K, N, ldb, transposed);

        if (src_dt == data_type::f16 && dst_dt == data_type::f16)
            return zen_weight_reorder(
                    src, dst, zd::f16, zd::f16, K, N, ldb, transposed);

        if (src_dt == data_type::f32 && dst_dt == data_type::f32)
            return zen_weight_reorder(
                    src, dst, zd::f32, zd::f32, K, N, ldb, transposed);

        if (src_dt == data_type::s8 && dst_dt == data_type::s8)
            // int8 static-quant weight prepack.
            return zen_weight_reorder(
                    src, dst, zd::s8, gemm_src, K, N, ldb, transposed);

        if ((src_dt == data_type::s4 && dst_dt == data_type::s4)
                || (src_dt == data_type::u4 && dst_dt == data_type::u4))
            // WOQ 4-bit weight prepack; the gemm source (bf16) is recorded in
            // the packed descriptor and forwarded to the packer.
            return zen_weight_reorder(src, dst, to_zen_dt(src_dt), gemm_src, K,
                    N, ldb, transposed);

        if (src_dt == data_type::f32 && dst_dt == data_type::bf16) {
            // Convert f32 -> plain bf16 (contiguous `ab`), then prepack that
            // bf16 slice into the Zen blocked layout. The conversion avoids the
            // backend's f32->bf16 fringe-N bug (see header note).
            status_t st = convert_plain(
                    src, convert_buf, zd::f32, zd::bf16, K, N, ldb, src_is_ab);
            if (st == success)
                st = zen_weight_reorder(convert_buf, dst, zd::bf16, zd::bf16, K,
                        N, /*ldb=*/N, /*transposed=*/false);
            return st;
        }

        if (src_dt == data_type::f32 && dst_dt == data_type::f16) {
            // Convert f32 -> plain f16 (contiguous `ab`), then prepack that
            // f16 slice into the Zen blocked layout (same two-step pattern as
            // f32->bf16).
            status_t st = convert_plain(
                    src, convert_buf, zd::f32, zd::f16, K, N, ldb, src_is_ab);
            if (st == success)
                st = zen_weight_reorder(convert_buf, dst, zd::f16, zd::f16, K,
                        N, /*ldb=*/N, /*transposed=*/false);
            return st;
        }

        if (utils::one_of(src_dt, data_type::bf16, data_type::f16)
                && dst_dt == data_type::f32) {
            // Convert to plain f32 (contiguous `ab`), then prepack that f32
            // slice (same two-step pattern as f32->bf16).
            status_t st = convert_plain(src, convert_buf, to_zen_dt(src_dt),
                    zd::f32, K, N, ldb, src_is_ab);
            if (st == success)
                st = zen_weight_reorder(convert_buf, dst, zd::f32, zd::f32, K,
                        N, /*ldb=*/N, /*transposed=*/false);
            return st;
        }

        if (src_dt == data_type::f32 && dst_dt == data_type::s8) {
            // int8 prepack from an f32 weight reference: cast f32 -> plain s8
            // (contiguous `ab`), then prepack that s8 slice into the Zen
            // blocked layout.
            status_t st
                    = f32_to_s8_plain(src, convert_buf, K, N, ldb, src_is_ab);
            if (st == success)
                st = zen_weight_reorder(convert_buf, dst, zd::s8, gemm_src, K,
                        N, /*ldb=*/N, /*transposed=*/false);
            return st;
        }

        if (src_dt == data_type::f32
                && utils::one_of(dst_dt, data_type::s4, data_type::u4)) {
            // WOQ prepack from an f32 weight reference: cast f32 -> packed
            // 4-bit (contiguous `ab`), then prepack that 4-bit slice.
            status_t st = f32_to_int4_plain(
                    src, convert_buf, K, N, ldb, src_is_ab, dst_dt);
            if (st == success)
                st = zen_weight_reorder(convert_buf, dst, to_zen_dt(dst_dt),
                        gemm_src, K, N, /*ldb=*/N, /*transposed=*/false);
            return st;
        }

        return status::unimplemented;
    };

    for (int64_t b = 0; b < batch; b++) {
        const void *src_slice
                = src_base + static_cast<size_t>(b) * src_slice_bytes;
        void *dst_slice = dst_base + static_cast<size_t>(b) * dst_slice_bytes;
        CHECK(prepack_slice(src_slice, dst_slice));
    }

    return status::success;
#endif
}

status_t zen_reorder_t::execute_unpack(const exec_ctx_t &ctx) const {
#if !DNNL_X64_USE_ZEN
    MAYBE_UNUSED(ctx);
    return status::unimplemented;
#else
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const int ndims = src_d.ndims();
    const bool batched = ndims == 3;
    const auto &zpd = src_d.zen_packed_desc();
    const bool nk = zpd.weights_transposed;
    const int kdim = nk ? ndims - 1 : ndims - 2;
    const int ndim = nk ? ndims - 2 : ndims - 1;
    const int64_t K = src_d.dims()[kdim];
    const int64_t N = src_d.dims()[ndim];
    const int64_t batch = batched ? src_d.dims()[0] : 1;

    unpack_dst_layout_t l;
    if (!get_unpack_dst_layout(dst_d, kdim, ndim, l))
        return status::runtime_error;

    // f32/bf16/f16 pack with src == wei; s8 uses the recorded gemm source
    // (s8 or u8) to pick the s8s8 or u8s8 layout.
    const zd wei = to_zen_dt(src_d.data_type());
    const zd gemm_src = src_d.data_type() == data_type::s8
            ? to_zen_dt(zpd.gemm_src_dt)
            : wei;

    const auto *src_base = CTX_IN_MEM(const uint8_t *, DNNL_ARG_FROM);
    const size_t elem = dst_d.data_type_size();
    auto *dst_base = CTX_OUT_MEM(uint8_t *, DNNL_ARG_TO)
            + static_cast<size_t>(dst_d.offset0()) * elem;
    const size_t dst_slice_bytes = static_cast<size_t>(l.batch_stride) * elem;

    // A conversion unpacks each slice into the scratchpad in the packed
    // dtype, with the dst layout, then converts it into dst. Both are dense,
    // so the conversion walks the K*N elements in memory order.
    void *convert_buf = nullptr;
    if (src_d.data_type() != dst_d.data_type()) {
        convert_buf = ctx.get_scratchpad_grantor().get<void>(
                memory_tracking::names::key_reorder_space);
        if (convert_buf == nullptr) return status::out_of_memory;
    }

    for (int64_t b = 0; b < batch; b++) {
        void *dst_slice = dst_base + static_cast<size_t>(b) * dst_slice_bytes;
        CHECK(zen_weight_reorder(
                src_base + static_cast<size_t>(b) * zpd.per_slice_size,
                convert_buf ? convert_buf : dst_slice, wei, gemm_src, K, N,
                l.ldb, l.transposed, /*unpack=*/true));
        if (convert_buf)
            CHECK(convert_plain(convert_buf, dst_slice, wei,
                    to_zen_dt(dst_d.data_type()), K, N, /*ldb=*/N,
                    /*src_is_ab=*/true));
    }
    return status::success;
#endif
}

} // namespace reorder
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
