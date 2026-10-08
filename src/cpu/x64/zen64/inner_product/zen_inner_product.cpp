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

#include "cpu/x64/zen64/inner_product/zen_inner_product.hpp"

#include <string>

#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/matmul_inner_product.hpp" // init_matmul_md + matmul_desc machinery
#include "cpu/x64/zen64/common/zen_format_tag.hpp" // init_zen_packed_md

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace inner_product {

using namespace data_type;

namespace {

// Like matmul_inner_product's create_matmul_pd, but stops at the first impl
// whose name identifies zen_matmul (the shared helper only accepts brg_matmul).
status_t create_zen_matmul_pd(std::shared_ptr<primitive_desc_t> &matmul_pd,
        const engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *dst_md,
        const memory_desc_t *bia_md, const primitive_attr_t *attr) {
    auto matmul_desc = matmul_desc_t();
    CHECK(matmul_desc_init(&matmul_desc, src_md, wei_md, bia_md, dst_md,
            /*reduce_md=*/nullptr, matmul_reduce_kind::src));

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&matmul_desc, attr, nullptr);
    while (it != it.end()) {
        matmul_pd = *(++it);
        if (!matmul_pd) return status::unimplemented;
        if (std::string(matmul_pd->name()).find("zen") != std::string::npos)
            return status::success;
    }
    return status::unimplemented;
}

} // namespace

status_t zen_inner_product_fwd_t::pd_t::init(const engine_t *engine) {
    using namespace utils;
    using smask_t = primitive_attr_t::skip_mask_t;

    // IP-specific gates only; the nested zen_matmul validates dtypes/post-ops.
    // Both fwd prop_kinds are accepted; init_matmul_params picks the weights path.
    VDISPATCH_INNER_PRODUCT(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_INNER_PRODUCT(
            ::dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tAMD),
            "This implementation only supports AMD CPUs");
    VDISPATCH_INNER_PRODUCT(mayiuse(avx512_core), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_INNER_PRODUCT(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    // 2D only (plain Linear); spatial IP declines to matmul_inner_product/brgemm.
    VDISPATCH_INNER_PRODUCT(ndims() == 2, VERBOSE_BAD_NDIMS, "src", ndims());
    VDISPATCH_INNER_PRODUCT(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    const auto src_dt = src_md(0)->data_type;
    const auto wei_dt = weights_md(0)->data_type;
    const bool is_f16 = src_dt == f16 || wei_dt == f16;
    VDISPATCH_INNER_PRODUCT(
            one_of(wei_dt, f32, bf16, f16), VERBOSE_UNSUPPORTED_DT);
    // F16 requires AVX512-FP16 (Zen5+), matching nested zen_matmul.
    VDISPATCH_INNER_PRODUCT(IMPLICATION(is_f16, mayiuse(avx512_core_fp16)),
            VERBOSE_UNSUPPORTED_ISA);

    // Post-ops (+sum_dt) may deviate; nested zen_matmul checks the exact set.
    // Skip acc-mode only for f16 so --attr-acc-mode=f16 can reach zen_matmul.
    // f32/bf16 keep the default check (strict|relaxed|any), which rejects
    // acc-mode=f32 and does not widen those configs.
    auto skip_mask = smask_t::post_ops | smask_t::sum_dt;
    if (is_f16) skip_mask = skip_mask | smask_t::accumulation_mode;
    VDISPATCH_INNER_PRODUCT(
            attr()->has_default_values(skip_mask, dst_md(0)->data_type),
            VERBOSE_UNSUPPORTED_ATTR);

    VDISPATCH_INNER_PRODUCT_SC(
            init_matmul_params(engine), "init_matmul_params");

    init_scratchpad();
    return status::success;
}

status_t zen_inner_product_fwd_t::pd_t::set_training_formats() {
    using namespace format_tag;
    assert(desc()->prop_kind != prop_kind::forward_inference);
    // Backward shares src/weights, so resolve open descriptors to the plain
    // layout it expects (never zen_packed, nor the ba transpose). 2D: tag == ab.
    if (memory_desc_wrapper(src_md_).format_any())
        CHECK(memory_desc_init_by_tag(src_md_, ab));
    if (memory_desc_wrapper(weights_md_).format_any())
        CHECK(memory_desc_init_by_tag(weights_md_, ab));
    if (memory_desc_wrapper(dst_md_).format_any())
        CHECK(memory_desc_init_by_tag(dst_md_, ab));
    if (with_bias() && memory_desc_wrapper(bias_md_).format_any())
        CHECK(memory_desc_init_by_tag(bias_md_, x));
    return status::success;
}

status_t zen_inner_product_fwd_t::pd_t::init_matmul_params(
        const engine_t *engine) {
    using namespace format_tag;
    const auto src_dt = src_md(0)->data_type;
    const auto wei_dt = weights_md(0)->data_type;

    // Prepack (zen_packed) needs inference weights, either open (we advertise,
    // framework reorders once) or already packed from a prior query.
    const bool inference = desc()->prop_kind == prop_kind::forward_inference;
    const bool wei_format_any = memory_desc_wrapper(weights_md(0)).format_any();
    const bool wei_already_packed = zen::is_zen_packed(*weights_md(0));

    // Inference may pick the zen_packed prepack layout; training must expose
    // plain layouts the backward primitives can consume.
    if (inference) {
        VDISPATCH_INNER_PRODUCT(set_default_params() == status::success,
                VERBOSE_UNSUPPORTED_TAG);
    } else {
        VDISPATCH_INNER_PRODUCT_SC(
                set_training_formats(), VERBOSE_UNSUPPORTED_TAG);
    }

    // Unpadded IC so the matmul (K=IC) never over-reads; plain 2D src/dst.
    VDISPATCH_INNER_PRODUCT(
            IC_total() == IC_total_padded(), VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_INNER_PRODUCT(memory_desc_wrapper(src_md(0)).matches_tag(ab),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_INNER_PRODUCT(memory_desc_wrapper(dst_md(0)).matches_tag(ab),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");

    const dim_t K = IC_total();
    const dim_t N = OC();

    // 2D matmul descriptors: src [MB, IC] ab, dst [MB, OC] ab, bias [1, OC] ab.
    memory_desc_t mm_src_md {}, mm_wei_md {}, mm_dst_md {}, mm_bia_md {};
    VDISPATCH_INNER_PRODUCT_SC(
            init_matmul_md(mm_src_md, *src_md(), ab), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_INNER_PRODUCT_SC(
            init_matmul_md(mm_dst_md, *dst_md(), ab), VERBOSE_UNSUPPORTED_TAG);

    const bool use_zen_packed
            = inference && (wei_format_any || wei_already_packed);
    if (use_zen_packed) {
        // matmul weights = zen_packed [IC, OC]. The packed bytes are normalized
        // [K=IC, N=OC], so this md is identical whether advertised or pre-packed.
        dims_t wei_2d = {K, N};
        VDISPATCH_INNER_PRODUCT_SC(
                memory_desc_init_by_tag(mm_wei_md, 2, wei_2d, wei_dt, ab),
                VERBOSE_UNSUPPORTED_TAG);
        VDISPATCH_INNER_PRODUCT_SC(
                zen::init_zen_packed_md(mm_wei_md, src_dt, K, N, /*batch=*/1,
                        /*weights_transposed=*/false),
                VERBOSE_UNSUPPORTED_TAG);
        // Advertise zen_packed [OC, IC] only for open weights, so the framework
        // reorders once. A pre-packed md is kept as-is (no second reorder).
        if (wei_format_any) {
            VDISPATCH_INNER_PRODUCT_SC(
                    zen::init_zen_packed_md(weights_md_, src_dt, K, N,
                            /*batch=*/1, /*weights_transposed=*/true),
                    VERBOSE_UNSUPPORTED_TAG);
        }
    } else {
        // Plain path: hand the matmul the IP weights [OC, IC] as an [IC, OC] view
        // of the same bytes. oi -> ba (transposed view); io -> ab (direct view).
        // A concrete tag keeps zen_matmul on its plain path (packs each call).
        const memory_desc_wrapper wd(weights_md(0));
        format_tag_t mm_wei_tag = format_tag::undef;
        if (wd.matches_tag(ab))
            mm_wei_tag = ba;
        else if (wd.matches_tag(ba))
            mm_wei_tag = ab;
        VDISPATCH_INNER_PRODUCT(mm_wei_tag != format_tag::undef,
                VERBOSE_UNSUPPORTED_TAG_S, "weights");
        VDISPATCH_INNER_PRODUCT_SC(init_matmul_md(mm_wei_md, *weights_md(),
                                           mm_wei_tag, /*swap_dims=*/true),
                VERBOSE_UNSUPPORTED_TAG);
    }

    if (with_bias()) {
        dims_t bia_2d = {1, N};
        VDISPATCH_INNER_PRODUCT_SC(
                memory_desc_init_by_tag(mm_bia_md, 2, bia_2d,
                        weights_md(1)->data_type, format_tag::ab),
                VERBOSE_UNSUPPORTED_TAG);
    }

    // Resolve format_any on binary post-op src1 so the IP pd advertises a
    // concrete layout and rejects unplaceable post-ops (nested matmul no-ops it).
    VDISPATCH_INNER_PRODUCT_SC(
            attr_.set_default_formats(dst_md(0)), VERBOSE_UNSUPPORTED_POSTOP);

    // Nested matmul pd (must resolve to zen_matmul); it owns validation,
    // post-ops, and compute.
    primitive_attr_t matmul_attr = *attr();
    VDISPATCH_INNER_PRODUCT_SC(
            create_zen_matmul_pd(matmul_pd_, engine, &mm_src_md, &mm_wei_md,
                    &mm_dst_md, with_bias() ? &mm_bia_md : nullptr,
                    &matmul_attr),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "matmul");
    VDISPATCH_INNER_PRODUCT(
            matmul_pd_ != nullptr, VERBOSE_PRIMITIVE_CREATION_FAIL, "matmul");

    return status::success;
}

status_t zen_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    // Reuse the DNNL_ARG_* memories as-is: src/dst/bias flatten to the 2D mds and
    // the IP weights buffer is read as the matmul's [IC,OC] view (packed or ba).
    exec_args_t matmul_args = ctx.args();
    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, matmul_->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);

    return matmul_->execute(matmul_ctx);
}

} // namespace inner_product
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
