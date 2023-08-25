/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include <float.h>
#include <functional>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

// TODO: refactor the driver to avoid using extra flags of a memory descriptor.
#include "src/common/memory_desc.hpp"

#include "tests/test_isa_common.hpp"

#include "utils/parallel.hpp"
#include "utils/parser.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "brgemm/brgemm.hpp"

#if defined(DNNL_X64) && DNNL_X64 == 1 && DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
template <>
struct dnnl_api_traits<dnnl::impl::cpu::x64::brgemm_kernel_t *> {
    static void destroy(dnnl::impl::cpu::x64::brgemm_kernel_t *t) {
        DNN_SAFE_V(dnnl::impl::cpu::x64::brgemm_kernel_destroy(t));
    }
};
#endif

namespace brgemm {

#if defined(DNNL_X64) && DNNL_X64 == 1 && DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE

/// Initializes BRGEMM attributes from an input string.
///
/// @param brgattr Output BRGEMM attributes.
/// @param str Input string of values in the format: KEY:VALUE[+KEY:VALUE[...]].
///     `KEY` follows exact name of the brgemm_attr_t object members and their
///     `VALUE` follow the member type. enum and boolean types are treated as
///     integers.
///
dnnl_status_t brgemm_attr_init(
        dnnl::impl::cpu::x64::brgemm_attr_t *brgattr, const prb_t *prb) {
    using namespace dnnl::impl::cpu::x64;

    const auto &str = prb->brgemm_attr;
    if (str.empty()) return dnnl_success;

    size_t entry_pos = 0;
    while (entry_pos != std::string::npos) {
        auto key_value_str = parser::get_substr(str, entry_pos, '+');
        size_t value_pos = 0;
        auto key_str = parser::get_substr(key_value_str, value_pos, ':');
        auto value_str = parser::get_substr(key_value_str, value_pos, '\0');

#define PROCESS_SETTING_KEY_VAL(setting, key) \
    if (key_str.compare(STRINGIFY(key)) == 0) \
        brgattr->setting = std::stoi(value_str);

#define PROCESS_KEY_VAL(setting) PROCESS_SETTING_KEY_VAL(setting, setting)

        // TODO: `max_top_vpad` and `max_bottom_vpad` do not affect anything in
        // the kernel call and reference computation so far since
        // batch_element_t struct is not adjusted to incorporate different pad
        // values.
        // PROCESS_KEY_VAL(max_top_vpad);
        // PROCESS_KEY_VAL(max_bottom_vpad);
        PROCESS_KEY_VAL(hint_expected_A_size);
        PROCESS_KEY_VAL(hint_expected_B_size);
        PROCESS_KEY_VAL(hint_expected_C_size);
        PROCESS_KEY_VAL(wary_tail_read);
        PROCESS_KEY_VAL(generate_skip_accumulation);
        // TODO: `bd_mask` can't be passed to the kernel at this moment, that's
        // why `bd_mask_level` has to stay `0` for now until it's enabled.
        // PROCESS_KEY_VAL(bd_mask_level);
        PROCESS_KEY_VAL(use_uker);
        PROCESS_KEY_VAL(use_interleave_stores);
        PROCESS_KEY_VAL(postops_only);
        PROCESS_KEY_VAL(hint_bd_block);
        PROCESS_KEY_VAL(hint_bd_block2);
        PROCESS_KEY_VAL(hint_ld_block);
        PROCESS_KEY_VAL(hint_ld_block2);

        PROCESS_SETTING_KEY_VAL(hint_prfA.dist1, hint_prfA_dist1);
        PROCESS_SETTING_KEY_VAL(hint_prfA.dist2, hint_prfA_dist2);
        PROCESS_SETTING_KEY_VAL(hint_prfB.dist1, hint_prfB_dist1);
        PROCESS_SETTING_KEY_VAL(hint_prfB.dist2, hint_prfB_dist2);
        PROCESS_SETTING_KEY_VAL(hint_prfC.dist1, hint_prfC_dist1);
        PROCESS_SETTING_KEY_VAL(hint_prfC.dist2, hint_prfC_dist2);

#undef PROCESS_SETTING_KEY_VAL
#undef PROCESS_KEY_VAL

        if (key_str.find(STRINGIFY(hint_innermost_loop)) != std::string::npos)
            brgattr->hint_innermost_loop
                    = static_cast<brgemm_kernel_innermost_loop_t>(
                            std::stoi(value_str));
        if (key_str.find(STRINGIFY(hint_loop_order)) != std::string::npos)
            brgattr->hint_loop_order = static_cast<brgemm_kernel_loop_order_t>(
                    std::stoi(value_str));
        if (key_str.find(STRINGIFY(hint_prefetching)) != std::string::npos)
            brgattr->hint_prefetching
                    = static_cast<brgemm_kernel_prefetching_t>(
                            std::stoi(value_str));
        if (key_str.find(STRINGIFY(hint_load_nt_A)) != std::string::npos)
            brgattr->hint_load_nt_A = static_cast<brgemm_kernel_hint_nt_t>(
                    std::stoi(value_str));
        if (key_str.find(STRINGIFY(hint_load_nt_B)) != std::string::npos)
            brgattr->hint_load_nt_B = static_cast<brgemm_kernel_hint_nt_t>(
                    std::stoi(value_str));
    }

    // `max_bs` is handled directly through the driver interface.
    brgattr->max_bs = prb->batch_size;

    // `fpmath_mode` is handled directly through the driver interface.
    brgattr->fpmath_mode = prb->attr.fpmath_mode;

    return dnnl_success;
}

std::string prepare_wei_format_string(
        dnnl_data_type_t dt, int64_t ldb, bool is_vnni_layout) {
    // `dt` affects the choice of last inner block (for VNNI-friendliness).
    // `n` affects the choice of B block.
    std::string wtag("BA16a");
    wtag += std::to_string(ldb) + "b";
    if (is_vnni_layout) {
        switch (dt) {
            case dnnl_f32: break;
            case dnnl_f16:
            case dnnl_bf16: wtag += "2a"; break;
            case dnnl_u8:
            case dnnl_s8: wtag += "4a"; break;
            default: assert(!"unsupported data type");
        }
    }

    return wtag;
}

int fill_data(data_kind_t kind, const prb_t *prb, const cfg_t &cfg,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    cfg_t::density_args_t density_args;
    density_args.data_kind = kind;
    density_args.n_acc = prb->k;
    const auto density = cfg.get_density(density_args);

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::bernoulli_distribution b_dist(density);

        // make sure the first element is positive
        if (idx_start == 0) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            mem_fp.set_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = density == 1.f ? true : b_dist(b_seed);
            float val = is_one * gen(int_seed);
            mem_fp.set_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    auto is_xf16 = [](dnnl_data_type_t dt) {
        return dt == dnnl_bf16 || dt == dnnl_f16;
    };
    if (!IMPLICATION(is_xf16(prb->bia_dt) || is_xf16(prb->dst_dt()),
                is_xf16(prb->wei_dt()))) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
    skip_unimplemented_data_type(
            {prb->src_dt(), prb->wei_dt(), prb->bia_dt, prb->dst_dt()},
            prb->dir, res);
    skip_unimplemented_sum_po(
            prb->attr, res, dnnl_gemm, prb->src_dt(), prb->dst_dt());
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_gemm);

    // Unconditionally skip remaining unimplemented cases.
    // TODO: stop doing it.
    res->state = SKIPPED;
    res->reason = CASE_NOT_SUPPORTED;
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // Reorder does not support s8 and zp compensations for arbitrary shapes,
    // so skip unsupported cases.
    // Note: this check must be done here to avoid runtime error in benchdnn due
    // to failed reorder creation.
    // TODO: enable this support and remove this check.
    const bool is_bad_ldb = prb->get_ldb() % 16 > 0 || prb->get_ldb() > 64;
    const bool req_s8_comp = prb->src_dt() == dnnl_s8;
    const bool req_zp_comp = !prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    if (is_bad_ldb && (req_s8_comp || req_zp_comp)) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const auto dt = prb->get_dt(kind);
    const float trh = dt == dnnl_f32 ? 1e-6f : epsilon_dt(dt);
    cmp.set_threshold(trh);
    cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?
}

// A special wrapper needed to match internal infrastructure.
dnnl_status_t brgemm_kernel_execute_postops_wrapper(
        const dnnl::impl::cpu::x64::brgemm_kernel_t *brgemm_kernel,
        int batch_size,
        const dnnl::impl::cpu::x64::brgemm_batch_element_t *batch_element,
        void *acc_ptr, void *dst_ptr,
        const dnnl::impl::cpu::x64::brgemm_post_ops_data_t &post_ops_data,
        void *scratchpad_ptr, const dnnl_stream_t &stream,
        const std::vector<dnnl_exec_arg_t> &dnnl_args) {
    brgemm_kernel_execute_postops(brgemm_kernel, batch_size, batch_element,
            acc_ptr, dst_ptr, post_ops_data, scratchpad_ptr);
    return dnnl_success;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == bench_mode_t::list) return res->state = LISTED, OK;

    skip_start(res);
    if (res->state == SKIPPED) return OK;

    // Need this here as brgemm has no primitive creation step
    skip_invalid_prb(prb, res);
    if (res->state == SKIPPED) return OK;

    bool use_dst_as_acc = false;
    if (prb->bia_dt == dnnl_data_type_undef && prb->acc_dt() == prb->dst_dt()
            && prb->attr.is_def(/* skip_fmpath = */ true))
        use_dst_as_acc = true;

    // Fuse batch size into K dimension which follows the library usage of the
    // kernel batch size setting.
    const dnnl_dims_t src_dims = {prb->m, prb->k * prb->batch_size};
    const dnnl_dims_t wei_dims = {prb->k * prb->batch_size, prb->n};

    dims_t src_strides = {prb->get_lda(), 1};
    dims_t dst_strides = {prb->get_ldd(), 1};
    dims_t acc_strides = use_dst_as_acc ? dst_strides : dims_t();

    auto dst_md = dnn_mem_t::init_md(prb->ndims, prb->dst_dims.data(),
            prb->dst_dt(), prb->dtag, dst_strides);

    using namespace dnnl::impl::cpu::x64;

    brgemm_t brgemm_desc;
    // Supports only address model for now as only affects the way memory is
    // passed to `brgemm_batch_element_t` object.
    brgemm_batch_kind_t batch_kind = brgemm_batch_kind_t::brgemm_addr;
    brgemm_layout_t layout = brgemm_layout_t::brgemm_row_major;

    // Pass `isa_undef` for now since internal work with it or rather isa bits
    // than isa values directly which causes misalignment between public enum
    // and internal values.
    // TODO: re-consider enabling isa values.
    const auto isa_undef = cpu_isa_t::isa_undef;

    // Create BRGeMM descriptor, analogous to primitive descriptor creation
    const auto status_init = brgemm_desc_init(&brgemm_desc, isa_undef,
            batch_kind, prb->src_dt(), prb->wei_dt(), false /* transA */,
            false /* transB */, layout, prb->alpha, prb->beta, prb->get_lda(),
            prb->get_ldb(), prb->get_ldc(use_dst_as_acc), prb->m, prb->n,
            prb->k, nullptr /* strides */);
    SAFE(check_dnnl_status(status_init, prb, res), WARN);
    if (res->state == SKIPPED) return OK;

    attr_args_t attr_args;
    auto wei_scale = prb->attr.scales.get(DNNL_ARG_WEIGHTS);
    if (wei_scale.policy == policy_t::PER_OC) {
        attr_args.prepare_scales(prb->attr, DNNL_ARG_WEIGHTS, 2);
    }
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    SAFE(check_dnnl_status(brgemm_desc_set_postops(&brgemm_desc, dnnl_attr,
                                   dst_md, prb->get_ldd(), prb->bia_dt),
                 prb, res),
            WARN);
    if (res->state == SKIPPED) return OK;

    brgemm_attr_t brgemm_attr;
    DNN_SAFE(brgemm_attr_init(&brgemm_attr, prb), WARN);
    SAFE(check_dnnl_status(
                 brgemm_desc_set_attr(&brgemm_desc, brgemm_attr), prb, res),
            WARN);
    if (res->state == SKIPPED) return OK;

    // Create BRGeMM kernel, analogous to primitive creation.
    // ctx_init can here be used to select core type on hetero ISA with
    // tbb
    brgemm_kernel_t *brgemm_kernel_;
    {
        auto brgemm_kernel_addr = &brgemm_kernel_;
        DNN_SAFE(create_in_thr_ctx(prb->ctx_init, brgemm_kernel_create,
                         brgemm_kernel_addr, brgemm_desc),
                WARN);
    }
    auto brgemm_kernel = make_benchdnn_dnnl_wrapper(brgemm_kernel_);

    const auto is_tmm = brgemm_desc.is_tmm;
    if (is_tmm) {
        char palette[AMX_PALETTE_SIZE] = {};
        DNN_SAFE(brgemm_init_tiles(brgemm_desc, palette), WARN);
        DNN_SAFE(amx_tile_configure(palette), WARN);
    }

    auto src_md = dnn_mem_t::init_md(
            prb->ndims, src_dims, prb->src_dt(), prb->stag, src_strides);

    // Create weights memory descriptor with VNNI-friendly format.
    // Note: LDB is not passed here. This is because it's super difficult to
    // incorporate stride on top of blocking - oneDNN API doesn't provide any
    // calls to support both options together. Submemory descriptor, which is
    // the only one who can create such memory desc, can't return the size of
    // memory. Thus, it requires two memories and we need to pass a memory
    // handle from bigger one (where LDB is an actual dim value) to smaller, but
    // there's some reorder bug resulting in an error.
    const auto wtag = prepare_wei_format_string(
            prb->wei_dt(), prb->get_ldb(), brgemm_desc.is_b_data_layout_vnni());
    BENCHDNN_PRINT(6, "wtag: %s\n", wtag.c_str());
    auto wei_md = dnn_mem_t::init_md(prb->ndims, wei_dims, prb->wei_dt(), wtag);

    const size_t wei_offset_s8s8 = dnnl_memory_desc_get_size(wei_md);
    // Prepare and assign extra for wei_md when s8s8 compensation, or source
    // zero point reduction values are needed.
    dnnl::impl::memory_extra_desc_t wei_md_extra {};
    wei_md_extra.flags = dnnl::impl::memory_extra_flags::none;
    if (prb->get_dt(SRC) == dnnl_s8 && prb->get_dt(WEI) == dnnl_s8) {
        wei_md_extra.flags
                |= dnnl::impl::memory_extra_flags::compensation_conv_s8s8;
        wei_md_extra.compensation_mask = 2; // N dimension
    }
    static_cast<dnnl_memory_desc_t>(wei_md)->extra = wei_md_extra;

    const size_t wei_offset_zp = wei_offset_s8s8
            + (wei_md_extra.flags != dnnl::impl::memory_extra_flags::none
                            ? prb->get_ldb() * sizeof(int32_t)
                            : 0);

    const bool need_src_comp = !prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    if (need_src_comp) {
        wei_md_extra.flags |= dnnl::impl::memory_extra_flags::
                compensation_conv_asymmetric_src;
        wei_md_extra.asymm_compensation_mask = 2; // N dimension
    }
    static_cast<dnnl_memory_desc_t>(wei_md)->extra = wei_md_extra;

    // Same as dst_md but with a pre-defined data type according to doc.
    auto acc_md = dnn_mem_t::init_md(prb->ndims, prb->dst_dims.data(),
            prb->acc_dt(), tag::abx, acc_strides);

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> bia_md {};
    if (prb->bia_dt != dnnl_data_type_undef) {
        const dnnl_dims_t bia_dims = {1, prb->n};
        bia_md = dnn_mem_t::init_md(
                prb->ndims, bia_dims, prb->bia_dt, tag::abx);
    }

    if (bench_mode == bench_mode_t::init) return res->state = INITIALIZED, OK;

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t acc_dt(acc_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t bia_dt;
    const char *bia_dt_ptr = nullptr;
    if (prb->bia_dt != dnnl_data_type_undef) {
        bia_dt = dnn_mem_t(bia_md, test_engine);
        bia_dt_ptr = (const char *)bia_dt;
    }

    dnn_mem_t src_fp(src_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t wei_fp(wei_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t acc_fp(acc_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t dst_fp(dst_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t bia_fp;
    if (prb->bia_dt != dnnl_data_type_undef)
        bia_fp = dnn_mem_t(bia_md, dnnl_f32, tag::abx, ref_engine);

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, BIA, DST});
    TIME_FILL(SAFE(fill_data(SRC, prb, cfg, src_dt, src_fp, res), WARN));
    TIME_FILL(SAFE(fill_data(WEI, prb, cfg, wei_dt, wei_fp, res), WARN));
    const int sum_idx = prb->attr.post_ops.find(attr_t::post_ops_t::SUM);
    if ((prb->beta != 0) || brgemm_attr.generate_skip_accumulation) {
        TIME_FILL(SAFE(fill_data(DST, prb, cfg, acc_dt, acc_fp, res), WARN));
        // Beta requires same values for reference and the kernel.
        if (use_dst_as_acc) {
            SAFE(dst_fp.reorder(acc_fp), WARN);
            SAFE(dst_dt.reorder(dst_fp), WARN);
        }
    }
    if (sum_idx >= 0)
        TIME_FILL(SAFE(fill_data(DST, prb, cfg, dst_dt, dst_fp, res), WARN));
    if (prb->bia_dt != dnnl_data_type_undef)
        TIME_FILL(SAFE(fill_data(BIA, prb, cfg, bia_dt, bia_fp, res), WARN));

    // "Library" args are needed to get dst for comparison.
    // "Reference" are used as usual.
    args_t args, ref_args;
    args.set(DNNL_ARG_DST, dst_dt);

    std::vector<brgemm_batch_element_t> v_batch_element(prb->batch_size);
    const char *src_ptr = (const char *)src_dt;
    const char *wei_ptr = (const char *)wei_dt;
    // Note: batch_size is incorporated into K dimension.
    // That's why each source batch has an offset of `k`.
    // Weights have more complicated case. Weights are in double-blocked format,
    // which becomes triple-blocked for bf16 and int8 to become VNNI-friendly.
    // Because of this and batch_size incorporation, offsets below DO NOT work
    // with K not divisible by K block size and batch_size > 1.
    // The problem is it can't be handled properly when batch size is fused,
    // but this allows enable s8s8 and zero-points compensation cases easier.
    int block_size = 0;
    switch (prb->wei_dt()) {
        case dnnl_f32: block_size = 16; break;
        case dnnl_f16: block_size = 16; break;
        case dnnl_bf16: block_size = 32; break;
        case dnnl_u8:
        case dnnl_s8: block_size = 64; break;
        default: break;
    }
    (void)block_size;
    assert(block_size > 1);
    assert(IMPLICATION(prb->batch_size > 1, prb->k % block_size == 0));

    const int64_t src_batch_offset = prb->k;
    const int64_t wei_batch_offset = prb->get_ldb() * prb->k;
    BENCHDNN_PRINT(6, "src_batch_offset=%ld wei_batch_offset=%ld\n",
            (long)src_batch_offset, (long)wei_batch_offset);

    for (size_t i = 0; i < v_batch_element.size(); i++) {
        v_batch_element[i].ptr.A
                = src_ptr + i * src_batch_offset * src_dt.sizeof_dt();
        v_batch_element[i].ptr.B
                = wei_ptr + i * wei_batch_offset * wei_dt.sizeof_dt();
    }

    // Brgemm takes single pointer oscale, but relies on a combination of arg
    // scales attributes. This helps to reuse attributes from primitives, but
    // requires them to pre-compute oscale = src_scale * wei_scale[:]
    auto src_scale = prb->attr.scales.get(DNNL_ARG_SRC);
    auto attr_scale = !wei_scale.is_def() ? wei_scale : src_scale;

    const int64_t count = attr_scale.policy == policy_t::COMMON ? 1 : prb->n;
    dnn_mem_t scales(1, &count, dnnl_f32, tag::x, get_test_engine());
    for (int64_t c = 0; c < count; ++c)
        scales.set_elem(c, prb->scales[c]);

    // Handle output scale common policy separately since the implementation
    // always expects them to be of vector length in case of `common` policy.
    std::vector<float> v16_scales(16, prb->scales[0]);
    const float *scales_ptr = attr_scale.policy == policy_t::COMMON
            ? v16_scales.data()
            : (const float *)scales;

    assert(prb->attr.scales.get(DNNL_ARG_DST).policy == policy_t::COMMON);
    const int64_t dst_scales_count = 1;
    dnn_mem_t dst_scales(
            1, &dst_scales_count, dnnl_f32, tag::x, get_test_engine());
    for (int64_t c = 0; c < dst_scales_count; ++c)
        // precompute inverted dst scales as expected in brgemm implementation
        dst_scales.set_elem(c, 1.f / prb->dst_scales[c]);

    // Handle output scale common policy separately since the implementation
    // always expects them to be of vector length in case of `common` policy.
    std::vector<float> v16_dst_scales(16, 1.f / prb->dst_scales[0]);
    const float *dst_scales_ptr = v16_dst_scales.data();

    char *acc_ptr = (char *)acc_dt;

    const int32_t *dst_zp_ptr = (const int32_t *)prb->dst_zp;
    char *src_comp_ptr = (char *)wei_dt + wei_offset_zp;
    int32_t zp_a_val
            = !prb->attr.zero_points.is_def(DNNL_ARG_SRC) ? prb->src_zp[0] : 0;

    if (!prb->attr.zero_points.is_def(DNNL_ARG_WEIGHTS)) {
        // TODO: weights zero point is not supported yet.
        // It requires enabling f32 -> u8 reorder with compensation on the
        // library side. When enabled, it produces incorrect results for cases
        // with K=1. Likely there's a bug inside. Postpone supporting it.
        return res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED, OK;
    }

    if (prb->attr.post_ops.binary_index() >= 0) {
        // TODO: binary post-op is not supported yet.
        return res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED, OK;
    }

    brgemm_post_ops_data_t post_ops_data(
            /* bias */ bia_dt_ptr,
            /* scales */ scales_ptr, /* binary_post_ops_rhs */ nullptr,
            /* oc_logical_off */ 0, /* dst_row_logical_off */ 0,
            /* data_C_ptr_ */ acc_ptr, /* first_mb_matrix_addr_off */ 0,
            /* a_zp_compensations */ src_comp_ptr,
            /* b_zp_compensations */ nullptr,
            /* c_zp_values */ dst_zp_ptr,
            /* skip_accumulation */ brgemm_attr.generate_skip_accumulation,
            /* zp_a_val */ zp_a_val,
            /* do_only_comp */ false,
            /* do_only_zp_a_val */ false,
            /* dst_scales */ dst_scales_ptr);

    auto scratchpad_size = brgemm_desc.get_wsp_buffer_size();
    std::vector<char> scratchpad(scratchpad_size);
    // Note: hardware lacking native s8s8 support expects compensation buffer
    // passed through a scratchpad argument in postops execution call.
    const bool need_hidden_compensation = scratchpad_size == 0
            && prb->get_dt(SRC) == dnnl_s8 && prb->get_dt(WEI) == dnnl_s8;
    char *scratchpad_ptr = need_hidden_compensation
            ? ((char *)wei_dt + wei_offset_s8s8)
            : scratchpad.data();

    char *dst_ptr = (char *)dst_dt;
    if (use_dst_as_acc) acc_ptr = dst_ptr;

    brgemm_kernel_execute_postops(brgemm_kernel, prb->batch_size,
            v_batch_element.data(), acc_ptr, dst_ptr, post_ops_data,
            scratchpad_ptr);
    res->state = EXECUTED;

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
        if (prb->bia_dt != dnnl_data_type_undef)
            ref_args.set(DNNL_ARG_BIAS, bia_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);
        // Passing accumulator values for `generate_skip_accumulation` check.
        ref_args.set(DNNL_ARG_SRC_1, acc_fp);
        // A hack to pass brgemm attributes to reference since some members
        // change the computation flow for correctness validation.
        dnn_mem_t workspace(src_md, ref_engine, {false, (void *)&brgemm_attr});
        workspace.map();
        ref_args.set(DNNL_ARG_WORKSPACE, workspace);

        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    }

    // Create a bind to match internals to run performance measurements.
    perf_function_t perf_func = std::bind(brgemm_kernel_execute_postops_wrapper,
            brgemm_kernel_, prb->batch_size, v_batch_element.data(), acc_ptr,
            dst_ptr, post_ops_data, scratchpad_ptr, std::placeholders::_1,
            std::placeholders::_2);
    measure_perf(prb->ctx_exe, res, perf_func, args);

    if (is_tmm) DNN_SAFE(amx_tile_release(), WARN);

    return OK;
}

#else

int doit(const prb_t *prb, res_t *res) {
    return OK;
}

#endif

} // namespace brgemm
