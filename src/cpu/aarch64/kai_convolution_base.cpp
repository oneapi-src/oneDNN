/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/kai_convolution_base.hpp"

#include <algorithm>
#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/kai_utils.hpp"

#include "kai/ops/bfloat.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/gemm/ndrange.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace data_type;
using namespace kai_utils;

namespace {

bool bias_ok(const kai_convolution_fwd_base_t::pd_t &pd) {
    return !pd.with_bias()
            || pd.invariant_bia_md()->data_type == pd.dst_md()->data_type
            || pd.invariant_bia_md()->data_type == pd.gemm_dst_dt_;
}

bool dense_nhwc(const memory_desc_t &md) {
    const memory_desc_wrapper mdw(md);
    return mdw.is_plain() && mdw.is_dense() && mdw.offset0() == 0
            && mdw.matches_tag(format_tag::nhwc);
}

bool regular_swd_ok(const cpu_convolution_fwd_pd_t &pd) {
    const auto src_dt = pd.invariant_src_md()->data_type;
    const auto wei_dt = pd.invariant_wei_md()->data_type;
    const auto dst_dt = pd.invariant_dst_md()->data_type;

    return (src_dt == f32 && wei_dt == f32 && dst_dt == f32)
            || (src_dt == bf16 && wei_dt == bf16
                    && utils::one_of(dst_dt, bf16, f32))
            || (src_dt == f16 && wei_dt == f16
                    && utils::one_of(dst_dt, f16, f32));
}

double weight_reorder_work(const kai_convolution_fwd_base_t::pd_t &pd) {
    return static_cast<double>(pd.OC()) * pd.IC() * pd.KH() * pd.KW();
}

double kernel_execute_work(const kai_convolution_fwd_base_t::pd_t &pd) {
    return weight_reorder_work(pd) * pd.MB() * pd.OH() * pd.OW();
}

bool accumulation_mode_ok(const cpu_convolution_fwd_pd_t &pd) {
    const auto acc_mode = pd.attr()->acc_mode_;
    if (utils::one_of(acc_mode, accumulation_mode::strict,
                accumulation_mode::relaxed, accumulation_mode::any,
                accumulation_mode::f32))
        return true;

    if (acc_mode == accumulation_mode::f16) {
        return utils::everyone_is(data_type::f16,
                pd.invariant_src_md()->data_type,
                pd.invariant_wei_md()->data_type,
                pd.invariant_dst_md()->data_type);
    }

    return false;
}

} // namespace

std::unique_ptr<kai::ops::IGemmCommon>
kai_convolution_fwd_base_t::pd_t::create_kai_gemm() const {
    return kai_utils::create_kai_gemm(
            *args_, src_md()->data_type, gemm_weights_dt_, gemm_dst_dt_);
}

int kai_convolution_fwd_base_t::pd_t::kernel_maxthreads() const {
    return args_->_maxthreads;
}

bool kai_convolution_fwd_base_t::pd_t::fixed_format() const {
    return args_ && args_->_fixed_format;
}

unsigned int kai_convolution_fwd_base_t::pd_t::gemm_m() const {
    return static_cast<unsigned int>(MB() * OH() * OW());
}

unsigned int kai_convolution_fwd_base_t::pd_t::gemm_k() const {
    return static_cast<unsigned int>(IC() * KH() * KW());
}

bool kai_convolution_fwd_base_t::pd_t::direct_1x1_kernel_ok() const {
    return KH() == 1 && KW() == 1;
}

bool kai_convolution_fwd_base_t::pd_t::direct_1x1_padding_ok() const {
    return padT() == 0 && padL() == 0;
}

bool kai_convolution_fwd_base_t::pd_t::direct_1x1_output_samples_in_bounds()
        const {
    return OH() > 0 && OW() > 0 && (OH() - 1) * KSH() < IH()
            && (OW() - 1) * KSW() < IW();
}

status_t kai_convolution_fwd_base_t::pd_t::init(const engine_t *engine) {
    using primitive_mask_t = primitive_attr_t::skip_mask_t;

    const bool try_fixed_format = weights_md_.format_kind == format_kind::any;

    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(ndims() == 4, VERBOSE_BAD_NDIMS, "src", ndims());
    VDISPATCH_CONV(!with_groups(), VERBOSE_UNSUPPORTED_FEATURE, "groups");
    VDISPATCH_CONV(regular_swd_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_CONV(attr()->has_default_values(primitive_mask_t::fpmath_mode
                                   | primitive_mask_t::accumulation_mode
                                   | primitive_mask_t::post_ops,
                           dst_md()->data_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(accumulation_mode_ok(*this),
            "accumulation mode is not valid for the data type combination");
    VDISPATCH_CONV(set_default_formats_common(format_tag::nhwc,
                           format_tag::hwio, format_tag::nhwc),
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_CONV_SC(attr_.set_default_formats(dst_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");

    VDISPATCH_CONV(dense_nhwc(src_md_), VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_CONV(dense_nhwc(dst_md_), VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_CONV(memory_desc_matches_tag(weights_md_, format_tag::hwio),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");
    VDISPATCH_CONV(memory_desc_wrapper(weights_md_).is_dense()
                    && weights_md_.offset0 == 0,
            VERBOSE_UNSUPPORTED_TAG_S, "weights");
    VDISPATCH_CONV(!with_bias()
                    || (memory_desc_matches_tag(bias_md_, format_tag::x)
                            && memory_desc_wrapper(bias_md_).is_dense()
                            && bias_md_.offset0 == 0),
            VERBOSE_UNSUPPORTED_TAG_S, "bias");

    gemm_dst_dt_ = dst_md()->data_type;
    if (with_bias() && invariant_bia_md()->data_type == data_type::f32
            && utils::one_of(
                    dst_md()->data_type, data_type::f16, data_type::bf16)) {
        gemm_dst_dt_ = data_type::f32;
    }

    // brgconv is faster for small OC * IC. We replicate the ACL heuristic to
    // avoid edge case regressions blocking removing ACL, longer term we should
    // re-assess this heuristic specifically for KleidiAI conv
    const bool any_f16 = utils::one_of(data_type::f16,
            invariant_src_md()->data_type, invariant_wei_md()->data_type,
            invariant_dst_md()->data_type);
    VDISPATCH_CONV(OC() * IC() > 2048 || !mayiuse(sve) || any_f16,
            "brgconv:sve is faster for small OC * IC");

    VDISPATCH_CONV(bias_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);

    const bool fast_mode = use_fast_mode(*src_md(), *attr());
    const size_t src_dt_size = types::data_type_size(src_md()->data_type);

    VDISPATCH_CONV(num_sum_post_ops(attr_.post_ops_) <= 1,
            "supports at most one sum post op");
    const auto post_ops_fusion
            = create_post_ops_fusion(attr_.post_ops_, !with_bias());
    CHECK(post_ops.init(engine, attr_.post_ops_, *dst_md(),
            post_ops_fusion.fallback_start_index));
    has_post_ops_fallback_ = post_ops_fusion.has_fallback(attr_.post_ops_);

    name_ = impl_base_name();
    if (has_post_ops_fallback_) name_ += "+post_ops_fallback";

    gemm_weights_dt_ = weights_md()->data_type;
    run_weight_reorder_ = false;
    use_dst_reorder_ = false;
    dst_reorder_pd_.reset();
    tmp_dst_md_ = memory_desc_t {};

    CHECK(init_datapath(engine));

    const int max_threads = dnnl_get_current_num_threads();
    const int num_threads = threads_for_kernel_execute(
            kernel_execute_work(*this), max_threads);
    args_ = std::make_shared<kai::ops::GemmArgs>(get_cpu_info(), gemm_m(), OC(),
            gemm_k(), gemm_k_sections(), gemm_n_batches(), gemm_n_multi(),
            uses_indirect_gemm(), post_ops_fusion.activation, num_threads,
            try_fixed_format, fast_mode, post_ops_fusion.accumulate);

    std::unique_ptr<kai::ops::IGemmCommon> kernel = create_kai_gemm();
    const bool fixed_format_failed = fixed_format()
            && (!kernel
                    || !is_fixed_format(kernel->get_config().weight_format));

    if (fixed_format_failed) {
        args_->_fixed_format = false;
        kernel = create_kai_gemm();
    }
    VDISPATCH_CONV(kernel, VERBOSE_UNSUPPORTED_DT_CFG);

    kai::ops::GemmConfig kernel_cfg = kernel->get_config();

    if (fixed_format()) {
        constexpr dim_t O_dim = 0;
        constexpr dim_t I_dim = 1;
        constexpr dim_t H_dim = 2;
        constexpr dim_t W_dim = 3;
        weight_format_to_memory_desc(weights_md_, kernel_cfg.weight_format,
                I_dim, O_dim, {W_dim, H_dim});
    }

    run_weight_reorder_ = !fixed_format() && kernel->B_is_pretransposed();

    auto scratchpad = scratchpad_registry().registrar();
    if (kernel->get_working_size() != 0) {
        // Match ACL's GEMM workspace alignment. Cache-line alignment alone
        // can regress interleaved convolutions on large packed-input buffers.
        scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                kernel->get_working_size(), 1, 4096, 4096);
    }

    if (run_weight_reorder_) {
        scratchpad.book(memory_tracking::names::key_conv_permuted_weights,
                kernel->get_B_pretransposed_array_size(), 1);
    }

    if (gemm_dst_dt_ != dst_md()->data_type) {
        CHECK(memory_desc_init_by_tag(tmp_dst_md_, dst_md()->ndims,
                dst_md()->dims, gemm_dst_dt_, format_tag::nhwc));
        VDISPATCH_CONV_SC(reorder_primitive_desc_create(dst_reorder_pd_, engine,
                                  &tmp_dst_md_, dst_md()),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "dst reorder");
        use_dst_reorder_ = true;

        const memory_desc_wrapper tmp_dst_d(&tmp_dst_md_);
        scratchpad.book(memory_tracking::names::key_conv_ncsp_dst,
                tmp_dst_d.size(), 1, 64, 64);
        scratchpad.book(memory_tracking::names::key_nested,
                dst_reorder_pd_->scratchpad_registry());
    }

    if (post_ops.has_sum()) {
        const memory_desc_wrapper dst_d(dst_md());
        scratchpad.book(memory_tracking::names::key_generic_acc, dst_d.size(),
                1, 64, 64);
    }

    book_datapath_scratchpad(scratchpad, src_dt_size);

    post_ops.init_scratchpad(scratchpad);

    return status::success;
}

status_t kai_convolution_fwd_base_t::init(engine_t *engine) {
    if (pd()->dst_reorder_pd_)
        CHECK(pd()->dst_reorder_pd_->create_primitive(dst_reorder_, engine));
    post_ops_ = pd()->post_ops;
    CHECK(post_ops_.init_primitives(engine));
    return status::success;
}

status_t kai_convolution_fwd_base_t::execute(const exec_ctx_t &ctx) const {
    const auto *pd = this->pd();
    const auto *src_md = pd->src_md();
    const auto *dst_md = pd->dst_md();
    const auto *weights_md = pd->weights_md();
    const bool run_weight_reorder = pd->run_weight_reorder_;
    const bool with_bias = pd->with_bias();
    const bool use_dst_reorder = pd->use_dst_reorder_;
    const bool fixed_format = pd->fixed_format();
    const bool has_post_ops_fallback = pd->has_post_ops_fallback_;
    const auto &post_ops = post_ops_;
    const bool post_ops_has_sum = post_ops.has_sum();
    const dim_t OC = pd->OC();
    const dim_t OH = pd->OH();
    const dim_t OW = pd->OW();
    const auto &scratchpad = ctx.get_scratchpad_grantor();

    const int max_threads = dnnl_get_current_num_threads();
    // Kernel selection and workspace sizes depend on the thread count. Reuse
    // the creation-time configuration even if the execution context changes.
    int kernel_num_threads = std::min(pd->kernel_maxthreads(), max_threads);

    std::unique_ptr<kai::ops::IGemmCommon> kernel = pd->create_kai_gemm();
    if (!kernel) return status::runtime_error;

    if (get_verbose(verbose_t::profile_externals)) {
        std::cout << "profile_externals: " << kernel->get_config().filter
                  << std::endl;
    }

    const kai::ops::ndrange_t window_size = kernel->get_window_size();
    const auto thread_partition
            = make_thread_partition(kernel_num_threads, window_size);
    kernel_num_threads = thread_partition.nthr;

    kernel->set_nthreads(kernel_num_threads);

    const auto *src_base = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto *raw_wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *wei_base = const_cast<void *>(raw_wei);
    if (run_weight_reorder) {
        wei_base = scratchpad.get<void>(
                memory_tracking::names::key_conv_permuted_weights);
    }

    void *dst_base = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *bias_base
            = with_bias ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS) : nullptr;

    constexpr int wei_o_dim = 0;

    // Tag matching ignores strides on singleton dimensions. GEMM may flatten
    // those dimensions into M or K, so use the canonical dense strides rather
    // than the arbitrary strides supplied for dimensions of size one.
    const int ld_src = static_cast<int>(pd->IC());
    const int ld_dst = static_cast<int>(OC);
    const int ld_wei = static_cast<int>(fixed_format
                    ? weights_md->format_desc.blocking.strides[wei_o_dim]
                    : OC);

    const int src_batch_stride
            = static_cast<int>(pd->IC() * pd->IH() * pd->IW());
    const int src_h_stride = static_cast<int>(pd->IW()) * ld_src;
    const int dst_h_stride = static_cast<int>(OW * OC);
    const int dst_batch_stride = static_cast<int>(OH * OW * OC);
    const size_t src_dt_size = types::data_type_size(src_md->data_type);
    const size_t src_col_stride_bytes
            = static_cast<size_t>(ld_src) * src_dt_size;
    const size_t src_h_stride_bytes
            = static_cast<size_t>(src_h_stride) * src_dt_size;
    const size_t src_batch_stride_bytes
            = static_cast<size_t>(src_batch_stride) * src_dt_size;
    void *kernel_dst_base = nullptr;
    if (!use_dst_reorder) {
        kernel_dst_base = post_ops_has_sum
                ? scratchpad.get<void>(memory_tracking::names::key_generic_acc)
                : dst_base;
    } else {
        kernel_dst_base = scratchpad.get<void>(
                memory_tracking::names::key_conv_ncsp_dst);
    }

    if (run_weight_reorder) {
        parallel_pretranspose_B_array(*kernel, wei_base, raw_wei, ld_wei, 0,
                false, kernel_num_threads);
    }

    if (kernel->get_working_size() != 0) {
        kernel->set_working_space(scratchpad.get<void>(
                memory_tracking::names::key_gemm_asm_tmp_buffer));
    }

    CHECK(setup_kernel_arrays(kernel_call_args_t {ctx, *pd, *kernel, scratchpad,
            max_threads, kernel_num_threads, src_base, wei_base,
            kernel_dst_base, bias_base, ld_src, ld_wei, ld_dst, src_h_stride,
            src_batch_stride, dst_h_stride, dst_batch_stride, src_dt_size,
            src_col_stride_bytes, src_h_stride_bytes, src_batch_stride_bytes}));

    parallel_execute(*kernel, window_size, thread_partition);

    void *post_ops_src = kernel_dst_base;
    if (use_dst_reorder) {
        post_ops_src = post_ops_has_sum
                ? scratchpad.get<void>(memory_tracking::names::key_generic_acc)
                : dst_base;

        if (!use_dst_reorder || !dst_reorder_) return status::runtime_error;

        auto *engine = ctx.stream()->engine();
        std::unique_ptr<memory_t, memory_deleter_t> tmp_dst_mem(new memory_t(
                engine, &pd->tmp_dst_md_, use_runtime_ptr, kernel_dst_base));
        std::unique_ptr<memory_t, memory_deleter_t> dst_mem(
                new memory_t(engine, dst_md, use_runtime_ptr, post_ops_src));

        exec_args_t reorder_args;
        reorder_args[DNNL_ARG_SRC] = {tmp_dst_mem.get(), true};
        reorder_args[DNNL_ARG_DST] = {dst_mem.get(), false};
        exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));

        auto *nested_grantor = memory_tracking::create_nested_grantor(
                ctx.get_scratchpad_grantor(),
                memory_tracking::names::key_nested,
                dst_reorder_->pd()->scratchpad_registry());
        reorder_ctx.set_scratchpad_grantor(nested_grantor);
        CHECK(dst_reorder_->execute(reorder_ctx));
    }

    if (has_post_ops_fallback) {
        if (post_ops_has_sum)
            CHECK(post_ops.execute(ctx, post_ops_src, dst_base));
        else
            CHECK(post_ops.execute(ctx, post_ops_src));
    }

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
