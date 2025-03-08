/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/gpu_engine.hpp"
#include "gpu/gpu_stream.hpp"
#include "gpu/gpu_zero_points_conv.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

status_t gpu_reorder_pd_t::maybe_create_zp_precompute_conv_pd(
        impl::engine_t *dst_engine) {
    memory_desc_wrapper dst_mdw(dst_md());
    auto &extra = dst_mdw.extra();
    auto needs_conv = memory_extra_flags::compensation_gpu_conv_asymmetric_src;
    auto is_dst_gpu = (dst_engine->kind() == engine_kind::gpu);
    do_zp_precomp_conv_ = is_dst_gpu && (extra.flags & needs_conv);
    if (!do_zp_precomp_conv_) return status::success;

    using namespace memory_extra_flags;
    const auto out_type = data_type::f32;
    primitive_attr_t attr;
    const bool is_bwd_d
            = extra.flags & compensation_gpu_conv_asymmetric_src_bwd;
    auto prop = (is_bwd_d) ? prop_kind::backward_data
                           : prop_kind::forward_inference;
    CHECK(create_zp_precompute_conv_pd(zp_precomp_conv_pd_, dst_engine, attr,
            dst_md(), extra.idhw, extra.odhw, extra.pdhw, extra.ddhw, out_type,
            prop));

    using namespace memory_tracking::names;
    auto gpu_align = utils::downcast<gpu::engine_t *>(dst_engine)
                             ->get_buffer_alignment();
    auto scratchpad = scratchpad_registry().registrar();
    const auto &registry = zp_precomp_conv_pd_->scratchpad_registry();
    memory_desc_wrapper wspace((is_bwd_d) ? zp_precomp_conv_pd_->diff_dst_md()
                                          : zp_precomp_conv_pd_->src_md());
    scratchpad.book(key_conv_tr_src, wspace.size(), 1, gpu_align);
    scratchpad.book(key_conv_tails, registry.size(), 1, gpu_align);
    return status::success;
}

status_t gpu_reorder_pd_t::maybe_create_zp_precompute_conv(
        std::shared_ptr<impl::primitive_t> &zp_precomp_conv,
        impl::engine_t *engine, gpu::primitive_t *primitive) const {
    if (!do_zp_precomp_conv_) return status::success;
    return primitive->create_nested_primitive(
            zp_precomp_conv, zp_precomp_conv_pd_, engine);
}

status_t gpu_reorder_pd_t::maybe_exec_zp_precompute_conv(const exec_ctx_t &ctx,
        const std::shared_ptr<impl::primitive_t> &zp_precomp_conv) const {
    using namespace memory_tracking::names;
    if (!do_zp_precomp_conv_) return status::success;

    const bool is_bwd_d = (zp_precomp_conv_pd_->get_prop_kind()
            == prop_kind::backward_data);
    auto *gpu_stream = utils::downcast<gpu::stream_t *>(ctx.stream());
    auto conv_md_in = (is_bwd_d) ? zp_precomp_conv_pd_->diff_dst_md()
                                 : zp_precomp_conv_pd_->src_md();
    auto scratchpad
            = ctx.get_scratchpad_grantor().get_memory_storage(key_conv_tr_src);
    std::unique_ptr<memory_t, memory_deleter_t> wspace;
    CHECK(safe_ptr_assign(wspace,
            new memory_t(ctx.stream()->engine(), conv_md_in,
                    std::move(scratchpad))));
    CHECK(gpu_stream->fill(*wspace->memory_storage(), 0x01,
            memory_desc_wrapper(conv_md_in).size(),
            gpu_stream->ctx().get_deps(), gpu_stream->ctx().get_deps()));

    exec_args_t r_args;
    auto arg_in = (is_bwd_d) ? DNNL_ARG_DIFF_DST : DNNL_ARG_SRC;
    auto arg_out = (is_bwd_d) ? DNNL_ARG_DIFF_SRC : DNNL_ARG_DST;
    r_args[arg_in] = memory_arg_t {(memory_t *)wspace.get(), true};
    r_args[DNNL_ARG_WEIGHTS] = memory_arg_t {ctx.output(DNNL_ARG_TO), true};
    r_args[arg_out] = memory_arg_t {ctx.output(DNNL_ARG_TO), false};
    exec_ctx_t r_ctx(ctx, std::move(r_args));

    nested_scratchpad_t ns(ctx, key_conv_tails, zp_precomp_conv);
    r_ctx.set_scratchpad_grantor(ns.grantor());
    return zp_precomp_conv->execute(r_ctx);
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
