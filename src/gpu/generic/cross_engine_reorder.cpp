/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "common/reorder.hpp"
#include "common/utils.hpp"

#include "gpu/generic/cross_engine_reorder.hpp"
#include "gpu/gpu_engine.hpp"
#include "gpu/gpu_stream.hpp"
#include "gpu/gpu_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {

void cross_engine_reorder_t::pd_t::init_scratchpad(impl::engine_t *gpu_engine) {
    if (do_reorder_) {
        using namespace memory_tracking::names;
        auto gpu_align = utils::downcast<gpu::engine_t *>(gpu_engine)
                                 ->get_buffer_alignment();
        auto scratchpad = scratchpad_registry().registrar();
        auto needs_dst = desc()->src_engine_kind == reorder_engine_kind_;
        memory_desc_wrapper wspace((needs_dst) ? dst_md() : src_md());
        scratchpad.book(key_reorder_cross_space, wspace.size(), 1, gpu_align);
        scratchpad.book(key_nested, reorder_pd_->scratchpad_registry().size(),
                1, gpu_align);
    }
}

status_t cross_engine_reorder_t::pd_t::init(impl::engine_t *engine,
        impl::engine_t *src_engine, impl::engine_t *dst_engine) {
    VDISPATCH_REORDER(src_engine != dst_engine, VERBOSE_BAD_ENGINE_KIND);
    VDISPATCH_REORDER(utils::one_of(engine_kind::gpu, src_engine->kind(),
                              dst_engine->kind()),
            VERBOSE_BAD_ENGINE_KIND);
    VDISPATCH_REORDER(attr_ok(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_REORDER(extra_ok(true), VERBOSE_UNSUPPORTED_MD_FLAG, "extra_ok");

    memory_desc_wrapper src_mdw(src_md());
    memory_desc_wrapper dst_mdw(dst_md());

    VDISPATCH_REORDER(!src_mdw.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    gpu::quantization_t src_quant {attr(), src_mdw, DNNL_ARG_SRC};
    gpu::quantization_t dst_quant {attr(), dst_mdw, DNNL_ARG_DST};
    gpu::sum_quantization_t sum_quant {attr()};
    bool with_sum_ab = src_quant.with_scale() || src_quant.with_zp()
            || dst_quant.with_scale() || dst_quant.with_zp()
            || sum_quant.with_scale() || sum_quant.with_zp();
    do_reorder_ = with_sum_ab || src_mdw != dst_mdw;

    impl::engine_t *reorder_engine
            = src_engine->kind() == engine_kind::gpu ? src_engine : dst_engine;

    primitive_attr_t r_attr(*attr());
    if (!r_attr.is_initialized()) return status::out_of_memory;

    auto clean_src_md = *src_md();
    auto clean_dst_md = *dst_md();
    clean_src_md.extra = clean_dst_md.extra = {};
    VDISPATCH_REORDER_SC(
            reorder_primitive_desc_create(reorder_pd_, reorder_engine,
                    &clean_src_md, &clean_dst_md, &r_attr),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "reorder");

    reorder_pd_t::init_desc(
            src_engine->kind(), dst_engine->kind(), true /* is_cross_engine */);

    VDISPATCH_REORDER_SC(maybe_create_zp_precompute_conv_pd(dst_engine),
            "failed to create nested zp precompute convolution");
    init_scratchpad(
            (dst_engine->kind() == engine_kind::gpu) ? dst_engine : src_engine);
    return status::success;
}

status_t cross_engine_reorder_t::init(impl::engine_t *engine) {
    CHECK(pd()->maybe_create_zp_precompute_conv(
            zp_precomp_conv_, engine, this));
    if (!pd()->do_reorder_) return status::success;
    return create_nested_primitive(reorder_, pd()->reorder_pd_, engine);
}

status_t cross_engine_reorder_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    auto *gpu_stream = utils::downcast<gpu::stream_t *>(ctx.stream());

    status_t status = status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_TO);

    std::unique_ptr<memory_t, memory_deleter_t> wspace;
    if (pd()->do_reorder_) {
        auto src_engine_kind = pd()->desc()->src_engine_kind;
        auto reorder_engine_kind = pd()->reorder_engine_kind_;
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                key_reorder_cross_space);
        auto wspace_md = src_engine_kind == reorder_engine_kind
                ? pd()->dst_md()
                : pd()->src_md();
        CHECK(safe_ptr_assign(wspace,
                new memory_t(ctx.stream()->engine(), wspace_md,
                        std::move(scratchpad))));
    }

    auto exec_reorder = [&](const memory_t *src_mem, const memory_t *dst_mem,
                                const memory_t *src_scales_mem,
                                const memory_t *dst_scales_mem) {
        exec_args_t r_args;
        r_args[DNNL_ARG_SRC]
                = memory_arg_t {const_cast<memory_t *>(src_mem), true};
        r_args[DNNL_ARG_DST]
                = memory_arg_t {const_cast<memory_t *>(dst_mem), false};
        r_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC]
                = memory_arg_t {const_cast<memory_t *>(src_scales_mem), true};
        r_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST]
                = memory_arg_t {const_cast<memory_t *>(dst_scales_mem), true};

        exec_ctx_t r_ctx(ctx, std::move(r_args));

        nested_scratchpad_t ns(ctx, key_nested, reorder_);
        r_ctx.set_scratchpad_grantor(ns.grantor());
        return reorder_->execute(r_ctx);
    };

    if (pd()->desc()->src_engine_kind == engine_kind::gpu) {
        // GPU -> CPU or GPU -> GPU
        memory_desc_wrapper dst_mdw(pd()->dst_md());
        if (pd()->do_reorder_) {
            if (pd()->beta() != 0.f) {
                status = gpu_stream->copy(dst, *wspace->memory_storage(),
                        dst_mdw.size(), gpu_stream->ctx().get_deps(),
                        gpu_stream->ctx().get_deps());
            }
            if (status == status::success)
                status = exec_reorder(ctx.input(DNNL_ARG_FROM), wspace.get(),
                        ctx.input(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC),
                        ctx.input(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST));
        }
        if (status == status::success) {
            status = gpu_stream->copy(
                    pd()->do_reorder_ ? *wspace->memory_storage() : src, dst,
                    dst_mdw.size(), gpu_stream->ctx().get_deps(),
                    gpu_stream->ctx().get_deps());
        }
    } else {
        // CPU -> GPU
        memory_desc_wrapper src_mdw(pd()->src_md());
        status = gpu_stream->copy(src,
                pd()->do_reorder_ ? *wspace->memory_storage() : dst,
                src_mdw.size(), gpu_stream->ctx().get_deps(),
                gpu_stream->ctx().get_deps());
        if (status == status::success && pd()->do_reorder_) {
            status = exec_reorder(wspace.get(), ctx.output(DNNL_ARG_TO),
                    ctx.input(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC),
                    ctx.input(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST));
        }
        if (status == status::success)
            status = pd()->maybe_exec_zp_precompute_conv(ctx, zp_precomp_conv_);
    }
    return status;
}

} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
