/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "ocl/cross_engine_reorder.hpp"
#include "common/utils.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

static dnnl_engine_kind_t cross_engine_reorder_engine_kind = dnnl_gpu;

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_reorder_set_engine_kind(
        dnnl_engine_kind_t engine_kind) {
    cross_engine_reorder_engine_kind = engine_kind;
    return dnnl_success;
}

namespace dnnl {
namespace impl {
namespace ocl {

status_t cross_engine_reorder_t::pd_t::init() {
    bool args_ok = src_engine() != dst_engine()
            && utils::one_of(engine_kind::gpu, src_engine()->kind(),
                    dst_engine()->kind());

    if (!args_ok) return status::unimplemented;

    reorder_engine_kind_ = cross_engine_reorder_engine_kind;

    memory_desc_wrapper src_mdw(src_md());
    memory_desc_wrapper dst_mdw(dst_md());

    // Do not use CPU reorders in the following cases:
    // 1. For 4o8i8o4i-like formats as they assume GPU-specific permutation
    // 2. For GPU <-> GPU reorders with different engines
    if (src_mdw.matches_one_of_tag(
                format_tag::OIhw4o8i8o4i, format_tag::gOIhw4o8i8o4i)
            || dst_mdw.matches_one_of_tag(
                    format_tag::OIhw4o8i8o4i, format_tag::gOIhw4o8i8o4i)
            || utils::everyone_is(engine_kind::gpu, src_engine()->kind(),
                    dst_engine()->kind())) {
        reorder_engine_kind_ = engine_kind::gpu;
    }

    engine_t *reorder_engine = src_engine()->kind() == reorder_engine_kind_
            ? src_engine()
            : dst_engine();

    auto r_impls = reorder_engine->get_reorder_implementation_list();
    const primitive_attr_t r_attr(*attr());
    for (auto r = r_impls; *r; ++r) {
        reorder_pd_t *r_pd = nullptr;
        if ((*r)(&r_pd, reorder_engine, &r_attr, reorder_engine, src_md(),
                    reorder_engine, dst_md())
                == status::success) {
            reorder_.reset(r_pd);
            break;
        }
    }

    if (!reorder_) return status::unimplemented;

    return status::success;
}

status_t cross_engine_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_TO);

    auto exec_reorder = [&](const memory_t *src_mem, const memory_t *dst_mem) {
        exec_args_t r_args;
        r_args[DNNL_ARG_SRC]
                = memory_arg_t {const_cast<memory_t *>(src_mem), true};
        r_args[DNNL_ARG_DST]
                = memory_arg_t {const_cast<memory_t *>(dst_mem), false};

        exec_ctx_t r_ctx(ctx.stream(), std::move(r_args));
        return reorder_->execute(r_ctx);
    };

    status_t status = status::success;
    if (pd()->src_engine()->kind() == pd()->reorder_engine_kind_) {
        if (do_reorder_) {
            status = exec_reorder(ctx.input(DNNL_ARG_FROM), temp_buf.get());
        }
        if (status == status::success) {
            memory_desc_wrapper dst_mdw(pd()->dst_md());
            status = compute_stream->copy(
                    do_reorder_ ? *temp_buf->memory_storage() : src, dst,
                    dst_mdw.size());
        }
    } else {
        memory_desc_wrapper src_mdw(pd()->src_md());
        status = compute_stream->copy(src,
                do_reorder_ ? *temp_buf->memory_storage() : dst,
                src_mdw.size());
        if (status == status::success && do_reorder_)
            status = exec_reorder(temp_buf.get(), ctx.output(DNNL_ARG_TO));
    }
    return status;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl
