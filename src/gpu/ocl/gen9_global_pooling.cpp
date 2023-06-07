/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "gpu/ocl/gen9_global_pooling.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

int calculate_spatial_chunk(const pool_conf_t &conf, engine_t *engine) {
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    const int hw_threads = compute_engine->device_info()->hw_threads();
    const bool is_xe_hp_plus = compute_engine->is_xe_hp()
            || compute_engine->is_xe_hpg() || compute_engine->is_xe_hpc();

    const int spatial_dim = conf.id * conf.ih * conf.iw;
    int chunk_size = spatial_dim;

    // Experimentally selected values for XeHP family
    const int desired_wi_per_thread = is_xe_hp_plus && conf.is_plain ? 1024 : 4;

    const auto get_work_items_num = [&]() {
        return conf.c * conf.mb * utils::div_up(spatial_dim, chunk_size);
    };
    while (get_work_items_num() < hw_threads * desired_wi_per_thread
            && chunk_size > 1) {
        chunk_size = utils::div_up(chunk_size, 2);
    }
    return chunk_size;
}

static status_t init_conf_common(pool_conf_t &conf, offsets_t &off,
        const pooling_pd_t *pd, engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    set_default_pool_conf(conf, *pd->desc(), *pd->invariant_src_md(),
            *pd->invariant_dst_md(), *pd->attr());

    if (conf.iw != conf.kw || conf.ih != conf.kh || conf.ow * conf.oh != 1)
        return status::unimplemented;

    const memory_desc_wrapper src_mdw(pd->invariant_src_md());
    const memory_desc_wrapper dst_mdw(pd->invariant_dst_md());
    const auto &padded_src_dims = src_mdw.padded_dims();
    const auto &padded_dst_dims = dst_mdw.padded_dims();
    if (utils::array_product(padded_src_dims + 2, conf.ndims - 2)
                    != conf.id * conf.ih * conf.iw
            || utils::array_product(padded_dst_dims + 2, conf.ndims - 2)
                    != conf.od * conf.oh * conf.ow)
        return status::unimplemented;

    using namespace dnnl::impl::alg_kind;
    if (!conf.is_backward) {
        if (conf.alg == pooling_max) {
            // gen9_global_pooling_fwd doesn't support zero padding.
            if (conf.mb != conf.mb_padded || conf.c != conf.c_padded)
                return status::unimplemented;
        }
        // heuristics: for small shapes, gen9_pooling_fwd provides better perf.
        if (conf.kd * conf.kh * conf.kw < 128) return status::unimplemented;
    }

    set_offsets(src_mdw, off.src_off);
    set_offsets(dst_mdw, off.dst_off);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    conf.is_plain = src_mdw.is_plain();
    conf.global_pool_spatial_chunk = calculate_spatial_chunk(conf, engine);

    const int spatial_dim_padded = utils::rnd_up(
            conf.id * conf.ih * conf.iw, conf.global_pool_spatial_chunk);
    conf.dispatch = compute_engine->create_dispatch(src_mdw.md_);
    conf.dispatch.define_dim("MB", 0, conf.mb_padded);
    conf.dispatch.define_dim("C", 1, conf.c_padded);
    if (conf.is_backward) {
        conf.dispatch.define_dim("SPATIAL", 2, spatial_dim_padded,
                conf.global_pool_spatial_chunk);
        conf.sub_group_size = compute_engine->device_info()->max_subgroup_size(
                src_mdw.data_type());
        if (conf.c % conf.sub_group_size != 0) conf.vectorize = false;
        if ((src_mdw.blocking_desc().strides[1] != 1) || !src_mdw.is_plain()
                || (dst_mdw.blocking_desc().strides[1] != 1)
                || !dst_mdw.is_plain())
            conf.vectorize = false;
        if (conf.vectorize) {
            CHECK(conf.dispatch.vectorize_dim("C", conf.sub_group_size));
        }
    }
    conf.dispatch.generate();

    conf.attr_info = attr_info_t::create(pd->attr());

    return status::success;
};

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const pool_conf_t &conf, const offsets_t &off,
        const post_ops_t &post_ops) {
    using namespace dnnl::impl::alg_kind;
    kernel_ctx.set_data_type(conf.src_dt);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("C", conf.c);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("SPATIAL_DIM", conf.id * conf.ih * conf.iw);
    kernel_ctx.define_int("SPATIAL_CHUNK", conf.global_pool_spatial_chunk);
    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("IS_BWD", conf.is_backward);
    kernel_ctx.define_int("IS_FWD", !conf.is_backward);
    kernel_ctx.define_int("IS_VECTORIZED", conf.vectorize);

    kernel_ctx.define_int("ALG_MAX", (conf.alg == pooling_max));
    kernel_ctx.define_int(
            "ALG_AVG_NP", (conf.alg == pooling_avg_exclude_padding));
    kernel_ctx.define_int(
            "ALG_AVG_P", (conf.alg == pooling_avg_include_padding));
    kernel_ctx.define_int("NEED_ZERO_PADDING",
            (conf.mb != conf.mb_padded || conf.c != conf.c_padded));

    CHECK(def_attr_info(
            kernel_ctx, conf.attr_info, post_ops, conf.dst_md_info.dims));

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);
    def_offsets(off.dst_off, kernel_ctx, "DST", conf.ndims);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t gen9_global_pooling_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t gen9_global_pooling_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off, attr()->post_ops_);
}

void gen9_global_pooling_fwd_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    size_t size = utils::array_product(
            conf.dst_md_info.padded_dims, conf.dst_md_info.ndims);
    scratchpad.book(memory_tracking::names::key_pool_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);

    scratchpad.book(memory_tracking::names::key_nested,
            reduction_pd_->scratchpad_registry());
}

status_t gen9_global_pooling_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    if (!reduction_p_) {
        auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
        auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
        auto &ws = CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE);

        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, ws);
        arg_list.set(2, dst);

        auto nd_range = pd()->conf.dispatch.nd_range();

        return parallel_for(ctx, nd_range, kernel_, arg_list);
    } else {

        exec_args_t reduction_args;
        reduction_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_SRC);
        reduction_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
        exec_ctx_t reduction_ctx(ctx, std::move(reduction_args));

        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested, reduction_p_);
        reduction_ctx.set_scratchpad_grantor(ns.grantor());

        // Executing the reduction kernel
        return reduction_p_->execute(reduction_ctx);
    }
}

status_t gen9_global_pooling_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t gen9_global_pooling_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off, attr()->post_ops_);
}

status_t gen9_global_pooling_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, ws);
    arg_list.set(2, diff_dst);

    auto nd_range = pd()->conf.dispatch.nd_range();

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
