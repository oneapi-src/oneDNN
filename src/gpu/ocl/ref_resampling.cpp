/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "gpu/ocl/ref_resampling.hpp"
#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// -------- Common functions ----------- //

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const resampling_conf_t &conf, const resampling_desc_t *desc) {
    switch (desc->alg_kind) {
        case alg_kind::resampling_nearest:
            kernel_ctx.define_int("RESAMPLING_ALG_NEAREST", 1);
            break;
        case alg_kind::resampling_linear:
            kernel_ctx.define_int("RESAMPLING_ALG_LINEAR", 1);
            break;
        default: return status::unimplemented;
    }

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.MB);
    kernel_ctx.define_int("C", conf.C);
    kernel_ctx.define_int("ID", conf.ID);
    kernel_ctx.define_int("IH", conf.IH);
    kernel_ctx.define_int("IW", conf.IW);
    kernel_ctx.define_int("OD", conf.OD);
    kernel_ctx.define_int("OH", conf.OH);
    kernel_ctx.define_int("OW", conf.OW);
    kernel_ctx.define_float("FD", conf.FD);
    kernel_ctx.define_float("FH", conf.FH);
    kernel_ctx.define_float("FW", conf.FW);

    def_offsets(conf.off.src_off, kernel_ctx, "SRC", conf.ndims);
    def_offsets(conf.off.dst_off, kernel_ctx, "DST", conf.ndims);

    def_dispatch(kernel_ctx, conf.dispatch);
    return status::success;
}

// ---------- ref_resampling_fwd_t ------------ //

status_t ref_resampling_fwd_t::pd_t::init_conf(engine_t *engine) {

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(dst_md());

    conf.dispatch.define_dim("MB", 0, dst_md()->padded_dims[0]);
    conf.dispatch.define_dim("C", 1, dst_md()->padded_dims[1]);
    conf.dispatch.define_dim("OD", nstl::max(2, dst_md()->ndims - 3), OD());
    conf.dispatch.define_dim("OH", nstl::max(2, dst_md()->ndims - 2), OH());
    conf.dispatch.define_dim("OW", nstl::max(2, dst_md()->ndims - 1), OW());
    conf.dispatch.generate();

    conf.ndims = dst_md()->ndims;

    const memory_desc_wrapper src_d(src_md());
    set_offsets(src_d, conf.off.src_off);

    const memory_desc_wrapper dst_d(dst_md());
    set_offsets(dst_d, conf.off.dst_off);

    conf.MB = MB();
    conf.C = C();
    conf.ID = ID();
    conf.IH = IH();
    conf.IW = IW();
    conf.OD = OD();
    conf.OH = OH();
    conf.OW = OW();
    conf.FD = FD();
    conf.FH = FH();
    conf.FW = FW();

    conf.attr_info = attr_info_t::create(attr());

    return status::success;
}

status_t ref_resampling_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(src_md()->data_type);
    kernel_ctx.define_int("IS_FWD", 1);

    status_t status = init_kernel_ctx_common(kernel_ctx, conf, desc());

    def_data_type(kernel_ctx, src_md()->data_type, "SRC");
    def_data_type(kernel_ctx, dst_md()->data_type, "DST");

    // Set post-op variables
    CHECK(def_attr_info(
            kernel_ctx, conf.attr_info, attr()->post_ops_, dst_md()->dims));

    return status;
}

status_t ref_resampling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    append_post_ops_to_arg_list(ctx, arg_list, 2, pd()->attr()->post_ops_);

    auto nd_range = pd()->conf.dispatch.nd_range();

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

// -------- ref_resampling_bwd_t ---------- //

status_t ref_resampling_bwd_t::pd_t::init_conf(engine_t *engine) {

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(diff_src_md());

    conf.dispatch.define_dim("MB", 0, diff_src_md()->padded_dims[0]);
    conf.dispatch.define_dim("C", 1, diff_src_md()->padded_dims[1]);
    conf.dispatch.define_dim(
            "ID", nstl::max(2, diff_src_md()->ndims - 3), ID());
    conf.dispatch.define_dim(
            "IH", nstl::max(2, diff_src_md()->ndims - 2), IH());
    conf.dispatch.define_dim(
            "IW", nstl::max(2, diff_src_md()->ndims - 1), IW());
    conf.dispatch.generate();

    conf.ndims = diff_dst_md()->ndims;

    const memory_desc_wrapper diff_src_d(diff_src_md());
    set_offsets(diff_src_d, conf.off.src_off);

    const memory_desc_wrapper diff_dst_d(diff_dst_md());
    set_offsets(diff_dst_d, conf.off.dst_off);

    conf.MB = MB();
    conf.C = C();
    conf.ID = ID();
    conf.IH = IH();
    conf.IW = IW();
    conf.OD = OD();
    conf.OH = OH();
    conf.OW = OW();
    conf.FD = FD();
    conf.FH = FH();
    conf.FW = FW();

    conf.attr_info = attr_info_t::create(attr());

    return status::success;
}

status_t ref_resampling_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(diff_src_md()->data_type);
    kernel_ctx.define_int("IS_BWD", 1);

    status_t status = init_kernel_ctx_common(kernel_ctx, conf, desc());

    def_data_type(kernel_ctx, diff_src_md()->data_type, "SRC");
    def_data_type(kernel_ctx, diff_dst_md()->data_type, "DST");

    return status;
}

status_t ref_resampling_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, diff_dst);

    auto nd_range = pd()->conf.dispatch.nd_range();

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
