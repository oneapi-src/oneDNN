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

#include "gpu/ocl/vectorized_resampling.hpp"
#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// -------- Common functions ----------- //

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const resampling_conf_t &conf, const resampling_desc_t *desc) {

    using namespace alg_kind;

    status_t status = status::success;

    kernel_ctx.define_int("IS_BWD", 1);

    switch (desc->alg_kind) {
        case resampling_nearest:
            kernel_ctx.define_int("RESAMPLING_ALG_NEAREST", 1);
            break;
        case resampling_linear:
            kernel_ctx.define_int("RESAMPLING_ALG_LINEAR", 1);
            break;
        default: status = status::unimplemented;
    }
    if (status != status::success) return status;

    const memory_desc_wrapper diff_src_d(desc->diff_src_desc);
    const memory_desc_wrapper diff_dst_d(desc->diff_dst_desc);
    const int ndims = diff_dst_d.ndims();

    // Compute strides and set variables
    kernel_ctx.define_int("MB_STRIDE", conf.padded_strides[0]);
    kernel_ctx.define_int("C_STRIDE", conf.padded_strides[1]);
    kernel_ctx.define_int(
            "ID_STRIDE", ndims < 5 ? 1 : conf.padded_strides[ndims - 3]);
    kernel_ctx.define_int(
            "IH_STRIDE", ndims < 4 ? 1 : conf.padded_strides[ndims - 2]);
    kernel_ctx.define_int(
            "IW_STRIDE", ndims < 3 ? 1 : conf.padded_strides[ndims - 1]);

    // kernel_ctx.define_int("VECT_SIZE", conf.vect_size);
    kernel_ctx.define_int("VECT_DT_N", conf.vect_size);
    kernel_ctx.define_int("GWS_WITH_SG_DEFAULT", 1);
    kernel_ctx.define_int("GWS_LWS0_DEFAULT", conf.lws[0]);
    kernel_ctx.define_int("GWS_LWS1_DEFAULT", conf.lws[1]);
    kernel_ctx.define_int("GWS_LWS2_DEFAULT", conf.lws[2]);
    kernel_ctx.define_int("GWS_SGS_DEFAULT", conf.sub_group_size);

    const size_t dst_size = types::data_type_size(diff_dst_d.data_type());
    kernel_ctx.define_int("ALIGNED_READ",
            conf.C * dst_size % (4 * conf.vect_size) == 0 ? 1 : 0);
    const size_t src_size = types::data_type_size(diff_src_d.data_type());
    kernel_ctx.define_int("ALIGNED_WRITE",
            conf.C * src_size % (4 * conf.vect_size) == 0 ? 1 : 0);

    kernel_ctx.define_int("NDIMS", ndims);
    kernel_ctx.define_int("MB", conf.MB);
    kernel_ctx.define_int("C", conf.C);
    kernel_ctx.define_int("PADDED_C", conf.padded_c);
    kernel_ctx.define_int("ID", conf.ID);
    kernel_ctx.define_int("IH", conf.IH);
    kernel_ctx.define_int("IW", conf.IW);
    kernel_ctx.define_int("OD", conf.OD);
    kernel_ctx.define_int("OH", conf.OH);
    kernel_ctx.define_int("OW", conf.OW);
    kernel_ctx.define_float("FD", conf.FD);
    kernel_ctx.define_float("FH", conf.FH);
    kernel_ctx.define_float("FW", conf.FW);

    const int max_d = std::max(1, (int)std::ceil(conf.FD * 1.5 - 0.5));
    const int max_h = std::max(1, (int)std::ceil(conf.FH * 1.5 - 0.5));
    const int max_w = std::max(1, (int)std::ceil(conf.FW * 1.5 - 0.5));
    kernel_ctx.define_int("MAX_NUM_D", max_d);
    kernel_ctx.define_int("MAX_NUM_H", max_h);
    kernel_ctx.define_int("MAX_NUM_W", max_w);

    def_offsets(conf.off.src_off, kernel_ctx, "SRC", ndims);
    def_offsets(conf.off.dst_off, kernel_ctx, "DST", ndims);
    return status::success;
}

status_t vectorized_resampling_bwd_t::pd_t::init_conf(engine_t *engine) {
    using namespace data_type;
    assert(engine->kind() == engine_kind::gpu);
    bool ok = !is_fwd() && set_default_params() == status::success
            && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    const memory_desc_wrapper diff_src_d(diff_src_md());
    const memory_desc_wrapper diff_dst_d(diff_dst_md());

    // Restriction: Only works for axb cases
    using namespace dnnl::impl::format_tag;
    const bool is_axb = (diff_src_d.matches_one_of_tag(acb, acdb, acdeb)
                    != format_tag::undef
            && diff_src_d.matches_one_of_tag(abc, abcd, abcde)
                    == format_tag::undef);
    if (!is_axb) { return status::unimplemented; }

    // ------- Heuristics -------- //
    // Tuned for PVC
    // TODO: Use hw config to determine optimal heuristics

    conf.vect_size = 4;
    conf.lws[0] = 512;
    conf.sub_group_size = 32;

    // For large cases where cache reuse is less likely, use smaller lws to increase parallelism via thread dispatching
    dim_t num_wi_estimate = diff_src_md()->padded_dims[0]
            * diff_src_md()->padded_dims[1] * ID() * IH() * IW()
            / conf.vect_size;
    if (num_wi_estimate > 1 >> 20) { conf.lws[0] = 256; }

    // ------ End of Heuristics ------- //

    // Padded C: multiple of sub_group_size and vect_size (subgroup padding), and at least vect_size * sub_group_size
    const int c_divisor = math::lcm(conf.sub_group_size, conf.vect_size);
    conf.padded_c = utils::rnd_up(diff_src_md()->padded_dims[1], c_divisor);
    conf.padded_c
            = std::max(conf.padded_c, conf.vect_size * conf.sub_group_size);

    // lws: Multiple of sub_group_size
    conf.lws[0] = utils::rnd_up(conf.lws[0], conf.sub_group_size);
    conf.lws[1] = conf.lws[2] = 1;

    // gws: multiple of lws and padded C, and each other dim
    const int gws_divisor = math::lcm((int)conf.lws[0], (int)conf.padded_c);
    conf.gws[0] = diff_src_md()->padded_dims[0] * conf.padded_c * ID() * IH()
            * IW() / conf.vect_size;
    conf.gws[0] = utils::rnd_up(conf.gws[0], gws_divisor);

    conf.gws[1] = conf.gws[2] = 1;

    // Copy src/dst shapes to conf
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

    // Highly-upsampled cases are not supported
    // TODO: Implement multiple linear calculations per work item
    // to eliminate this requirement
    const int max_d = std::max(1, (int)std::ceil(conf.FD * 1.5 - 0.5));
    const int max_h = std::max(1, (int)std::ceil(conf.FH * 1.5 - 0.5));
    const int max_w = std::max(1, (int)std::ceil(conf.FW * 1.5 - 0.5));
    const int max_num_linear_calcs = 2 * (max_d + max_h + max_w);
    if (max_num_linear_calcs > conf.sub_group_size) {
        return status::unimplemented;
    }

    // Compute strides after vect_size is taken into account.
    const blocking_desc_t &blocks = diff_src_md()->format_desc.blocking;
    const dim_t c_dim = diff_src_d.padded_dims()[1];
    const dim_t stride_c = blocks.strides[1];

    for (int i = 0; i < ndims(); i++) {
        if (blocks.strides[i] < stride_c || i == 1) {
            conf.padded_strides[i] = blocks.strides[i];
        } else {
            conf.padded_strides[i] = blocks.strides[i] * conf.padded_c / c_dim
                    / conf.vect_size; // Scale up to the newly-padded size
        }
    }

    set_offsets(diff_src_d, conf.off.src_off);
    set_offsets(diff_dst_d, conf.off.dst_off);

    return status::success;
}

status_t vectorized_resampling_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(diff_dst_md()->data_type);
    kernel_ctx.define_int("IS_BWD", 1);

    status_t status = init_kernel_ctx_common(kernel_ctx, conf, desc());

    def_data_type(kernel_ctx, diff_dst_md()->data_type, "SRC");
    def_data_type(kernel_ctx, diff_src_md()->data_type, "DST");

    return status;
}

status_t vectorized_resampling_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, diff_dst);

    const resampling_conf_t &conf = pd()->conf;
    compute::nd_range_t nd_range(conf.gws, conf.lws);

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
