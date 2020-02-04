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

#include "gpu/ocl/ref_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace format_tag;

status_t ref_shuffle_init_conf(
        shuffle_conf_t &jshfl, const shuffle_pd_t *pd, offsets_t &off) {

    const bool is_fwd = pd->is_fwd();

    const memory_desc_wrapper input_md(
            is_fwd ? pd->src_md() : pd->diff_dst_md());
    jshfl.data_type = input_md.data_type();

    const int axis = pd->axis();
    jshfl.axis = axis;

    const int axis_size = pd->axis_size();
    const int group_size = pd->group_size();
    jshfl.transpose_row = is_fwd ? group_size : axis_size / group_size;
    jshfl.transpose_col = is_fwd ? axis_size / group_size : group_size;
    jshfl.axis_size = axis_size;
    jshfl.group_size = group_size;

    auto dims = pd->desc()->data_desc.dims;
    auto ndims = pd->desc()->data_desc.ndims;
    const size_t outer_size = utils::array_product(dims, axis);
    const size_t inner_size
            = utils::array_product(dims + axis + 1, ndims - axis - 1);
    const size_t dim = axis_size * inner_size;
    jshfl.outer_size = outer_size;
    jshfl.inner_size = inner_size;
    jshfl.dim = dim;
    jshfl.ndims = ndims;

    jshfl.gws_d[0] = nstl::max(size_t(1), inner_size);
    jshfl.gws_d[1] = nstl::max(1, axis_size);
    jshfl.gws_d[2] = nstl::max(size_t(1), outer_size);

    set_offsets(input_md, off.src_off);

    return status::success;
}

status_t ref_shuffle_init_const_def(compute::kernel_ctx_t &kernel_ctx,
        const shuffle_conf_t &jshfl, const offsets_t &off) {

    kernel_ctx.set_data_type(jshfl.data_type);
    kernel_ctx.define_int("NDIMS", jshfl.ndims);
    kernel_ctx.define_int("AXIS", jshfl.axis);
    kernel_ctx.define_int("AXIS_SIZE", jshfl.axis_size);
    kernel_ctx.define_int("GROUP_SIZE", jshfl.group_size);
    kernel_ctx.define_int("TRANSPOSE_ROW", jshfl.transpose_row);
    kernel_ctx.define_int("TRANSPOSE_COL", jshfl.transpose_col);
    kernel_ctx.define_int("INNER_SIZE", jshfl.inner_size);
    kernel_ctx.define_int("OUTER_SIZE", jshfl.outer_size);

    def_offsets(off.src_off, kernel_ctx, "SRC", jshfl.ndims);
    return status::success;
}

template <dnnl_format_tag_t tag>
status_t ref_shuffle_t::execute_(const exec_ctx_t &ctx) const {
    auto &src = pd()->is_fwd() ? CTX_IN_STORAGE(DNNL_ARG_SRC)
                               : CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &dst = pd()->is_fwd() ? CTX_OUT_STORAGE(DNNL_ARG_DST)
                               : CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    const auto &jshfl = pd()->jshfl_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);

    auto nd_range = compute::nd_range_t(jshfl.gws_d);
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}
template status_t ref_shuffle_t::execute_<any>(const exec_ctx_t &ctx) const;

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
