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

#ifndef JIT_REF_SHUFFLE_KERNEL_HPP
#define JIT_REF_SHUFFLE_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"
#include "ocl_shuffle_pd.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_ref_shuffle_kernel {

    jit_ref_shuffle_kernel(const jit_shuffle_conf_t &ajshfl) : jshfl(ajshfl) {}

    ~jit_ref_shuffle_kernel() {}

    static status_t init_conf(jit_shuffle_conf_t &jshfl, const shuffle_pd_t *pd,
            jit_offsets &jit_off) {

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

        set_offsets(input_md, jit_off.src_off);

        return status::success;
    }

    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            const jit_shuffle_conf_t &jshfl, const jit_offsets &jit_off) {

        kernel_ctx.set_data_type(jshfl.data_type);
        kernel_ctx.define_int("NDIMS", jshfl.ndims);
        kernel_ctx.define_int("AXIS", jshfl.axis);
        kernel_ctx.define_int("AXIS_SIZE", jshfl.axis_size);
        kernel_ctx.define_int("GROUP_SIZE", jshfl.group_size);
        kernel_ctx.define_int("TRANSPOSE_ROW", jshfl.transpose_row);
        kernel_ctx.define_int("TRANSPOSE_COL", jshfl.transpose_col);
        kernel_ctx.define_int("INNER_SIZE", jshfl.inner_size);
        kernel_ctx.define_int("OUTER_SIZE", jshfl.outer_size);

        def_offsets(jit_off.src_off, kernel_ctx, "SRC", jshfl.ndims);
        return status::success;
    }

    jit_shuffle_conf_t jshfl;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif // JIT_REF_SHUFFLE_KERNEL_HPP
