/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include <math.h>

#include "common/primitive_exec_types.hpp"

#include "gpu/ocl/gen9_concat.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

std::pair<int, int> gen9_concat_t::pd_t::calculate_iter_dim_idx_chunk(
        int num_threads) const {
    if (conf.ndims == 1) return std::make_pair(0, 1);
    const auto &dst_dims = conf.dst_md_info.padded_dims;
    int max_dim_idx = -1;
    int max_dim = -1;
    for (int dim_idx = conf.ndims - 1; dim_idx >= 0; dim_idx--) {
        if (dst_dims[dim_idx] > max_dim && dim_idx != conf.concat_axis) {
            max_dim = dst_dims[dim_idx];
            max_dim_idx = dim_idx;
        }
    }
    const int iter_dim_idx = max_dim_idx;
    const int all_elems = utils::array_product(dst_dims, conf.ndims);
    const int max_iter_dim_chunk = 1024;
    const int min_threads = num_threads * 4;
    int iter_dim_chunk = std::min(dst_dims[iter_dim_idx], max_iter_dim_chunk);
    const auto get_num_threads = [&]() {
        return ceil(static_cast<float>(all_elems)
                / (iter_dim_chunk * conf.sub_group_size));
    };
    while (get_num_threads() < min_threads && iter_dim_chunk > 1) {
        iter_dim_chunk = ceil(iter_dim_chunk / 2.0f);
    }
    return std::make_pair(iter_dim_idx, iter_dim_chunk);
}

bool gen9_concat_t::pd_t::can_use_sub_group_size(
        const compute::compute_engine_t *compute_engine, int sub_group_size) {
    auto is_dim_dense = [](const memory_desc_wrapper &mdw, int dim_idx) {
        return mdw.blocking_desc().strides[dim_idx] == 1;
    };
    auto get_dim_block = [](const memory_desc_wrapper &mdw, int dim_idx) {
        const auto &blk = mdw.blocking_desc();
        if (blk.inner_nblks == 0
                || blk.inner_idxs[blk.inner_nblks - 1] != dim_idx)
            return static_cast<dnnl_dim_t>(1);
        return blk.inner_blks[blk.inner_nblks - 1];
    };

    const concat_pd_t *pd = this;
    const int c_idx = 1;
    const memory_desc_wrapper dst_mdw(pd->dst_md());
    const bool is_dst_blocked = dst_mdw.blocking_desc().inner_nblks > 0;
    bool is_concat_axis_aligned = true;
    bool layouts_compatible = is_dst_blocked
            ? get_dim_block(dst_mdw, c_idx) % sub_group_size == 0
            : is_dim_dense(dst_mdw, c_idx);
    for (int i = 0; i < conf.n; ++i) {
        const memory_desc_wrapper src_mdw(pd->src_md(i));
        is_concat_axis_aligned = is_concat_axis_aligned
                && src_mdw.md_->dims[conf.concat_axis] % sub_group_size == 0;
        if (is_dst_blocked) {
            layouts_compatible = layouts_compatible
                    && get_dim_block(src_mdw, c_idx) % sub_group_size == 0;
        } else {
            layouts_compatible
                    = layouts_compatible && is_dim_dense(src_mdw, c_idx);
        }
    }
    return is_concat_axis_aligned && layouts_compatible
            && compute_engine->mayiuse_sub_group(sub_group_size);
}

int gen9_concat_t::pd_t::calculate_sub_group_size(
        const compute::compute_engine_t *compute_engine) {
    // Subgroups are used only for concatenation over C dimension
    if (conf.concat_axis != 1) return 1;
    for (int sub_group_size : {16, 8}) {
        if (can_use_sub_group_size(compute_engine, sub_group_size)) {
            return sub_group_size;
        }
    }
    return 1;
}

status_t gen9_concat_t::pd_t::init_conf(engine_t *engine) {
    const concat_pd_t *pd = this;

    const memory_desc_wrapper dst_mdw(pd->dst_md());
    const auto dst_dims = dst_mdw.md_->padded_dims;

    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.dst_type = dst_mdw.data_type();
    conf.src_type = memory_desc_wrapper(pd->src_md(0)).data_type();
    conf.ndims = dst_mdw.ndims();
    const auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(dst_mdw.md_);
    conf.n = pd->n_inputs();
    conf.concat_axis = pd->concat_dim();

    int concat_axis_end = 0;
    for (int i = 0; i < conf.n; ++i) {
        const memory_desc_wrapper src_mdw(pd->src_md(i));
        concat_axis_end += src_mdw.md_->dims[conf.concat_axis];
        conf.offset[i] = concat_axis_end;
        conf.src_md_infos[i] = memory_desc_info_t::create(pd->src_md(i));
    }

    conf.sub_group_size = calculate_sub_group_size(compute_engine);
    std::tie(conf.iter_dim_idx, conf.iter_dim_chunk)
            = calculate_iter_dim_idx_chunk(
                    compute_engine->device_info()->hw_threads());

    if (dst_mdw.blocking_desc().inner_nblks == 0
            && (conf.sub_group_size == 1
                    || (conf.ndims > 2 && conf.iter_dim_chunk == 1))) {
        return status::unimplemented;
    }

    for (int dim_idx = 0; dim_idx < MAX_NDIMS; dim_idx++) {
        const int dim_block
                = conf.iter_dim_idx == dim_idx ? conf.iter_dim_chunk : 1;
        const int dim_size = conf.ndims > dim_idx ? dst_dims[dim_idx] : 1;
        conf.dispatch.define_dim(
                utils::format("D%d", dim_idx), 0, dim_size, dim_block);
    }
    if (conf.sub_group_size > 1) {
        conf.dispatch.vectorize_dim("D1", conf.sub_group_size);
    }
    conf.dispatch.generate();

    return status::success;
}

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const concat_conf_t &conf) {
    for (int i = 0; i < conf.n; ++i) {
        kernel_ctx.define_int(utils::format("SRC%d_END", i), conf.offset[i]);
        def_memory_desc_info(kernel_ctx, conf.src_md_infos[i],
                utils::format("SRC%d", i).c_str());
    }
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    kernel_ctx.set_data_type(conf.src_type);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("CONCAT_AXIS", conf.concat_axis);
    kernel_ctx.define_int("NUM_INPUTS", conf.n);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("VECT_DT_N", 1);
    kernel_ctx.define_int("ITER_DIM_PADDED_SIZE",
            conf.dst_md_info.padded_dims[conf.iter_dim_idx]);
    kernel_ctx.define_int("ITER_DIM_IDX", conf.iter_dim_idx);
    kernel_ctx.define_int("ITER_DIM_CHUNK", conf.iter_dim_chunk);

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t gen9_concat_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

status_t gen9_concat_t::execute_concat(const exec_ctx_t &ctx) const {
    status_t status;
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dst);
    for (int i = 0; i < 16; ++i) {
        auto &src = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + i);
        arg_list.set(i + 1, src);
    }

    auto nd_range = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel, arg_list);
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
