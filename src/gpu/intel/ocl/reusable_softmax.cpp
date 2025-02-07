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

#include <numeric>
#include "gpu/intel/ocl/reusable_softmax.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace gpu_utils;

class softmax_lws_strategy_t : public compute::lws_strategy_t {
public:
    bool is_included(const compute::mapped_block_t &blocks) const override {
        for (const block_t &block : inc_blocks) {
            if (blocks.get_dim_idx() == block.dim_idx) { return true; }
        }
        return false;
    };

    void include(dim_idx_t dim, size_t size) {
        inc_blocks.emplace_back(dim, into<dim_t>(size), 1);
    }

private:
    using compute::lws_strategy_t::lws_strategy_t;
    compute::range_t create_lws(compute::range_t &gws,
            const compute::gws_bin_mapping_t &mapper) const override {
        auto lws = compute::range_t::one(gws.ndims());

        for (size_t i = 0; i < gws.ndims(); i++) {
            const auto &bins = mapper.get_bins(i);
            if (bins.empty()) continue;
            for (const block_t &inc_block : inc_blocks) {
                if (bins[0].get_dim_idx() == inc_block.dim_idx) {
                    lws[i] *= into<size_t>(inc_block.block);
                }
            }
        }

        return lws;
    };

    std::vector<block_t> inc_blocks;
};

namespace softmax_dims_t {
dim_idx_t mb = 0;
dim_idx_t ic = 1;
dim_idx_t sp0 = 2;
dim_idx_t sp1 = 3;
dim_idx_t sp2 = 4;
dim_idx_t workers = 5; // artificial dimension partitions reductions
}; // namespace softmax_dims_t

static std::vector<dim_idx_t> get_dims(size_t ndims) {
    std::vector<dim_idx_t> ret(ndims);
    uint8_t idx = 0;
    ret[idx++] = softmax_dims_t::mb;
    ret[idx++] = softmax_dims_t::ic;
    if (ndims >= 3) ret[idx++] = softmax_dims_t::sp0;
    if (ndims >= 4) ret[idx++] = softmax_dims_t::sp1;
    if (ndims >= 5) ret[idx++] = softmax_dims_t::sp2;
    return ret;
}

status_t reusable_softmax_fwd_t::pd_t::init_dispatch_default_reusable(
        gpu::engine_t *engine) {
    using dims_vec_t = std::vector<dim_idx_t>;

    dims_vec_t src_dim_ids(memory_desc_wrapper(src_md()).ndims());
    std::iota(src_dim_ids.begin(), src_dim_ids.end(), 0);

    dims_vec_t dispatch_dim_ids = src_dim_ids;
    dispatch_dim_ids.erase(dispatch_dim_ids.begin() + (desc()->softmax_axis));

    compute::named_buffer_t src_buf("SRC", *src_md(), src_dim_ids);
    compute::named_buffer_t dst_buf("DST", src_buf);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    compute::reusable_dispatch_config_t dispatch_config(
            compute_engine, std::move(dispatch_dim_ids));
    CHECK(dispatch_config.register_buffer(src_buf));
    CHECK(dispatch_config.register_buffer(dst_buf));

    compute::reusable_dispatch_t dispatch;
    const auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    CHECK(dispatch_config.generate(dispatch,
            compute::default_lws_strategy_t(compute_engine, gpu_attr)));

    conf.gws_params = dispatch.get_compile_params();
    rt_conf.gws_params = dispatch.get_runtime_params();

    return status::success;
}

status_t reusable_softmax_fwd_t::pd_t::init_dispatch_workgroup_per_reduction(
        gpu::engine_t *engine, const size_t num_workers_per_workgroup) {

    const memory_desc_wrapper src_mdw(src_md());
    std::vector<dim_idx_t> dims_ids = get_dims(src_mdw.ndims());
    auto sizes = src_mdw.dims(); // TODO: dynamic worker policy
    const size_t softmax_axis = static_cast<size_t>(desc()->softmax_axis);
    const dim_t softmax_axis_size = sizes[desc()->softmax_axis];

    // set number of work items per reduction block
    rt_conf.softmax_chunk_size = dnnl::impl::utils::div_up(
            softmax_axis_size, num_workers_per_workgroup);

    // apply constraint checks based on chosen workgroup_size
    VDISPATCH_SOFTMAX(rt_conf.softmax_chunk_size < 64, VERBOSE_BAD_PARAM,
            "indivisible axis reduction size");

    // source buffer gets new dimension: multiple workers per reduction block
    compute::named_buffer_t src_buf("SRC");

    // keep original input buffer geometry for addressing
    compute::named_buffer_t ori_buf("ORIGINAL");
    for (size_t i = 0; i < dims_ids.size(); i++) {
        ori_buf.append_block(dims_ids[i], sizes[i]);
    }

    for (size_t i = 0; i < dims_ids.size(); i++) {
        if (i == softmax_axis) {
            src_buf.append_block(
                    softmax_dims_t::workers, num_workers_per_workgroup);
            src_buf.append_block(dims_ids[i], rt_conf.softmax_chunk_size);
        }
        if (i != softmax_axis) { src_buf.append_block(dims_ids[i], sizes[i]); }
    }

    // Account for reduction axis indivisible by num workers; num
    // workers times elements per worker will exceed reduction axis
    // length. Reset strides outside reduction dim to original input
    // buffer strides prior to worker partitioning (folding)
    const size_t folded_axis_size
            = num_workers_per_workgroup * rt_conf.softmax_chunk_size;
    for (size_t i = 0; i < softmax_axis; i++) {
        src_buf.format_desc.blocking.strides[i]
                = (src_buf.format_desc.blocking.strides[i] / folded_axis_size)
                * softmax_axis_size;
    }

    compute::named_buffer_t dst_buf("DST", src_buf);

    // dispatch: all dims except reduction dimension plus workers dimension
    std::vector<dim_idx_t> dispatch_dims = std::move(dims_ids);
    dispatch_dims[softmax_axis] = softmax_dims_t::workers;

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    compute::reusable_dispatch_config_t dispatch_config(
            compute_engine, std::move(dispatch_dims));
    CHECK(dispatch_config.register_buffer(src_buf));
    CHECK(dispatch_config.register_buffer(dst_buf));
    CHECK(dispatch_config.register_buffer(ori_buf));

    const auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    compute::reusable_dispatch_t dispatch;
    auto lws_strat = softmax_lws_strategy_t(compute_engine, gpu_attr);
    lws_strat.include(softmax_dims_t::workers, num_workers_per_workgroup);
    CHECK(dispatch_config.generate(dispatch, lws_strat));
    conf.gws_params = dispatch.get_compile_params();
    rt_conf.gws_params = dispatch.get_runtime_params();

    auto dispatch_lws = dispatch.get_runtime_params().nd_range.local_range();
    auto dispatch_gws = dispatch.get_runtime_params().nd_range.global_range();

    auto *device_info = compute_engine->device_info();
    const size_t multiple_of_sg_lws
            = utils::rnd_up(dispatch_lws[0], device_info->max_subgroup_size());

    compute::range_t softmax_gws
            = {multiple_of_sg_lws, dispatch_gws[1], dispatch_gws[2]};
    compute::range_t softmax_lws
            = {multiple_of_sg_lws, dispatch_lws[1], dispatch_lws[2]};
    compute::nd_range_t softmax_ndrange(softmax_gws, softmax_lws);
    rt_conf.gws_params.nd_range = softmax_ndrange;

    return status::success;
}

compute::kernel_ctx_t reusable_softmax_params_t::get_kernel_ctx() const {
    compute::kernel_ctx_t kernel_ctx;
    kernel_ctx.define_int("LOGSOFTMAX", is_logsoftmax);
    kernel_ctx.define_int("MANY_REDUCTIONS_PER_WORKGROUP",
            algorithm_number == many_reductions_per_workgroup);
    kernel_ctx.define_int("USE_SUBGROUP_REDUCTION",
            algorithm_number == one_reduction_per_subgroup);
    kernel_ctx.define_int("USE_WORKGROUP_REDUCTION",
            algorithm_number == one_reduction_per_workgroup);
    kernel_ctx.add_option("-cl-std=CL2.0");

    kernel_ctx.set_data_type(src_data_type);
    def_data_type(kernel_ctx, src_data_type, "SRC");
    def_data_type(kernel_ctx, dst_data_type, "DST");

    gws_params.def_kernel_macros(kernel_ctx);
    return kernel_ctx;
}

status_t reusable_softmax_fwd_t::execute_generic(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &src_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &dst_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.append(src_scale);
    arg_list.append(dst_scale);
    arg_list.append(pd()->rt_conf.softmax_axis_size);
    arg_list.append(pd()->rt_conf.softmax_axis_stride);
    arg_list.append(pd()->rt_conf.softmax_chunk_size);
    arg_list.append(pd()->rt_conf.gws_params.get());

    auto status = parallel_for(
            ctx, pd()->rt_conf.gws_params.nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
