/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"

#include "common/utils.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/gpu_primitive_attr.hpp"
#include "gpu/intel/ocl/reduction/reduction_utils.hpp"
#include "gpu/intel/ocl/reduction/reusable_ref_reduction.hpp"
#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace gpu_utils;

namespace { // Use an anonymous namespace to avoid collisions with ocl:atomic
namespace reduction_dims {
dim_idx_t outer = 0;
dim_idx_t reduction = 1;
dim_idx_t inner = 2;
} // namespace reduction_dims
} // namespace
static const std::vector<dim_idx_t> dims {
        reduction_dims::outer,
        reduction_dims::reduction,
        reduction_dims::inner,
};
static const std::vector<dim_idx_t> dispatch_dims {
        reduction_dims::outer,
        reduction_dims::inner,
};

ref_reduction_conf_t::ref_reduction_conf_t(const reduction_subproblem_t &subprb,
        reduction_alg_kind_t alg, data_type_t src_dt, data_type_t dst_dt,
        const compute::device_info_t &device_info,
        gpu_primitive_attr_t *gpu_attr)
    : reduction_stride(subprb.reduction_block.stride)
    , reduction_size(subprb.reduction_block.block)
    , num_dst_elems(into<size_t>(
              subprb.outer_block.block * subprb.inner_block.block)) {
    conf.alg = alg;
    conf.src_dt = src_dt;
    conf.dst_dt = dst_dt;
    auto arch = device_info.gpu_arch();
    const int base_threads_per_eu
            = compute::device_info_t::threads_per_eu(arch);
    conf.threads_per_eu
            = gpu_attr ? gpu_attr->threads_per_eu() : base_threads_per_eu;
}

status_t ref_reduction_conf_t::init_dispatcher(
        const reduction_subproblem_t &subprb,
        const compute::compute_engine_t &engine,
        gpu_primitive_attr_t *gpu_attr) {

    compute::named_buffer_t src_buf("SRC");
    std::array<dim_t, 3> dim_sizes = {subprb.outer_block.block,
            subprb.reduction_block.block, subprb.inner_block.block};
    for (size_t i = 0; i < 3; i++) {
        src_buf.append_block(dims[i], dim_sizes[i]);
    }
    compute::named_buffer_t dst_buf("DST", src_buf);
    dst_buf.remove_dim(reduction_dims::reduction);

    compute::reusable_dispatch_config_t config(&engine, dispatch_dims);
    CHECK(config.register_buffer(src_buf));
    CHECK(config.register_buffer(dst_buf));

    compute::reusable_dispatch_t dispatch;
    CHECK(config.generate(
            dispatch, compute::default_lws_strategy_t(&engine, gpu_attr)));
    conf.params = dispatch.get_compile_params();
    rt_conf = dispatch.get_runtime_params();

    return status::success;
}

void reusable_ref_reduction_t::pd_t::init_scratchpad() {
    // Only need scratchpads for the first 2 phases, since we can reuse them
    // and memory requirements are monotonically decreasing each phase.
    const uint32_t keys[2] = {memory_tracking::names::key_reduction,
            memory_tracking::names::key_reduction_1};

    auto scratchpad = scratchpad_registry().registrar();
    for (size_t i = 0; i < std::min(phases.size(), size_t {2}); i++) {
        const ref_reduction_conf_t &phase = phases[i];
        const size_t dt_size = types::data_type_size(phase.conf.dst_dt);
        scratchpad.book(
                keys[i], phase.num_dst_elems, dt_size, OCL_BUFFER_ALIGNMENT);
    }
}

status_t reusable_ref_reduction_t::pd_t::init_conf(impl::engine_t *engine) {
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const int ndims = src_mdw.ndims();
    const dim_t *src_dims = src_mdw.dims();
    const dim_t *src_padded_dims = src_mdw.padded_dims();
    const dim_t *dst_dims = dst_mdw.dims();

    bool is_reduction_dim[DNNL_MAX_NDIMS];
    for (int i = 0; i < ndims; i++) {
        // Actually reduced dimensions
        if (src_dims[i] != dst_dims[i]) {
            is_reduction_dim[i] = true;
            continue;
        }

        // Size-1 dims can be treated as reducible (at no cost):
        if (src_dims[i] == 1 && src_padded_dims[i] == 1) {
            is_reduction_dim[i] = true;
            continue;
        }

        is_reduction_dim[i] = false;
    }

    std::vector<reduction_subproblem_t> subprbs;
    CHECK(generate_reduction_phases(src_md(), dst_md(), subprbs));

    //DST zero-padding not supported on reduction dims
    reduction_subproblem_t &last_subprb = subprbs.back();
    for (const auto &zpad : last_subprb.dst_zpads) {
        if (is_reduction_dim[zpad.dim_idx]) return status::unimplemented;
    }

    // DST zero-padding is not supported for algs which modify input 0s
    // (DST zero-padding on non-reduced dims always accompanied by SRC zero-padding)
    using namespace alg_kind;
    bool alg_changes_zeros = desc()->eps != 0
            && utils::one_of(desc()->alg_kind, reduction_norm_lp_max,
                    reduction_norm_lp_sum, reduction_norm_lp_power_p_max,
                    reduction_norm_lp_power_p_sum);
    if (alg_changes_zeros && !last_subprb.dst_zpads.empty()) {
        return status::unimplemented;
    }

    // SRC zero-padding on reduced dims is not supported if alg is affected by zeros.
    reduction_subproblem_t &first_subprb = subprbs.front();
    const bool alg_affected_by_zeros = utils::one_of(
            desc()->alg_kind, reduction_min, reduction_max, reduction_mul);
    for (const auto &zpad : first_subprb.src_zpads) {
        if (alg_affected_by_zeros && is_reduction_dim[zpad.dim_idx]) {
            return status::unimplemented;
        }
    }

    const compute::compute_engine_t *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);
    auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());

    data_type_t accum_data_type = types::default_accum_data_type(
            src_mdw.data_type(), dst_mdw.data_type());
    for (size_t i = 0; i < subprbs.size(); i++) {
        const bool is_first = (i == 0);
        const bool is_final = (i == subprbs.size() - 1);
        reduction_alg_kind_t phase_alg
                = from_alg(desc()->alg_kind, is_first, is_final);
        data_type_t src_dt = is_first ? src_mdw.data_type() : accum_data_type;
        data_type_t dst_dt = is_final ? dst_mdw.data_type() : accum_data_type;

        phases.emplace_back(subprbs[i], phase_alg, src_dt, dst_dt,
                *compute_engine->device_info(), gpu_attr);
        auto &phase = phases.back();
        CHECK(phase.init_dispatcher(subprbs[i], *compute_engine, gpu_attr));
    }

    // Compute div from basic mdw dims
    div = 1;
    for (int i = 0; i < src_mdw.ndims(); i++) {
        if (is_reduction_dim[i]) div *= src_dims[i];
    }
    return status::success;
}

static void init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const ref_reduction_key_params_t &conf) {
    using namespace alg_kind;

    // Data types
    kernel_ctx.set_data_type(conf.src_dt);
    def_data_type(kernel_ctx, conf.dst_dt, "DST");

    // Dispatcher
    conf.params.def_kernel_macros(kernel_ctx);

    // Alg kinds
    def_reduction_alg_kinds(kernel_ctx);
    kernel_ctx.define_int("REDUCTION_ALG", to_int(conf.alg));
}

status_t ref_reduction_key_params_t::get_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    primitive_attr_t ocl_attr;
    CHECK(ocl_attr.set_gpu_attr(gpu_primitive_attr_t(threads_per_eu)));
    kernel_ctx = compute::kernel_ctx_t(&ocl_attr);

    init_kernel_ctx_common(kernel_ctx, *this);
    return status::success;
}

status_t reusable_ref_reduction_t::execute(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    std::unique_ptr<memory_storage_t> sp_reduce[2]
            = {ctx.get_scratchpad_grantor().get_memory_storage(
                       memory_tracking::names::key_reduction),
                    ctx.get_scratchpad_grantor().get_memory_storage(
                            memory_tracking::names::key_reduction_1)};

    for (size_t i = 0; i < kernels_.size(); i++) {
        const auto &kernel = kernels_[i];
        const auto &phase = pd()->phases[i];
        const auto &nd_range = phase.rt_conf.nd_range;

        bool use_int32_offset = phase.conf.params.use_int32_offset;
        const auto &append_off
                = [use_int32_offset](
                          compute::kernel_arg_list_t &arg_list, dim_t off) {
                      if (use_int32_offset) {
                          arg_list.append(into<int32_t>(off));
                      } else {
                          arg_list.append(off);
                      }
                  };

        // Set up the reduction arg list
        compute::kernel_arg_list_t reduction_arg_list;

        memory_storage_t &src_mem = (i == 0) ? src : *sp_reduce[(i - 1) % 2];
        memory_storage_t &dst_mem
                = (i == kernels_.size() - 1) ? dst : *sp_reduce[i % 2];

        reduction_arg_list.append(src_mem);
        reduction_arg_list.append(dst_mem);
        append_off(reduction_arg_list, phase.reduction_stride);
        append_off(reduction_arg_list, into<dim_t>(phase.reduction_size));
        reduction_arg_list.append(pd()->div);
        reduction_arg_list.append(pd()->desc()->p);
        reduction_arg_list.append(pd()->desc()->eps);
        reduction_arg_list.append(phase.rt_conf.get());

        CHECK(parallel_for(ctx, nd_range, kernel, reduction_arg_list));
    }

    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
