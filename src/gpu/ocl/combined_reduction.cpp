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

#include "gpu/ocl/combined_reduction.hpp"
#include "common/scratchpad.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static reduction_phase_t init_phase(int start, int end, int num_reductions,
        data_type_t src_type, data_type_t dst_type,
        compute::nd_range_t nd_range, bool is_final, bool is_first) {
    reduction_phase_t phase;
    phase.initial_size = start;
    phase.reduction_size = num_reductions;
    phase.final_size = end;
    phase.num_reduction_chunks
            = utils::div_up(phase.initial_size, phase.reduction_size);
    phase.src_type = src_type;
    phase.dst_type = dst_type;
    phase.nd_range = nd_range;
    phase.is_final = is_final;
    phase.is_first = is_first;
    return phase;
}

void combined_reduction_t::pd_t::init_scratchpad() {
    // Only need scratchpads for the first 2 phases, since we can reuse them
    // and memory requirements are monotonically decreasing each phase.
    uint32_t keys[2] = {memory_tracking::names::key_reduction,
            memory_tracking::names::key_reduction_1};

    for (int phase_num = 0;
            phase_num < std::min(2, (int)conf.phases.size() - 1); phase_num++) {
        const size_t sp_data_size
                = types::data_type_size(conf.phases[phase_num].dst_type);
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(keys[phase_num], conf.sp_size[phase_num], sp_data_size,
                OCL_BUFFER_ALIGNMENT);
    }
}

status_t combined_reduction_t::pd_t::init_conf(engine_t *engine) {
    // To start, check for compatibility
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const int ndims = src_mdw.ndims();

    const dnnl_dim_t *src_dims = src_mdw.dims();
    const blocking_desc_t &blk = src_mdw.blocking_desc();

    const dnnl_dim_t *dst_dims = dst_mdw.dims();
    const blocking_desc_t &dst_blk = dst_mdw.blocking_desc();

    const compute::compute_engine_t *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);

    // Require same src/dst blocking
    if (blk.inner_nblks != dst_blk.inner_nblks) { // Same number of blocks
        return status::unimplemented;
    }

    for (int i = 0; i < blk.inner_nblks; i++) {
        if (blk.inner_idxs[i]
                != dst_blk.inner_idxs[i]) { // Same blocking permutation
            return status::unimplemented;
        }
        if (blk.inner_blks[i] != dst_blk.inner_blks[i]) { // Same blocking sizes
            return status::unimplemented;
        }
    }

    // Zero padding is not implemented when dim is reduced
    // Or when doing an LP/P alg (not zero-preserving)
    using namespace alg_kind;
    for (int i = 0; i < blk.inner_nblks; i++) {
        // Needs zero padding
        if (dst_mdw.padded_dims()[blk.inner_idxs[i]]
                != dst_mdw.dims()[blk.inner_idxs[i]]) {
            // non-zero-preserving alg
            switch (desc()->alg_kind) {
                case reduction_norm_lp_max:
                case reduction_norm_lp_sum:
                case reduction_norm_lp_power_p_max:
                case reduction_norm_lp_power_p_sum:
                    return status::unimplemented;
                default: break;
            }
            // Dim reduced
            if (dst_mdw.dims()[blk.inner_idxs[i]]
                    != src_mdw.dims()[blk.inner_idxs[i]]) {
                return status::unimplemented;
            }
        }
    }

    // Determine ordering of dims by stride
    dims_t dim_perm = {0}, dst_perm = {0};
    for (int i = 0; i < ndims; i++) {
        // Src
        dim_t stride = blk.strides[i];
        int dim_idx = i;
        for (int j = 0; j < i; j++) {
            if (stride > blk.strides[dim_perm[j]]) {
                // Insert this stride/idx into dim_perms
                stride = blk.strides[dim_perm[j]];
                int tmp_perm = dim_perm[j];
                dim_perm[j] = dim_idx;
                dim_idx = tmp_perm;
            }
        }
        dim_perm[i] = dim_idx;

        // Same for dst
        stride = dst_blk.strides[i];
        dim_idx = i;
        for (int j = 0; j < i; j++) {
            if (stride > dst_blk.strides[dst_perm[j]]) {
                // Insert this stride/idx into dim_perms
                stride = dst_blk.strides[dst_perm[j]];
                int tmp_perm = dst_perm[j];
                dst_perm[j] = dim_idx;
                dim_idx = tmp_perm;
            }
        }
        dst_perm[i] = dim_idx;
    }

    dims_t block_sizes, dst_blocks;
    src_mdw.compute_blocks(block_sizes);
    dst_mdw.compute_blocks(dst_blocks);
    // Determine extended (plain+blocked) dim structure
    dim_t extended_dim_order[2 * MAX_NDIMS], extended_dst_order[2 * MAX_NDIMS];
    dim_t extended_dim_size[2 * MAX_NDIMS];
    const int num_comp_dims = ndims + blk.inner_nblks;
    for (int i = 0; i < ndims; i++) { // plain
        extended_dim_order[i] = dim_perm[i];
        extended_dim_size[i]
                = src_mdw.padded_dims()[dim_perm[i]] / block_sizes[dim_perm[i]];
        extended_dst_order[i] = dst_perm[i];
    }
    for (int i = 0; i < blk.inner_nblks; i++) { // blocked
        extended_dim_order[i + ndims] = blk.inner_idxs[i];
        extended_dim_size[i + ndims] = blk.inner_blks[i];
        extended_dst_order[i + ndims] = dst_blk.inner_idxs[i];
    }

    // Only allow same src/dst format tags and permutations
    // TODO: Relax src/dst format matching
    for (int i = 0; i < num_comp_dims; i++) {
        if (extended_dim_order[i] != extended_dst_order[i]) {
            return status::unimplemented;
        }
    }

    // Convert composite structure to reduced dims
    dim_t extended_reduced_dims[2 * MAX_NDIMS];
    for (int i = 0; i < num_comp_dims; i++) {
        extended_reduced_dims[i] = (src_dims[extended_dim_order[i]]
                                           != dst_dims[extended_dim_order[i]])
                ? 1
                : 0;
    }

    // Finally, the check: Make sure all reduced dims are sequential
    // i.e. extended_reduced_dims has no 10...1 pattern
    for (int i = 0; i < num_comp_dims - 2; i++) {
        if (extended_reduced_dims[i] == 0
                || extended_reduced_dims[i + 1] == 1) {
            continue;
        }
        // Now we have the 10 pattern -- look for all 0's to the right
        for (int j = i + 1; j < num_comp_dims; j++) {
            if (extended_reduced_dims[j] == 1) { return status::unimplemented; }
        }
        break;
    }

    // Get information about composite outer/reduced/inner dimensions
    int num_outer_dims = 0, num_reduced_dims = 0, num_inner_dims = 0;
    bool left_side = true;
    for (int i = 0; i < num_comp_dims; i++) {
        if (extended_reduced_dims[i] == 1) {
            left_side = false;
            num_reduced_dims += 1;
            continue;
        }

        if (left_side) {
            num_outer_dims += 1;
        } else {
            num_inner_dims += 1;
        }
    }

    // Compute composite dim sizes
    int outer_dim_size = 1;
    for (int i = 0; i < num_outer_dims; i++) {
        outer_dim_size *= extended_dim_size[i];
    }
    int reduced_dim_size = 1;
    for (int i = 0; i < num_reduced_dims; i++) {
        reduced_dim_size *= extended_dim_size[i + num_outer_dims];
    }
    int inner_dim_size = 1;
    for (int i = 0; i < num_inner_dims; i++) {
        inner_dim_size
                *= extended_dim_size[i + num_outer_dims + num_reduced_dims];
    }

    // Set up conf variables that don't change between phases
    conf.ndims = ndims;
    conf.alg = desc()->alg_kind;
    conf.power = desc()->p;
    conf.eps = desc()->eps;

    conf.div = reduced_dim_size;
    conf.outer_dim_size = outer_dim_size;
    conf.inner_dim_size = inner_dim_size;

    conf.attr_info = attr_info_t::create(attr());

    // Heuristics based on testing on PVC
    conf.sub_group_size = compute_engine->device_info()->max_subgroup_size();

    const int target_reduction_size = 8;

    // Pad the inner dim to a multiple of subgroup size
    conf.inner_dim_per_sg = std::min(reduced_dim_size,
            std::max(1, conf.sub_group_size / conf.inner_dim_size));
    conf.gws_inner_dim_size = utils::rnd_up(
            conf.inner_dim_per_sg * conf.inner_dim_size, conf.sub_group_size);

    while (reduced_dim_size > 1) {
        data_type_t src_data_type;
        bool is_first;
        if (reduced_dim_size == conf.div) {
            src_data_type = src_mdw.data_type();
            is_first = true;
        } else {
            src_data_type = types::default_accum_data_type(
                    src_mdw.data_type(), data_type::undef);
            is_first = false;
        }

        // Compute the number of phases left
        const int horiz_reductions
                = utils::div_up(reduced_dim_size, conf.inner_dim_per_sg);
        const int num_remaining_phases = std::floor(
                std::log(horiz_reductions) / std::log(target_reduction_size));
        const int red_per_phase
                = std::pow(reduced_dim_size, 1.0f / num_remaining_phases);

        int reduction_size;
        bool is_final;
        data_type_t dst_data_type;
        if (num_remaining_phases > 1) {
            reduction_size = red_per_phase;
            is_final = false;
            dst_data_type = types::default_accum_data_type(
                    src_mdw.data_type(), data_type::undef);
        } else {
            reduction_size = reduced_dim_size;
            is_final = true;
            dst_data_type = dst_mdw.data_type();
        }

        const int phase_start = reduced_dim_size;
        const int phase_reductions = reduction_size;
        const int phase_end = utils::div_up(phase_start, phase_reductions);

        // Set scratchpad sizes
        const int phase_num = conf.phases.size();
        if (!is_final && phase_num < 2) {
            conf.sp_size[phase_num]
                    = conf.outer_dim_size * phase_end * conf.inner_dim_size;
        }

        compute::dispatch_t dispatch = compute_engine->create_dispatch();
        size_t gws[3] = {1, 1, 1}, lws[3] = {1, 1, 1};
        gws[0] *= outer_dim_size * conf.gws_inner_dim_size * phase_end;

        // Set lws + pad gws simultaneously
        // - lws multiple of sub_group_size
        // - gws multiple of lws
        lws[0] = utils::rnd_up(std::min((int)gws[0], 256), conf.sub_group_size);
        gws[0] = utils::rnd_up(gws[0], lws[0]);
        compute::nd_range_t nd_range(gws, lws);

        conf.phases.push_back(init_phase(phase_start, phase_end,
                phase_reductions, src_data_type, dst_data_type, nd_range,
                is_final, is_first));
        reduced_dim_size = phase_end;
    }
    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const reduction_conf_t &conf, const reduction_phase_t &phase) {
    using namespace alg_kind;

    kernel_ctx.set_data_type(phase.src_type);

    // 1 ==> Use subgroups
    kernel_ctx.define_int("GWS_WITH_SG_DEFAULT", 1);
    kernel_ctx.define_int("GWS_SGS_DEFAULT", conf.sub_group_size);
    kernel_ctx.define_int("GWS_LWS0_DEFAULT", phase.nd_range.local_range()[0]);
    kernel_ctx.define_int("GWS_LWS1_DEFAULT", 1);
    kernel_ctx.define_int("GWS_LWS2_DEFAULT", 1);
    kernel_ctx.define_int("INNER_DIMS_PER_WI", conf.inner_dim_per_sg);

    kernel_ctx.define_int("REDUCTION_END_SIZE", phase.final_size);
    kernel_ctx.define_int("REDUCTION_SIZE", phase.initial_size);
    kernel_ctx.define_int("REDUCTION_CHUNK_SIZE", phase.num_reduction_chunks);
    kernel_ctx.define_int("OUTER_DIM_STRIDE",
            phase.num_reduction_chunks * conf.gws_inner_dim_size);
    kernel_ctx.define_int("DIV", conf.div);
    kernel_ctx.define_int("OUTER_DIM_SIZE", conf.outer_dim_size);
    kernel_ctx.define_int("INNER_DIM_SIZE", conf.inner_dim_size);
    kernel_ctx.define_int("PADDED_INNER_DIM_SIZE", conf.gws_inner_dim_size);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("POWER", conf.power);
    kernel_ctx.define_float("EPS", conf.eps);

    int sg_reduction_per_wi
            = utils::div_up(phase.reduction_size, conf.inner_dim_per_sg);
    sg_reduction_per_wi = std::min(conf.div, sg_reduction_per_wi);
    kernel_ctx.define_int("REDUCTIONS_PER_WI",
            sg_reduction_per_wi); // Can change between phases
    kernel_ctx.define_int("IS_FINAL", phase.is_final);
    kernel_ctx.define_int("IS_FIRST", phase.is_first);

    // Block loading is supported when inner dims are a multiple of 4 bytes
    if ((types::data_type_size(phase.src_type) * conf.inner_dim_size
                * conf.inner_dim_per_sg)
                    % 4
            == 0) {
        kernel_ctx.define_int("WITH_BLOCK_READ", 1);
    } else {
        kernel_ctx.define_int("WITH_BLOCK_READ", 0);
    }

    switch (conf.alg) {
        case reduction_max: kernel_ctx.define_int("IS_MAX", 1); break;
        case reduction_min: kernel_ctx.define_int("IS_MIN", 1); break;
        case reduction_mean: kernel_ctx.define_int("IS_MEAN", 1); break;
        case reduction_sum: kernel_ctx.define_int("IS_SUM", 1); break;
        case reduction_mul: kernel_ctx.define_int("IS_MUL", 1); break;
        case reduction_norm_lp_max:
            kernel_ctx.define_int("IS_LP_MAX", 1);
            break;
        case reduction_norm_lp_sum:
            kernel_ctx.define_int("IS_LP_SUM", 1);
            break;
        case reduction_norm_lp_power_p_max:
            kernel_ctx.define_int("IS_P_MAX", 1);
            break;
        case reduction_norm_lp_power_p_sum:
            kernel_ctx.define_int("IS_P_SUM", 1);
            break;
        default: return status::invalid_arguments;
    }

    def_data_type(kernel_ctx, phase.src_type, "SRC");
    def_data_type(kernel_ctx, phase.dst_type, "DST");

    return status::success;
}

status_t combined_reduction_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx,
        const reduction_phase_t &phase) const {
    return init_kernel_ctx_common(kernel_ctx, conf, phase);
}

status_t combined_reduction_t::execute_combined(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    std::unique_ptr<memory_storage_t> sp_reduce[2]
            = {ctx.get_scratchpad_grantor().get_memory_storage(
                       memory_tracking::names::key_reduction),
                    ctx.get_scratchpad_grantor().get_memory_storage(
                            memory_tracking::names::key_reduction_1)};
    const auto &conf = pd()->conf;

    status_t status = status::success;
    for (size_t i = 0; i < kernels.size(); i++) {
        auto &kernel = kernels[i];
        auto &phase = conf.phases[i];
        auto nd_range = phase.nd_range;

        // Set up the reduction arg list
        compute::kernel_arg_list_t reduction_arg_list;

        if (i == 0) {
            reduction_arg_list.set(0, src);
        } else {
            reduction_arg_list.set(0, *sp_reduce[(i - 1) % 2]);
        }

        if (i == kernels.size() - 1) {
            reduction_arg_list.set(1, dst);
        } else {
            reduction_arg_list.set(1, *sp_reduce[i % 2]);
        }

        status = parallel_for(ctx, nd_range, kernel, reduction_arg_list);
        CHECK(status);
    }
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
