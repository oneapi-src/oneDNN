/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include "gpu/intel/ocl/reduction/combined_reduction.hpp"
#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/ocl/reduction/reduction_utils.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

// Returns the next factor of big_num less than (or equal to) target
dim_t get_previous_factor(dim_t big_num, dim_t target) {
    for (dim_t i = 0; i < target; i++) {
        if (big_num % (target - i) == 0) return target - i;
    }
    return 1;
}

static bool can_block_read(dim_t upper_bound, dim_t stride, data_type_t dt) {
    // If size-1 dimension, can always block read
    if (upper_bound == 1) return true;

    // Otherwise, the stride has to be 4-byte aligned
    return (stride * static_cast<dim_t>(types::data_type_size(dt)) % 4 == 0);
}

bool reduction_phase_conf_t::can_use_block_reads() {
    const dim_t inner_dim_per_sg
            = nstl::clamp(subgroup_size / inner_block.block, dim_t {1},
                    reduction_block.block);
    const dim_t num_horiz_reductions = reduction_block.block / inner_dim_per_sg;

    // Block loading
    // 2 requirements:
    //  1) Pointer is 4-byte aligned (pointer has 3 access patterns, one for each dimension)
    const bool can_block_read_outer = can_block_read(outer_block.block,
            reduction_block.block * inner_block.block, src_type);
    const bool can_block_read_reduction = can_block_read(num_horiz_reductions,
            inner_dim_per_sg * inner_block.block, src_type);
    const bool can_block_read_inner
            = can_block_read(inner_block.block, subgroup_size, src_type);

    //  2) All work items in a subgroup call the load function (inner dim and reduction sizes are coherent with subgroup size)
    const bool using_all_simd_channels
            = (inner_block.block * inner_dim_per_sg % subgroup_size == 0);
    const bool aligned_reduction = (reduction_block.block
            == num_horiz_reductions * inner_dim_per_sg);

    return can_block_read_outer && can_block_read_reduction
            && can_block_read_inner && using_all_simd_channels
            && aligned_reduction;
}

reduction_phase_conf_t::reduction_phase_conf_t(
        const reduction_subproblem_t &subprb, data_type_t src_type,
        data_type_t dst_type, const compute::compute_engine_t *compute_engine,
        bool large_grf_mode)
    : reduction_subproblem_t(subprb)
    , src_type(src_type)
    , dst_type(dst_type)
    , subgroup_size(compute_engine->device_info()->max_subgroup_size()) {
    // Short-circuit if zero-dim is present
    gpu_assert(reduction_block.block != 0) << "Reducing over 0 elements";
    if (outer_block.block == 0 || inner_block.block == 0) {
        nd_range = compute::nd_range_t({0}, {into<size_t>(subgroup_size)});
        return;
    }
    with_block_reads = can_use_block_reads();

    const int num_EU = compute_engine->device_info()->eu_count();
    const int max_wg_size = static_cast<int>(
            compute_engine->device_info()->max_wg_size(large_grf_mode));
    compute::gpu_arch_t arch = compute_engine->device_info()->gpu_arch();
    int threads_per_eu
            = large_grf_mode ? 4 : compute::device_info_t::threads_per_eu(arch);
    int num_threads = num_EU * threads_per_eu;

    // inner_dim can either be:
    // 1. packed into a single subgroup (small inner dim), or
    // 2. split among several subgroups (large inner dim)
    const dim_t num_packed_inner_dims
            = nstl::clamp(subgroup_size / inner_block.block, dim_t {1},
                    reduction_block.block);
    const dim_t num_split_inner_dims
            = utils::div_up(inner_block.block, subgroup_size);

    int max_slm = utils::div_up(
            num_threads, outer_block.block * num_split_inner_dims);
    max_slm = nstl::min(max_slm, max_wg_size / subgroup_size);
    slm_reductions = [this, &num_packed_inner_dims, &max_slm]() {
        const dim_t rem_red = reduction_block.block / num_packed_inner_dims;
        // XXX: max_div no longer required
        int n_slm = into<int>(nstl::min(rem_red, into<dim_t>(max_slm)));
        return gpu_utils::dev_getenv("combined_reduction_n_slm", n_slm);
    }();
    dim_t num_subgroups
            = outer_block.block * num_split_inner_dims * slm_reductions;

    // Increase num_outer_idxs to use persistent threading to reduce the number of subgroups
    // and avoid overdispatching
    outer_tile_size = [this, &arch, &num_threads, &num_subgroups]() -> int {
        // Enable >1 block sizes only for PVC+, to avoid oldest-first thread arbitration
        dim_t block_size = 1;
        if (arch >= compute::gpu_arch_t::xe_hpc) {
            block_size = num_subgroups / num_threads;
            block_size = get_previous_factor(outer_block.block, block_size);
        }
        return gpu_utils::dev_getenv(
                "combined_reduction_num_outer", into<int>(block_size));
    }();
    gpu_assert(outer_block.block % outer_tile_size == 0)
            << "Invalid choice of persistent thread outer idxs";
    num_subgroups /= outer_tile_size;

    // Compute the nd_range for this phase
    compute::range_t gws(into<size_t>(num_subgroups * subgroup_size));
    compute::range_t lws(into<size_t>(slm_reductions * subgroup_size));
    nd_range = compute::nd_range_t(gws, lws);

    is_first = false;
    is_final = false;
}

void combined_reduction_t::pd_t::init_scratchpad() {
    // Only need scratchpads for the first 2 phases, since we can reuse them
    // and memory requirements are monotonically decreasing each phase.
    const uint32_t keys[2] = {memory_tracking::names::key_reduction,
            memory_tracking::names::key_reduction_1};

    auto scratchpad = scratchpad_registry().registrar();
    const size_t num_phases = phases.size();
    const size_t num_scratchpads = std::min(num_phases - 1, size_t {2});
    for (size_t i = 0; i < num_scratchpads; i++) {
        const reduction_phase_conf_t &phase = phases[i];
        const size_t sp_data_size = types::data_type_size(phase.dst_type);
        const size_t num_dst_elems = static_cast<size_t>(
                phase.outer_block.block * phase.inner_block.block);
        scratchpad.book(
                keys[i], num_dst_elems, sp_data_size, OCL_BUFFER_ALIGNMENT);
    }
}

// Further subdivides a subproblem, by applying part of the reduction
std::array<reduction_subproblem_t, 2> subdivide_subproblem(
        const reduction_subproblem_t &subprb, dim_t reduction_size) {
    const block_t &reduction_block = subprb.reduction_block;
    assert(reduction_block.block % reduction_size == 0);
    const dim_t remaining_reduction = reduction_block.block / reduction_size;
    const dim_t inner = subprb.inner_block.block;
    const dim_t outer = subprb.outer_block.block;

    reduction_subproblem_t prb0(
            inner, reduction_size, outer * remaining_reduction);

    block_t next_reduction(1, remaining_reduction, inner);
    reduction_subproblem_t prb1(inner, remaining_reduction, outer);

    prb0.src_zpads = subprb.src_zpads;
    prb1.dst_zpads = subprb.dst_zpads;

    return {std::move(prb0), std::move(prb1)};
}

status_t split_into_phases(const reduction_subproblem_t &subprb,
        data_type_t accum_data_type,
        const compute::compute_engine_t *compute_engine,
        std::vector<reduction_phase_conf_t> &phases, bool large_grf_mode) {
    const dim_t reduction_elems = subprb.reduction_block.block;
    reduction_phase_conf_t try_phase(subprb, accum_data_type, accum_data_type,
            compute_engine, large_grf_mode);
    // Zero-dim short circuit
    if (try_phase.outer_block.block == 0 || try_phase.inner_block.block == 0) {
        phases.emplace_back(try_phase);
        return status::success;
    }

    //Heuristic:
    // subsplitting has a high cost due to launching multiple sequential threads,
    // so only split when parallelism is low and reductions per thread is large
    const bool low_parallelism = [&compute_engine, &large_grf_mode,
                                         &try_phase]() {
        compute::gpu_arch_t arch = compute_engine->device_info()->gpu_arch();
        int threads_per_EU = large_grf_mode
                ? 4
                : compute::device_info_t::threads_per_eu(arch);
        const int num_EU = compute_engine->device_info()->eu_count();
        const int min_threads = gpu_utils::dev_getenv(
                "combined_reduction_occ_thresh", threads_per_EU * num_EU / 2);
        const int dispatched_threads
                = into<int>(try_phase.nd_range.global_range()[0]
                        / into<size_t>(try_phase.subgroup_size));
        return dispatched_threads < min_threads;
    }();
    const bool large_reduction = [&try_phase]() {
        const int slm_red = into<int>(try_phase.nd_range.local_range()[0]
                / into<size_t>(try_phase.subgroup_size));
        const dim_t sg_red = nstl::clamp(
                try_phase.subgroup_size / try_phase.inner_block.block,
                dim_t {1}, try_phase.reduction_block.block);
        const dim_t red_per_thread
                = try_phase.reduction_block.block / slm_red / sg_red;
        const int red_thresh
                = gpu_utils::dev_getenv("combined_reduction_split_thresh", 128);
        return red_per_thread >= red_thresh;
    }();
    if (!large_reduction || !low_parallelism) {
        phases.emplace_back(try_phase);
        return status::success;
    }

    // Split into 2 phases
    dim_t reduction_end = static_cast<dim_t>(std::sqrt(reduction_elems));
    reduction_end = get_previous_factor(reduction_elems, reduction_end);

    auto subdivided
            = subdivide_subproblem(subprb, reduction_elems / reduction_end);
    phases.emplace_back(subdivided[0], accum_data_type, accum_data_type,
            compute_engine, large_grf_mode);
    if (reduction_end > 1) {
        phases.emplace_back(subdivided[1], accum_data_type, accum_data_type,
                compute_engine, large_grf_mode);
    }
    return status::success;
}

status_t combined_reduction_t::pd_t::init_conf(impl::engine_t *engine) {
    // To start, check for compatibility
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const int ndims = src_mdw.ndims();
    const dim_t *src_dims = src_mdw.dims();
    const dim_t *src_padded_dims = src_mdw.padded_dims();
    const dim_t *dst_dims = dst_mdw.dims();

    // Implementation uses int for offset calculations
    if (src_mdw.nelems(true) > INT_MAX || dst_mdw.nelems(true) > INT_MAX)
        return status::unimplemented;

    for (int i = 0; i < ndims; i++) {
        // Actually reduced dimensions
        if (src_dims[i] != dst_dims[i]) {
            conf.is_reduction_dim[i] = true;
            continue;
        }

        // Size-1 dims can be treated as reducible (at no cost):
        if (src_dims[i] == 1 && src_padded_dims[i] == 1) {
            conf.is_reduction_dim[i] = true;
            continue;
        }

        conf.is_reduction_dim[i] = false;
    }

    using namespace alg_kind;
    std::vector<reduction_subproblem_t> subprbs;
    CHECK(generate_reduction_phases(src_md(), dst_md(), subprbs));

    // Heuristic: Checking for src zero padding in the reduction loop is slow.
    // For now, if the reduction dim for any subproblem contains zero-padded elements,
    // only allow algs which can safely accumulate them without affecting the result.
    const bool alg_affected_by_zeros = utils::one_of(
            desc()->alg_kind, reduction_min, reduction_max, reduction_mul);
    bool accumulating_src_zpad = false;
    for (const auto &subprb : subprbs) {
        for (const auto &zpad : subprb.src_zpads) {
            if (conf.is_reduction_dim[zpad.dim_idx]) {
                accumulating_src_zpad = true;
                break;
            }
        }
    }

    if (accumulating_src_zpad && alg_affected_by_zeros) {
        return status::unimplemented;
    }

    const compute::compute_engine_t *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);

    auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    bool large_grf_mode = gpu_attr && gpu_attr->threads_per_eu() == 4;
    // Further break up phases if needed, for parallelism
    data_type_t accum_data_type = types::default_accum_data_type(
            src_mdw.data_type(), data_type::undef);
    for (auto &subprb : subprbs) {
        CHECK(split_into_phases(subprb, accum_data_type, compute_engine, phases,
                large_grf_mode));
    }

    // Compute div from basic mdw dims
    conf.div = 1;
    for (int i = 0; i < src_mdw.ndims(); i++) {
        if (conf.is_reduction_dim[i]) conf.div *= src_dims[i];
    }

    // Set conf values
    conf.alg = desc()->alg_kind;
    conf.power = desc()->p;
    conf.eps = desc()->eps;
    conf.attr_info = attr_info_t::create(attr());

    // Set variables that matter for first/last phases
    phases.front().is_first = true;
    phases.front().src_type = src_mdw.data_type();

    phases.back().is_final = true;
    phases.back().dst_type = dst_mdw.data_type();

    // Post-ops require a bit more information
    if (attr()->post_ops_.len() > 0) {
        conf.ndims = dst_mdw.ndims();
        conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
        set_offsets(dst_mdw, conf.off.dst_off);
    }

    return status::success;
}

void def_zero_pad(compute::kernel_ctx_t &kernel_ctx, const char *prefix,
        const zero_padding_t &zpad, size_t idx) {
    const std::string size_name = utils::format("%s_Z%zu_SIZE", prefix, idx);
    kernel_ctx.define_int(size_name, zpad.data_size);

    // Inner block defines
    {
        const std::string padded_name
                = utils::format("%s_Z%zu_SIZE0", prefix, idx);
        const std::string stride_name
                = utils::format("%s_Z%zu_STRIDE0", prefix, idx);
        kernel_ctx.define_int(padded_name, zpad.inner_size);
        kernel_ctx.define_int(stride_name, zpad.inner_stride);
    }
    // Outer block defines
    {
        const std::string padded_name
                = utils::format("%s_Z%zu_SIZE1", prefix, idx);
        const std::string stride_name
                = utils::format("%s_Z%zu_STRIDE1", prefix, idx);
        kernel_ctx.define_int(padded_name, zpad.outer_size);
        kernel_ctx.define_int(stride_name, zpad.outer_stride);
    }
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const reduction_conf_t &conf, const reduction_phase_conf_t &phase) {
    using namespace alg_kind;

    def_reduction_alg_kinds(kernel_ctx);
    reduction_alg_kind_t alg
            = from_alg(conf.alg, phase.is_first, phase.is_final);
    reduction_alg_kind_t secondary_alg
            = from_alg(conf.alg, false, phase.is_final);
    kernel_ctx.define_int("REDUCTION_ALG", to_int(alg));
    kernel_ctx.define_int("SECONDARY_REDUCTION_ALG", to_int(secondary_alg));

    kernel_ctx.set_data_type(phase.src_type);

    kernel_ctx.define_int("SUBGROUP_SIZE", phase.subgroup_size);
    const auto &lws = phase.nd_range.local_range();
    if (!lws) return status::runtime_error;
    kernel_ctx.define_int("LWS_SIZE", static_cast<int64_t>(lws[0]));

    kernel_ctx.define_int("DIV", conf.div);
    kernel_ctx.define_float("POWER", conf.power);
    kernel_ctx.define_float("EPS", conf.eps);

    kernel_ctx.define_int("OUTER_DIM_SIZE", phase.outer_block.block);
    kernel_ctx.define_int("REDUCTION_SIZE", phase.reduction_block.block);
    kernel_ctx.define_int("INNER_DIM_SIZE", phase.inner_block.block);

    kernel_ctx.define_int("OUTER_TILE_SIZE", phase.outer_tile_size);

    kernel_ctx.define_int("IS_FINAL", phase.is_final);
    kernel_ctx.define_int("IS_FIRST", phase.is_first);

    kernel_ctx.define_int("WITH_BLOCK_READ", phase.with_block_reads ? 1 : 0);

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

    // Def zero-padding variables
    kernel_ctx.define_int(
            "NUM_SRC_ZPAD", static_cast<int64_t>(phase.src_zpads.size()));
    for (size_t i = 0; i < phase.src_zpads.size(); i++) {
        def_zero_pad(kernel_ctx, "SRC", phase.src_zpads[i], i);
    }
    kernel_ctx.define_int(
            "NUM_DST_ZPAD", static_cast<int64_t>(phase.dst_zpads.size()));
    for (size_t i = 0; i < phase.dst_zpads.size(); i++) {
        def_zero_pad(kernel_ctx, "DST", phase.dst_zpads[i], i);
        const std::string is_reduced_name
                = utils::format("DST_Z%zu_IS_REDUCED", i);
        kernel_ctx.define_int(is_reduced_name,
                conf.is_reduction_dim[phase.dst_zpads[i].dim_idx]);
    }

    def_data_type(kernel_ctx, phase.src_type, "SRC");
    def_data_type(kernel_ctx, phase.dst_type, "DST");

    return status::success;
}

status_t combined_reduction_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx,
        const reduction_phase_conf_t &phase) const {
    status_t status = init_kernel_ctx_common(kernel_ctx, conf, phase);
    if (status != status_t::dnnl_success) return status;

    // Set post-op macros
    auto empty_po = post_ops_t();
    const auto &actual_po = &attr()->post_ops_;
    const post_ops_t *po = phase.is_final ? actual_po : &empty_po;

    CHECK(def_attr_info(kernel_ctx, conf.attr_info, *po, *dst_md()));
    if (attr()->post_ops_.len() > 0 && phase.is_final) {
        // Can only do this for the final phase, since it overwrites def_data_type for DST
        def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");
        def_offsets(conf.off.dst_off, kernel_ctx, "DST", conf.ndims);
    }

    return status;
}

status_t combined_reduction_t::execute_combined(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    std::unique_ptr<memory_storage_t> sp_reduce[2]
            = {ctx.get_scratchpad_grantor().get_memory_storage(
                       memory_tracking::names::key_reduction),
                    ctx.get_scratchpad_grantor().get_memory_storage(
                            memory_tracking::names::key_reduction_1)};

    status_t status = status::success;
    for (size_t i = 0; i < kernels_.size(); i++) {
        auto &kernel = kernels_[i];
        auto &phase = pd()->phases[i];
        auto nd_range = phase.nd_range;

        // Set up the reduction arg list
        compute::kernel_arg_list_t reduction_arg_list;

        memory_storage_t &src_mem = (i == 0) ? src : *sp_reduce[(i - 1) % 2];
        memory_storage_t &dst_mem
                = (i == kernels_.size() - 1) ? dst : *sp_reduce[i % 2];

        reduction_arg_list.set(0, src_mem);
        reduction_arg_list.set(1, dst_mem);

        // nullify post ops unless it's the final phase
        auto empty_po = post_ops_t();
        const auto &actual_po = &pd()->attr()->post_ops_;
        const post_ops_t *po = phase.is_final ? actual_po : &empty_po;
        append_post_ops_to_arg_list(ctx, reduction_arg_list, 2, *po);

        status = parallel_for(ctx, nd_range, kernel, reduction_arg_list);
        CHECK(status);
    }
    return status;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
