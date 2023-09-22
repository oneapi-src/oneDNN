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

#include "gpu/ocl/combined_reduction.hpp"
#include "common/scratchpad.hpp"
#include "gpu/block_structure.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
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
    , subgroup_size(compute_engine->device_info()->max_subgroup_size())
    , with_block_reads(can_use_block_reads()) {

    const int num_EU = compute_engine->device_info()->eu_count();
    const int max_wg_size = static_cast<int>(
            compute_engine->device_info()->max_wg_size(large_grf_mode));

    // inner_dim can either be:
    // 1. packed into a single subgroup (small inner dim), or
    // 2. split among several subgroups (large inner dim)
    const dim_t num_packed_inner_dims
            = nstl::clamp(subgroup_size / inner_block.block, dim_t {1},
                    reduction_block.block);
    const dim_t num_split_inner_dims
            = utils::div_up(inner_block.block, subgroup_size); // S per I

    const dim_t num_horiz_reductions
            = reduction_block.block / num_packed_inner_dims;

    dim_t num_subgroups = outer_block.block * num_split_inner_dims;

    // We need to determine 2 variables according to some heuristic:
    // 1. Vector size (increases block load size)
    // 2. Threads per EU (decreases scheduling overhead, in this case)

    // Vector size requirements:
    // 1. (required) reductions and inner_dim aligned with no tails on either one
    // 2. (heuristic) Block loads should not exceed maximum instruction load size
    // 3. (heuristic) EUs should not become unsaturated due to vector size
    int nvec = 1;
    bool reduce_vec = false;
    if (with_block_reads) {
        const size_t single_load_size = types::data_type_size(src_type)
                * static_cast<size_t>(subgroup_size);
        const int max_load_size = 256; // Set on ATS-M, may depend on arch
        const int max_vect_size
                = static_cast<int>(max_load_size / single_load_size);

        for (int N : {8, 4, 2}) {
            // Related to EU saturation
            if (num_subgroups / N < num_EU) continue;
            // Related to block load size
            if (N > max_vect_size) continue;
            if (num_horiz_reductions % N == 0) {
                if (num_split_inner_dims == 1
                        || num_split_inner_dims % N == 0) {
                    nvec = N;
                    reduce_vec = (num_split_inner_dims == 1);
                    break;
                }
            }
        }
    }
    vect_size = nvec;
    reduce_vector = reduce_vec;

    if (!reduce_vector) num_subgroups /= vect_size;

    // Compute the number of threads per EU - this has no major impact
    // on average time, but can improve the best times on
    // close-to-cache-size problems with high parallelism
    const dim_t max_threads = num_subgroups / num_EU;
    dim_t threads_per_wg
            = nstl::clamp(static_cast<dim_t>(max_wg_size / subgroup_size),
                    dim_t {1}, max_threads);
    threads_per_wg = get_previous_factor(num_subgroups, threads_per_wg);

    // Compute the nd_range for this phase
    size_t gws[3] = {1, 1, 1}, lws[3] = {1, 1, 1};
    gws[0] = static_cast<size_t>(num_subgroups * subgroup_size);
    lws[0] = static_cast<size_t>(threads_per_wg * subgroup_size);
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

    const int subgroup_size
            = compute_engine->device_info()->max_subgroup_size();
    const dim_t inner_elems = subprb.inner_block.block;
    const dim_t reduction_elems = subprb.reduction_block.block;
    const dim_t outer_elems = subprb.outer_block.block;

    const dim_t inner_dim_per_sg
            = nstl::max(dim_t {1}, subgroup_size / inner_elems);
    const int num_EU = compute_engine->device_info()->eu_count();
    const dim_t num_sg_per_red_end
            = outer_elems * utils::div_up(inner_elems, subgroup_size);

    //Heuristics:
    // EU_mult: reduce parallelism to at most num_EU*EU_mult (reduces scheduling overhead?)
    const int EU_mult = 20;
    // Target single_phase_threshold horizontal reductions with each phase
    const int single_phase_threshold = 128;

    // Estimate the number of phases remaining, and divide it up evenly around this target
    int N = static_cast<int>(std::ceil(std::log2(reduction_elems)
            / std::log2(single_phase_threshold * inner_dim_per_sg)));
    N = std::max(1, N); // N must be positive
    dim_t reduction_end = static_cast<dim_t>(
            std::pow(reduction_elems, 1.0f - 1.0f / static_cast<float>(N)));

    // Reduce parallelism and finalize reduction_end
    reduction_end = nstl::clamp(
            num_EU * EU_mult / num_sg_per_red_end, dim_t {1}, reduction_end);
    reduction_end = get_previous_factor(reduction_elems, reduction_end);

    // Create the phase and recursively enter
    dim_t reduction_size = reduction_elems / reduction_end;

    if (reduction_end == 1) {
        phases.emplace_back(subprb, accum_data_type, accum_data_type,
                compute_engine, large_grf_mode);
        return status::success;
    } else {
        // Subdivide the subproblem by reducing by reduction_size first
        auto subdivided = subdivide_subproblem(subprb, reduction_size);
        phases.emplace_back(subdivided[0], accum_data_type, accum_data_type,
                compute_engine, large_grf_mode);
        return split_into_phases(subdivided[1], accum_data_type, compute_engine,
                phases, large_grf_mode);
    }
}

status_t combined_reduction_t::pd_t::init_conf(engine_t *engine) {
    // To start, check for compatibility
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const int ndims = src_mdw.ndims();
    const dim_t *src_dims = src_mdw.dims();
    const dim_t *src_padded_dims = src_mdw.padded_dims();
    const dim_t *dst_dims = dst_mdw.dims();

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

    kernel_ctx.set_data_type(phase.src_type);

    // Used for packing small inner vectors into a subgroup
    const dim_t inner_dim_per_sg
            = nstl::clamp(phase.subgroup_size / phase.inner_block.block,
                    dim_t {1}, phase.reduction_block.block);
    const dim_t num_horiz_reductions
            = phase.reduction_block.block / inner_dim_per_sg;

    kernel_ctx.define_int("SUBGROUP_SIZE", phase.subgroup_size);
    kernel_ctx.define_int(
            "LWS_SIZE", static_cast<int64_t>(phase.nd_range.local_range()[0]));

    kernel_ctx.define_int("DIV", conf.div);
    kernel_ctx.define_float("POWER", conf.power);
    kernel_ctx.define_float("EPS", conf.eps);

    kernel_ctx.define_int("OUTER_DIM_SIZE", phase.outer_block.block);
    kernel_ctx.define_int("REDUCTION_SIZE", phase.reduction_block.block);
    kernel_ctx.define_int("INNER_DIM_SIZE", phase.inner_block.block);

    kernel_ctx.define_int("IS_FINAL", phase.is_final);
    kernel_ctx.define_int("IS_FIRST", phase.is_first);

    kernel_ctx.define_int("VECT_DT_N", phase.vect_size);
    kernel_ctx.define_int("REDUCE_VECTOR", phase.reduce_vector ? 1 : 0);

    // Because the reduction loop is quite tight, we can override the compiler's
    // loop unrolling logic to increase it a lot and get a bit more speed
    // Heuristic determined on ATS-m, set to exclude the possibility of
    // exceeding the instruction cache
    const dim_t max_unroll = 256;
    const dim_t unroll_factor = nstl::clamp(
            num_horiz_reductions / (phase.reduce_vector ? phase.vect_size : 1),
            dim_t {1}, max_unroll);
    kernel_ctx.define_int("UNROLL_FACTOR", unroll_factor);

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
    CHECK(def_attr_info(
            kernel_ctx, conf.attr_info, attr()->post_ops_, dst_md()->dims));
    if (attr()->post_ops_.len() > 0) {
        if (phase.is_final) {
            // Can only do this for the final phase, since it overwrites def_data_type for DST
            def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");
        }
        def_offsets(conf.off.dst_off, kernel_ctx, "DST", conf.ndims);
    }

    return status;
}

status_t combined_reduction_t::execute_combined(const exec_ctx_t &ctx) const {
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

        append_post_ops_to_arg_list(
                ctx, reduction_arg_list, 2, pd()->attr()->post_ops_);

        status = parallel_for(ctx, nd_range, kernel, reduction_arg_list);
        CHECK(status);
    }
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
