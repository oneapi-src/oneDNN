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

using extended_dims_t = dim_t[2 * DNNL_MAX_NDIMS];

static reduction_phase_t init_phase(dim_t outer_dim_size, dim_t reduction_size,
        dim_t inner_dim_size, data_type_t src_type, data_type_t dst_type,
        const compute::compute_engine_t *compute_engine) {
    reduction_phase_t phase;
    phase.outer_dim_size = outer_dim_size;
    phase.reduction_size = reduction_size;
    phase.inner_dim_size = inner_dim_size;
    const int subgroup_size
            = compute_engine->device_info()->max_subgroup_size();
    const int num_EU = compute_engine->device_info()->eu_count();
    const size_t max_wg_size = compute_engine->device_info()->max_wg_size();

    // Derive relevant constants defined by the problem's shape
    // inner_dim can either be:
    // 1. packed into a single subgroup (small inner dim), or
    // 2. split among several subgroups (large inner dim)
    const int num_packed_inner_dims
            = nstl::clamp(subgroup_size / phase.inner_dim_size, (dim_t)1,
                    phase.reduction_size);
    const dim_t num_split_inner_dims
            = utils::div_up(phase.inner_dim_size, subgroup_size); // S per I
    const bool inner_dim_aligned = (subgroup_size % phase.inner_dim_size == 0
            || phase.inner_dim_size % subgroup_size == 0);

    const dim_t num_horiz_reductions
            = phase.reduction_size / num_packed_inner_dims;
    const int num_tail_reductions
            = phase.reduction_size % num_packed_inner_dims;
    const bool reductions_aligned = (num_tail_reductions == 0);

    int num_subgroups = phase.outer_dim_size * num_split_inner_dims;

    // We need to determine 2 variables according to some heuristic:
    // 1. Vector size (increases block load size)
    // 2. Threads per EU (decreases scheduling overhead, in this case)

    // Vector size requirements:
    // 1. (required) reductions and inner_dim aligned with no tails on either one
    // 2. (heuristic) Block loads should not exceed maximum instruction load size
    // 3. (heuristic) EUs should not become unsaturated due to vector size
    int nvec = 1;
    bool reduce_vec = false;
    if (reductions_aligned && inner_dim_aligned) {
        const size_t single_load_size
                = types::data_type_size(src_type) * subgroup_size;
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
    phase.vect_size = nvec;
    phase.reduce_vector = reduce_vec;

    if (!phase.reduce_vector) num_subgroups /= phase.vect_size;

    // Compute the number of threads per EU - this has no major impact
    // on average time, but can improve the best times on
    // close-to-cache-size problems with high parallelism
    const int max_threads = num_subgroups / num_EU;
    int threads_per_wg = nstl::clamp(
            static_cast<int>(max_wg_size / subgroup_size), 1, max_threads);
    threads_per_wg = get_previous_factor(num_subgroups, threads_per_wg);

    // Compute the nd_range for this phase
    size_t gws[3] = {1, 1, 1}, lws[3] = {1, 1, 1};
    gws[0] = num_subgroups * subgroup_size;
    lws[0] = threads_per_wg * subgroup_size;
    phase.nd_range = compute::nd_range_t(gws, lws);

    phase.src_type = src_type;
    phase.dst_type = dst_type;
    phase.is_first = false;
    phase.is_final = false;
    return phase;
}

void combined_reduction_t::pd_t::init_scratchpad() {
    // Only need scratchpads for the first 2 phases, since we can reuse them
    // and memory requirements are monotonically decreasing each phase.
    const uint32_t keys[2] = {memory_tracking::names::key_reduction,
            memory_tracking::names::key_reduction_1};

    auto scratchpad = scratchpad_registry().registrar();
    const int num_phases = static_cast<int>(conf.phases.size());
    const int num_scratchpads = std::min(num_phases - 1, 2);
    for (int i = 0; i < num_scratchpads; i++) {
        const reduction_phase_t &phase = conf.phases[i];
        const size_t sp_data_size = types::data_type_size(phase.dst_type);
        const int num_dst_elems = phase.outer_dim_size * phase.inner_dim_size;
        scratchpad.book(
                keys[i], num_dst_elems, sp_data_size, OCL_BUFFER_ALIGNMENT);
    }
}

// Represents the memory layout of memory_desc_wrapper, where plain and blocked dims
// each get their own full dimension. I.E. aBc16b would have ndims=4: 3 plain plus
// one blocked.
struct layout_t {
public:
    layout_t(const memory_desc_wrapper &mdw) {
        const dim_t *dims = mdw.dims();
        const dim_t *padded_dims = mdw.dims();
        const blocking_desc_t &blk = mdw.blocking_desc();
        const dim_t *strides = blk.strides;
        const int plain_ndims = mdw.ndims();

        ndims_ = plain_ndims + blk.inner_nblks;

        for (int i = 0; i < plain_ndims; i++) {
            dim_t src_stride = strides[i];
            dim_t dim_idx = i;
            for (int j = 0; j < i; j++) {
                if (src_stride > strides_[j]) {
                    // Insert this stride/idx into ordering
                    nstl::swap(perm_[j], dim_idx);
                    nstl::swap(strides_[j], src_stride);
                }
            }
            perm_[i] = dim_idx;
            strides_[i] = src_stride;
        }

        for (int i = 0; i < plain_ndims; i++) {
            dims_[i] = dims[perm_[i]];
            padded_dims_[i] = padded_dims[perm_[i]];
        }
        // Incorporate inner blocks into permutations/strides
        int inner_elems = 1;
        for (int i = blk.inner_nblks - 1; i >= 0; --i) {
            const dim_t blk_idx = blk.inner_idxs[i];
            const dim_t blk_size = blk.inner_blks[i];
            perm_[i + plain_ndims] = blk_idx;
            strides_[i + plain_ndims] = inner_elems;
            inner_elems *= blk_size;

            // Split up blocked dims into different components
            // (loses some information about padding)
            padded_dims_[i + plain_ndims] = blk.inner_blks[i];
            padded_dims_[blk_idx]
                    = utils::div_up(padded_dims_[blk_idx], blk_size);
            dims_[i + plain_ndims] = std::min(dims_[blk_idx], blk_size);
            dims_[blk_idx] = utils::div_up(dims_[blk_idx], blk_size);
        }
    }

    inline int ndims() const { return ndims_; }
    inline dim_t *perm() { return perm_; }
    inline dim_t *strides() { return strides_; }
    inline dim_t *dims() { return dims_; }
    inline dim_t *padded_dims() { return padded_dims_; }

private:
    int ndims_;
    // strides_, dims_, and padded_dims_ are ordered by decreasing stride
    // perm_[i] is the original mdw index for each dimension before reordering,
    // encoding the format tag (i.e. aBc16b => perm_ = {0, 1, 2, 1})
    extended_dims_t perm_, strides_;
    extended_dims_t dims_, padded_dims_;
};

status_t set_reduction_phases(dim_t outer_elems, dim_t reduction_elems,
        dim_t inner_elems, data_type_t accum_data_type, int subgroup_size,
        const compute::compute_engine_t *compute_engine,
        std::vector<reduction_phase_t> &phases) {
    // Recursive end condition: reduction_elems == 1
    if (reduction_elems == 1) return status::success;

    const dim_t inner_dim_per_sg
            = nstl::max((dim_t)1, subgroup_size / inner_elems);
    const int num_EU = compute_engine->device_info()->eu_count();
    const dim_t num_sg_per_red_end
            = outer_elems * utils::div_up(inner_elems, subgroup_size);

    //Heuristics:
    // EU_mult: reduce parallelism to at most num_EU*EU_mult (reduces scheduling overhead?)
    const int EU_mult = 20;
    // Target single_phase_threshold horizontal reductions with each phase
    const int single_phase_threshold = 128;

    // Estimate the number of phases remaining, and divide it up evenly around this target
    int N = (int)std::ceil(std::log2(reduction_elems)
            / std::log2(single_phase_threshold * inner_dim_per_sg));
    N = std::max(1, N); // N must be positive
    dim_t reduction_end
            = static_cast<dim_t>(std::pow(reduction_elems, (float)(N - 1) / N));

    // Reduce parallelism and finalize reduction_end
    reduction_end = nstl::clamp(
            num_EU * EU_mult / num_sg_per_red_end, (dim_t)1, reduction_end);
    reduction_end = get_previous_factor(reduction_elems, reduction_end);

    // Create the phase and recursively enter
    dim_t reduction_size = reduction_elems / reduction_end;
    phases.push_back(init_phase(outer_elems * reduction_end, reduction_size,
            inner_elems, accum_data_type, accum_data_type, compute_engine));

    return set_reduction_phases(outer_elems, reduction_end, inner_elems,
            accum_data_type, subgroup_size, compute_engine, phases);
}

status_t combined_reduction_t::pd_t::init_conf(engine_t *engine) {
    // To start, check for compatibility
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const int ndims = src_mdw.ndims();
    const dim_t *src_dims = src_mdw.dims();
    const dim_t *src_padded_dims = src_mdw.padded_dims();
    const dim_t *dst_dims = dst_mdw.dims();

    dims_t is_dim_reduced;
    for (int i = 0; i < ndims; i++) {
        // Actually reduced dimensions
        if (src_dims[i] != dst_dims[i]) {
            is_dim_reduced[i] = true;
            continue;
        }

        // Size-1 dims can be treated as reducible (at no cost):
        if (src_dims[i] == 1 && src_padded_dims[i] == 1) {
            is_dim_reduced[i] = true;
            continue;
        }

        is_dim_reduced[i] = false;
    }

    // Zero padding is not supported when dim is reduced
    // Or when doing an LP/P alg (not zero-preserving)
    const blocking_desc_t &dst_blk = dst_mdw.blocking_desc();
    using namespace alg_kind;
    for (int i = 0; i < dst_blk.inner_nblks; i++) {
        // Needs zero padding
        if (dst_mdw.padded_dims()[dst_blk.inner_idxs[i]]
                != dst_mdw.dims()[dst_blk.inner_idxs[i]]) {
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
            if (is_dim_reduced[dst_blk.inner_idxs[i]]) {
                return status::unimplemented;
            }
        }
    }

    // Convert plain/blocking dim structure to singular extended structure
    layout_t src_ext(src_mdw);
    layout_t dst_ext(dst_mdw);

    // Requirement: non-reduced dimensions appear in the same order in src and dst
    dim_t *src_ext_padded_dims = src_ext.padded_dims();
    dim_t *src_ext_perm = src_ext.perm();
    extended_dims_t non_reduced_src_perm;
    extended_dims_t non_reduced_src_dims;
    int non_reduced_src_ndims = 0;
    for (int i = 0; i < src_ext.ndims(); i++) {
        if (is_dim_reduced[src_ext_perm[i]]) continue;

        non_reduced_src_perm[non_reduced_src_ndims] = src_ext_perm[i];
        non_reduced_src_dims[non_reduced_src_ndims++] = src_ext_padded_dims[i];
    }

    dim_t *dst_ext_padded_dims = dst_ext.padded_dims();
    dim_t *dst_ext_perm = dst_ext.perm();
    extended_dims_t non_reduced_dst_perm;
    extended_dims_t non_reduced_dst_dims;
    int non_reduced_dst_ndims = 0;
    for (int i = 0; i < dst_ext.ndims(); i++) {
        if (is_dim_reduced[dst_ext_perm[i]]) continue;

        non_reduced_dst_perm[non_reduced_dst_ndims] = dst_ext_perm[i];
        non_reduced_dst_dims[non_reduced_dst_ndims++] = dst_ext_padded_dims[i];
    }

    // same number of non-reduced dims (plain + blocked)
    if (non_reduced_src_ndims != non_reduced_dst_ndims) {
        return status::unimplemented;
    }

    // non-reduced dims have the same order (plain + blocked)
    if (!utils::array_cmp(non_reduced_src_perm, non_reduced_dst_perm,
                non_reduced_src_ndims)) {
        return status::unimplemented;
    }

    // non-reduced dims have the same sizes (plain + blocked)
    if (!utils::array_cmp(non_reduced_src_dims, non_reduced_dst_dims,
                non_reduced_src_ndims)) {
        return status::unimplemented;
    }

    bool ext_reduced_dim[2 * DNNL_MAX_NDIMS];
    dim_t *src_ext_dims = src_ext.dims();
    for (int i = 0; i < src_ext.ndims(); i++) {
        if (is_dim_reduced[src_ext_perm[i]]) {
            ext_reduced_dim[i] = true;
            continue;
        }

        // Size-1 dims -- need to redo this check here since blocking
        // can create size-1 "dimensions" in the outer/plain slots
        if (src_ext_dims[i] == 1 && src_ext_padded_dims[i] == 1) {
            ext_reduced_dim[i] = true;
            continue;
        }

        // Remaining dims CANNOT be reduced
        ext_reduced_dim[i] = false;
    }

    const compute::compute_engine_t *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);

    conf.sub_group_size = compute_engine->device_info()->max_subgroup_size();
    // Starting from the innermost dimension, find the reduction dimensions and group neighboring
    // ones to be reduced simultaneously.
    data_type_t accum_data_type = types::default_accum_data_type(
            src_mdw.data_type(), data_type::undef);
    dim_t inner_elems = 1;
    dim_t outer_elems
            = utils::array_product(src_ext.padded_dims(), src_ext.ndims());
    for (int dim_idx = src_ext.ndims() - 1; dim_idx >= 0; --dim_idx) {
        // Check if this dimension is reduced
        if (ext_reduced_dim[dim_idx]) {
            int start_reduction_idx = dim_idx;
            int num_reduced_dims = 1;
            while (num_reduced_dims <= start_reduction_idx
                    && ext_reduced_dim[start_reduction_idx
                            - num_reduced_dims]) {
                ++num_reduced_dims;
            }
            int end_reduction_idx = start_reduction_idx - num_reduced_dims + 1;

            // Compute outer, reduction, and inner sizes for this chunk
            dim_t reduction_elems = utils::array_product(
                    src_ext_padded_dims + end_reduction_idx, num_reduced_dims);
            outer_elems /= reduction_elems;

            // Skip this phase if it's just a size-1 dimension (nop phase)
            if (reduction_elems > 1) {
                status_t status = set_reduction_phases(outer_elems,
                        reduction_elems, inner_elems, accum_data_type,
                        conf.sub_group_size, compute_engine, conf.phases);
                if (status != status::success) {
                    // Some error hit within the set_reduction_phases function
                    return status::unimplemented;
                }
            }
            dim_idx = end_reduction_idx;
        } else {
            // Just move this dimension to the inside
            outer_elems /= src_ext.padded_dims()[dim_idx];
            inner_elems *= src_ext.padded_dims()[dim_idx];
        }
    }

    // Compute div from basic mdw dims
    conf.div = 1;
    for (int i = 0; i < src_mdw.ndims(); i++) {
        if (is_dim_reduced[i]) conf.div *= src_dims[i];
    }

    // Set conf values
    conf.alg = desc()->alg_kind;
    conf.power = desc()->p;
    conf.eps = desc()->eps;
    conf.attr_info = attr_info_t::create(attr());

    // Set variables that matter for first/last phases
    conf.phases.front().is_first = true;
    conf.phases.front().src_type = src_mdw.data_type();

    conf.phases.back().is_final = true;
    conf.phases.back().dst_type = dst_mdw.data_type();

    // Post-ops require a bit more information
    if (attr()->post_ops_.len() > 0) {
        conf.ndims = dst_mdw.ndims();
        conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
        set_offsets(dst_mdw, conf.off.dst_off);
    }

    return status::success;
}

static bool can_block_read(dim_t upper_bound, dim_t stride, data_type_t dt) {
    // If size-1 dimension, can always block read
    if (upper_bound == 1) return true;

    // Otherwise, the stride has to be 4-byte aligned
    return (stride * types::data_type_size(dt) % 4 == 0);
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const reduction_conf_t &conf, const reduction_phase_t &phase) {
    using namespace alg_kind;

    kernel_ctx.set_data_type(phase.src_type);

    // Used for packing small inner vectors into a subgroup
    const dim_t inner_dim_per_sg
            = nstl::clamp(conf.sub_group_size / phase.inner_dim_size, (dim_t)1,
                    phase.reduction_size);
    const dim_t num_horiz_reductions = phase.reduction_size / inner_dim_per_sg;

    kernel_ctx.define_int("SUBGROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("LWS_SIZE", phase.nd_range.local_range()[0]);

    kernel_ctx.define_int("DIV", conf.div);
    kernel_ctx.define_int("POWER", conf.power);
    kernel_ctx.define_float("EPS", conf.eps);

    kernel_ctx.define_int("OUTER_DIM_SIZE", phase.outer_dim_size);
    kernel_ctx.define_int("REDUCTION_SIZE", phase.reduction_size);
    kernel_ctx.define_int("INNER_DIM_SIZE", phase.inner_dim_size);

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
            (dim_t)1, max_unroll);
    kernel_ctx.define_int("UNROLL_FACTOR", unroll_factor);

    // Block loading
    // 2 requirements:
    //  1) Pointer is 4-byte aligned (pointer has 3 access patterns, one for each dimension)
    const bool can_block_read_outer = can_block_read(phase.outer_dim_size,
            phase.reduction_size * phase.inner_dim_size, phase.src_type);
    const bool can_block_read_reduction = can_block_read(num_horiz_reductions,
            inner_dim_per_sg * phase.inner_dim_size, phase.src_type);
    const bool can_block_read_inner = can_block_read(
            phase.inner_dim_size, conf.sub_group_size, phase.src_type);

    //  2) All work items in a subgroup call the load function (inner dim and reduction sizes are coherent with subgroup size)
    const bool using_all_simd_channels
            = (phase.inner_dim_size * inner_dim_per_sg % conf.sub_group_size
                    == 0);
    const bool aligned_reduction
            = (phase.reduction_size == num_horiz_reductions * inner_dim_per_sg);

    const bool can_use_block_reads = can_block_read_outer
            && can_block_read_reduction && can_block_read_inner
            && using_all_simd_channels && aligned_reduction;

    kernel_ctx.define_int("WITH_BLOCK_READ", can_use_block_reads ? 1 : 0);

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
    status_t status = init_kernel_ctx_common(kernel_ctx, conf, phase);
    if (status != status_t::dnnl_success) return status;

    // Set post-op macros
    CHECK(def_attr_info(kernel_ctx, conf.attr_info, attr()->post_ops_));
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
    const auto &conf = pd()->conf;

    status_t status = status::success;
    for (size_t i = 0; i < kernels_.size(); i++) {
        auto &kernel = kernels_[i];
        auto &phase = conf.phases[i];
        auto nd_range = phase.nd_range;

        // Set up the reduction arg list
        compute::kernel_arg_list_t reduction_arg_list;

        if (i == 0) {
            reduction_arg_list.set(0, src);
        } else {
            reduction_arg_list.set(0, *sp_reduce[(i - 1) % 2]);
        }

        if (i == kernels_.size() - 1) {
            reduction_arg_list.set(1, dst);
        } else {
            reduction_arg_list.set(1, *sp_reduce[i % 2]);
        }

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
