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
#include "gpu/ocl/gen9_reduction.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using extended_dims_t = dim_t[2 * DNNL_MAX_NDIMS];

static reduction_phase_t init_phase(dim_t start, dim_t end,
        dim_t num_reductions, data_type_t src_type, data_type_t dst_type,
        compute::nd_range_t nd_range) {
    reduction_phase_t phase;
    phase.initial_size = start;
    phase.reduction_size = num_reductions;
    phase.final_size = end;
    phase.num_reduction_chunks
            = utils::div_up(phase.initial_size, phase.reduction_size);
    phase.src_type = src_type;
    phase.dst_type = dst_type;
    phase.nd_range = nd_range;
    phase.is_first = false;
    phase.is_final = false;
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

        m_ndims = plain_ndims + blk.inner_nblks;

        for (int i = 0; i < plain_ndims; i++) {
            dim_t src_stride = strides[i];
            dim_t dim_idx = i;
            for (int j = 0; j < i; j++) {
                if (src_stride > strides[perm_[j]]) {
                    // Insert this stride/idx into ordering
                    src_stride = strides[perm_[j]];
                    nstl::swap(perm_[j], dim_idx);
                    nstl::swap(strides_[j], src_stride);
                }
            }
            perm_[i] = dim_idx;
            strides_[i] = src_stride;
        }
        for (int i = 0; i < plain_ndims; i++) {
            dims_[i] = dims[i];
            padded_dims_[i] = padded_dims[i];
        }
        // Incorporate inner blocks into permutations/strides
        int inner_elems = 1;
        for (int i = blk.inner_nblks - 1; i >= 0; --i) {
            perm_[i + plain_ndims] = plain_ndims + i;
            strides_[i + plain_ndims] = inner_elems;
            inner_elems *= blk.inner_blks[i];

            // Split up blocked dims into different components
            // (loses some information about padding)
            padded_dims_[i + plain_ndims] = blk.inner_blks[i];
            padded_dims_[blk.inner_idxs[i]] = utils::div_up(
                    padded_dims_[blk.inner_idxs[i]], blk.inner_blks[i]);
            dims_[i + plain_ndims]
                    = std::min(dims_[blk.inner_idxs[i]], blk.inner_blks[i]);
            dims_[blk.inner_idxs[i]] = utils::div_up(
                    dims_[blk.inner_idxs[i]], blk.inner_blks[i]);
        }
    }

    inline int ndims() const { return m_ndims; }
    inline dim_t *perm() { return perm_; }
    inline dim_t *strides() { return strides_; }
    inline dim_t *dims() { return dims_; }
    inline dim_t *padded_dims() { return padded_dims_; }

private:
    int m_ndims;
    extended_dims_t perm_, strides_;
    extended_dims_t dims_, padded_dims_;
};

status_t combined_reduction_t::pd_t::init_conf(engine_t *engine) {
    // To start, check for compatibility
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const int ndims = src_mdw.ndims();
    const dim_t *src_dims = src_mdw.dims();
    const dim_t *dst_dims = dst_mdw.dims();

    dims_t is_dim_reduced;
    for (int i = 0; i < ndims; i++) {
        is_dim_reduced[i] = src_dims[i] != dst_dims[i];
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

    // If this shape is sparse, use a different
    // implementation (unless it's ref, use this anyway)
    constexpr float required_density = 1.0f / 16;
    const int padded_elems
            = utils::array_product(src_mdw.padded_dims(), src_mdw.ndims());
    const int nelems = utils::array_product(src_mdw.dims(), src_mdw.ndims());
    if ((float)nelems / padded_elems < required_density) {
        // Use gen9 if possible
        if (gen9_reduction_t::is_compatible(src_mdw, dst_mdw, nullptr)) {
            return status::unimplemented;
        }
    }

    // Convert plain/blocking dim structure to singular extended structure
    layout_t src_ext(src_mdw);
    layout_t dst_ext(dst_mdw);

    // Requirement: src/dst have same dim ordering
    if (src_ext.ndims() != dst_ext.ndims()) return status::unimplemented;
    if (!utils::array_cmp(src_ext.perm(), dst_ext.perm(), src_ext.ndims())) {
        return status::unimplemented;
    }

    // Requirement: each dim must remain unchanged (src=dst) or be completely reduced (src!=1, dst=1)
    dim_t *src_ext_dims = src_ext.dims();
    dim_t *dst_ext_dims = dst_ext.dims();
    dim_t *src_ext_padded_dims = src_ext.padded_dims();
    dim_t *dst_ext_padded_dims = dst_ext.padded_dims();
    bool ext_reduced_dim[2 * DNNL_MAX_NDIMS];
    for (int i = 0; i < src_ext.ndims(); i++) {
        if (src_ext_dims[i] == dst_ext_dims[i]) {
            ext_reduced_dim[i] = false;
            continue;
        }
        if (dst_ext_dims[i] != 1) return status::unimplemented;
        ext_reduced_dim[i] = true;
    }

    // Same for padded dims (can't change blocking structure)
    for (int i = 0; i < src_ext.ndims(); i++) {
        if (src_ext_padded_dims[i] == dst_ext_padded_dims[i]) continue;
        if (dst_ext_padded_dims[i] != 1) return status::unimplemented;
    }

    const compute::compute_engine_t *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);

    // Split the extended dims into 3 sets of dims: outer, inner, and reduction
    int separate_ndims[3] = {0};
    extended_dims_t src_sep_dims[3];
    dim_t *ext_perm = src_ext.perm();
    int state = 0; // 0=outer, 1=reduction, 2=inner
    for (int i = 0; i < src_ext.ndims(); i++) {
        // If in outer dims and reduced, move to reduced dims
        // If in reduced dims and not reduced, move to inner dims
        if ((state == 0 && ext_reduced_dim[ext_perm[i]])
                || (state == 1 && !ext_reduced_dim[ext_perm[i]])) {
            state += 1;
        }
        // If in inner dims and reduced, unimplemented
        if (state == 2 && ext_reduced_dim[ext_perm[i]]) {
            return status::unimplemented;
        }
        src_sep_dims[state][separate_ndims[state]]
                = src_ext.padded_dims()[src_ext.perm()[i]];
        separate_ndims[state] += 1;
    }

    // Combine each separate dim to a single composite one
    dim_t composite_dims[3] = {1};
    composite_dims[0]
            = utils::array_product(src_sep_dims[0], separate_ndims[0]);
    composite_dims[1]
            = utils::array_product(src_sep_dims[1], separate_ndims[1]);
    composite_dims[2]
            = utils::array_product(src_sep_dims[2], separate_ndims[2]);

    // Set up conf variables that don't change between phases
    conf.ndims = ndims;
    conf.alg = desc()->alg_kind;
    conf.power = desc()->p;
    conf.eps = desc()->eps;

    conf.outer_dim_size = composite_dims[0];
    conf.div = composite_dims[1];
    conf.inner_dim_size = composite_dims[2];

    conf.attr_info = attr_info_t::create(attr());

    // Heuristics based on testing on PVC
    conf.sub_group_size = compute_engine->device_info()->max_subgroup_size();

    // Each phase will attempt to perform this many horizontal reductions
    int target_horiz_reductions = 16;
    // Increase work per wi past target so we spawn less than max_subgroups subgroups
    const int max_subgroups = 1024;
    // If there are fewer than this many horizontal reductions,
    // perform all remaining reductions in a single phase
    const int single_phase_threshold = 40;

    // LP algs require more flops, so they should do fewer reductions
    switch (conf.alg) {
        case reduction_norm_lp_max:
        case reduction_norm_lp_sum:
        case reduction_norm_lp_power_p_max:
        case reduction_norm_lp_power_p_sum: target_horiz_reductions = 4;
        default: break;
    }

    // Pad the inner dim to a multiple of subgroup size
    conf.inner_dim_per_sg = std::min(composite_dims[1],
            std::max((dim_t)1, conf.sub_group_size / conf.inner_dim_size));
    conf.gws_inner_dim_size = utils::rnd_up(
            conf.inner_dim_per_sg * conf.inner_dim_size, conf.sub_group_size);

    const dim_t sg_per_inner_dim
            = utils::div_up(conf.inner_dim_size, conf.sub_group_size);

    // Each phase actually consists of 2 stages:
    // 1. Parallelized across work items in a subgroup: reduce by num_horizontal_reductions
    // 2. Between work items in a subgroup: reduce by conf.inner_dim_per_sg
    // The minimum possible reduction per subgroup is conf.inner_dim_per_sg
    dim_t reduced_dim_size = composite_dims[1];
    while (reduced_dim_size > 1) {
        data_type_t src_data_type = types::default_accum_data_type(
                src_mdw.data_type(), data_type::undef);
        data_type_t dst_data_type = types::default_accum_data_type(
                src_mdw.data_type(), data_type::undef);

        // Heuristic:
        // Keep horizontal reductions at the target:
        dim_t reduction_size = target_horiz_reductions * conf.inner_dim_per_sg;

        // Except when:
        // 1) total horizontal_reductions < minimum: reduce everything (another phase isn't worth it)
        const dim_t horiz_reductions
                = utils::div_up(reduced_dim_size, conf.inner_dim_per_sg);
        if (horiz_reductions <= single_phase_threshold) {
            reduction_size = reduced_dim_size;
        }

        // 2) total subgroups > max: increase reduction since some parallelism is lost due to high dispatching
        dim_t reduction_end = utils::div_up(reduced_dim_size, reduction_size);
        dim_t num_dst_elems
                = conf.outer_dim_size * conf.inner_dim_size * reduction_end;
        int num_subgroups = utils::div_up(
                num_dst_elems * sg_per_inner_dim, conf.inner_dim_size);

        // Outer dims are independent, so base this heuristic on "subgroups per outer dim"
        if (num_subgroups
                > max_subgroups * conf.outer_dim_size * sg_per_inner_dim) {
            reduction_size *= (float)num_subgroups
                    / (max_subgroups * conf.outer_dim_size * sg_per_inner_dim);

            reduction_end = utils::div_up(reduced_dim_size, reduction_size);
            num_dst_elems
                    = conf.outer_dim_size * conf.inner_dim_size * reduction_end;
            num_subgroups = utils::div_up(
                    num_dst_elems * sg_per_inner_dim, conf.inner_dim_size);
        }
        // End Heuristic

        reduction_end = utils::div_up(reduced_dim_size, reduction_size);

        // Shrink reduction_size without changing the final shape
        // ex: div_up(41,16)=3, div_up(41,3)=14 - only 14 reductions needed, not 16
        reduction_size = utils::div_up(reduced_dim_size, reduction_end);
        reduction_size = utils::rnd_up(reduction_size, conf.inner_dim_per_sg);

        // Clamp reduction size according to 2-phase kernel
        reduction_size = std::min(reduced_dim_size,
                std::max(conf.inner_dim_per_sg, reduction_size));

        num_dst_elems
                = conf.outer_dim_size * conf.inner_dim_size * reduction_end;
        num_subgroups = utils::div_up(
                num_dst_elems * sg_per_inner_dim, conf.inner_dim_size);

        const dim_t phase_start = reduced_dim_size;
        const dim_t phase_reductions = reduction_size;
        const dim_t phase_end = utils::div_up(phase_start, phase_reductions);

        // Set scratchpad sizes
        const int phase_num = static_cast<int>(conf.phases.size());
        if (phase_num < 2) conf.sp_size[phase_num] = num_dst_elems;

        compute::dispatch_t dispatch = compute_engine->create_dispatch();
        size_t gws[3] = {1, 1, 1}, lws[3] = {1, 1, 1};
        gws[0] = num_subgroups * conf.sub_group_size;

        // Set lws + pad gws simultaneously
        // - lws multiple of sub_group_size
        // - gws multiple of lws
        lws[0] = conf.sub_group_size;
        gws[0] = utils::rnd_up(gws[0], lws[0]);
        compute::nd_range_t nd_range(gws, lws);

        conf.phases.push_back(init_phase(phase_start, phase_end,
                phase_reductions, src_data_type, dst_data_type, nd_range));

        reduced_dim_size = phase_end;
    }

    // Set variables that matter for first/last phases
    conf.phases.front().is_first = true;
    conf.phases.front().src_type = src_mdw.data_type();

    conf.phases.back().is_final = true;
    conf.phases.back().dst_type = dst_mdw.data_type();

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
    kernel_ctx.define_int("REDUCTION_START_SIZE", phase.initial_size);
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

    kernel_ctx.define_int("REDUCTION_SIZE", phase.reduction_size);
    int sg_reduction_per_wi
            = utils::div_up(phase.reduction_size, conf.inner_dim_per_sg);
    sg_reduction_per_wi = std::min(conf.div, sg_reduction_per_wi);
    kernel_ctx.define_int("REDUCTIONS_PER_WI", sg_reduction_per_wi);
    kernel_ctx.define_int("IS_FINAL", phase.is_final);
    kernel_ctx.define_int("IS_FIRST", phase.is_first);

    // Block loading is supported when inner dims are a multiple of 4 bytes
    const size_t src_dt_size = types::data_type_size(phase.src_type);
    const int nelems_per_sg = conf.inner_dim_size * conf.inner_dim_per_sg;
    const size_t read_bytes = src_dt_size * nelems_per_sg;
    const bool use_block_reads = (read_bytes % 4 == 0)
            && (phase.initial_size % nelems_per_sg == 0);
    kernel_ctx.define_int("WITH_BLOCK_READ", use_block_reads ? 1 : 0);

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

    kernel_ctx.define_int("UNROLL_AMOUNT", std::min(sg_reduction_per_wi, 8));

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

        status = parallel_for(ctx, nd_range, kernel, reduction_arg_list);
        CHECK(status);
    }
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
