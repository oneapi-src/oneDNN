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

#include <limits>
#include "common/eltwise_pd.hpp"
#include "common/utils.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gpu_primitive_attr.hpp"
#include "gpu/intel/ocl/reduction/atomic_reduction.hpp"
#include "gpu/intel/ocl/reduction/reduction_utils.hpp"
#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace gpu_utils;

class atomic_lws_strategy_t : public compute::lws_strategy_t {
public:
    bool is_included(const compute::mapped_block_t &blocks) const override {
        for (const block_t &block : inc_blocks) {
            if (blocks.get_dim_idx() == into<size_t>(block.dim_idx)) {
                return true;
            }
        }
        return false;
    };

    void include(dim_idx_t dim, size_t size) {
        inc_blocks.emplace_back(into<dim_t>(dim), into<dim_t>(size), 1);
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
                if (bins[0].get_dim_idx() == into<size_t>(inc_block.dim_idx)) {
                    lws[i] *= into<size_t>(inc_block.block);
                }
            }
        }

        return lws;
    };

    std::vector<block_t> inc_blocks;
};

// 3 relevant blocks: inner, reduction, and outer
// inner is broken up into subgroup, vector, and inner_group
//  -- vector block is implicit: it's not dispatched or broadcasted,
//      its indexing is handled manually to make use of compiled constants
// reduction is broken up into global, local, and loop
// outer is left unchanged
namespace reduction_dims {
dim_idx_t subgroup = 0;
// implicit vector = 1
dim_idx_t inner_group = 2;
dim_idx_t global = 3;
dim_idx_t local = 4;
dim_idx_t loop = 5;
dim_idx_t outer = 6;
} // namespace reduction_dims

atomic_reduction_conf_t::atomic_reduction_conf_t(
        const reduction_subproblem_t &subprb, reduction_alg_kind_t alg,
        reduction_alg_kind_t secondary_alg, data_type_t src_type,
        data_type_t dst_type, const compute::device_info_t &device_info,
        gpu_primitive_attr_t *gpu_attr)
    : reduction_subproblem_t(subprb) {
    conf.src_type = src_type;
    conf.dst_type = dst_type;
    conf.subgroup_size = device_info.max_subgroup_size();
    // Short-circuit if zero-dim is present
    gpu_assert(reduction_block.block != 0) << "Reducing over 0 elements";
    if (outer_block.block == 0 || inner_block.block == 0) return;

    auto arch = device_info.gpu_arch();
    const int base_threads_per_eu
            = compute::device_info_t::threads_per_eu(arch);
    conf.threads_per_eu
            = gpu_attr ? gpu_attr->threads_per_eu() : base_threads_per_eu;

    const bool large_grf_mode = conf.threads_per_eu == 4;
    const size_t max_wg_size = device_info.max_wg_size(large_grf_mode);
    const int eu_count = device_info.eu_count();
    const size_t max_sg_per_wg = utils::div_up(max_wg_size, conf.subgroup_size);

    // number of subgroups (threads) to saturate the GPU
    const int target_subgroups = eu_count * conf.threads_per_eu;

    const dim_t max_local_size
            = std::min(into<dim_t>(max_sg_per_wg), reduction_block.block);
    dim_t wg_per_inner = utils::div_up(inner_block.block, conf.subgroup_size);
    const dim_t max_num_sg = max_local_size * wg_per_inner * outer_block.block;

    // Atomic accumulation comes with a lot of overhead:
    // 1. Need to initialize data via the copy engine beforehand
    // 2. The atomic accumulation is still O(N) in the worst-case
    // 3. Need to have a finalization kernel afterward for some algs/dts
    // Therefore, only use atomic accumulation if we gain *enough* parallelism
    const int sparsity_threshold = 16;
    if (target_subgroups / max_num_sg > sparsity_threshold) {
        const int target_per_phase
                = into<int>(std::cbrt(reduction_block.block));
        conf.global_acc = target_per_phase;
        conf.local_acc = utils::rnd_up_pow2(target_per_phase);
    } else {
        conf.global_acc = 1;
        conf.local_acc = utils::rnd_up_pow2(
                into<dim_t>(std::sqrt(reduction_block.block)));
    }
    if (conf.local_acc > reduction_block.block) conf.local_acc /= 2;
    conf.local_acc = std::min(conf.local_acc, into<dim_t>(max_sg_per_wg));

    // If using atomic accumulation, mean is performed in separate kernel
    // XXX: dividing before the final accumulation can lead to lower accuracy
    if (conf.global_acc > 1 && alg == reduction_alg_kind_t::mean) {
        alg = secondary_alg = reduction_alg_kind_t::sum;
    }
    conf.alg = alg;
    conf.secondary_alg = secondary_alg;

    // Increase vector size to increase block size, without reducing saturation
    bool is_pre_xe_hp = arch < compute::gpu_arch_t::xe_hp;
    const int max_load_size = is_pre_xe_hp ? 128 : 256;
    conf.vect_size = 1;
    for (auto vec : {8, 4, 2}) {
        const dim_t num_sg = conf.local_acc * conf.global_acc * wg_per_inner
                / vec * outer_block.block;
        // Don't unsaturate
        if (num_sg < target_subgroups) continue;

        // vec * subgroup_size has to divide inner_block.block
        if (inner_block.block % (vec * conf.subgroup_size) != 0) continue;

        // Limit maximum vector size based on max load size
        if (vec * conf.subgroup_size
                        * into<int>(types::data_type_size(src_type))
                > max_load_size) {
            continue;
        }

        // Increasing vec size has the following effects:
        conf.vect_size = vec;
        wg_per_inner /= vec;
        break;
    }

    // Compute the unroll factor to minimize read-accumulate iters
    const auto &compute_unroll = [](dim_t loop_size) {
        const int max_unroll = 128;
        int unroll;
        if (max_unroll > loop_size) {
            // XXX: This encodes the reduction loop size directly into the kernel,
            // which limits reusability for these cases (small reduction sizes) in
            // exchange for fast execution time.
            unroll = into<int>(loop_size);
        } else {
            int min_iters = std::numeric_limits<int>::max();
            for (int u = max_unroll; u > 0; u--) {
                const int unroll_iters = into<int>(loop_size / u);
                const int extra_iters = loop_size % u;
                const int total_iters = unroll_iters + extra_iters;

                if (total_iters < min_iters) {
                    unroll = u;
                    min_iters = total_iters;
                }
            }
        }
        return unroll;
    };
    const int full_loop_size = into<int>(utils::div_up(
            reduction_block.block, conf.global_acc * conf.local_acc));
    const int tail_loop_size = into<int>(
            reduction_block.block / (conf.global_acc * conf.local_acc));
    conf.full_unroll_factor = compute_unroll(full_loop_size);
    conf.tail_unroll_factor = compute_unroll(tail_loop_size);
}

status_t atomic_reduction_conf_t::init_dispatcher(
        const compute::compute_engine_t *engine,
        const gpu_primitive_attr_t *gpu_attr) {
    const std::vector<dim_idx_t> dispatch_dims = {
            reduction_dims::outer,
            reduction_dims::local,
            reduction_dims::global,
            reduction_dims::inner_group,
            reduction_dims::subgroup,
    };
    const std::vector<dim_idx_t> all_dims = {
            reduction_dims::outer,
            reduction_dims::loop,
            reduction_dims::local,
            reduction_dims::global,
            reduction_dims::inner_group,
            reduction_dims::subgroup,
    };
    compute::named_buffer_t src("SRC");
    std::array<dim_t, 6> sizes = {
            outer_block.block,
            conf.global_acc,
            conf.local_acc,
            utils::div_up(reduction_block.block,
                    conf.global_acc * conf.local_acc), // not dispatched
            inner_block.block / conf.vect_size / conf.subgroup_size,
            conf.subgroup_size,
    };
    for (size_t dim_idx = 0; dim_idx < all_dims.size(); dim_idx++) {
        src.append_block(all_dims[dim_idx], sizes[dim_idx]);
    }
    // the loop dim may have padding - update the outer block's stride to avoid it
    dim_idx_t src_outer_idx = src.get_dim_idx(reduction_dims::outer);
    gpu_assert(src_outer_idx != dim_idx::invalid);
    src.format_desc.blocking.strides[src_outer_idx]
            = outer_block.stride / conf.vect_size;

    compute::named_buffer_t dst("DST", src);
    dst.remove_dim(reduction_dims::loop);
    dst.remove_dim(reduction_dims::local); // broadcasted
    dst.remove_dim(reduction_dims::global); // broadcasted

    // Broadcast src's global/local dims, since we index the reduction dims manually
    src.remove_dim(reduction_dims::global, false);
    src.remove_dim(reduction_dims::local, false);

    // Once again, loop dim padding causes issues
    dim_idx_t dst_outer_idx = dst.get_dim_idx(reduction_dims::outer);
    gpu_assert(dst_outer_idx != dim_idx::invalid);
    dst.format_desc.blocking.strides[dst_outer_idx]
            = inner_block.block / conf.vect_size;

    // Create the dispatcher
    compute::reusable_dispatch_config_t config(engine, dispatch_dims);
    CHECK(config.register_buffer(src));
    CHECK(config.register_buffer(dst));
    CHECK(config.define_dim_index(
            "ATOMIC", reduction_dims::global, conf.global_acc));
    CHECK(config.define_dim_index(
            "LOCAL", reduction_dims::local, conf.local_acc));
    CHECK(config.use_subgroup(
            src.get_name(), into<size_t>(conf.subgroup_size)));

    compute::reusable_dispatch_t dispatch;
    atomic_lws_strategy_t lws_strat(engine, gpu_attr);
    lws_strat.include(reduction_dims::local, into<size_t>(conf.local_acc));
    lws_strat.include(
            reduction_dims::subgroup, into<size_t>(conf.subgroup_size));
    CHECK(config.generate(dispatch, lws_strat));
    conf.params = dispatch.get_compile_params();
    rt_conf = dispatch.get_runtime_params();

    return status::success;
}

void atomic_reduction_t::pd_t::init_scratchpad() {
    // Only need scratchpads for the first 2 phases, since we can reuse them
    // and memory requirements are monotonically decreasing each phase.
    const uint32_t keys[2] = {memory_tracking::names::key_reduction,
            memory_tracking::names::key_reduction_1};

    // If we have to use a finalization kernel, we need another scratchpad
    size_t num_phases = phases.size();
    if (needs_finalization) num_phases++;

    auto scratchpad = scratchpad_registry().registrar();
    const size_t num_scratchpads = std::min(num_phases - 1, size_t {2});
    for (size_t i = 0; i < num_scratchpads; i++) {
        const atomic_reduction_conf_t &phase = phases[i];
        const size_t sp_data_size = types::data_type_size(phase.conf.dst_type);
        const size_t num_dst_elems = into<size_t>(
                phase.outer_block.block * phase.inner_block.block);
        scratchpad.book(
                keys[i], num_dst_elems, sp_data_size, OCL_BUFFER_ALIGNMENT);
    }
}

status_t atomic_reduction_t::pd_t::init_conf(impl::engine_t *engine) {
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
            src_mdw.data_type(), data_type::undef);
    for (size_t i = 0; i < subprbs.size(); i++) {
        const bool is_first = (i == 0);
        const bool is_final = (i == subprbs.size() - 1);
        data_type_t src_dt = is_first ? src_mdw.data_type() : accum_data_type;
        data_type_t dst_dt = is_final ? dst_mdw.data_type() : accum_data_type;
        reduction_alg_kind_t alg
                = from_alg(desc()->alg_kind, is_first, is_final);
        reduction_alg_kind_t secondary_alg
                = from_alg(desc()->alg_kind, false, is_final);

        phases.emplace_back(subprbs[i], alg, secondary_alg, src_dt, dst_dt,
                *compute_engine->device_info(), gpu_attr);
        atomic_reduction_conf_t &phase = phases.back();
        if (phase.inner_block.block % phase.conf.subgroup_size != 0) {
            return status::unimplemented;
        }
        CHECK(phase.init_dispatcher(compute_engine, gpu_attr));
    }

    for (atomic_reduction_conf_t &phase : phases) {
        if (phase.conf.global_acc > 1) {
            bool ok = compute_engine->mayiuse(
                    compute::device_ext_t::ext_float_atomics);

            // Due to hardware support and initialization logic, only
            // f32 sum/mean (initialized to 0) and f32 min (initialized to inf)
            // are supported. Better filling logic could enable f16 atomic operations.
            ok = ok && phase.conf.dst_type == data_type::f32
                    && utils::one_of(phase.conf.alg, reduction_alg_kind_t::mean,
                            reduction_alg_kind_t::sum,
                            reduction_alg_kind_t::min);
            if (!ok) return status::unimplemented;
        }
    }

    // Compute div from basic mdw dims
    div = 1;
    for (int i = 0; i < src_mdw.ndims(); i++) {
        if (is_reduction_dim[i]) div *= src_dims[i];
    }

    power = desc()->p;
    eps = desc()->eps;

    // If the final kernel uses atomic accumulation, we need a finalization kernel
    bool alg_needs_finalization = utils::one_of(desc()->alg_kind,
            reduction_mean, reduction_norm_lp_max, reduction_norm_lp_sum,
            reduction_norm_lp_power_p_max, reduction_norm_lp_power_p_sum);
    if (alg_needs_finalization && phases.back().conf.global_acc > 1) {
        needs_finalization = true;
        CHECK(init_finalization_pd(engine));
    } else {
        needs_finalization = false;
    }

    return status::success;
}

status_t atomic_reduction_t::pd_t::init_finalization_pd(
        impl::engine_t *engine) {
    eltwise_desc_t eltwise_desc;
    memory_desc_t eltwise_mem_desc(*dst_md());
    // XXX: Just for mean currently
    if (desc()->alg_kind != alg_kind::reduction_mean) {
        return status::unimplemented;
    }
    CHECK(eltwise_desc_init(&eltwise_desc, prop_kind_t::dnnl_forward,
            alg_kind_t::dnnl_eltwise_linear, &eltwise_mem_desc,
            &eltwise_mem_desc, nullptr, nullptr, 1.0f / static_cast<float>(div),
            0));

    primitive_attr_t eltwise_attr(*attr());
    if (!eltwise_attr.is_initialized()) return status::out_of_memory;
    primitive_desc_iterator_t it(engine,
            reinterpret_cast<op_desc_t *>(&eltwise_desc), &eltwise_attr,
            nullptr);
    if (!it.is_initialized()) return status::invalid_arguments;
    eltwise_pd_ = *(++it);

    return eltwise_pd_ ? status::success : status::invalid_arguments;
}

static void init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const atomic_reduction_key_params_t &conf) {
    using namespace alg_kind;

    kernel_ctx.set_data_type(conf.src_type);
    def_data_type(kernel_ctx, conf.src_type, "SRC");
    def_data_type(kernel_ctx, conf.dst_type, "DST");

    conf.params.def_kernel_macros(kernel_ctx);

    // All of the variables needed to compute strides
    kernel_ctx.define_int("LOCAL_SIZE", conf.local_acc);
    kernel_ctx.define_int("ATOMIC_REDUCTION_SIZE", conf.global_acc);
    // End stride vars

    kernel_ctx.define_int("FULL_UNROLL_FACTOR", conf.full_unroll_factor);
    kernel_ctx.define_int("TAIL_UNROLL_FACTOR", conf.tail_unroll_factor);

    // To use atomic_global_add
    kernel_ctx.add_option("-cl-std=CL2.0");

    kernel_ctx.define_int("VECT_DT_N", conf.vect_size);

    def_reduction_alg_kinds(kernel_ctx);
    kernel_ctx.define_int("REDUCTION_ALG", to_int(conf.alg));
    kernel_ctx.define_int(
            "SECONDARY_REDUCTION_ALG", to_int(conf.secondary_alg));
}

status_t atomic_reduction_key_params_t::get_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    primitive_attr_t ocl_attr;
    CHECK(ocl_attr.set_gpu_attr(gpu_primitive_attr_t(threads_per_eu)));
    kernel_ctx = compute::kernel_ctx_t(&ocl_attr);

    init_kernel_ctx_common(kernel_ctx, *this);
    return status::success;
}

status_t atomic_reduction_t::execute_atomic(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    std::unique_ptr<memory_storage_t> sp_reduce[2]
            = {ctx.get_scratchpad_grantor().get_memory_storage(
                       memory_tracking::names::key_reduction),
                    ctx.get_scratchpad_grantor().get_memory_storage(
                            memory_tracking::names::key_reduction_1)};

    const size_t num_phases = pd()->phases.size();
    memory_storage_t &final_mem = (pd()->needs_finalization)
            ? *sp_reduce[(num_phases - 1) % 2]
            : dst;

    for (size_t i = 0; i < kernels_.size(); i++) {
        auto &kernel = kernels_[i];
        auto &phase = pd()->phases[i];
        auto &nd_range = phase.rt_conf.nd_range;

        // Set up the reduction arg list
        compute::kernel_arg_list_t reduction_arg_list;

        memory_storage_t &src_mem = (i == 0) ? src : *sp_reduce[(i - 1) % 2];
        memory_storage_t &dst_mem
                = (i == kernels_.size() - 1) ? final_mem : *sp_reduce[i % 2];

        // Initialize dst if we're using atomic (global) accumulation
        if (phase.conf.global_acc > 1) {
            // min -> fill with inf (11111111), otherwise sum/mean fill with 0
            uint8_t pattern
                    = phase.conf.alg == reduction_alg_kind_t::min ? 255 : 0;
            const size_t dst_data_size
                    = types::data_type_size(phase.conf.dst_type);
            const size_t num_dst_elems = into<size_t>(
                    phase.outer_block.block * phase.inner_block.block);
            size_t dst_size = num_dst_elems * dst_data_size;
            compute::compute_stream_t *compute_stream
                    = utils::downcast<compute::compute_stream_t *>(
                            ctx.stream());
            CHECK(compute_stream->fill(dst_mem, pattern, dst_size,
                    compute_stream->ctx().get_deps(),
                    compute_stream->ctx().get_deps()));
        }

        reduction_arg_list.set(0, src_mem);
        reduction_arg_list.set(1, dst_mem);
        reduction_arg_list.append(into<int>(phase.inner_block.block));
        reduction_arg_list.append(pd()->div);
        reduction_arg_list.append(pd()->power);
        reduction_arg_list.append(pd()->eps);
        reduction_arg_list.append(into<int>(phase.reduction_block.block));
        reduction_arg_list.append(phase.rt_conf.get());

        CHECK(parallel_for(ctx, nd_range, kernel, reduction_arg_list));
    }

    // Run a finalization kernel if needed
    if (pd()->needs_finalization) {
        exec_args_t eltwise_args;
        std::unique_ptr<memory_t, memory_deleter_t> eltwise_src;
        CHECK(safe_ptr_assign(eltwise_src,
                new memory_t(ctx.stream()->engine(), pd()->dst_md(0),
                        std::move(sp_reduce[(num_phases - 1) % 2]))));
        eltwise_args[DNNL_ARG_SRC] = memory_arg_t {eltwise_src.get(), true};
        eltwise_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
        exec_ctx_t eltwise_ctx(ctx, std::move(eltwise_args));

        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested, eltwise_p_);
        eltwise_ctx.set_scratchpad_grantor(ns.grantor());

        CHECK(eltwise_p_->execute(eltwise_ctx));
    }

    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
