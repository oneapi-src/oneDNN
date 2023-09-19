/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/ocl/atomic_reduction.hpp"
#include "common/eltwise_pd.hpp"
#include "common/scratchpad.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

atomic_reduction_conf_t::atomic_reduction_conf_t(
        const reduction_subproblem_t &subprb, data_type_t src_type,
        data_type_t dst_type, bool is_first, bool is_final,
        const compute::device_info_t &device_info, bool large_grf_mode)
    : reduction_subproblem_t(subprb)
    , src_type(src_type)
    , dst_type(dst_type)
    , is_first(is_first)
    , is_final(is_final)
    , subgroup_size(device_info.max_subgroup_size()) {
    auto arch = device_info.gpu_arch();
    const int threads_per_eu = compute::device_info_t::threads_per_eu(arch);
    const size_t max_wg_size = device_info.max_wg_size(large_grf_mode);
    const int eu_count = device_info.eu_count();
    const size_t max_sg_per_wg = utils::div_up(max_wg_size, subgroup_size);

    // number of subgroups (threads) to saturate the GPU
    const int target_subgroups = eu_count * threads_per_eu;

    const dim_t max_local_size = std::min(
            static_cast<dim_t>(max_sg_per_wg), reduction_block.block);
    dim_t wg_per_inner = utils::div_up(inner_block.block, subgroup_size);
    const dim_t max_num_sg = max_local_size * wg_per_inner * outer_block.block;

    // Atomic accumulation comes with a lot of overhead:
    // 1. Need to initialize data via the copy engine beforehand
    // 2. The atomic accumulation is still O(N) in the worst-case
    // 3. Need to have a finalization kernel afterward for some algs/dts
    // Therefore, only use atomic accumulation if we gain *enough* parallelism
    const int sparsity_threshold = 16;
    dim_t local_size;
    if (target_subgroups / max_num_sg > sparsity_threshold) {
        const int target_per_phase
                = static_cast<int>(std::cbrt(reduction_block.block));
        global_acc = target_per_phase;
        local_size = utils::rnd_up_pow2(target_per_phase);
    } else {
        global_acc = 1;
        local_size = utils::rnd_up_pow2(
                static_cast<dim_t>(std::sqrt(reduction_block.block)));
    }
    if (local_size > reduction_block.block) local_size /= 2;
    local_size = std::min(local_size, static_cast<dim_t>(max_sg_per_wg));

    // Increase vector size to increase block size, without reducing saturation
    bool is_pre_xe_hp = arch < compute::gpu_arch_t::xe_hp;
    const int max_load_size = is_pre_xe_hp ? 128 : 256;
    vect_size = 1;
    for (auto vec : {8, 4, 2}) {
        const dim_t num_sg = local_size * global_acc * wg_per_inner / vec
                * outer_block.block;
        // Don't unsaturate
        if (num_sg < target_subgroups) continue;

        // vec * subgroup_size has to divide inner_block.block
        if (inner_block.block % (vec * subgroup_size) != 0) continue;

        // Limit maximum vector size based on max load size
        if (vec * subgroup_size
                        * static_cast<int>(types::data_type_size(src_type))
                > max_load_size) {
            continue;
        }

        // Increasing vec size has the following effects:
        vect_size = vec;
        wg_per_inner /= vec;
        break;
    }

    const size_t sgs = static_cast<size_t>(subgroup_size);
    const size_t wgpi = static_cast<size_t>(wg_per_inner);
    const size_t tpw = static_cast<size_t>(local_size);
    const size_t nglobal = static_cast<size_t>(global_acc);

    size_t lws[3] = {sgs, 1, tpw};
    size_t gws[3] = {
            sgs, wgpi * static_cast<size_t>(outer_block.block), nglobal * tpw};

    nd_range = compute::nd_range_t(gws, lws);
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
        const size_t sp_data_size = types::data_type_size(phase.dst_type);
        const size_t num_dst_elems = static_cast<size_t>(
                phase.outer_block.block * phase.inner_block.block);
        scratchpad.book(
                keys[i], num_dst_elems, sp_data_size, OCL_BUFFER_ALIGNMENT);
    }
}

status_t atomic_reduction_t::pd_t::init_conf(engine_t *engine) {
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

    std::vector<reduction_subproblem_t> subprbs;
    CHECK(generate_reduction_phases(src_md(), dst_md(), subprbs));

    //DST zero-padding not supported on reduction dims
    reduction_subproblem_t &last_subprb = subprbs.back();
    for (const auto &zpad : last_subprb.dst_zpads) {
        if (conf.is_reduction_dim[zpad.dim_idx]) {
            return status::unimplemented;
        }
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
        if (alg_affected_by_zeros && conf.is_reduction_dim[zpad.dim_idx]) {
            return status::unimplemented;
        }
    }

    const compute::compute_engine_t *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);
    auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    bool large_grf_mode = gpu_attr && gpu_attr->threads_per_eu() == 4;

    data_type_t accum_data_type = types::default_accum_data_type(
            src_mdw.data_type(), data_type::undef);
    for (size_t i = 0; i < subprbs.size(); i++) {
        const bool is_first = (i == 0);
        const bool is_final = (i == subprbs.size() - 1);
        data_type_t src_dt = is_first ? src_mdw.data_type() : accum_data_type;
        data_type_t dst_dt = is_final ? dst_mdw.data_type() : accum_data_type;

        phases.emplace_back(subprbs[i], src_dt, dst_dt, is_first, is_final,
                *compute_engine->device_info(), large_grf_mode);
        atomic_reduction_conf_t &phase = phases.back();
        if (phase.inner_block.block % phase.subgroup_size != 0) {
            return status::unimplemented;
        }
    }

    for (atomic_reduction_conf_t &phase : phases) {
        if (phase.global_acc > 1) {
            bool ok = compute_engine->mayiuse(
                    compute::device_ext_t::ext_float_atomics);

            // Due to hardware support and initialization logic, only
            // f32 sum/mean (initialized to 0) and f32 min (initialized to inf)
            // are supported. Better filling logic could enable f16 atomic operations.
            ok = ok && phase.dst_type == data_type::f32
                    && utils::one_of(desc()->alg_kind, reduction_mean,
                            reduction_sum, reduction_min);
            if (!ok) return status::unimplemented;
        }
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

    // If the final kernel uses atomic accumulation, we need a finalization kernel
    bool alg_needs_finalization = utils::one_of(conf.alg, reduction_mean,
            reduction_norm_lp_max, reduction_norm_lp_sum,
            reduction_norm_lp_power_p_max, reduction_norm_lp_power_p_sum);
    if (alg_needs_finalization && phases.back().global_acc > 1) {
        needs_finalization = true;
        CHECK(init_finalization_pd(engine));
    } else {
        needs_finalization = false;
    }

    return status::success;
}

status_t atomic_reduction_t::pd_t::init_finalization_pd(engine_t *engine) {
    eltwise_desc_t eltwise_desc;
    memory_desc_t eltwise_mem_desc(*dst_md());
    // XXX: Just for mean currently
    if (conf.alg != alg_kind::reduction_mean) return status::unimplemented;
    CHECK(eltwise_desc_init(&eltwise_desc, prop_kind_t::dnnl_forward,
            alg_kind_t::dnnl_eltwise_linear, &eltwise_mem_desc,
            &eltwise_mem_desc, nullptr, nullptr,
            1.0f / static_cast<float>(conf.div), 0));

    primitive_attr_t eltwise_attr(*attr());
    if (!eltwise_attr.is_initialized()) return status::out_of_memory;
    primitive_desc_iterator_t it(engine,
            reinterpret_cast<op_desc_t *>(&eltwise_desc), &eltwise_attr,
            nullptr);
    if (!it.is_initialized()) return status::invalid_arguments;
    eltwise_pd_ = *(++it);

    return eltwise_pd_ ? status::success : status::invalid_arguments;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const reduction_conf_t &conf, const atomic_reduction_conf_t &phase) {
    using namespace alg_kind;

    kernel_ctx.set_data_type(phase.src_type);

    // All of the variables needed to compute strides
    kernel_ctx.define_int("SUBGROUP_SIZE", phase.subgroup_size);
    kernel_ctx.define_int("LOCAL_SIZE",
            static_cast<int64_t>(phase.nd_range.local_range()[2]));
    kernel_ctx.define_int("REDUCTION_SIZE", phase.reduction_block.block);
    kernel_ctx.define_int("INNER_DIM_SIZE", phase.inner_block.block);
    kernel_ctx.define_int("ATOMIC_REDUCTION_SIZE", phase.global_acc);
    // End stride vars

    // To use atomic_global_add
    kernel_ctx.add_option("-cl-std=CL2.0");

    kernel_ctx.define_int("DIV", conf.div);
    kernel_ctx.define_float("POWER", conf.power);
    kernel_ctx.define_float("EPS", conf.eps);

    kernel_ctx.define_int("IS_FINAL", phase.is_final);
    kernel_ctx.define_int("IS_FIRST", phase.is_first);

    kernel_ctx.define_int("VECT_DT_N", phase.vect_size);

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

status_t atomic_reduction_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx,
        const atomic_reduction_conf_t &phase) const {
    CHECK(init_kernel_ctx_common(kernel_ctx, conf, phase));
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
        auto &nd_range = phase.nd_range;

        // Set up the reduction arg list
        compute::kernel_arg_list_t reduction_arg_list;

        memory_storage_t &src_mem = (i == 0) ? src : *sp_reduce[(i - 1) % 2];
        memory_storage_t &dst_mem
                = (i == kernels_.size() - 1) ? final_mem : *sp_reduce[i % 2];

        // Initialize dst if we're using atomic (global) accumulation
        if (phase.global_acc > 1) {
            // min -> fill with inf (11111111), otherwise sum/mean fill with 0
            uint8_t pattern
                    = pd()->conf.alg == alg_kind::reduction_min ? 255 : 0;
            const size_t dst_data_size = types::data_type_size(phase.dst_type);
            const size_t num_dst_elems = static_cast<size_t>(
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

        CHECK(parallel_for(ctx, nd_range, kernel, reduction_arg_list));
    }

    // Run a finalization kernel if needed
    if (pd()->needs_finalization) {
        exec_args_t eltwise_args;
        std::unique_ptr<memory_t> eltwise_src;
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
} // namespace gpu
} // namespace impl
} // namespace dnnl
