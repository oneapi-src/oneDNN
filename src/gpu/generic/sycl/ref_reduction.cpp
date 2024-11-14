#include "ref_reduction.hpp"

#include "gpu/generic/sycl/engine.hpp"
#include "gpu/generic/sycl/reduction_kernels.hpp"

#include <numeric>

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

inline int round_up_to_nearest_multiple(int val, int multiplier) {
    const int diff = val % multiplier;
    if (diff > 0) { val += (multiplier - diff); }
    return val;
}

static int get_max_col_tile_dim(int col_lim, int max_wg_size, int max_sg_size) {
    auto const max_tile_col = max_wg_size / max_sg_size;
    return std::min(col_lim, max_tile_col);
}

static int get_max_row_tile_dim(
        int row_lim, int tile_col, int max_wg_size, int max_sg_size) {
    const auto max_row_tile = max_wg_size / tile_col;
    const auto ideal_row_tile
            = round_up_to_nearest_multiple(row_lim, max_sg_size);
    auto ub_row_tile = round_up_to_nearest_multiple(max_row_tile, max_sg_size);
    ub_row_tile = ub_row_tile > max_row_tile ? ub_row_tile - max_sg_size
                                             : ub_row_tile;
    return std::min(ideal_row_tile, ub_row_tile);
}

size_t ref_reduction_t::pd_t::compute_workspace_size(
        const std::vector<int> &dims, const std::vector<int> &axes,
        int reduce_size) {
    if (axes.size() == 1 || reduce_size == 1) { return 0; }

    auto out_sizes = get_first_two_out_sizes(dims, axes);
    return std::accumulate(
            out_sizes.begin(), out_sizes.end(), 1, std::multiplies<size_t>());
}

status_t ref_reduction_t::pd_t::init_scratchpad() {
    dim_t dims1[] = {out_size_vec_[0]};
    dim_t dims2[] = {out_size_vec_[1]};
    memory_desc_init_by_tag(
            scratch_md_1_, 1, dims1, data_type::f32, format_tag_t::dnnl_a);
    memory_desc_init_by_tag(
            scratch_md_2_, 1, dims2, data_type::f32, format_tag_t::dnnl_a);

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_reduction, out_size_vec_[0],
            types::data_type_size(data_type::f32));
    scratchpad.book(memory_tracking::names::key_reduction_1, out_size_vec_[1],
            types::data_type_size(data_type::f32));

    return status::success;
}

status_t ref_reduction_t::pd_t::init_out_scratchpad() {
    memory_desc_wrapper dst_wrap(dst_md());
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_reduction_out,
            dst_wrap.nelems(), types::data_type_size(data_type::f32));

    return status::success;
}

status_t ref_reduction_t::pd_t::init_reorder(impl::engine_t *engine) {
    reorder_src_md_ = *dst_md();
    reorder_src_md_.data_type = data_type::f32;
    CHECK(reorder_primitive_desc_create(
            reorder_pd_, engine, &reorder_src_md_, dst_md()));

    if (!reorder_pd_) { return status::invalid_arguments; }

    return status::success;
}

reduction_sizes_t ref_reduction_t::pd_t::get_reduction_sizes(
        const sycl_reduction_conf_t &conf) {
    size_t input_size = 1;
    for (size_t i = 0; i < xpu::sycl::md_t::max_dims; i++) {
        if (conf.src_dims[i] == -1) break;
        input_size *= conf.src_dims[i];
    }
    size_t reduction_size = 1;
    for (size_t i = 0; i < xpu::sycl::md_t::max_dims; i++) {
        if (conf.src_dims[i] == -1) break;
        reduction_size *= conf.src_dims[conf.axes[i]];
    }
    const auto output_size = input_size / reduction_size;
    return {input_size, reduction_size, output_size};
}

void ref_reduction_t::pd_t::squeeze_dims_and_axes(
        const memory_desc_wrapper &src_wrap, const std::vector<bool> &axes_mask,
        std::vector<int> &squeezed_dims, std::vector<int> &squeezed_axis) {
    const auto &dims = src_wrap.dims();
    int new_axis = 0;
    for (int i = 0; i < src_wrap.ndims(); i++) {
        int jump = i;
        int new_dim = dims[i];
        if (dims[i] == 1) { continue; }
        while (axes_mask[i] && jump + 1 < src_wrap.ndims()
                && (axes_mask[jump + 1] || dims[jump + 1] == 1)) {
            new_dim *= dims[jump + 1];
            ++jump;
        }
        if (axes_mask[i]) { squeezed_axis.push_back(new_axis); }
        i = jump;
        squeezed_dims.push_back(new_dim);
        new_axis++;
    }
}

std::vector<int> ref_reduction_t::pd_t::get_first_two_out_sizes(
        const std::vector<int> &dims, const std::vector<int> &axes) {
    std::vector<int> result {};
    auto sorted_axes = axes;
    std::sort(sorted_axes.begin(), sorted_axes.end(), std::greater<size_t>());
    size_t total_size = std::accumulate(
            dims.begin(), dims.end(), 1, std::multiplies<int>());

    auto const size = std::min(sorted_axes.size(), 2UL);
    for (size_t i = 0; i < size; i++) {
        total_size /= dims[sorted_axes[i]];
        result.push_back(total_size);
    }
    return result;
}

status_t ref_reduction_t::pd_t::init_conf(impl::engine_t *engine) {
    auto *sycl_engine = utils::downcast<const impl::xpu::sycl::engine_impl_t *>(
            engine->impl());
    const ::sycl::device &sycl_device = sycl_engine->device();
    bool supports_subgroup
            = (sycl_device.get_info<::sycl::info::device::max_num_sub_groups>()
                    > 0);
    if (!supports_subgroup) return status::unimplemented;

    const size_t max_work_group_size = std::min<size_t>(256,
            sycl_device.get_info<::sycl::info::device::max_work_group_size>());

    const auto max_work_item_sizes
            = sycl_device
                      .get_info<::sycl::info::device::max_work_item_sizes<3>>();
    const auto max_wg_size
            = std::min(max_work_item_sizes[2], max_work_group_size);

#if defined(DNNL_SYCL_CUDA) || defined(DNNL_SYCL_HIP)
    const auto max_sg_size = 32;
#else
    const auto subgroup_sizes
            = sycl_device.get_info<::sycl::info::device::sub_group_sizes>();
    const auto max_sg_size
            = *std::max_element(subgroup_sizes.begin(), subgroup_sizes.end());
#endif

    bool supports_atomics = false;
    for (const auto &cap :
            sycl_device.get_info<
                    ::sycl::info::device::atomic_memory_scope_capabilities>()) {
        if (cap == ::sycl::memory_scope::work_group) {
            supports_atomics = true;
            break;
        }
    }

    max_wg_size_ = max_wg_size;
    max_sg_size_ = max_sg_size;

    sycl_reduction_conf_t init_conf;
    init_conf.alg = desc()->alg_kind;
    init_conf.p = desc()->p;
    init_conf.eps = desc()->eps;
    init_conf.src_md = xpu::sycl::md_t(src_md());
    init_conf.dst_md = xpu::sycl::md_t(dst_md());
    init_conf.post_ops = sycl_post_ops_t(attr(), dst_md());

    memory_desc_wrapper src_wrap(src_md());
    memory_desc_wrapper dst_wrap(dst_md());
    init_conf.src_dt = src_wrap.data_type();
    init_conf.dst_dt = dst_wrap.data_type();

    for (int i = 0; i < xpu::sycl::md_t::max_dims; ++i) {
        init_conf.src_dims[i] = -1;
        init_conf.axes[i] = -1;
    }

    std::vector<bool> axes_mask(src_wrap.ndims());
    int arr_idx = 0;
    for (int i = 0; i < src_wrap.ndims(); ++i) {
        init_conf.src_dims[i] = src_wrap.dims()[i];

        if (src_wrap.dims()[i] != 1 && dst_wrap.dims()[i] == 1) {
            init_conf.axes[arr_idx] = i;
            axes_mask[i] = true;
            arr_idx++;
        }
    }
    init_conf.num_dims = src_wrap.ndims();
    init_conf.num_axes = arr_idx;

    std::vector<int> new_dims;
    std::vector<int> new_axes;
    squeeze_dims_and_axes(src_wrap, axes_mask, new_dims, new_axes);
    std::sort(new_axes.begin(), new_axes.end(), std::greater<int>());

    squeezed_dims_ = new_dims;
    squeezed_axes_ = new_axes;

    auto num_dims = new_dims.size();
    num_reductions_ = new_axes.size();
    out_size_vec_ = get_first_two_out_sizes(new_dims, new_axes);

    if (num_reductions_ == 1) {
        auto const dims_begin = new_dims.begin();
        auto const dims_end = new_dims.end();
        auto const axis = new_axes[0];

        // Setup internal params
        init_conf.batch_size = std::accumulate(
                dims_begin, dims_begin + axis, 1, std::multiplies<int>());
        init_conf.reduce_size = new_dims[axis];
        init_conf.stride_size = std::accumulate(
                dims_begin + axis + 1, dims_end, 1, std::multiplies<int>());

    } else if ((num_dims - num_reductions_) == 1) {
        assert(num_dims == 3 && num_reductions_ == 2);
        assert(new_axes[1] == 0 && new_axes[0] == 2);

        init_conf.batch_size = new_dims[0] * new_dims[1];
        init_conf.reduce_size = new_dims[2];
        init_conf.stride_size = 1;
        init_conf.batch_groups = new_dims[1];
        multi_reduction_ = false;
        num_reductions_ = 1;
    } else {
        multi_reduction_ = true;
        CHECK(init_scratchpad());
    }

    if (init_conf.stride_size == 1) {
        init_conf.transpose = false;
        init_conf.bank_offset = false;
    } else if (init_conf.stride_size > 4) {
        init_conf.transpose = true;
        init_conf.bank_offset = true;
    } else {
        init_conf.transpose = false;
        init_conf.bank_offset = true;
    }

    auto dims = squeezed_dims_;
    for (size_t red_iter = 0; red_iter < num_reductions_; ++red_iter) {
        auto conf = init_conf;
        const auto &axes = squeezed_axes_;
        auto dims_begin = dims.begin();
        auto dims_end = dims.end();
        auto axis = axes[red_iter];
        conf.is_first_iter = (red_iter == 0);
        conf.is_last_iter = (red_iter == num_reductions_ - 1);
        conf.batch_size = std::accumulate(
                dims_begin, dims_begin + axis, 1, std::multiplies<int>());
        conf.reduce_size = dims[axis];
        if (axis < static_cast<int>(dims.size() - 1)) {
            conf.stride_size = std::accumulate(
                    dims_begin + axis + 1, dims_end, 1, std::multiplies<int>());
        } else {
            conf.stride_size = 1;
        }

        needs_atomic_reduction_ = conf.batch_groups != -1;
        const auto batch_groups
                = conf.batch_groups == -1 ? conf.batch_size : conf.batch_groups;
        const auto max_wg_size = max_wg_size_;
        const auto max_sg_size = max_sg_size_;
        int tile_col = get_max_col_tile_dim(
                conf.stride_size, max_wg_size, max_sg_size);
        tile_col = std::min(tile_col, sycl_reduction_conf_t::local_col_wg);
        int tile_row = get_max_row_tile_dim(
                conf.reduce_size, tile_col, max_wg_size, max_sg_size);
        tile_row = std::min(tile_row, sycl_reduction_conf_t::local_row_wg);
        conf.tile_col = tile_col;
        conf.tile_row = tile_row;
        local_ranges_.emplace_back(range_t {1, tile_row, tile_col});

        auto global_col
                = round_up_to_nearest_multiple(conf.stride_size, tile_col);
        auto global_row
                = round_up_to_nearest_multiple(conf.reduce_size, tile_row);
        global_ranges_.emplace_back(
                range_t {conf.batch_size, global_row, global_col});
        needs_atomic_reduction_ = needs_atomic_reduction_
                || (global_row > std::min(
                            tile_row, conf.num_sg_reductions * max_sg_size));

        VDISPATCH_REDUCTION(
                IMPLICATION(needs_atomic_reduction_, supports_atomics),
                "Implementation needs to perform atomic reduction, but atomics "
                "are not supported by current device");
        VDISPATCH_REDUCTION(
                IMPLICATION(needs_atomic_reduction_, !attr()->deterministic_),
                "Atomic reduction is only supported in non-deterministic mode");
        VDISPATCH_REDUCTION(IMPLICATION(needs_atomic_reduction_,
                                    conf.alg != alg_kind::reduction_mul),
                "Algorithm Mul is not supported with atomic reduction");
        VDISPATCH_REDUCTION(IMPLICATION(needs_atomic_reduction_,
                                    attr()->post_ops_.find(dnnl_sum) == -1),
                "Sum postop is not supported with atomic reduction");

        const size_t dt_size = data_type_size(data_type::f32);
        local_mem_sizes_.push_back(
                ((tile_row + conf.bank_offset) * (tile_col + conf.bank_offset))
                * dt_size);

        needs_reorder_ = needs_atomic_reduction_
                && dst_wrap.data_type() != data_type::f32;
        if (needs_reorder_) { conf.dst_dt = data_type::f32; }

        if (multi_reduction_) {
            if (red_iter != 0) { conf.src_dt = data_type::f32; }
            if (red_iter != num_reductions_ - 1) {
                conf.dst_dt = data_type::f32;
            }
        }

        conf.batch_groups = batch_groups;
        confs_.push_back(conf);
        dims[axes[red_iter]] = 1;
    }

    if (needs_reorder_) { CHECK(init_reorder(engine)); }
    if (needs_atomic_reduction_) { CHECK(init_out_scratchpad()); }

    return status::success;
}

status_t ref_reduction_t::init(impl::engine_t *engine) {
    const auto reduction_kid = ::sycl::get_kernel_id<reduction_kernel_fwd_t>();
    CHECK(create_kernel(engine, reduction_kid, &kernel_));

    if (pd()->needs_atomic_reduction_) {
        const auto init_kid = ::sycl::get_kernel_id<init_kernel_t>();
        const auto finalize_kid
                = ::sycl::get_kernel_id<atomic_finalize_kernel_t>();
        CHECK(create_kernel(engine, init_kid, &init_kernel_));
        CHECK(create_kernel(engine, finalize_kid, &finalize_kernel_));
    }

    if (pd()->needs_reorder_) {
        CHECK(pd()->reorder_pd_->create_primitive(reorder_p_, engine));
    }

    return status::success;
}

status_t ref_reduction_t::execute(const exec_ctx_t &ctx) const {
    auto dst_wrap = memory_desc_wrapper(pd()->dst_md());
    auto scratch_wrap = memory_desc_wrapper(pd()->reorder_src_md_);
    const bool needs_atomic_reduction = pd()->needs_atomic_reduction_;
    const bool needs_reorder = pd()->needs_reorder_;
    for (size_t i = 0; i < pd()->num_reductions_; ++i) {
        const auto &conf = pd()->confs_[i];

        if (needs_reorder
                && ((pd()->multi_reduction_ && pd()->num_reductions_ - 1 == i)
                        || i == 0)) {
            CHECK(parallel_for(ctx, init_kernel_, [&](::sycl::handler &cgh) {
                auto out = CTX_OUT_SCRATCH_KERNEL_MEMORY(key_reduction_out);
                init_kernel_t kernel(out, pd()->desc()->alg_kind);
                cgh.parallel_for(
                        ::sycl::range<1>(scratch_wrap.nelems()), kernel);
            }));
        }

        if (!needs_reorder
                && (needs_atomic_reduction
                        && ((pd()->multi_reduction_
                                    && pd()->num_reductions_ - 1 == i)
                                || i == 0))) {
            CHECK(parallel_for(ctx, init_kernel_, [&](::sycl::handler &cgh) {
                auto out = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
                init_kernel_t kernel(out, pd()->desc()->alg_kind);
                cgh.parallel_for(::sycl::range<1>(dst_wrap.nelems()), kernel);
            }));
        }

        const size_t local_mem_size_bytes = pd()->local_mem_sizes_[i];
        const auto &global_range = pd()->global_ranges_[i];
        const auto &local_range = pd()->local_ranges_[i];
        CHECK(parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
            auto src_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
            auto dst_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);

            auto temp_arg1_in
                    = xpu::sycl::memory_storage_base_t::empty_in_memory_arg(
                            ctx.stream(), cgh);
            auto temp_arg1_out
                    = xpu::sycl::memory_storage_base_t::empty_out_memory_arg(
                            ctx.stream(), cgh);
            auto temp_arg2_in
                    = xpu::sycl::memory_storage_base_t::empty_in_memory_arg(
                            ctx.stream(), cgh);
            auto temp_arg2_out
                    = xpu::sycl::memory_storage_base_t::empty_out_memory_arg(
                            ctx.stream(), cgh);
            auto out_scratch
                    = xpu::sycl::memory_storage_base_t::empty_out_memory_arg(
                            ctx.stream(), cgh);

            if (pd()->multi_reduction_) {
                temp_arg1_in = CTX_IN_SCRATCH_KERNEL_MEMORY(key_reduction);
                temp_arg1_out = CTX_OUT_SCRATCH_KERNEL_MEMORY(key_reduction);
                temp_arg2_in = CTX_IN_SCRATCH_KERNEL_MEMORY(key_reduction_1);
                temp_arg2_out = CTX_OUT_SCRATCH_KERNEL_MEMORY(key_reduction_1);
            }
            if (pd()->needs_reorder_) {
                out_scratch = CTX_OUT_SCRATCH_KERNEL_MEMORY(key_reduction_out);
            }

            auto local_mem = ::sycl::local_accessor<uint8_t, 1>(
                    ::sycl::range<1>(local_mem_size_bytes), cgh);

            auto src = i == 0    ? src_arg
                    : i % 2 != 0 ? temp_arg1_in
                                 : temp_arg2_in;
            auto dst = i == pd()->num_reductions_ - 1
                    ? (pd()->needs_reorder_ ? out_scratch : dst_arg)
                    : i % 2 != 0 ? temp_arg2_out
                                 : temp_arg1_out;

            reduction_kernel_fwd_t reduction_kernel(src, dst, conf,
                    needs_atomic_reduction, local_mem, cgh, ctx);
            ::sycl::nd_range<3> range(::sycl::range<3>(global_range.x,
                                              global_range.y, global_range.z),
                    ::sycl::range<3>(
                            local_range.x, local_range.y, local_range.z));
            cgh.parallel_for(range, reduction_kernel);
        }));
    }

    const auto &conf = pd()->confs_[0];
    const auto alg = conf.alg;
    sycl_post_ops_t post_ops = sycl_post_ops_t(
            pd()->attr(), memory_desc_wrapper(pd()->dst_md()));
    if (needs_atomic_reduction
            && (utils::one_of(alg, alg_kind::reduction_norm_lp_max,
                        alg_kind::reduction_norm_lp_sum,
                        alg_kind::reduction_norm_lp_power_p_max,
                        alg_kind::reduction_norm_lp_power_p_sum,
                        alg_kind::reduction_mean)
                    || pd()->attr()->post_ops_.len() != 0)) {
        float full_reduce_size = 1.f;
        if (alg == alg_kind::reduction_mean) {
            for (auto &axis : pd()->squeezed_axes_) {
                full_reduce_size *= pd()->squeezed_dims_[axis];
            }
        }

        CHECK(parallel_for(ctx, finalize_kernel_, [&](::sycl::handler &cgh) {
            auto out = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
            auto out_scratch = CTX_OUT_SCRATCH_KERNEL_MEMORY(key_reduction_out);
            atomic_finalize_kernel_t kernel(cgh, ctx,
                    (pd()->needs_reorder_ ? conf.local_mem_dt
                                          : pd()->dst_md()->data_type),
                    (pd()->needs_reorder_ ? out_scratch : out), alg, conf.p,
                    conf.eps, post_ops, conf.dst_md, full_reduce_size);
            cgh.parallel_for(::sycl::range<1>(dst_wrap.nelems()), kernel);
        }));
    }

    if (!needs_reorder) { return status::success; }

    std::unique_ptr<memory_t, memory_deleter_t> scratch_mem;
    auto scratchpad_storage = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_reduction_out);
    CHECK(safe_ptr_assign(scratch_mem,
            new memory_t(ctx.stream()->engine(), &pd()->reorder_src_md_,
                    std::move(scratchpad_storage))));

    exec_args_t reorder_args;
    reorder_args[DNNL_ARG_SRC] = memory_arg_t {scratch_mem.get(), true};
    reorder_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
    exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));

    return reorder_p_->execute(reorder_ctx);
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
