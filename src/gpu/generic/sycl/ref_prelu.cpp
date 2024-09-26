/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/generic/sycl/ref_prelu.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/prelu_kernels.hpp"
#include "xpu/sycl/stream_impl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_prelu_fwd_t::pd_t::init_conf() {
    if (has_zero_dim_memory()) return status::success;

    conf_ = sycl_prelu_conf_t();

    const memory_desc_wrapper data_d(src_md(0));
    const memory_desc_wrapper weights_d(weights_md(0));
    conf_.data_md = xpu::sycl::md_t(src_md(0));
    conf_.weights_md = xpu::sycl::md_t(weights_md(0));
    conf_.dst_md = xpu::sycl::md_t(dst_md(0));
    conf_.ndims = ndims();
    conf_.mask = utils::get_dims_mask(data_d.dims(), weights_d.dims(), ndims());

    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.work_amount = memory_desc_wrapper(src_md(0)).nelems();
    conf_.work_amount_wei = memory_desc_wrapper(weights_md(0)).nelems();
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (conf_.work_amount + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;

    return status::success;
}

status_t ref_prelu_fwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<prelu_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_prelu_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
        int tot_work = nelems_A;
        prelu_fwd_kernel_vec_t prelu_fwd_kernel(pd()->conf_, cgh, ctx);
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;
        int work_per_wg = wg_size * block_size;
        int n_wgs = (tot_work + work_per_wg - 1) / work_per_wg;
        int n_thr = n_wgs * wg_size;
        cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size), prelu_fwd_kernel);
    });
}

status_t ref_prelu_bwd_t::pd_t::init_reduction(impl::engine_t *engine) {
    if (reduce_diff_weights_) {
        reduction_desc_t rdesc;
        scratch_md_ = memory_desc_t(*src_md(0));
        scratch_md_.data_type = diff_weights_md()->data_type;
        CHECK(reduction_desc_init(&rdesc, dnnl_alg_kind_t::dnnl_reduction_sum,
                &scratch_md_, diff_weights_md(0), 0, 0));
        primitive_attr_t reduction_attr(*attr());
        if (!reduction_attr.is_initialized()) return status::out_of_memory;

        primitive_desc_iterator_t it(
                engine, (op_desc_t *)&rdesc, &reduction_attr, nullptr);
        if (!it.is_initialized()) return status::invalid_arguments;
        while (++it != it.end()) {
            reduction_pd_ = *it;
            if (reduction_pd_) break;
        }

        if (!reduction_pd_) { return status::invalid_arguments; }
    }

    return status::success;
}

void ref_prelu_bwd_t::pd_t::init_scratchpad() {
    if (reduce_diff_weights_) {
        auto scratchpad = scratchpad_registry().registrar();
        size_t size
                = utils::array_product(src_md()->padded_dims, src_md()->ndims);
        scratchpad.book(memory_tracking::names::key_prelu_reduction, size,
                types::data_type_size(scratch_md_.data_type));

        scratchpad.book(memory_tracking::names::key_nested,
                reduction_pd_->scratchpad_registry());
    }
}

status_t ref_prelu_bwd_t::pd_t::init_conf() {
    if (has_zero_dim_memory()) return status::success;
    conf_ = sycl_prelu_conf_t();
    conf_.data_md = xpu::sycl::md_t(src_md(0));
    conf_.weights_md = xpu::sycl::md_t(weights_md(0));
    conf_.diff_data_md = xpu::sycl::md_t(diff_src_md(0));
    conf_.diff_weights_md = xpu::sycl::md_t(diff_weights_md(0));
    conf_.diff_dst_md = xpu::sycl::md_t(diff_dst_md(0));
    conf_.ndims = ndims();

    const memory_desc_wrapper weights_d(weights_md(0));
    const memory_desc_wrapper data_d(src_md(0));
    conf_.bcast_type = dnnl::impl::get_rhs_arg_broadcasting_strategy(
            *weights_d.md_, data_d);
    reduce_diff_weights_
            = (conf_.bcast_type == broadcasting_strategy_t::scalar);
    conf_.mask = utils::get_dims_mask(data_d.dims(), weights_d.dims(), ndims());
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.work_amount_src = memory_desc_wrapper(src_md(0)).nelems();
    conf_.work_amount = memory_desc_wrapper(weights_md(0)).nelems();
    conf_.work_load = conf_.work_amount_src / conf_.work_amount;
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (conf_.work_amount_src + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;

    return status::success;
}

status_t ref_prelu_bwd_t::init(impl::engine_t *engine) {
    if (pd()->reduce_diff_weights_) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
        CHECK(pd()->reduction_pd_->create_primitive_nested(p, engine));
        reduction_p_ = p.first;
    }

    const auto kid = ::sycl::get_kernel_id<prelu_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_prelu_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    std::unique_ptr<memory_t, memory_deleter_t> scratch_mem;
    if (pd()->reduce_diff_weights_) {
        auto scratchpad_storage
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_prelu_reduction);
        CHECK(safe_ptr_assign(scratch_mem,
                new memory_t(ctx.stream()->engine(), &pd()->scratch_md_,
                        std::move(scratchpad_storage))));
    }

    auto status = parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
        int tot_work = nelems_A;

        prelu_bwd_kernel_vec_t prelu_bwd_kernel(
                pd()->conf_, cgh, ctx, pd()->reduce_diff_weights_, scratch_mem);
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;
        int work_per_wg = wg_size * block_size;
        int n_wgs = (tot_work + work_per_wg - 1) / work_per_wg;
        int n_thr = n_wgs * wg_size;
        cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size), prelu_bwd_kernel);
    });

    CHECK(status);

    if (pd()->reduce_diff_weights_) {
        exec_args_t reduction_args;
        reduction_args[DNNL_ARG_SRC] = memory_arg_t {scratch_mem.get(), true};
        reduction_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DIFF_WEIGHTS);
        exec_ctx_t reduction_ctx(ctx, std::move(reduction_args));

        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested, reduction_p_);
        reduction_ctx.set_scratchpad_grantor(ns.grantor());
        return reduction_p_->execute(reduction_ctx);
    }

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
