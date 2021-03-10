/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "gpu/ocl/gemm/gemm_with_post_ops.hpp"
#include "gpu/gemm/gpu_gemm_utils.hpp"

#ifdef DNNL_WITH_SYCL
#include "dnnl_sycl.hpp"
#include "sycl/sycl_memory_storage.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gemm_with_post_ops_t::pd_t::init(engine_t *engine) {

    if (attr()->post_ops_.len() == 0) return status::unimplemented;

    auto attributes_with_po = attr()->clone();
    attributes_with_po->set_scratchpad_mode(scratchpad_mode::user);

    auto attributes_without_po = attr()->clone();
    attributes_without_po->post_ops_.entry_.clear();
    attributes_without_po->set_scratchpad_mode(scratchpad_mode::user);

    const auto gemm_desc = desc();

    const auto impl_list
            = engine->get_implementation_list((op_desc_t *)gemm_desc);

    dnnl::impl::primitive_desc_t *gemm_candidate_pd = nullptr;

    status_t pd_create_res = status_t::dnnl_unimplemented;
    unsigned list_idx = 0;
    do {
        if (impl_list[list_idx]
                != &primitive_desc_t::create<ocl::gemm_with_post_ops_t::pd_t>) {
            pd_create_res = impl_list[list_idx](&gemm_candidate_pd,
                    (op_desc_t *)gemm_desc, attributes_with_po, engine,
                    nullptr);

            // exit if gemm kernel support post ops
            if (pd_create_res == status_t::dnnl_success) {
                delete gemm_candidate_pd;
                return status::unimplemented;
            }

            pd_create_res = impl_list[list_idx](&gemm_candidate_pd,
                    (op_desc_t *)gemm_desc, attributes_without_po, engine,
                    nullptr);
        }
    } while (pd_create_res != status_t::dnnl_success
            && impl_list[++list_idx] != nullptr);

    CHECK(pd_create_res);
    gemm_pd_.reset(gemm_candidate_pd);

    desc_.c_desc = *gemm_pd_->dst_md();

    // initilize post op worker
    dnnl_eltwise_desc_t po_worker_desc;
    dnnl_memory_desc_t po_input_mem_desc(*dst_md(0));
    dnnl_eltwise_forward_desc_init(&po_worker_desc,
            dnnl_prop_kind_t::dnnl_forward_inference,
            dnnl_alg_kind_t::dnnl_eltwise_linear, &po_input_mem_desc, 1, 0);
    dnnl_primitive_desc_iterator it(
            engine, (op_desc_t *)&po_worker_desc, attributes_with_po, nullptr);
    if (!it.is_initialized()) return status::invalid_arguments;
    ++it;
    post_op_worker_pd_.reset(it.fetch_once());
    if (post_op_worker_pd_) {
        use_scratchpad_with_post_op_worker
                = attributes_with_po->post_ops_.find(primitive_kind_t::dnnl_sum)
                != -1;
    } else {
        return status::unimplemented;
    }
    init_scratchpad();

    return status::success;
}

void gemm_with_post_ops_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    if (use_scratchpad_with_post_op_worker) {
        size_t size
                = utils::array_product(dst_md()->padded_dims, dst_md()->ndims);
        scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer, size,
                types::data_type_size(dst_md()->data_type));
    }
    scratchpad.book(memory_tracking::names::key_nested,
            gemm_pd_->scratchpad_registry());
    scratchpad.book(memory_tracking::names::key_nested,
            post_op_worker_pd_->scratchpad_registry());
}

status_t gemm_with_post_ops_t::execute(const gemm_exec_ctx_t &ctx) const {
    std::unique_ptr<memory_t> c_mem_before_po_worker;
    status_t exec_status;

    if (pd()->use_scratchpad()) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_tmp_buffer);
        CHECK(safe_ptr_assign(c_mem_before_po_worker,
                new memory_t(ctx.stream()->engine(), pd()->dst_md(0),
                        memory_flags_t::use_runtime_ptr,
                        scratchpad->data_handle())));

        gemm_exec_args_t args(ctx.args());
        args.c = c_mem_before_po_worker->memory_storage();

        exec_ctx_t tmp_exec_ctx(ctx.stream());
        tmp_exec_ctx.set_resource_mapper(ctx.get_resource_mapper());
        tmp_exec_ctx.set_scratchpad_grantor(&ctx.get_scratchpad_grantor());
        nested_scratchpad_t ns(
                tmp_exec_ctx, memory_tracking::names::key_nested, gemm_prim_);

        gemm_exec_ctx_t gemm_ex_ctx
                = gemm_exec_ctx_t(ctx.stream(), args, ctx.desc());
        gemm_ex_ctx.set_resource_mapper(ctx.get_resource_mapper());
        gemm_ex_ctx.set_scratchpad_grantor(ns.grantor());

        exec_status = gemm_utils::gpu_gemm(gemm_prim_)->execute(gemm_ex_ctx);
    } else {
        exec_status = gemm_utils::gpu_gemm(gemm_prim_)->execute(ctx);
    }

    CHECK(exec_status);

    std::unique_ptr<memory_t> dst_mem;
    const auto *dst_memory_storage = &GEMM_CTX_ARG_STORAGE(c);
#ifdef DNNL_WITH_SYCL
    auto *sycl_memory_storage
            = utils::downcast<const impl::sycl::sycl_memory_storage_base_t *>(
                    dst_memory_storage);
    auto m_kind = sycl_memory_storage->memory_kind();

    memory_t *memory_ptr;
    ::dnnl_sycl_interop_memory_create(&memory_ptr, pd()->dst_md(0),
            ctx.stream()->engine(), m_kind, dst_memory_storage->data_handle());
    CHECK(safe_ptr_assign(dst_mem, memory_ptr));
#else
    CHECK(safe_ptr_assign(dst_mem,
            new memory_t(ctx.stream()->engine(), pd()->dst_md(0),
                    memory_flags_t::use_runtime_ptr,
                    dst_memory_storage->data_handle())));
#endif

    exec_args_t po_worker_args;
    if (pd()->use_scratchpad()) {
        po_worker_args[DNNL_ARG_SRC]
                = memory_arg_t {c_mem_before_po_worker.get(), true};
    } else {
        po_worker_args[DNNL_ARG_SRC] = memory_arg_t {dst_mem.get(), true};
    }
    po_worker_args[DNNL_ARG_DST] = memory_arg_t {dst_mem.get(), false};
    exec_ctx_t po_exec_ctx(ctx.stream(), std::move(po_worker_args));
    po_exec_ctx.set_resource_mapper(ctx.get_resource_mapper());

    po_exec_ctx.set_scratchpad_grantor(&ctx.get_scratchpad_grantor());
    nested_scratchpad_t ns(po_exec_ctx, memory_tracking::names::key_nested,
            post_op_worker_prim_);
    po_exec_ctx.set_scratchpad_grantor(ns.grantor());

    exec_status = post_op_worker_prim_->execute(po_exec_ctx);

    return exec_status;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
