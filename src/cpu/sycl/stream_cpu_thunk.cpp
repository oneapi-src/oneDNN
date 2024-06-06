/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/primitive_iface.hpp"
#include "common/utils.hpp"

#include "cpu/sycl/stream_cpu_thunk.hpp"

using namespace dnnl::impl::cpu::sycl;

void dnnl_impl_sycl_cpu_thunk(const thunk_params_t *params) {
    using namespace dnnl::impl;

    auto *submit_ctx = params->submit_ctx_ptr;
    auto *prim_iface = submit_ctx->prim_iface;

    assert(params->size == submit_ctx->sycl_mem_storages.size());
    for (size_t i = 0; i < params->size; i++) {
        auto *mem_storage = submit_ctx->sycl_mem_storages[i];
        void *handle = mem_storage->data_handle();
        void *host_ptr = reinterpret_cast<void *>(params->native_pointers[i]);
        submit_ctx->exec_ctx.register_memory_mapping(handle, host_ptr);
    }

    prim_iface->execute(submit_ctx->exec_ctx);

    const_cast<primitive_iface_t *>(prim_iface)->release();

    delete submit_ctx;
}
