/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "dnnl_memory.hpp"
#include "dnnl_reorder.hpp"

#if DNNL_WITH_SYCL
#include <CL/sycl.hpp>
#endif

int dnn_mem_t::reorder(const dnn_mem_t &rhs, const attr_bundle_t *attr_bundle) {
    if (this == &rhs) return OK;
    return execute_reorder(rhs, *this, attr_bundle);
}

dnn_mem_t dnn_mem_t::create_from_host_ptr(
        const dnnl_memory_desc_t &md, dnnl_engine_t engine, void *host_ptr) {
    dnnl_engine_kind_t eng_kind;
    DNN_SAFE_V(dnnl_engine_get_kind(engine, &eng_kind));

    // XXX: allows to construct CPU memory only.
    assert(eng_kind == dnnl_cpu);
    (void)eng_kind;

    std::shared_ptr<void> handle;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    using buf_type = cl::sycl::buffer<uint8_t, 1>;
    size_t sz = dnnl_memory_desc_get_size(&md);
    handle.reset(new buf_type((uint8_t *)host_ptr, cl::sycl::range<1>(sz)),
            [](void *ptr) { delete (buf_type *)ptr; });

#else
    handle.reset(host_ptr, [](void *) {});
#endif
    return dnn_mem_t(md, engine, handle.get());
}
