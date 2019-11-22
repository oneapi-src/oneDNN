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

#include "dnnl.hpp"
#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory.hpp"
#include "sycl/sycl_memory_storage.hpp"

namespace dnnl {

memory::memory(with_sycl_tag, const desc &md, const engine &eng, void *handle,
        bool is_usm) {
    using namespace dnnl::impl;
    using namespace dnnl::impl::sycl;

    engine_t *eng_c = eng.get();
    const memory_desc_t *md_c = &md.data;

    if (!eng_c || eng_c->runtime_kind() != runtime_kind::sycl)
        error::wrap_c_api(dnnl::impl::status::invalid_arguments,
                "could not create a memory");

    const auto mdw = memory_desc_wrapper(&md.data);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        error::wrap_c_api(dnnl::impl::status::invalid_arguments,
                "could not create a memory");

    size_t size = memory_desc_wrapper(md_c).size();
    unsigned flags = (handle == DNNL_MEMORY_ALLOCATE)
            ? memory_flags_t::alloc
            : memory_flags_t::use_runtime_ptr;

    std::unique_ptr<memory_storage_t> mem_storage;
#ifdef DNNL_SYCL_DPCPP
    if (is_usm) {
        mem_storage.reset(
                new sycl_usm_memory_storage_t(eng_c, flags, size, handle));
    }
#endif
    if (!is_usm) {
        mem_storage.reset(
                new sycl_buffer_memory_storage_t(eng_c, flags, size, handle));
    }
    if (!mem_storage)
        error::wrap_c_api(
                dnnl::impl::status::out_of_memory, "could not create a memory");

    auto *mem = new memory_t(eng_c, md_c, std::move(mem_storage), true);
    reset(mem);
}

} // namespace dnnl
