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

namespace dnnl {

memory::memory(with_sycl_tag, const desc &md, const engine &eng, bool is_usm) {
    using namespace dnnl::impl;

    engine_t *eng_c = eng.get();
    const memory_desc_t *md_c = &md.data;

    if (!eng_c || eng_c->runtime_kind() != runtime_kind::sycl)
        error::wrap_c_api(
                status::invalid_arguments, "could not create a memory");

    if (md_c->format_kind == dnnl::impl::format_kind::any)
        error::wrap_c_api(
                status::invalid_arguments, "could not create a memory");

    size_t size = memory_desc_wrapper(md_c).size();

    memory_storage_t *mem_storage_ptr;
    status_t status = eng_c->create_memory_storage(
            &mem_storage_ptr, memory_flags_t::alloc, size, 0, nullptr);
    if (status != status::success)
        error::wrap_c_api(status, "could not create a memory");

    auto *mem = new memory_t(eng_c, md_c, mem_storage_ptr, true);
    reset(mem);
}

} // namespace dnnl
