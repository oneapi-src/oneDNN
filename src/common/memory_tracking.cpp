/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "memory_tracking.hpp"
#include "primitive_exec_types.hpp"

#include "engine.hpp"

namespace dnnl {
namespace impl {
namespace memory_tracking {

void *grantor_t::get_host_ptr(const memory_storage_t *mem_storage) const {
    return exec_ctx_->host_ptr(mem_storage);
}

bool grantor_t::is_cpu_engine(const memory_storage_t *mem_storage) const {
    if (!mem_storage) return false;
    auto engine = mem_storage->engine();
    assert(engine);
    if (engine->kind() == engine_kind::cpu) return true;
    return false;
}

} // namespace memory_tracking
} // namespace impl
} // namespace dnnl
