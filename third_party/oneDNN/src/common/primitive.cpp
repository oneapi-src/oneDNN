/*******************************************************************************
* Copyright 2016-2022 Intel Corporation
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

#include <string>

#include <assert.h>

#include "c_types_map.hpp"
#include "engine.hpp"

#if defined(DNNL_ENABLE_ITT_TASKS)
#include "ittnotify.hpp"
#endif

#include "primitive.hpp"
#include "primitive_desc.hpp"
#include "primitive_exec_types.hpp"
#include "reorder_pd.hpp"
#include "scratchpad_debug.hpp"
#include "stack_checker.hpp"
#include "stream.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::primitive_kind;

namespace dnnl {
namespace impl {

nested_scratchpad_t::nested_scratchpad_t(const exec_ctx_t &master_ctx, int key,
        const std::shared_ptr<primitive_t> &nested_p) {
    auto scratchpad = master_ctx.get_scratchpad_grantor();
    scratchpad_mem_storage_ = scratchpad.get_memory_storage(key);
    grantor_ = utils::make_unique<memory_tracking::grantor_t>(
            nested_p->pd()->scratchpad_registry().grantor(
                    scratchpad_mem_storage_.get(), master_ctx));
#ifdef DNNL_ENABLE_MEM_DEBUG
    if (scratchpad_debug::is_protect_scratchpad()) {
        scratchpad_debug::protect_scratchpad_buffer(
                grantor_->get_base_storage(), grantor_->get_registry());
    }
#endif
}

#ifdef DNNL_ENABLE_MEM_DEBUG
nested_scratchpad_t::~nested_scratchpad_t() {
    if (scratchpad_debug::is_protect_scratchpad()) {
        scratchpad_debug::unprotect_scratchpad_buffer(
                grantor_->get_base_storage(), grantor_->get_registry());
    }
}
#else
nested_scratchpad_t::~nested_scratchpad_t() = default;
#endif

} // namespace impl
} // namespace dnnl
