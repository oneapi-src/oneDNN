/*******************************************************************************
 * Copyright 2021 Intel Corporation
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

#include "backend/dnnl/scratchpad.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

thread_local dnnl::engine thread_local_scratchpad_t::eng_ {};
thread_local const impl::allocator_t *thread_local_scratchpad_t::alloc_ {};
thread_local std::shared_ptr<char> thread_local_scratchpad_t::buffer_ {};
thread_local size_t thread_local_scratchpad_t::size_ {0};

thread_local_scratchpad_t::thread_local_scratchpad_t(
        size_t size, const dnnl::engine &eng, const impl::allocator_t &alloc) {
    assertm(eng_.get(true) == nullptr ? true
                                      : eng.get_kind() == eng_.get_kind(),
            "engine is invalid");

    if (size > size_) {
        eng_ = eng;
        alloc_ = &alloc;

        buffer_.reset();

        // Try to expand the global scratchpad to the necessary size
        char *buf = reinterpret_cast<char *>(
                allocator::malloc(size, eng, &alloc));
        if (buf == nullptr) {
            // Recreate scratchpad with original capacity
            buf = reinterpret_cast<char *>(
                    allocator::malloc(size_, eng, &alloc));
            if (buf == nullptr) {
                size_ = 0;
            } else {
                buffer_.reset(buf,
                        std::bind(allocator::free, std::placeholders::_1, eng_,
                                alloc_));
            }
        } else {
            size_ = size;
            buffer_.reset(buf,
                    std::bind(allocator::free, std::placeholders::_1, eng_,
                            alloc_));
        }
    }
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
