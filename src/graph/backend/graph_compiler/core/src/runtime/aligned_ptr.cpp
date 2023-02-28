/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include "aligned_ptr.hpp"
#include <immintrin.h>
#include <runtime/config.hpp>
#include <util/parallel.hpp>
#include <util/simple_math.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using utils::divide_and_ceil;
using utils::parallel;

void generic_ptr_base_t::zeroout() const {
    static constexpr int page_size = 4096;
    int numthreads = runtime_config_t::get().get_num_threads();
    parallel(
            [&](uint64_t i, uint64_t n) {
                if (i != n - 1) {
                    memset(static_cast<char *>(ptr_) + i * page_size, 0,
                            page_size);
                } else {
                    memset(static_cast<char *>(ptr_) + i * page_size, 0,
                            size_ - i * page_size);
                }
            },
            0, divide_and_ceil(size_, page_size), 1, numthreads);
}
/**
 * Flush cache
 * */
void generic_ptr_base_t::flush_cache() const {
    static constexpr int cache_line_size = 64;
    int numthreads = runtime_config_t::get().get_num_threads();
    parallel(
            [&](uint64_t i, uint64_t n) {
                _mm_clflush(static_cast<const void *>(
                        static_cast<const char *>(ptr_) + i));
            },
            0, size_, cache_line_size, numthreads);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
