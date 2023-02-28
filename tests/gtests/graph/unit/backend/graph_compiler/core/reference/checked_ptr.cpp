
/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include <checked_ptr.hpp>
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/mman.h>
#endif
#include <assert.h>
#include <stdexcept>
#include <runtime/os.hpp>
#include <util/assert.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
static void fill_memory(char *start, char *end) {
    memset(start, 0xcc, end - start);
}

void *checked_ptr_policy_t::alloc(size_t sz, size_t alignment) {
    size_t page_sz = runtime::get_os_page_size();
    size_t data_size = utils::rnd_up(sz, page_sz);
    size_t real_sz = data_size + page_sz * 2;
    COMPILE_ASSERT(
            real_sz > 2 * page_sz, "At least 3 pages should be allocated");
#ifdef _WIN32
    throw std::runtime_error("checked_ptr_policy_t::alloc not implemented");
#else
    auto ret = mmap(nullptr, real_sz, PROT_READ | PROT_WRITE,
            MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    COMPILE_ASSERT(ret, "mmap failed");
    mprotect(ret, page_sz, PROT_NONE);
    auto protect_page_rhs = (char *)ret + real_sz - page_sz;
    mprotect(protect_page_rhs, page_sz, PROT_NONE);
    auto result = protect_page_rhs - utils::rnd_up(sz, alignment);
    fill_memory(result + sz, protect_page_rhs);
    fill_memory((char *)ret + page_sz, result);
    return result;
#endif
}

static void check(
        uint8_t *start, uint8_t *end, void *base_buffer, size_t real_sz) {
#ifdef _WIN32
    throw std::runtime_error("checked_ptr_policy_t::alloc not implemented");
#else
    for (uint8_t *p = start; p < end; p++) {
        if (*p != 0xcc) {
            fputs("Buffer overflow detected\n", stderr);
            munmap(base_buffer, real_sz);
            std::abort();
        }
    }
#endif
}
void checked_ptr_policy_t::dealloc(void *ptr, size_t sz) {
    size_t page_sz = runtime::get_os_page_size();
    size_t data_size = utils::rnd_up(sz, page_sz);
    size_t real_sz = data_size + page_sz * 2;
    auto buffer = (uint8_t *)(utils::rnd_dn((size_t)ptr, page_sz) - page_sz);
#ifdef _WIN32
    throw std::runtime_error("checked_ptr_policy_t::dealloc not implemented");
#else
    auto protect_page_rhs = (uint8_t *)buffer + real_sz - page_sz;
    check((uint8_t *)ptr + sz, protect_page_rhs, buffer, real_sz);
    check(buffer + page_sz, (uint8_t *)ptr, buffer, real_sz);
    munmap(buffer, real_sz);
#endif
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
