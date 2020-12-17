/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_BACKEND_DNNL_ALLOCATORS_HPP
#define LLGA_BACKEND_DNNL_ALLOCATORS_HPP

#include <sstream>

namespace llga {
namespace impl {
namespace dnnl_impl {
namespace utils {

class allocator {
public:
    constexpr static size_t tensor_memalignment = 4096;

    static char *malloc(size_t size) {
        void *ptr;
#ifdef _WIN32
        ptr = _aligned_malloc(size, tensor_memalignment);
        int rc = ((ptr) ? 0 : errno);
#else
        int rc = ::posix_memalign(&ptr, tensor_memalignment, size);
#endif /* _WIN32 */
        return (rc == 0) ? (char *)ptr : nullptr;
    }

    static void free(void *p) {
#ifdef _WIN32
        _aligned_free((void *)p);
#else
        ::free((void *)p);
#endif /* _WIN32 */
    }
};

} // namespace utils
} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif
