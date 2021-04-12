/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include "oneapi/dnnl/dnnl_graph_types.h"

#define MEM_ALIGNMENT (4096)

void *allocate(size_t mem_size, dnnl_graph_allocator_attr_t attr) {
    (void)attr;
    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(mem_size, MEM_ALIGNMENT);
    int ret = ((ptr) ? 0 : errno);
#else
    int ret = posix_memalign(&ptr, MEM_ALIGNMENT, mem_size);
#endif /* _WIN32 */
    return (ret == 0) ? ptr : NULL;
}

void deallocate(void *buffer) {
#ifdef _WIN32
    _aligned_free(buffer);
#else
    free(buffer);
#endif /* _WIN32 */
}
