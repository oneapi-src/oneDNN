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

#ifndef MEMORY_STORAGE_MAP_MANAGER_HPP
#define MEMORY_STORAGE_MAP_MANAGER_HPP

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"

#include <functional>
#include <unordered_map>

namespace mkldnn {
namespace impl {

// Service class to support mapping/unmapping functionality for memory storages.
// 
// The main responsibility of the class is to store context information
// between map/unmap calls in form of a callback function.
//
// Unmap call may require an additional context that should be created and
// stored between map/unmap calls. However there is no way to store this
// context in an immutable memory storage therefore we need some external
// entity to be used for that.
struct memory_storage_map_manager_t : public c_compatible {
    memory_storage_map_manager_t() = default;

    ~memory_storage_map_manager_t() {
        assert(registered_unmap_pointers.empty());
        assert(registered_unmap_callbacks.empty());
    }

    static memory_storage_map_manager_t &instance() {
        static memory_storage_map_manager_t map_manager;
        return map_manager;
    }

    status_t register_unmap(const memory_storage_t *mem_storage,
            void *mapped_ptr, const std::function<void()> &callback) {
        // TODO: implementation is not thread-safe
        assert(registered_unmap_pointers.count(mem_storage) == 0);
        assert(registered_unmap_callbacks.count(mem_storage) == 0);

        registered_unmap_pointers[mem_storage] = mapped_ptr;
        registered_unmap_callbacks[mem_storage] = callback;

        return status::success;
    }

    status_t unmap(const memory_storage_t *mem_storage, const void *mapped_ptr) {
        assert(registered_unmap_pointers.count(mem_storage) == 1);
        assert(registered_unmap_callbacks.count(mem_storage) == 1);

        assert(mapped_ptr == registered_unmap_pointers[mem_storage]);

        registered_unmap_callbacks[mem_storage]();

        registered_unmap_pointers.erase(mem_storage);
        registered_unmap_callbacks.erase(mem_storage);

        return status::success;
    }

private:
    std::unordered_map<const memory_storage_t *, const void *> registered_unmap_pointers;
    std::unordered_map<const memory_storage_t *, std::function<void()>>
            registered_unmap_callbacks;
};

} // namespace impl
} // namespace mkldnn

#endif
