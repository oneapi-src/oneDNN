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

#ifndef LLGA_ALLOCATOR_HPP
#define LLGA_ALLOCATOR_HPP

#include <cstdio>

namespace llga {

// Base class for allocator, which will be inherited by FW allocator
class base_allocator {
public:
    /// A default destructor
    virtual ~base_allocator() = default;

    /// Generate persistent buffer, destoryed by deallocate_persistent()
    /// or Framework on its exit
    ///
    /// @param n Memory size to be allocated
    /// @param alignment Specify what address offset should be aligned with
    /// @returns A pointer to the allocated buffer
    virtual void *allocate_persistent(size_t n, int alignment) = 0;

    /// Destroy persistent buffer
    ///
    /// @param buffer A pointer to the previously allocated buffer
    virtual void deallocate_persistent(void *buffer) = 0;

    /// Generate output tensor buffer whose's lifecycle is expected to
    /// be within an iteration. Framework will be responsible
    /// for the deallocation for such memory buffer
    ///
    /// @param n Memory size to be allocated
    /// @returns A pointer to the allocated buffer
    virtual void *allocate_output(size_t n) = 0;

    /// Generate temp memory buffer, whose lifecycle should be managed by allocator
    ///
    /// @param n Memory size to be allocated
    /// @returns A pointer to the allocated buffer
    virtual void *allocate_temp(size_t n) = 0;
};

} // namespace llga
#endif
