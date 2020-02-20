/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <assert.h>
#include <memory>

#include "dnnl.h"

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "memory_storage.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
enum memory_flags_t { alloc = 0x1, use_runtime_ptr = 0x2, omit_zero_pad = 0x4 };
} // namespace impl
} // namespace dnnl

struct dnnl_memory : public dnnl::impl::c_compatible {
    dnnl_memory(dnnl::impl::engine_t *engine,
            const dnnl::impl::memory_desc_t *md, unsigned flags, void *handle);
    virtual ~dnnl_memory() {}

    /** returns memory's engine */
    dnnl::impl::engine_t *engine() const { return engine_; }
    /** returns memory's description */
    const dnnl::impl::memory_desc_t *md() const { return &md_; }
    /** returns the underlying memory storage */
    dnnl::impl::memory_storage_t *memory_storage() const {
        return memory_storage_.get();
    }
    /** returns data handle */
    dnnl::impl::status_t get_data_handle(void **handle) const {
        return memory_storage()->get_data_handle(handle);
    }

    /** sets data handle */
    dnnl::impl::status_t set_data_handle(void *handle) {
        using namespace dnnl::impl;

        void *old_handle;
        CHECK(memory_storage()->get_data_handle(&old_handle));

        if (handle != old_handle) {
            CHECK(memory_storage()->set_data_handle(handle));
        }
        return zero_pad();
    }

    /** zeros padding */
    dnnl::impl::status_t zero_pad() const;

protected:
    dnnl::impl::engine_t *engine_;
    const dnnl::impl::memory_desc_t md_;

private:
    template <dnnl::impl::data_type_t>
    dnnl::impl::status_t typed_zero_pad() const;

    dnnl_memory() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(dnnl_memory);

    std::unique_ptr<dnnl::impl::memory_storage_t> memory_storage_;
};

#endif
