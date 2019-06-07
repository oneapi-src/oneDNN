/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "memory_storage.hpp"
#include "nstl.hpp"

namespace mkldnn {
namespace impl {
enum memory_flags_t { alloc = 0x1, use_backend_ptr = 0x2 };
} // namespace impl
} // namespace mkldnn

struct mkldnn_memory : public mkldnn::impl::c_compatible {
    mkldnn_memory(mkldnn::impl::engine_t *engine,
            const mkldnn::impl::memory_desc_t *md, unsigned flags,
            void *handle);
    virtual ~mkldnn_memory() {}

    /** returns memory's engine */
    mkldnn::impl::engine_t *engine() const { return engine_; }
    /** returns memory's description */
    const mkldnn::impl::memory_desc_t *md() const { return &md_; }
    /** returns the underlying memory storage */
    mkldnn::impl::memory_storage_t *memory_storage() const {
        return memory_storage_.get();
    }
    /** returns data handle */
    mkldnn::impl::status_t get_data_handle(void **handle) const {
        return memory_storage()->get_data_handle(handle);
    }

    /** sets data handle */
    mkldnn::impl::status_t set_data_handle(void *handle) {
        using namespace mkldnn::impl;

        status_t status = memory_storage()->set_data_handle(handle);
        if (status != status::success)
            return status;
        return zero_pad();
    }

    /** zeros padding */
    mkldnn::impl::status_t zero_pad() const;

protected:
    mkldnn::impl::engine_t *engine_;
    const mkldnn::impl::memory_desc_t md_;

private:
    template <mkldnn::impl::data_type_t>
    mkldnn::impl::status_t typed_zero_pad() const;

    mkldnn_memory() = delete;
    MKLDNN_DISALLOW_COPY_AND_ASSIGN(mkldnn_memory);

    std::unique_ptr<mkldnn::impl::memory_storage_t> memory_storage_;
};

#endif
