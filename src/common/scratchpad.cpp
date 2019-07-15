/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <memory>

#include "engine.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "scratchpad.hpp"

namespace mkldnn {
namespace impl {

/* Allocating memory buffers on a page boundary to reduce TLB/page misses */
const size_t page_size = 2097152;

/*
  Implementation of the scratchpad_t interface that is compatible with
  a concurrent execution
*/
struct concurrent_scratchpad_t : public scratchpad_t {
    concurrent_scratchpad_t(engine_t *engine, size_t size) : size_(size) {
        memory_storage_t *mem_storage_ptr;
        auto status = engine->create_memory_storage(
                &mem_storage_ptr, size, page_size);
        assert(status == status::success);
        MAYBE_UNUSED(status);
        mem_storage_.reset(mem_storage_ptr);
    }

    virtual const memory_storage_t *get_memory_storage() const override {
        return mem_storage_.get();
    }

private:
    std::unique_ptr<memory_storage_t> mem_storage_;
    size_t size_;

    MKLDNN_DISALLOW_COPY_AND_ASSIGN(concurrent_scratchpad_t);
};

/*
  Implementation of the scratchpad_t interface that uses a global
  scratchpad
*/

struct global_scratchpad_t : public scratchpad_t {
    global_scratchpad_t(engine_t *engine, size_t size) {
        // TODO: check if engine is the same
        if (size > size_) {
            size_ = size;
            memory_storage_t *mem_storage_ptr;
            auto status = engine->create_memory_storage(
                    &mem_storage_ptr, size, page_size);
            assert(status == status::success);
            MAYBE_UNUSED(status);
            mem_storage_.reset(mem_storage_ptr);
        }
        reference_count_++;
    }

    ~global_scratchpad_t() {
        reference_count_--;
        if (reference_count_ == 0) {
            mem_storage_.reset();
            size_ = 0;
        }
    }

    virtual const memory_storage_t *get_memory_storage() const override {
        return mem_storage_.get();
    }

private:
    thread_local static std::unique_ptr<memory_storage_t> mem_storage_;
    thread_local static size_t size_;
    thread_local static unsigned int reference_count_;
};

thread_local std::unique_ptr<memory_storage_t>
        global_scratchpad_t::mem_storage_;
thread_local size_t global_scratchpad_t::size_ = 0;
thread_local unsigned int global_scratchpad_t::reference_count_ = 0;


/*
   Scratchpad creation routine
*/
scratchpad_t *create_scratchpad(engine_t *engine, size_t size) {
#ifndef MKLDNN_ENABLE_CONCURRENT_EXEC
    return new global_scratchpad_t(engine, size);
#else
    return new concurrent_scratchpad_t(engine, size);
#endif
}
}
}
