/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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
#include "utils.hpp"

#include "scratchpad.hpp"

namespace dnnl {
namespace impl {

namespace {

memory_storage_t *create_scratchpad_memory_storage(
        engine_t *engine, size_t size) {
    memory_storage_t *mem_storage;
    auto status = engine->create_memory_storage(&mem_storage, size);
    assert(status == status::success);
    MAYBE_UNUSED(status);
    return mem_storage;
}

} // namespace

/*
  Implementation of the scratchpad_t interface that is compatible with
  a concurrent execution
*/
struct concurrent_scratchpad_t : public scratchpad_t {
    concurrent_scratchpad_t(engine_t *engine, size_t size) {
        auto *mem_storage = create_scratchpad_memory_storage(engine, size);
        mem_storage_.reset(mem_storage);
    }

    virtual const memory_storage_t *get_memory_storage() const override {
        return mem_storage_.get();
    }

private:
    std::unique_ptr<memory_storage_t> mem_storage_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(concurrent_scratchpad_t);
};

/*
  Implementation of the scratchpad_t interface that uses a global
  scratchpad
*/

struct global_scratchpad_t : public scratchpad_t {
    global_scratchpad_t(engine_t *engine, size_t size) {
        UNUSED(engine);
        if (size > size_) {
            delete mem_storage_;
            mem_storage_ = create_scratchpad_memory_storage(engine, size);
            size_ = size;
        }
        reference_count_++;
    }

    ~global_scratchpad_t() {
        reference_count_--;
        if (reference_count_ == 0) {
            delete mem_storage_;
            mem_storage_ = nullptr;
            size_ = 0;
        }
    }

    virtual const memory_storage_t *get_memory_storage() const override {
        return mem_storage_;
    }

private:
    thread_local static memory_storage_t *mem_storage_;
    thread_local static size_t size_;
    thread_local static unsigned int reference_count_;
};

// CAVEAT: avoid having non-trivially-constructed thread-local objects. Their
// construction order may depends on the program execution and the final
// destruction order may be such that a thread-local object is destroyed
// before all its users are destroyed thus causing a crash at exit.
// Tested by tests/gtests/test_global_scratchad.cpp
thread_local memory_storage_t *global_scratchpad_t::mem_storage_ = nullptr;
thread_local size_t global_scratchpad_t::size_ = 0;
thread_local unsigned int global_scratchpad_t::reference_count_ = 0;

/*
   Scratchpad creation routine
*/
scratchpad_t *create_scratchpad(
        engine_t *engine, size_t size, bool use_global_scratchpad) {
#ifndef DNNL_ENABLE_CONCURRENT_EXEC
    /*
     * TODO: global scratchpad should be able to handle memory
     * from different engines.
     * lock global scratchpad to work with CPU engine only.
     */
    if (use_global_scratchpad && engine->kind() == engine_kind_t::dnnl_cpu)
        return new global_scratchpad_t(engine, size);
    else
        return new concurrent_scratchpad_t(engine, size);
#else
    UNUSED(use_global_scratchpad);
    return new concurrent_scratchpad_t(engine, size);
#endif
}

} // namespace impl
} // namespace dnnl
