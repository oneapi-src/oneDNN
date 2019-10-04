/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#ifndef CPU_ENGINE_HPP
#define CPU_ENGINE_HPP

#include <assert.h>

#include "dnnl.h"

#include "../common/engine.hpp"
#include "c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

class cpu_engine_impl_list_t {
public:
    static const engine_t::concat_primitive_desc_create_f *
    get_concat_implementation_list();
    static const engine_t::reorder_primitive_desc_create_f *
    get_reorder_implementation_list();
    static const engine_t::sum_primitive_desc_create_f *
    get_sum_implementation_list();
    static const engine_t::primitive_desc_create_f *get_implementation_list();
};

class cpu_engine_t : public engine_t {
public:
    cpu_engine_t() : engine_t(engine_kind::cpu, get_cpu_native_runtime()) {}

    /* implementation part */

    virtual status_t create_memory_storage(memory_storage_t **storage,
            unsigned flags, size_t size, size_t alignment,
            void *handle) override;

    virtual status_t create_stream(stream_t **stream, unsigned flags) override;

    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const override {
        return cpu_engine_impl_list_t::get_concat_implementation_list();
    }

    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list() const override {
        return cpu_engine_impl_list_t::get_reorder_implementation_list();
    }

    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const override {
        return cpu_engine_impl_list_t::get_sum_implementation_list();
    }

    virtual const primitive_desc_create_f *
    get_implementation_list() const override {
        return cpu_engine_impl_list_t::get_implementation_list();
    }
};

class cpu_engine_factory_t : public engine_factory_t {
public:
    virtual size_t count() const override { return 1; }
    virtual status_t engine_create(
            engine_t **engine, size_t index) const override {
        assert(index == 0);
        *engine = new cpu_engine_t();
        return status::success;
    };
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
