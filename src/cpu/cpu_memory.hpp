/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef CPU_MEMORY_HPP
#define CPU_MEMORY_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"
#include "utils.hpp"

namespace mkldnn { namespace impl { namespace cpu {

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

class cpu_memory: public primitive {
private:
    char *_memory_buffer;
    const bool _owns_memory;

protected:
    status_t execute_impl() { return success; }

public:
    cpu_memory(const memory_primitive_desc_t &mpd, char* ptr)
        : primitive(mpd, const_cast<impl::engine*>(mpd.base.engine), ready)
        , _memory_buffer(ptr)
        , _owns_memory(ptr == nullptr) {
        primitive_at_t input_at = { this, 0 };
        _input.push_back(input_at);
        _output.push_back(this);
        if (_memory_buffer == nullptr) {
            const size_t size = memory_desc_wrapper(mpd).size();
            _memory_buffer = static_cast<char *>(
                    mkldnn::impl::malloc(size, default_alignment));
        }
    }
    ~cpu_memory() { if (_owns_memory) mkldnn::impl::free(_memory_buffer); }

    bool owns_memory() const { return _owns_memory; }

    virtual char* memory(size_t index = 0) const
    { assert(index == 0); return _memory_buffer; }
    virtual const char* memory_const(size_t index = 0) const
    { assert(index == 0); return _memory_buffer; }

    /* static magic */
    static status_t memory_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkldnn::impl::engine &aengine);
    static const primitive_impl memory_implementation;
};

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
