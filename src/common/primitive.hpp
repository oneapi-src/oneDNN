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

#ifndef PRIMITIVE_HPP
#define PRIMITIVE_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"

struct mkldnn_primitive: public mkldnn::impl::c_compatible {
public:
    enum exec_state { ready, not_ready, error };

    typedef mkldnn::impl::nstl::vector<mkldnn::impl::primitive_at_t>
        input_vector;

    typedef mkldnn::impl::nstl::vector<const mkldnn::impl::primitive *>
        output_vector;

    mkldnn::impl::primitive_desc_t primitive_desc() const {
        return _primitive_desc;
    }

    mkldnn::impl::engine *engine() const { return _engine; }

    mkldnn::impl::primitive_kind_t kind() const {
        return _primitive_desc.base.primitive_kind;
    }

    virtual exec_state get_exec_state() const {
        return _exec_state;
    }

    bool inputs_ready() const {
        for (auto i = 0UL; i < _input.size(); i++)
            if (_input[i].primitive->get_exec_state() != ready)
                return false;
        return true;
    }

    mkldnn::impl::status_t execute() {
        if (!inputs_ready())
            return mkldnn::impl::status::not_ready;
        _exec_state = not_ready;
        mkldnn::impl::status_t status = execute_impl();
        _exec_state = (status == mkldnn::impl::status::success)
            ? ready : error;
        return status;
    }

    size_t input_count() const { return _input.size(); }
    const input_vector &input() const { return _input; }

    size_t output_count() const { return _output.size(); }
    const output_vector &output() const { return _output; }

    virtual char* memory(size_t index = 0) const {
        return output()[index]->memory();
    }
    virtual const char* memory_const(size_t index = 0) const {
        return output()[index]->memory_const();
    }

    virtual ~mkldnn_primitive() {}

protected:
    exec_state _exec_state;

    mkldnn::impl::engine *_engine;

    const mkldnn::impl::primitive_desc_t _primitive_desc;

    input_vector _input;
    output_vector _output;

    virtual mkldnn::impl::status_t execute_impl() = 0;

    mkldnn_primitive(const mkldnn::impl::primitive_desc_t& primitive_desc,
            mkldnn::impl::engine *engine, exec_state state = not_ready)
        : _exec_state(state)
        , _engine(engine)
        , _primitive_desc(primitive_desc)
    {}

private:
    mkldnn_primitive() = delete;
    mkldnn_primitive(const mkldnn_primitive &) = delete;
    mkldnn_primitive(mkldnn_primitive &&) = delete;
    mkldnn_primitive &operator=(const mkldnn_primitive &) = delete;
    mkldnn_primitive &operator=(mkldnn_primitive &&) = delete;
};

namespace mkldnn { namespace impl {

typedef status_t (*primitive_desc_init_f)(primitive_desc_t *primitive_desc,
        const op_desc_t &op_desc, const engine &aengine);

typedef status_t (*primitive_create_f)(primitive **aprimitive,
        const primitive_desc_t *primitive_desc, const primitive_at_t inputs[],
        const primitive *outputs[]);

struct primitive_impl {
    const primitive_create_f primitive_create;
};

status_t primitive_desc_init(primitive_desc_t *primitive_desc,
        const op_desc_t &op_desc, const engine &aengine);

status_t inline check_inputs_array(size_t n, const primitive_at_t inputs[]) {
    for (size_t i = 0; i < n; i++)
        if (inputs[i].primitive->output_count() <= inputs[i].output_index)
            return status::invalid_arguments;
    return status::success;
}

}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
