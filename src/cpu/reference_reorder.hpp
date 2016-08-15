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

#ifndef CPU_REFERENCE_REORDER_HPP
#define CPU_REFERENCE_REORDER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::primitive_kind;

template <impl::precision_t prec_i, impl::precision_t prec_o>
class reference_reorder: public primitive {
private:
    const impl::reorder_primitive_desc_t &_rpd;

    status_t execute();

protected:
    status_t execute_impl() {
        _exec_state = busy;
        status_t status = execute();
        _exec_state = done;
        return status;
    }

public:
    typedef typename precision2type<prec_i>::type data_i_t;
    typedef typename precision2type<prec_o>::type data_o_t;

    reference_reorder(const reorder_primitive_desc_t &rpd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(rpd, const_cast<impl::engine*>(rpd.base.engine), not_ready)
        , _rpd(_primitive_desc.reorder)
    {
        _input.push_back(inputs[0]);
        _output.push_back(outputs[0]);
    }
    ~reference_reorder() {}

    /* static magic */
    static status_t reorder_primitive_desc_init(
            primitive_desc_t *primitive_desc,
            const memory_primitive_desc_t *input,
            const memory_primitive_desc_t *output);
    static const primitive_impl implementation;
};

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

