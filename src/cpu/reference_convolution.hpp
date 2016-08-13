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

#ifndef CPU_REFERENCE_CONVOLUTION_HPP
#define CPU_REFERENCE_CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"

namespace mkldnn { namespace impl { namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::precision;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::primitive_kind;

template <impl::precision_t prec>
class reference_convolution: public primitive {
private:
    const impl::convolution_primitive_desc_t &_cpd;
    const bool _with_bias;

    status_t execute_forward();
    status_t execute_backward_data();
    status_t execute_backward_weights();
    status_t execute_backward_bias();

protected:
    status_t execute_impl() {
        status_t status = success;
        _exec_state = busy;
        switch (_cpd.convolution_desc.prop_kind) {
        case forward: status = execute_forward(); break;
        case backward_data: status = execute_backward_data(); break;
        case backward_weights: status = execute_backward_weights(); break;
        case backward_bias: status = execute_backward_bias(); break;
        default: assert(0 && "invalid prop_kind"); // should never happen
        }
        _exec_state = done;
        return status;
    }

public:
    typedef typename precision2type<prec>::type data_t;

    reference_convolution(const convolution_primitive_desc_t &cpd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(cpd, const_cast<impl::engine*>(cpd.base.engine), not_ready)
        , _cpd(_primitive_desc.convolution)
        , _with_bias(!memory_desc_wrapper(_cpd.bias_primitive_desc).is_zero())
    {
        for (int i = 0; i < 2 + _with_bias; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
    }
    ~reference_convolution() {}

    /* static magic */
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkldnn::impl::engine &aengine);
    static const primitive_impl implementation;
};

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
