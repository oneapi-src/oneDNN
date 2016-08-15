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

#ifndef CPU_gemm_INNER_PRODUCT_HPP
#define CPU_gemm_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_engine.hpp"
#include "primitive.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::precision;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::primitive_kind;

template <impl::precision_t prec>
class gemm_inner_product : public primitive {
private:
    const impl::inner_product_primitive_desc_t &_ippd;
    const bool _with_bias;

    status_t execute_forward();
    status_t execute_backward_data();
    status_t execute_backward_weights();
    status_t execute_backward_bias();

protected:
    status_t execute_impl()
    {
        switch (_ippd.inner_product_desc.prop_kind) {
        case forward: return execute_forward(); break;
        case backward_data: return execute_backward_data(); break;
        case backward_weights: return execute_backward_weights(); break;
        case backward_bias: return execute_backward_bias(); break;
        default: assert(0 && "invalid prop_kind"); // should never happen
        }
    }

public:
    typedef typename precision2type<prec>::type data_t;

    gemm_inner_product(const inner_product_primitive_desc_t &ippd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(ippd, const_cast<impl::engine *>(ippd.base.engine),
                not_ready)
        , _ippd(_primitive_desc.inner_product)
        , _with_bias(!memory_desc_wrapper(_ippd.bias_primitive_desc).is_zero())
    {
        for (int i = 0; i < 2 + _with_bias; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
    }
    ~gemm_inner_product() {}

    /* static magic */
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkldnn::impl::engine &aengine);
    static const primitive_impl implementation;
};
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
