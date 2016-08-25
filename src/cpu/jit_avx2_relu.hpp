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

#ifndef CPU_JIT_RELU_HPP
#define CPU_JIT_RELU_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "relu.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::precision_t prec>
class jit_avx2_relu:
    public relu<jit_avx2_relu<prec>> {
public:
    typedef typename prec_trait<prec>::type data_t;
    using relu<jit_avx2_relu<prec>>::relu;

    jit_avx2_relu(const relu_primitive_desc_t &rpd, const primitive_at_t *inputs,
            const primitive *outputs[]);
    ~jit_avx2_relu();

    static status_t set_default_parameters(relu_desc_t &relu_d);
    static status_t constraint(const relu_desc_t &relu_d);

    static const primitive_impl implementation;
private:
    size_t chunk_size, n_chunks, n_reminder_elems;

    void(*ker_main)(const void *);
    void(*ker_reminder)(const void *);

    data_t jit_negative_slope;
    struct xbyak_relu;
    xbyak_relu *jit_relu, *jit_relu_reminder;

    status_t execute_forward();
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
