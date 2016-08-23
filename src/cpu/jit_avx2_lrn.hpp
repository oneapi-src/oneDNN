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

#ifndef CPU_JIT_LRN_HPP
#define CPU_JIT_LRN_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "lrn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::precision_t prec>
class jit_avx2_lrn : public lrn<jit_avx2_lrn<prec>>
{
public:
    typedef typename prec_trait<prec>::type data_t;
    using lrn<jit_avx2_lrn<prec>>::lrn;

    jit_avx2_lrn(const lrn_primitive_desc_t &lpd, const primitive_at_t *inputs,
            const primitive *outputs[]);
    ~jit_avx2_lrn();

    static status_t set_default_parameters(lrn_desc_t &lrn_d);
    static status_t constraint(const lrn_desc_t &lrn_d);

    static const primitive_impl implementation;
private:
    /* Computes output h (x) w (x) 8 */
    void(*ker_first)(const void *);
    void(*ker_last)(const void *);
    void(*ker)(const void *);

    struct xbyak_lrn;
    xbyak_lrn *jit, *jit_first, *jit_last;

    status_t execute_forward();
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
