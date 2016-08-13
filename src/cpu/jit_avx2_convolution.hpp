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

#ifndef CPU_JIT_CONVOLUTION_HPP
#define CPU_JIT_CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"
#include "jit_avx2_generator.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::primitive_kind;

template <impl::precision_t prec>
class jit_avx2_convolution: public primitive {
private:
    const impl::convolution_primitive_desc_t &_cpd;
    const bool _with_bias;

    jit_convolution_param_t jcp;
    jit_avx2_generator* generator;
    void (*jit_ker)(void*);

    inline int offset_w(int gr, int oc, int ic, int iw, int ih, int bo, int bi);
    inline int offset_b(int ch, int bl);
    inline int offset_src(int img, int ch, int ih, int iw, int bl);
    inline int offset_dst(int img, int ch, int ih, int iw, int bl);

    // TODO: implement in cpp.
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

    jit_avx2_convolution(const convolution_primitive_desc_t &cpd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(cpd, const_cast<impl::engine*>(cpd.base.engine), not_ready)
        , _cpd(_primitive_desc.convolution)
        , _with_bias(!memory_desc_wrapper(_cpd.bias_primitive_desc).is_zero())
    {
        for (int i = 0; i < 2 + _with_bias; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
        jcp.SIMD_W = 8;

        const memory_desc_wrapper
            src_d(cpd.src_primitive_desc.memory_desc),
            weights_d(cpd.weights_primitive_desc.memory_desc),
            bias_d(cpd.bias_primitive_desc.memory_desc),
            dst_d(cpd.dst_primitive_desc.memory_desc);

        const bool w_groups = weights_d.ndims() == (src_d.ndims() + 1);
        const uint32_t w_idx_base = w_groups ? 1 : 0;
        jcp.ngroups = w_groups ? weights_d.dims()[0] : 1;
        jcp.mb = src_d.dims()[0];
        jcp.ic = weights_d.dims()[w_idx_base + 1];
        jcp.oc = weights_d.dims()[w_idx_base + 0];

        jcp.ih = src_d.dims()[2]; jcp.iw = src_d.dims()[3];
        jcp.oh = dst_d.dims()[2]; jcp.ow = dst_d.dims()[3];

        jcp.t_pad = _cpd.convolution_desc.padding[0];
        jcp.l_pad = _cpd.convolution_desc.padding[1];
        jcp.kh = weights_d.dims()[w_idx_base + 2];
        jcp.kw = weights_d.dims()[w_idx_base + 3];
        jcp.stride_h = _cpd.convolution_desc.strides[0];
        jcp.stride_w = _cpd.convolution_desc.strides[1];

        jcp.ic_block = (jcp.ic % jcp.SIMD_W) ? 1 : jcp.SIMD_W;
        jcp.nb_ic = jcp.ic / jcp.ic_block;

        jcp.oc_block = jcp.SIMD_W;
        jcp.nb_oc = jcp.oc / jcp.oc_block;

        jcp.ur_h = 1; /* no code-unrolling by h so far */
        jcp.ur_w = 3;
        jcp.nb_ic_blocking =  jcp.nb_oc_blocking = 1;
        for (int b = 4; b > 1; b--)
            if (jcp.nb_oc % b == 0) {
                jcp.nb_oc_blocking = b;
                break;
            }
        jcp.ur_w_tail =jcp.ow % jcp.ur_w;

        generator = new jit_avx2_generator(&jcp);
//TODO: if(generator == nullptr) return nullptr;
        jit_ker = (void (*)(void*))generator->getCode();
//TODO: if(jit_ker == nullptr) return nullptr;
    }
    ~jit_avx2_convolution() {}


    /* static magic */
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkl_dnn::impl::engine &aengine);
    static const primitive_impl implementation;
};

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
