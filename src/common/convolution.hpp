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

#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::precision;

template <typename convolution_impl>
class convolution: public primitive {
public:
    convolution(const convolution_primitive_desc_t &pd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(pd, const_cast<impl::engine *>(pd.base.engine), not_ready)
        , _cpd(_primitive_desc.convolution)
        , _is_training(_cpd.convolution_desc.prop_kind
                == prop_kind::forward_training)
        , _with_bias(!memory_desc_wrapper(_cpd.bias_primitive_desc).is_zero())
        , _with_groups(memory_desc_wrapper(_cpd.weights_primitive_desc).ndims()
                == (memory_desc_wrapper(_cpd.src_primitive_desc).ndims() + 1))
    {
        for (int i = 0; i < 2 + _with_bias; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
    }

    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkldnn::impl::engine &aengine);

protected:
    const impl::convolution_primitive_desc_t &_cpd;
    const bool _is_training;
    const bool _with_bias;
    const bool _with_groups;

    virtual status_t execute_impl() {
        switch (_cpd.convolution_desc.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_scoring:
            return execute_forward();
        case prop_kind::backward_data: return execute_backward_data();
        case prop_kind::backward_weights: return execute_backward_weights();
        case prop_kind::backward_bias: return execute_backward_bias();
        default: assert(!"invalid prop_kind");
        }
        return unimplemented;
    }

    static status_t create(primitive **aprimitive,
            const primitive_desc_t *primitive_desc,
            const primitive_at_t inputs[], const primitive *outputs[]);

    /* derivative class should manually define the implementation
     * static const primitive_impl implementation; */

    /* child classes might want to use parent default setter and constraint */
    template <typename Impl>
    static status_t set_default_parameters(convolution_desc_t &conv_d, ...);
    template <typename Impl>
    static status_t constraint(const convolution_desc_t &conv_d, ...)
    { return success; }

private:
    template <typename Impl = convolution_impl>
    using data_t = typename Impl::data_t;

    virtual status_t execute_forward() { return unimplemented; }
    virtual status_t execute_backward_data() { return unimplemented; }
    virtual status_t execute_backward_weights() { return unimplemented; }
    virtual status_t execute_backward_bias() { return unimplemented; }

    template <typename Impl>
    static status_t set_default_parameters(convolution_desc_t &conv_d,
            decltype(&Impl::set_default_parameters) _ = 0)
    { return Impl::set_default_parameters(conv_d); }

    template <typename Impl>
    static status_t constraint(const convolution_desc_t &conv_d,
            decltype(&Impl::constraint))
    { return Impl::constraint(conv_d); }
};

/* implementation */

template <typename convolution_impl>
status_t convolution<convolution_impl>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkldnn::impl::engine &aengine) {
    using namespace prop_kind;
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    auto conv_d = op_desc.convolution;

    bool args_ok = everyone_is(prec, conv_d.src_desc.precision,
            conv_d.weights_desc.precision,
            conv_d.bias_desc.format == memory_format::undef ? prec
            : conv_d.bias_desc.precision, conv_d.dst_desc.precision)
        && op_desc._kind == primitive_kind::convolution
        && one_of(conv_d.prop_kind, forward_training, forward_scoring,
                backward_data, backward_weights, backward_bias);
    if (!args_ok) return invalid_arguments;

    status_t status;
    status = set_default_parameters<convolution_impl>(conv_d, 0);
    if (status != success) return status;

    status = constraint<convolution_impl>(conv_d, 0);
    if (status != success) return status;

    /* final stage */
    convolution_primitive_desc_t cpd = {};
    cpd.base.primitive_kind = primitive_kind::convolution;
    cpd.base.engine = &aengine;
    cpd.base.implementation = reinterpret_cast<const void*>(
            &convolution_impl::implementation);
    cpd.convolution_desc = conv_d;
    CHECK(mkldnn_memory_primitive_desc_init(&cpd.src_primitive_desc,
                &conv_d.src_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&cpd.weights_primitive_desc,
                &conv_d.weights_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&cpd.bias_primitive_desc,
                &conv_d.bias_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&cpd.dst_primitive_desc,
                &conv_d.dst_desc, &aengine));

    primitive_desc->convolution = cpd;
    return success;
}

template <typename convolution_impl>
status_t convolution<convolution_impl>::create(
        primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind ==
            primitive_kind::convolution);
    auto &cpd = primitive_desc->convolution;
    *aprimitive = new convolution_impl(cpd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}

template <typename convolution_impl> template <typename Impl>
status_t convolution<convolution_impl>::set_default_parameters(
        convolution_desc_t &conv_d, ... ) {
    constexpr precision_t prec = data_trait<data_t<>>::prec;
    const bool with_bias = !memory_desc_wrapper(conv_d.bias_desc).is_zero();
    const bool with_groups = conv_d.weights_desc.tensor_desc.ndims
        == (conv_d.src_desc.tensor_desc.ndims + 1);

    if (conv_d.src_desc.format == any)
        CHECK(types::set_default_format<prec>(conv_d.src_desc, nchw));
    if (conv_d.weights_desc.format == any)
        CHECK(types::set_default_format<prec>(conv_d.weights_desc,
                    with_groups ? goihw : oihw));
    if (with_bias && conv_d.bias_desc.format == any)
        CHECK(types::set_default_format<prec>(conv_d.bias_desc, x));
    if (conv_d.dst_desc.format == any)
        CHECK(types::set_default_format<prec>(conv_d.dst_desc,
                    conv_d.src_desc.format));

    return success;
}

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
