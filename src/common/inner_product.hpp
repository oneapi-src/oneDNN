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

#ifndef INNER_PRODUCT_HPP
#define INNER_PRODUCT_HPP

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

template <typename inner_product_impl>
class inner_product: public primitive {
public:
    inner_product(const inner_product_primitive_desc_t &pd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(pd, const_cast<impl::engine *>(pd.base.engine), not_ready)
        , _ippd(_primitive_desc.inner_product)
        , _is_training(_ippd.inner_product_desc.prop_kind
                == prop_kind::forward_training)
        , _with_bias(!memory_desc_wrapper(_ippd.bias_primitive_desc).is_zero())
    {
        for (int i = 0; i < 2 + _with_bias; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
    }

    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkldnn::impl::engine &aengine);

protected:
    const impl::inner_product_primitive_desc_t &_ippd;
    const bool _is_training;
    const bool _with_bias;

    virtual status_t execute_impl() {
        switch (_ippd.inner_product_desc.prop_kind) {
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
    static status_t set_default_parameters(inner_product_desc_t &ip_d, ...);
    template <typename Impl>
    static status_t constraint(const inner_product_desc_t &ip_d, ...)
    { return success; }

private:
    template <typename Impl = inner_product_impl>
    using data_t = typename Impl::data_t;

    virtual status_t execute_forward() { return unimplemented; }
    virtual status_t execute_backward_data() { return unimplemented; }
    virtual status_t execute_backward_weights() { return unimplemented; }
    virtual status_t execute_backward_bias() { return unimplemented; }

    template <typename Impl>
    static status_t set_default_parameters(inner_product_desc_t &ip_d,
            decltype(&Impl::set_default_parameters) _ = 0)
    { return Impl::set_default_parameters(ip_d); }

    template <typename Impl>
    static status_t constraint(const inner_product_desc_t &ip_d,
            decltype(&Impl::constraint))
    { return Impl::constraint(ip_d); }
};

/* implementation */

template <typename inner_product_impl>
status_t inner_product<inner_product_impl>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkldnn::impl::engine &aengine) {
    using namespace prop_kind;
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    auto ip_d = op_desc.inner_product;

    bool args_ok = everyone_is(prec, ip_d.src_desc.precision,
            ip_d.weights_desc.precision,
            ip_d.bias_desc.format == memory_format::undef ? prec
            : ip_d.bias_desc.precision, ip_d.dst_desc.precision)
        && op_desc._kind == primitive_kind::inner_product
        && one_of(ip_d.prop_kind, forward_training, forward_scoring,
                backward_data, backward_weights, backward_bias);
    if (!args_ok) return invalid_arguments;

    status_t status;
    status = set_default_parameters<inner_product_impl>(ip_d, 0);
    if (status != success) return status;

    status = constraint<inner_product_impl>(ip_d, 0);
    if (status != success) return status;

    /* final stage */
    inner_product_primitive_desc_t ippd = {};
    ippd.base.primitive_kind = primitive_kind::inner_product;
    ippd.base.engine = &aengine;
    ippd.base.implementation = reinterpret_cast<const void*>(
            &inner_product_impl::implementation);
    ippd.inner_product_desc = ip_d;
    CHECK(mkldnn_memory_primitive_desc_init(&ippd.src_primitive_desc,
                &ip_d.src_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&ippd.weights_primitive_desc,
                &ip_d.weights_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&ippd.bias_primitive_desc,
                &ip_d.bias_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&ippd.dst_primitive_desc,
                &ip_d.dst_desc, &aengine));

    primitive_desc->inner_product = ippd;
    return success;
}

template <typename inner_product_impl>
status_t inner_product<inner_product_impl>::create(
        primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind ==
            primitive_kind::inner_product);
    auto &ippd = primitive_desc->inner_product;
    *aprimitive = new inner_product_impl(ippd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}

template <typename inner_product_impl> template <typename Impl>
status_t inner_product<inner_product_impl>::set_default_parameters(
        inner_product_desc_t &ip_d, ... ) {
    constexpr precision_t prec = data_trait<data_t<>>::prec;
    const bool with_bias = !memory_desc_wrapper(ip_d.bias_desc).is_zero();

    if (ip_d.src_desc.format == any) {
        if (ip_d.src_desc.tensor_desc.ndims == 4)
            CHECK(types::set_default_format<prec>(ip_d.src_desc, nchw));
        else if (ip_d.src_desc.tensor_desc.ndims == 2)
            CHECK(types::set_default_format<prec>(ip_d.src_desc, nc));
        else
            return unimplemented;
    }
    if (ip_d.weights_desc.format == any) {
        if (ip_d.weights_desc.tensor_desc.ndims == 4)
            CHECK(types::set_default_format<prec>(ip_d.weights_desc, oihw));
        else if (ip_d.src_desc.tensor_desc.ndims == 2)
            CHECK(types::set_default_format<prec>(ip_d.weights_desc, oi));
        else
            return unimplemented;
    }
    if (with_bias && ip_d.bias_desc.format == any)
        CHECK(types::set_default_format<prec>(ip_d.bias_desc, x));
    if (ip_d.dst_desc.format == any)
        CHECK(types::set_default_format<prec>(ip_d.dst_desc, nc));

    return success;
}

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
