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

#ifndef RELU_HPP
#define RELU_HPP

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

template <typename relu_impl>
class relu: public primitive {
public:
    relu(const relu_primitive_desc_t &pd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(pd, const_cast<impl::engine *>(pd.base.engine), not_ready)
        , _rpd(_primitive_desc.relu)
    {
        _input.push_back(inputs[0]);
        _output.push_back(outputs[0]);
    }

    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkldnn::impl::engine &aengine);

protected:
    const impl::relu_primitive_desc_t &_rpd;

    virtual status_t execute_impl() {
        switch (_rpd.relu_desc.prop_kind) {
        case prop_kind::forward: return execute_forward();
        case prop_kind::backward_data: return execute_backward_data();
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
    static status_t set_default_parameters(relu_desc_t &relu_d, ...);
    template <typename Impl>
    static status_t constraint(const relu_desc_t &relu_d, ...)
    { return success; }
    template <typename Impl>
    static memory_desc_t get_scratch(const relu_desc_t &relu_d, ...)
    { return relu_d.dst_desc; }

private:
    template <typename Impl = relu_impl>
    using data_t = typename Impl::data_t;

    virtual status_t execute_forward() { return unimplemented; }
    virtual status_t execute_backward_data() { return unimplemented; }

    template <typename Impl>
    static status_t set_default_parameters(relu_desc_t &relu_d,
            decltype(&Impl::set_default_parameters) _ = 0)
    { return Impl::set_default_parameters(relu_d); }

    template <typename Impl>
    static status_t constraint(const relu_desc_t &relu_d,
            decltype(&Impl::constraint))
    { return Impl::constraint(relu_d); }
};

/* implementation */

template <typename relu_impl>
status_t relu<relu_impl>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkldnn::impl::engine &aengine) {
    using namespace prop_kind;
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    auto relu_d = op_desc.relu;

    bool args_ok = everyone_is(prec, relu_d.src_desc.precision,
            relu_d.dst_desc.precision)
        && op_desc._kind == primitive_kind::relu
        && one_of(relu_d.prop_kind, forward, backward_data);
    if (!args_ok) return invalid_arguments;

    status_t status;
    status = set_default_parameters<relu_impl>(relu_d, 0);
    if (status != success) return status;

    status = constraint<relu_impl>(relu_d, 0);
    if (status != success) return status;

    /* final stage */
    relu_primitive_desc_t rpd = {};
    rpd.base.primitive_kind = primitive_kind::relu;
    rpd.base.engine = &aengine;
    rpd.base.implementation = reinterpret_cast<const void*>(
            &relu_impl::implementation);
    rpd.relu_desc = relu_d;
    CHECK(mkldnn_memory_primitive_desc_init(&rpd.src_primitive_desc,
                &relu_d.src_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&rpd.dst_primitive_desc,
                &relu_d.dst_desc, &aengine));

    primitive_desc->relu = rpd;
    return success;
}

template <typename relu_impl>
status_t relu<relu_impl>::create(
        primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == primitive_kind::relu);
    auto &rpd = primitive_desc->relu;
    *aprimitive = new relu_impl(rpd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}

template <typename relu_impl> template <typename Impl>
status_t relu<relu_impl>::set_default_parameters(
        relu_desc_t &relu_d, ... ) {
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    if (relu_d.src_desc.format == any)
        CHECK(types::set_default_format<prec>(relu_d.src_desc, nchw));
    if (relu_d.dst_desc.format == any)
        CHECK(types::set_default_format<prec>(relu_d.dst_desc,
                    relu_d.src_desc.format));

    return success;
}

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
