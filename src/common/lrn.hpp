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

#ifndef LRN_HPP
#define LRN_HPP

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

template <typename lrn_impl>
class lrn: public primitive {
public:
    lrn(const lrn_primitive_desc_t &pd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(pd, const_cast<impl::engine *>(pd.base.engine), not_ready)
        , _lpd(_primitive_desc.lrn)
        , _is_training(_lpd.lrn_desc.prop_kind == prop_kind::forward_training)
    {
        for (int i = 0; i < 1 + _is_training; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
    }

    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkldnn::impl::engine &aengine);

protected:
    const impl::lrn_primitive_desc_t &_lpd;
    const bool _is_training;

    virtual status_t execute_impl() {
        switch (_lpd.lrn_desc.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_scoring:
            return execute_forward();
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
    static status_t set_default_parameters(lrn_desc_t &lrn_d, ...);
    template <typename Impl>
    static status_t constraint(const lrn_desc_t &lrn_d, ...)
    { return success; }
    template <typename Impl>
    static memory_desc_t get_scratch(const lrn_desc_t &lrn_d, ...) {
        return lrn_d.prop_kind == prop_kind::forward_scoring
            ? types::zero<memory_desc_t>() : lrn_d.dst_desc;
    }

private:
    template <typename Impl = lrn_impl>
    using data_t = typename Impl::data_t;

    virtual status_t execute_forward() { return unimplemented; }
    virtual status_t execute_backward_data() { return unimplemented; }

    template <typename Impl>
    static status_t set_default_parameters(lrn_desc_t &lrn_d,
            decltype(&Impl::set_default_parameters) _ = 0)
    { return Impl::set_default_parameters(lrn_d); }

    template <typename Impl>
    static status_t constraint(const lrn_desc_t &lrn_d,
            decltype(&Impl::constraint))
    { return Impl::constraint(lrn_d); }
    template <typename Impl>
    static memory_desc_t get_scratch(const lrn_desc_t &lrn_d,
            decltype(&Impl::get_scratch))
    { return Impl::get_scratch(lrn_d); }
};

/* implementation */

template <typename lrn_impl>
status_t lrn<lrn_impl>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkldnn::impl::engine &aengine) {
    using namespace prop_kind;
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    auto lrn_d = op_desc.lrn;

    bool args_ok = everyone_is(prec, lrn_d.src_desc.precision,
            lrn_d.dst_desc.precision)
        && op_desc._kind == primitive_kind::lrn
        && one_of(lrn_d.prop_kind, forward_training, forward_scoring,
                backward_data);
    if (!args_ok) return invalid_arguments;

    status_t status;
    status = set_default_parameters<lrn_impl>(lrn_d, 0);
    if (status != success) return status;

    status = constraint<lrn_impl>(lrn_d, 0);
    if (status != success) return status;

    memory_desc_t scratch_desc = get_scratch<lrn_impl>(lrn_d, 0);

    /* final stage */
    lrn_primitive_desc_t lpd = {};
    lpd.base.primitive_kind = primitive_kind::lrn;
    lpd.base.engine = &aengine;
    lpd.base.implementation = reinterpret_cast<const void*>(
            &lrn_impl::implementation);
    lpd.lrn_desc = lrn_d;
    CHECK(mkldnn_memory_primitive_desc_init(&lpd.src_primitive_desc,
                &lrn_d.src_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&lpd.scratch_primitive_desc,
                &scratch_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&lpd.dst_primitive_desc,
                &lrn_d.dst_desc, &aengine));

    primitive_desc->lrn = lpd;
    return success;
}

template <typename lrn_impl>
status_t lrn<lrn_impl>::create(
        primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == primitive_kind::lrn);
    auto &lpd = primitive_desc->lrn;
    *aprimitive = new lrn_impl(lpd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}

template <typename lrn_impl> template <typename Impl>
status_t lrn<lrn_impl>::set_default_parameters(
        lrn_desc_t &lrn_d, ... ) {
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    if (lrn_d.src_desc.format == any)
        CHECK(types::set_default_format<prec>(lrn_d.src_desc, nchw));
    if (lrn_d.dst_desc.format == any)
        CHECK(types::set_default_format<prec>(lrn_d.dst_desc,
                    lrn_d.src_desc.format));

    return success;
}

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
