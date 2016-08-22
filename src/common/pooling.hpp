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

#ifndef POOLING_HPP
#define POOLING_HPP

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

template <typename pooling_impl>
class pooling: public primitive {
public:
    pooling(const pooling_primitive_desc_t &pd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(pd, const_cast<impl::engine *>(pd.base.engine), not_ready)
        , _ppd(_primitive_desc.pooling)
        , _is_training(_ppd.pooling_desc.prop_kind
                == prop_kind::forward_training)
    {
        for (int i = 0; i < 1 + _is_training; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
    }

    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkldnn::impl::engine &aengine);

protected:
    const impl::pooling_primitive_desc_t &_ppd;
    const bool _is_training;

    virtual status_t execute_impl() {
        switch (_ppd.pooling_desc.prop_kind) {
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
    static status_t set_default_parameters(pooling_desc_t &pool_d, ...);
    template <typename Impl>
    static status_t constraint(const pooling_desc_t &pool_d, ...)
    { return success; }
    template <typename Impl>
    static memory_desc_t get_indices_desc(const pooling_desc_t &pool_d, ...) {
        if (pool_d.prop_kind == prop_kind::forward_scoring)
            return types::zero<memory_desc_t>();
        auto indices_desc = pool_d.dst_desc;
        indices_desc.precision = data_trait<index_t<>>::prec;
        return indices_desc;
    }

private:
    template <typename Impl = pooling_impl>
    using data_t = typename Impl::data_t;
    template <typename Impl = pooling_impl>
    using index_t = typename Impl::index_t;

    virtual status_t execute_forward() { return unimplemented; }
    virtual status_t execute_backward_data() { return unimplemented; }

    template <typename Impl>
    static status_t set_default_parameters(pooling_desc_t &pool_d,
            decltype(&Impl::set_default_parameters) _ = 0)
    { return Impl::set_default_parameters(pool_d); }

    template <typename Impl>
    static status_t constraint(const pooling_desc_t &pool_d,
            decltype(&Impl::constraint))
    { return Impl::constraint(pool_d); }
    template <typename Impl>
    static memory_desc_t get_indices_desc(const pooling_desc_t &pool_d,
            decltype(&Impl::get_indices_desc))
    { return Impl::get_indices_desc(pool_d); }
};

/* implementation */

template <typename pooling_impl>
status_t pooling<pooling_impl>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkldnn::impl::engine &aengine) {
    using namespace prop_kind;
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    auto pool_d = op_desc.pooling;

    bool args_ok = everyone_is(prec, pool_d.src_desc.precision,
            pool_d.dst_desc.precision)
        && op_desc._kind == primitive_kind::pooling
        && one_of(pool_d.prop_kind, forward_training, forward_scoring,
                backward_data);
    if (!args_ok) return invalid_arguments;

    status_t status;
    status = set_default_parameters<pooling_impl>(pool_d, 0);
    if (status != success) return status;

    status = constraint<pooling_impl>(pool_d, 0);
    if (status != success) return status;

    memory_desc_t indices_desc = get_indices_desc<pooling_impl>(pool_d, 0);

    /* final stage */
    pooling_primitive_desc_t ppd = {};
    ppd.base.primitive_kind = primitive_kind::pooling;
    ppd.base.engine = &aengine;
    ppd.base.implementation = reinterpret_cast<const void*>(
            &pooling_impl::implementation);
    ppd.pooling_desc = pool_d;
    CHECK(mkldnn_memory_primitive_desc_init(&ppd.src_primitive_desc,
                &pool_d.src_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&ppd.indices_primitive_desc,
                &indices_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&ppd.dst_primitive_desc,
                &pool_d.dst_desc, &aengine));

    primitive_desc->pooling = ppd;
    return success;
}

template <typename pooling_impl>
status_t pooling<pooling_impl>::create(
        primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == primitive_kind::pooling);
    auto &ppd = primitive_desc->pooling;
    *aprimitive = new pooling_impl(ppd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}

template <typename pooling_impl> template <typename Impl>
status_t pooling<pooling_impl>::set_default_parameters(
        pooling_desc_t &pool_d, ... ) {
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    if (pool_d.src_desc.format == any)
        CHECK(types::set_default_format<prec>(pool_d.src_desc, nchw));
    if (pool_d.dst_desc.format == any)
        CHECK(types::set_default_format<prec>(pool_d.dst_desc,
                    pool_d.src_desc.format));

    return success;
}

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
