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

#ifndef BATCH_NORMALIZATION_HPP
#define BATCH_NORMALIZATION_HPP

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

template <typename batch_normalization_impl>
class batch_normalization: public primitive {
public:
    batch_normalization(const batch_normalization_primitive_desc_t &pd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(pd, const_cast<impl::engine *>(pd.base.engine), not_ready)
        , _bnpd(_primitive_desc.batch_normalization)
        , _is_training(_bnpd.batch_normalization_desc.prop_kind ==
                       prop_kind::forward_training) {
        for (int i = 0; i < 2 + _is_training; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
    }
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkldnn::impl::engine &aengine);

protected:
    const impl::batch_normalization_primitive_desc_t &_bnpd;
    const bool _is_training;

    virtual status_t execute_impl() {
        switch (_bnpd.batch_normalization_desc.prop_kind) {
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

    template <typename Impl>
    static status_t set_default_parameters(batch_normalization_desc_t &bnd,...);
    template <typename Impl>
    static status_t constraints(batch_normalization_desc_t &bnd,...);

    template <typename Impl>
    static memory_desc_t get_workspace(
        const batch_normalization_desc_t &bnd,...) {
        if (bnd.prop_kind == prop_kind::forward_scoring) {
            return types::zero<memory_desc_t>();
        } else {
            memory_desc_t _desc;
            tensor_desc_t _tensor = { 1, bnd.dst_desc.tensor_desc.dims[1] * 4 };
            mkldnn_memory_desc_init(&_desc, &_tensor,
                bnd.dst_desc.precision, mkldnn::impl::memory_format::x);
            return _desc;
        }
    }

private:
    template <typename Impl = batch_normalization_impl>
    using data_t = typename Impl::data_t;

    virtual status_t execute_forward() { return unimplemented; }
    virtual status_t execute_backward_data() { return unimplemented; }

    template <typename Impl>
    static status_t set_default_parameters(batch_normalization_desc_t &bnd,
            decltype(&Impl::set_default_parameters) _ = 0)
    { return Impl::set_default_parameters(bnd); }
    template <typename Impl>
    static status_t constraints(batch_normalization_desc_t &bnd,
            decltype(&Impl::constraints) _ = 0)
    { return Impl::constraints(bnd); }
    template <typename Impl>
    static memory_desc_t get_workspace(const batch_normalization_desc_t &bnd,
            decltype(&Impl::get_workspace))
    { return Impl::get_workspace(bnd); }
};

/* implementation */

template <typename batch_normalization_impl>
status_t batch_normalization<batch_normalization_impl>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkldnn::impl::engine &aengine) {
    using namespace prop_kind;
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    auto bnd = op_desc.batch_normalization;

    bool args_ok = everyone_is(prec, bnd.src_desc.precision,
            bnd.dst_desc.precision)
        && op_desc._kind == primitive_kind::batch_normalization
        && one_of(bnd.prop_kind, forward_training, forward_scoring,
                backward_data);
    if (!args_ok) return invalid_arguments;

    status_t status;
    status = set_default_parameters<batch_normalization_impl>(bnd, 0);
    if (status != success) return status;

    status = constraints<batch_normalization_impl>(bnd, 0);
    if (status != success) return status;

    memory_desc_t workspace_desc = get_workspace<
                                            batch_normalization_impl>(bnd, 0);

    /* final stage */
    batch_normalization_primitive_desc_t bnpd = {};
    bnpd.base.primitive_kind = primitive_kind::batch_normalization;
    bnpd.base.engine = &aengine;
    bnpd.base.implementation = reinterpret_cast<const void*>(
            &batch_normalization_impl::implementation);
    bnpd.batch_normalization_desc = bnd;

    CHECK(mkldnn_memory_primitive_desc_init(&bnpd.src_primitive_desc,
                &bnd.src_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&bnpd.dst_primitive_desc,
                &bnd.dst_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&bnpd.scaleshift_primitive_desc,
                &bnd.scaleshift_desc, &aengine));
    CHECK(mkldnn_memory_primitive_desc_init(&bnpd.workspace_primitive_desc,
                &workspace_desc, &aengine));

    primitive_desc->batch_normalization = bnpd;
    return success;
}

template <typename batch_normalization_impl>
status_t batch_normalization<batch_normalization_impl>::create(
        primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind ==
            primitive_kind::batch_normalization);
    auto &bnpd = primitive_desc->batch_normalization;
    *aprimitive = new batch_normalization_impl(bnpd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}

template <typename batch_normalization_impl> template <typename Impl>
status_t batch_normalization<batch_normalization_impl>::set_default_parameters(
        batch_normalization_desc_t &bnd, ... ) {
    constexpr precision_t prec = data_trait<data_t<>>::prec;

    if (bnd.src_desc.format == any)
        CHECK(types::set_default_format<prec>(bnd.src_desc, nchw));
    if (bnd.dst_desc.format == any)
        CHECK(types::set_default_format<prec>(bnd.dst_desc, bnd.src_desc.format));
    if (bnd.scaleshift_desc.format == any)
        CHECK(types::set_default_format<prec>(bnd.src_desc, mkldnn_nc));

    return success;
}

template <typename batch_normalization_impl> template <typename Impl>
status_t batch_normalization<batch_normalization_impl>::constraints(
        batch_normalization_desc_t &bnd, ... ) {
    if (bnd.scaleshift_desc.format !=  mkldnn_nc)
        return mkldnn_invalid_arguments;
    else
        return success;
}

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
