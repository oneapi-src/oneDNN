/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef REORDER_PRIMITIVE_DESC_HPP
#define REORDER_PRIMITIVE_DESC_HPP

#include "dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

struct reorder_primitive_desc_iface_t : public dnnl_primitive_desc {
    reorder_primitive_desc_iface_t(dnnl::impl::primitive_desc_t *pd,
            dnnl::impl::engine_t *engine, dnnl::impl::engine_t *src_engine,
            dnnl::impl::engine_t *dst_engine)
        : dnnl_primitive_desc(pd, engine)
        , src_engine_(src_engine)
        , dst_engine_(dst_engine)
        , scratchpad_engine_(nullptr) {}

    dnnl::impl::engine_t *src_engine() const override { return src_engine_; }
    dnnl::impl::engine_t *dst_engine() const override { return dst_engine_; }

    dnnl::impl::engine_t *scratchpad_engine() const override {
        return scratchpad_engine_;
    }

    dnnl::impl::status_t query(
            dnnl::impl::query_t what, int idx, void *result) const override {
        auto status = dnnl::impl::status::success;
        switch (what) {
            case dnnl::impl::query::reorder_src_engine:
                *(dnnl::impl::engine_t **)result = src_engine();
                break;
            case dnnl::impl::query::reorder_dst_engine:
                *(dnnl::impl::engine_t **)result = dst_engine();
                break;
            default: status = dnnl_primitive_desc::query(what, idx, result);
        }
        return status;
    }

private:
    dnnl::impl::engine_t *src_engine_;
    dnnl::impl::engine_t *dst_engine_;
    dnnl::impl::engine_t *scratchpad_engine_;
};
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
