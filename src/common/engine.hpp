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

#ifndef ENGINE_HPP
#define ENGINE_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "utils.hpp"
#include "primitive.hpp"
#include "reorder.hpp"

struct mkldnn_engine: public mkldnn::impl::c_compatible {
protected:
    mkldnn::impl::engine_kind_t _kind;
public:
    mkldnn_engine(): _kind(mkldnn::impl::engine_kind::any_engine) {}
    mkldnn_engine(mkldnn::impl::engine_kind_t kind): _kind(kind) {}
    virtual ~mkldnn_engine() {}

    virtual bool is_lazy() const = 0;
    virtual bool is_ok() const = 0;
    mkldnn::impl::engine_kind_t kind() const { return _kind; }

    virtual mkldnn::impl::status_t submit(size_t n,
            mkldnn::impl::primitive *primitives[],
            mkldnn::impl::primitive **error_primitive) = 0;

    /* primitives' descriptor initializators
     * the default one guarantees to return at least an empty list,
     * so no need to check the return value on NULL */
    virtual mkldnn::impl::primitive_desc_init_f *get_primitive_inits() const;
    virtual mkldnn::impl::reorder_primitive_desc_init_f *get_reorder_inits() const;
};

namespace mkldnn { namespace impl {

class engine_factory: public c_compatible {
public:
    virtual size_t count() = 0;
    virtual engine_kind_t kind() = 0;
    virtual status_t engine_create(engine **engine, size_t index) = 0;
};

}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
