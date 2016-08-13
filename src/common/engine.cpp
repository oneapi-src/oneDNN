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

#include "mkldnn.h"
#include "engine.hpp"
#include "nstl.hpp"

#include "c_types_map.hpp"
#include "../cpu/cpu_engine.hpp"

namespace mkldnn { namespace impl {

// TODO: we need some caching+refcounting mechanism so that an engine could not
// be created twice and is only destroyed when the refcount is 0

// With STL we would've used vector. Alas, we cannot use STL..
engine_factory *engine_factories[] = {
    &cpu::engine_factory,
    &cpu::engine_factory_lazy,
    NULL,
};

static inline engine_factory *get_engine_factory(engine_kind_t kind)
{
    for (engine_factory **ef = engine_factories; *ef; ef++)
        if ((*ef)->kind() == kind)
            return *ef;
    return NULL;
}

}}

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

primitive_desc_init_f *engine::get_primitive_inits() const {
    static primitive_desc_init_f empty_list[] = { nullptr };
    return empty_list;
}

reorder_primitive_desc_init_f *engine::get_reorder_inits() const {
    static reorder_primitive_desc_init_f reorder_empty_list[] = { nullptr };
    return reorder_empty_list;
}

size_t mkldnn_engine_get_count(engine_kind_t kind) {
    engine_factory *ef = get_engine_factory(kind);
    return ef != NULL ? ef->count() : 0;
}

status_t mkldnn_engine_create(engine **engine,
        engine_kind_t kind, size_t index) {
    if (engine == NULL)
        return invalid_arguments;

    engine_factory *ef = get_engine_factory(kind);
    if (ef == NULL || index >= ef->count())
        return invalid_arguments;

    return ef->engine_create(engine, index);
}

status_t mkldnn_engine_get_kind(engine *engine, engine_kind_t *kind) {
    if (engine == NULL || !engine->is_ok())
        return invalid_arguments;
    *kind = engine->kind();
    return success;
}

status_t mkldnn_engine_get_is_lazy(engine *engine, int *is_lazy) {
    if (engine == NULL || !engine->is_ok())
        return invalid_arguments;
    *is_lazy = engine->is_lazy();
    return success;
}

status_t mkldnn_engine_destroy(engine *engine) {
    delete engine;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
