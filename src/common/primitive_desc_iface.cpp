/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"

#include "engine.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_iface.hpp"
#include "primitive_iterator.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;

dnnl_primitive_desc::dnnl_primitive_desc(
        const std::shared_ptr<primitive_desc_t> &pd, engine_t *engine)
    : pd_(pd), engine_(engine) {}

status_t dnnl_primitive_desc::create_primitive_iface(
        std::pair<primitive_iface_t *, bool> &primitive_iface,
        const cache_blob_t &cache_blob) const {
    // Step 1: create impl::primitive_t or get it from primitive cache
    std::pair<std::shared_ptr<primitive_t>, bool> p;
    auto status = pd_->create_primitive(p, engine(), cache_blob);
    if (status != status::success) return status;
    // Step 2: create primitive_iface_t, init and return it to user
    primitive_iface_t *p_iface = nullptr;
    CHECK(safe_ptr_assign(p_iface, new primitive_iface_t(p.first, engine())));
    status = p_iface->init();
    if (status != status::success) {
        p_iface->release();
        return status;
    }
    primitive_iface = std::make_pair(p_iface, p.second);
    return status::success;
}

const std::shared_ptr<primitive_desc_t> &dnnl_primitive_desc::impl() const {
    return pd_;
}

dnnl::impl::engine_t *dnnl_primitive_desc::engine() const {
    return engine_;
}
const dnnl::impl::primitive_attr_t *dnnl_primitive_desc::attr() const {
    return pd_->attr();
}

const char *dnnl_primitive_desc::info() const {
    return pd_->info(engine_);
}

dnnl::impl::engine_t *dnnl_primitive_desc::src_engine() const {
    return engine_;
}
dnnl::impl::engine_t *dnnl_primitive_desc::dst_engine() const {
    return engine_;
}

dnnl::impl::engine_t *dnnl_primitive_desc::scratchpad_engine() const {
    return engine_;
}

status_t dnnl_primitive_desc::query(query_t what, int idx, void *result) const {
    auto status = status::success;
    switch (what) {
        case query::engine: *(engine_t **)result = engine(); break;
        case query::cache_blob_id_size_s64:
            *(dim_t *)result = (dim_t)pd_->get_cache_blob_id(engine()).size();
            break;
        case query::cache_blob_id:
            *(const uint8_t **)result = pd_->get_cache_blob_id(engine()).empty()
                    ? nullptr
                    : pd_->get_cache_blob_id(engine()).data();
            break;

        default: status = pd_->query(what, idx, result);
    }
    return status;
}

status_t dnnl_primitive_desc_get_attr(
        const primitive_desc_iface_t *primitive_desc_iface,
        const primitive_attr_t **attr) {
    if (utils::any_null(primitive_desc_iface, attr)) return invalid_arguments;

    *attr = primitive_desc_iface->attr();
    return success;
}

status_t dnnl_primitive_desc_clone(
        primitive_desc_iface_t **primitive_desc_iface,
        const primitive_desc_iface_t *existing_primitive_desc_iface) {
    if (utils::any_null(primitive_desc_iface, existing_primitive_desc_iface))
        return invalid_arguments;

    return safe_ptr_assign(*primitive_desc_iface,
            new primitive_desc_iface_t(existing_primitive_desc_iface->impl(),
                    existing_primitive_desc_iface->engine()));
}

status_t dnnl_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface,
        const_c_op_desc_t c_op_desc, const primitive_attr_t *attr,
        engine_t *engine, const primitive_desc_iface_t *hint_fwd_pd) {
    primitive_desc_iterator_t *it;
    status_t status = dnnl_primitive_desc_iterator_create(
            &it, c_op_desc, attr, engine, hint_fwd_pd);
    if (status != status::success) return status;

    primitive_desc_iface_t *pd_iface
            = new primitive_desc_iface_t(*(*it), engine);
    dnnl_primitive_desc_iterator_destroy(it);
    if (pd_iface == nullptr) return out_of_memory;

    *primitive_desc_iface = pd_iface;

    return success;
}
