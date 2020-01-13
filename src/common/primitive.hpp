/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef PRIMITIVE_HPP
#define PRIMITIVE_HPP

#include <assert.h>

#include "dnnl.h"

#include "c_types_map.hpp"
#include "memory_storage.hpp"
#include "memory_tracking.hpp"
#include "primitive_desc.hpp"
#include "primitive_exec_types.hpp"
#include "scratchpad.hpp"

#include <type_traits>

namespace dnnl {
namespace impl {

struct primitive_t : public c_compatible {
    primitive_t(const primitive_desc_t *pd) : pd_(pd->clone()) {}
    virtual ~primitive_t() = default;

    virtual status_t init(engine_t *engine) { return status::success; }
    std::shared_ptr<primitive_desc_t> pd() const { return pd_; }
    primitive_kind_t kind() const { return pd_->kind(); }
    virtual status_t execute(const exec_ctx_t &ctx) const = 0;

protected:
    std::shared_ptr<primitive_desc_t> pd_;

private:
    primitive_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(primitive_t);
};

} // namespace impl
} // namespace dnnl

#define ARG_TYPE(t) \
    typename std::remove_cv<typename std::remove_pointer<t>::type>::type

#define CTX_IN_MEM(type, arg) \
    static_cast<const ARG_TYPE(type) *>(CTX_IN_STORAGE(arg).data_handle())

#define CTX_OUT_MEM(type, arg) \
    static_cast<ARG_TYPE(type) *>(CTX_OUT_STORAGE(arg).data_handle())

struct dnnl_primitive : public dnnl::impl::c_compatible {
    dnnl_primitive(const std::shared_ptr<dnnl::impl::primitive_t> &primitive,
            dnnl::impl::engine_t *engine, bool use_global_scratchpad = false);

    dnnl::impl::engine_t *engine() const;
    const dnnl::impl::primitive_desc_iface_t *pd() const;
    const std::shared_ptr<dnnl::impl::primitive_t> &get_primitive() const;
    dnnl::impl::status_t execute(dnnl::impl::exec_ctx_t &ctx) const;

private:
    std::shared_ptr<dnnl::impl::primitive_t> primitive_;
    std::unique_ptr<dnnl::impl::scratchpad_t> scratchpad_;
    std::unique_ptr<dnnl::impl::primitive_desc_iface_t> pd_;

    dnnl_primitive() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(dnnl_primitive);
};

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
