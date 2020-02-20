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

#ifndef PRIMITIVE_IMPL_HPP
#define PRIMITIVE_IMPL_HPP

#include <assert.h>

#include "dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "primitive_exec_types.hpp"

namespace dnnl {
namespace impl {

struct primitive_impl_t : public c_compatible {
    primitive_impl_t(const primitive_desc_t *pd) : pd_(pd->clone()) {}
    virtual ~primitive_impl_t() { delete pd_; }

    virtual status_t init() { return status::success; }
    engine_t *engine() const { return pd_->engine(); }
    const primitive_desc_t *pd() const { return pd_; }
    primitive_kind_t kind() const { return pd_->kind(); }
    virtual status_t execute(const exec_ctx_t &ctx) const = 0;

protected:
    const primitive_desc_t *pd_;

private:
    primitive_impl_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(primitive_impl_t);
};

} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
