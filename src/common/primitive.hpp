/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "primitive_exec_types.hpp"
#include "primitive_impl.hpp"
#include "scratchpad.hpp"

#include <type_traits>

#define ARG_TYPE(t) \
    typename std::remove_cv<typename std::remove_pointer<t>::type>::type

#define CTX_IN_MEM(type, arg) \
    static_cast<const ARG_TYPE(type) *>(CTX_IN_STORAGE(arg).data_handle())

#define CTX_OUT_MEM(type, arg) \
    static_cast<ARG_TYPE(type) *>(CTX_OUT_STORAGE(arg).data_handle())

struct mkldnn_primitive : public mkldnn::impl::c_compatible {
    mkldnn_primitive(const std::shared_ptr<mkldnn::impl::primitive_impl_t>
                             &primitive_impl,
            bool use_global_scratchpad);

    mkldnn::impl::status_t init();
    mkldnn::impl::engine_t *engine() const;
    const mkldnn::impl::primitive_desc_t *pd() const;
    const std::shared_ptr<mkldnn::impl::primitive_impl_t> &
    get_primitive_impl() const;
    mkldnn::impl::status_t execute(mkldnn::impl::exec_ctx_t &ctx) const;

    ~mkldnn_primitive();

private:
    std::shared_ptr<mkldnn::impl::primitive_impl_t> primitive_impl_;
    void *scratchpad_buffer_;
    mkldnn::impl::scratchpad_t *global_scratchpad_;

    mkldnn_primitive() = delete;
    MKLDNN_DISALLOW_COPY_AND_ASSIGN(mkldnn_primitive);
};

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
