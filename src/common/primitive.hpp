/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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
#include <atomic>

#include "dnnl.h"

#include "c_types_map.hpp"
#include "memory_storage.hpp"
#include "memory_tracking.hpp"
#include "primitive_exec_types.hpp"
#include "primitive_impl.hpp"
#include "scratchpad.hpp"

#include <type_traits>

#define ARG_TYPE(t) \
    typename std::remove_cv<typename std::remove_pointer<t>::type>::type

#define CTX_IN_MEM(type, arg) \
    static_cast<const ARG_TYPE(type) *>(ctx.host_ptr(arg))

#define CTX_OUT_MEM(type, arg) static_cast<ARG_TYPE(type) *>(ctx.host_ptr(arg))

namespace dnnl {
namespace impl {

status_t primitive_execute(const primitive_t *primitive, exec_ctx_t &ctx);

}
} // namespace dnnl

/** \brief A pure virtual primitive class
 *
 * Primitive contains links to its inputs & outputs, though it does not track
 * their readiness on execution step.
 *
 * @remark @b Rational.
 *   Dependencies are essential through-out the whole MKL-DNN library, so it
 *   makes sense to include them on the very low level. On the other hand,
 *   tracking them should be a task for corresponding essence, like scheduler,
 *   stream or whatever. Primitive itself should know nothing about the
 *   environment it is running in.
 *
 * @note
 *   To make user experience better we should provide API which allows
 *   achieving the best (or good enough) performance when creating primitives
 *   in natural order: i.e. from bottom to top for forward pass and from top to
 *   bottom for backward pass. Please consider restriction [1] in Level 0.
 */
struct dnnl_primitive : public dnnl::impl::c_compatible {
    dnnl_primitive(
            const std::shared_ptr<dnnl::impl::primitive_impl_t> &primitive_impl,
            bool use_global_scratchpad);

    dnnl::impl::status_t init();
    dnnl::impl::engine_t *engine() const;
    const dnnl::impl::primitive_desc_t *pd() const;
    const std::shared_ptr<dnnl::impl::primitive_impl_t> &
    get_primitive_impl() const;
    dnnl::impl::status_t execute(dnnl::impl::exec_ctx_t &ctx) const;

    void retain() { counter_++; }

    void release() {
        if (--counter_ == 0) { delete this; }
    }

protected:
    ~dnnl_primitive() = default;

private:
    std::atomic<int> counter_;
    std::shared_ptr<dnnl::impl::primitive_impl_t> primitive_impl_;
    std::unique_ptr<dnnl::impl::scratchpad_t> scratchpad_;

    dnnl_primitive() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(dnnl_primitive);
};

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
