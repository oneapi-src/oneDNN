/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef XPU_SYCL_ENGINE_ID_HPP
#define XPU_SYCL_ENGINE_ID_HPP

#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

struct engine_id_impl_t : public impl::engine_id_impl_t {

    engine_id_impl_t(const ::sycl::device &device,
            const ::sycl::context &context, engine_kind_t kind,
            runtime_kind_t runtime_kind, size_t index)
        : impl::engine_id_impl_t(kind, runtime_kind, index)
        , device_(device)
        , context_(context) {}

    ~engine_id_impl_t() override = default;

private:
    bool compare_resource(
            const impl::engine_id_impl_t *id_impl) const override {
        const auto *typed_id
                = utils::downcast<const engine_id_impl_t *>(id_impl);
        return device_ == typed_id->device_ && context_ == typed_id->context_;
    }

    size_t hash_resource() const override {
        size_t seed = 0;
        seed = hash_combine(seed, device_);
        seed = hash_combine(seed, context_);
        return seed;
    }

    ::sycl::device device_;
    ::sycl::context context_;
};

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
