/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef GPU_SYCL_SYCL_GPU_ENGINE_HPP
#define GPU_SYCL_SYCL_GPU_ENGINE_HPP

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_engine_base.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

class sycl_gpu_engine_t : public impl::sycl::sycl_engine_base_t {
public:
    using sycl_engine_base_t::context;
    using sycl_engine_base_t::create_stream;
    using sycl_engine_base_t::device;

    sycl_gpu_engine_t(
            const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
        : sycl_engine_base_t(engine_kind::gpu, dev, ctx, index) {
        assert(dev.is_gpu());
    }

    status_t create_memory_storage(memory_storage_t **storage, unsigned flags,
            size_t size, void *handle) override {
        return sycl_engine_base_t::create_memory_storage(
                storage, flags, size, handle);
    }

    status_t create_stream(stream_t **stream, unsigned flags) override {
        return sycl_engine_base_t::create_stream(stream, flags);
    }

    const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return gpu::gpu_impl_list_t::get_reorder_implementation_list(
                src_md, dst_md);
    }

    const impl_list_item_t *get_concat_implementation_list() const override {
        return gpu::gpu_impl_list_t::get_concat_implementation_list();
    }

    const impl_list_item_t *get_sum_implementation_list() const override {
        return gpu::gpu_impl_list_t::get_sum_implementation_list();
    }

    const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc) const override {
        return gpu::gpu_impl_list_t::get_implementation_list(desc);
    }

protected:
    ~sycl_gpu_engine_t() override = default;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
