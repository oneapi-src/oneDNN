/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_GPU_PRIMITIVE_HPP
#define GPU_GPU_PRIMITIVE_HPP

#include "common/cache_blob.hpp"
#include "common/primitive.hpp"
#include "common/primitive_exec_types.hpp"

#include "gpu/gpu_resource.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct primitive_t : public impl::primitive_t {
    using impl::primitive_t::primitive_t;

    struct compute_block_t {
        compute_block_t(impl::primitive_t *primitive) : primitive_(primitive) {}
        virtual ~compute_block_t() = default;

        status_t get_cache_blob_size(
                impl::engine_t *engine, size_t *size) const {
            if (primitive_)
                return primitive_->get_cache_blob_size(engine, size);
            return get_cache_blob_size_impl(engine, size);
        }

        status_t get_cache_blob(
                impl::engine_t *engine, cache_blob_t &blob) const {
            if (primitive_) return primitive_->get_cache_blob(engine, blob);
            return get_cache_blob_impl(engine, blob);
        }

        bool empty() const { return empty_impl(); }

        const impl::primitive_t *primitive() const { return primitive_; }

    private:
        virtual bool empty_impl() const { return !bool(primitive_); }

        virtual status_t get_cache_blob_size_impl(
                impl::engine_t *engine, size_t *size) const {
            assert(!"unexpected");
            return status::runtime_error;
        }
        virtual status_t get_cache_blob_impl(
                impl::engine_t *engine, cache_blob_t &blob) const {
            assert(!"unexpected");
            return status::runtime_error;
        }

        // "primitive" is a common compute block for all vendors and kernel
        // languages.
        impl::primitive_t *primitive_;
    };

    status_t create_nested_primitive(
            std::shared_ptr<impl::primitive_t> &primitive,
            const std::shared_ptr<primitive_desc_t> &pd,
            impl::engine_t *engine) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
        CHECK(pd->create_primitive_nested(p, engine, cache_blob()));

        if (p.second == cache_state_t::kernel_hit) {
            creation_cached_state_ = cache_state_t::nested_primitive_hit;
        }
        primitive = p.first;
        register_compute_block(new compute_block_t(primitive.get()));
        return status::success;
    }

    status_t get_cache_blob_size(
            impl::engine_t *engine, size_t *size) const override {
        if (!size) return status::invalid_arguments;
        // Query binary size for each created kernel.
        for (const auto &cb : compute_blocks()) {
            if (cb->empty()) continue;
            CHECK(cb->get_cache_blob_size(engine, size));
        }
        return status::success;
    }

    status_t get_cache_blob(
            impl::engine_t *engine, cache_blob_t &blob) const override {
        for (const auto &cb : compute_blocks()) {
            if (cb->empty()) continue;
            CHECK(cb->get_cache_blob(engine, blob));
        }
        return status::success;
    }

    status_t create_resource(
            impl::engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;
        auto r = utils::make_unique<gpu_resource_t>();
        if (!r) return status::out_of_memory;
        CHECK(init_res_storage(engine, r.get()));
        mapper.add(this, std::move(r));

        for (const auto &cb : compute_blocks()) {
            if (cb->empty()) continue;
            // Check that the compute block is a "primitive".
            if (cb->primitive())
                CHECK(cb->primitive()->create_resource(engine, mapper));
        }
        return status::success;
    }

protected:
    virtual status_t init_res_storage(
            impl::engine_t *engine, gpu_resource_t *r) const {
        return status::success;
    }

    void register_compute_block(compute_block_t *cb) {
        compute_blocks_.emplace_back(cb);
    }

    const std::vector<std::unique_ptr<compute_block_t>> &
    compute_blocks() const {
        return compute_blocks_;
    }

private:
    void register_primitive(impl::primitive_t *primitive) {
        compute_blocks_.emplace_back(new compute_block_t(primitive));
    }

    std::vector<std::unique_ptr<compute_block_t>> compute_blocks_;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
