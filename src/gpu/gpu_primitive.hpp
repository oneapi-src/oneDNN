/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include <cassert>

#ifndef DISABLE_VERBOSE
#include <iostream>
#include <sstream>
#include "common/verbose.hpp"
#endif

#include "common/cache_blob.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm_exec_types.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/kernel_cache.hpp"

#define CTX_GPU_RES_STORAGE(arg) \
    (*(ctx.get_resource_mapper() \
                    ->template get<gpu_resource_t>(this) \
                    ->get_memory_storage(arg)))

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_primitive_t : public primitive_t {
    using primitive_t::primitive_t;

    struct compute_block_t {
        enum class kind_t { kernel, primitive };

        compute_block_t(const compute::kernel_t &kernel)
            : kind_(kind_t::kernel), kernel_(kernel), primitive_(nullptr) {}
        compute_block_t(const primitive_t *primitive)
            : kind_(kind_t::primitive), primitive_(primitive) {}

        bool is_kernel() const { return kind_ == kind_t::kernel; }
        bool is_primitive() const { return kind_ == kind_t::primitive; }
        explicit operator bool() const { return kernel_ || primitive_; }

        const primitive_t *primitive() const { return primitive_; }
        compute::kernel_t kernel() const { return kernel_; }
        kind_t kind() const { return kind_; }

    private:
        kind_t kind_;
        compute::kernel_t kernel_;
        const primitive_t *primitive_;
    };

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;
        auto r = utils::make_unique<gpu_resource_t>();
        if (!r) return status::out_of_memory;
        CHECK(init_res_storage(engine, r.get()));
        mapper.add(this, std::move(r));

        for (const auto &cb : compute_blocks()) {
            if (!cb) continue;
            if (cb.kind() == compute_block_t::kind_t::primitive)
                CHECK(cb.primitive()->create_resource(engine, mapper));
        }
        return status::success;
    }

    status_t get_cache_blob_size(
            engine_t *engine, size_t *size) const override {
        if (!size) return status::invalid_arguments;
        if (version_ != -1) (*size) += sizeof(version_);
        // Query binary size for each created kernel.
        for (const auto &cb : compute_blocks()) {
            if (!cb) continue;

            switch (cb.kind()) {
                case compute_block_t::kind_t::kernel: {
                    size_t sz = 0;
                    CHECK(cb.kernel().get_binary_size(engine, &sz));
                    // We need additional sizeof(size_t) bytes to store the size
                    // of the binary when packing.
                    (*size) += sz + sizeof(size_t);
                    break;
                }
                case compute_block_t::kind_t::primitive:
                    CHECK(cb.primitive()->get_cache_blob_size(engine, size));
                    break;
                default: assert(!"unexpected"); return status::runtime_error;
            }
        }
        return status::success;
    }

    status_t get_cache_blob(
            engine_t *engine, cache_blob_t &blob) const override {
        if (version_ != -1)
            CHECK(blob.add_value((const uint8_t *)&version_, sizeof(version_)));
        for (const auto &cb : compute_blocks()) {
            if (!cb) continue;

            switch (cb.kind()) {
                case compute_block_t::kind_t::kernel: {
                    // Get a binary for each kernel within current primitive.
                    compute::binary_t binary;
                    CHECK(cb.kernel().get_binary(engine, binary));
                    CHECK(blob.add_binary(binary.data(), binary.size()));
                    break;
                }
                case compute_block_t::kind_t::primitive:
                    CHECK(cb.primitive()->get_cache_blob(engine, blob));
                    break;
                default: assert(!"unexpected"); return status::runtime_error;
            }
        }
        return status::success;
    }

    status_t create_kernel(engine_t *engine, compute::kernel_t *kernel,
            jit::jit_generator_base *jitter) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        CHECK(compute_engine->create_kernel(kernel, jitter, cache_blob()));
        CHECK(register_kernels({*kernel}));
        return status::success;
    }

    status_t create_kernels(engine_t *engine,
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        CHECK(compute_engine->create_kernels(
                kernels, kernel_names, kernel_ctx, cache_blob()));
        CHECK(register_kernels(*kernels));
        return status::success;
    }

    status_t create_kernel(engine_t *engine, compute::kernel_t *kernel,
            const char *kernel_name, const compute::kernel_ctx_t &kernel_ctx) {

        std::vector<compute::kernel_t> kernels(1);
        auto status
                = create_kernels(engine, &kernels, {kernel_name}, kernel_ctx);
        if (status == status::success) *kernel = kernels[0];
        return status;
    }

    template <typename T>
    status_t create_kernels(engine_t *engine,
            std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names, const T &params) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        if (cache_blob())
            return compute_engine->create_kernels_from_cache_blob(
                    cache_blob(), kernels, kernel_names);

        auto key = std::make_shared<trivial_key_container_t<T>>(
                params, compute_engine->engine_id());
        gpu_assert(key->key.is_valid());

        CHECK(get_cached_kernels<typename trivial_key_t<T>::value_type>(
                std::move(key), engine, kernels, kernel_names));

        CHECK(register_kernels(kernels));

        return status::success;
    }

    template <typename T>
    status_t create_kernel(engine_t *engine, compute::kernel_t &kernel,
            const char *kernel_name, const T &params) {
        std::vector<compute::kernel_t> kernels(1);
        CHECK(create_kernels(engine, kernels, {kernel_name}, params));
        kernel = kernels[0];
        return status::success;
    }

    status_t create_nested_primitive(std::shared_ptr<primitive_t> &primitive,
            const std::shared_ptr<primitive_desc_t> &pd, engine_t *engine) {
        CHECK(pd->create_primitive(primitive, engine, cache_blob()));
        register_primitive(primitive.get());
        return status::success;
    }

    // TODO: use inheritance for exec_ctx_t to get rid of such places...
    static status_t parallel_for(const gemm_exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) {
        auto compute_stream
                = utils::downcast<compute::compute_stream_t *>(ctx.stream());
        return parallel_for(*compute_stream, range, kernel, arg_list,
                compute_stream->ctx().get_deps(),
                compute_stream->ctx().get_deps());
    }

    static status_t parallel_for(const exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) {
        auto compute_stream
                = utils::downcast<compute::compute_stream_t *>(ctx.stream());
        return parallel_for(*compute_stream, range, kernel, arg_list,
                compute_stream->ctx().get_deps(),
                compute_stream->ctx().get_deps());
    }

    // Intel GPU hardware has a limitation on the size of work group dimensions to
    // be at most uint32_t. This function works around that by passing an offset
    // argument. The OpenCL native offset cannot be used due to lack of SYCL
    // interop support.
    static status_t large_parallel_for(const exec_ctx_t &ctx,
            const compute::nd_range_t &nd_range,
            const compute::kernel_t &kernel,
            compute::kernel_arg_list_t &arg_list, int offset_idx) {

        auto global_range = nd_range.global_range();
        auto local_range = nd_range.local_range();

        size_t off_inc[3] = {};
        for (int i = 0; i < 3; i++)
            off_inc[i] = local_range ? UINT32_MAX * local_range[i] : UINT32_MAX;

        int64x3_t offset_arg = {};
        auto &offset = offset_arg.array;
        for_(offset[2] = 0; static_cast<size_t>(offset[2]) < global_range[2];
                offset[2] += off_inc[2])
        for_(offset[1] = 0; static_cast<size_t>(offset[1]) < global_range[1];
                offset[1] += off_inc[1])
        for_(offset[0] = 0; static_cast<size_t>(offset[0]) < global_range[0];
                offset[0] += off_inc[0])
        {
            arg_list.set(offset_idx, offset_arg);
            size_t range[3];
            for (int i = 0; i < 3; i++)
                range[i] = std::min(off_inc[i], global_range[i] - offset[i]);

            CHECK(parallel_for(ctx, compute::nd_range_t(3, range, local_range),
                    kernel, arg_list));
        }
        return status::success;
    }

protected:
    int32_t version() const { return version_; }

    void set_version(int32_t version) { version_ = version; }

    void register_primitive(const primitive_t *primitive) {
        registered_compute_blocks_.emplace_back(primitive);
    }

    status_t register_kernels(const std::vector<compute::kernel_t> &kernels) {
        for (const auto &k : kernels) {
            CHECK(k.dump());
            registered_compute_blocks_.emplace_back(k);
        }
        return status::success;
    }

    virtual status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const {
        return status::success;
    }

private:
    const std::vector<compute_block_t> &compute_blocks() const {
        return registered_compute_blocks_;
    }

    static status_t parallel_for(stream_t &stream,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list,
            const compute::event_t &deps, compute::event_t &out_dep) {
        return kernel.parallel_for(stream, range, arg_list, deps, out_dep);
    }

    std::vector<compute_block_t> registered_compute_blocks_;

    // Persistent cache versioning is not used by default. To enable versioning
    // the primitive should:
    // 1) Set the version via set_version() in case of non-cached initialization
    // 2) Retrieve the version from the cache blob and set it via set_version()
    //    in case of cached initialization
    int32_t version_ = -1;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
