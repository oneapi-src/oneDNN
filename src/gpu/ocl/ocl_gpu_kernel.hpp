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

#ifndef GPU_OCL_OCL_GPU_KERNEL_HPP
#define GPU_OCL_OCL_GPU_KERNEL_HPP

#include <assert.h>
#include <string>
#include <CL/cl.h>

#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_gpu_kernel_cache_t;

class ocl_gpu_kernel_t : public compute::kernel_impl_t {
public:
    ocl_gpu_kernel_t(const std::shared_ptr<compute::binary_t> &binary,
            const std::string &kernel_name,
            const std::vector<gpu::compute::scalar_type_t> &arg_types)
        : state_(state_t::binary)
        , ocl_kernel_(nullptr)
        , binary_(binary)
        , binary_size_(binary->size())
        , kernel_name_(kernel_name)
        , arg_types_(arg_types) {
        MAYBE_UNUSED(state_);
    }

    ~ocl_gpu_kernel_t() override;

    cl_kernel ocl_kernel() const {
        assert(state_ == state_t::kernel);
        return ocl_kernel_;
    }

    status_t parallel_for(stream_t &stream, const compute::nd_range_t &range,
            const compute::kernel_arg_list_t &arg_list) override;

    status_t realize(compute::kernel_t *kernel, const engine_t *engine,
            compute::program_list_t *programs) const override;

    const char *name() const {
        assert(state_ == state_t::binary);
        return kernel_name_.c_str();
    }

    const std::shared_ptr<compute::binary_t> &binary() const override {
        assert(state_ == state_t::binary);
        return binary_;
    }

    status_t binary(engine_t *engine, compute::binary_t &binary) const override;

    const std::vector<gpu::compute::scalar_type_t> &arg_types() const override {
        return arg_types_;
    }

    void clear() override {
        assert(state_ == state_t::binary);
        binary_->clear();
        kernel_name_.clear();
        arg_types_.clear();
    }

    status_t binary_size(size_t *binary_size) const override {
        (*binary_size) = binary_size_;
        return status::success;
    }

    enum class state_t { binary, kernel };

protected:
    ocl_gpu_kernel_t(cl_kernel ocl_kernel,
            const std::vector<gpu::compute::scalar_type_t> &arg_types);

    state_t state_;
    cl_kernel ocl_kernel_;
    std::shared_ptr<compute::binary_t> binary_;
    // When DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE is defined the binary_ is
    // cleared via `clear()` to reduce memory footprint. Because of that the
    // binary size is stored separately to avoid querying it.
    size_t binary_size_;
    std::string kernel_name_;

    std::vector<gpu::compute::scalar_type_t> arg_types_;

    std::shared_ptr<ocl_gpu_kernel_cache_t> cache_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_OCL_GPU_KERNEL_HPP
