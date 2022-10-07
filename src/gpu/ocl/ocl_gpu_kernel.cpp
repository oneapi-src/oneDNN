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

#include <assert.h>
#include <string>
#include <CL/cl.h>

#include "gpu/ocl/ocl_gpu_kernel.hpp"

#include "common/utils.hpp"
#include "gpu/compute/program_list.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_usm_utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/profile.hpp"
#include "gpu/profile.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

namespace {
// RAII helper to lock/unlock cl_kernel for enqueueing.
struct enqueue_lock_t {
    enqueue_lock_t(std::atomic<bool> *is_locked) : is_locked(is_locked) {
        try_lock();
    }

    ~enqueue_lock_t() {
        if (owns_lock) unlock();
    }

    // Tries to lock cl_kernel for enqueueing.
    void try_lock() {
        bool expected = false;
        owns_lock = is_locked->compare_exchange_weak(expected, true);
    }

    // Unlocks cl_kernel after enqueueing.
    void unlock() const { is_locked->store(false); }

    std::atomic<bool> *is_locked;
    bool owns_lock = false;
};
} // namespace

ocl_gpu_kernel_t::~ocl_gpu_kernel_t() {
    if (ocl_kernel_) OCL_CHECK_V(clReleaseKernel(ocl_kernel_));
}

status_t ocl_gpu_kernel_t::parallel_for(stream_t &stream,
        const compute::nd_range_t &range,
        const compute::kernel_arg_list_t &arg_list) {
    assert(state_ == state_t::kernel);

    auto *ocl_stream = utils::downcast<ocl_stream_t *>(&stream);
    cl_command_queue queue = ocl_stream->queue();

    assert(ocl_kernel_ && "kernel is NULL");

    cl_kernel enqueue_kernel = ocl_kernel_;
    ocl_wrapper_t<cl_kernel> cloned_kernel;

    // Try to lock the mutex to use the original cl_kernel without cloning.
    enqueue_lock_t enqueue_lock(&is_locked_);
    if (!enqueue_lock.owns_lock) {
        // Failed to lock so clone the kernel to avoid concurrent access to
        // the same cl_kernel object.
        cl_kernel _cloned_kernel;
        CHECK(clone_kernel(ocl_kernel_, &_cloned_kernel));
        cloned_kernel = make_ocl_wrapper(_cloned_kernel);
        enqueue_kernel = cloned_kernel.get();
    }

    for (int i = 0; i < arg_list.nargs(); ++i) {
        auto &arg = arg_list.get(i);
        cl_int set_err = CL_SUCCESS;
        if (arg.is_global()) {
            auto *mem_storage
                    = static_cast<const memory_storage_t *>(arg.value());
            if (!mem_storage->is_null()) {
                auto *ocl_mem_storage
                        = utils::downcast<const ocl_memory_storage_base_t *>(
                                mem_storage);

                // Validate that the OpenCL contexts match for execution
                // context and memory.
                auto stream_ocl_ctx
                        = utils::downcast<ocl_gpu_engine_t *>(stream.engine())
                                  ->context();
                auto memory_storage_ocl_ctx
                        = utils::downcast<ocl_gpu_engine_t *>(
                                ocl_mem_storage->engine())
                                  ->context();
                if (stream_ocl_ctx != memory_storage_ocl_ctx) {
                    MAYBE_REPORT_ERROR(
                            "mismatched OpenCL context for primitive/memory");
                    return status::invalid_arguments;
                }

                switch (ocl_mem_storage->memory_kind()) {
                    case memory_kind::buffer: {
                        auto *m = utils::downcast<
                                const ocl_buffer_memory_storage_t *>(
                                ocl_mem_storage);
                        auto ocl_mem = m->mem_object();
                        set_err = clSetKernelArg(
                                enqueue_kernel, i, sizeof(cl_mem), &ocl_mem);
                        break;
                    }
                    case memory_kind::usm: {
                        auto *m = utils::downcast<
                                const ocl_usm_memory_storage_t *>(
                                ocl_mem_storage);
                        auto *usm_ptr = m->usm_ptr();
                        CHECK(usm::set_kernel_arg_usm(
                                stream.engine(), enqueue_kernel, i, usm_ptr));
                        break;
                    }
                    default: assert(!"not expected");
                }
            } else {
                cl_mem null_mem = nullptr;
                set_err = clSetKernelArg(
                        enqueue_kernel, i, sizeof(cl_mem), &null_mem);
            }
        } else if (arg.is_local()) {
            set_err = clSetKernelArg(
                    enqueue_kernel, i, arg.size(), arg.value());
        } else if (arg.is_svm_pointer()) {
#ifdef CL_VERSION_2_0
            set_err = clSetKernelArgSVMPointer(enqueue_kernel, i, arg.value());
#else
            return status::runtime_error; // SVM is not supported
#endif // CL_VERSION_2_0
        } else {
            // Convert if types do not match.
            typename std::aligned_storage<sizeof(float), sizeof(float)>::type
                    tmp_storage;
            void *cast_storage = &tmp_storage;
            auto cvt_arg = compute::kernel_arg_t::cast(
                    arg_types_[i], arg, cast_storage);
            set_err = clSetKernelArg(
                    enqueue_kernel, i, cvt_arg.size(), cvt_arg.value());
        }
        status_t status = convert_to_dnnl(set_err);
        if (status != status::success) return status;
    }

    cl_uint ndims = static_cast<cl_uint>(range.ndims());
    if (range.is_zero()) { return status::success; }
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(queue, enqueue_kernel, ndims, nullptr,
            range.global_range(), range.local_range(), 0, nullptr,
            is_profiling_enabled() ? &event : nullptr);
    OCL_CHECK(err);
    if (is_profiling_enabled()) register_profiling_event(event);
    return status::success;
}

status_t ocl_gpu_kernel_t::realize(compute::kernel_t *kernel,
        const engine_t *engine, compute::program_list_t *programs) const {
    assert(state_ == state_t::binary);
    if (!binary_) return status::success;

    cl_int err;
    if (programs) {
        auto *p = programs->get<cl_program>(binary_.get());
        if (p) {
            auto k = make_ocl_wrapper(clCreateKernel(p, name(), &err));
            OCL_CHECK(err);
            (*kernel) = compute::kernel_t(new ocl_gpu_kernel_t(k, arg_types_));
            return status::success;
        }
    }

    auto *compute_engine = utils::downcast<const ocl_gpu_engine_t *>(engine);
    cl_device_id dev = compute_engine->device();
    cl_context ctx = compute_engine->context();
    const unsigned char *binary_buffer = binary_->data();
    size_t binary_size = binary_->size();
    assert(binary_size > 0);

    auto program = make_ocl_wrapper(clCreateProgramWithBinary(
            ctx, 1, &dev, &binary_size, &binary_buffer, nullptr, &err));
    OCL_CHECK(err);
    err = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
    OCL_CHECK(err);

    auto ocl_kernel = make_ocl_wrapper(clCreateKernel(program, name(), &err));
    OCL_CHECK(err);
    (*kernel) = compute::kernel_t(new ocl_gpu_kernel_t(ocl_kernel, arg_types_));

    if (programs) {
        programs->add(binary_.get(), program.get());
        program.release();
    }

    return status::success;
}

status_t ocl_gpu_kernel_t::binary(
        engine_t *engine, compute::binary_t &binary) const {
    const auto *ocl_engine = utils::downcast<const ocl_gpu_engine_t *>(engine);
    std::shared_ptr<compute::binary_t> shared_binary;
    CHECK(get_ocl_program_binary(
            ocl_kernel_, ocl_engine->device(), shared_binary));
    binary = std::move(*shared_binary);
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
