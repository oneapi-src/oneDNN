/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "common/rw_mutex.hpp"
#include "common/utils.hpp"
#include "gpu/ocl/ocl_context.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_usm_utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Kernel wrapper storing a per-thread copy of cl_kernel.
class kernel_wrapper_t {
public:
    kernel_wrapper_t(cl_kernel kernel = nullptr) : kernel_(kernel) {}

    operator cl_kernel() const { return kernel_; }

    status_t set_arg(int arg_index, size_t arg_size, const void *arg_value) {
        cl_int err = clSetKernelArg(kernel_, arg_index, arg_size, arg_value);
        return convert_to_dnnl(err);
    }

    status_t set_svm_arg(int arg_index, const void *arg_value) {
#ifdef CL_VERSION_2_0
        cl_int err = clSetKernelArgSVMPointer(kernel_, arg_index, arg_value);
        return convert_to_dnnl(err);
#else
        // SVM is not supported.
        UNUSED(arg_index);
        UNUSED(arg_value);
        return status::runtime_error;
#endif
    }

    status_t set_usm_arg(
            engine_t *engine, int arg_index, const void *arg_value) {
        return usm::set_kernel_arg_usm(engine, kernel_, arg_index, arg_value);
    }

private:
    cl_kernel kernel_;
};

class ocl_gpu_kernel_cache_t {
public:
    ocl_gpu_kernel_cache_t(cl_kernel main_kernel) : main_kernel_(main_kernel) {}

    ~ocl_gpu_kernel_cache_t() {
        for (auto &kv : kernels_) {
            OCL_CHECK_V(clReleaseKernel(kv.second));
        }
    }

    status_t get(kernel_wrapper_t **kernel) {
        auto id = std::this_thread::get_id();
        {
            utils::lock_read_t lock_read(mutex_);
            auto it = kernels_.find(id);
            if (it != kernels_.end()) {
                *kernel = &it->second;
                return status::success;
            }
        }

        // No copy for this thread, clone the original kernel and save the
        // copy.
        cl_kernel cloned_kernel;
        CHECK(clone_kernel(main_kernel_, &cloned_kernel));

        utils::lock_write_t lock_write(mutex_);
        auto ret = kernels_.emplace(id, cloned_kernel);
        *kernel = &ret.first->second;
        return status::success;
    }

private:
    cl_kernel main_kernel_;
    std::unordered_map<std::thread::id, kernel_wrapper_t> kernels_;
    utils::rw_mutex_t mutex_;
};

ocl_gpu_kernel_t::ocl_gpu_kernel_t(cl_kernel ocl_kernel,
        const std::vector<gpu::compute::scalar_type_t> &arg_types)
    : ocl_kernel_(ocl_kernel), arg_types_(arg_types), save_events_(false) {
    OCL_CHECK_V(clRetainKernel(ocl_kernel_));
    cache_ = std::make_shared<ocl_gpu_kernel_cache_t>(ocl_kernel_);
}

ocl_gpu_kernel_t::~ocl_gpu_kernel_t() {
    if (ocl_kernel_) OCL_CHECK_V(clReleaseKernel(ocl_kernel_));
}

status_t ocl_gpu_kernel_t::get_binary(
        const engine_t *engine, compute::binary_t &binary) const {
    auto *ocl_engine = utils::downcast<const ocl_gpu_engine_t *>(engine);
    return get_ocl_program_binary(ocl_kernel(), ocl_engine->device(), binary);
}

status_t ocl_gpu_kernel_t::get_binary_size(
        const engine_t *engine, size_t *binary_size) const {
    auto *ocl_engine = utils::downcast<const ocl_gpu_engine_t *>(engine);
    return get_ocl_program_binary_size(
            ocl_kernel(), ocl_engine->device(), binary_size);
}

status_t ocl_gpu_kernel_t::parallel_for(stream_t &stream,
        const compute::nd_range_t &range,
        const compute::kernel_arg_list_t &arg_list,
        const compute::event_t &deps, compute::event_t &out_dep) {

    auto *ocl_stream = utils::downcast<ocl_stream_t *>(&stream);
    cl_command_queue queue = ocl_stream->queue();

    kernel_wrapper_t *kernel = nullptr;
    CHECK(cache_->get(&kernel));
    CHECK(gpu::compute::check_scalar_arguments(arg_list, arg_types_));
    for (int i = 0; i < arg_list.nargs(); ++i) {
        auto &arg = arg_list.get(i);
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
                        CHECK(kernel->set_arg(i, sizeof(cl_mem), &ocl_mem));
                        break;
                    }
                    case memory_kind::usm: {
                        auto *m = utils::downcast<
                                const ocl_usm_memory_storage_t *>(
                                ocl_mem_storage);
                        auto *usm_ptr = m->usm_ptr();
                        CHECK(kernel->set_usm_arg(stream.engine(), i, usm_ptr));
                        break;
                    }
                    default: assert(!"not expected");
                }
            } else {
                if (usm::is_usm_supported(stream.engine())) {
                    CHECK(kernel->set_usm_arg(stream.engine(), i, nullptr));
                } else {
                    cl_mem null_mem = nullptr;
                    CHECK(kernel->set_arg(i, sizeof(cl_mem), &null_mem));
                }
            }
        } else if (arg.is_local()) {
            CHECK(kernel->set_arg(i, arg.size(), arg.value()));
        } else if (arg.is_svm_pointer()) {
            CHECK(kernel->set_svm_arg(i, arg.value()));
        } else {
            CHECK(kernel->set_arg(i, arg.size(), arg.value()));
        }
    }

    cl_uint ndims = static_cast<cl_uint>(range.ndims());
    if (range.is_zero()) { return status::success; }

    ocl_wrapper_t<cl_event> event;
    if (ocl_stream->flags() & stream_flags::out_of_order) {
        const auto &event_wrappers = ocl_event_t::from(deps).events;
        std::vector<cl_event> events(
                event_wrappers.begin(), event_wrappers.end());

        cl_uint num_events = (cl_uint)events.size();
        const cl_event *events_data = num_events ? events.data() : nullptr;
        cl_int err = clEnqueueNDRangeKernel(queue, *kernel, ndims, nullptr,
                range.global_range(), range.local_range(), num_events,
                events_data, &event.unwrap());
        OCL_CHECK(err);
        ocl_event_t::from(out_dep).events = {event};
    } else {
        bool save_event = save_events_ || stream.is_profiling_enabled();
        cl_int err = clEnqueueNDRangeKernel(queue, *kernel, ndims, nullptr,
                range.global_range(), range.local_range(), 0, nullptr,
                save_event ? &event.unwrap() : nullptr);
        OCL_CHECK(err);
    }

    if (stream.is_profiling_enabled()) {
        ocl_stream->profiler().register_event(
                utils::make_unique<ocl_event_t>(std::move(event)));
    }

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
