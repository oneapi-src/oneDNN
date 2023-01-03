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

#include "gpu/sycl/sycl_interop_gpu_kernel.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/profile.hpp"
#include "gpu/zero_pad_struct.h"
#include "sycl/level_zero_utils.hpp"
#include "sycl/profile.hpp"
#include "sycl/sycl_c_types_map.hpp"
#include "sycl/sycl_stream.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;

static void set_scalar_arg(
        ::sycl::handler &cgh, int index, size_t size, const void *value) {
    switch (size) {
        case sizeof(uint8_t):
            cgh.set_arg(index, *static_cast<const uint8_t *>(value));
            break;
        case sizeof(uint16_t):
            cgh.set_arg(index, *static_cast<const uint16_t *>(value));
            break;
        case sizeof(uint32_t):
            cgh.set_arg(index, *static_cast<const uint32_t *>(value));
            break;
        case sizeof(uint64_t):
            cgh.set_arg(index, *static_cast<const uint64_t *>(value));
            break;
        case sizeof(zero_pad_mask_t):
            cgh.set_arg(index, *static_cast<const zero_pad_mask_t *>(value));
            break;
        default:
            assert(!"Please add another case");
            throw std::runtime_error("Internal error");
    }
}

status_t sycl_interop_gpu_kernel_t::parallel_for(stream_t &stream,
        const gpu::compute::nd_range_t &range,
        const gpu::compute::kernel_arg_list_t &arg_list) {
    if (range.is_zero()) return status::success;
    auto *sycl_stream = utils::downcast<sycl_stream_t *>(&stream);
    auto &queue = sycl_stream->queue();
    sycl_gpu_engine_t *sycl_engine
            = utils::downcast<sycl_gpu_engine_t *>(sycl_stream->engine());

    // XXX: DPCPP/L0 does not support non-uniform work-groups and does not
    // provide any diagnostics. This is to catch potential issues on oneDNN
    // side.
    if (sycl_engine->backend() == backend_t::level0 && range.local_range()) {
        for (size_t i = 0; i < range.ndims(); i++) {
            size_t gws = range.global_range()[i];
            size_t lws = range.local_range()[i];
            if (lws > 0 && gws % lws != 0) {
                if (get_verbose()) {
                    printf("onednn_verbose,gpu,error,Level Zero backend only "
                           "supports uniform work-groups\n");
                    fflush(nullptr);
                }
                return status::invalid_arguments;
            }
        }
    }
    CHECK(gpu::compute::check_scalar_arguments(arg_list, arg_types_));

    auto event = queue.submit([&](::sycl::handler &cgh) {
        cgh.depends_on(sycl_stream->get_deps());
        for (int i = 0; i < arg_list.nargs(); ++i) {
            auto &arg = arg_list.get(i);
            if (arg.is_global()) {
                auto *mem_storage
                        = static_cast<const memory_storage_t *>(arg.value());
                if (*mem_storage) {
                    auto *sycl_mem_storage = utils::downcast<
                            const sycl_memory_storage_base_t *>(mem_storage);
                    switch (sycl_mem_storage->memory_kind()) {
                        case memory_kind::buffer: {
                            auto *m = utils::downcast<
                                    const sycl_buffer_memory_storage_t *>(
                                    mem_storage);
                            auto &sycl_buf = m->buffer();
                            cgh.set_arg((int)i,
                                    sycl_buf.get_access<
                                            ::sycl::access::mode::read_write>(
                                            cgh));
                            break;
                        }
                        case memory_kind::usm: {
                            auto *m = utils::downcast<
                                    const sycl_usm_memory_storage_t *>(
                                    mem_storage);
                            cgh.set_arg((int)i, m->usm_ptr());
                            break;
                        }
                        default: assert(!"not expected");
                    }
                } else {
                    cgh.set_arg((int)i, nullptr);
                }
            } else if (arg.is_local()) {
                auto acc = compat::local_accessor<uint8_t, 1>(
                        ::sycl::range<1>(arg.size()), cgh);
                cgh.set_arg((int)i, acc);
            } else {
                set_scalar_arg(cgh, (int)i, arg.size(), arg.value());
            }
        }
        if (range.local_range()) {
            auto sycl_nd_range = to_sycl_nd_range(range);
            cgh.parallel_for(sycl_nd_range, *sycl_kernel_);
        } else {
            auto *global_range = range.global_range();
            auto sycl_range = ::sycl::range<3>(
                    global_range[2], global_range[1], global_range[0]);
            cgh.parallel_for(sycl_range, *sycl_kernel_);
        }
    });

    if (gpu::is_profiling_enabled()) register_profile_event(event);
    sycl_stream->set_deps({event});
    return status::success;
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
