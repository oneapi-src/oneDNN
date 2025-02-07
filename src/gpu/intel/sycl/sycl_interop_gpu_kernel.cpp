/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "gpu/intel/sycl/sycl_interop_gpu_kernel.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/ocl/types_interop.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/sycl/l0/utils.hpp"
#include "gpu/intel/sycl/stream.hpp"
#include "gpu/intel/sycl/utils.hpp"
#include "gpu/intel/utils.hpp"
#include "xpu/sycl/c_types_map.hpp"
#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

using namespace impl::gpu::intel::sycl;

static void set_scalar_arg(::sycl::handler &cgh, int index,
        intel::compute::scalar_type_t type, const void *value) {
    using scalar_type_t = intel::compute::scalar_type_t;
    switch (type) {
        case scalar_type_t::_char:
        case scalar_type_t::_uchar:
            cgh.set_arg(index, *static_cast<const uint8_t *>(value));
            break;
        case scalar_type_t::_bfloat16:
        case scalar_type_t::_half:
        case scalar_type_t::_short:
        case scalar_type_t::_ushort:
            cgh.set_arg(index, *static_cast<const uint16_t *>(value));
            break;
        case scalar_type_t::_float:
        case scalar_type_t::_int:
        case scalar_type_t::_uint:
            cgh.set_arg(index, *static_cast<const uint32_t *>(value));
            break;
        case scalar_type_t::_double:
        case scalar_type_t::_long:
        case scalar_type_t::_ulong:
            cgh.set_arg(index, *static_cast<const uint64_t *>(value));
            break;
        case scalar_type_t::_zero_pad_mask_t:
            cgh.set_arg(index, *static_cast<const zero_pad_mask_t *>(value));
            break;
        case scalar_type_t::_int64x2_t:
            cgh.set_arg(index, *static_cast<const int64x2_t *>(value));
            break;
        case scalar_type_t::_int64x3_t:
            cgh.set_arg(index, *static_cast<const int64x3_t *>(value));
            break;
        case scalar_type_t::_int64x4_t:
            cgh.set_arg(index, *static_cast<const int64x4_t *>(value));
            break;
        case scalar_type_t::_int64x5_t:
            cgh.set_arg(index, *static_cast<const int64x5_t *>(value));
            break;
        case scalar_type_t::_int64x6_t:
            cgh.set_arg(index, *static_cast<const int64x6_t *>(value));
            break;
        case scalar_type_t::_dispatch_gws_rt_params_t:
            cgh.set_arg(index,
                    *static_cast<const dispatch_gws_rt_params_t *>(value));
            break;
        default:
            gpu_error_not_expected() << "Unimplemented scalar_type_t";
            throw std::runtime_error("Internal error");
    }
}

status_t sycl_interop_gpu_kernel_t::parallel_for(impl::stream_t &stream,
        const gpu::intel::compute::nd_range_t &range,
        const gpu::intel::compute::kernel_arg_list_t &arg_list,
        const xpu::event_t &deps, xpu::event_t &out_dep) {
    if (range.is_zero()) return status::success;
    auto *gpu_stream = utils::downcast<gpu::stream_t *>(&stream);
    auto &queue
            = *utils::downcast<xpu::sycl::stream_impl_t *>(gpu_stream->impl())
                       ->queue();
    const auto *sycl_engine_impl
            = utils::downcast<const xpu::sycl::engine_impl_t *>(
                    stream.engine()->impl());

    // XXX: DPCPP/L0 does not support non-uniform work-groups and does not
    // provide any diagnostics. This is to catch potential issues on oneDNN
    // side.
    if (sycl_engine_impl->backend() == xpu::sycl::backend_t::level0
            && range.local_range()) {
        for (size_t i = 0; i < range.ndims(); i++) {
            size_t gws = range.global_range()[i];
            size_t lws = range.local_range()[i];
            if (lws > 0 && gws % lws != 0) {
                VERROR(common, level_zero,
                        "only uniform work-groups are supported");
                return status::invalid_arguments;
            }
        }
    }
    CHECK(check_scalar_arguments(arg_list));

    auto event = queue.submit([&](::sycl::handler &cgh) {
        cgh.depends_on(xpu::sycl::event_t::from(deps).events);
        for (int i = 0; i < arg_list.nargs(); ++i) {
            auto &arg = arg_list.get(i);
            if (arg.is_global()) {
                auto *mem_storage
                        = static_cast<const memory_storage_t *>(arg.value());
                if (*mem_storage) {
                    auto *sycl_mem_storage = utils::downcast<
                            const xpu::sycl::memory_storage_base_t *>(
                            mem_storage);
                    switch (sycl_mem_storage->memory_kind()) {
                        case xpu::sycl::memory_kind::buffer: {
                            auto *m = utils::downcast<
                                    const xpu::sycl::buffer_memory_storage_t *>(
                                    mem_storage);
                            auto &sycl_buf = m->buffer();
                            cgh.set_arg((int)i,
                                    sycl_buf.get_access<
                                            ::sycl::access::mode::read_write>(
                                            cgh));
                            break;
                        }
                        case xpu::sycl::memory_kind::usm: {
                            auto *m = utils::downcast<
                                    const xpu::sycl::usm_memory_storage_t *>(
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
                auto acc = xpu::sycl::compat::local_accessor<uint8_t, 1>(
                        ::sycl::range<1>(arg.size()), cgh);
                cgh.set_arg((int)i, std::move(acc));
            } else {
                set_scalar_arg(cgh, (int)i, arg.scalar_type(), arg.value());
            }
        }
        if (range.local_range()) {
            auto sycl_nd_range = gpu::intel::sycl::to_sycl_nd_range(range);
            cgh.parallel_for(sycl_nd_range, *sycl_kernel_);
        } else {
            const auto &global_range = range.global_range();
            auto sycl_range = ::sycl::range<3>(
                    global_range.ndims() >= 3 ? global_range[2] : 1,
                    global_range.ndims() >= 2 ? global_range[1] : 1,
                    global_range[0]);
            cgh.parallel_for(sycl_range, *sycl_kernel_);
        }
    });

    if (stream.is_profiling_enabled()) {
        auto sycl_event = utils::make_unique<xpu::sycl::event_t>(
                std::vector<::sycl::event> {event});
        gpu_stream->profiler().register_event(std::move(sycl_event));
    }

    xpu::sycl::event_t::from(out_dep).events = {std::move(event)};
    return status::success;
}

status_t sycl_interop_gpu_kernel_t::dump() const {
    xpu::binary_t binary;
    CHECK(gpu::intel::sycl::get_kernel_binary(sycl_kernel(), binary));
    return gpu::intel::gpu_utils::dump_kernel_binary(binary, name());
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
