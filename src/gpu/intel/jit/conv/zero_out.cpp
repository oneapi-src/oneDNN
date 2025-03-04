/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/jit/conv/zero_out.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

const size_t zero_out_kernel_desc_t::bytes_per_thr = 128;

std::string zero_out_kernel_desc_t::kernel_name() const {
    return "zero_out";
}

exec_config_t zero_out_kernel_desc_t::exec_cfg(
        const impl::engine_t *engine) const {
    return exec_config_t(hw_t(engine), regs_, simd_);
}
compute::range_t zero_out_kernel_desc_t::local_range() const {
    return compute::range_t(into<size_t>(simd_));
}

void zero_out_kernel_desc_t::init_kernel_iface(
        kernel_iface_t &kernel_iface) const {
    kernel_iface.register_arg("size", type_t::u32());
    kernel_iface.register_arg("ptr", type_t::byte_ptr());
}

void zero_out_kernel_desc_t::init_kernel_info(kernel_info_t &kernel_info,
        const kernel_params_base_t &_params,
        const impl::engine_t *engine) const {
    auto &params = static_cast<const zero_out_kernel_params_t &>(_params);
    for (int i = 0; i < kernel_info.nargs(); i++) {
        auto &name = kernel_info.arg_name(i);
        auto &var = kernel_info.arg_var(i);
        if (var.type().is_ptr()) continue;
        gpu_assert(name == "size") << "Unknown scalar argument: " << name;
        kernel_info.set_internal_arg(name, into<uint32_t>(params.size));
    }
    kernel_info.set_nd_range(nd_range(simd_, params.size));
}

status_t zero_out_kernel_desc_t::create_kernel(compute::kernel_t &kernel,
        gpu_primitive_t *primitive, impl::engine_t *engine) const {
    return primitive->create_kernel(
            engine, kernel, kernel_name().c_str(), *this);
}

status_t zero_out_kernel_desc_t::create_generator(
        const compute::compute_engine_t &engine,
        compute::kernel_t &kernel) const {
    ir_generator_t<zero_out_kernel_t> ir_gen(*this);
    return engine.create_kernel(&kernel, &ir_gen);
}

serialization_stream_t zero_out_kernel_desc_t::serialize() const {
    return serialization_stream_t(regs_, simd_, dpas_);
}

zero_out_kernel_desc_t zero_out_kernel_desc_t::deserialize(
        const serialization_stream_t &s) {
    zero_out_kernel_desc_t desc;
    deserializer_t d(s);
    d.pop(desc.regs_);
    d.pop(desc.simd_);
    d.pop(desc.dpas_);
    return desc;
}

compute::nd_range_t zero_out_kernel_desc_t::nd_range(int simd, size_t size) {
    return compute::nd_range_t(
            into<size_t>(
                    utils::div_up(size, zero_out_kernel_desc_t::bytes_per_thr)
                    * simd),
            simd);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
