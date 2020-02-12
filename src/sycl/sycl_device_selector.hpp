/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#ifndef SYCL_DEVICE_SELECTOR_HPP
#define SYCL_DEVICE_SELECTOR_HPP

#include <functional>
#include <CL/sycl.hpp>

namespace dnnl {
namespace impl {
namespace sycl {

template <typename UnderlyingType>
class fixed_device_selector_t : public cl::sycl::device_selector {
public:
    fixed_device_selector_t(const cl::sycl::device &device,
            std::function<void(const cl::sycl::device &, UnderlyingType &)>
                    func)
        : fixed_device_(device), func_(func) {
        if ((fixed_device_.is_cpu() || fixed_device_.is_gpu())) {
            func_(device, underlying_fixed_device_);
        }
    }

    virtual int operator()(const cl::sycl::device &device) const override {
        // Never choose devices other than fixed_device_
        // XXX: there is no reliable way to compare SYCL devices so try heuristics:
        // 1) For CPU and GPU SYCL devices compare their OpenCL devices
        // 2) For Host device assume it's always unique
        if (underlying_fixed_device_) {
            if (!device.is_cpu() && !device.is_gpu()) return -1;
            UnderlyingType underlying_dev;
            func_(device, underlying_dev);
            return (underlying_dev == underlying_fixed_device_ ? 1 : -1);
        }
        assert(fixed_device_.is_host());
        return device.is_host() ? 1 : -1;
    }

private:
    cl::sycl::device fixed_device_;
    UnderlyingType underlying_fixed_device_ = nullptr;
    std::function<void(const cl::sycl::device &, UnderlyingType &)> func_;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
