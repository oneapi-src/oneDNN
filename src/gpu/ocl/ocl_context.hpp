/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_OCL_OCL_CONTEXT_HPP
#define GPU_OCL_OCL_CONTEXT_HPP

#include "gpu/compute/context.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ocl_event_t final : compute::event_t {
    ocl_event_t() = default;
    ocl_event_t(const std::vector<ocl_wrapper_t<cl_event>> &events)
        : events(events) {}
    ocl_event_t(std::vector<ocl_wrapper_t<cl_event>> &&events)
        : events(std::move(events)) {}
    ocl_event_t(ocl_wrapper_t<cl_event> &&event) {
        events.emplace_back(std::move(event));
    }

    const ocl_wrapper_t<cl_event> &operator[](size_t i) const {
        return events[i];
    }
    ocl_wrapper_t<cl_event> &operator[](size_t i) { return events[i]; }
    size_t size() const { return events.size(); }

    static ocl_event_t &from(compute::event_t &event) {
        return *utils::downcast<ocl_event_t *>(&event);
    }
    static const ocl_event_t &from(const compute::event_t &event) {
        return *utils::downcast<const ocl_event_t *>(&event);
    }
    std::unique_ptr<compute::event_t> clone() const {
        return std::unique_ptr<compute::event_t>(new ocl_event_t(*this));
    }

    void append(const compute::event_t &event) {
        auto &other = *utils::downcast<const ocl_event_t *>(&event);
        events.insert(events.end(), other.events.begin(), other.events.end());
    };

    std::vector<ocl_wrapper_t<cl_event>> events;
};

struct ocl_context_t final : public gpu::compute::context_t {
    ocl_context_t() = default;
    ocl_context_t(const std::vector<ocl_wrapper_t<cl_event>> &&events)
        : events_(std::move(events)) {};
    ocl_context_t(const ocl_context_t &) = default;
    ~ocl_context_t() = default;

    ocl_context_t &operator=(const ocl_context_t &other) {
        events_ = other.events_;
        return *this;
    }

    ocl_event_t &get_ocl_deps() { return events_; }
    const ocl_event_t &get_ocl_deps() const { return events_; }
    gpu::compute::event_t &get_deps() override { return events_; }
    const gpu::compute::event_t &get_deps() const override { return events_; }

    void set_deps(std::vector<ocl_wrapper_t<cl_event>> &&event) {
        events_ = ocl_event_t(std::move(event));
    }
    void set_deps(ocl_event_t &&events) { events_ = std::move(events); };

    void append_deps(const compute::event_t &event) override {
        events_.append(event);
    }

private:
    ocl_event_t events_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
