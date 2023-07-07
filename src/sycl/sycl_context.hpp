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

#ifndef SYCL_SYCL_CONTEXT_HPP
#define SYCL_SYCL_CONTEXT_HPP

#include "gpu/compute/context.hpp"
#include "oneapi/dnnl/dnnl_sycl.h"

namespace dnnl {
namespace impl {
namespace sycl {

struct sycl_event_t : public gpu::compute::event_t {
    sycl_event_t() = default;
    sycl_event_t(const std::vector<::sycl::event> &event) : events(event) {}
    sycl_event_t(std::vector<::sycl::event> &&event)
        : events(std::move(event)) {}
    sycl_event_t(const sycl_event_t &) = default;
    sycl_event_t &operator=(sycl_event_t other) {
        events = other.events;
        return *this;
    }

    const ::sycl::event &operator[](size_t i) const { return events[i]; }
    ::sycl::event &operator[](size_t i) { return events[i]; }
    size_t size() const { return events.size(); }

    static sycl_event_t &from(gpu::compute::event_t &event) {
        return *utils::downcast<sycl_event_t *>(&event);
    }
    static const sycl_event_t &from(const gpu::compute::event_t &event) {
        return *utils::downcast<const sycl_event_t *>(&event);
    }
    std::unique_ptr<gpu::compute::event_t> clone() const {
        return std::unique_ptr<gpu::compute::event_t>(new sycl_event_t(*this));
    }

    void append(const gpu::compute::event_t &event) {
        auto &other = *utils::downcast<const sycl_event_t *>(&event);
        events.insert(events.end(), other.events.begin(), other.events.end());
    }

    std::vector<::sycl::event> events;
};

struct sycl_context_t final : public gpu::compute::context_t {
    sycl_context_t() = default;
    sycl_context_t(const std::vector<::sycl::event> &&events)
        : events_(std::move(events)) {};
    sycl_context_t(const sycl_context_t &) = default;
    ~sycl_context_t() = default;

    sycl_context_t &operator=(const sycl_context_t &other) {
        events_ = other.events_;
        return *this;
    }

    sycl_event_t &get_sycl_deps() { return events_; }
    const sycl_event_t &get_sycl_deps() const { return events_; }
    gpu::compute::event_t &get_deps() override { return events_; }
    const gpu::compute::event_t &get_deps() const override { return events_; }

    void set_deps(std::vector<::sycl::event> &&event) {
        events_ = sycl_event_t(std::move(event));
    }
    void set_deps(sycl_event_t &&events) { events_ = std::move(events); };

    void append_deps(const gpu::compute::event_t &event) override {
        events_.append(event);
    }

private:
    sycl_event_t events_;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl
#endif
