/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef XPU_SYCL_CONTEXT_HPP
#define XPU_SYCL_CONTEXT_HPP

#include "oneapi/dnnl/dnnl_sycl.h"

#include "xpu/context.hpp"
#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

struct event_t : public xpu::event_t {
    event_t() = default;
    event_t(const std::vector<::sycl::event> &event) : events(event) {}
    event_t(std::vector<::sycl::event> &&event) : events(std::move(event)) {}
    event_t(const event_t &) = default;
    event_t &operator=(const event_t &other) {
        events = other.events;
        return *this;
    }

    ~event_t() override = default;

    const ::sycl::event &operator[](size_t i) const { return events[i]; }
    ::sycl::event &operator[](size_t i) { return events[i]; }
    size_t size() const { return events.size(); }

    static event_t &from(xpu::event_t &event) {
        return *utils::downcast<event_t *>(&event);
    }
    static const event_t &from(const xpu::event_t &event) {
        return *utils::downcast<const event_t *>(&event);
    }
    std::unique_ptr<xpu::event_t> clone() const override {
        return std::unique_ptr<xpu::event_t>(new event_t(*this));
    }

    void append(const xpu::event_t &event) {
        auto &other = *utils::downcast<const event_t *>(&event);
        events.insert(events.end(), other.events.begin(), other.events.end());
    }

    std::vector<::sycl::event> events;
};

struct context_t final : public xpu::context_t {
    context_t() = default;
    context_t(std::vector<::sycl::event> events)
        : events_(std::move(events)) {};
    context_t(const context_t &) = default;
    ~context_t() override = default;

    context_t &operator=(const context_t &other) {
        events_ = other.events_;
        return *this;
    }

    event_t &get_sycl_deps() { return events_; }
    const event_t &get_sycl_deps() const { return events_; }
    xpu::event_t &get_deps() override { return events_; }
    const xpu::event_t &get_deps() const override { return events_; }

    void set_deps(std::vector<::sycl::event> &&event) {
        events_ = event_t(std::move(event));
    }
    void set_deps(event_t &&events) { events_ = std::move(events); };

    void append_deps(const xpu::event_t &event) override {
        events_.append(event);
    }

private:
    event_t events_;
};

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
#endif
