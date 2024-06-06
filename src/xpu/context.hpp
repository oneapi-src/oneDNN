/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef XPU_CONTEXT_HPP
#define XPU_CONTEXT_HPP

#include <memory>

namespace dnnl {
namespace impl {
namespace xpu {

class event_t {
public:
    virtual ~event_t() = 0;
    virtual std::unique_ptr<event_t> clone() const = 0;
};
inline event_t::~event_t() = default;

// Abstract class for runtime inputs and outputs
class context_t {
public:
    virtual event_t &get_deps() = 0;
    virtual const event_t &get_deps() const = 0;

    virtual void append_deps(const event_t &event) = 0;
    virtual ~context_t() = default;
};

} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
