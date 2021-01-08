/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef INTERFACE_ID_HPP
#define INTERFACE_ID_HPP

#include <atomic>
#include <cstddef>

struct dnnl_graph_id {
public:
    using id_t = size_t;
    id_t id() const { return id_; }

    dnnl_graph_id() : id_(++counter) {};
    dnnl_graph_id(const dnnl_graph_id &other) : id_(other.id()) {};
    dnnl_graph_id &operator=(const dnnl_graph_id &other) = delete;

protected:
    static std::atomic<id_t> counter;
    ~dnnl_graph_id() = default;

private:
    const id_t id_;
};

namespace dnnl {
namespace graph {
namespace impl {
using id = ::dnnl_graph_id;
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
