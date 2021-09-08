/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
#ifndef BACKEND_DNNL_PASSES_COMPILE_OPS_HPP
#define BACKEND_DNNL_PASSES_COMPILE_OPS_HPP

#include <memory>
#include <vector>
#include <unordered_map>

#include "dnnl.hpp"

#include "interface/c_types_map.hpp"

#include "backend/dnnl/passes/op_executable.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

class executable_mgr {
public:
    executable_mgr() = default;

    // Disable assignment and copy
    executable_mgr(const executable_mgr &) = delete;
    executable_mgr(executable_mgr &&) = delete;
    executable_mgr &operator=(const executable_mgr &) = delete;
    executable_mgr &operator=(executable_mgr &&) = delete;

    int64_t init_executable() {
        auto ret = data_.insert({counter++, std::shared_ptr<op_executable>()});
        return ret.first->first;
    }

    std::shared_ptr<op_executable> &get_executable(int64_t key) {
        return data_[key];
    }

private:
    std::unordered_map<int64_t, std::shared_ptr<op_executable>> data_;
    int64_t counter {0};
};

impl::status_t compile_ops(std::vector<std::shared_ptr<impl::op_t>> &subgraph,
        const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr,
        executable_mgr &exec_mgr, pd_cache_t &pd_cache);

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
