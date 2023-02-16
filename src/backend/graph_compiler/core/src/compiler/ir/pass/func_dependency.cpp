/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include "func_dependency.hpp"
#include <vector>
#include "../viewer.hpp"
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
class func_dependency_impl_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    std::unordered_set<func_t> &set_;
    std::vector<func_t> &dep_;

    func_dependency_impl_t(
            std::vector<func_t> &dep, std::unordered_set<func_t> &set)
        : set_(set), dep_(dep) {}
    void view(call_c v) override {
        func_t f = std::dynamic_pointer_cast<func_base>(v->func_);
        if (f) {
            if (set_.find(f) == set_.end()) {
                set_.insert(f);
                dep_.push_back(f);
            }
        }
        for (auto &arg : v->args_) {
            dispatch(arg);
        }
    }

    void view(func_addr_c v) override {
        auto f = v->func_;
        if (set_.find(f) == set_.end()) {
            set_.insert(f);
            dep_.push_back(f);
        }
    }
};

func_c func_dependency_finder_t::operator()(func_c f) {
    std::unordered_set<func_t> set;
    func_dependency_impl_t impl(dep_, set);
    impl.dispatch(f);
    return f;
}

stmt_c func_dependency_finder_t::operator()(stmt_c f) {
    std::unordered_set<func_t> set;
    func_dependency_impl_t impl(dep_, set);
    impl.dispatch(f);
    return f;
}

func_c func_dependency_finder_t::operator()(
        func_c f, std::unordered_set<func_t> &set) {
    func_dependency_impl_t impl(dep_, set);
    impl.dispatch(f);
    return f;
}

stmt_c func_dependency_finder_t::operator()(
        stmt_c f, std::unordered_set<func_t> &set) {
    func_dependency_impl_t impl(dep_, set);
    impl.dispatch(f);
    return f;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
