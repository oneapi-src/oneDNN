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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_WEAKPTR_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_WEAKPTR_UTILS_HPP

#include <memory>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {

template <typename T>
bool is_uninitialized_weakptr(const std::weak_ptr<T> &weak) {
    return !weak.owner_before(std::weak_ptr<T> {})
            && !std::weak_ptr<T> {}.owner_before(weak);
}

template <typename T>
static T *get_raw_from_weakptr(const std::weak_ptr<T> &wptr) {
    if (is_uninitialized_weakptr(wptr)) return nullptr;
    auto raw = wptr.lock();
    assert(raw);
    return raw.get();
}

template <typename T>
struct weakptr_hashset_t {
    using impl_t = std::unordered_map<T *, std::weak_ptr<T>>;
    impl_t impl_;
    template <typename IterT>
    struct iterator_t {
        IterT impl_itr_;
        std::weak_ptr<T> operator*() { return impl_itr_->second; }
        iterator_t &operator++() {
            ++impl_itr_;
            return *this;
        }
        iterator_t operator++(int) {
            iterator old = *this;
            ++impl_itr_;
            return old;
        }
        bool operator!=(const iterator_t &other) const {
            return impl_itr_ != other.impl_itr_;
        }
    };
    using iterator = iterator_t<typename impl_t::iterator>;
    using const_iterator = iterator_t<typename impl_t::const_iterator>;
    iterator begin() noexcept { return iterator {impl_.begin()}; }
    iterator end() noexcept { return iterator {impl_.end()}; }
    const_iterator begin() const noexcept {
        return const_iterator {impl_.begin()};
    }
    const_iterator end() const noexcept { return const_iterator {impl_.end()}; }

    const_iterator find(std::weak_ptr<T> v) const {
        return const_iterator {impl_.find(v.lock().get())};
    }

    iterator find(std::weak_ptr<T> v) {
        return iterator {impl_.find(v.lock().get())};
    }

    bool has(const std::weak_ptr<T> &v) const {
        return impl_.find(v.lock().get()) != impl_.end();
    }

    void merge(const weakptr_hashset_t<T> &other) {
        impl_.insert(other.impl_.begin(), other.impl_.end());
    }

    void insert(const std::weak_ptr<T> &v) {
        auto ptr = v.lock();
        impl_[ptr.get()] = v;
    }

    void insert(const std::shared_ptr<T> &v) { impl_[v.get()] = v; }

    void erase(const std::weak_ptr<T> &v) { impl_.erase(v.lock().get()); }

    weakptr_hashset_t() = default;
    weakptr_hashset_t(std::initializer_list<std::weak_ptr<T>> initv) {
        for (auto &v : initv) {
            auto ptr = v.lock();
            impl_[ptr.get()] = v;
        }
    }

    size_t size() const { return impl_.size(); }

    bool operator==(const weakptr_hashset_t<T> &other) const {
        if (size() != other.size()) { return false; }
        for (auto &kv : impl_) {
            if (other.impl_.find(kv.first) == other.impl_.end()) {
                return false;
            }
        }
        return true;
    }
};

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
