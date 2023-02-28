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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_ALIGNED_PTR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_ALIGNED_PTR_HPP

#include <memory.h>
#include <stddef.h>
#include <utility>
#include <util/def.hpp>
#include <util/os.hpp>
#include <util/simple_math.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
/**
 * Allocator and RAII memory manager for aligned memory
 * @param sz the memory size, in bytes
 * @param aligment the alignment in bytes
 * */
struct SC_INTERNAL_API generic_ptr_base_t {
    void *ptr_;
    size_t size_;
    /**
     * Sets the memory buffer to 0
     * */
    void zeroout() const;
    /**
     * Flush cache
     * */
    void flush_cache() const;
};

struct aligned_ptr_policy_t {
    static void *alloc(size_t sz, size_t alignment) {
        return aligned_alloc(
                alignment, utils::divide_and_ceil(sz, alignment) * alignment);
    }

    static void dealloc(void *ptr, size_t sz) { aligned_free(ptr); }
};

/**
 * Allocator and RAII memory manager for aligned memory
 * @param sz the memory size, in bytes
 * @param aligment the alignment in bytes
 * */
template <typename Policy>
struct raii_ptr_t : protected generic_ptr_base_t {
    using generic_ptr_base_t::flush_cache;
    using generic_ptr_base_t::ptr_;
    using generic_ptr_base_t::size_;
    using generic_ptr_base_t::zeroout;
    raii_ptr_t(size_t sz, size_t alignment = 64) {
        ptr_ = Policy::alloc(sz, alignment);
        size_ = sz;
    }

    raii_ptr_t(void *ptr, size_t sz) : generic_ptr_base_t {ptr, sz} {}
    /**
     * Move another ptr to this
     * */
    raii_ptr_t(raii_ptr_t &&other) {
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
        size_ = other.size_;
        other.size_ = 0;
    }

    raii_ptr_t &operator=(raii_ptr_t &&other) {
        if (&other == this) { return *this; }
        if (ptr_) { Policy::dealloc(ptr_, size_); }
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
        size_ = other.size_;
        other.size_ = 0;
        return *this;
    }
    raii_ptr_t copy() const {
        if (ptr_) {
            size_t alignment = 64;
            auto newptr = Policy::alloc(size_, alignment);
            memcpy(newptr, ptr_, size_);
            return raii_ptr_t {newptr, size_};
        }
        return raii_ptr_t {};
    }
    raii_ptr_t() : generic_ptr_base_t {nullptr, 0} {}
    ~raii_ptr_t() {
        if (ptr_) { Policy::dealloc(ptr_, size_); }
    }
};

using generic_aligned_ptr_t = raii_ptr_t<aligned_ptr_policy_t>;

template <typename T, typename Base = generic_aligned_ptr_t>
struct aligned_ptr_t : Base {
    using Base::flush_cache;
    using Base::ptr_;
    using Base::size_;
    using Base::zeroout;
    /**
     * Creates a typed aligned memory buffer
     * @param counts the count of the elements
     * @param aligment the alignment in bytes
     * */
    aligned_ptr_t(size_t counts, size_t alignment = 64)
        : Base(counts * sizeof(T), alignment) {}
    aligned_ptr_t(aligned_ptr_t &&other) : Base(std::move(other)) {}
    aligned_ptr_t() : Base() {}
    aligned_ptr_t &operator=(aligned_ptr_t &&other) {
        Base::operator=(std::move(other));
        return *this;
    }
    aligned_ptr_t copy() { return Base::copy(); }

    T *get() { return reinterpret_cast<T *>(ptr_); }
    const T *get() const { return reinterpret_cast<const T *>(ptr_); }
    T &operator[](size_t index) { return get()[index]; }

    size_t size() const { return size_ / sizeof(T); }
    T *begin() { return get(); }
    const T *begin() const { return get(); }
    T *end() { return get() + size(); }
    const T *end() const { return get() + size(); }

    T *data() { return get(); };
    const T *data() const { return get(); };
    /**
     * Fills the buffer
     * @param f the functor which generates values for each elements of the
     *      buffer. Should have the prototype T func(size_t v);, where v is
     * the index in the buffer and it should returns the value to fill on
     * the index
     * */
    template <typename Func>
    void fill(Func f) {
        T *ptr = get();
        for (size_t i = 0; i < size_ / sizeof(T); i++) {
            ptr[i] = f(i);
        }
    }

private:
    aligned_ptr_t(Base &&other) : Base(std::move(other)) {}
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
