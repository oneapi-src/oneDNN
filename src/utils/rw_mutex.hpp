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

#ifndef UTILS_RW_MUTEX_HPP
#define UTILS_RW_MUTEX_HPP

#include <memory>

// The following code is derived from
// https://github.com/oneapi-src/oneDNN/blob/master/src/common/rw_mutex.hpp

// As shared_mutex was introduced only in C++17
// a custom implementation of read-write lock pattern is used
namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

struct rw_mutex_t {
    rw_mutex_t();
    void lock_read();
    void lock_write();
    void unlock_read();
    void unlock_write();
    ~rw_mutex_t();

    rw_mutex_t(const rw_mutex_t &) = delete;
    rw_mutex_t &operator=(const rw_mutex_t &) = delete;

private:
    struct rw_mutex_impl_t;
    std::unique_ptr<rw_mutex_impl_t> rw_mutex_impl_;
};

struct lock_read_t {
    explicit lock_read_t(rw_mutex_t &rw_mutex);
    ~lock_read_t();

    lock_read_t(const lock_read_t &) = delete;
    lock_read_t &operator=(const lock_read_t &) = delete;

private:
    rw_mutex_t &rw_mutex_;
};

struct lock_write_t {
    explicit lock_write_t(rw_mutex_t &rw_mutex_t);
    ~lock_write_t();

    lock_write_t(const lock_write_t &) = delete;
    lock_write_t &operator=(const lock_write_t &) = delete;

private:
    rw_mutex_t &rw_mutex_;
};

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
