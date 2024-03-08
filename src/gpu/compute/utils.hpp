/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_COMPUTE_UTILS_HPP
#define GPU_COMPUTE_UTILS_HPP

#include <array>
#include <cassert>
#include <sstream>
#include <tuple>
#include <vector>

#include "common/utils.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

using binary_t = std::vector<uint8_t>;
using device_uuid_t = std::tuple<uint64_t, uint64_t>;

struct device_uuid_hasher_t {
    size_t operator()(const device_uuid_t &uuid) const {
        const size_t seed = hash_combine(0, std::get<0>(uuid));
        return hash_combine(seed, std::get<1>(uuid));
    }
};

// Stores global/local ranges to use for kernel enqueueing
class nd_range_t {
public:
    static constexpr size_t max_ndims = 3;
    using work_size_t = std::array<size_t, nd_range_t::max_ndims>;
    nd_range_t() {
        global_range_[0] = 1;
        global_range_[1] = 1;
        global_range_[2] = 1;
        with_local_range_ = false;
    }

    nd_range_t(size_t n, const size_t *global_range,
            const size_t *local_range = nullptr) {

        assert(n <= max_ndims);
        with_local_range_ = bool(local_range);

        for (size_t i = 0; i < max_ndims; ++i) {
            global_range_[i] = (i < n) ? global_range[i] : 1;
            if (with_local_range_) {
                local_range_[i] = (i < n) ? local_range[i] : 1;
            }
        }
    }

    nd_range_t(const size_t *global_range, const size_t *local_range = nullptr)
        : nd_range_t(max_ndims, global_range, local_range) {}

    template <typename int_type>
    nd_range_t(std::initializer_list<int_type> global_range,
            std::initializer_list<int_type> local_range = {}) {
        with_local_range_ = (local_range.size() > 0);
        if (with_local_range_) {
            assert(global_range.size() == local_range.size());
        }
        size_t n = global_range.size();
        for (size_t i = 0; i < max_ndims; i++) {
            global_range_[i] = (i < n) ? *(global_range.begin() + i) : 1;
            if (with_local_range_) {
                local_range_[i] = (i < n) ? *(local_range.begin() + i) : 1;
            }
        }
    }

    template <typename int_type>
    nd_range_t(const std::vector<int_type> &global_range,
            const std::vector<int_type> &local_range = {}) {
        with_local_range_ = (local_range.size() > 0);
        if (with_local_range_) {
            assert(global_range.size() == local_range.size());
        }
        size_t n = global_range.size();
        for (size_t i = 0; i < max_ndims; i++) {
            global_range_[i] = (i < n) ? global_range[i] : 1;
            if (with_local_range_) {
                local_range_[i] = (i < n) ? local_range[i] : 1;
            }
        }
    }

    size_t ndims() const { return max_ndims; }
    const size_t *global_range() const { return global_range_; }

    const size_t *local_range() const {
        return with_local_range_ ? local_range_ : nullptr;
    }

    bool is_zero() const {
        return global_range_[0] == 0 || global_range_[1] == 0
                || global_range_[2] == 0;
    }

    std::string str() const {
        std::stringstream oss;
        oss << "gws = [" << global_range_[0] << ", " << global_range_[1] << ", "
            << global_range_[2] << "] lws = ";
        if (local_range()) {
            oss << "[" << local_range_[0] << ", " << local_range_[1] << ", "
                << local_range_[2] << "]";
        } else {
            oss << "(nil)";
        }
        return oss.str();
    }

private:
    size_t global_range_[max_ndims];
    size_t local_range_[max_ndims];
    bool with_local_range_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_UTILS_HPP
