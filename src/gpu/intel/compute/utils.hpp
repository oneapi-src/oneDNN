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

#ifndef GPU_INTEL_COMPUTE_UTILS_HPP
#define GPU_INTEL_COMPUTE_UTILS_HPP

#include <array>
#include <cassert>
#include <sstream>
#include <tuple>
#include <vector>

#include "common/utils.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

class range_t {
public:
    static constexpr size_t max_ndims = 3;
    constexpr range_t() = default;
    range_t(size_t dim0) : ndims_(1), dims_ {dim0, 0, 0} {}
    range_t(size_t dim0, size_t dim1) : ndims_(2), dims_ {dim0, dim1, 0} {}
    range_t(size_t dim0, size_t dim1, size_t dim2)
        : ndims_(3), dims_ {dim0, dim1, dim2} {}

    template <typename int_type>
    range_t(const std::vector<int_type> &dims) {
        gpu_assert(dims.size() <= max_ndims)
                << "Too many dimensions for range_t";
        ndims_ = dims.size();
        for (size_t i = 0; i < dims.size(); i++) {
            dims_[i] = into<size_t>(dims[i]);
        }
    }

    // Initialize a valid nd-range with one element, to be mutated
    static range_t one(size_t ndims = max_ndims) {
        range_t ret;
        ret.ndims_ = ndims;
        for (size_t i = 0; i < ndims; i++) {
            ret.dims_[i] = 1;
        }
        return ret;
    }

    // Initialize a valid nd-range with zero elements, to have dimensions set individually
    static range_t empty(size_t ndims = max_ndims) {
        range_t ret;
        ret.ndims_ = ndims;
        return ret;
    }

    size_t &operator[](size_t idx) {
        gpu_assert(idx < ndims_) << "range index " << idx
                                 << " overflows range ndims of " << ndims_;
        return dims_[idx];
    }
    size_t operator[](size_t idx) const {
        gpu_assert(idx < ndims_) << "range index " << idx
                                 << " overflows range ndims of " << ndims_;
        return dims_[idx];
    }
    size_t ndims() const { return ndims_; }
    size_t nelems() const {
        if (ndims_ == 0) return 0;
        return utils::array_product(dims_.data(), ndims_);
    };
    const size_t *data() const { return dims_.data(); }

    bool operator==(const range_t &rhs) const {
        if (ndims_ != rhs.ndims_) return false;
        for (size_t i = 0; i < ndims_; i++) {
            if (dims_[i] != rhs.dims_[i]) return false;
        }
        return true;
    }
    bool operator!=(const range_t &rhs) const { return !operator==(rhs); }

    operator bool() const { return ndims_ > 0; }

    std::string str() const {
        if (ndims_ == 0) return "(nil)";

        std::stringstream oss;
        oss << "[";
        for (size_t i = 0; i < ndims(); i++) {
            if (i > 0) oss << ", ";
            oss << dims_[i];
        }
        oss << "]";
        return oss.str();
    }

private:
    size_t ndims_ = 0;
    std::array<size_t, max_ndims> dims_ = {0, 0, 0};
};

// Stores global/local ranges to use for kernel enqueueing
class nd_range_t {
public:
    nd_range_t() = default;
    explicit nd_range_t(
            const range_t &global_range, const range_t &local_range = range_t())
        : global_range_(global_range), local_range_(local_range) {
        if (local_range_) {
            gpu_assert(local_range_.ndims() == global_range_.ndims())
                    << "Incompatible gws/lws dimensions";
            for (size_t i = 0; i < local_range_.ndims(); i++) {
                gpu_assert(local_range_[i] != 0) << "Invalid local work size";
            }
        }
    }

    size_t ndims() const { return global_range_.ndims(); }
    const range_t &global_range() const { return global_range_; }

    const range_t &local_range() const { return local_range_; }

    bool is_zero() const { return (global_range_.nelems() == 0); }

    std::string str() const {
        std::stringstream oss;
        oss << "gws = " << global_range_.str();
        oss << " lws = ";
        if (local_range_) {
            oss << local_range_.str();
        } else {
            oss << "(nil)";
        }
        return oss.str();
    }

private:
    range_t global_range_;
    range_t local_range_;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_COMPUTE_UTILS_HPP
