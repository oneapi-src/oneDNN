/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_GPU_PRIMITIVE_ATTR_HPP
#define GPU_GPU_PRIMITIVE_ATTR_HPP

#include "common/primitive_attr.hpp"
#include "common/serialization_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_primitive_attr_t : public primitive_attr_item_t {
    gpu_primitive_attr_t(int threads_per_eu = 0)
        : threads_per_eu_(threads_per_eu) {}

    std::unique_ptr<primitive_attr_item_t> clone() const override {
        return utils::make_unique<gpu_primitive_attr_t>(threads_per_eu_);
    }

    bool has_default_values() const override { return threads_per_eu_ == 0; }

    bool is_equal(const primitive_attr_item_t &other) const override {
        auto *other_ptr = utils::downcast<const gpu_primitive_attr_t *>(&other);
        return threads_per_eu_ == other_ptr->threads_per_eu_;
    }

    size_t get_hash() const override { return threads_per_eu_; }

    void serialize(serialization_stream_t &stream) const override {
        stream.write(&threads_per_eu_);
    }

    int threads_per_eu() const { return threads_per_eu_; }

private:
    int threads_per_eu_;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
