/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "common/primitive_attr_quant.hpp"
#include "common/primitive_hashing.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {

const quant_entry_t &default_quant_entry() {
    static const quant_entry_t default_quant_entry;
    return default_quant_entry;
}

size_t quant_entry_t::get_hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, mask_);
    seed = hash_combine(seed, static_cast<size_t>(data_type_));
    seed = hash_combine(seed, group_ndims_);
    if (group_ndims_ > 0)
        seed = primitive_hashing::get_array_hash(
                seed, group_dims_, group_ndims_);
    return seed;
}

void quant_entry_t::serialize(serialization_stream_t &sstream) const {
    sstream.write(&mask_);
    sstream.write(&data_type_);
    sstream.write(&group_ndims_);
    if (group_ndims_ > 0) sstream.write(group_dims_, group_ndims_);
}

std::string quant_entry_t::get_verbose() const {
    std::string s;
    s.append(std::to_string(mask_));
    s.append(":").append(dnnl_dt2str(data_type_));
    if (group_ndims_ > 0) {
        s.append(":")
                .append(std::to_string(group_dims_[0]))
                .append("x")
                .append(std::to_string(group_dims_[1]));
    }
    return s;
}

std::ostream &operator<<(std::ostream &ss, const quant_entry_t &e) {
    ss << e.get_verbose();
    return ss;
}

size_t scales_t::get_hash() const {
    size_t seed = 0;
    // Go through scales for all arguments.
    for (const auto &e : scales_) {
        seed = hash_combine(seed, e.first);
        seed = hash_combine(seed, e.second.get_hash());
    }
    return seed;
}

void scales_t::serialize(serialization_stream_t &sstream) const {
    for (const auto &e : scales_) {
        sstream.write(&e.first);
        e.second.serialize(sstream);
    }
}

std::string scales_t::get_verbose() const {
    std::string s;
    std::string empty_delim, attr_delim = "+";
    std::string delim = empty_delim;
    for (const auto &scale : scales_) {
        const auto &q = scale.second;
        if (q.has_default_values()) continue;

        int arg = scale.first;
        s.append(delim)
                .append(arg2str(arg))
                .append(":")
                .append(q.get_verbose());
        delim = attr_delim;
    }
    return s;
}

size_t zero_points_t::get_hash() const {
    size_t seed = 0;
    // Go through zero_points for all arguments.
    for (const auto &e : zero_points_) {
        seed = hash_combine(seed, e.first);
        seed = hash_combine(seed, e.second.get_hash());
    }
    return seed;
}

void zero_points_t::serialize(serialization_stream_t &sstream) const {
    for (const auto &e : zero_points_) {
        sstream.write(&e.first);
        e.second.serialize(sstream);
    }
}

std::string zero_points_t::get_verbose() const {
    std::string s;
    std::string empty_delim, attr_delim = "+";
    std::string delim = empty_delim;
    for (const auto &zero_point : zero_points_) {
        const auto &q = zero_point.second;
        if (q.has_default_values()) continue;

        int arg = zero_point.first;
        s.append(delim)
                .append(arg2str(arg))
                .append(":")
                .append(q.get_verbose());
        delim = attr_delim;
    }
    return s;
}

} // namespace impl
} // namespace dnnl
