/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef PRIMITIVE_HASHING_HPP
#define PRIMITIVE_HASHING_HPP

#include "c_types_map.hpp"
#include "mkldnn.h"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace primitive_hashing {

struct key_t {
    key_t(mkldnn_primitive_kind_t primitive_kind, const op_desc_t *op_desc,
            const primitive_attr_t *attr, const std::type_index &impl_id,
            int impl_nthr)
        : primitive_kind_(primitive_kind)
        , op_desc_(op_desc)
        , attr_(attr)
        , impl_id_(impl_id)
        , impl_nthr_(impl_nthr) {}

    bool operator==(const key_t &rhs) const {
        MKLDNN_SHORT_CIRCUIT_SELF_COMPARISON(rhs);
        return false;
    }

private:
    mkldnn_primitive_kind_t primitive_kind_;
    const op_desc_t *op_desc_;
    const primitive_attr_t *attr_;
    std::type_index impl_id_;
    int impl_nthr_;
};

} // namespace primitive_hashing
} // namespace impl
} // namespace mkldnn

// inject a specialization of std::hash for key_t in std namespace
namespace std {
template <>
struct hash<mkldnn::impl::primitive_hashing::key_t> {
    using argument_type = mkldnn::impl::primitive_hashing::key_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &key) const {
        size_t seed = 0;
        return seed;
    }
};
} // namespace std

#endif
