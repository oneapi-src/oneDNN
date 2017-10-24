/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::utils;

namespace mkldnn {
namespace impl {

status_t scales_t::set(int count, int mask, const float *scales) {
    if (count != count_)
        cleanup();

    count_ = count;
    mask_ = mask;

    if (count_ == 1) {
        scales_ = scales_buf_;
        utils::array_set(scales_, scales[0], scales_buf_size);
    } else {
        scales_ = (float *)impl::malloc(count_ * sizeof(*scales_), 64);
        if (scales_ == nullptr)
            return status::out_of_memory;

        for (int c = 0; c < count_; ++c)
            scales_[c] = scales[c];
    }

    return status::success;
}

}
}

status_t primitive_attr_t::set_round_mode(round_mode_t round_mode) {
    using namespace mkldnn::impl::round_mode;

    const bool ok = one_of(round_mode, nearest, down);
    if (!ok)
        return invalid_arguments;

    round_mode_ = round_mode;
    return success;
}

status_t mkldnn_primitive_attr_create(primitive_attr_t **attr) {
    if (attr == nullptr)
        return invalid_arguments;

    return safe_ptr_assign<mkldnn_primitive_attr>(*attr,
            new mkldnn_primitive_attr);
}

status_t mkldnn_primitive_attr_clone(primitive_attr_t **attr,
        const primitive_attr_t *existing_attr) {
    if (any_null(attr, existing_attr))
        return invalid_arguments;

    return safe_ptr_assign<mkldnn_primitive_attr>(*attr,
            existing_attr->clone());
}

status_t mkldnn_primitive_attr_destroy(primitive_attr_t *attr) {
    if (attr)
        delete attr;

    return success;
}

status_t mkldnn_primitive_attr_get_int_output_round_mode(
        const primitive_attr_t *attr, round_mode_t *round_mode) {
    if (any_null(attr, round_mode))
        return invalid_arguments;

    *round_mode = attr->round_mode_;

    return success;
}

status_t mkldnn_primitive_attr_set_int_output_round_mode(
        primitive_attr_t *attr, round_mode_t round_mode) {
    if (any_null(attr))
        return invalid_arguments;

    return attr->set_round_mode(round_mode);
}

status_t mkldnn_primitive_attr_get_output_scales(const primitive_attr_t *attr,
        int *count, int *mask, const float **scales) {
    if (any_null(attr, count, mask, scales))
        return invalid_arguments;

    *count = attr->output_scales_.count_;
    *mask = attr->output_scales_.mask_;
    *scales = attr->output_scales_.scales_;

    return success;
}

status_t mkldnn_primitive_attr_set_output_scales(primitive_attr_t *attr,
        int count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0;
    if (!ok)
        return invalid_arguments;

    return attr->output_scales_.set(count, mask, scales);
}
