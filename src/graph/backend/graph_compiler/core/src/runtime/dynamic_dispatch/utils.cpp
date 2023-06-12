/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include <limits>
#include <string.h>
#include "../dispatch_key.hpp"
#include "utils.hpp"
#include <util/simple_math.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
void deep_copy_dynamic_tensor(
        runtime::dynamic_tensor_t *out, const runtime::dynamic_tensor_t *in) {
    out->ndims_ = in->ndims_;
    out->dyn_mask_ = in->dyn_mask_;
    memcpy(out->dims_, in->dims_, sizeof(int64_t) * in->ndims_);
}

uint64_t calculate_blocking_dims(void *placeholder, uint64_t *format) {
    auto *dyn_tsr = reinterpret_cast<dynamic_tensor_t *>(placeholder);
    auto *fmt = reinterpret_cast<dispatch_key *>(format);
    uint64_t size = 1;
    if (fmt->is_plain()) {
        for (int i = 0; i < dyn_tsr->ndims_; i++) {
            size *= dyn_tsr->dims_[i];
        }
    } else {
        constexpr int max_format_dims = dispatch_key::meta::FORMAT_BITS / 4;
        uint16_t count[max_format_dims] = {0};
        uint16_t idx = 0;
        bool first_block = true;
        while (idx < max_format_dims) {
            auto axis = fmt->get(idx);
            if (axis == 0xF) { break; }
            count[axis]++;
            if (count[axis] == 2) {
                uint16_t block;
                if (first_block) {
                    first_block = false;
                    block = fmt->get_block1();
                } else {
                    block = fmt->get_block2();
                }
                size = size / dyn_tsr->dims_[axis]
                        * utils::divide_and_ceil(dyn_tsr->dims_[axis], block)
                        * block;
            } else {
                size = size * dyn_tsr->dims_[axis];
            }
            idx++;
        }
    }
    return size;
}

dispatch_key get_impl_dispatch_key(int impl) {
    dispatch_key ret(0);
    ret.set_impl_alg(impl);
    return ret;
}
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
