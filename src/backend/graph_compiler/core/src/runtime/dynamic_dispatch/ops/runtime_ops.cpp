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
#include <stdint.h>
#include "impl_type.hpp"
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/dynamic_dispatch/op_dispatch_tables.hpp>
#include <runtime/dynamic_dispatch/utils.hpp>
#include <util/utils.hpp>
namespace sc {
extern "C" void calculate_shape_of_tensor_op(void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, int *shape_idxs, int ndims) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);
    runtime::dispatch_key *out_fmt_st
            = reinterpret_cast<runtime::dispatch_key *>(out_fmt);
    runtime::dispatch_key *in_fmt_st
            = reinterpret_cast<runtime::dispatch_key *>(in_fmt);
    assert(out_dyn_tsr->ndims_ == ndims);
    constexpr int max_format_dims
            = runtime::dispatch_key::meta::FORMAT_BITS / 4;
    uint16_t count[max_format_dims] = {0};
    for (int n = 0; n < ndims; n++) {
        int ret = 1;
        bool first_block = true;
        for (int i = 0; i < in_fmt_st->ndims(); i++) {
            auto &idx = shape_idxs[n];
            auto axis = in_fmt_st->get(i);
            ++count[axis];
            if (axis == idx) {
                // vnni format may have count==3 but we do not concern.
                if (count[axis] == 1) {
                    ret *= in_dyn_tsr->dims_[axis];
                } else if (count[axis] == 2) {
                    uint16_t block;
                    if (first_block) {
                        block = in_fmt_st->get_block1();
                    } else {
                        block = in_fmt_st->get_block2();
                    }
                    ret = ret / in_dyn_tsr->dims_[axis]
                            * utils::divide_and_ceil(
                                    in_dyn_tsr->dims_[axis], block)
                            * block;
                }
            }
            if (count[axis] == 2) { first_block = false; }
        }
        reinterpret_cast<int *>(out_dyn_tsr->data_)[n] = ret;
    }
}
} // namespace sc
