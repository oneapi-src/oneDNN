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
extern "C" void calculate_shape_of_tensor_op(
        void *out, void *in, int *shape_idxs, int ndims) {
    runtime::dynamic_tensor_t *out_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(out);
    runtime::dynamic_tensor_t *in_dyn_tsr
            = reinterpret_cast<runtime::dynamic_tensor_t *>(in);
    assert(out_dyn_tsr->ndims_ == ndims);
    for (int n = 0; n < ndims; n++) {
        int ret = 1;
        int dim = static_cast<int>(in_dyn_tsr->dims_[shape_idxs[n]]);
        int block = runtime::get_dyn_cfg_single(dim);
        ret = utils::divide_and_ceil(dim, block) * block;
        reinterpret_cast<int *>(out_dyn_tsr->data_)[n] = ret;
    }
}
} // namespace sc
