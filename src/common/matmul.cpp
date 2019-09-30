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

#include <assert.h>

#include "dnnl.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::types;

status_t dnnl_matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_md, const memory_desc_t *weights_md,
        const memory_desc_t *bias_md, const memory_desc_t *dst_md) {
    bool args_ok = !any_null(matmul_desc, src_md, weights_md, dst_md);
    if (!args_ok) return status::invalid_arguments;

    auto op_d = matmul_desc_t();
    op_d.primitive_kind = primitive_kind::matmul;

    op_d.src_desc = *src_md;
    op_d.weights_desc = *weights_md;
    if (bias_md) op_d.bias_desc = *bias_md;
    op_d.dst_desc = *dst_md;

    const int ndims = op_d.dst_desc.ndims;
    bool ok = op_d.src_desc.ndims == ndims && op_d.weights_desc.ndims == ndims;

    int offset = 0;
    if (ndims == 3) {
        // check: batch
        ok = ok
                && everyone_is(op_d.dst_desc.dims[0], op_d.src_desc.dims[0],
                        op_d.weights_desc.dims[0]);
        offset = 1;
    }

    // check: m, n, k
    ok = ok && op_d.dst_desc.dims[offset + 0] == op_d.src_desc.dims[offset + 0]
            && op_d.dst_desc.dims[offset + 1]
                    == op_d.weights_desc.dims[offset + 1]
            && op_d.src_desc.dims[offset + 1]
                    == op_d.weights_desc.dims[offset + 0];
    if (!ok) return status::invalid_arguments;

    // bias check
    if (op_d.bias_desc.ndims != 0) {
        bool bias_ok = op_d.bias_desc.ndims == ndims
                && one_of(op_d.bias_desc.dims[0], 1, op_d.dst_desc.dims[0])
                && one_of(op_d.bias_desc.dims[1], 1, op_d.dst_desc.dims[1])
                && IMPLICATION(ndims == 3,
                        one_of(op_d.bias_desc.dims[2], 1,
                                op_d.dst_desc.dims[2]));
        if (!bias_ok) return status::invalid_arguments;
    }

    op_d.accum_data_type = types::default_accum_data_type(
            op_d.src_desc.data_type, op_d.weights_desc.data_type,
            op_d.dst_desc.data_type, prop_kind::forward);

    *matmul_desc = op_d;
    return status::success;
}
