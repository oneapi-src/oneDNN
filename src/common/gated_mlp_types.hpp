/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef COMMON_GATED_MLP_TYPES_HPP
#define COMMON_GATED_MLP_TYPES_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/memory_desc.hpp"
#include "common/opdesc.hpp"

namespace dnnl {
namespace impl {

// A descriptor for a gated mlp (GLU) operation.
struct gated_mlp_desc_t : public op_desc_t  {
    gated_mlp_desc_t() : op_desc_t(primitive_kind::gated_mlp) {}

    std::unique_ptr<op_desc_t> clone() const override {
        return utils::make_unique<gated_mlp_desc_t>(*this);
    }

    memory_desc_t src_desc; /* input vector */
    memory_desc_t W_gate_desc; /* weights for gated portion */
    memory_desc_t W_up_desc; /* weights for linear portion */
    memory_desc_t W_down_desc; /* weights for final FC out */
    memory_desc_t dst_desc;

    //TODO: add enum for type of activation, swish relu sigmoid...
    //TODO: zp + scale?

    dnnl_dim_t mb_sz() const { return src_desc.dims[0]; }
    dnnl_dim_t ic_sz() const { return src_desc.dims[1]; }
    dnnl_dim_t oc_sz() const { return W_gate_desc.dims[1]; }
};

} // namespace impl
} // namespace dnnl

#endif // COMMON_GATED_MLP_TYPES_HPP
