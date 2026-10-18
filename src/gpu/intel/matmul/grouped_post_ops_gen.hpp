/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GPU_INTEL_MATMUL_GROUPED_POST_OPS_GEN_HPP
#define GPU_INTEL_MATMUL_GROUPED_POST_OPS_GEN_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/serialization.hpp"

#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

enum class po_kind_t {
    none,
    eltwise,
    binary_grouped_scale,
    binary_dense_scale,
    binary_nvfp4_scale
};

// Serializable description of a post-op entry, holding what is needed to
// generate its code. Floats are stored as their bit patterns to keep the
// structure trivially serializable.
struct grouped_po_desc_t {
    po_kind_t kind = po_kind_t::none;
    alg_kind_t alg = alg_kind::undef;
    uint32_t alpha = 0;
    uint32_t beta = 0;
    uint32_t scale = 0;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(grouped_po_desc_t);

bool is_eltwise_alg_supported(alg_kind_t alg);

int find_po_in_chain(const po_kind_t *po_chain, po_kind_t kind);

status_t check_post_op_chain(const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_desc, dim_t ngroups, po_kind_t *po_chain,
        data_type_t *scale_arr);

void set_binary_scales_dt(const primitive_attr_t &attr,
        const po_kind_t *po_chain, data_type_t *scale_arr);

std::string generate_post_ops_microgemm_header(
        const grouped_po_desc_t *po, int po_len);

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY

#endif // GPU_INTEL_MATMUL_GROUPED_POST_OPS_GEN_HPP
