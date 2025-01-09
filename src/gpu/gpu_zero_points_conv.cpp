/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include <cassert>
#include <vector>

#include "common/convolution_pd.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/gpu_zero_points_conv.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

status_t create_zp_precompute_conv_pd(std::shared_ptr<primitive_desc_t> &retn,
        dnnl::impl::engine_t *eng, const primitive_attr_t &attr,
        const memory_desc_t *wei, const dim_t *idhw, const dim_t *odhw,
        const dim_t *pdhw, const dim_t *ddhw, data_type_t out_type,
        prop_kind_t prop, bool has_offset0) {
    using namespace memory_extra_flags;
    auto real_wei = *wei;
    const int off = (!idhw[1]) ? 2 + !idhw[2] : !idhw[0];
    const bool with_groups = (real_wei.ndims == (6 - off));
    if (real_wei.extra.flags & compensation_gpu_conv_asymmetric_src_swap) {
        static_assert(DNNL_MAX_NDIMS == 12, "DNNL_MAX_NDIMS is not 12");
        std::array<int, DNNL_MAX_NDIMS> perm_grp
                = {0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::array<int, DNNL_MAX_NDIMS> perm_no_grp
                = {1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        CHECK(memory_desc_permute_axes(real_wei, *wei,
                (with_groups) ? perm_grp.data() : perm_no_grp.data()));
    }
    real_wei.extra = memory_extra_desc_t();

    const auto &dims = real_wei.dims;
    const bool is_fwd = ((prop == prop_kind::forward_training)
            || (prop == prop_kind::forward_inference));
    const bool is_bwd_d = (prop == prop_kind::backward_data);
    assert((off < 3) && (real_wei.ndims >= 5 - off) && (is_fwd || is_bwd_d));
    MAYBE_UNUSED(is_fwd);

    using memory_dims = std::vector<dim_t>;
    memory_dims S1 {1, 1, 1};
    memory_dims P1 {0, 0, 0};
    // dim order for weights: [G,] OC, IC, [[[D,] H,] W]
    memory_dims dims_in {1,
            (with_groups) ? dims[0] * dims[2 - is_bwd_d] : dims[1 - is_bwd_d]};
    memory_dims dims_out {1,
            (with_groups) ? dims[0] * dims[1 + is_bwd_d] : dims[0 + is_bwd_d]};
    for (int i = off; i < 3; i++) {
        const auto k_idx = 2 + with_groups + i - off;
        const auto KD = (dims[k_idx] - 1) * (ddhw[i] + 1) + 1;
        dims_in.emplace_back(idhw[i]);
        dims_out.emplace_back(odhw[i]);
        P1[i] = dims_out.back() - dims_in.back() - 1 + KD - pdhw[i];
    }

    memory_desc_t in, out;
    CHECK(memory_desc_init_by_tag(out, int(dims_out.size()), dims_out.data(),
            out_type, format_tag::any));
    CHECK(memory_desc_init_by_tag(in, int(dims_in.size()), dims_in.data(),
            data_type::s8, format_tag::any));

    if (has_offset0) {
        auto out_type_size = types::data_type_size(out_type);
        auto offset0 = memory_desc_wrapper(real_wei).size(0, false);
        assert(offset0 % out_type_size == 0);
        out.offset0 = offset0 / out_type_size;
    }
    auto conv_desc = convolution_desc_t();
    CHECK(dnnl::impl::conv_desc_init(&conv_desc, prop,
            alg_kind::convolution_direct, (is_bwd_d) ? &out : &in, &real_wei,
            nullptr, (is_bwd_d) ? &in : &out, S1.data() + off, ddhw + off,
            pdhw + off, P1.data() + off));
    primitive_desc_iterator_t it(eng, (op_desc_t *)&conv_desc, &attr, nullptr);
    if (!it.is_initialized()) return status::out_of_memory;
    retn = *(++it);
    return (retn) ? status::success : status::unimplemented;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
