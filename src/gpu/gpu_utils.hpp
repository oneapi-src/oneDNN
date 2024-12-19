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

#ifndef GPU_GPU_UTILS_HPP
#define GPU_GPU_UTILS_HPP

#include <cassert>
#include <map>
#include <vector>

#include "common/convolution_pd.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/primitive_exec_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

inline dim_t get_attr_oscales_count(int mask, const memory_desc_wrapper &md) {
    dim_t count = 1;
    if (mask == 0) return count;

    for (int d = 0; d < md.ndims(); d++) {
        const int dim_mask = 1 << d;
        if (dim_mask & mask) count *= md.dims()[d];
    }

    return count;
}

inline status_t create_zp_precompute_conv_pd(
        std::shared_ptr<primitive_desc_t> &retn, dnnl::impl::engine_t *eng,
        const primitive_attr_t &attr, const memory_desc_t *wei,
        const dim_t *idhw, const dim_t *odhw, const dim_t *pdhw,
        const dim_t *ddhw, data_type_t out_type, prop_kind_t prop) {
    using namespace memory_extra_flags;
    auto real_wei = *wei;
    const int off = (!idhw[1]) ? 2 + !idhw[2] : !idhw[0];
    const bool with_groups = (real_wei.ndims == (6 - off));
    if (real_wei.extra.flags & compensation_gpu_conv_asymmetric_src_swap) {
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

    auto out_type_size = types::data_type_size(out_type);
    auto offset0 = memory_desc_wrapper(real_wei).size(0, false);
    assert(offset0 % out_type_size == 0);
    out.offset0 = offset0 / out_type_size;

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

class scales_query_t {
public:
    bool has_default_values() const { return scales_.has_default_values(); }
    int get_mask() const { return scales_.mask_; }
    size_t get_count() const { return count_; }
    data_type_t get_data_type() const { return scales_.data_type_; }
    dim_t get_group() const {
        if (scales_.ndims_ < 2) return 1;
        const auto g0 = scales_.group_dims_[0];
        const auto g1 = scales_.group_dims_[1];
        assert(utils::one_of(1, g0, g1));
        return g0 > 1 ? g0 : g1;
    }
    // Returns a dimension to which the group should be applied.
    int get_group_dim() const {
        // If groups are not identified, they should be set to `1`, and
        // it shouldn't hurt to divide by 1 any dim. Just use 0th for that.
        if (scales_.ndims_ < 2) return 0;
        const auto g0 = scales_.group_dims_[0];
        const auto g1 = scales_.group_dims_[1];
        assert(utils::one_of(1, g0, g1));
        UNUSED(g1);
        const int g_dim = g0 > 1 ? 0 : 1;
        return ndims_ - scales_.ndims_ + g_dim;
    }

    memory_storage_t &get_scales(const exec_ctx_t &ctx) const {
        return CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | arg_);
    }

    scales_query_t() = default;
    scales_query_t(const primitive_attr_t *attr, const memory_desc_wrapper &mdw,
            int arg)
        : arg_(arg), ndims_(mdw.ndims()) {
        scales_ = attr->scales_.get(arg);
        count_ = get_attr_oscales_count(scales_.mask_, mdw);
    }

private:
    runtime_scales_t scales_;
    dim_t count_ = 0;
    int arg_ = 0;
    int ndims_ = 0;
};

class zero_points_query_t {
public:
    bool has_default_values() const { return zps_.has_default_values(arg_); }
    int get_mask() const {
        int mask = zps_.get(arg_);
        return mask;
    }
    size_t get_count() const { return count_; }
    data_type_t get_data_type() const { return zps_.get_data_type(arg_); }
    dim_t get_group() const {
        if (zps_.get_groups_ndims(arg_) < 2) return 1;
        const auto g0 = zps_.get_groups(arg_)[0];
        const auto g1 = zps_.get_groups(arg_)[1];
        assert(utils::one_of(1, g0, g1));
        return g0 > 1 ? g0 : g1;
    }
    // Returns a dimension to which the group should be applied.
    int get_group_dim() const {
        // If groups are not identified, they should be set to `1`, and
        // it shouldn't hurt to divide by 1 any dim. Just use 0th for that.
        if (zps_.get_groups_ndims(arg_) < 2) return 0;
        const auto g0 = zps_.get_groups(arg_)[0];
        const auto g1 = zps_.get_groups(arg_)[1];
        assert(utils::one_of(1, g0, g1));
        UNUSED(g1);
        const int g_dim = g0 > 1 ? 0 : 1;
        return ndims_ - zps_.get_groups_ndims(arg_) + g_dim;
    }

    memory_storage_t &get_zero_points(const exec_ctx_t &ctx) const {
        return CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | arg_);
    }

    zero_points_query_t() = default;
    zero_points_query_t(const primitive_attr_t *attr,
            const memory_desc_wrapper &mdw, int arg)
        : arg_(arg), ndims_(mdw.ndims()) {
        zps_ = attr->zero_points_;
        int mask = zps_.get(arg);
        count_ = get_attr_oscales_count(mask, mdw);
    }

private:
    zero_points_t zps_;
    dim_t count_ = 0;
    int arg_ = 0;
    int ndims_ = 0;
};

struct quantization_t {
public:
    bool with_scale() const { return !scale_.has_default_values(); }
    int scale_mask() const { return scale_.get_mask(); }
    size_t num_scales() const { return scale_.get_count(); }
    data_type_t scale_dt() const { return scale_.get_data_type(); }
    dim_t scale_group() const { return scale_.get_group(); }
    int scale_group_dim() const { return scale_.get_group_dim(); }
    memory_storage_t &scales(const exec_ctx_t &ctx) const {
        return scale_.get_scales(ctx);
    }

    bool with_zp() const { return !zp_.has_default_values(); }
    int zp_mask() const { return zp_.get_mask(); }
    size_t num_zps() const { return zp_.get_count(); }
    data_type_t zp_dt() const { return zp_.get_data_type(); }
    dim_t zp_group() const { return zp_.get_group(); }
    int zp_group_dim() const { return zp_.get_group_dim(); }
    memory_storage_t &zero_points(const exec_ctx_t &ctx) const {
        return zp_.get_zero_points(ctx);
    }

    quantization_t(const primitive_attr_t *attr, const memory_desc_wrapper &mdw,
            int arg)
        : scale_(attr, mdw, arg), zp_(attr, mdw, arg) {}
    quantization_t() = default;

private:
    scales_query_t scale_;
    zero_points_query_t zp_;
};

struct sum_quantization_t {
public:
    bool with_scale() const { return scale_ != 0; }
    int scale_mask() const { return 0; }
    size_t num_scales() const { return (size_t)(with_scale()); }
    float scales() const { return scale_; }

    bool with_zp() const { return zp_ != 0; }
    int zp_mask() const { return 0; }
    size_t num_zps() const { return (size_t)(with_zp()); }
    int zero_points() const { return zp_; }

    sum_quantization_t(const primitive_attr_t *attr) {
        const auto &post_ops = attr->post_ops_;
        const int sum_idx = post_ops.find(primitive_kind::sum);
        if (sum_idx != -1) {
            const auto &sum = post_ops.entry_[sum_idx].sum;
            scale_ = sum.scale;
            zp_ = sum.zero_point;
        }
    }
    sum_quantization_t() = default;

private:
    float scale_ = 0;
    int zp_ = 0;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
