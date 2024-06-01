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

#include <map>
#include <vector>

#include "oneapi/dnnl/dnnl.h"

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

class scales_query_t {
public:
    bool has_default_values() const { return scales_.has_default_values(); }
    int get_mask() const { return scales_.mask_; }
    size_t get_count() const { return count_; }
    memory_storage_t &get_scales(const exec_ctx_t &ctx) const {
        return CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | arg_);
    }

    scales_query_t() = default;
    scales_query_t(const primitive_attr_t *attr, const memory_desc_wrapper &mdw,
            int arg)
        : arg_(arg) {
        scales_ = attr->scales_.get(arg);
        count_ = get_attr_oscales_count(scales_.mask_, mdw);
    }

private:
    runtime_scales_t scales_;
    dim_t count_ = 0;
    int arg_ = 0;
};

class zero_points_query_t {
public:
    bool has_default_values() const { return zps_.has_default_values(arg_); }
    int get_mask() const {
        int mask = zps_.get(arg_);
        return mask;
    }
    size_t get_count() const { return count_; }
    memory_storage_t &get_zero_points(const exec_ctx_t &ctx) const {
        return CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | arg_);
    }

    zero_points_query_t() = default;
    zero_points_query_t(const primitive_attr_t *attr,
            const memory_desc_wrapper &mdw, int arg)
        : arg_(arg) {
        zps_ = attr->zero_points_;
        int mask = zps_.get(arg);
        count_ = get_attr_oscales_count(mask, mdw);
    }

private:
    zero_points_t zps_;
    dim_t count_ = 0;
    int arg_ = 0;
};

struct quantization_t {
public:
    bool with_scale() const { return !scale_.has_default_values(); }
    int scale_mask() const { return scale_.get_mask(); }
    size_t num_scales() const { return scale_.get_count(); }
    memory_storage_t &scales(const exec_ctx_t &ctx) const {
        return scale_.get_scales(ctx);
    }

    bool with_zp() const { return !zp_.has_default_values(); }
    int zp_mask() const { return zp_.get_mask(); }
    size_t num_zps() const { return zp_.get_count(); }
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
