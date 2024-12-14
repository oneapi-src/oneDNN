/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_ATTR_QUANT_HPP
#define COMMON_PRIMITIVE_ATTR_QUANT_HPP

// NOTE: Objects declared in this header are moved out from primitive_attr.hpp due
// to micro_sdpa primitive. Int8 support requires at least two primitive_attr
// objects to be used inside sdpa_desc_t object which triggers a deleted
// copy-ctor of primitive_attr_t, which is there because of RNN scales still
// rely on static scales and manage dynamically-allocated memory.
//
// As a result, micro_sdpa uses scales and zero-points objects directly and
// requires a dedicated header for that, otherwise, it's going to be a circular
// dependency between headers when it comes to inclusion of opdesc.hpp which
// sdpa_desc_t is a part of.

#include "utils.hpp"

#include <algorithm>
#include <map>
#include <vector>

namespace dnnl {
namespace impl {

const primitive_attr_t &default_attr();
struct runtime_scales_t;
const runtime_scales_t &default_runtime_scale();

struct runtime_scales_t : public c_compatible {
    runtime_scales_t() = default;

    runtime_scales_t &operator=(const runtime_scales_t &rhs) {
        mask_ = rhs.mask_;
        is_set_ = rhs.is_set_;
        ndims_ = rhs.ndims_;
        if (ndims_ > 0) utils::array_copy(group_dims_, rhs.group_dims_, ndims_);
        data_type_ = rhs.data_type_;
        return *this;
    }

    status_t set(int mask) { return set(0, mask, {}, data_type::f32); }

    status_t set(int ndims, int mask, const dims_t group_dims,
            data_type_t data_type = data_type::f32) {
        mask_ = mask;
        is_set_ = true;
        ndims_ = ndims;
        if (ndims > 0) utils::array_copy(group_dims_, group_dims, ndims);
        data_type_ = data_type;
        return status::success;
    }

    bool operator==(const runtime_scales_t &rhs) const {
        return mask_ == rhs.mask_ && is_set_ == rhs.is_set_
                && ndims_ == rhs.ndims_
                && IMPLICATION(ndims_ > 0,
                        utils::array_cmp(group_dims_, rhs.group_dims_, ndims_))
                && data_type_ == rhs.data_type_;
    }

    bool has_default_values() const { return *this == default_runtime_scale(); }

    bool has_default_groups() const { return 0 == ndims_; }
    bool has_default_data_type() const { return data_type_ == data_type::f32; }

    // TODO: replace with `-1` to remove `is_set_`.
    // Hide `mask_` under `private:` to force interface usage.
    int mask_ = 0;
    bool is_set_ = false;
    int ndims_ = 0;
    dims_t group_dims_ = {};
    data_type_t data_type_ = data_type::f32;
};

struct arg_scales_t : public c_compatible {
    arg_scales_t() = default;

    const runtime_scales_t &get(int arg) const {
        static const runtime_scales_t default_scales;
        const auto it = scales_.find(arg);
        if (it == scales_.end()) return default_scales;
        return it->second;
    }

    status_t set(int arg, const runtime_scales_t &scale) {
        if (!check_arg(arg)) return status::invalid_arguments;
        scales_[arg] = scale;
        return status::success;
    }

    bool operator==(const arg_scales_t &rhs) const {
        return scales_ == rhs.scales_;
    }

    bool has_default_values(const std::vector<int> &skip_args = {}) const {
        auto predicate = [](const runtime_scales_t &s) {
            return s.has_default_values();
        };
        return has_default_property(skip_args, predicate);
    }

    bool has_default_data_type(const std::vector<int> &skip_args = {}) const {
        auto predicate = [](const runtime_scales_t &s) {
            return s.has_default_data_type();
        };
        return has_default_property(skip_args, predicate);
    }

    bool has_default_groups(const std::vector<int> &skip_args = {}) const {
        auto predicate = [](const runtime_scales_t &s) {
            return s.has_default_groups();
        };
        return has_default_property(skip_args, predicate);
    }

    status_t set(int arg, int mask) {
        return set(arg, mask, 0, {}, data_type::f32);
    }

    status_t set(int arg, int mask, int ndims, const dims_t group_dims,
            data_type_t data_type) {
        if (!check_arg(arg)) return status::invalid_arguments;
        return scales_[arg].set(ndims, mask, group_dims, data_type);
    }

    // TODO: move to `private` and keep a single interface per entry.
    status_t get(int arg, int *mask, bool *is_set, int *ndims = nullptr,
            dims_t group_dims = nullptr,
            data_type_t *data_type = nullptr) const {
        if (!check_arg(arg)) return status::invalid_arguments;
        const auto &s = get(arg);
        if (mask) *mask = s.mask_;
        if (is_set) *is_set = s.is_set_;
        if (ndims) *ndims = s.ndims_;
        if (group_dims && s.ndims_ > 0)
            utils::array_copy(group_dims, s.group_dims_, s.ndims_);
        if (data_type) *data_type = s.data_type_;
        return status::success;
    }

    data_type_t get_data_type(int arg) const {
        data_type_t data_type;
        auto st = get(arg, nullptr, nullptr, nullptr, nullptr, &data_type);
        if (st != status::success) return data_type::undef;
        return data_type;
    }

    status_t reset(int arg) {
        if (!check_arg(arg)) return status::invalid_arguments;
        const auto it = scales_.find(arg);
        if (it != scales_.end()) scales_.erase(it);
        return status::success;
    }

    status_t copy_from(const arg_scales_t &other) {
        for (auto it = other.scales_.begin(); it != other.scales_.end(); ++it) {
            // Find an entry that can match the arguments without constructing a
            // new object.
            if (scales_.count(it->first) == 1) {
                auto &entry = scales_[it->first];
                if (entry == it->second) continue;
            }

            CHECK(set(it->first, it->second));
        }
        return status::success;
    }

    std::map<int, runtime_scales_t> scales_;

private:
    bool check_arg(int arg) const {
        // binary
        for (const auto &sa : {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1}) {
            if (arg == sa) return true;
        }
        // concat
        if (arg & DNNL_ARG_MULTIPLE_SRC) return true;
        // convolution
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == sa) return true;
        }
        // depth-wise convolution post op
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == (DNNL_ARG_ATTR_POST_OP_DW | sa)) return true;
        }
        // sdpa
        if (arg == DNNL_ARG_SRC_2) return true;
        return false;
    }

    bool has_default_property(const std::vector<int> &skip_args,
            bool (*predicate)(const runtime_scales_t &)) const {
        for (const auto &s : scales_) {
            if (!predicate(s.second)) {
                bool skip = false;
                for (const auto &skip_a : skip_args)
                    if (s.first == skip_a) {
                        skip = true;
                        break;
                    }
                if (skip) continue;
                return false;
            }
        }
        return true;
    }
};

struct zero_points_t : public c_compatible {
    bool operator==(const zero_points_t &rhs) const {
        return mask_src == rhs.mask_src && mask_wei == rhs.mask_wei
                && mask_dst == rhs.mask_dst && is_set_src == rhs.is_set_src
                && is_set_wei == rhs.is_set_wei && is_set_dst == rhs.is_set_dst
                && data_type_wei == rhs.data_type_wei
                && group_ndims_wei == rhs.group_ndims_wei
                && IMPLICATION(group_ndims_wei > 0,
                        utils::array_cmp(group_dims_wei, rhs.group_dims_wei,
                                group_ndims_wei))
                && data_type_src == rhs.data_type_src
                && group_ndims_src == rhs.group_ndims_src
                && IMPLICATION(group_ndims_src > 0,
                        utils::array_cmp(group_dims_src, rhs.group_dims_src,
                                group_ndims_src));
    }

    // arg-specific checks
    bool common(int arg) const { return get_mask(arg) == 0; }
    bool per_dim_1(int arg) const { return get_mask(arg) == 2; }
    bool has_default_values(int arg) const {
        return is_set(arg) == false && has_default_data_type(arg);
    }
    bool has_default_groups(int arg) const {
        return IMPLICATION(arg == DNNL_ARG_WEIGHTS, group_ndims_wei == 0)
                && IMPLICATION(arg == DNNL_ARG_SRC, group_ndims_src == 0);
    }
    bool has_default_data_type(int arg) const {
        return get_data_type(arg) == data_type::s32;
    }
    // same checks but for all supported arguments at once
    bool common() const { return check_all(&zero_points_t::common); }
    bool has_default_values() const {
        return check_all(&zero_points_t::has_default_values);
    }
    bool has_default_groups() const {
        return check_all(&zero_points_t::has_default_groups);
    }
    bool has_default_data_type() const {
        return check_all(&zero_points_t::has_default_data_type);
    }

    status_t get(int arg, int *mask, data_type_t *dt = nullptr) const;

    int get(int arg) const; // Returns 0 if dimension is unset

    data_type_t get_data_type(int arg) const {
        if (arg == DNNL_ARG_WEIGHTS) return data_type_wei;
        if (arg == DNNL_ARG_SRC) return data_type_src;
        return data_type::s32;
    }

    const dim_t *get_groups(int arg) const {
        if (arg == DNNL_ARG_WEIGHTS) return group_dims_wei;
        if (arg == DNNL_ARG_SRC) return group_dims_src;
        return nullptr;
    }

    int get_groups_ndims(int arg) const {
        if (arg == DNNL_ARG_WEIGHTS) return group_ndims_wei;
        if (arg == DNNL_ARG_SRC) return group_ndims_src;
        return 0;
    }

    status_t set(int arg, int mask, int ndims, const dims_t group_dims,
            data_type_t data_type);

    status_t set(int arg, int mask) {
        return set(arg, mask, 0, nullptr, data_type::s32);
    }

    status_t set(int arg) { return set(arg, 0); }

private:
    bool is_set_src = false, is_set_wei = false, is_set_dst = false;
    int mask_src = 0, mask_wei = 0, mask_dst = 0;
    data_type_t data_type_wei = data_type::s32;
    int group_ndims_wei = 0;
    dims_t group_dims_wei {};
    // TODO: A temporary solution until a single quant abstraction is
    // introduced.
    data_type_t data_type_src = data_type::s32;
    int group_ndims_src = 0;
    dims_t group_dims_src {};

    int get_mask(int arg) const {
        int mask = 0;
        switch (arg) {
            case DNNL_ARG_SRC: mask = mask_src; break;
            case DNNL_ARG_WEIGHTS: mask = mask_wei; break;
            case DNNL_ARG_DST: mask = mask_dst; break;
            default: mask = 0;
        }
        return mask;
    }

    bool is_set(int arg) const {
        bool arg_is_set = false;
        switch (arg) {
            case DNNL_ARG_SRC: arg_is_set = is_set_src; break;
            case DNNL_ARG_WEIGHTS: arg_is_set = is_set_wei; break;
            case DNNL_ARG_DST: arg_is_set = is_set_dst; break;
            default: arg_is_set = 0;
        }
        return arg_is_set;
    }

    bool check_all(bool (zero_points_t::*f)(int) const) const {
        for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST})
            if (!(this->*f)(arg)) return false;
        return true;
    }
};

} // namespace impl
} // namespace dnnl

#endif
