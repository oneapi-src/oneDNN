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

#include "common/serialization_stream.hpp"
#include "common/utils.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

namespace dnnl {
namespace impl {

struct quant_entry_t;
const quant_entry_t &default_quant_entry();

struct quant_entry_t : public c_compatible {
    quant_entry_t() = default;

    // `set(...)` approach is taken over constructors as the usage model assumes
    // the change of state of this object but it doesn't require its destruction
    // which would come with some performance price which prevails in this case.
    status_t set(int mask, data_type_t data_type) {
        return set(mask, data_type, 0, {});
    }
    status_t set(int mask, data_type_t data_type, int group_ndims,
            const dims_t group_dims) {
        mask_ = mask;
        data_type_ = data_type;
        group_ndims_ = group_ndims;
        if (group_ndims_ > 0) {
            utils::array_copy(group_dims_, group_dims, group_ndims_);
        }
        return status::success;
    }
    status_t set(const quant_entry_t &other) {
        return set(other.mask_, other.data_type_, other.group_ndims_,
                other.group_dims_);
    }

    quant_entry_t &operator=(const quant_entry_t &rhs) {
        auto st = this->set(rhs);
        assert(st == status::success);
        UNUSED(st);
        return *this;
    }

    bool has_default_values() const { return *this == default_quant_entry(); }
    bool has_default_groups() const {
        return this->group_ndims_ == default_quant_entry().group_ndims_;
    }

    int get_mask() const { return mask_; }
    data_type_t get_data_type() const { return data_type_; }
    dim_t get_group(int d) const {
        // If groups were not requested, return `1` for convenience.
        if (group_ndims_ == default_quant_entry().group_ndims_) return 1;
        // But if they were, any out of bound access would return `0` and likely
        // lead to a division by zero which is fast to catch.
        if (d >= group_ndims_) return 0;
        return group_dims_[d];
    }

    // Note: keep the definition here to satisfy the
    // `gtests/internals/test_comparison_operators` linking requirements which
    // mandates bodies to be in the header file.
    bool operator==(const quant_entry_t &rhs) const {
        return mask_ == rhs.mask_ && data_type_ == rhs.data_type_
                && group_ndims_ == rhs.group_ndims_
                && IMPLICATION(group_ndims_ > 0,
                        utils::array_cmp(
                                group_dims_, rhs.group_dims_, group_ndims_));
    }

    size_t get_hash() const;

    void serialize(serialization_stream_t &sstream) const;

    std::string get_verbose() const;

private:
    // Note: INT_MIN is used on purpose to avoid potential issues when
    // `(mask & bit)` expression will return `true`. `INT_MIN` is represented
    // as `10...0` in bits and will avoid such situations.
    int mask_ = INT_MIN;
    data_type_t data_type_ = data_type::undef;
    int group_ndims_ = 0;
    dims_t group_dims_ {};
};

std::ostream &operator<<(std::ostream &ss, const quant_entry_t &e);

struct scales_t : public c_compatible {
    scales_t() = default;

    const quant_entry_t &get(int arg) const {
        const auto it = scales_.find(arg);
        if (it == scales_.end()) return default_quant_entry();
        return it->second;
    }

    // See `set(...)` comment for `quant_entry_t` for a design choice
    // explanation.
    status_t set(int arg, int mask) {
        return set(arg, mask, default_data_type, 0, {});
    }
    status_t set(int arg, int mask, data_type_t data_type, int group_ndims,
            const dims_t group_dims) {
        if (!check_arg(arg)) return status::invalid_arguments;
        CHECK(scales_[arg].set(mask, data_type, group_ndims, group_dims));
        return status::success;
    }
    // Use this interface with `default_quant_entry` when need to remove a
    // specific scale.
    status_t set(int arg, const quant_entry_t &other) {
        return scales_[arg].set(other);
    }

    // This interface is different from the one below and is just a shortcut.
    bool has_default_values(int arg) const {
        return get(arg).has_default_values();
    }

    // This interface is used to make sure that other than `supported_args` have
    // default values. It's to make sure that non-allowed arguments were not
    // passed to the library.
    bool has_default_values(const std::vector<int> &supported_args = {}) const {
        auto predicate
                = [](const quant_entry_t &s) { return s.has_default_values(); };
        return has_default_property(supported_args, predicate);
    }

    // This interface is used to make sure that other than `supported_args` have
    // default values. It's to make sure that non-allowed arguments were not
    // passed to the library.
    bool has_default_data_type(
            const std::vector<int> &supported_args = {}) const {
        auto predicate = [](const quant_entry_t &s) {
            // Note: `data_type::undef` represents `default_quant_entry`.
            return utils::one_of(
                    s.get_data_type(), default_data_type, data_type::undef);
        };
        return has_default_property(supported_args, predicate);
    }
    // This interface checks specific argument. It exists because quant_entry_t
    // doesn't have a notion of default data_type, only scales do.
    // Note: can be removed once the library unconditionally supports data type
    // for scales for every implementation, then this call can be removed as to
    // make a proper load, the data type must be queried.
    bool has_default_data_type(int arg) const {
        // Note: `data_type::undef` represents `default_quant_entry`.
        return utils::one_of(
                get(arg).get_data_type(), default_data_type, data_type::undef);
    }

    // This interface is different from the one below and is just a shortcut.
    bool has_default_groups(int arg) const {
        return get(arg).has_default_groups();
    }

    // This interface is used to make sure that other than `supported_args` have
    // default values. It's to make sure that non-allowed arguments were not
    // passed to the library.
    bool has_default_groups(const std::vector<int> &supported_args = {}) const {
        auto predicate
                = [](const quant_entry_t &s) { return s.has_default_groups(); };
        return has_default_property(supported_args, predicate);
    }

    int get_mask(int arg) const { return get(arg).get_mask(); }
    data_type_t get_data_type(int arg) const {
        return get(arg).get_data_type();
    }
    dim_t get_group(int arg, int d) const { return get(arg).get_group(d); }

    bool operator==(const scales_t &rhs) const {
        return scales_ == rhs.scales_;
    }

    size_t get_hash() const;

    void serialize(serialization_stream_t &sstream) const;

    std::string get_verbose() const;

private:
    // Sorted property of `std::map` is used for hashing.
    std::map<int, quant_entry_t> scales_;
    static constexpr data_type_t default_data_type = data_type::f32;

    bool check_arg(int arg) const {
        // regular
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == sa) return true;
        }
        // binary
        for (const auto &sa : {DNNL_ARG_SRC_1}) {
            if (arg == sa) return true;
        }
        // concat
        if (arg & DNNL_ARG_MULTIPLE_SRC) return true;
        // depth-wise convolution post op
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == (DNNL_ARG_ATTR_POST_OP_DW | sa)) return true;
        }
        // sdpa
        if (arg == DNNL_ARG_SRC_2) return true;
        return false;
    }

    // The function makes sure that if any argument was specified by user, that
    // only `supported_args` have their value customized, rest unsupported
    // values were not updated.
    bool has_default_property(const std::vector<int> &supported_args,
            bool (*predicate)(const quant_entry_t &)) const {
        for (const auto &s : scales_) {
            // Arg passed the condition, check the next one.
            if (predicate(s.second)) continue;

            bool allow_non_default = false;
            for (const auto &supported_arg : supported_args)
                if (s.first == supported_arg) {
                    allow_non_default = true;
                    break;
                }
            if (allow_non_default) continue;
            return false;
        }
        return true;
    }
};

struct zero_points_t : public c_compatible {
    zero_points_t() = default;

    const quant_entry_t &get(int arg) const {
        const auto it = zero_points_.find(arg);
        if (it == zero_points_.end()) return default_quant_entry();
        return it->second;
    }

    // See `set(...)` comment for `quant_entry_t` for a design choice
    // explanation.
    status_t set(int arg, int mask) {
        return set(arg, mask, default_data_type, 0, {});
    }
    status_t set(int arg, int mask, data_type_t data_type, int group_ndims,
            const dims_t group_dims) {
        if (!check_arg(arg)) return status::invalid_arguments;
        CHECK(zero_points_[arg].set(mask, data_type, group_ndims, group_dims));
        return status::success;
    }
    // Use this interface with `default_quant_entry` when need to remove a
    // specific zero_point.
    status_t set(int arg, const quant_entry_t &other) {
        return zero_points_[arg].set(other);
    }

    // This interface is different from the one below and is just a shortcut.
    bool has_default_values(int arg) const {
        return get(arg).has_default_values();
    }

    // This interface is used to make sure that other than `supported_args` have
    // default values. It's to make sure that non-allowed arguments were not
    // passed to the library.
    bool has_default_values(const std::vector<int> &supported_args = {}) const {
        auto predicate
                = [](const quant_entry_t &s) { return s.has_default_values(); };
        return has_default_property(supported_args, predicate);
    }

    // This interface checks the content of all zero_points, and allows to
    // ignore certain arguments.
    bool has_default_data_type(
            const std::vector<int> &supported_args = {}) const {
        auto predicate = [](const quant_entry_t &s) {
            // Note: `data_type::undef` represents `default_quant_entry`.
            return utils::one_of(
                    s.get_data_type(), default_data_type, data_type::undef);
        };
        return has_default_property(supported_args, predicate);
    }
    // This interface checks specific argument. It exists because quant_entry_t
    // doesn't have a notion of default data_type, only zero_points do.
    // Note: can be removed once the library unconditionally supports data type
    // for zero_points for every implementation, then this call can be removed
    // as to make a proper load, the data type must be queried.
    bool has_default_data_type(int arg) const {
        // Note: `data_type::undef` represents `default_quant_entry`.
        return utils::one_of(
                get(arg).get_data_type(), default_data_type, data_type::undef);
    }

    // This interface is different from the one below and is just a shortcut.
    bool has_default_groups(int arg) const {
        return get(arg).has_default_groups();
    }

    // This interface is used to make sure that other than `supported_args` have
    // default values. It's to make sure that non-allowed arguments were not
    // passed to the library.
    bool has_default_groups(const std::vector<int> &supported_args = {}) const {
        auto predicate
                = [](const quant_entry_t &s) { return s.has_default_groups(); };
        return has_default_property(supported_args, predicate);
    }

    int get_mask(int arg) const { return get(arg).get_mask(); }
    data_type_t get_data_type(int arg) const {
        return get(arg).get_data_type();
    }
    dim_t get_group(int arg, int d) const { return get(arg).get_group(d); }

    bool operator==(const zero_points_t &rhs) const {
        return zero_points_ == rhs.zero_points_;
    }

    size_t get_hash() const;

    void serialize(serialization_stream_t &sstream) const;

    std::string get_verbose() const;

private:
    // Sorted property of `std::map` is used for hashing.
    std::map<int, quant_entry_t> zero_points_;
    static constexpr data_type_t default_data_type = data_type::s32;

    bool check_arg(int arg) const {
        // regular
        // gemm internal primitive would use DNNL_ARG_A, DNNL_ARG_B, DNNL_ARG_C,
        // which match to DNNL_ARG_WEIGHTS, DNNL_ARG_SRC, DNNL_ARG_DST. They
        // are defined in gpu internals, thus, not spelled here.
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == sa) return true;
        }
        // sdpa
        if (arg == DNNL_ARG_SRC_2) return true;
        return false;
    }

    // The function makes sure that if any argument was specified by user, that
    // only `supported_args` have their value customized, rest unsupported
    // values were not updated.
    bool has_default_property(const std::vector<int> &supported_args,
            bool (*predicate)(const quant_entry_t &)) const {
        for (const auto &s : zero_points_) {
            // Arg passed the condition, check the next one.
            if (predicate(s.second)) continue;

            bool allow_non_default = false;
            for (const auto &supported_arg : supported_args)
                if (s.first == supported_arg) {
                    allow_non_default = true;
                    break;
                }
            if (allow_non_default) continue;
            return false;
        }
        return true;
    }
};

} // namespace impl
} // namespace dnnl

#endif
