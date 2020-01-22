/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#ifndef PRIMITIVE_ATTR_HPP
#define PRIMITIVE_ATTR_HPP

#include <map>
#include <initializer_list>

#include "dnnl.h"

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

const primitive_attr_t &default_attr();

struct rnn_data_qparams_t : public c_compatible {
    rnn_data_qparams_t() : scale_(1.), shift_(0.) {}
    bool has_default_values() const { return (scale_ == 1. && shift_ == 0.); }
    bool defined() const {
        return !is_runtime_value(scale_) && !is_runtime_value(shift_);
    }

    status_t set(float scale, float shift) {
        scale_ = scale;
        shift_ = shift;
        return status::success;
    }

    bool operator==(const rnn_data_qparams_t &rhs) const {
        bool ret = scale_ == rhs.scale_ && shift_ == rhs.shift_;
        return ret;
    }

    float scale_;
    float shift_;
};

struct rnn_tparams_t : public c_compatible {
    rnn_tparams_t()
        : test_mode_(false), scales_(nullptr), ngates_(0), cscale_(0.0f) {}

    rnn_tparams_t(const rnn_tparams_t &rhs) : rnn_tparams_t() {
        set(rhs.test_mode_, rhs.ngates_, rhs.scales_, rhs.cscale_);
    }

    ~rnn_tparams_t() {
        test_mode_ = false;
        if (scales_ != nullptr) impl::free(scales_);
        scales_ = nullptr;
        ngates_ = 0;
        cscale_ = 0.0f;
    }

    bool operator==(const rnn_tparams_t &rhs) const {
        bool ret = test_mode_ == rhs.test_mode_ && ngates_ == rhs.ngates_
                && cscale_ == rhs.cscale_;

        if (!ret) return ret;

        if (scales_) {
            for (dim_t g = 0; g < ngates_; g++) {
                if (scales_[g] != rhs.scales_[g]) { return false; }
            }
        }
        return true;
    }

    rnn_tparams_t &operator=(const rnn_tparams_t &rhs) {
        DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
        status_t status
                = set(rhs.test_mode_, rhs.ngates_, rhs.scales_, rhs.cscale_);
        assert(status == status::success);
        (void)status;
        return *this;
    }

    bool has_default_values() const {
        return (test_mode_ == false && scales_ == nullptr && ngates_ == 0
                && cscale_ == 0.0f);
    }

    status_t set(bool mode, dim_t ngates, const float *scales, float cscale) {
        test_mode_ = mode;
        ngates_ = ngates;
        scales_ = nullptr;
        if (scales != nullptr) {
            scales_ = (float *)impl::malloc(ngates_ * sizeof(*scales_), 64);
            if (scales_ == nullptr) return status::out_of_memory;
            utils::array_copy(scales_, scales, ngates_);
        }

        cscale_ = cscale;

        return status::success;
    }

    bool test_mode_; /* we could also use scale_ == nullptr as a test to check test_mode*/
    float *scales_;
    dim_t ngates_; /* ngates is equel to the number of scales */
    float cscale_; /* =0.0f if no c state */
};

struct scales_t : public c_compatible {
    scales_t() : count_(1), mask_(0), scales_(scales_buf_) { set(1.); }
    scales_t(dim_t count, int mask, const float *scales)
        : scales_(scales_buf_) {
        set(count, mask, scales);
    }

    scales_t(const scales_t &rhs) : scales_t() {
        set(rhs.count_, rhs.mask_, rhs.scales_);
    }

    ~scales_t() { cleanup(); }

    scales_t &operator=(const scales_t &rhs) {
        DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
        status_t status = set(rhs.count_, rhs.mask_, rhs.scales_);
        assert(status == status::success);
        (void)status;
        return *this;
    }

    bool operator==(const scales_t &rhs) const {
        bool ret = count_ == rhs.count_ && mask_ == rhs.mask_
                && !utils::any_null(scales_, rhs.scales_)
                && defined() == rhs.defined()
                && IMPLICATION(defined(),
                        utils::array_cmp(scales_, rhs.scales_, count_));
        return ret;
    }

    bool has_default_values() const {
        for (dim_t c = 0; c < count_; ++c) {
            if (scales_[c] != 1.) return false;
        }
        return true;
    }

    bool defined() const { return !is_runtime_value(scales_[0]); }

    status_t set(dim_t count, int mask, const float *scales);
    status_t set(float single_scale) { return this->set(1, 0, &single_scale); }

    dim_t count_;
    int mask_;
    float *scales_;

private:
    enum { scales_buf_size = 16 };
    float scales_buf_[scales_buf_size];

    void cleanup() {
        if (scales_ != scales_buf_ && scales_ != nullptr) impl::free(scales_);

        count_ = 1;
        mask_ = 0;
        scales_ = scales_buf_;
    }
};

struct arg_scales_t : public c_compatible {
    arg_scales_t() {
        for (const auto &sa : {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1}) {
            set(sa, 1.f);
        }
    }

    const scales_t &get(int arg) const {
        static const scales_t default_scales;
        const auto it = scales_.find(arg);
        if (it == scales_.end()) return default_scales;
        return it->second;
    }

    bool operator==(const arg_scales_t &rhs) const {
        return scales_ == rhs.scales_;
    }

    bool has_default_values() const {
        for (const auto &s : scales_) {
            if (!s.second.has_default_values()) return false;
        }
        return true;
    }

    status_t get(int arg, dim_t *count, int *mask, const float **scales) const;
    status_t set(int arg, dim_t count, int mask, const float *scales);
    status_t set(int arg, float single_scale) {
        return set(arg, 1, 0, &single_scale);
    }

    std::map<int, scales_t> scales_;

private:
    bool check_arg(int arg) const {
        for (const auto &sa : {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1}) {
            if (arg == sa) return true;
        }
        return false;
    }
};

struct zero_points_t : public c_compatible {
    bool operator==(const zero_points_t &rhs) const {
        auto eq = [](int a, int b) {
            return a == b || (is_runtime_value(a) && is_runtime_value(b));
        };
        return eq(zero_point_src, rhs.zero_point_src)
                && eq(zero_point_wei, rhs.zero_point_wei)
                && eq(zero_point_dst, rhs.zero_point_dst);
    }

    // arg-specific checks
    bool common(int arg) const { return true; }
    bool defined(int arg) const { return !is_runtime_value(*get(arg)); }
    bool has_default_values(int arg) const {
        return *get(arg) == 0 && get_mask(arg) == 0;
    }

    // same checks but for all supported arguments at once
    bool common() const { return check_all(&zero_points_t::common); }
    bool defined() const { return check_all(&zero_points_t::defined); }
    bool has_default_values() const {
        return check_all(&zero_points_t::has_default_values);
    }

    const int *get(int arg) const {
        arg &= ~DNNL_ARG_ATTR_ZERO_POINTS;
        switch (arg) {
            case DNNL_ARG_SRC: return &zero_point_src;
            case DNNL_ARG_WEIGHTS: return &zero_point_wei;
            case DNNL_ARG_DST: return &zero_point_dst;
        }
        static int zero = 0;
        return &zero;
    }

    status_t get(
            int arg, dim_t *count, int *mask, const int **zero_points) const;

    status_t set(int arg, dim_t count, int mask, const int *zero_points);
    status_t set(int arg, int single_zero_points) {
        return set(arg, 1, 0, &single_zero_points);
    }

private:
    // TODO: support count and mask
    int zero_point_src = 0, zero_point_wei = 0, zero_point_dst = 0;
    int mask_src = 0, mask_wei = 0, mask_dst = 0;

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

    bool check_all(bool (zero_points_t::*f)(int) const) const {
        for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST})
            if (!(this->*f)(arg)) return false;
        return true;
    }
};

} // namespace impl
} // namespace dnnl

struct dnnl_post_ops : public dnnl::impl::c_compatible {
    struct entry_t {
        struct eltwise_t {
            dnnl::impl::alg_kind_t alg;
            float scale, alpha, beta;
        };

        dnnl::impl::primitive_kind_t kind;
        union {
            struct {
                float scale;
            } sum;
            eltwise_t eltwise;
        };

        bool is_eltwise(bool require_scale_one = false) const {
            using namespace dnnl::impl;
            return kind == primitive_kind::eltwise
                    && IMPLICATION(require_scale_one, eltwise.scale == 1.f);
        }

        bool is_relu(bool require_scale_one = true,
                bool require_nslope_zero = true) const {
            using namespace dnnl::impl;
            return is_eltwise(require_scale_one)
                    && eltwise.alg == alg_kind::eltwise_relu
                    && IMPLICATION(require_nslope_zero, eltwise.alpha == 0.f);
        }

        bool is_sum(bool require_scale_one = true) const {
            using namespace dnnl::impl;
            return kind == primitive_kind::sum
                    && IMPLICATION(require_scale_one, sum.scale == 1.f);
        }

        bool operator==(const entry_t &rhs) const {
            using namespace dnnl::impl;
            if (kind != rhs.kind) { return false; }
            bool ret = true;
            switch (kind) {
                case primitive_kind::eltwise:
                    ret = eltwise.alg == rhs.eltwise.alg
                            && eltwise.scale == rhs.eltwise.scale
                            && eltwise.alpha == rhs.eltwise.alpha
                            && eltwise.beta == rhs.eltwise.beta;
                    break;
                case primitive_kind::sum:
                    ret = sum.scale == rhs.sum.scale;
                    break;
                default: assert(!"unsupported post_op");
            }
            return ret;
        }

        bool operator!=(const entry_t &rhs) const {
            return !this->operator==(rhs);
        }
    };

    dnnl_post_ops() : len_(0) {}

    dnnl::impl::status_t append_sum(float scale);
    dnnl::impl::status_t append_eltwise(
            float scale, dnnl::impl::alg_kind_t alg, float alpha, float beta);

    int find(dnnl::impl::primitive_kind_t kind, int start = 0,
            int stop = -1) const {
        if (stop == -1) stop = len_;
        stop = dnnl::impl::nstl::min(stop, len_);
        for (int idx = start; idx < stop; ++idx)
            if (entry_[idx].kind == kind) return idx;
        return -1;
    }

    bool defined() const;
    bool has_default_values() const { return len_ == 0; }

    bool contain(dnnl::impl::primitive_kind_t kind, int index) const {
        return find(kind, index, index + 1) == index;
    }

    bool operator==(const dnnl_post_ops &rhs) const {
        bool ret = len_ == rhs.len_
                && dnnl::impl::utils::array_cmp(entry_, rhs.entry_, len_);
        return ret;
    }

    enum { capacity = 4 };

    int len_;
    entry_t entry_[capacity];
};

struct dnnl_primitive_attr : public dnnl::impl::c_compatible {
    dnnl_primitive_attr()
        : scratchpad_mode_(dnnl::impl::scratchpad_mode::library) {}

    dnnl_primitive_attr *clone() const {
        return new dnnl_primitive_attr(*this);
    }

    enum class skip_mask_t : unsigned {
        none = 0,
        oscale = 1u << 0,
        oscale_runtime = (unsigned)oscale | (1u << 1),
        scales = 1u << 2,
        zero_points = 1u << 3,
        zero_points_runtime = (unsigned)zero_points | (1u << 4),
        post_ops = 1u << 5,
        rnn_data_qparams = 1u << 6,
        rnn_weights_qparams = 1u << 7,
        rnn_tparams = 1u << 8,
    };

    /** Returns true if the attributes have default values.
     *
     * @note The scratchpad_mode_ is not take into account */
    bool has_default_values(skip_mask_t mask = skip_mask_t::none) const;

    /** Returns true if the attributes are fully defined. */
    bool defined(skip_mask_t mask = skip_mask_t::none) const;

    bool operator==(const dnnl_primitive_attr &rhs) const {
        bool ret = scratchpad_mode_ == rhs.scratchpad_mode_
                && output_scales_ == rhs.output_scales_
                && scales_ == rhs.scales_ && zero_points_ == rhs.zero_points_
                && post_ops_ == rhs.post_ops_
                && rnn_data_qparams_ == rhs.rnn_data_qparams_
                && rnn_weights_qparams_ == rhs.rnn_weights_qparams_
                && rnn_tparams_ == rhs.rnn_tparams_;
        return ret;
    }

    dnnl::impl::status_t set_scratchpad_mode(
            dnnl::impl::scratchpad_mode_t scratchpad_mode);
    dnnl::impl::status_t set_post_ops(const dnnl::impl::post_ops_t &post_ops);

    // NOTE: make sure that the types below have overloaded comparison operator
    dnnl::impl::scales_t output_scales_;
    dnnl::impl::arg_scales_t scales_;
    dnnl::impl::zero_points_t zero_points_;
    dnnl::impl::scratchpad_mode_t scratchpad_mode_;
    dnnl::impl::post_ops_t post_ops_;
    dnnl::impl::rnn_data_qparams_t rnn_data_qparams_;
    dnnl::impl::scales_t rnn_weights_qparams_;
    dnnl::impl::rnn_tparams_t rnn_tparams_;
};

inline dnnl_primitive_attr::skip_mask_t operator|(
        dnnl_primitive_attr::skip_mask_t lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    return static_cast<dnnl_primitive_attr::skip_mask_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}
inline dnnl_primitive_attr::skip_mask_t operator&(
        dnnl_primitive_attr::skip_mask_t lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    return static_cast<dnnl_primitive_attr::skip_mask_t>(
            static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}
inline dnnl_primitive_attr::skip_mask_t &operator|=(
        dnnl_primitive_attr::skip_mask_t &lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    lhs = lhs | rhs;
    return lhs;
}
inline dnnl_primitive_attr::skip_mask_t &operator&=(
        dnnl_primitive_attr::skip_mask_t &lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    lhs = lhs & rhs;
    return lhs;
}
inline bool operator!=(dnnl_primitive_attr::skip_mask_t lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    return (static_cast<unsigned>(lhs) != static_cast<unsigned>(rhs));
}
inline dnnl_primitive_attr::skip_mask_t operator~(
        dnnl_primitive_attr::skip_mask_t rhs) {
    return static_cast<dnnl_primitive_attr::skip_mask_t>(
            ~static_cast<unsigned>(rhs));
}

#endif
