/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_JIT_IR_MESSAGE_PATTERNS_HPP
#define GPU_JIT_IR_MESSAGE_PATTERNS_HPP

#include <sstream>
#include <string>

#include "common/type_helpers.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/message_patterns.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Tagged types

// Layout is represented as a sorted list of strides. Each entry corresponds to
// a variable used for offset calculation. Naively, the setup is such that any
// offset would be calculated as the sum of the f(entry_var) * stride plus some
// base_offset. A stride is considered complex when f(entry_var) != dim. The
// function f and the base offset are not stored in this structure. The base
// offset is assumed to be generated from simplifying expressions like
// (f(entry_var) + c) * stride, so it results in the same effective alignments.
// The resulting overflow/underflow is handled by the entry can_overflow.
template <typename dim_type_t>
struct stride_layout_t {
    static const int MAX_NDIMS = 12;
    stride_layout_t(int type_size)
        : buffer_size(0), type_size(type_size), ndims(0) {};

    struct stride_entry_t {
        stride_entry_t() = default;
        stride_entry_t(dim_t size, dim_t stride)
            : size(size), stride(stride) {};
        dim_t size = 0;
        dim_t stride = 0;
    };

    struct stride_dim_t {
        stride_dim_t() = default;
        stride_dim_t(dim_type_t dim, dim_t size, dim_t stride,
                bool can_overflow, bool is_complex)
            : dim(dim)
            , size(size)
            , stride(stride)
            , can_overflow(can_overflow)
            , is_complex(is_complex) {}

        dim_type_t dim;
        dim_t size = 0;
        dim_t stride = 0;

        // whether this dimension can result in buffer overflows
        bool can_overflow;

        // A stride is considered complex if its offset is not simply calculated
        // as dim * stride;
        bool is_complex;

        bool operator<(const stride_dim_t &other) const {
            if (stride < other.stride)
                return true;
            else if (stride == other.stride) {
                if (size > other.size)
                    return true;
                else if (size == other.size)
                    return dim.id() < other.dim.id();
            }
            return false;
        }
        std::string str() const {
            std::ostringstream oss;
            oss << dim << ":" << size << "*" << stride;
            return oss.str();
        }
    };

    using stride_array_t = std::array<stride_dim_t, MAX_NDIMS>;

    typename stride_array_t::iterator strides_end() {
        return strides.begin() + ndims;
    }
    typename stride_array_t::const_iterator strides_end() const {
        return strides.begin() + ndims;
    }
    std::string str() const {
        std::ostringstream oss;
        oss << "buffer_size:" << buffer_size;
        for (auto i = strides.begin(); i != strides_end(); i++) {
            oss << " " << i->str();
        }
        return oss.str();
    }

    const stride_dim_t &operator[](int i) const {
        ir_assert(i < MAX_NDIMS);
        return strides[i];
    }
    stride_dim_t &operator[](int i) {
        ir_assert(i < MAX_NDIMS);
        return strides[i];
    }

    dim_t buffer_size;
    dim_t type_size;
    dim_t ndims;
    stride_array_t strides;
};

template <typename dim_type_t>
struct send_hint_t {
    send_hint_t()
        : type_id_(send_type_id_t::empty), type_size_(0), ref_block_size_(0) {};
    send_hint_t(dim_t type_size, dim_t ref_block_size)
        : type_id_(send_type_id_t::empty)
        , type_size_(type_size)
        , ref_block_size_(ref_block_size) {};
    static const dim_t unset = 0;
    enum send_type_id_t {
        empty = 0,
        uniform_blocked = 1,
        uniform_2d = 2,
    };
    enum send_dim_idx { block = 0, w = 1, h = 2 };
    using slayout_t = stride_layout_t<dim_type_t>;
    using hint_t = send_hint_t<dim_type_t>;
    const dim_t &operator[](dim_type_t i) const { return hint_[i.id()]; }
    dim_t &operator[](dim_type_t i) { return hint_[i.id()]; }
    std::string str() const {
        std::ostringstream oss;
        oss << "hint:";
        bool is_empty = true;
        for (int id = 0; id < dim_type_t::max_id(); id++) {
            auto i = dim_type_t::from_id(id);
            if (hint_[i.id()] != unset) {
                oss << " " << i.str() << ":" << hint_[i.id()];
                is_empty = false;
            }
        }
        if (is_empty) oss << " (empty)";
        return oss.str();
    }

    dim_t size(send_dim_idx dim = send_dim_idx::block) const {
        assert((dim == send_dim_idx::block) || is_uniform_2d());
        int s = 1;
        for (size_t i = 0; i < hint_.size(); ++i) {
            if (hint_[i] != unset
                    && (dim == send_dim_idx::block || dim & w_dims_[i]))
                s *= hint_[i];
        }
        return s;
    }

    void add_stride(const typename slayout_t::stride_dim_t &i,
            send_dim_idx send_idx = send_dim_idx::block) {
        strides_.push_back(i);
        dim_t base;
        switch (send_idx) {
            case send_dim_idx::block:
                set_type(send_type_id_t::uniform_blocked);
                base = block_rem();
                break;
            case send_dim_idx::w:
            case send_dim_idx::h:
                set_dim(i.dim, send_idx);
                set_type(send_type_id_t::uniform_2d);
                base = rem(send_idx);
                break;
        }
        base = std::min(base, i.size);
        hint_[i.dim.id()] = hint_[i.dim.id()] == hint_t::unset
                ? base
                : hint_[i.dim.id()] * base;
    }

    void set_dim(dim_type_t idx, send_dim_idx i) { w_dims_[idx.id()] |= i; }
    bool is_w_dim(dim_type_t idx) const {
        return w_dims_[idx.id()] & send_dim_idx::w;
    }
    bool is_h_dim(dim_type_t idx) const {
        return w_dims_[idx.id()] & send_dim_idx::h;
    }

    bool is_uniform_blocked() const { return type_id_ == uniform_blocked; }
    bool is_uniform_2d() const { return type_id_ == uniform_2d; }
    send_type_id_t get_type() const { return type_id_; }
    void set_type(send_type_id_t type) {
        ir_assert(utils::one_of(type_id_, type, send_type_id_t::empty));
        type_id_ = type;
    }
    dim_t ref_2d_width() const { return block_width / type_size_; }
    static const dim_t block_width = 64; // Max blocked width in bytes
    static const dim_t block_height = 32; // Max rows of blocked width
    dim_t rem(send_dim_idx i) const {
        switch (i) {
            case send_dim_idx::w: return width_rem();
            case send_dim_idx::h: return height_rem();
            case send_dim_idx::block: return block_rem();
        }
        return -1;
    }

    dim_t block_rem() const { return ref_block_size_ / size(); };
    dim_t width_rem() const { return ref_2d_width() / size(); };
    dim_t height_rem() const {
        dim_t height = size() / ref_2d_width();
        return !!height ? block_height / height : height;
    };
    dim_t surface_pitch() const {
        dim_t val = 0;
        for (auto s : strides_) {
            if (is_h_dim(s.dim)) { val = s.stride; }
        }
        return val * type_size_;
    };

    dim_t surface_width() const {
        dim_t val = 0;
        for (auto s : strides_) {
            if (is_w_dim(s.dim)) val = hint_[s.dim.id()] * s.stride;
        }
        return val * type_size_;
    };

private:
    send_type_id_t type_id_;
    dim_t type_size_;
    dim_t ref_block_size_;
    std::array<dim_t, dim_type_t::max_id()> hint_ = {0};
    std::array<dim_t, dim_type_t::max_id()> w_dims_ = {0};
    std::vector<typename slayout_t::stride_dim_t> strides_;
};

template <typename dim_type_t>
struct uniform_send_idiom_t final {
    uniform_send_idiom_t(dim_t min_size, bool check_2d = false)
        : min_size(min_size), check_2d(check_2d) {}
    constexpr uniform_send_idiom_t(const uniform_send_idiom_t &) = default;

    using hint_t = send_hint_t<dim_type_t>;
    using slayout_t = stride_layout_t<dim_type_t>;

    dim_t min_size;
    bool check_2d;
    static const dim_t block_alignment = 16;
    static const dim_t width_alignment = 4;
    static const dim_t pitch_alignment = 8;
    static const dim_t surface_width_min_size = 64;
    static const dim_t surface_width_alignment = 4;
    dim_t ref_block_size(const slayout_t &layout) const {
        return block_load_min_size() / layout.type_size;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "uniform " << min_size << " byte send ";
        return oss.str();
    }
    dim_t block_load_min_size() const { return min_size; }

    std::vector<hint_t> get_hints(const slayout_t &layout,
            typename slayout_t::stride_array_t::const_iterator i,
            const hint_t &hint, bool valid_2d = true,
            bool valid_block = true) const {
        // BASE CASE CHECK

        if (hint.is_uniform_blocked() && !valid_block) return {};
        if (hint.is_uniform_2d() && !valid_2d) return {};

        // Base case: whole layout has been processed and the hint satisfies
        // alignment checks.
        if (i == layout.strides_end()) {
            bool valid_block_util
                    = hint.is_uniform_blocked() && hint.block_rem() <= 1;
            bool valid_2d_util = hint.is_uniform_2d()
                    && (hint.size() * layout.type_size) >= min_size;
            if (valid_block_util || valid_2d_util)
                return {hint};
            else
                return {};
        }

        // INITIAL ALIGNMENT CHECKS

        auto i_stride_bytes = i->stride * layout.type_size;

        // The stride_layout_t structure is sorted by stride, therefore if the
        // stride exceeds the current load stride, it is impossible to pack any
        // more data into a blocked load instruction.
        if (valid_block && i_stride_bytes > block_load_min_size()
                && hint.block_rem() > 1)
            valid_block = false;

        auto width_stride = hint.width_rem() > 1 ? std::max(layout.type_size,
                                    hint_t::block_width / hint.width_rem())
                                                 : hint_t::block_width;

        // Check hint is potentially valid
        if (valid_2d && width_stride < i_stride_bytes && hint.width_rem() > 8) {
            valid_2d = false;
        }

        if (!(valid_block || valid_2d)) return std::vector<hint_t> {};

        // Get 2D or block hints which skip the current dimension
        std::vector<hint_t> skip_hints = [&] {
            bool is_aligned_block
                    = valid_block && i_stride_bytes % block_alignment == 0;
            bool is_aligned_2d
                    = valid_2d && i_stride_bytes % width_alignment == 0;

            // The 2d send instruction zero extends any data outside the surface
            // width and surface height, so always true for 2d.
            bool no_partial_overflow = !i->can_overflow
                    || (i_stride_bytes % block_load_min_size() == 0
                            && layout.buffer_size % block_load_min_size() == 0);
            is_aligned_block &= no_partial_overflow;

            // Get all hints skipping this dimension
            if (is_aligned_block || is_aligned_2d) {
                return get_hints(
                        layout, i + 1, hint, is_aligned_2d, is_aligned_block);
            } else {
                return std::vector<hint_t> {};
            }
        }();

        // Get block hints which use the current dimension
        std::vector<hint_t> use_blocked = [&] {
            if (hint.block_rem() <= 1 || hint.is_uniform_2d() || !valid_block)
                return std::vector<hint_t> {};

            // Check data is contiguous
            auto stride = std::max(
                    layout.type_size, block_load_min_size() / hint.block_rem());
            if (stride != i_stride_bytes || i->is_complex) {
                return std::vector<hint_t> {};
            }

            // Alignment check is unnecessary as the loads stride enforces proper
            // alignment
            if ((i->size >= hint.block_rem() && i->size % hint.block_rem() == 0)
                    || (i->size < hint.block_rem()
                            && hint.block_rem() % i->size == 0)) {
                // No masking means the final load block must be divisible by the
                // load_size
                hint_t new_hint = hint;
                new_hint.add_stride(*i, hint_t::send_dim_idx::block);
                return get_hints(layout, i + 1, new_hint, /*valid_2d=*/false,
                        valid_block);
            } else {
                // Dimension cannot be packed into a blocked load
                return std::vector<hint_t> {};
            }
        }();

        std::vector<hint_t> use_height_hints = [&] {
            if (!valid_2d || !(hint.height_rem() > 1)
                    || hint.surface_width() < surface_width_min_size
                    || i->is_complex) {
                return std::vector<hint_t> {};
            }

            // Attempt to pack data into block_height
            if (hint.surface_pitch() == 0) {
                if (i_stride_bytes % pitch_alignment != 0
                        || i_stride_bytes < hint.surface_width()) {
                    return std::vector<hint_t> {};
                }
            } else {
                // Check data is contiguous
                dim_t stride = (hint_t::block_height / hint.height_rem())
                        * hint.surface_pitch();
                if (stride != i_stride_bytes) { return std::vector<hint_t> {}; }
            }

            hint_t new_hint = hint;
            new_hint.add_stride(*i, hint_t::send_dim_idx::h);
            return get_hints(layout, i + 1, new_hint, valid_2d,
                    /*valid_block=*/false);
        }();

        // Get hints which use the current dimension as Width
        std::vector<hint_t> use_width_hints = [&] {
            // Cannot pack this dimension;
            if (i_stride_bytes != width_stride || i->is_complex
                    || hint.surface_pitch() != 0 || hint.width_rem() <= 1
                    || !valid_2d) {
                return std::vector<hint_t> {};
            }

            // No need to check block_width alignment, it is enforced by the
            // surface_pitch and fact block_width is loaded in power of 2
            // sizes.
            if (i->size >= hint.width_rem()) {
                hint_t new_hint = hint;
                new_hint.add_stride(*i, hint_t::send_dim_idx::w);

                auto ret = [&]() {
                    // Get blocked load using 2d send
                    if (i->size > hint.width_rem()
                            && i->size % hint.width_rem() == 0) {
                        new_hint.add_stride(*i, hint_t::send_dim_idx::h);
                        return get_hints(layout, i + 1, new_hint, valid_2d,
                                /*valid_block=*/false);
                    }
                    return std::vector<hint_t> {};
                }();

                int new_surface_width = i->size * i_stride_bytes;
                if (new_surface_width % surface_width_alignment) {
                    // Surface width must be aligned to max(4,
                    // elem_size). The elem_size requirement is
                    // implicitly enforced by the stride.
                    return std::vector<hint_t> {};
                }

                auto ret2 = get_hints(
                        layout, i + 1, new_hint, valid_2d, valid_block);
                ret.insert(ret.end(), ret2.begin(), ret2.end());
                return ret;

                // Accept correctly aligned send with w size < rem as long as
                // long as surface width will be > min.
            } else if (hint.width_rem() % i->size == 0) {
                // Cannot partially pack as we need surface_width >=
                // surface_width_min_size() >= block_width
                if (i->size * i_stride_bytes < surface_width_min_size) {
                    return std::vector<hint_t> {};
                }
                hint_t new_hint = hint;
                new_hint.add_stride(*i, hint_t::send_dim_idx::w);

                auto ret = get_hints(
                        layout, i + 1, new_hint, valid_2d, valid_block);
                return ret;
            }
            return std::vector<hint_t> {};
        }();

        use_width_hints.insert(use_width_hints.end(), use_height_hints.begin(),
                use_height_hints.end());
        use_width_hints.insert(
                use_width_hints.end(), use_blocked.begin(), use_blocked.end());
        use_width_hints.insert(
                use_width_hints.end(), skip_hints.begin(), skip_hints.end());

        return use_width_hints;
    }

    std::vector<hint_t> get_hints(const slayout_t &layout) const {
        hint_t hint(layout.type_size, ref_block_size(layout));
        auto ret = get_hints(layout, layout.strides.begin(), hint, check_2d,
                /*valid_block=*/true);
        std::vector<hint_t> filtered_ret(ret.size());
        auto it = std::copy_if(
                ret.begin(), ret.end(), filtered_ret.begin(), [&](hint_t &h) {
                    if (h.is_uniform_2d()) {
                        bool w_dim_set = false, h_dim_set = false;
                        for (auto &i : layout.strides) {
                            // Currently send_plan permits 2d sends only when a
                            // single dim is assigned each to h and w.
                            if ((h.is_w_dim(i.dim) && w_dim_set)
                                    || (h.is_h_dim(i.dim) && h_dim_set)
                                    || (h.is_h_dim(i.dim) && h.is_w_dim(i.dim)))
                                return false;
                            // W must be assigned innermost dim.
                            if (h.is_w_dim(i.dim) && i.stride != 1)
                                return false;

                            w_dim_set |= h.is_w_dim(i.dim);
                            h_dim_set |= h.is_h_dim(i.dim);
                        }
                    }
                    return true;
                });
        filtered_ret.resize(std::distance(filtered_ret.begin(), it));
        std::sort(filtered_ret.begin(), filtered_ret.end(),
                [&](const hint_t &a, const hint_t &b) {
                    return a.size() > b.size();
                });
        std::sort(
                ret.begin(), ret.end(), [&](const hint_t &a, const hint_t &b) {
                    return a.size() > b.size();
                });
        if (ret.size() && filtered_ret.size()
                && ret[0].size() > filtered_ret[0].size())
            ir_warning() << "Optimal send hint disabled: " << ret[0]
                         << std::endl;

        return filtered_ret;
    }
};

class stmt_t;
template <typename dim_id_t>
struct send_pattern_t;

template <typename dim_id_t>
class send_matcher_t : public ir_visitor_t {
public:
    static bool is_match(
            const send_pattern_t<dim_id_t> &pattern, const stmt_t &stmt) {
        send_matcher_t matcher(pattern);
        matcher.visit(stmt);
        return matcher.is_match_;
    };

    void _visit(const func_impl_t &obj) override {
        if (!obj.is<send_t>()) return;

        auto &s = obj.as<send_t>();

        if (pattern.is_uniform_blocked()) {
            // Larger blocked or 2D messages are a strict improvement
            if ((s.is_block() || s.is_2d())
                    && s.access_size() >= pattern.data().size())
                return;
        } else {
            if (s.is_2d() && s.access_size() >= pattern.data().size()) return;
        }

        is_match_ = false;
    }

private:
    send_matcher_t(const send_pattern_t<dim_id_t> &pattern)
        : pattern(pattern), is_match_(true) {}
    send_pattern_t<dim_id_t> pattern;
    bool is_match_;
};

// Uniform wrapper for storing load patterns generalized from hint
template <typename dim_id_t>
struct send_pattern_t {
    using hint_t = send_hint_t<dim_id_t>;
    send_pattern_t() : type_id_(hint_t::send_type_id_t::empty) {};
    send_pattern_t(const hint_t &data)
        : type_id_(data.get_type()), data_(data) {};
    void operator=(const hint_t &data) {
        type_id_ = data.get_type();
        data_ = data;
    }

    bool is_empty() const { return type_id_ == hint_t::send_type_id_t::empty; }
    hint_t data() const { return data_; }
    bool is_uniform_blocked() const {
        return type_id_ == hint_t::send_type_id_t::uniform_blocked;
    }
    bool is_uniform_2d() const {
        return type_id_ == hint_t::send_type_id_t::uniform_2d;
    }
    bool matches(const stmt_t &stmt) const {
        return send_matcher_t<send_pattern_t::dim_type_>::is_match(*this, stmt);
    }
    std::string str() const { return data_.str(); }

private:
    using dim_type_ = dim_id_t;
    typename hint_t::send_type_id_t type_id_;
    hint_t data_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
