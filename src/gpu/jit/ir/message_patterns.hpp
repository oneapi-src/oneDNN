/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/jit/ir/message_patterns.hpp"
#include "common/type_helpers.hpp"
#include "gpu/jit/utils/utils.hpp"
#include "gpu/jit/ir/message.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Tagged types
enum type_id_t {
    empty = 0,
    uniform_blocked = 1,
    uniform_2d = 2,
};

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
        stride_dim_t(const stride_dim_t &other) = default;

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
    send_hint_t() : type_id_(type_id_t::empty) {};
    static const dim_t unset = 0;
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

    send_hint_t lcm(const send_hint_t &other) {
        send_hint_t ret;
        for (int i = 0; i < dim_type_t::NDIMS; i++) {
            if (hint_[i] == unset)
                ret.hint_[i] = other.hint_[i];
            else if (other.hint_[i] == unset)
                ret.hint_[i] = hint_[i];
            else
                ret.hint_[i] = math::lcm(hint_[i], other.hint_[i]);
        }
        return ret;
    }
    enum dim_idx { block = 0, w = 1, h = 2 };

    dim_t size(dim_idx dim = dim_idx::block) const {
        assert((dim == dim_idx::block) || is_uniform_2d());
        int s = 1;
        for (size_t i = 0; i < hint_.size(); ++i) {
            if (hint_[i] != unset
                    && (dim == dim_idx::block || dim & w_dims_[i]))
                s *= hint_[i];
        }
        return s;
    }

    dim_t height_size() const {
        assert(is_uniform_2d());
        return size(dim_idx::h);
    }

    void set_w_dim(dim_type_t idx) { w_dims_[idx.id()] |= dim_idx::w; }
    void set_h_dim(dim_type_t idx) { w_dims_[idx.id()] |= dim_idx::h; }
    bool is_w_dim(dim_type_t idx) const {
        return w_dims_[idx.id()] & dim_idx::w;
    }
    bool is_h_dim(dim_type_t idx) const {
        return w_dims_[idx.id()] & dim_idx::h;
    }

    dim_t block_size(dim_t base) const { return size(); }

    bool is_uniform_blocked() const { return type_id_ == uniform_blocked; }
    bool is_uniform_2d() const { return type_id_ == uniform_2d; }
    type_id_t get_type() const { return type_id_; }
    void set_type(type_id_t type) { type_id_ = type; }

private:
    type_id_t type_id_;
    std::array<dim_t, dim_type_t::max_id()> hint_ = {0};
    std::array<dim_t, dim_type_t::max_id()> w_dims_ = {0};
};

template <typename dim_type_t>
struct uniform_send_idiom_t final {
    uniform_send_idiom_t(dim_t block_min_size, double b, bool check_2d = false)
        : size(block_min_size), min_utilization(b), check_2d(check_2d) {}
    constexpr uniform_send_idiom_t(const uniform_send_idiom_t &) = default;

    using hint_t = send_hint_t<dim_type_t>;
    using slayout_t = stride_layout_t<dim_type_t>;

    dim_t size;
    double min_utilization;
    bool check_2d;
    static const dim_t block_alignment = 16;
    static const dim_t width_alignment = 4;
    static const dim_t pitch_alignment = 8;
    static const dim_t surface_width_min_size = 64;
    static const dim_t surface_width_alignment = 4;
    static const dim_t block_width = 64; // Max blocked width in bytes
    static const dim_t block_height = 32; // Max rows of blocked width

    double utilization_2d(int size) const {
        return 1.0 * size / (block_width * block_height);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "uniform " << size << " byte send, util: " << min_utilization;
        return oss.str();
    }
    dim_t load_size() const { return size; }

    dim_t ref_block_size(const slayout_t &layout) const {
        return load_size() / layout.type_size;
    }
    dim_t ref_2d_width(const slayout_t &layout) const {
        return block_width / layout.type_size;
    }
    dim_t ref_2d_height() const { return block_height; }

    std::vector<hint_t> get_hints(const slayout_t &layout,
            typename slayout_t::stride_array_t::const_iterator i,
            const hint_t &hint, bool valid_2d = true, bool valid_block = true) const {
        // BASE CASE CHECK

        if (hint.is_uniform_blocked() && !valid_block) return {};
        if (hint.is_uniform_2d() && !valid_2d) return {};

        auto block_rem = [&]() { return ref_block_size(layout) / hint.size(); };
        auto width_rem
                = [&]() { return ref_2d_width(layout) / hint.size(); };
        auto height_rem = [&]() {
            dim_t height = hint.size() / ref_2d_width(layout);
            return !!height ? ref_2d_height() / height : height;
        };
        auto surface_pitch = [&](const slayout_t &layout) {
            dim_t val = 0;
            if (layout.strides.begin() == i) return val;
            auto iter = layout.strides.begin();
            while (iter != i) {
                if (hint.is_h_dim(iter->dim)) val = iter->stride;
                ++iter;
            }
            return val * layout.type_size;
        };

        auto surface_width = [&](const slayout_t &layout) {
            dim_t val = 0;
            if (layout.strides.begin() == i) return val;
            auto iter = layout.strides.begin();
            while (iter != i) {
                if (hint.is_w_dim(iter->dim))
                    val = hint[iter->dim] * iter->stride;
                ++iter;
            }
            return val * layout.type_size;
        };

        // Base case: whole layout has been processed and the hint satisfies
        // alignment checks.
        if (i == layout.strides_end()) {
            bool valid_block_util
                    = hint.is_uniform_blocked() && block_rem() <= 1;
            bool valid_2d_util = hint.is_uniform_2d()
                    && utilization_2d(hint.size() * layout.type_size)
                            >= min_utilization;
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
        if (valid_block && i_stride_bytes > load_size() && block_rem() > 1)
            valid_block = false;

        auto width_stride = width_rem() > 1
                ? std::max(layout.type_size, block_width / width_rem())
                : block_width;

        // Check hint is potentially valid
        if (valid_2d && width_stride < i_stride_bytes
                && min_utilization * width_rem() > 1) {
            valid_2d = false;
        }

        if(!(valid_block || valid_2d))
            return std::vector<hint_t>{};

        // SKIP CURRENT DIM

        // Get hints which skip the current dimension
        std::vector<hint_t> skip_hints = [&] {
            bool is_aligned_block
                    = valid_block && i_stride_bytes % block_alignment == 0;
            bool is_aligned_2d
                    = valid_2d && i_stride_bytes % width_alignment == 0;

            // The 2d send instruction zero extends any data outside the surface
            // width and surface height, so always true for 2d.
            bool no_partial_overflow = !i->can_overflow
                    || (i_stride_bytes % load_size() == 0
                            && layout.buffer_size % load_size() == 0);
            is_aligned_block &= no_partial_overflow;

            // Get all hints skipping this dimension
            if (is_aligned_block || is_aligned_2d)
                return get_hints(layout, i + 1, hint, is_aligned_2d, is_aligned_block);
            else {
                return std::vector<hint_t> {};
            }
        }();

        // USE CURRENT DIM

        // Get hints which use the current dimension
        std::vector<hint_t> use_hints = [&] {
            if (block_rem() <= 1 || hint.is_uniform_2d() || !valid_block)
                return std::vector<hint_t> {};

            // Check data is contiguous
            auto stride = std::max(layout.type_size, load_size() / block_rem());
            if (stride != i_stride_bytes || i->is_complex) {
                return std::vector<hint_t> {};
            }

            // Alignment check is unnecessary as the loads stride enforces proper
            // alignment
            if (i->size >= block_rem() && i->size % block_rem() == 0) {
                // No masking means the final load block must be divisible by the
                // load_size
                hint_t new_hint = hint;
                new_hint[i->dim] = block_rem();
                new_hint.set_type(type_id_t::uniform_blocked);
                return get_hints(layout, i + 1, new_hint, /*valid_2d=*/false,
                        valid_block);

            } else if (block_rem() % i->size == 0) {
                // No masking means we cannot handle any overflow
                hint_t new_hint = hint;
                new_hint[i->dim] = i->size;
                new_hint.set_type(type_id_t::uniform_blocked);
                return get_hints(layout, i + 1, new_hint, /*valid_2d=*/false,
                        valid_block);
            } else {
                // Dimension cannot be packed into a blocked load
                return std::vector<hint_t> {};
            }
        }();

        auto pack_2d = [&](int size, hint_t new_hint) {
            new_hint.set_type(type_id_t::uniform_2d);
            new_hint.set_h_dim(i->dim);
            if (size >= height_rem()) {
                new_hint[i->dim] = new_hint[i->dim] == hint_t::unset
                        ? height_rem()
                        : new_hint[i->dim] * height_rem();
                auto ret = get_hints(layout, i + 1, new_hint, valid_2d,
                        /*valid_block=*/false);
                ir_info() << "Fully height packed " << i->dim << "Resulted in "
                          << ret.size() << " hints:\n";
                for (auto &i : ret)
                    ir_info() << i << "\n";
                ir_info() << "\n";
                return ret;
            } else {
                new_hint[i->dim] = new_hint[i->dim] == hint_t::unset
                        ? size
                        : new_hint[i->dim] * size;
                auto ret = get_hints(layout, i + 1, new_hint, valid_2d,
                        /*valid_block=*/false);
                ir_info() << "Fully height packed " << i->dim << "Resulted in "
                          << ret.size() << " hints:\n";
                for (auto &i : ret)
                    ir_info() << i << "\n";
                ir_info() << "\n";
                return ret;
            }
        };

        std::vector<hint_t> use_height_hints = [&] {
            ir_info() << "Packing " << i->dim << " into height\n";
            if (!valid_2d || !(height_rem() > 1)
                    || surface_width(layout) < surface_width_min_size
                    || i->is_complex) {
                ir_info() << "Cannot pack into height: block_height: "
                          << height_rem()
                          << " surface width: " << surface_width(layout)
                          << " is_complex: " << i->is_complex << "\n";
                return std::vector<hint_t> {};
            }

            // Attempt to pack data into block_height
            if (surface_pitch(layout) == 0) {
                if (i_stride_bytes % pitch_alignment != 0
                        || i_stride_bytes < surface_width(layout)) {
                    ir_info() << "Cannot pack height as data is not aligned\n";
                    return std::vector<hint_t> {};
                }
            } else {
                // Check data is contiguous
                auto stride
                        = (block_height / height_rem()) * surface_pitch(layout);
                if (stride != i_stride_bytes) {
                    ir_info()
                            << "Cannot pack height as data is not contiguous\n";
                    return std::vector<hint_t> {};
                }
            }

            return pack_2d(i->size, hint);
        }();

        // Get hints which use the current dimension as Width
        std::vector<hint_t> use_width_hints = [&] {
            ir_info() << "Packing " << i->dim << " into width\n";
            // Cannot pack this dimension;
            if (i_stride_bytes != width_stride || i->is_complex
                    || surface_pitch(layout) != 0 || width_rem() <= 1
                    || !valid_2d) {
                ir_info() << "Cannot pack width i_stride_bytes: "
                          << i_stride_bytes
                          << " pitch: " << surface_pitch(layout)
                          << " width_rem: " << width_rem() << "\n";
                return std::vector<hint_t> {};
            }

            // Width must be bound to innermost unit stride block
            if (i->stride != 1) { return std::vector<hint_t> {}; }

            // No need to check block_width alignment, it is enforced by the
            // surface_pitch and fact block_width is loaded in power of 2
            // sizes.
            if (i->size >= width_rem()) {
                hint_t new_hint = hint;
                new_hint.set_type(type_id_t::uniform_2d);
                new_hint[i->dim] = width_rem();
                new_hint.set_w_dim(i->dim);
                ir_info() << "Fully Packing " << i->dim << " into width\n";

                auto ret = [&]() {
                    // Get blocked load using 2d send
                    if (i->size > width_rem()
                            && i->size % width_rem() == 0) {
                        ir_info() << "Fully Packing " << i->dim
                                  << "height spillover\n";
                        return pack_2d(i->size / width_rem(), new_hint);
                    }
                    ir_info()
                            << "No height spillover to pack: size: " << i->size
                            << "block_width: " << width_rem() << "\n";
                    return std::vector<hint_t> {};
                }();

                int new_surface_width = i->size * i_stride_bytes;
                if (new_surface_width % surface_width_alignment) {
                    // Surface width must be aligned to max(4,
                    // elem_size). The elem_size requirement is
                    // implicitly enforced by the stride.
                    ir_info() << "Cannot pack width due to "
                                 "alignment: "
                              << i->dim << ": " << i->size << "\n";
                    return std::vector<hint_t> {};
                }

                auto ret2 = get_hints(
                        layout, i + 1, new_hint, valid_2d, valid_block);
                ret.insert(ret.end(), ret2.begin(), ret2.end());
                ir_info() << "Fully width packed " << i->dim << "Resulted in "
                          << ret.size() << " hints:\n";
                for (auto &i : ret)
                    ir_info() << i << "\n";
                ir_info() << "\n";
                return ret;

                // Accept correctly aligned send with w size < rem as long as
                // long as surface width will be > min.
            } else if (width_rem() % i->size == 0) {
                // Cannot partially pack as we need surface_width >=
                // surface_width_min_size() >= block_width
                if (i->size * i_stride_bytes < surface_width_min_size) {
                    return std::vector<hint_t> {};
                }
                hint_t new_hint = hint;
                new_hint[i->dim] = i->size;
                new_hint.set_type(type_id_t::uniform_2d);
                new_hint.set_w_dim(i->dim);

                auto ret = get_hints(
                        layout, i + 1, new_hint, valid_2d, valid_block);
                ir_info() << "Sub width packed " << i->dim << "Resulted in "
                          << ret.size() << " hints:\n";
                for (auto &i : ret)
                    ir_info() << i << "\n";
                ir_info() << "\n";
                return ret;
            }
            return std::vector<hint_t> {};
        }();

        // Hint priority is 1. Width 2. Height 3. skip
        use_width_hints.insert(use_width_hints.end(), use_height_hints.begin(),
                use_height_hints.end());
        use_width_hints.insert(
                use_width_hints.end(), use_hints.begin(), use_hints.end());
        use_width_hints.insert(
                use_width_hints.end(), skip_hints.begin(), skip_hints.end());

        return use_width_hints;
    }

    std::vector<hint_t> get_hints(const slayout_t &layout) const {
        ir_info() << "Getting Hints for: " << layout << "\n";
        hint_t hint;
        auto ret = get_hints(layout, layout.strides.begin(), hint, check_2d,
                /*valid_block=*/true);
        return ret;
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
            if ((s.is_block() || s.is_2d()) && s.access_size() >= pattern.data().size())
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
    send_pattern_t() : type_id_(type_id_t::empty) {};
    send_pattern_t(const hint_t &data)
        : type_id_(data.get_type()), data_(data) {};
    void operator=(const hint_t &data) {
        type_id_ = data.get_type();
        data_ = data;
    }

    bool is_empty() const { return type_id_ == empty; }
    hint_t data() const { return data_; }
    bool is_uniform_blocked() const { return type_id_ == uniform_blocked; }
    bool is_uniform_2d() const { return type_id_ == uniform_2d; }
    bool matches(const stmt_t &stmt) const{
        return send_matcher_t<send_pattern_t::dim_type_>::is_match(*this, stmt);
    }
    std::string str() const {
        return data_.str();
    }

private:
    using dim_type_ = dim_id_t;
    type_id_t type_id_;
    hint_t data_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
