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

#include "common/type_helpers.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

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
struct block_hint_t {
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

    block_hint_t lcm(const block_hint_t &other) {
        block_hint_t ret;
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

    dim_t size() const {
        int s = unset;
        for (auto i : hint_) {
            if (i != unset) s = (s == unset) ? i : s * i;
        }
        return s;
    }

private:
    std::array<dim_t, dim_type_t::max_id()> hint_ = {0};
};

template <typename dim_type_t>
struct send_idiom_t {
    virtual std::vector<block_hint_t<dim_type_t>> get_hints(
            const stride_layout_t<dim_type_t> &layout) const = 0;
};

// The uniform blocked pattern corresponds to a sequence blocked send
// instructions which load a multiple of size data.
struct uniform_blocked_pattern_t {
    constexpr uniform_blocked_pattern_t(dim_t size, dim_t alignment)
        : size(size), alignment(alignment) {}
    constexpr uniform_blocked_pattern_t(const uniform_blocked_pattern_t &)
            = default;

    std::string str() const {
        std::ostringstream oss;
        oss << "uniform " << size << " byte blocked";
        return oss.str();
    }
    dim_t size;
    dim_t alignment;
};

template <typename dim_type_t>
struct uniform_blocked_idiom_t final : public send_idiom_t<dim_type_t> {
    constexpr uniform_blocked_idiom_t(const uniform_blocked_pattern_t &data)
        : data_(data) {}

    using hint_t = block_hint_t<dim_type_t>;
    using slayout_t = stride_layout_t<dim_type_t>;

    dim_t load_size() const { return data_.size; }
    dim_t alignment() const { return data_.alignment; }

    std::vector<hint_t> get_hints(const slayout_t &layout,
            typename slayout_t::stride_array_t::const_iterator i,
            const hint_t &hint, dim_t load_rem) const {
        // Base case: whole layout has been processed and the hint satisfies
        // alignment checks.
        if (i == layout.strides_end()) {
            if (load_rem)
                return {};
            else
                return {hint};
        }

        auto i_stride_bytes = i->stride * layout.type_size;

        // The stride_layout_t structure is sorted by stride, therefore if the
        // stride exceeds the current load stride, it is impossible to pack any
        // more data into a blocked load instruction.
        if (i_stride_bytes > load_size() && load_rem) { return {}; }

        // Get hints which skip the current dimension
        std::vector<hint_t> skip_hints = [&] {
            bool is_aligned = i_stride_bytes % alignment() == 0;

            bool no_partial_overflow = !i->can_overflow
                    || (i_stride_bytes % load_size() == 0
                            && layout.buffer_size % load_size() == 0);

            // Get all hints skipping this dimension
            if (is_aligned && no_partial_overflow)
                return get_hints(layout, i + 1, hint, load_rem);
            else {
                return std::vector<hint_t> {};
            }
        }();

        // Get hints which use the current dimension
        std::vector<hint_t> use_hints = [&] {
            if (load_rem == 0) { return std::vector<hint_t> {}; }

            // Check data is contiguous
            auto stride = std::max(layout.type_size, load_size() / load_rem);
            if (stride != i_stride_bytes || i->is_complex) {
                return std::vector<hint_t> {};
            }

            // Alignment check is unnecessary as the loads stride enforces proper
            // alignment
            if (i->size >= load_rem && i->size % load_rem == 0) {
                // No masking means the final load block must be divisible by the
                // load_size
                hint_t new_hint = hint;
                new_hint[i->dim] = load_rem;
                return get_hints(layout, i + 1, new_hint, 0);

            } else if (load_rem % i->size == 0) {
                // No masking means we cannot handle any overflow
                hint_t new_hint = hint;
                new_hint[i->dim] = i->size;
                return get_hints(layout, i + 1, new_hint, load_rem / i->size);
            } else {
                // Dimension cannot be packed into a blocked load
                return std::vector<hint_t> {};
            }
        }();

        use_hints.insert(use_hints.end(), skip_hints.begin(), skip_hints.end());
        return use_hints;
    }

    std::vector<hint_t> get_hints(const slayout_t &layout) const override {
        dim_t size = load_size() / layout.type_size;
        hint_t hint;
        return get_hints(layout, layout.strides.begin(), hint, size);
    }

private:
    uniform_blocked_pattern_t data_;
};

// The uniform blocked pattern corresponds to a sequence blocked send
// instructions which load a multiple of size data.
struct uniform_2d_pattern_t {
    constexpr uniform_2d_pattern_t(double utilization)
        : min_utilization(utilization) {};
    constexpr uniform_2d_pattern_t(const uniform_2d_pattern_t &) = default;

    std::string str() const {
        std::ostringstream oss;
        oss << "uniform (" << min_utilization << " utilization) 2d blocked";
        return oss.str();
    }

    uniform_2d_pattern_t with_size(int size) const {
        return uniform_2d_pattern_t(1.0 * size / (block_width * block_height));
    }

    static const dim_t width_alignment = 4;
    static const dim_t pitch_alignment = 8;
    static const dim_t surface_width_min_size = 64;
    static const dim_t surface_width_alignment = 4;

    static const dim_t block_width = 64; // Blocked width in bytes
    static const dim_t block_height = 32; // Rows of blocked width

    double min_utilization;
};

template <typename dim_id_t>
struct uniform_2d_idiom_t final : public send_idiom_t<dim_id_t> {
    constexpr uniform_2d_idiom_t(const uniform_2d_pattern_t &data)
        : data_(data) {};

    using hint_t = block_hint_t<dim_id_t>;
    using slayout_t = stride_layout_t<dim_id_t>;

    dim_t block_width() const { return data_.block_width; }
    dim_t block_height() const { return data_.block_height; }
    double min_utilization() const { return data_.min_utilization; }
    dim_t width_alignment() const {
        return uniform_2d_pattern_t::width_alignment;
    }
    dim_t surface_width_alignment() const {
        return uniform_2d_pattern_t::surface_width_alignment;
    }
    dim_t surface_width_min_size() const {
        return uniform_2d_pattern_t::surface_width_min_size;
    }
    dim_t pitch_alignment() const {
        return uniform_2d_pattern_t::pitch_alignment;
    }

    struct rem_t {
        dim_t block_width;
        dim_t block_height;
    };
    struct surface_t {
        static const dim_t unknown = 0;
        dim_t width = unknown;
        dim_t height = unknown;
        dim_t pitch = unknown;
    };

    std::vector<hint_t> get_hints(const slayout_t &layout,
            typename slayout_t::stride_array_t::const_iterator i,
            const hint_t &hint, rem_t load_rem, surface_t surface) const {
        // Base case: whole layout has been processed and the hint satisfies
        // alignment checks.
        if (i == layout.strides_end()) {
            double utilization
                    = static_cast<double>(hint.size() * layout.type_size)
                    / (block_width() * block_height());
            // (static_cast<double>(surface.width)
            //           / utils::rnd_up(surface.width, block_width()))
            // * (static_cast<double>(surface.height)
            //         / utils::rnd_up(surface.height, block_height()));
            ir_info() << "Hint " << hint << " with size: " << hint.size()
                      << " and utilization: " << utilization
                      << " vs min_utilization " << min_utilization() << "  ";
            if (utilization < min_utilization()) {
                ir_info() << "Hint Failed\n";
                return {};
            } else {
                ir_info() << "Hint Passed\n";
                return {hint};
            }
        }

        ir_info() << "Processing " << i->dim << ": " << i->size << "\n";
        auto i_stride_bytes = i->stride * layout.type_size;
        auto width_stride = load_rem.block_width ? std::max(layout.type_size,
                                    block_width() / load_rem.block_width)
                                                 : block_width();
        ir_info() << "Processing " << i->dim << ": " << i->size << "*"
                  << i_stride_bytes << "\n";

        // Check hint is potentially valid
        if (width_stride < i_stride_bytes
                && min_utilization() * load_rem.block_width > 1) {
            ir_info() << "Stopping search due to invalid width packing\n";
            return std::vector<hint_t> {};
        }

        // Get hints which skip the current dimension
        std::vector<hint_t> skip_hints = [&] {
            ir_info() << "Skipping " << i->dim << " packing\n";
            bool is_aligned = i_stride_bytes % width_alignment() == 0;

            // The 2d send instruction zero extends any data outside the surface
            // width and surface height
            bool no_partial_overflow = true;

            // Get all hints skipping this dimension
            if (is_aligned && no_partial_overflow)
                return get_hints(layout, i + 1, hint, load_rem, surface);
            else {
                ir_info() << "Cannot skip packing: " << i->dim << ": " << i->size
                          << " due to alignment\n";
                return std::vector<hint_t> {};
            }
        }();

        auto pack_2d = [&](int size, rem_t new_rem, hint_t new_hint,
                               surface_t new_surface) {
            if (size >= load_rem.block_height) {
                new_hint[i->dim] = new_hint[i->dim] == hint_t::unset
                        ? load_rem.block_height
                        : new_hint[i->dim] * load_rem.block_height;
                new_rem.block_height = 0;
                new_surface.height = new_surface.height == surface_t::unknown
                        ? size
                        : new_surface.height * size;
                auto ret = get_hints(
                        layout, i + 1, new_hint, new_rem, new_surface);
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
                if (load_rem.block_height % size == 0) {
                    new_rem.block_height = load_rem.block_height / size;
                } else {
                    // The 2d send instruction zero extends any data outside
                    // the surface width and surface height
                    new_rem.block_height = 0;
                }
                new_surface.height = new_surface.height == surface_t::unknown
                        ? size
                        : new_surface.height * size;
                ir_info() << "new_rem.block_height: " << new_rem.block_height
                          << "\n";
                auto ret = get_hints(
                        layout, i + 1, new_hint, new_rem, new_surface);
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
            if (!load_rem.block_height
                    || surface.width < surface_width_min_size()
                    || i->is_complex) {
                ir_info() << "Cannot pack into height: block_height: "
                          << load_rem.block_height
                          << " surface width: " << surface.width
                          << " is_complex: " << i->is_complex << "\n";
                return std::vector<hint_t> {};
            }

            // Attempt to pack data into block_height
            auto new_surface = surface;
            if (surface.pitch == surface_t::unknown) {
                if (i_stride_bytes % pitch_alignment() != 0
                        || i_stride_bytes < surface.width) {
                    ir_info() << "Cannot pack height as data is not aligned\n";
                    return std::vector<hint_t> {};
                }
                new_surface.pitch = i_stride_bytes;
            } else {
                // Check data is contiguous
                auto stride = (block_height() / load_rem.block_height)
                        * surface.pitch;
                if (stride != i_stride_bytes) {
                    ir_info()
                            << "Cannot pack height as data is not contiguous\n";
                    return std::vector<hint_t> {};
                }
            }

            return pack_2d(i->size, load_rem, hint, new_surface);
        }();

        // Get hints which use the current dimension
        std::vector<hint_t> use_width_hints = [&] {
            ir_info() << "Packing " << i->dim << " into width\n";
            // Cannot pack this dimension;
            if (i_stride_bytes != width_stride || i->is_complex
                    || surface.pitch != surface_t::unknown
                    || load_rem.block_width == 0) {
                ir_info() << "Cannot pack width i_stride_bytes: "
                          << i_stride_bytes << " pitch: " << surface.pitch
                          << " load_rem.block_width: " << load_rem.block_width
                          << "\n";
                return std::vector<hint_t> {};
            }

            // No need to check block_width alignment, it is enforced by the
            // surface_pitch and fact block_width is loaded in power of 2
            // sizes.
            if (i->size >= load_rem.block_width) {
                hint_t new_hint = hint;
                rem_t new_rem = load_rem;
                auto new_surface = surface;
                new_hint[i->dim] = load_rem.block_width;
                new_rem.block_width = 0;
                ir_info() << "Fully Packing " << i->dim << " into width\n";

                auto ret = [&]() {
                    // Get blocked load using 2d send
                    if (i->size > load_rem.block_width
                            && i->size % load_rem.block_width == 0) {
                        ir_info() << "Fully Packing " << i->dim
                                  << "height spillover\n";
                        auto blocked_surface = surface;
                        blocked_surface.pitch = block_width();
                        blocked_surface.width = block_width();
                        return pack_2d(i->size / load_rem.block_width, new_rem,
                                new_hint, blocked_surface);
                    }
                    ir_info()
                            << "No height spillover to pack: size: " << i->size
                            << "block_width: " << load_rem.block_width << "\n";
                    return std::vector<hint_t> {};
                }();

                new_surface.width = i->size * i_stride_bytes;
                if (new_surface.width % surface_width_alignment()) {
                    // Surface width must be aligned to max(4,
                    // elem_size). The elem_size requirement is
                    // implicitly enforced by the stride.
                    ir_info() << "Cannot pack width due to "
                                 "alignment: "
                              << i->dim << ": " << i->size << "\n";
                    return std::vector<hint_t> {};
                }

                auto ret2 = get_hints(
                        layout, i + 1, new_hint, new_rem, new_surface);
                ret.insert(ret.end(), ret2.begin(), ret2.end());
                ir_info() << "Fully width packed " << i->dim << "Resulted in "
                          << ret.size() << " hints:\n";
                for (auto &i : ret)
                    ir_info() << i << "\n";
                ir_info() << "\n";
                return ret;

            } else if (load_rem.block_width % i->size == 0) {
                // Cannot partially pack as we need surface_width >=
                // surface_width_min_size() >= block_width
                hint_t new_hint = hint;
                rem_t new_rem = load_rem;
                auto new_surface = surface;
                new_hint[i->dim] = i->size;
                new_rem.block_width = load_rem.block_width / i->size;
                new_surface.width = i->size * i_stride_bytes;

                auto ret = get_hints(
                        layout, i + 1, new_hint, new_rem, new_surface);
                ir_info() << "Sub width packed " << i->dim << "Resulted in "
                          << ret.size() << " hints:\n";
                for (auto &i : ret)
                    ir_info() << i << "\n";
                ir_info() << "\n";
                return ret;
            }
            return std::vector<hint_t> {};
        }();

        use_width_hints.insert(use_width_hints.end(), use_height_hints.begin(),
                use_height_hints.end());
        use_width_hints.insert(
                use_width_hints.end(), skip_hints.begin(), skip_hints.end());
        return use_width_hints;
    }

    std::vector<hint_t> get_hints(const slayout_t &layout) const override {
        ir_info() << "Getting Hints for: " << layout << "\n";
        rem_t rem;
        rem.block_width = block_width() / layout.type_size;
        rem.block_height = block_height();
        hint_t hint;
        auto ret = get_hints(
                layout, layout.strides.begin(), hint, rem, surface_t());
        // if (!ret.empty()) {
        //     std::cout << "Hints for: " << layout << "\n";
        //     for (auto &h : ret) {
        //         std::cout << h << "\n";
        //     }
        // } else
        //     std::cout << "No hints for " << layout << "\n";
        return ret;
    }

private:
    uniform_2d_pattern_t data_;
};
class stmt_t;
// Tagged union for storing load patterns
struct send_pattern_t {
    send_pattern_t() : type_id_(empty) {};
    send_pattern_t(const uniform_blocked_pattern_t &data)
        : type_id_(uniform_blocked), data_block_(data) {};
    void operator=(const uniform_blocked_pattern_t &data) {
        type_id_ = uniform_blocked;
        data_block_ = data;
    }
    send_pattern_t(const uniform_2d_pattern_t &data)
        : type_id_(uniform_2d), data_2d_(data) {};
    void operator=(const uniform_2d_pattern_t &data) {
        type_id_ = uniform_2d;
        data_2d_ = data;
    }

    bool is_empty() const { return type_id_ == empty; }
    bool is_uniform_blocked() const { return type_id_ == uniform_blocked; }
    bool is_uniform_2d() const { return type_id_ == uniform_2d; }
    bool matches(const stmt_t &stmt) const;
    const uniform_blocked_pattern_t &as_uniform_blocked() const {
        return data_block_;
    }
    const uniform_2d_pattern_t &as_uniform_2d() const { return data_2d_; }
    std::string str() const {
        switch (type_id_) {
            case uniform_blocked: return data_block_.str();
            case uniform_2d: return data_2d_.str();
            default: return "(empty)";
        }
    }

private:
    // Tagged types
    enum type_id_t {
        empty = 0,
        uniform_blocked = 1,
        uniform_2d = 2,
    };

    type_id_t type_id_;
    union {
        uniform_blocked_pattern_t data_block_;
        uniform_2d_pattern_t data_2d_;
    };
};

const std::vector<uniform_blocked_pattern_t> &get_uniform_blocked_patterns(
        compute::gpu_arch_t arch);
const std::vector<uniform_2d_pattern_t> &get_uniform_2d_patterns(
        compute::gpu_arch_t arch);
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
