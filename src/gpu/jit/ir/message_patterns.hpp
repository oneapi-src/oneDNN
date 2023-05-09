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
        dim_t size;
        dim_t stride;
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
        dim_t size;
        dim_t stride;

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
        for (int id = 0; id < dim_type_t::max_id(); id++) {
            auto i = dim_type_t::from_id(id);
            if (hint_[i.id()] > 1) {
                oss << " " << i.str() << ":" << hint_[i.id()];
            }
        }
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

private:
    std::array<dim_t, dim_type_t::max_id()> hint_ = {};
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

class stmt_t;
// Tagged union for storing load patterns
struct send_pattern_t {
    send_pattern_t() : type_id_(empty) {};
    send_pattern_t(const uniform_blocked_pattern_t &data)
        : type_id_(uniform_blocked), block_data_(data) {};
    void operator=(const uniform_blocked_pattern_t &data) {
        type_id_ = uniform_blocked;
        block_data_ = data;
    }

    bool is_empty() const { return type_id_ == empty; }
    bool is_uniform_blocked() const { return type_id_ == uniform_blocked; }
    bool matches(const stmt_t &stmt) const;
    const uniform_blocked_pattern_t &as_uniform_blocked() const {
        return block_data_;
    }
    std::string str() const {
        switch (type_id_) {
            case uniform_blocked: return block_data_.str();
            default: return "(empty)";
        }
    }

private:
    // Tagged types
    enum type_id_t {
        empty = 0,
        uniform_blocked = 1,
    };

    type_id_t type_id_;
    union {
        uniform_blocked_pattern_t block_data_;
    };
};

const std::vector<uniform_blocked_pattern_t> &get_uniform_blocked_patterns(
        compute::gpu_arch_t arch);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
