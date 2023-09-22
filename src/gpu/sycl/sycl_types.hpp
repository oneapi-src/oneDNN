/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_SYCL_SYCL_TYPES_HPP
#define GPU_SYCL_SYCL_TYPES_HPP

#include <limits>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

// The macros are expected to be called within a command group function object
// that is passed to `parallel_for`.
#define CTX_IN_SYCL_KERNEL_MEMORY(arg) \
    CTX_IN_STORAGE(arg).is_null() \
            ? sycl_memory_storage_base_t::empty_in_memory_arg( \
                    ctx.stream(), cgh) \
            : utils::downcast<const impl::sycl::sycl_memory_storage_base_t *>( \
                    &CTX_IN_STORAGE(arg)) \
                      ->get_in_memory_arg(ctx.stream(), cgh)

#define CTX_OUT_SYCL_KERNEL_MEMORY(arg) \
    CTX_OUT_STORAGE(arg).is_null() \
            ? sycl_memory_storage_base_t::empty_out_memory_arg( \
                    ctx.stream(), cgh) \
            : utils::downcast<const impl::sycl::sycl_memory_storage_base_t *>( \
                    &CTX_OUT_STORAGE(arg)) \
                      ->get_out_memory_arg(ctx.stream(), cgh)

#define CHECK_SYCL_KERNEL_ARG_TYPE(type) \
    static_assert(::sycl::is_device_copyable_v<type>)

template <::sycl::access_mode mode>
struct sycl_memory_arg_t {
    using acc_dt = uint8_t;
    using acc_t = ::sycl::accessor<acc_dt, 1, mode>;
    static sycl_memory_arg_t<mode> create_empty(const acc_t &dummy_acc) {
        sycl_memory_arg_t<mode> arg(nullptr, dummy_acc);
        arg.empty_ = true;
        return arg;
    }

    sycl_memory_arg_t(void *usm, const acc_t &dummy_acc)
        : empty_(false), usm_(usm), acc_(dummy_acc) {}
    sycl_memory_arg_t(const acc_t &acc)
        : empty_(false), usm_(nullptr), acc_(acc) {}
    // This method must be called only from inside a kernel.
    void *get_pointer() const {
        if (usm_) return usm_;
        return const_cast<acc_dt *>(
                acc_.template get_multi_ptr<::sycl::access::decorated::no>()
                        .get());
    }

    bool empty() const { return empty_; }

private:
    bool empty_;
    void *usm_;
    acc_t acc_;
};

// TODO: come up with better names?
using sycl_in_memory_arg_t = sycl_memory_arg_t<::sycl::access::mode::read>;
using sycl_out_memory_arg_t = sycl_memory_arg_t<::sycl::access::mode::write>;
using sycl_inout_memory_arg_t
        = sycl_memory_arg_t<::sycl::access::mode::read_write>;

// TODO: this class mimics memory_desc_t and makes sure it can be passed
// to SYCL kernels as a kernel argument. SYCL puts restrictions on kernel
// arguments, e.g. those cannot contain unions.
struct sycl_md_t {
    // There is a limitation on total size of kernel arguments hence using
    // reduced number of supported dimensions and int32_t for dimensions.
    static constexpr int max_dims = 6;

    using dim32_t = int32_t;
    using dims32_t = dim32_t[max_dims];

    data_type_t data_type() const { return data_type_; }

    dim32_t ndims() const { return ndims_; }
    dim32_t offset0() const { return offset0_; }
    dim32_t inner_nblks() const { return inner_nblks_; }

    const dims32_t &dims() const { return dims_; }
    const dims32_t &padded_dims() const { return padded_dims_; }
    const dims32_t &padded_offsets() const { return padded_offsets_; }
    const dims32_t &strides() const { return strides_; }
    const dims32_t &inner_blks() const { return inner_blks_; }
    const dims32_t &inner_idxs() const { return inner_idxs_; }

    sycl_md_t() = default;
    sycl_md_t(const memory_desc_t *md) {
        memory_desc_wrapper mdw(md);

        assert(mdw.format_kind() == format_kind::blocked);
        assert(mdw.ndims() <= max_dims);

        const auto &blk = mdw.blocking_desc();

        data_type_ = mdw.data_type();
#define CHECK_AND_ASSIGN(lhs, rhs) \
    assert((rhs) <= INT32_MAX); \
    (lhs) = (rhs)

        CHECK_AND_ASSIGN(ndims_, mdw.ndims());
        CHECK_AND_ASSIGN(offset0_, mdw.offset0());
        CHECK_AND_ASSIGN(inner_nblks_, blk.inner_nblks);

        for (int d = 0; d < mdw.ndims(); d++) {
            CHECK_AND_ASSIGN(dims_[d], mdw.dims()[d]);
            CHECK_AND_ASSIGN(padded_dims_[d], mdw.padded_dims()[d]);
            CHECK_AND_ASSIGN(padded_offsets_[d], mdw.padded_offsets()[d]);
            CHECK_AND_ASSIGN(strides_[d], blk.strides[d]);
            CHECK_AND_ASSIGN(inner_blks_[d], blk.inner_blks[d]);
            CHECK_AND_ASSIGN(inner_idxs_[d], blk.inner_idxs[d]);
        }
#undef CHECK_AND_ASSIGN
    }

    template <typename... Args>
    dim_t off(Args... args) const {
        dims_t pos = {args...};
        return off_v(pos, false);
    }

    dim_t off_v(const dims_t pos, bool is_pos_padded = false) const {
        dims_t pos_copy = {0};
        for (int d = 0; d < ndims(); ++d)
            pos_copy[d] = pos[d] + (is_pos_padded ? 0 : padded_offsets()[d]);
        dim_t phys_offset = offset0();

        if (inner_nblks() > 0) {
            dim_t blk_stride = 1;
            for (int iblk = inner_nblks() - 1; iblk >= 0; --iblk) {
                const int d = inner_idxs()[iblk];

                dim_t p;
                if (pos_copy[d] <= INT32_MAX) {
                    p = (int32_t)pos_copy[d] % (int32_t)inner_blks()[iblk];
                    pos_copy[d] = (int32_t)pos_copy[d]
                            / (int32_t)inner_blks()[iblk];
                } else {
                    p = pos_copy[d] % inner_blks()[iblk];
                    pos_copy[d] /= inner_blks()[iblk];
                }

                phys_offset += p * blk_stride;

                blk_stride *= inner_blks()[iblk];
            }
        }

        for (int d = 0; d < ndims(); ++d) {
            const dim_t p = pos_copy[d];
            phys_offset += p * strides()[d];
        }
        return phys_offset;
    }

    dim_t off_l(dim_t l_offset, bool is_pos_padded = false) const {
        dims_t pos;
        for (int rd = 0; rd < ndims(); ++rd) {
            const int d = ndims() - 1 - rd;
            const dim_t cur_dim = is_pos_padded ? padded_dims()[d] : dims()[d];
            if (l_offset <= INT32_MAX && cur_dim <= INT32_MAX) {
                pos[d] = (int32_t)l_offset % (int32_t)cur_dim;
                l_offset = (int32_t)l_offset / (int32_t)cur_dim;
            } else {
                pos[d] = l_offset % cur_dim;
                l_offset /= cur_dim;
            }
        }
        return off_v(pos, is_pos_padded);
    }

private:
    data_type_t data_type_;

    dim32_t ndims_;

    dims32_t dims_;
    dims32_t padded_dims_;
    dims32_t padded_offsets_;
    dim32_t offset0_;

    dims32_t strides_;
    dim32_t inner_nblks_;
    dims32_t inner_blks_;
    dims32_t inner_idxs_;
};

struct bfloat16_t {
    uint16_t raw_bits_;
    bfloat16_t() = default;
    constexpr bfloat16_t(uint16_t r) : raw_bits_(r) {}
    bfloat16_t(float f) { (*this) = f; }

    bfloat16_t &operator=(float f) {
        auto iraw = utils::bit_cast<std::array<uint16_t, 2>>(f);

        if (::sycl::isnormal(f)) {
            // FP_NORMAL: round to nearest even and truncate
            const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
            const uint32_t int_raw
                    = utils::bit_cast<uint32_t>(f) + rounding_bias;
            iraw = utils::bit_cast<std::array<uint16_t, 2>>(int_raw);
            raw_bits_ = iraw[1];
        } else if (::sycl::isinf(f)) {
            // FP_INFINITE
            raw_bits_ = iraw[1];
        } else if (::sycl::isnan(f)) {
            // FP_NAN: truncate and set MSB of the mantissa force QNAN
            raw_bits_ = iraw[1];
            raw_bits_ |= 1 << 6;
        } else {
            // FP_SUBNORMAL, FP_ZERO: sign preserving zero (denormal go to zero)
            raw_bits_ = iraw[1];
            raw_bits_ &= 0x8000;
        }

        return *this;
    }

    operator float() const {
        std::array<uint16_t, 2> iraw = {{0, raw_bits_}};
        return utils::bit_cast<float>(iraw);
    }
};

using float16_t = ::sycl::half;

// Add a check for every SYCL kernel argument type.
//
// Exception: sycl_memory_arg_t doesn't pass the check because it contains
// sycl::accessor which is not device copyable. However, it is treated by the
// compiler in a special way allowing it not to satisfy the requirement.
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_md_t);
CHECK_SYCL_KERNEL_ARG_TYPE(bfloat16_t);

template <data_type_t>
struct sycl_prec_traits;

template <>
struct sycl_prec_traits<data_type::f16> {
    using type = float16_t;
};
template <>
struct sycl_prec_traits<data_type::bf16> {
    using type = bfloat16_t;
};
template <>
struct sycl_prec_traits<data_type::f32> {
    using type = float;
};
template <>
struct sycl_prec_traits<data_type::s32> {
    using type = int32_t;
};
template <>
struct sycl_prec_traits<data_type::s8> {
    using type = int8_t;
};
template <>
struct sycl_prec_traits<data_type::u8> {
    using type = uint8_t;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

namespace std {

template <>
class numeric_limits<dnnl::impl::gpu::sycl::bfloat16_t> {
public:
    static constexpr dnnl::impl::gpu::sycl::bfloat16_t lowest() {
        return {uint16_t(0xff7f)};
    }
    static constexpr dnnl::impl::gpu::sycl::bfloat16_t max() {
        return {uint16_t(0x7f7f)};
    }
    static constexpr int digits = 8;
    static constexpr dnnl::impl::gpu::sycl::bfloat16_t epsilon() {
        return {uint16_t((0x7f - (digits - 1)) << (digits - 1))};
    }
};

} // namespace std

#endif
