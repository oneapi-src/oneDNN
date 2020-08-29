/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

template <::sycl::access_mode mode>
struct sycl_memory_arg_t {
    using acc_t = ::sycl::accessor<uint8_t, 1, mode>;

    sycl_memory_arg_t(void *usm, const acc_t &dummy_acc)
        : usm_(usm), acc_(dummy_acc) {}
    sycl_memory_arg_t(const acc_t &acc) : usm_(nullptr), acc_(acc) {}
    // This method must be called only from inside a kernel.
    void *get_pointer() { return usm_ ? usm_ : acc_.get_pointer().get(); }

private:
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

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
