/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_SYCL_BINARY_KERNELS_HPP
#define GPU_SYCL_BINARY_KERNELS_HPP

#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct binary_kernel_vec_t {
    static constexpr int vec_len = 8;
    static constexpr int max_supported_ndims = 5;

    binary_kernel_vec_t(const sycl_binary_conf_t &conf,
            xpu::sycl::in_memory_arg_t &src0, xpu::sycl::in_memory_arg_t &src1,
            xpu::sycl::out_memory_arg_t &dst,
            xpu::sycl::in_memory_arg_t &src0_scale,
            xpu::sycl::in_memory_arg_t &src1_scale, data_type_t scales_dt,
            xpu::sycl::in_memory_arg_t &po1_src,
            xpu::sycl::in_memory_arg_t &po2_src,
            xpu::sycl::in_memory_arg_t &po3_src,
            xpu::sycl::in_memory_arg_t &po4_src,
            xpu::sycl::in_memory_arg_t &po5_src)
        : conf_(conf)
        , src0_(src0)
        , src1_(src1)
        , dst_(dst)
        , src0_scale_(src0_scale)
        , src1_scale_(src1_scale)
        , scales_dt_(scales_dt)
        , po1_src_(po1_src)
        , po2_src_(po2_src)
        , po3_src_(po3_src)
        , po4_src_(po4_src)
        , po5_src_(po5_src) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;

        size_t base_idx = offset_t * conf_.block_size;
        size_t vec_base_idx = base_idx / vec_len;

        size_t sg_base_idx = (wg_offset_t + sg_offset_t) * conf_.block_size;

        const float sm_0 = (conf_.do_scale_src0
                        ? load_float_value(scales_dt_, src0_scale_ptr(), 0)
                        : 1.f);

        const float sm_1 = (conf_.do_scale_src1
                        ? load_float_value(scales_dt_, src1_scale_ptr(), 0)
                        : 1.f);

        dims_t dims, off;
        bool any_broadcast = false;
        for (int i = 0; i < max_supported_ndims; i++) {
            dims[i] = (i < src0_md().ndims()) ? src0_md().dims()[i] : 1;
            if (i < src0_md().ndims()) {
                any_broadcast |= conf_.broadcast_dims[i];
            }
        }
        if (!any_broadcast && conf_.post_ops.get_post_op() == 0
                && sg_base_idx + (sg.get_local_range()[0] * conf_.block_size)
                        < conf_.wk_size) {
            for (int i = 0; i < conf_.block_size / vec_len; i++) {
                auto src0_vec = load_float_vec<vec_len>(
                        src0_md().data_type(), src0_ptr(), vec_base_idx + i);
                auto src1_vec = load_float_vec<vec_len>(
                        src1_md().data_type(), src1_ptr(), vec_base_idx + i);
                auto dst_vec = load_float_vec<vec_len>(
                        dst_md().data_type(), dst_ptr(), vec_base_idx + i);

                if (conf_.do_scale_src0)
                    src0_vec *= ::sycl::vec<float, vec_len>(sm_0);
                if (conf_.do_scale_src1)
                    src1_vec *= ::sycl::vec<float, vec_len>(sm_1);

                auto acc_vec = compute_alg(src0_vec, src1_vec, conf_.alg_kind);
                // TODO: Adding post-ops seems to be interfering with compiler's
                // optimizations. Figure out how to make the compiler to generate
                // the right code.
                acc_vec = conf_.post_ops.apply(acc_vec, dst_vec);
                store_float_vec(dst_md().data_type(), acc_vec, dst_ptr(),
                        vec_base_idx + i);
            }
        } else {
            for (int i = 0; i < conf_.block_size; i++) {
                int idx = base_idx + i;
                if (idx < conf_.wk_size) {
                    utils::l_dims_by_l_offset(
                            off, idx, dims, max_supported_ndims);

                    for (int i = 0; i < max_supported_ndims; i++) {
                        if (conf_.broadcast_dims[i] && i < src0_md().ndims()) {
                            off[i] = 0;
                        }
                    }

                    int idx1 = src1_md().off_v(off);

                    auto src0 = load_float_value(
                            src0_md().data_type(), src0_ptr(), idx);
                    auto src1 = load_float_value(
                            src1_md().data_type(), src1_ptr(), idx1);
                    auto dst = load_float_value(
                            dst_md().data_type(), dst_ptr(), idx);

                    if (conf_.do_scale_src0) src0 *= sm_0;
                    if (conf_.do_scale_src1) src1 *= sm_1;

                    auto acc = compute_alg_n(src0, src1, conf_.alg_kind);
                    ::sycl::vec<float, 16> post_po_sr = post_op_src_val(idx);
                    acc = conf_.post_ops.apply(acc, dst, post_po_sr);
                    store_float_value(
                            dst_md().data_type(), acc, dst_ptr(), idx);
                }
            }
        }
    }

private:
    const xpu::sycl::md_t &src0_md() const { return conf_.src0_md; }
    const xpu::sycl::md_t &src1_md() const { return conf_.src1_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }

    void *src0_ptr() const { return src0_.get_pointer(); }
    void *src1_ptr() const { return src1_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    float *src0_scale_ptr() const {
        return static_cast<float *>(src0_scale_.get_pointer());
    }
    float *src1_scale_ptr() const {
        return static_cast<float *>(src1_scale_.get_pointer());
    }

    inline ::sycl::vec<float, 16> post_op_src_val(dim_t data_l_off) const {
        ::sycl::vec<float, 16> post_po_sr;
        const auto maxPostPo = conf_.post_ops.get_post_op();

        for (dim_t po_idx = 0; po_idx < maxPostPo; po_idx++) {
            float res = 0.0f;
            if (po_idx == 0)
                res = get_post_op_val(po1_src_, po_idx, data_l_off);
            else if (po_idx == 1)
                res = get_post_op_val(po2_src_, po_idx, data_l_off);
            else if (po_idx == 2)
                res = get_post_op_val(po3_src_, po_idx, data_l_off);
            else if (po_idx == 3)
                res = get_post_op_val(po4_src_, po_idx, data_l_off);
            else if (po_idx == 4)
                res = get_post_op_val(po5_src_, po_idx, data_l_off);

            post_po_sr[po_idx] = res;
        }
        return post_po_sr;
    }

    float get_post_op_val(const xpu::sycl::in_memory_arg_t &bin_src_op,
            dim_t &idx, dim_t offset) const {
        auto src1_desc = conf_.binary_src_arr[idx];

        const auto off = get_binary_src1_off(
                src1_desc, offset, dst_md().dims(), dst_md().ndims());

        auto dst = load_float_value(
                src1_desc.data_type(), bin_src_op.get_pointer(), off);
        return dst;
    }

    dim_t get_binary_src1_off(const xpu::sycl::md_t &src1_md, dim_t l_offset,
            const xpu::sycl::md_t::dims32_t &dst_dims,
            const xpu::sycl::md_t::dim32_t &dst_ndims) const {
        const dim_t mask_binary_po
                = get_dims_mask(dst_dims, src1_md.dims(), dst_ndims);
        return get_po_tensor_off(
                src1_md, l_offset, dst_dims, dst_ndims, mask_binary_po);
    }

    inline dim_t get_dims_mask(const xpu::sycl::md_t::dims32_t &dims1,
            const xpu::sycl::md_t::dims32_t &dims2, const dim_t &ndims,
            bool skip_dim_of_one = false) const {
        dim_t mask = 0;
        for (dim_t d = 0; d < ndims; ++d) {
            // Disable mask_bit for dimensions of `1` by request.
            dim_t mask_bit = skip_dim_of_one && dims1[d] == 1 ? 0 : (1 << d);
            mask += dims1[d] == dims2[d] ? mask_bit : 0;
        }
        return mask;
    }

    inline dim_t get_po_tensor_off(const xpu::sycl::md_t &tensor_md,
            dim_t l_offset, const xpu::sycl::md_t::dims32_t &dst_dims,
            const dim_t &dst_ndims, const dim_t &mask) const {
        dims_t l_dims_po {};
        get_l_dims_po(l_dims_po, l_offset, dst_dims, dst_ndims, mask);

        return tensor_md.off_v(l_dims_po);
    }

    inline void get_l_dims_po(dims_t l_dims_po, dim_t l_offset,
            const xpu::sycl::md_t::dims32_t &dst_dims, const dim_t &dst_ndims,
            const dim_t &mask) const {

        l_dims_by_l_offset(l_dims_po, l_offset, dst_dims, dst_ndims);
        utils::apply_mask_on_dims(l_dims_po, dst_ndims, mask);
    }

    inline void l_dims_by_l_offset(dims_t dims_pos, dim_t l_offset,
            const xpu::sycl::md_t::dims32_t &dims, const dim_t &ndims) const {
        for (dim_t rd = 0; rd < ndims; ++rd) {
            const dim_t d = ndims - 1 - rd;
            /* switch to faster 32-bit division when possible. */
            if (l_offset <= INT32_MAX && dims[d] <= INT32_MAX) {
                dims_pos[d] = (int32_t)l_offset % (int32_t)dims[d];
                l_offset = (int32_t)l_offset / (int32_t)dims[d];
            } else {
                dims_pos[d] = l_offset % dims[d];
                l_offset /= dims[d];
            }
        }
    }

    template <int width>
    ::sycl::vec<float, width> compute_alg(::sycl::vec<float, width> src0,
            ::sycl::vec<float, width> src1, alg_kind_t alg) const {
        switch (alg) {
            case alg_kind::binary_add: return src0 + src1;
            case alg_kind::binary_div: return src0 / src1;
            case alg_kind::binary_max: return ::sycl::fmax(src0, src1);
            case alg_kind::binary_min: return ::sycl::fmin(src0, src1);
            case alg_kind::binary_mul: return src0 * src1;
            case alg_kind::binary_sub: return src0 - src1;
            case alg_kind::binary_ge:
                return ((src0 >= src1) * -1).template convert<float>();
            case alg_kind::binary_gt:
                return ((src0 > src1) * -1).template convert<float>();
            case alg_kind::binary_le:
                return ((src0 <= src1) * -1).template convert<float>();
            case alg_kind::binary_lt:
                return ((src0 < src1) * -1).template convert<float>();
            case alg_kind::binary_eq:
                return ((src0 == src1) * -1).template convert<float>();
            case alg_kind::binary_ne:
                return ((src0 != src1) * -1).template convert<float>();
            default: return ::sycl::vec<float, width> {NAN};
        }
    }

    template <typename T>
    T compute_alg_n(T src0, T src1, alg_kind_t alg) const {
        switch (alg) {
            case alg_kind::binary_add: return src0 + src1;
            case alg_kind::binary_div: return src0 / src1;
            case alg_kind::binary_max: return ::sycl::max(src0, src1);
            case alg_kind::binary_min: return ::sycl::min(src0, src1);
            case alg_kind::binary_mul: return src0 * src1;
            case alg_kind::binary_sub: return src0 - src1;
            case alg_kind::binary_ge: return ((src0 >= src1));
            case alg_kind::binary_gt: return ((src0 > src1));
            case alg_kind::binary_le: return ((src0 <= src1));
            case alg_kind::binary_lt: return ((src0 < src1));
            case alg_kind::binary_eq: return ((src0 == src1));
            case alg_kind::binary_ne: return ((src0 != src1));
            default: return (T)(999);
        }
    }

    sycl_binary_conf_t conf_;

    xpu::sycl::in_memory_arg_t src0_;
    xpu::sycl::in_memory_arg_t src1_;
    xpu::sycl::out_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t src0_scale_;
    xpu::sycl::in_memory_arg_t src1_scale_;
    data_type_t scales_dt_;
    xpu::sycl::in_memory_arg_t po1_src_;
    xpu::sycl::in_memory_arg_t po2_src_;
    xpu::sycl::in_memory_arg_t po3_src_;
    xpu::sycl::in_memory_arg_t po4_src_;
    xpu::sycl::in_memory_arg_t po5_src_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
