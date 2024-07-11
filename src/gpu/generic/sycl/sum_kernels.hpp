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

#ifndef GPU_SYCL_SUM_KERNELS_HPP
#define GPU_SYCL_SUM_KERNELS_HPP

#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct sum_kernel_vec_t {
    static constexpr int vec_len = 8;
    static constexpr int max_supported_ndims = 6;

    sum_kernel_vec_t(const sycl_sum_conf_t &conf,
            xpu::sycl::in_memory_arg_t &src0, xpu::sycl::in_memory_arg_t &src1,
            xpu::sycl::in_memory_arg_t &src2, xpu::sycl::in_memory_arg_t &src3,
            xpu::sycl::in_memory_arg_t &src4, xpu::sycl::in_memory_arg_t &src5,
            xpu::sycl::in_memory_arg_t &src6, xpu::sycl::in_memory_arg_t &src7,
            xpu::sycl::in_memory_arg_t &src8, xpu::sycl::in_memory_arg_t &src9,
            xpu::sycl::in_memory_arg_t &src10,
            xpu::sycl::in_memory_arg_t &src11,
            xpu::sycl::in_memory_arg_t &src12,
            xpu::sycl::in_memory_arg_t &src13,
            xpu::sycl::in_memory_arg_t &src14,
            xpu::sycl::in_memory_arg_t &src15, xpu::sycl::out_memory_arg_t &dst)
        : conf_(conf)
        , src0_(src0)
        , src1_(src1)
        , src2_(src2)
        , src3_(src3)
        , src4_(src4)
        , src5_(src5)
        , src6_(src6)
        , src7_(src7)
        , src8_(src8)
        , src9_(src9)
        , src10_(src10)
        , src11_(src11)
        , src12_(src12)
        , src13_(src13)
        , src14_(src14)
        , src15_(src15)
        , dst_(dst) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;

        size_t base_idx = offset_t * conf_.block_size;

        dims_t dims, off, strides;
        for (int i = 0; i < max_supported_ndims; i++) {
            dims[i] = (i < conf_.dst_md.ndims()) ? conf_.dst_md.dims()[i] : 1;
            strides[i] = (i < conf_.dst_md.ndims()) ? conf_.dst_md.strides()[i]
                                                    : INT_MAX;
        }

        for (int i = 0; i < conf_.block_size; i++) {
            int idx = base_idx + i;
            if (idx < conf_.wk_size) {
                for (int i = 0; i < max_supported_ndims; i++) {
                    off[i] = idx / strides[i] % dims[i];
                }
                auto result = load_float_val(src0_ptr(), conf_.src_md[0], off);

#define ONEDNN_SYCL_SUM_ADD_ARG(ARG_N) \
    if (conf_.n > ARG_N) \
        result += conf_.src_scales[ARG_N] \
                * load_float_val( \
                        src##ARG_N##_ptr(), conf_.src_md[ARG_N], off);

                ONEDNN_SYCL_SUM_ADD_ARG(1)
                ONEDNN_SYCL_SUM_ADD_ARG(2)
                ONEDNN_SYCL_SUM_ADD_ARG(3)
                ONEDNN_SYCL_SUM_ADD_ARG(4)
                ONEDNN_SYCL_SUM_ADD_ARG(5)
                ONEDNN_SYCL_SUM_ADD_ARG(6)
                ONEDNN_SYCL_SUM_ADD_ARG(7)
                ONEDNN_SYCL_SUM_ADD_ARG(8)
                ONEDNN_SYCL_SUM_ADD_ARG(9)
                ONEDNN_SYCL_SUM_ADD_ARG(11)
                ONEDNN_SYCL_SUM_ADD_ARG(11)
                ONEDNN_SYCL_SUM_ADD_ARG(12)
                ONEDNN_SYCL_SUM_ADD_ARG(13)
                ONEDNN_SYCL_SUM_ADD_ARG(14)
                ONEDNN_SYCL_SUM_ADD_ARG(15)
#undef ONEDNN_SYCL_SUM_ADD_ARG

                store_float_value(
                        conf_.dst_md.data_type(), result, dst_ptr(), idx);
            }
        }
    }

private:
    float load_float_val(
            const void *ptr, xpu::sycl::md_t md, dims_t off) const {
        return load_float_value(md.data_type(), ptr, md.off_v(off));
    }

    void *src0_ptr() const { return src0_.get_pointer(); }
    void *src1_ptr() const { return src1_.get_pointer(); }
    void *src2_ptr() const { return src2_.get_pointer(); }
    void *src3_ptr() const { return src3_.get_pointer(); }
    void *src4_ptr() const { return src4_.get_pointer(); }
    void *src5_ptr() const { return src5_.get_pointer(); }
    void *src6_ptr() const { return src6_.get_pointer(); }
    void *src7_ptr() const { return src7_.get_pointer(); }
    void *src8_ptr() const { return src8_.get_pointer(); }
    void *src9_ptr() const { return src9_.get_pointer(); }
    void *src10_ptr() const { return src10_.get_pointer(); }
    void *src11_ptr() const { return src11_.get_pointer(); }
    void *src12_ptr() const { return src12_.get_pointer(); }
    void *src13_ptr() const { return src13_.get_pointer(); }
    void *src14_ptr() const { return src14_.get_pointer(); }
    void *src15_ptr() const { return src15_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    sycl_sum_conf_t conf_;

    xpu::sycl::in_memory_arg_t src0_;
    xpu::sycl::in_memory_arg_t src1_;
    xpu::sycl::in_memory_arg_t src2_;
    xpu::sycl::in_memory_arg_t src3_;
    xpu::sycl::in_memory_arg_t src4_;
    xpu::sycl::in_memory_arg_t src5_;
    xpu::sycl::in_memory_arg_t src6_;
    xpu::sycl::in_memory_arg_t src7_;
    xpu::sycl::in_memory_arg_t src8_;
    xpu::sycl::in_memory_arg_t src9_;
    xpu::sycl::in_memory_arg_t src10_;
    xpu::sycl::in_memory_arg_t src11_;
    xpu::sycl::in_memory_arg_t src12_;
    xpu::sycl::in_memory_arg_t src13_;
    xpu::sycl::in_memory_arg_t src14_;
    xpu::sycl::in_memory_arg_t src15_;
    xpu::sycl::out_memory_arg_t dst_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
