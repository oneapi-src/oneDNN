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

#ifndef GPU_SYCL_SOFTMAX_KERNELS_HPP
#define GPU_SYCL_SOFTMAX_KERNELS_HPP

#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct softmax_fwd_kernel_vec_t {
    softmax_fwd_kernel_vec_t(const sycl_softmax_conf_t &conf,
            sycl_in_memory_arg_t &src, sycl_in_memory_arg_t &scale_src,
            sycl_in_memory_arg_t &scale_dst, sycl_out_memory_arg_t &dst)
        : conf_(conf)
        , src_(src)
        , scale_src_(scale_src)
        , scale_dst_(scale_dst)
        , dst_(dst) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;
        size_t base_idx = offset_t * conf_.block_size;

        auto operation = [=](dim_t &ou, dim_t &in) {
            float space_denom = 0;
            float space_max = -FLT_MAX;
            dim_t ou_in_offset = ou * conf_.channels * conf_.inner_size + in;

            for (dim_t c = 0; c < conf_.channels; c++) {
                size_t off
                        = src_md().off_l(ou_in_offset + c * conf_.inner_size);
                float s = load_float_value(
                        src_md().data_type(), src_ptr(), off);
                space_max = nstl::max(space_max, s);
            }
            for (dim_t c = 0; c < conf_.channels; c++) {
                size_t src_off
                        = src_md().off_l(ou_in_offset + c * conf_.inner_size);
                float s = load_float_value(
                        src_md().data_type(), src_ptr(), src_off);
                float d = s - space_max;
                space_denom += ::sycl::exp((float)d);
            }
            if (conf_.alg_kind == alg_kind::softmax_log) {
                space_denom = ::sycl::log((float)space_denom);
            }
            for (dim_t c = 0; c < conf_.channels; c++) {
                size_t src_off
                        = src_md().off_l(ou_in_offset + c * conf_.inner_size);
                float s = load_float_value(
                        src_md().data_type(), src_ptr(), src_off);
                float d = s - space_max;
                if (conf_.alg_kind == alg_kind::softmax_accurate) {
                    d = ::sycl::exp((float)d);
                }
                float sd = space_denom;
                if (conf_.alg_kind == alg_kind::softmax_accurate) {
                    d /= sd;
                } else if (conf_.alg_kind == alg_kind::softmax_log) {
                    d -= sd;
                }

                float scale = 1.0f;
                scale = conf_.do_scale_src
                        ? load_float_value(data_type::f32, scale_src_ptr(), 0)
                        : scale;
                scale /= conf_.do_scale_dst
                        ? load_float_value(data_type::f32, scale_dst_ptr(), 0)
                        : 1.0f;

                d = (d * scale);
                size_t dst_off
                        = dst_md().off_l(ou_in_offset + c * conf_.inner_size);
                store_float_value(dst_md().data_type(), d, dst_ptr(), dst_off);
            }
        };

        for (dim_t blk_idx = 0; blk_idx < conf_.block_size; blk_idx++) {
            dim_t idx = base_idx + blk_idx;

            if (idx < conf_.wk_size) {
                dim_t in = (idx / (1)) % conf_.inner_size;
                dim_t ou = (idx / (conf_.inner_size)) % conf_.outer_size;
                operation(ou, in);
            }
        }
    }

private:
    const sycl_md_t &src_md() const { return conf_.src_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }

    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *scale_src_ptr() const { return scale_src_.get_pointer(); }
    void *scale_dst_ptr() const { return scale_dst_.get_pointer(); }

    sycl_softmax_conf_t conf_;
    sycl_in_memory_arg_t src_;
    sycl_in_memory_arg_t scale_src_;
    sycl_in_memory_arg_t scale_dst_;
    sycl_out_memory_arg_t dst_;
};

struct softmax_bwd_kernel_vec_t {
    softmax_bwd_kernel_vec_t(const sycl_softmax_conf_t &conf,
            sycl_in_memory_arg_t &dst, sycl_in_memory_arg_t &diff_dst,
            sycl_out_memory_arg_t &diff_src)
        : conf_(conf), dst_(dst), diff_dst_(diff_dst), diff_src_(diff_src) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;
        size_t base_idx = offset_t * conf_.block_size;

        auto operation = [=](dim_t &ou, dim_t &in) {
            dim_t ou_in_offset = ou * conf_.channels * conf_.inner_size + in;
            float sbr = 0;
            for (dim_t c = 0; c < conf_.channels; ++c) {
                auto diff_dst_off = diff_dst_md().off_l(
                        ou_in_offset + c * conf_.inner_size);
                float dd = load_float_value(diff_dst_md().data_type(),
                        diff_dst_ptr(), diff_dst_off);
                if (conf_.alg_kind == alg_kind::softmax_accurate) {
                    auto dst_off = dst_md().off_l(
                            ou_in_offset + c * conf_.inner_size);
                    float d = load_float_value(
                            dst_md().data_type(), dst_ptr(), dst_off);
                    sbr += dd * d;
                } else if (conf_.alg_kind == alg_kind::softmax_log) {
                    sbr += dd;
                }
            }

            for (dim_t c = 0; c < conf_.channels; ++c) {
                auto diff_dst_off = diff_dst_md().off_l(
                        ou_in_offset + c * conf_.inner_size);
                auto dst_off
                        = dst_md().off_l(ou_in_offset + c * conf_.inner_size);

                float d = load_float_value(
                        dst_md().data_type(), dst_ptr(), dst_off);
                float dd = load_float_value(diff_dst_md().data_type(),
                        diff_dst_ptr(), diff_dst_off);

                float val = 0;
                if (conf_.alg_kind == alg_kind::softmax_accurate) {
                    val = d * (dd - sbr);
                } else if (conf_.alg_kind == alg_kind::softmax_log) {
                    val = dd - ::sycl::exp(d) * sbr;
                }

                auto diff_src_off = diff_src_md().off_l(
                        ou_in_offset + c * conf_.inner_size);
                store_float_value(diff_src_md().data_type(), val,
                        diff_src_ptr(), diff_src_off);
            }
        };

        for (dim_t i = 0; i < conf_.block_size; i++) {
            dim_t idx = base_idx + i;
            if (idx < conf_.wk_size) {
                dim_t in = (idx / 1) % conf_.inner_size;
                dim_t ou = (idx / conf_.inner_size) % conf_.outer_size;
                operation(ou, in);
            }
        }
    }

private:
    const sycl_md_t &dst_md() const { return conf_.dst_md; }
    const sycl_md_t &diff_dst_md() const { return conf_.diff_dst_md; }
    const sycl_md_t &diff_src_md() const { return conf_.diff_src_md; }

    void *dst_ptr() const { return dst_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }
    void *diff_src_ptr() const { return diff_src_.get_pointer(); }

    sycl_softmax_conf_t conf_;
    sycl_in_memory_arg_t dst_;
    sycl_in_memory_arg_t diff_dst_;
    sycl_out_memory_arg_t diff_src_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
