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

#ifndef GPU_SYCL_BINARY_KERNELS_HPP
#define GPU_SYCL_BINARY_KERNELS_HPP

#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct binary_kernel_vec_t {
    static constexpr int vec_len = 8;

    binary_kernel_vec_t(const sycl_binary_conf_t &conf,
            sycl_in_memory_arg_t &src0, sycl_in_memory_arg_t &src1,
            sycl_out_memory_arg_t &dst, sycl_in_memory_arg_t &src0_scale,
            sycl_in_memory_arg_t &src1_scale, data_type_t scales_dt)
        : conf_(conf)
        , src0_(src0)
        , src1_(src1)
        , dst_(dst)
        , src0_scale_(src0_scale)
        , src1_scale_(src1_scale)
        , scales_dt_(scales_dt) {}

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

        if (sg_base_idx + (sg.get_local_range()[0] * conf_.block_size)
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
                    auto src0 = load_float_value(
                            src0_md().data_type(), src0_ptr(), idx);
                    auto src1 = load_float_value(
                            src1_md().data_type(), src1_ptr(), idx);
                    auto dst = load_float_value(
                            dst_md().data_type(), dst_ptr(), idx);

                    if (conf_.do_scale_src0) src0 *= sm_0;
                    if (conf_.do_scale_src1) src1 *= sm_1;

                    auto acc = compute_alg_n(src0, src1, conf_.alg_kind);
                    acc = conf_.post_ops.apply(acc, dst);
                    store_float_value(
                            dst_md().data_type(), acc, dst_ptr(), idx);
                }
            }
        }
    }

private:
    const sycl_md_t &src0_md() const { return conf_.src0_md; }
    const sycl_md_t &src1_md() const { return conf_.src1_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }

    void *src0_ptr() const { return src0_.get_pointer(); }
    void *src1_ptr() const { return src1_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    float *src0_scale_ptr() const {
        return static_cast<float *>(src0_scale_.get_pointer());
    }
    float *src1_scale_ptr() const {
        return static_cast<float *>(src1_scale_.get_pointer());
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

    sycl_in_memory_arg_t src0_;
    sycl_in_memory_arg_t src1_;
    sycl_out_memory_arg_t dst_;
    sycl_in_memory_arg_t src0_scale_;
    sycl_in_memory_arg_t src1_scale_;
    data_type_t scales_dt_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
