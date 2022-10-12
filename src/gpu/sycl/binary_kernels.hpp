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
            sycl_in_memory_arg_t &src1_scale)
        : conf_(conf)
        , src0_(src0)
        , src1_(src1)
        , dst_(dst)
        , src0_scale_(src0_scale)
        , src1_scale_(src1_scale) {}

    [[sycl::reqd_sub_group_size(32)]] void operator()(
            ::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();

        size_t base = ((item.get_group(0) * conf_.wg_size
                               + sg.get_group_id()[0] * sg.get_local_range()[0])
                                      * conf_.block_size
                              + sg.get_local_id() * conf_.block_size)
                / vec_len;

        for (int i = 0; i < conf_.block_size / vec_len; i++) {
            auto src0_vec = load_float_vec<vec_len>(
                    src0_md().data_type(), src0_ptr(), base + i);
            auto src1_vec = load_float_vec<vec_len>(
                    src1_md().data_type(), src1_ptr(), base + i);

            auto dst_vec = load_float_vec<vec_len>(
                    dst_md().data_type(), dst_ptr(), base + i);

            if (conf_.do_scale_src0) src0_vec *= src0_scale_ptr()[0];
            if (conf_.do_scale_src1) src1_vec *= src1_scale_ptr()[0];

            auto acc_vec = compute_alg(src0_vec, src1_vec, conf_.alg_kind);
            // TODO: Adding post-ops seems to be interfering with compiler's
            // optimizations. Figure out how to make the compiler to generate
            // the right code.
            acc_vec = conf_.post_ops.apply(acc_vec, dst_vec);
            store_float_vec(dst_md().data_type(), acc_vec, dst_ptr(), base + i);
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
            case alg_kind::binary_max: return math::max_vec(src0, src1);
            case alg_kind::binary_min: return math::min_vec(src0, src1);
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

    sycl_binary_conf_t conf_;

    sycl_in_memory_arg_t src0_;
    sycl_in_memory_arg_t src1_;
    sycl_out_memory_arg_t dst_;
    sycl_in_memory_arg_t src0_scale_;
    sycl_in_memory_arg_t src1_scale_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
