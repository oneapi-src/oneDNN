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

#ifndef GPU_SYCL_PRELU_KERNELS_HPP
#define GPU_SYCL_PRELU_KERNELS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"

#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_math_utils.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

static constexpr int max_supported_ndims = 5;

struct prelu_fwd_kernel_vec_t {
    static constexpr int vec_len = 8;

    prelu_fwd_kernel_vec_t(const sycl_prelu_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_in_memory_arg_t &weights,
            sycl_out_memory_arg_t &dst)
        : conf_(conf), data_(data), weights_(weights), dst_(dst) {}

    void operator()(::sycl::nd_item<1> item) const {

        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();

        const int mask = conf_.mask;
        const dim_t work_amount = conf_.work_amount;
        size_t nthr = conf_.n_thr;
        if ((dim_t)ithr >= work_amount) return;
        dim_t start {0}, end {0};
        dims_t dims_d, off;

        for (int j = 0; j < max_supported_ndims; j++) {
            off[j] = 0;
            dims_d[j] = (data_md().dims()[j] != 0) ? data_md().dims()[j] : 1;
        }

        balance211(work_amount, nthr, ithr, start, end);
        if (conf_.ndims == 1) {
            utils::nd_iterator_init(start, off[0], dims_d[0]);
        } else if (conf_.ndims == 2) {
            utils::nd_iterator_init(
                    start, off[0], dims_d[0], off[1], dims_d[1]);

        } else if (conf_.ndims == 3) {
            utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2]);
        } else if (conf_.ndims == 4) {
            utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2], off[3], dims_d[3]);
        } else if (conf_.ndims == 5) {
            utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2], off[3], dims_d[3], off[4], dims_d[4]);
        }

        for (dim_t iwork = start; iwork < end; ++iwork) {
            dim_t data_off = offset(data_md(), off);
            ;
            dim_t weight_off = weights_offset(mask, weights_md(), off);
            auto src_val = load_float_value(
                    data_md().data_type(), data_ptr(), data_off);
            auto weights_val = load_float_value(
                    weights_md().data_type(), weights_ptr(), weight_off);
            auto res = math::relu_fwd(src_val, weights_val);
            store_float_value(data_md().data_type(), res, dst_ptr(), data_off);
            if (conf_.ndims == 1) {
                utils::nd_iterator_step(off[0], dims_d[0]);
            }
            if (conf_.ndims == 2) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1]);
            }
            if (conf_.ndims == 3) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                        off[2], dims_d[2]);
            }
            if (conf_.ndims == 4) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                        off[2], dims_d[2], off[3], dims_d[3]);
            }
            if (conf_.ndims == 5) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                        off[2], dims_d[2], off[3], dims_d[3], off[4],
                        dims_d[4]);
            }
        }
    }

private:
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &weights_md() const { return conf_.weights_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *weights_ptr() const { return weights_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    static dim_t offset(const sycl_md_t &mem, dims_t dims) {
        const int ndims = mem.ndims();
        switch (ndims) {
            case 1: return mem.off(dims[0]);
            case 2: return mem.off(dims[0], dims[1]);
            case 3: return mem.off(dims[0], dims[1], dims[2]);
            case 4: return mem.off(dims[0], dims[1], dims[2], dims[3]);
            case 5: return mem.off(dims[0], dims[1], dims[2], dims[3], dims[4]);
            default: return -1;
        }
        return -1;
    }

    static dim_t weights_offset(
            const int mask, const sycl_md_t &mem, dims_t &dims) {
        dims_t dims_w {};
        std::copy(dims, dims + max_supported_ndims, dims_w);
        utils::apply_mask_on_dims(dims_w, mem.ndims(), mask);
        return offset(mem, dims_w);
    }

    sycl_prelu_conf_t conf_;
    sycl_in_memory_arg_t data_;
    sycl_in_memory_arg_t weights_;
    sycl_out_memory_arg_t dst_;
};

struct prelu_bwd_kernel_vec_t {
    static constexpr int vec_len = 8;

    prelu_bwd_kernel_vec_t(const sycl_prelu_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_out_memory_arg_t &diff_data,
            sycl_in_memory_arg_t &weights, sycl_out_memory_arg_t &diff_weights,
            sycl_in_memory_arg_t &diff_dst, sycl_out_memory_arg_t &scratchpad)
        : conf_(conf)
        , data_(data)
        , diff_data_(diff_data)
        , weights_(weights)
        , diff_weights_(diff_weights)
        , diff_dst_(diff_dst)
        , scratchpad_(scratchpad) {}

    void operator()(::sycl::nd_item<1> item) const {
        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        switch (conf_.bcast_type) {
            case broadcasting_strategy_t::scalar:
                calculate_scalar(data_ptr(), weights_ptr(), diff_weights_ptr(),
                        diff_dst_ptr(), diff_data_ptr(), ithr);
                reduce_scalar(diff_weights_ptr(), ithr);
                break;
            case broadcasting_strategy_t::no_broadcast:
                calculate_no_broadcast(data_ptr(), weights_ptr(),
                        diff_weights_ptr(), diff_dst_ptr(), diff_data_ptr(),
                        ithr);
                break;
            case broadcasting_strategy_t::per_oc:
            case broadcasting_strategy_t::per_oc_spatial:
            case broadcasting_strategy_t::per_mb_spatial:
            case broadcasting_strategy_t::per_mb_w:
            case broadcasting_strategy_t::per_w:
            case broadcasting_strategy_t::shared_axes:
                calculate_shared_axes(data_ptr(), weights_ptr(),
                        diff_weights_ptr(), diff_dst_ptr(), diff_data_ptr(),
                        ithr, item);
                break;
            default: return;
        }
    }

    float ker(const float *src, const float *weights, const float *diff_dst,
            float *diff_src, dim_t data_off, dim_t weight_off) const {

        float src_val
                = load_float_value(data_md().data_type(), data_ptr(), data_off);
        float diff_dst_val = load_float_value(
                diff_dst_md().data_type(), diff_dst_ptr(), data_off);
        float weights_val = load_float_value(
                weights_md().data_type(), weights_ptr(), weight_off);

        float diff_src_res = ::dnnl::impl::math::relu_bwd_use_dst(
                diff_dst_val, src_val, weights_val);
        float diff_weight_res = src_val > 0 ? 0 : (diff_dst_val * src_val);
        store_float_value(
                data_md().data_type(), diff_src_res, diff_src, data_off);
        return diff_weight_res;
    }

    void reduce_scalar(float *diff_weights, size_t i) const {
        const size_t nthr = conf_.n_thr;
        const dim_t work_amount = conf_.work_amount_src;
        const int thread_count = nstl::min((dim_t)nthr, work_amount);
        double s = 0;
        int c = 0;
        for (dim_t a = 0; a < thread_count; a++) {
            auto la = load_float_value(
                    weights_md().data_type(), scratchpad_ptr(), i);
            if (!std::isnan(la)) c = c + 1;
        }
        if (c == (thread_count)) {
            for (dim_t j = 0; j < thread_count; j++) {
                auto la = load_float_value(
                        weights_md().data_type(), scratchpad_ptr(), j);
                s = s + la;
            }
            store_float_value(
                    weights_md().data_type(), s, diff_weights_ptr(), 0);
        }
    }

    void set_reduction_buffers(
            const dim_t work_amount, dim_t &group_size, dim_t &buf_size) const {
        float sqrt = std::sqrt(work_amount);
        group_size = std::ceil(sqrt);
        buf_size = std::floor(sqrt);
        if (group_size * buf_size < work_amount) group_size++;
    }

private:
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &weights_md() const { return conf_.weights_md; }
    const sycl_md_t &diff_data_md() const { return conf_.diff_data_md; }
    const sycl_md_t &diff_weights_md() const { return conf_.diff_weights_md; }
    const sycl_md_t &diff_dst_md() const { return conf_.diff_dst_md; }

    float *data_ptr() const { return (float *)(data_.get_pointer()); }
    float *weights_ptr() const { return (float *)(weights_.get_pointer()); }
    float *diff_data_ptr() const { return (float *)(diff_data_.get_pointer()); }
    float *diff_weights_ptr() const {
        return (float *)(diff_weights_.get_pointer());
    }
    float *diff_dst_ptr() const { return (float *)(diff_dst_.get_pointer()); }
    float *scratchpad_ptr() const {
        return (float *)(scratchpad_.get_pointer());
    }

    static dim_t offset(const sycl_md_t &mem, dims_t dims) {
        const int ndims = mem.ndims();
        switch (ndims) {
            case 1: return mem.off(dims[0]);
            case 2: return mem.off(dims[0], dims[1]);
            case 3: return mem.off(dims[0], dims[1], dims[2]);
            case 4: return mem.off(dims[0], dims[1], dims[2], dims[3]);
            case 5: return mem.off(dims[0], dims[1], dims[2], dims[3], dims[4]);
            default: return -1;
        }
        return -1;
    }

    static dim_t weights_offset(
            const int mask, const sycl_md_t &mem, dims_t &dims) {
        dims_t dims_w {};
        std::copy(dims, dims + max_supported_ndims, dims_w);
        utils::apply_mask_on_dims(dims_w, mem.ndims(), mask);
        return offset(mem, dims_w);
    }

    void calculate_scalar(const float *src, const float *weights,
            float *diff_weights, const float *diff_dst, float *diff_src,
            size_t ithr) const {

        const size_t nthr = conf_.n_thr;
        const dim_t work_amount = conf_.work_amount_src;

        if ((dim_t)ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dims_t dims_d, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            off[i] = 0;
            dims_d[i] = (data_md().dims()[i] != 0) ? data_md().dims()[i] : 1;
        }

        balance211(work_amount, nthr, ithr, start, end);
        const dim_t workload = end - start;

        if (conf_.ndims == 1) {
            utils::nd_iterator_init(start, off[0], dims_d[0]);
        }
        if (conf_.ndims == 2) {
            utils::nd_iterator_init(
                    start, off[0], dims_d[0], off[1], dims_d[1]);
        }
        if (conf_.ndims == 3) {
            utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2]);
        }
        if (conf_.ndims == 4) {
            utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2], off[3], dims_d[3]);
        }
        if (conf_.ndims == 5) {
            utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2], off[3], dims_d[3], off[4], dims_d[4]);
        }

        dim_t group_size, buf_size;
        set_reduction_buffers(workload, group_size, buf_size);
        dim_t offset_buf {0}, group_off {0}, data_size {buf_size};
        float s = 0;
        float r = 0;
        for (dim_t iwork = start; iwork < end; ++iwork) {
            const auto data_off = offset(data_md(), off);
            const auto weight_off = 0;
            float src_val = load_float_value(
                    data_md().data_type(), data_ptr(), data_off);
            float diff_dst_val = load_float_value(
                    diff_dst_md().data_type(), diff_dst_ptr(), data_off);
            float weights_val = load_float_value(
                    weights_md().data_type(), weights_ptr(), weight_off);
            float diff_src_res = ::dnnl::impl::math::relu_bwd_use_dst(
                    diff_dst_val, src_val, weights_val);
            float diff_weight_res = src_val > 0 ? 0 : (diff_dst_val * src_val);
            store_float_value(
                    data_md().data_type(), diff_src_res, diff_src, data_off);

            s = s + diff_weight_res;
            if (++offset_buf == data_size) {
                r = r + s;
                offset_buf = 0;
                group_off++;
                s = 0;
                data_size = ((group_off + 1) * buf_size <= workload)
                        ? buf_size
                        : workload - (group_off * buf_size);
            }
            if (conf_.ndims == 1) {
                utils::nd_iterator_step(off[0], dims_d[0]);
            }
            if (conf_.ndims == 2) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1]);
            }
            if (conf_.ndims == 3) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                        off[2], dims_d[2]);
            }
            if (conf_.ndims == 4) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                        off[2], dims_d[2], off[3], dims_d[3]);
            }
            if (conf_.ndims == 5) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                        off[2], dims_d[2], off[3], dims_d[3], off[4],
                        dims_d[4]);
            }
        }
        store_float_value(weights_md().data_type(), r, scratchpad_ptr(), ithr);
    }

    void calculate_no_broadcast(const float *src, const float *weights,
            float *diff_weights, const float *diff_dst, float *diff_src,
            size_t ithr) const {
        const size_t nthr = conf_.n_thr;
        const dim_t work_amount = conf_.work_amount_src;
        const int mask = conf_.mask;

        if ((dim_t)ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dims_t dims_d, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            off[i] = 0;
            dims_d[i] = (data_md().dims()[i] != 0) ? data_md().dims()[i] : 1;
        }

        balance211(work_amount, nthr, ithr, start, end);
        if (conf_.ndims == 1) {
            utils::nd_iterator_init(start, off[0], dims_d[0]);
        }
        if (conf_.ndims == 2) {
            utils::nd_iterator_init(
                    start, off[0], dims_d[0], off[1], dims_d[1]);
        }
        if (conf_.ndims == 3) {
            utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2]);
        }
        if (conf_.ndims == 4) {
            utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2], off[3], dims_d[3]);
        }
        if (conf_.ndims == 5) {
            utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2], off[3], dims_d[3], off[4], dims_d[4]);
        }

        for (dim_t iwork = start; iwork < end; ++iwork) {
            const auto data_off = offset(data_md(), off);
            const auto weight_off = weights_offset(mask, weights_md(), off);
            const auto res = ker(
                    src, weights, diff_dst, diff_src, data_off, weight_off);

            store_float_value(
                    weights_md().data_type(), res, diff_weights, weight_off);
            if (conf_.ndims == 1) {
                utils::nd_iterator_step(off[0], dims_d[0]);
            }
            if (conf_.ndims == 2) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1]);
            }
            if (conf_.ndims == 3) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                        off[2], dims_d[2]);
            }
            if (conf_.ndims == 4) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                        off[2], dims_d[2], off[3], dims_d[3]);
            }
            if (conf_.ndims == 5) {
                utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                        off[2], dims_d[2], off[3], dims_d[3], off[4],
                        dims_d[4]);
            }
        }
    }

    void calculate_shared_axes(const float *src, const float *weights,
            float *diff_weights, const float *diff_dst, float *diff_src,
            size_t ith, ::sycl::nd_item<1> item) const {

        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        dims_t dims_d, dims_w;
        for (int i = 0; i < max_supported_ndims; i++) {
            dims_d[i] = (data_md().dims()[i] != 0) ? data_md().dims()[i] : 1;
            dims_w[i] = (weights_md().dims()[i] != 0) ? weights_md().dims()[i]
                                                      : 1;
        }

        const size_t nthr = conf_.n_thr;
        const dim_t work_amount = conf_.work_amount;
        if ((dim_t)ithr >= work_amount) return;
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        dim_t group_size, buf_size;
        const dim_t workload = conf_.work_load;
        set_reduction_buffers(workload, group_size, buf_size);

        dims_t off_w, off_d, dims_start, dims_end;

        if (conf_.ndims == 1) {
            utils::nd_iterator_init(start, off_w[0], dims_w[0]);
        }
        if (conf_.ndims == 2) {
            utils::nd_iterator_init(
                    start, off_w[0], dims_w[0], off_w[1], dims_w[1]);
        }
        if (conf_.ndims == 3) {
            utils::nd_iterator_init(start, off_w[0], dims_w[0], off_w[1],
                    dims_w[1], off_w[2], dims_w[2]);
        }
        if (conf_.ndims == 4) {
            utils::nd_iterator_init(start, off_w[0], dims_w[0], off_w[1],
                    dims_w[1], off_w[2], dims_w[2], off_w[3], dims_w[3]);
        }
        if (conf_.ndims == 5) {
            utils::nd_iterator_init(start, off_w[0], dims_w[0], off_w[1],
                    dims_w[1], off_w[2], dims_w[2], off_w[3], dims_w[3],
                    off_w[4], dims_w[4]);
        }

        for (dim_t iwork = start; iwork < end; ++iwork) {
            auto weight_off = offset(weights_md(), off_w);

            for (int i = 0; i < max_supported_ndims; i++) {
                dims_start[i] = (dims_d[i] == dims_w[i]) ? off_w[i] : 0;
                dims_end[i]
                        = (dims_d[i] == dims_w[i]) ? off_w[i] + 1 : dims_d[i];
            }
            dim_t buf_off {0}, group_off {0}, data_size {buf_size};

            float res = 0;
            float st = 0;
            for_(off_d[0] = dims_start[0]; off_d[0] < dims_end[0]; ++off_d[0])
            for_(off_d[1] = dims_start[1]; off_d[1] < dims_end[1]; ++off_d[1])
            for_(off_d[2] = dims_start[2]; off_d[2] < dims_end[2]; ++off_d[2])
            for_(off_d[3] = dims_start[3]; off_d[3] < dims_end[3]; ++off_d[3])
            for (off_d[4] = dims_start[4]; off_d[4] < dims_end[4]; ++off_d[4]) {
                auto data_off = offset(data_md(), off_d);
                const auto diff_weight = ker(
                        src, weights, diff_dst, diff_src, data_off, weight_off);
                st = st + diff_weight;
                buf_off = buf_off + 1;
                if (buf_off == data_size) {
                    group_off = group_off + 1;
                    buf_off = 0;
                    data_size = ((group_off + 1) * buf_size <= workload)
                            ? buf_size
                            : workload - (group_off * buf_size);

                    res = res + st;
                    st = 0;
                }
            }

            store_float_value(weights_md().data_type(), res, diff_weights_ptr(),
                    weight_off);
            if (conf_.ndims == 1) {
                utils::nd_iterator_step(off_w[0], dims_w[0]);
            }
            if (conf_.ndims == 2) {
                utils::nd_iterator_step(
                        off_w[0], dims_w[0], off_w[1], dims_w[1]);
            }
            if (conf_.ndims == 3) {
                utils::nd_iterator_step(off_w[0], dims_w[0], off_w[1],
                        dims_w[1], off_w[2], dims_w[2]);
            }
            if (conf_.ndims == 4) {
                utils::nd_iterator_step(off_w[0], dims_w[0], off_w[1],
                        dims_w[1], off_w[2], dims_w[2], off_w[3], dims_w[3]);
            }
            if (conf_.ndims == 5) {
                utils::nd_iterator_step(off_w[0], dims_w[0], off_w[1],
                        dims_w[1], off_w[2], dims_w[2], off_w[3], dims_w[3],
                        off_w[4], dims_w[4]);
            }
        }
    }

    sycl_prelu_conf_t conf_;
    sycl_in_memory_arg_t data_;
    sycl_out_memory_arg_t diff_data_;
    sycl_in_memory_arg_t weights_;
    sycl_out_memory_arg_t diff_weights_;
    sycl_in_memory_arg_t diff_dst_;
    sycl_out_memory_arg_t scratchpad_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
