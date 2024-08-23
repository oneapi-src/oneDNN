/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_PRELU_KERNELS_HPP
#define GPU_GENERIC_SYCL_PRELU_KERNELS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

static constexpr int max_supported_ndims = 5;

struct prelu_fwd_kernel_vec_t {
    static constexpr int vec_len = 8;

    prelu_fwd_kernel_vec_t(const sycl_prelu_conf_t &conf, ::sycl::handler &cgh,
            const exec_ctx_t &ctx)
        : conf_(conf)
        , data_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , weights_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WEIGHTS))
        , dst_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST)) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t data_mem(data_, conf_.data_md);
        memory_tensor_t weights_mem(weights_, conf_.weights_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);

        size_t ithr = item.get_global_id(0);

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
            dim_t weight_off = weights_offset(mask, weights_md(), off);
            auto src_val = data_mem.load(data_off);
            auto weights_val = weights_mem.load(weight_off);
            auto res = math::relu_fwd(src_val, weights_val);
            dst_mem.store(res, data_off);
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
    const xpu::sycl::md_t &data_md() const { return conf_.data_md; }
    const xpu::sycl::md_t &weights_md() const { return conf_.weights_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }

    static dim_t offset(const xpu::sycl::md_t &mem, dims_t dims) {
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
            const int mask, const xpu::sycl::md_t &mem, dims_t &dims) {
        dims_t dims_w {};
        std::copy(dims, dims + max_supported_ndims, dims_w);
        utils::apply_mask_on_dims(dims_w, mem.ndims(), mask);
        return offset(mem, dims_w);
    }

    sycl_prelu_conf_t conf_;
    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::in_memory_arg_t weights_;
    xpu::sycl::out_memory_arg_t dst_;
};

struct prelu_bwd_kernel_vec_t {
    static constexpr int vec_len = 8;

    prelu_bwd_kernel_vec_t(const sycl_prelu_conf_t &conf, ::sycl::handler &cgh,
            const exec_ctx_t &ctx, bool reduce_diff_weights,
            std::unique_ptr<memory_t, memory_deleter_t> &scratch_mem)
        : conf_(conf)
        , data_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , diff_data_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC))
        , weights_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WEIGHTS))
        , diff_weights_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_WEIGHTS))
        , diff_dst_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST))
        , scratchpad_(reduce_diff_weights
                          ? utils::downcast<
                                  const xpu::sycl::memory_storage_base_t *>(
                                  scratch_mem->memory_storage())
                                    ->get_out_memory_arg(ctx.stream(), cgh)
                          : xpu::sycl::memory_storage_base_t::
                                  empty_out_memory_arg(ctx.stream(), cgh)) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t data_mem(data_, conf_.data_md);
        memory_tensor_t diff_data_mem(diff_data_, conf_.diff_data_md);
        memory_tensor_t weights_mem(weights_, conf_.weights_md);
        memory_tensor_t diff_weights_mem(diff_weights_, conf_.diff_weights_md);
        memory_tensor_t diff_dst_mem(diff_dst_, conf_.diff_dst_md);
        memory_plain_t scratchpad_mem(
                scratchpad_, conf_.weights_md.data_type());

        size_t ithr = item.get_global_id(0);
        switch (conf_.bcast_type) {
            case broadcasting_strategy_t::scalar:
                calculate_scalar(data_mem, weights_mem, scratchpad_mem,
                        diff_dst_mem, diff_data_mem, ithr);
                break;
            case broadcasting_strategy_t::no_broadcast:
                calculate_no_broadcast(data_mem, weights_mem, diff_weights_mem,
                        diff_dst_mem, diff_data_mem, ithr);
                break;
            default:
                calculate_shared_axes(data_mem, weights_mem, diff_weights_mem,
                        diff_dst_mem, diff_data_mem, ithr, item);
                break;
        }
    }

    float ker(const in_memory_tensor_t &data_mem,
            const in_memory_tensor_t &weights_mem,
            const in_memory_tensor_t &diff_dst_mem,
            out_memory_tensor_t &diff_src_mem, dim_t data_off,
            dim_t weight_off) const {

        float src_val = data_mem.load(data_off);
        float diff_dst_val = diff_dst_mem.load(data_off);
        float weights_val = weights_mem.load(weight_off);

        float diff_src_res = ::dnnl::impl::math::relu_bwd_use_dst(
                diff_dst_val, src_val, weights_val);
        float diff_weight_res = src_val > 0 ? 0 : (diff_dst_val * src_val);
        diff_src_mem.store(diff_src_res, data_off);
        return diff_weight_res;
    }

    void set_reduction_buffers(
            const dim_t work_amount, dim_t &group_size, dim_t &buf_size) const {
        float sqrt = std::sqrt(work_amount);
        group_size = std::ceil(sqrt);
        buf_size = std::floor(sqrt);
        if (group_size * buf_size < work_amount) group_size++;
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.data_md; }
    const xpu::sycl::md_t &weights_md() const { return conf_.weights_md; }
    const xpu::sycl::md_t &diff_data_md() const { return conf_.diff_data_md; }
    const xpu::sycl::md_t &diff_weights_md() const {
        return conf_.diff_weights_md;
    }
    const xpu::sycl::md_t &diff_dst_md() const { return conf_.diff_dst_md; }

    static dim_t offset(const xpu::sycl::md_t &mem, dims_t dims) {
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
            const int mask, const xpu::sycl::md_t &mem, dims_t &dims) {
        dims_t dims_w {};
        std::copy(dims, dims + max_supported_ndims, dims_w);
        utils::apply_mask_on_dims(dims_w, mem.ndims(), mask);
        return offset(mem, dims_w);
    }

    void calculate_scalar(const in_memory_tensor_t &data_mem,
            const in_memory_tensor_t &weights_mem,
            out_memory_plain_t &scratchpad_mem,
            const in_memory_tensor_t &diff_dst_mem,
            out_memory_tensor_t &diff_src_mem, size_t ithr) const {

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
            const auto weight_off = 0;
            float src_val = data_mem.load(data_off);
            float diff_dst_val = diff_dst_mem.load(data_off);
            float weights_val = weights_mem.load(weight_off);
            float diff_src_res = ::dnnl::impl::math::relu_bwd_use_dst(
                    diff_dst_val, src_val, weights_val);
            float diff_weight_res = src_val > 0 ? 0 : (diff_dst_val * src_val);
            diff_src_mem.store(diff_src_res, data_off);
            scratchpad_mem.store(diff_weight_res, data_off);

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

    void calculate_no_broadcast(const in_memory_tensor_t &data_mem,
            const in_memory_tensor_t &weights_mem,
            out_memory_tensor_t &diff_weights_mem,
            const in_memory_tensor_t &diff_dst_mem,
            out_memory_tensor_t &diff_src_mem, size_t ithr) const {
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
            const auto res = ker(data_mem, weights_mem, diff_dst_mem,
                    diff_src_mem, data_off, weight_off);

            diff_weights_mem.store(res, weight_off);
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

    void calculate_shared_axes(const in_memory_tensor_t &data_mem,
            const in_memory_tensor_t &weights_mem,
            out_memory_tensor_t &diff_weights_mem,
            const in_memory_tensor_t &diff_dst_mem,
            out_memory_tensor_t &diff_src_mem, size_t ith,
            ::sycl::nd_item<1> item) const {

        size_t ithr = item.get_global_id(0);
        dims_t dims_d, dims_w;
        for (int i = 0; i < max_supported_ndims; i++) {
            dim_t data_dim_i = data_md().dims()[i];
            dim_t data_ndims = data_md().ndims();
            dims_d[i] = (data_dim_i > 0 && i < data_ndims) ? data_dim_i : 1;
            dim_t weights_dim_i = weights_md().dims()[i];
            dim_t weights_ndims = weights_md().ndims();
            dims_w[i] = (weights_dim_i > 0 && i < weights_ndims) ? weights_dim_i
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
                const auto diff_weight = ker(data_mem, weights_mem,
                        diff_dst_mem, diff_src_mem, data_off, weight_off);
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

            diff_weights_mem.store(res, weight_off);
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
    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::out_memory_arg_t diff_data_;
    xpu::sycl::in_memory_arg_t weights_;
    xpu::sycl::out_memory_arg_t diff_weights_;
    xpu::sycl::in_memory_arg_t diff_dst_;
    xpu::sycl::out_memory_arg_t scratchpad_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
