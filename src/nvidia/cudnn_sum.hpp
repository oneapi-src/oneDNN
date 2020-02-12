/***************************************************************************
 *  Copyright 2020 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 **************************************************************************/

#ifndef CUDNN_SUM_HPP
#define CUDNN_SUM_HPP
#include "gpu/ocl/ref_sum.hpp"
#include "nvidia/sycl_cuda_engine.hpp"
#include "nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace cuda {

struct cudnn_ref_sum_t : public ::dnnl::impl::gpu::ocl::ref_sum_t {

    using base_t = dnnl::impl::gpu::ocl::ref_sum_t;
    using base_t::base_t;
    using base_pd_t = base_t::pd_t;

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        DECLARE_SUM_PD_T("ref:any", cudnn_ref_sum_t);
        // This function can be used for backend that does not support
        // blocking on f32, so it can convert the blocked format to nchw. Since
        // the final destination will preserve the blocking, the last reorder
        // to put the accumulated result to the final output will add the
        // blocking back.
        virtual void define_dst_acc_md() override {
            dst_acc_md_ = dst_md_;
            dst_acc_md_.data_type = dnnl_f32;
            if ((dst_md_.data_type == data_type::s8)
                    && (memory_desc_matches_nchw_vect_c(&dst_md_))) {
                dst_acc_md_.format_desc.blocking.inner_nblks = 0;
                dst_acc_md_.format_desc.blocking.inner_idxs[0] = 0;
                dst_acc_md_.format_desc.blocking.inner_blks[0] = 0;
                dst_acc_md_.format_desc.blocking.strides[dst_acc_md_.ndims - 1]
                        = 1;
                for (int i = dst_acc_md_.ndims - 2; i >= 0; i--) {
                    dst_acc_md_.format_desc.blocking.strides[i]
                            = dst_acc_md_.format_desc.blocking.strides[i + 1]
                            * dst_acc_md_.dims[i + 1];
                }
            }
        }
    };
};

} // namespace cuda
} // namespace impl
} // namespace dnnl

#endif // CUDNN_SUM_HPP
