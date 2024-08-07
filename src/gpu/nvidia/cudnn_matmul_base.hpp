/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_BASE_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_BASE_HPP

#include "gpu/gpu_matmul_pd.hpp"

#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/nvidia/cudnn_matmul_executor.hpp"
#include "gpu/nvidia/cudnn_matmul_impl.hpp"
#include "gpu/nvidia/cudnn_matmul_lt_impl.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_base_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_base_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;
        virtual status_t init(impl::engine_t *engine) = 0;

        // Use scratchpad memory and reorder from it in two scenarios:
        // * Bias dt is different from dst dt.
        // * Dst dt is not f32. cuBLAS only supports s8s8f32.
        bool reorder_required() const {
            return dst_md()->data_type != data_type::f32
                    || (with_bias()
                            && (weights_md(1)->data_type
                                    != dst_md()->data_type));
        }

    protected:
        bool blocking_ok() const {
            std::vector<const memory_desc_t *> mds
                    = {src_md(), dst_md(), weights_md(0)};
            if (with_bias()) mds.push_back(weights_md(1));
            for (const memory_desc_t *md : mds) {
                memory_desc_wrapper mdw(md);
                if (mdw.is_blocking_desc()) {
                    if (mdw.blocking_desc().inner_nblks != 0) { return false; }
                }
            }
            return true;
        }
    };
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
