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

#include "nvidia/cudnn_deconvolution.hpp"
#include "nvidia/sycl_cuda_scoped_context.hpp"
#include "nvidia/sycl_cuda_stream.hpp"
#include "nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace cuda {
status_t cudnn_deconvolution_bwd_weights_t::execute_bias(
        const exec_ctx_t &ctx) const {
    memory_desc_wrapper wrap(pd()->diff_dst_md(0));
    if (wrap.size() == 0) { return status::success; }

    cuda::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<cuda::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        auto bias_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS);
        auto y_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);

        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            auto bias = sc.memory<void *>(ih, bias_acc);
            auto y = sc.memory<void *>(ih, y_acc);

            pd()->impl_->execute_bias(handle, y, bias);
        });
    });
}

} // namespace cuda
} // namespace impl
} // namespace dnnl
