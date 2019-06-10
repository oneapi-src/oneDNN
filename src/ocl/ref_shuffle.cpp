/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "ocl/ref_shuffle.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace format_tag;

template <int data_type_size>
template <mkldnn_format_tag_t tag>
status_t ref_shuffle_t<data_type_size>::execute_(const exec_ctx_t &ctx) const {
    auto &src = pd()->is_fwd()
                    ? CTX_IN_STORAGE(MKLDNN_ARG_SRC)
                    : CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &dst = pd()->is_fwd()
                    ? CTX_OUT_STORAGE(MKLDNN_ARG_DST)
                    : CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC);

    const auto &jshfl = pd()->jshfl_;

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, dst);

    auto nd_range = cl_nd_range_t(jshfl.gws_d);
    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}
template status_t ref_shuffle_t<4>::execute_<any>(const exec_ctx_t &ctx) const;
template status_t ref_shuffle_t<2>::execute_<any>(const exec_ctx_t &ctx) const;
template status_t ref_shuffle_t<1>::execute_<any>(const exec_ctx_t &ctx) const;

} // namespace ocl
} // namespace impl
} // namespace mkldnn
