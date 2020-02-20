/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

namespace dnnl {
namespace impl {
namespace ocl {

using namespace format_tag;

template <dnnl_format_tag_t tag>
status_t ref_shuffle_t::execute_(const exec_ctx_t &ctx) const {
    auto &src = pd()->is_fwd() ? CTX_IN_STORAGE(DNNL_ARG_SRC)
                               : CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &dst = pd()->is_fwd() ? CTX_OUT_STORAGE(DNNL_ARG_DST)
                               : CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    const auto &jshfl = pd()->jshfl_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);

    auto nd_range = compute::nd_range_t(jshfl.gws_d);
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}
template status_t ref_shuffle_t::execute_<any>(const exec_ctx_t &ctx) const;

} // namespace ocl
} // namespace impl
} // namespace dnnl
