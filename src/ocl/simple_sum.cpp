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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/mkldnn_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "ocl/cl_stream.hpp"

#include "ocl/simple_sum.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <data_type_t data_type>
status_t simple_sum_t<data_type>::execute(const exec_ctx_t &ctx) const {
    auto &output = CTX_OUT_STORAGE(MKLDNN_ARG_DST);

    const int num_arrs = pd()->n_inputs();
    const memory_desc_wrapper o_d(pd()->dst_md());
    const size_t nelems = o_d.nelems();

    for (int a = 0; a < num_arrs; ++a) {

        auto &input = CTX_IN_STORAGE(MKLDNN_ARG_MULTIPLE_SRC + a);
        const float scale = pd()->scales()[a];

        kernel_.set_arg(0, input);
        kernel_.set_arg(1, output);
        kernel_.set_arg(2, scale);
        kernel_.set_arg(3, a);

        auto &executor = *(
                utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

        auto nd_range = cl_nd_range_t({ nelems });
        status_t status = executor.parallel_for(nd_range, kernel_);
        if (status != status::success)
            return status;
    }
    return status::success;
}

template struct simple_sum_t<data_type::f32>;

} // namespace ocl
} // namespace impl
} // namespace mkldnn
