/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_DNNL_OPERATORS_SUM_HPP
#define BACKEND_DNNL_OPERATORS_SUM_HPP

#include <vector>

#include "backend/dnnl/tensor.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

struct sum : public dnnl::sum {
    using super = dnnl::sum;

    static void compute(const scale_t &scales, const std::vector<tensor> &srcs,
            tensor &dst, const dnnl::engine &p_engine, impl::allocator_t *alc,
            const dnnl::stream &p_stream) {
        UNUSED(alc);
        auto src_descs = utils::fmap(srcs, [](const tensor &t) {
            // "upcast" vector<tensor::desc> to vector<memory::desc>
            return static_cast<memory::desc>(t.get_desc());
        });
        auto pd = primitive_desc(scales, src_descs, p_engine);

        dst.reinit_if_possible(p_stream, pd.dst_desc());

        exec_args args {{DNNL_ARG_DST, dst}};
        for (int i = 0; i < srcs.size(); ++i) {
            args.insert(
                    {DNNL_ARG_MULTIPLE_SRC + i, srcs[static_cast<size_t>(i)]});
        }
        super(pd).execute(p_stream, args);
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
