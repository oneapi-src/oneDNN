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

#ifndef LLGA_BACKEND_DNNL_OPERATORS_REORDER_HPP
#define LLGA_BACKEND_DNNL_OPERATORS_REORDER_HPP

#include <vector>

#include "backend/dnnl/tensor.hpp"
#include "common.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace {
enum reorder_input { kSrc };
enum reorder_output { kDst };
} // namespace

struct reorder : public dnnl::reorder, public kernel_base {
    using super = dnnl::reorder;

private:
    primitive_desc pd_;
    dnnl::engine eng_;

public:
    impl::status_t compile_impl(const node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(anode);
        using desc = tensor::desc;

        const desc src {inputs.at(reorder_input::kSrc)};
        const desc dst {outputs.at(reorder_output::kDst)};

        eng_ = make_dnnl_engine(*aengine);
        // TODO(wuxun): consider reorder between different engines
        pd_ = primitive_desc(
                /*src_engine=*/eng_, src, /*dst_engine=*/eng_, dst);
        return status::success;
    }

    impl::status_t execute_impl(const node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        UNUSED(anode);
        impl::allocator_t *alc = astream->get_engine()->get_allocator();
        tensor src_ts {inputs.at(reorder_input::kSrc), eng_, alc};
        tensor dst_ts {outputs.at(reorder_output::kDst), eng_, alc};

        dnnl::stream s(eng_);
        super(pd_).execute(s, src_ts, dst_ts);
        s.wait();
        return status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
