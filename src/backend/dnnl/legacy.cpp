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

#include <limits>

#include "interface/backend.hpp"
#include "interface/value.hpp"

#include "common.hpp"
#include "dnnl_backend.hpp"
#include "tensor.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

void fill_layout_info(impl::logical_tensor_t *lt, const tensor::desc &td) {
    const impl::logical_tensor_wrapper ltw(lt);
    if (ltw.is_any()) { // we only reset any format
#ifdef DNNL_GRAPH_LAYOUT_DEBUG
        const int ndims = td.data.ndims;
        if (ndims <= 1) { // scratchpads mem
            const dnnl_dims_t &dims = td.data.dims;
            lt->ndims = ndims;
            std::copy(dims, dims + ndims, lt->dims);
            lt->data_type = static_cast<impl::data_type_t>(td.data.data_type);
        }
#endif // DNNL_GRAPH_LAYOUT_DEBUG

        impl::utils::optional<size_t> layout_id
                = dnnl_backend::get_singleton().set_mem_desc(td);
        lt->layout.layout_id = static_cast<int64_t>(layout_id.value());
        lt->layout_type = impl::layout_type::opaque;
    }
}

void fill_layout_info(
        std::shared_ptr<impl::value_t> &val, const tensor::desc &td) {
    impl::logical_tensor_t lt = val->get_logical_tensor();
    const impl::logical_tensor_wrapper ltw(lt);
    if (ltw.is_any()) { // we only reset any format
        val->set_layout_id(static_cast<int64_t>(
                dnnl_backend::get_singleton().set_mem_desc(td).value()));
    }
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
