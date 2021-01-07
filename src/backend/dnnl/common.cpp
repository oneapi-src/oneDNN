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

#include "backend/dnnl/backend.hpp"
#include "backend/dnnl/common.hpp"
#include "interface/backend.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

tensor::desc tensor::desc::get_tensor_desc(const impl::logical_tensor_t &lt) {
    const impl::logical_tensor_wrapper ltw(lt);
    const tensor::dims dims = ltw.vdims();
    const auto dtype = static_cast<tensor::desc::data_type>(ltw.data_type());

    if (ltw.is_any()) {
        auto fmtag = tensor::desc::format_tag::any;
        return tensor::desc {dims, dtype, fmtag};
    } else if (ltw.is_strided()) {
        const tensor::dims strides = ltw.vstrides();
        return tensor::desc {dims, dtype, strides};
    } else {
        const int64_t layout_id = ltw.layout_id();
        auto backend = backend_manager::get_backend("dnnl");
        const impl::utils::optional<impl::utils::any> &td
                = std::dynamic_pointer_cast<dnnl_backend>(backend)
                          ->get_mem_desc(static_cast<size_t>(layout_id));
        return impl::utils::any_cast<tensor::desc>(td.value());
    }
}

void fill_layout_info(impl::logical_tensor_t *lt, const tensor::desc &td) {
    const impl::logical_tensor_wrapper ltw(lt);
    if (ltw.is_any()) { // we only reset any format
        auto backend = backend_manager::get_backend("dnnl");
        impl::utils::optional<size_t> layout_id
                = std::dynamic_pointer_cast<dnnl_backend>(backend)
                          ->set_mem_desc(td);
        lt->layout.layout_id = static_cast<int64_t>(layout_id.value());
        lt->layout_type = impl::layout_type::opaque;
    }
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
