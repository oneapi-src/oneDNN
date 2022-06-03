/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef DNN_GRAPH_TYPES_HPP
#define DNN_GRAPH_TYPES_HPP

#include <algorithm>
#include <vector>

#include <oneapi/dnnl/dnnl_graph.hpp>

namespace benchdnnext {

using graph_dt = dnnl::graph::logical_tensor::data_type;

inline bool is_low_precision(const std::vector<graph_dt> &dtypes) {
    return std::find_if(dtypes.begin(), dtypes.end(), [](graph_dt dt) {
        return dt == graph_dt::u8 || dt == graph_dt::s8;
    }) != dtypes.end();
}

inline bool with_typecast(const std::vector<graph_dt> &dtypes) {
    return std::find_if(dtypes.begin(), dtypes.end(),
                   [](graph_dt dt) { return dt == graph_dt::bf16; })
            != dtypes.end()
            && is_low_precision(dtypes);
}

inline graph_dt set_main_op_dtype(graph_dt dtype) {
    return is_low_precision({dtype}) ? graph_dt::f32 : dtype;
}

inline graph_dt dequantize_dtype(graph_dt dtype) {
    return is_low_precision({dtype}) ? graph_dt::f32 : dtype;
}

} // namespace benchdnnext

#endif // DNN_GRAPH_TYPES_HPP
