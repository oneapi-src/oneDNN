/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_FLEX_REWRITE_HPP
#define BENCHDNN_GRAPH_FLEX_REWRITE_HPP

#include <map>
#include <string>

#include "deserialize.hpp"

namespace graph {

struct flex_rewrite {
    flex_rewrite(const std::map<size_t, std::string> in_shapes,
            const std::map<size_t, std::string> op_attrs, const int64_t mb)
        : in_shapes_(in_shapes), op_attrs_(op_attrs), mb_(mb) {}

    void rewrite(deserialized_graph &dgraph);

private:
    // input shape info from CML
    std::map<size_t, std::string> in_shapes_;
    // input attributes from CML
    std::map<size_t, std::string> op_attrs_;
    int64_t mb_;

    void split_ncx(const std::string &data_format, dims_t &in, int64_t &n,
            int64_t &c, dims_t &x) const;
    void merge_ncx(const std::string &data_format, dims_t &out, int64_t n,
            int64_t c, const dims_t &x) const;
    void split_oix(const std::string &data_format, dims_t &in, dims_t &oi,
            dims_t &x) const;
    void broadcast(const dims_t &x, const dims_t &y, dims_t &z) const;
    // Returns `pad_begin` and `pad_end` for each dimension.
    void cal_pads(dims_t &pads_begin, dims_t &pads_end,
            const deserialized_op &aop, const dims_t &spatial_dims,
            const dims_t &strides, const dims_t &kernel, bool deconv) const;
    void infer_output_shape(deserialized_graph &dgraph, bool change_stride);
    void inports_shape_rewrite(deserialized_graph &dgraph, bool &change_stride);
    bool get_inport_shape_stride(const std::string &in_shape,
            std::string &shape, std::string &stride, std::string &msg);
    void op_attrs_rewrite(deserialized_graph &dgraph);
    void quantized_graph_rewrite(deserialized_graph &dgraph);
    void update_output_info(deserialized_op &aop, deserialized_graph &dgraph,
            bool change_stride);
};

} // namespace graph

#endif
