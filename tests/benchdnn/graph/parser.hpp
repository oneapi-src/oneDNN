/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_PARSER_HPP
#define BENCHDNN_GRAPH_PARSER_HPP

#include <map>
#include <string>

#include "allocator.hpp"
#include "dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "utils.hpp"

extern dnnl_engine_kind_t engine_tgt_kind;

namespace graph {

bool parse_input_shapes(
        std::vector<std::map<size_t, std::string>> &in_shapes_vec,
        const char *str, const std::string &option_name = "in-shapes");

bool parse_op_attrs(std::vector<std::map<size_t, std::string>> &op_attrs_vec,
        const char *str);

bool parse_graph_expected_n_partitions(
        std::vector<size_t> &expected_n_partition_vec, const char *str);

bool parse_graph_fpmath_mode(
        std::vector<graph_fpmath_mode_t> &fpmath_mode_vec, const char *str);

bool parse_input_file(std::string &json_file, const char *str);

bool parse_dt(std::vector<dnnl_data_type_t> &dt,
        std::vector<std::map<size_t, dnnl_data_type_t>> &dt_map,
        const char *str, const std::string &option_name = "dt");

std::map<std::string, std::string> parse_attrs(const std::string &attrs_str);

// Convert f32 vec attrs string into f32 vec
// e.g. 1.0x2.2 -> {1.0, 2.2}
std::vector<float> string_to_f32_vec(const std::string &val_str);

// Convert shape string from cml into dims_t
// e.g. 1x2x3 -> {1,2,3}
dims_t string_to_shape(const std::string &shape_str);

} // namespace graph
#endif
