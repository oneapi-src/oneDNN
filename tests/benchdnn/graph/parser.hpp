/*******************************************************************************
* Copyright 2022 Intel Corporation
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

namespace graph {

bool parse_input_shapes(
        std::vector<std::map<size_t, std::string>> &in_shapes_vec,
        const char *str);

bool parse_op_attrs(std::vector<std::map<size_t, std::string>> &op_attrs_vec,
        const char *str);

bool parse_input_file(std::string &json_file, const char *str);

std::map<std::string, std::string> parse_attrs(const std::string &attrs_str);

// Convert f32 vec attrs string into f32 vec
// e.g. 1.0x2.2 -> {1.0, 2.2}
std::vector<float> string_to_f32_vec(const std::string &val_str);

// Convert shape string from cml into dims_t
// e.g. 1x2x3 -> {1,2,3}
dims_t string_to_shape(const std::string &shape_str);

} // namespace graph
#endif
