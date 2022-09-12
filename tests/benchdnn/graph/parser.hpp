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

#include <fstream>
#include <stdio.h>
#include <string>

namespace graph {

static bool parse_string(
        std::string &val, const char *str, const std::string &option_name) {
    const std::string pattern = std::string("--") + option_name + "=";
    if (pattern.find(str, 0, pattern.size()) == std::string::npos) return false;
    str = str + pattern.size();
    if (*str == '\0') return val = "", true;
    return val = str, true;
}

void parse_key_value(std::vector<std::map<size_t, std::string>> &res_v,
        std::string key_val_str) {
    std::string::size_type case_pos = 0;
    std::string::size_type case_end, key_pos, key_end, val_pos, val_end;
    while (true) {
        case_end = key_val_str.find(',', case_pos);
        std::string case_str
                = key_val_str.substr(case_pos, case_end - case_pos);
        if (case_str.size() == 0) {
            res_v.push_back({{0, "default"}});
            if (case_end == std::string::npos) break;
            case_pos = case_end;
            ++case_pos;
            continue;
        }
        key_pos = 0;
        std::map<size_t, std::string> key_val_case;
        key_val_case.clear();
        while ((key_end = case_str.find(':', key_pos)) != std::string::npos) {
            if ((val_pos = case_str.find_first_not_of(':', key_end))
                    == std::string::npos)
                break;
            val_end = case_str.find('+', val_pos);
            std::string key_str = case_str.substr(key_pos, key_end - key_pos);
            std::string val_str = case_str.substr(val_pos, val_end - val_pos);
            auto key_num = size_t(stoll(key_str));
            if (key_val_case.count(key_num)) {
                fprintf(stdout,
                        "graph: Parser: repeat id `%zd`, will use first value "
                        "for it.\n",
                        key_num);
            } else {
                key_val_case.emplace(stoll(key_str), val_str);
            }
            key_pos = val_end;
            if (key_pos != std::string::npos) ++key_pos;
        }
        res_v.push_back(key_val_case);
        if (case_end == std::string::npos) break;
        case_pos = case_end;
        ++case_pos;
    }
}

static bool parse_input_shapes(
        std::vector<std::map<size_t, std::string>> &in_shapes_vec,
        const char *str) {
    std::string in_shapes_str;
    auto result = parse_string(in_shapes_str, str, "in-shapes");
    if (result && in_shapes_str != "") {
        in_shapes_vec.clear();
        parse_key_value(in_shapes_vec, in_shapes_str);
    }
    return result;
}

static bool parse_op_attrs(
        std::vector<std::map<size_t, std::string>> &op_attrs_vec,
        const char *str) {
    std::string op_attrs_str;
    auto result = parse_string(op_attrs_str, str, "op-attrs");
    if (result && op_attrs_str != "") {
        op_attrs_vec.clear();
        parse_key_value(op_attrs_vec, op_attrs_str);
    }
    return result;
}

static bool parse_input_file(std::string &json_file, const char *str) {
    return parse_string(json_file, str, "case");
}

void catch_unknown_options(const char *str) {
    const std::string pattern = "--";
    if (pattern.find(str, 0, pattern.size()) != std::string::npos) {
        fprintf(stderr, "graph: ERROR: unknown option: `%s`, exiting...\n",
                str);
        exit(2);
    }
}
} // namespace graph
#endif
