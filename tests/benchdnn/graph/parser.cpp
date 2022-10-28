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

#include "utils/parser.hpp"

#include "parser.hpp"

namespace graph {

using namespace parser;

namespace {
bool parse_string(
        std::string &val, const char *str, const std::string &option_name) {
    const std::string pattern = std::string("--") + option_name + "=";
    if (pattern.find(str, 0, pattern.size()) == std::string::npos) return false;
    str = str + pattern.size();
    return val = str, true;
}

void parse_key_value(std::vector<std::map<size_t, std::string>> &res_v,
        const std::string &key_val_str) {
    if (key_val_str.empty()) return;
    res_v.clear();

    // Expected format: KEY1:VAL1[+KEY2:VAL2...]
    std::string::size_type case_pos = 0;
    while (case_pos != std::string::npos) {
        std::string case_str = get_substr(key_val_str, case_pos, ',');
        // Process empty entry when several passed: `--OPT=,KEY1:VAL1...`.
        if (case_str.empty()) {
            res_v.push_back({{0, "default"}});
            continue;
        }

        std::string::size_type val_pos = 0;
        std::map<size_t, std::string> key_val_case;
        key_val_case.clear();
        while (val_pos < case_str.size()) {
            std::string single_key_val = get_substr(case_str, val_pos, '+');
            if (single_key_val.empty()) continue;

            std::string::size_type key_pos = 0;
            std::string key_str = get_substr(single_key_val, key_pos, ':');
            std::string val_str
                    = single_key_val.substr(key_pos, val_pos - key_pos);
            auto key_num = size_t(stoll(key_str));
            if (key_val_case.count(key_num) || single_key_val.empty()) {
                fprintf(stderr, "graph: Parser: repeat id `%zd`, exiting...\n",
                        key_num);
                exit(2);
            }
            key_val_case.emplace(stoll(key_str), val_str);
        }
        res_v.push_back(key_val_case);
    }
}
} // namespace

bool parse_input_shapes(
        std::vector<std::map<size_t, std::string>> &in_shapes_vec,
        const char *str) {
    std::string in_shapes_str;
    if (!parse_string(in_shapes_str, str, "in-shapes")) return false;
    return parse_key_value(in_shapes_vec, in_shapes_str), true;
}

bool parse_op_attrs(std::vector<std::map<size_t, std::string>> &op_attrs_vec,
        const char *str) {
    std::string op_attrs_str;
    if (!parse_string(op_attrs_str, str, "op-attrs")) return false;
    return parse_key_value(op_attrs_vec, op_attrs_str), true;
}

std::map<std::string, std::string> parse_attrs(const std::string &attrs_str) {
    std::map<std::string, std::string> attrs_map;
    std::string::size_type key_pos = 0;
    std::string::size_type key_end, val_pos, val_end;
    std::map<size_t, std::string> key_val_case;
    while ((key_end = attrs_str.find(':', key_pos)) != std::string::npos) {
        if ((val_pos = attrs_str.find_first_not_of(':', key_end))
                == std::string::npos)
            break;
        val_end = attrs_str.find('*', val_pos);
        std::string key_str = attrs_str.substr(key_pos, key_end - key_pos);
        std::string val_str = attrs_str.substr(val_pos, val_end - val_pos);
        // Validation of input happens at `deserialized_op::create()`.
        if (attrs_map.count(key_str)) {
            attrs_map[key_str] = val_str;
            BENCHDNN_PRINT(0, "Repeat attr: %s, will use last value for it.\n",
                    key_str.c_str());
        } else {
            attrs_map.emplace(key_str, val_str);
        }
        key_pos = val_end;
        if (key_pos != std::string::npos) ++key_pos;
    }
    return attrs_map;
}

// Convert f32 vec attrs string into f32 vec
// e.g. 1.0x2.2 -> {1.0, 2.2}
std::vector<float> string_to_f32_vec(const std::string &val_str) {
    std::vector<float> f32_vec;
    std::string::size_type pos = 0;
    while (pos != std::string::npos) {
        std::string num_str = get_substr(val_str, pos, 'x');
        if (!num_str.empty()) {
            f32_vec.emplace_back(atof(num_str.c_str()));
        } else {
            fprintf(stderr,
                    "graph: Parser: invalid attr value `%s`, exiting...\n",
                    val_str.c_str());
            exit(2);
        }
    }
    return f32_vec;
}

// Convert shape string from cml into dims_t
// e.g. 1x2x3 -> {1,2,3}
dims_t string_to_shape(const std::string &shape_str) {
    dims_t shape;
    std::string::size_type pos = 0;
    while (pos != std::string::npos) {
        std::string dim_str = get_substr(shape_str, pos, 'x');
        if (!dim_str.empty()) {
            shape.emplace_back(atof(dim_str.c_str()));
        } else {
            fprintf(stderr,
                    "graph: Parser: invalid shape value `%s`, exiting...\n",
                    shape_str.c_str());
            exit(2);
        }
    }
    return shape;
}

bool parse_input_file(std::string &json_file, const char *str) {
    return parse_string(json_file, str, "case");
}

} // namespace graph
