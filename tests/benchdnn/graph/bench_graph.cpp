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

#include <stdio.h>
#include <string>

#include "flex_rewrite.hpp"
#include "graph.hpp"
#include "parser.hpp"
#include "utils/parser.hpp"

namespace graph {

void check_correctness(const settings_t &s) {
    for_(const auto &i_in_shapes : s.in_shapes_vec)
    for_(const auto &i_op_attrs : s.op_attrs_vec)
    for_(const auto &i_fpmath_mode : s.fpmath_mode)
    for (const auto &i_mb : s.mb) {
        deserialized_graph dg;
        dg.load(locate_file(s.json_file));
        flex_rewrite fw(i_in_shapes, i_op_attrs, i_mb);
        fw.rewrite(dg);
        const prb_t prb(dg, i_fpmath_mode);
        const auto &cpp_pstr
                = case_to_str(s.json_file, i_in_shapes, i_op_attrs, i_mb);
        const char *pstr = cpp_pstr.c_str();
        BENCHDNN_PRINT(1, "run: %s\n", pstr);
        res_t res {};
        doit(&prb, &res);
        parse_result(res, pstr);
        if (has_bench_mode_bit(mode_bit_t::perf)) {
            perf_report_t pr(cpp_pstr, s.perf_template);
            pr.report(&res, pstr);
        }
    }
}

int bench(int argc, char **argv) {
    driver_name = "graph";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};

    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_input_shapes(s.in_shapes_vec, argv[0])
                || parse_op_attrs(s.op_attrs_vec, argv[0])
                || parse_attr_fpmath_mode(
                        s.fpmath_mode, def.fpmath_mode, argv[0])
                || parse_mb(s.mb, def.mb, argv[0]) || parse_reset(s, argv[0]);
        if (!parsed_options) {
            if (!parse_input_file(s.json_file, argv[0]))
                catch_unknown_options(argv[0]);
            check_correctness(s);
        }
    }
    return OK;
}
} // namespace graph
