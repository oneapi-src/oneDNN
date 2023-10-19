/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_GRAPH_HPP
#define BENCHDNN_GRAPH_GRAPH_HPP

#define UNMAP 0
#define MAP 1

#include <iostream>
#include <map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common.hpp"
#include "deserialize.hpp"
#include "dnn_types.hpp"
#include "dnnl_debug.hpp"
#include "utils.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace graph {

using namespace dnnl::graph;

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }
    std::string json_file;
    std::vector<std::map<size_t, std::string>> in_shapes_vec {{{0, "default"}}};
    std::vector<std::map<size_t, std::string>> op_attrs_vec {{{0, "default"}}};

    const char *perf_template_csv
            = "perf,%engine%,%DESC%,"
              "%-time%,%0time%";
    const char *perf_template_def = "perf,%engine%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

// TODO evaluate prb_t struct
struct prb_t {
    prb_t(const deserialized_graph &dg,
            const attr_t::fpmath_mode_t &fpmath_mode)
        : dg(dg) {
        switch (fpmath_mode.mode) {
            case dnnl_fpmath_mode_strict:
                this->fpmath_mode = dnnl::fpmath_mode::strict;
                break;
            case dnnl_fpmath_mode_bf16:
                this->fpmath_mode = dnnl::fpmath_mode::bf16;
                break;
            case dnnl_fpmath_mode_f16:
                this->fpmath_mode = dnnl::fpmath_mode::f16;
                break;
            case dnnl_fpmath_mode_any:
                this->fpmath_mode = dnnl::fpmath_mode::any;
                break;
            case dnnl_fpmath_mode_tf32:
                this->fpmath_mode = dnnl::fpmath_mode::tf32;
                break;
        }
    }

    deserialized_graph dg;
    dnnl::fpmath_mode fpmath_mode;
};

std::string case_to_str(const std::string &json_file,
        const std::map<size_t, std::string> &in_shapes,
        const std::map<size_t, std::string> &op_attrs, const int64_t mb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const std::string case_str, const char *perf_template)
        : base_perf_report_t(perf_template), case_str_(case_str) {}
    void dump_desc(std::ostream &s) const override { s << case_str_; }
    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

private:
    const std::string case_str_;
};

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);
} // namespace graph

#endif
