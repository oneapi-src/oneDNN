/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef PRELU_HPP
#define PRELU_HPP

#include "dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace prelu {

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }
    std::vector<dims_t> dims;
    std::vector<dir_t> dir {FWD_D};
    std::vector<dnnl_data_type_t> dt {dnnl_f32};
    std::vector<std::string> tag {tag::abx};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%dir%,%dt%,%tag%,%DESC%,%-time%,%"
              "0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t {
    prb_t(const std::vector<dims_t> &dims, dir_t dir, dnnl_data_type_t dt,
            const std::string &tag, const attr_t &attr, int64_t mb = 0)
        : dims(dims)
        , dir(dir)
        , dt(dt)
        , tag(tag)
        , attr(attr)
        , ndims((int)dims[0].size()) {
        if (mb) this->dims[0][0] = mb;
    }
    ~prb_t() {}

    std::vector<dims_t> dims;
    dir_t dir;
    dnnl_data_type_t dt;
    std::string tag;
    attr_t attr;
    int ndims;
};

std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *prb, const res_t *res, const char *prb_str) {
        prb_ = prb;
        tag_ = normalize_tag(prb_->tag, prb_->ndims);
        base_report(res, prb_str);
    }
    void dump_desc(std::ostream &s) const override { s << prb_->dims; }

    void dump_desc_csv(std::ostream &s) const override { s << prb_->dims; }

    const dir_t *dir() const override { return &prb_->dir; }
    const dnnl_data_type_t *dt() const override { return &prb_->dt; }
    const std::string *tag() const override { return &tag_; }

private:
    const prb_t *prb_ = NULL;
    std::string tag_;
};

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &dst);
void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_weights);
int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace prelu

#endif
