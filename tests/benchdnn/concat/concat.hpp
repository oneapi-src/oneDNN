/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef CONCAT_HPP
#define CONCAT_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/perf_report.hpp"

namespace concat {

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_vdims_t prb_vdims;

    std::vector<dnnl_data_type_t> sdt {dnnl_f32}, ddt {dnnl_f32};
    std::vector<std::vector<std::string>> stag {{tag::abx}};
    std::vector<std::string> dtag {tag::undef};
    std::vector<int> axis {1};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%sdt%,%ddt%,%stag%,%dtag%,%axis%,%DESC%,%-"
              "time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_vdims_t {
    prb_t(const prb_vdims_t &prb_vdims, dnnl_data_type_t sdt,
            dnnl_data_type_t ddt, const std::vector<std::string> &stag,
            const std::string &dtag, int axis, const attr_t &attr)
        : prb_vdims_t(prb_vdims)
        , sdt(sdt)
        , ddt(ddt)
        , stag(stag)
        , dtag(dtag)
        , axis(axis)
        , attr(attr) {
        dst_dims[axis] = axis_size();
    }
    ~prb_t() {}

    dnnl_data_type_t sdt, ddt;
    std::vector<std::string> stag;
    std::string dtag;
    int axis;
    attr_t attr;

    int64_t axis_size() const {
        int64_t as = 0;
        for (int i = 0; i < n_inputs(); ++i)
            as += vdims[i].at(axis);
        return as;
    }
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , sdt_({p_->sdt})
        , stag_({})
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {
        for (size_t d = 0; d < p_->stag.size(); d++)
            stag_.push_back(normalize_tag(p_->stag[d], p_->ndims));
    }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    const int *axis() const override { return &p_->axis; }
    const std::vector<dnnl_data_type_t> *sdt() const override { return &sdt_; }
    const dnnl_data_type_t *ddt() const override { return &p_->ddt; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<dnnl_data_type_t> sdt_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

void compute_ref(
        const prb_t *prb, const std::vector<dnn_mem_t> &src, dnn_mem_t &dst);
void check_known_skipped_case(const prb_t *prb, res_t *res);
int fill_src(int input_idx, dnnl_data_type_t dt, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp);
int init_pd(dnnl_engine_t engine, const prb_t *prb, dnnl_primitive_desc_t &cpd,
        res_t *res, dir_t dir, const_dnnl_primitive_desc_t hint);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace concat

#endif
