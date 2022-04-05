/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef SUM_HPP
#define SUM_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace sum {

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_dims_t prb_dims;

    std::vector<std::vector<dnnl_data_type_t>> sdt {{dnnl_f32, dnnl_f32}};
    std::vector<dnnl_data_type_t> ddt {dnnl_f32};
    std::vector<std::vector<std::string>> stag {{tag::abx}};
    std::vector<std::string> dtag {tag::undef};
    std::vector<std::vector<float>> input_scales {{1}};

    const char *perf_template_csv() const {
        static const std::string args = "%sdt%,%ddt%,%stag%,%dtag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_dims_t {
    prb_t(const prb_dims_t &prb_dims, const std::vector<dnnl_data_type_t> &sdt,
            dnnl_data_type_t ddt, const std::vector<std::string> &stag,
            const std::string &dtag, const std::vector<float> &input_scales,
            const attr_t &attr)
        : prb_dims_t(prb_dims)
        , sdt(sdt)
        , ddt(ddt)
        , stag(stag)
        , dtag(dtag)
        , input_scales(sdt.size())
        , attr(attr) {
        // if there is a single scale then broadcast it
        for (int i_input = 0; i_input < n_inputs(); i_input++)
            this->input_scales[i_input] = ((int)input_scales.size() == 1)
                    ? input_scales[0]
                    : input_scales[i_input];
    }
    ~prb_t() {}

    dir_t dir = FLAG_FWD; // Lack of prop_kind, always considered as forward.
    std::vector<dnnl_data_type_t> sdt;
    dnnl_data_type_t ddt;
    std::vector<std::string> stag;
    std::string dtag;
    std::vector<float> input_scales;
    attr_t attr;

    int n_inputs() const { return (int)sdt.size(); }
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , stag_({})
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {
        for (size_t d = 0; d < p_->stag.size(); d++)
            stag_.push_back(normalize_tag(p_->stag[d], p_->ndims));
    }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_dims_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    const std::string *name() const override { return &p_->name; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->sdt;
    }
    const dnnl_data_type_t *ddt() const override { return &p_->ddt; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace sum

#endif
