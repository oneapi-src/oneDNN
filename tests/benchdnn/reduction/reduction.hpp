/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef REDUCTION_HPP
#define REDUCTION_HPP

#include "oneapi/dnnl/dnnl.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace reduction {

enum alg_t {
    undef,
    min,
    max,
    mul,
    sum,
    mean,
    norm_lp_max,
    norm_lp_sum,
    norm_lp_power_p_max,
    norm_lp_power_p_sum,
    reduction_min = min,
    reduction_max = max,
    reduction_mul = mul,
    reduction_sum = sum,
    reduction_mean = mean,
    reduction_norm_lp_max = norm_lp_max,
    reduction_norm_lp_sum = norm_lp_sum,
    reduction_norm_lp_power_p_max = norm_lp_power_p_max,
    reduction_norm_lp_power_p_sum = norm_lp_power_p_sum,
};

alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_vdims_t prb_vdims;

    std::vector<dnnl_data_type_t> sdt {dnnl_f32};
    std::vector<dnnl_data_type_t> ddt {dnnl_f32};
    std::vector<std::string> stag {tag::abx};
    std::vector<std::string> dtag {tag::any};
    std::vector<alg_t> alg {alg_t::sum};
    std::vector<float> p {1.0f}, eps {0.0f};

    const char *perf_template_csv() const {
        static const std::string args = "%sdt%,%ddt%,%stag%,%dtag%,%alg%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_vdims_t {
    prb_t(const prb_vdims_t &prb_vdims, dnnl_data_type_t sdt,
            dnnl_data_type_t ddt, const std::string &stag,
            const std::string &dtag, alg_t alg, float p, float eps,
            const attr_t &attr)
        : prb_vdims_t(prb_vdims)
        , sdt(sdt)
        , ddt(ddt)
        , stag(stag)
        , dtag(dtag)
        , alg(alg)
        , p(p)
        , eps(eps)
        , attr(attr) {}

    dir_t dir = FLAG_FWD; // Lack of prop_kind, always considered as forward.
    dnnl_data_type_t sdt, ddt;
    std::string stag, dtag;
    alg_t alg;
    float p, eps;
    attr_t attr;
};

std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , prb_(prb)
        , sdt_({prb_->sdt})
        , stag_({normalize_tag(prb_->stag, prb_->ndims)})
        , dtag_(normalize_tag(prb_->dtag, prb_->ndims)) {}

    void dump_alg(std::ostream &s) const override { s << alg2str(prb_->alg); }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*prb_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    const attr_t *attr() const override { return &prb_->attr; }
    const std::string *name() const override { return &prb_->name; }
    const std::vector<dnnl_data_type_t> *sdt() const override { return &sdt_; }
    const dnnl_data_type_t *ddt() const override { return &prb_->ddt; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *prb_;
    std::vector<dnnl_data_type_t> sdt_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args);

int fill_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp);
int fill_dst(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp);

} // namespace reduction

#endif
