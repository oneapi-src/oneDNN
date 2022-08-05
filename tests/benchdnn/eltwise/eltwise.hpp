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

#ifndef ELTWISE_HPP
#define ELTWISE_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace eltwise {

using alg_t = attr_t::post_ops_t::kind_t;

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_dims_t prb_dims;

    std::vector<dir_t> dir {FWD_D};
    std::vector<dnnl_data_type_t> dt {dnnl_f32};
    std::vector<std::string> tag {tag::abx};
    std::vector<alg_t> alg {alg_t::RELU};
    std::vector<float> alpha {0.f}, beta {0.f};

    const char *perf_template_csv() const {
        static const std::string args = "%dir%,%dt%,%tag%,%alg%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_dims_t {
    prb_t(const prb_dims_t &prb_dims, dir_t dir, dnnl_data_type_t dt,
            const std::string &tag, alg_t alg, float alpha, float beta,
            bool inplace, const attr_t &attr, const thr_ctx_t &ctx_init,
            const thr_ctx_t &ctx_exe, int64_t mb = 0)
        : prb_dims_t(prb_dims)
        , dir(dir)
        , dt(dt)
        , tag(tag)
        , alg(alg)
        , alpha(alpha)
        , beta(beta)
        , inplace(inplace)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , user_mb(mb) {
        if (mb) dims[0] = mb;
    }
    ~prb_t() {}

    dir_t dir;
    dnnl_data_type_t dt;
    std::string tag;
    alg_t alg;
    float alpha, beta;
    bool inplace;
    attr_t attr;
    const thr_ctx_t ctx_init, ctx_exe;
    int64_t user_mb;

    bool use_dst() const {
        return alg == alg_t::RELU_DST || alg == alg_t::TANH_DST
                || alg == alg_t::ELU_DST || alg == alg_t::SQRT_DST
                || alg == alg_t::LOGISTIC_DST || alg == alg_t::EXP_DST
                || alg == alg_t::CLIP_V2_DST;
    }
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , tag_(normalize_tag(p_->tag, p_->ndims)) {}

    void dump_alg(std::ostream &s) const override { s << p_->alg; }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_dims_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const std::string *name() const override { return &p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const dnnl_data_type_t *dt() const override { return &p_->dt; }
    const int64_t *user_mb() const override { return &p_->user_mb; }
    const std::string *tag() const override { return &tag_; }

private:
    const prb_t *p_;
    std::string tag_;
};

int fill_data(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp);
bool check_abs_err(const prb_t *prb, const float &s, const float &trh);
float get_eltwise_zero_trust_percent(const prb_t *prb);
float get_eltwise_threshold(dnnl_data_type_t dt, alg_t alg, bool is_fwd = true);

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args);

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace eltwise

#endif
