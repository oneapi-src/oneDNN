/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace eltwise {

using alg_t = attr_t::post_ops_t::kind_t;

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    dims_t dims;

    std::vector<dir_t> dir {FWD_D};
    std::vector<dnnl_data_type_t> dt {dnnl_f32};
    std::vector<std::string> tag {tag::abx};
    std::vector<alg_t> alg {alg_t::RELU};
    std::vector<float> scales {0, 0.25, -0.25}, alpha {scales}, beta {scales};
    std::vector<int64_t> mb {0};
    std::vector<bool> inplace {true};
    bool allow_unimpl = false;

    const char *perf_template_csv
            = "perf,%engine%,%dir%,%dt%,%tag%,%alg%,%DESC%,%-time%,%0time%";
    const char *perf_template_def = "perf,%engine%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t {
    prb_t(const dims_t &dims, dir_t dir, dnnl_data_type_t dt,
            const std::string &tag, alg_t alg, float alpha, float beta,
            bool inplace, int64_t mb = 0)
        : dims(dims)
        , dir(dir)
        , dt(dt)
        , tag(tag)
        , alg(alg)
        , alpha(alpha)
        , beta(beta)
        , inplace(inplace)
        , ndims((int)dims.size()) {
        if (mb) this->dims[0] = mb;
    }
    ~prb_t() {}

    dims_t dims;
    dir_t dir;
    dnnl_data_type_t dt;
    std::string tag;
    alg_t alg;
    float alpha, beta;
    bool inplace;
    int ndims;

    bool use_dst() const {
        return alg == alg_t::RELU_DST || alg == alg_t::TANH_DST
                || alg == alg_t::ELU_DST || alg == alg_t::SQRT_DST
                || alg == alg_t::LOGISTIC_DST || alg == alg_t::EXP_DST;
    }
};
std::ostream &operator<<(std::ostream &s, const prb_t &p);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *p, const res_t *r, const char *prb_str) {
        p_ = p;
        base_report(r, prb_str);
    }

    void dump_alg(std::ostream &s) const override { s << p_->alg; }

    void dump_desc(std::ostream &s) const override { s << p_->dims; }

    void dump_desc_csv(std::ostream &s) const override { s << p_->dims; }

    const dir_t *dir() const override { return &p_->dir; }
    const dnnl_data_type_t *dt() const override { return &p_->dt; }
    const std::string *tag() const override { return &p_->tag; }

private:
    const prb_t *p_ = NULL;
};

bool check_extreme_values(const float &a, const float &b, alg_t alg);
void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src, dnn_mem_t &dst);
void compute_ref_bwd(const prb_t *p, const dnn_mem_t &src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_src);

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

} // namespace eltwise

#endif
