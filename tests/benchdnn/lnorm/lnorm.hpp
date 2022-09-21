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

#ifndef LNORM_HPP
#define LNORM_HPP

#include <assert.h>
#include <limits.h>
#include <numeric>
#include <stdint.h>

#include <iostream>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

#include "bnorm/bnorm.hpp"

namespace lnorm {

using check_alg_t = bnorm::check_alg_t;
using flags_t = bnorm::flags_t;
const flags_t NONE = bnorm::NONE;
const flags_t GLOB_STATS = bnorm::GLOB_STATS;
const flags_t USE_SCALE = bnorm::USE_SCALE;
const flags_t USE_SHIFT = bnorm::USE_SHIFT;
const auto flags2str = bnorm::flags2str;
flags_t str2flags(const char *str);

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_dims_t prb_dims;

    std::vector<dir_t> dir {FWD_D};
    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::vector<std::string>> tag {{tag::abx, tag::any}};
    std::vector<std::string> stat_tag {tag::any};
    std::vector<flags_t> flags {NONE};
    check_alg_t check_alg = check_alg_t::ALG_AUTO;

    const char *perf_template_csv() const {
        static const std::string args = "%dir%,%dt%,%tag%,%stat_tag%,%flags%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_dims_t {
    prb_t(const prb_dims_t &prb_dims, const std::vector<std::string> &tag,
            const std::string &stat_tag, dir_t dir,
            const std::vector<dnnl_data_type_t> &dt, flags_t flags,
            const attr_t &attr, const thr_ctx_t &ctx_init,
            const thr_ctx_t &ctx_exe, bool inplace, check_alg_t check_alg)
        : prb_dims_t(prb_dims)
        , check_alg(check_alg)
        , tag(tag)
        , stat_tag(stat_tag)
        , dir(dir)
        , dt(dt)
        , flags(flags)
        , inplace(inplace)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , scales(NULL) {
        n = 1;
        for (int d = 0; d < ndims - 1; d++)
            n *= dims[d];
        c = dims[ndims - 1];
        eps = 1.f / 16;

        // Broadcast data types if needed
        if (dt.size() == 1) {
            const auto val = dt[0]; // Need a copy here.
            this->dt.assign(2, val);
        }
        if (tag.size() == 1) { this->tag.push_back(tag::any); }

        generate_oscales();
    }
    ~prb_t() {
        if (scales) zfree(scales);
    }

    check_alg_t check_alg;
    std::vector<std::string> tag;
    std::string stat_tag;
    dir_t dir;
    std::vector<dnnl_data_type_t> dt;
    flags_t flags;
    bool inplace;
    attr_t attr;
    const thr_ctx_t ctx_init, ctx_exe;
    int64_t n, c;
    float eps;

    bool use_sc() const { return flags & USE_SCALE; }
    bool use_sh() const { return flags & USE_SHIFT; }

    float *scales;
    void generate_oscales();
};

std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , stat_tag_(normalize_tag(p_->stat_tag, p_->ndims - 1)) {
        for (size_t d = 0; d < p_->tag.size(); d++)
            tag_.push_back(normalize_tag(p_->tag[d], p_->ndims));
    }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_dims_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    void dump_flags(std::ostream &s) const override {
        s << flags2str(p_->flags);
    }

    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const std::string *name() const override { return &p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const std::vector<std::string> *stag() const override { return &tag_; }
    const std::string *stat_tag() const override { return &stat_tag_; }

private:
    const prb_t *p_;
    std::vector<std::string> tag_;
    std::string stat_tag_;
};

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace lnorm

#endif
