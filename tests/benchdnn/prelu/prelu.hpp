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

#ifndef PRELU_HPP
#define PRELU_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace prelu {

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_vdims_t prb_vdims;

    std::vector<dir_t> dir {FWD_D};
    std::vector<std::vector<dnnl_data_type_t>> sdt {{dnnl_f32, dnnl_f32}};
    std::vector<std::vector<std::string>> stag {{tag::abx, tag::any}};

    const char *perf_template_csv() const {
        static const std::string args = "%dir%,%sdt%,%stag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_vdims_t {
    prb_t(const prb_vdims_t &prb_vdims, dir_t dir,
            const std::vector<dnnl_data_type_t> &sdt,
            const std::vector<std::string> &stag, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe)
        : prb_vdims_t(prb_vdims)
        , dir(dir)
        , sdt(sdt)
        , stag(stag)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe) {}
    ~prb_t() {}

    dir_t dir;
    std::vector<dnnl_data_type_t> sdt;
    std::vector<std::string> stag;
    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;
};

std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template), prb_(prb), stag_({}) {
        for (size_t d = 0; d < prb_->stag.size(); d++)
            stag_.push_back(normalize_tag(prb_->stag[d], prb_->ndims));
    }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*prb_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    const thr_ctx_t *ctx_init() const override { return &prb_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &prb_->ctx_exe; }
    const std::string *name() const override { return &prb_->name; }
    const dir_t *dir() const override { return &prb_->dir; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &prb_->sdt;
    }
    const std::vector<std::string> *stag() const override { return &stag_; }

private:
    const prb_t *prb_;
    std::vector<std::string> stag_;
};

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args);
int fill_data(data_kind_t kind, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp);
int setup_prelu_po(const_dnnl_primitive_desc_t pd, std::vector<int> &args,
        std::vector<dnn_mem_t> &ref_mem, std::vector<dnn_mem_t> &prim_mem);
void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace prelu

#endif
