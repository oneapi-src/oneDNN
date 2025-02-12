/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

    bool has_single_setup() const override {
        return dir.size() == 1 && sdt.size() == 1 && stag.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public prb_vdims_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.prb_vdims, s.dir[0], s.sdt[0], s.stag[0],
                s.attributes.front(), s.ctx_init[0], s.ctx_exe[0],
                s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_vdims_t &prb_vdims, dir_t dir,
            const std::vector<dnnl_data_type_t> &sdt,
            const std::vector<std::string> &stag, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe,
            const impl_filter_t &impl_filter)
        : prb_vdims_t(prb_vdims)
        , dir(dir)
        , sdt(sdt)
        , stag(stag)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , impl_filter(impl_filter) {
        // Broadcast data types if needed
        if (sdt.size() == 1) {
            const auto val = sdt[0]; // Need a copy here.
            this->sdt.assign(2, val);
        }

        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    dir_t dir;
    std::vector<dnnl_data_type_t> sdt;
    std::vector<std::string> stag;
    bool inplace = false; // Lacks placement, always considered `false`.
    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;
    impl_filter_t impl_filter;

    // Used to construct memory desc when dimensions are runtime since such mds
    // can't be used directly from query and memory objects can't be constructed.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const {
        assert(!"No runtime dimensions support for this driver!");
        return make_benchdnn_dnnl_wrapper<dnnl_memory_desc_t>(nullptr);
    }

    const char *str() const { return repro.c_str(); }

private:
    std::string repro;

    std::string set_repro_line();
};

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

    const attr_t *attr() const override { return &prb_->attr; }
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

int fill_data(data_kind_t kind, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp);

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args);
void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args);
std::vector<int> supported_exec_args(dir_t dir);
int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res,
        dnnl_primitive_t prim_ref = nullptr);

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace prelu

#endif
