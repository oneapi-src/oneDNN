/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#ifndef IP_HPP
#define IP_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace ip {

struct desc_t {
    int64_t mb, oc, ic, id, ih, iw;
    std::string name;
    int ndims;

    dims_t src_dims() const;
    dims_t wei_dims() const;
    dims_t bia_dims() const;
    dims_t dst_dims() const;
    int64_t desc_nelems(int arg, int mask) const;
};
int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

typedef struct dt_conf_t {
    dnnl_data_type_t dt;
    double min, max; /* representative */
    double f_min, f_max; /* fill range */
    int f_base; /* fill base, use 0 */
    double f_sparsity; /* amount of non-zeros, default 0.25 */
    double f_scale; /* fill scale, scaling factor for integer generated data */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

extern const _dt_conf_t conf_f32;
extern const _dt_conf_t conf_bf16bf16f32;

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    desc_t desc {};

    std::vector<dir_t> dir {FWD_B};
    std::vector<const dt_conf_t *> cfg {conf_f32};
    std::vector<std::string> stag {tag::any}, wtag {tag::any}, dtag {tag::any};

    const char *perf_template_csv() const {
        static const std::string args = "%dir%,%cfg%,%stag%,%wtag%,%dtag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public desc_t {
    prb_t(const desc_t &desc, int64_t mb, dir_t dir, const dt_conf_t *cfg,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe)
        : desc_t(desc)
        , dir(dir)
        , cfg(cfg)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , user_mb(mb)
        , ops(0)
        , src_scales(NULL)
        , wei_scales(NULL)
        , dst_scales(NULL) {
        if (mb) this->mb = mb;
        count_ops();
        src_scales = generate_scales(DNNL_ARG_SRC);
        wei_scales = generate_scales(DNNL_ARG_WEIGHTS);
        dst_scales = generate_scales(DNNL_ARG_DST);
    }
    ~prb_t() {
        if (src_scales) zfree(src_scales);
        if (wei_scales) zfree(wei_scales);
        if (dst_scales) zfree(dst_scales);
    }

    dir_t dir;
    const dt_conf_t *cfg;
    std::string stag, wtag, dtag;
    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;
    int64_t user_mb;

    double ops;
    float *src_scales, *wei_scales, *dst_scales;

    void count_ops() {
        if (ops > 0) return;
        ops = 2. * mb * ic * oc * id * ih * iw;
    };

    dt_conf_t get_dt_conf(data_kind_t dk) const {
        return (attr.fpmath_mode == dnnl_fpmath_mode_bf16 && cfg == conf_f32)
                ? conf_bf16bf16f32[dk]
                : cfg[dk];
    }

    float *generate_scales(int arg) const;

    // Used to construct memory desc when dimensions are runtime since such mds
    // can't be used directly from query and memory objects can't be constructed.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const {
        assert(!"No runtime dimensions support for this driver!");
        return make_benchdnn_dnnl_wrapper<dnnl_memory_desc_t>(nullptr);
    }

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

const dt_conf_t *str2cfg(const char *str);
std::ostream &operator<<(std::ostream &s, const dt_conf_t *cfg);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , wtag_(normalize_tag(p_->wtag, p_->ndims))
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_cfg(std::ostream &s) const override { s << p_->cfg; }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->mb << ',' << p_->oc << ',' << p_->ic << ',' << p_->id << ','
          << p_->ih << ',' << p_->iw;
    }

    double ops() const override { return p_->ops; }
    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const int64_t *user_mb() const override { return &p_->user_mb; }
    const std::string *name() const override { return &p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *wtag() const override { return &wtag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<std::string> stag_;
    std::string wtag_, dtag_;
};

inline size_t src_off_f(const prb_t *prb, int64_t mb, int64_t ic, int64_t id,
        int64_t ih, int64_t iw) {
    return (((mb * prb->ic + ic) * prb->id + id) * prb->ih + ih) * prb->iw + iw;
}

inline size_t wei_off_f(const prb_t *prb, int64_t oc, int64_t ic, int64_t id,
        int64_t ih, int64_t iw) {
    return (((oc * prb->ic + ic) * prb->id + id) * prb->ih + ih) * prb->iw + iw;
}

inline size_t bia_off_f(const prb_t *prb, int64_t oc) {
    return oc;
}

inline size_t dst_off_f(const prb_t *prb, int64_t mb, int64_t oc) {
    return mb * prb->oc + oc;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args);
void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args);

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int doit(const prb_t *prb, res_t *res);

int bench(int argc, char **argv);
} // namespace ip

#endif
