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

#ifndef POOL_HPP
#define POOL_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include <iostream>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace pool {

enum alg_t { MAX, AVG_NP, AVG_P };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct desc_t {
    int64_t mb, ic;
    int64_t id, ih, iw;
    int64_t od, oh, ow;
    int64_t kd, kh, kw;
    int64_t sd, sh, sw;
    int64_t pd, ph, pw;

    const char *name;
    int ndims;
};

int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

/** configuration structure, that controls initial data filling + error check
 *
 * dt defines pooling precision
 *
 * for each type (SRC and DST) the values are filled as follows:
 * if (rand() > f_sparsity) then:
 *     v <-- f_base // it is guaranteed each kernel window
 *                  // has at least one non-zero element
 * else:
 *     v <-- f_min + rand() * f_step % (f_max - f_min)
 *
 * on final check the resulting values should be in [min .. max] range, the
 * relative difference should not exceed eps
 */
typedef struct dt_conf_t {
    dnnl_data_type_t dt;
    double min, max; /* representative */
    int f_min, f_max; /* fill range */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

extern const _dt_conf_t conf_f32;

const dt_conf_t *str2cfg(const char *str);
std::ostream &operator<<(std::ostream &s, const dt_conf_t *cfg);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    desc_t desc {};

    std::vector<dir_t> dir {FWD_D};
    std::vector<const dt_conf_t *> cfg {conf_f32};
    std::vector<std::string> tag {tag::abx};
    std::vector<alg_t> alg {MAX};
    std::vector<int64_t> mb {0};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%name%,%dir%,%cfg%,%tag%,%alg%,%DESC%,%-"
              "time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%name%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public desc_t {
    prb_t(const desc_t &desc, dir_t dir, const dt_conf_t *cfg,
            const std::string &tag, alg_t alg, int64_t mb = 0)
        : desc_t(desc), dir(dir), cfg(cfg), tag(tag), alg(alg) {
        if (mb) this->mb = mb;
    }
    ~prb_t() {}

    dir_t dir;
    const dt_conf_t *cfg;
    std::string tag;
    alg_t alg;

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);
};
std::ostream &operator<<(std::ostream &s, const prb_t &p);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *p, const res_t *r, const char *prb_str) {
        p_ = p;
        tag_ = fmt_tag2str(convert_tag(p_->tag, p_->ndims));
        base_report(r, prb_str);
    }

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_cfg(std::ostream &s) const override { s << p_->cfg; }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->mb << ','

          << p_->ic << ',' << p_->id << ',' << p_->ih << ',' << p_->iw << ','

          << p_->od << ',' << p_->oh << ',' << p_->ow << ','

          << p_->kd << ',' << p_->kh << ',' << p_->kw << ','

          << p_->sd << ',' << p_->sh << ',' << p_->sw << ','

          << p_->pd << ',' << p_->ph << ',' << p_->pw;
    }

    const char *name() const override { return p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::string *tag() const override { return &tag_; }

private:
    const prb_t *p_ = NULL;
    std::string tag_;
};

inline int64_t src_off_f(const prb_t *p, int64_t mb, int64_t ic, int64_t id,
        int64_t ih, int64_t iw) {
    return (((mb * p->ic + ic) * p->id + id) * p->ih + ih) * p->iw + iw;
}

inline void inv_src_off_f(const prb_t *p, int64_t off, int64_t &mb, int64_t &ic,
        int64_t &id, int64_t &ih, int64_t &iw) {
    iw = off % p->iw;
    off /= p->iw;
    ih = off % p->ih;
    off /= p->ih;
    id = off % p->id;
    off /= p->id;
    ic = off % p->ic;
    off /= p->ic;
    mb = off % p->mb;
    off /= p->mb;
    assert(off == 0);
}

inline int64_t dst_off_f(const prb_t *p, int64_t mb, int64_t ic, int64_t od,
        int64_t oh, int64_t ow) {
    return (((mb * p->ic + ic) * p->od + od) * p->oh + oh) * p->ow + ow;
}

inline void inv_dst_off_f(const prb_t *p, int64_t off, int64_t &mb, int64_t &ic,
        int64_t &od, int64_t &oh, int64_t &ow) {
    ow = off % p->ow;
    off /= p->ow;
    oh = off % p->oh;
    off /= p->oh;
    od = off % p->od;
    off /= p->od;
    ic = off % p->ic;
    off /= p->ic;
    mb = off % p->mb;
    off /= p->mb;
    assert(off == 0);
}

inline int64_t ker_off_f(const prb_t *p, int64_t kd, int64_t kh, int64_t kw) {
    return (kd * p->kh + kh) * p->kw + kw;
}

inline int64_t get_num_summands(
        const prb_t *p, int64_t d, int64_t h, int64_t w) {
    const int64_t ID = p->id, IH = p->ih, IW = p->iw;
    const int64_t KD = p->kd, KH = p->kh, KW = p->kw;
    const int64_t PD = p->pd, PH = p->ph, PW = p->pw;
    const int64_t SD = p->sd, SH = p->sh, SW = p->sw;

    auto d_start = MAX2(d * SD - PD, 0);
    auto h_start = MAX2(h * SH - PH, 0);
    auto w_start = MAX2(w * SW - PW, 0);
    auto d_end = MIN2(d * SD - PD + KD, ID);
    auto h_end = MIN2(h * SH - PH + KH, IH);
    auto w_end = MIN2(w * SW - PW + KW, IW);

    return p->alg == AVG_P
            ? KD * KH * KW
            : (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
}

void compute_ref_fwd(
        const prb_t *p, const dnn_mem_t &src, dnn_mem_t &dst, dnn_mem_t &ws);
void compute_ref_bwd(const prb_t *p, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, const dnn_mem_t &ws);

int compare_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r);
int compare_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r);
int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r);
int fill_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r);
int fill_ws(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r);

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

} // namespace pool

#endif
