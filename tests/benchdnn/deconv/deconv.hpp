/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef DECONV_HPP
#define DECONV_HPP

#include <iostream>

#include <assert.h>
#include <stdint.h>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/compare.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace deconv {

enum alg_t {
    UNDEF,
    DIRECT,
    WINO,
    AUTO,
    deconvolution_direct = DIRECT,
    deconvolution_wino = WINO,
    deconvolution_auto = AUTO,
};
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
alg_t alg_kind2alg(dnnl_alg_kind_t alg);

struct desc_t {
    int64_t g, mb;
    int64_t ic, id, ih, iw;
    int64_t oc, od, oh, ow;
    int64_t kd, kh, kw;
    int64_t sd, sh, sw;
    int64_t pd, ph, pw;
    int64_t pd_r, ph_r, pw_r; // End side padding for each dimension
    int64_t dd, dh, dw;
    bool has_groups;

    std::string name;
    int ndims;

    // Initialize dependent opposite-side paddings values from the shape
    // parameters.
    void init_pad_r() {
        pw_r = opp_pad(iw, ow, kw, sw, pw, dw);
        ph_r = opp_pad(ih, oh, kh, sh, ph, dh);
        pd_r = opp_pad(id, od, kd, sd, pd, dd);
    }

    int64_t desc_nelems(int arg, int mask) const;

    dims_t src_dims() const;
    dims_t wei_dims() const;
    dims_t bia_dims() const;
    dims_t dst_dims() const;
    dims_t strides() const;
    dims_t dilations() const;
    dims_t padding() const;
    dims_t padding_r() const;

private:
    int64_t opp_pad(int64_t i, int64_t o, int64_t k, int64_t s, int64_t p,
            int64_t d) const {
        return (i - 1) * s - o + ((k - 1) * (d + 1) + 1) - p;
    }
};

int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

/** configuration structure, that controls initial data filling + error check
 *
 * dt defines deconvolution precision
 *
 * for each type (SRC, WEI, BIA, and DST) the values are filled as follows:
 * if (rand() > f_sparsity) then:
 *     v <-- f_base // it is guaranteed each kernel window
 *                  // has at least one non-zero element
 * else:
 *     v <-- f_min + rand() * f_step % (f_max - f_min)
 *
 *
 * on final check the resulting values should be in [min .. max] range, the
 * relative difference should not exceed eps
 */
typedef struct dt_conf_t {
    dnnl_data_type_t dt;
    double min, max; /* representative */
    int f_min, f_max; /* fill range */
    int f_base; /* fill base, use 0 */
    int f_step; /* fill step, use 1 */
    double f_sparsity; /* amount of non-zeros, default 0.25 */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

extern const _dt_conf_t conf_f32;
extern const _dt_conf_t conf_f32_with_bf16_fpmath;
extern const _dt_conf_t conf_f32_with_tf32_fpmath;

const dt_conf_t *str2cfg(const char *str);
std::ostream &operator<<(std::ostream &s, const dt_conf_t *cfg);
const dt_conf_t *auto_cfg(const alg_t alg, const dt_conf_t *cfg);

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
    std::vector<alg_t> alg {DIRECT};

    const char *perf_template_csv() const {
        static const std::string args
                = "%dir%,%cfg%,%stag%,%wtag%,%dtag%,%alg%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dir.size() == 1 && cfg.size() == 1 && stag.size() == 1
                && wtag.size() == 1 && dtag.size() == 1 && alg.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public desc_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.desc, s.dir[0], s.cfg[0], s.stag[0], s.wtag[0], s.dtag[0],
                s.alg[0],
                settings_t::get_attr(s.scales[0], s.zero_points[0],
                        s.post_ops[0], s.scratchpad_mode[0], s.fpmath_mode[0]),
                s.ctx_init[0], s.ctx_exe[0], s.mb[0]) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const desc_t &desc, dir_t dir, const dt_conf_t *cfg,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, alg_t alg, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe, int64_t mb = 0)
        : desc_t(desc)
        , dir(dir)
        , cfg(cfg)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , alg(alg)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , user_mb(mb)
        , ops(0) {
        if (mb) this->mb = mb;
        count_ops();
        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    dir_t dir;
    const dt_conf_t *cfg;
    std::string stag, wtag, dtag;
    alg_t alg;
    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;
    int64_t user_mb;

    double ops;

    void count_ops();

    const dt_conf_t &get_dt_conf(data_kind_t dk) const {
        if (cfg == conf_f32) {
            switch (attr.fpmath_mode) {
                case dnnl_fpmath_mode_bf16:
                    return conf_f32_with_bf16_fpmath[dk];
                case dnnl_fpmath_mode_tf32:
                    return conf_f32_with_tf32_fpmath[dk];
                default: return cfg[dk];
            }
        }
        return cfg[dk];
    }

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
        : base_perf_report_t(perf_template)
        , p_(prb)
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , wtag_(normalize_tag(p_->wtag, p_->ndims))
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_cfg(std::ostream &s) const override { s << p_->cfg; }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->g << ',' << p_->mb << ','

          << p_->ic << ',' << p_->id << ',' << p_->ih << ',' << p_->iw << ','

          << p_->oc << ',' << p_->od << ',' << p_->oh << ',' << p_->ow << ','

          << p_->kd << ',' << p_->kh << ',' << p_->kw << ','

          << p_->sd << ',' << p_->sh << ',' << p_->sw << ','

          << p_->pd << ',' << p_->ph << ',' << p_->pw << ','

          << p_->dd << ',' << p_->dh << ',' << p_->dw;
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

int transpose_data_wei(
        const prb_t *prb, const dnn_mem_t &wei, const dnn_mem_t &wei_tr);

inline int64_t src_off_f(const prb_t *prb, int64_t mb, int64_t g, int64_t ic,
        int64_t id, int64_t ih, int64_t iw) {
    return (((mb * prb->ic + g * prb->ic / prb->g + ic) * prb->id + id)
                           * prb->ih
                   + ih)
            * prb->iw
            + iw;
}

inline int64_t wei_off_f(const prb_t *prb, int64_t g, int64_t oc, int64_t ic,
        int64_t kd, int64_t kh, int64_t kw) {
    return ((((g * prb->oc / prb->g + oc) * prb->ic / prb->g + ic) * prb->kd
                    + kd) * prb->kh
                   + kh)
            * prb->kw
            + kw;
}

inline int64_t bia_off_f(const prb_t *prb, int64_t g, int64_t oc) {
    return g * prb->oc / prb->g + oc;
}

inline int64_t dst_off_f(const prb_t *prb, int64_t mb, int64_t g, int64_t oc,
        int64_t od, int64_t oh, int64_t ow) {
    return (((mb * prb->oc + g * prb->oc / prb->g + oc) * prb->od + od)
                           * prb->oh
                   + oh)
            * prb->ow
            + ow;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args);
void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args);
std::vector<int> supported_exec_args(dir_t dir);
int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref = nullptr);

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace deconv

#endif
