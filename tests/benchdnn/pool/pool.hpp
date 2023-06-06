/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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
#include "utils/cfg.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace pool {

enum alg_t {
    undef,
    max,
    avg_np,
    avg_p,
    pooling_max = max,
    pooling_avg_exclude_padding = avg_np,
    pooling_avg_include_padding = avg_p,
};
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct desc_t {
    int64_t mb, ic;
    int64_t id, ih, iw;
    int64_t od, oh, ow;
    int64_t kd, kh, kw;
    int64_t dd, dh, dw;
    int64_t sd, sh, sw;
    int64_t pd, ph, pw;
    int64_t pd_r, ph_r, pw_r; // End side padding for each dimension

    std::string name;
    int ndims;

    // Initialize dependent opposite-side paddings values from the shape
    // parameters
    void init_pad_r() {
        pw_r = opp_pad(iw, ow, kw, dw, sw, pw);
        ph_r = opp_pad(ih, oh, kh, dh, sh, ph);
        pd_r = opp_pad(id, od, kd, dd, sd, pd);
    }

    dims_t src_dims() const;
    dims_t dst_dims() const;
    dims_t strides() const;
    dims_t kernel() const;
    dims_t dilations() const;
    dims_t padding() const;
    dims_t padding_r() const;

private:
    int64_t opp_pad(
            int64_t i, int64_t o, int64_t k, int64_t d, int64_t s, int64_t p) {
        return (o - 1) * s - i + ((k - 1) * (d + 1) + 1) - p;
    }
};

int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

std::string str2cfg(const char *str);

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    desc_t desc {};

    std::vector<dir_t> dir {FWD_D};
    std::vector<std::string> cfg {std::string()};
    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::string> tag {tag::abx};
    std::vector<alg_t> alg {max};

    const char *perf_template_csv() const {
        static const std::string args = "%dir%,%sdt%,%tag%,%alg%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dir.size() == 1 && dt.size() == 1 && tag.size() == 1
                && alg.size() == 1 && base_settings_t::has_single_setup();
    }
};

struct prb_t : public desc_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.desc, s.dir[0], s.dt[0], s.tag[0], s.alg[0],
                settings_t::get_attr(s.scales[0], s.zero_points[0],
                        s.post_ops[0], s.scratchpad_mode[0], s.fpmath_mode[0]),
                s.ctx_init[0], s.ctx_exe[0], s.mb[0]) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const desc_t &desc, dir_t dir,
            const std::vector<dnnl_data_type_t> &dt, const std::string &tag,
            alg_t alg, const attr_t &attr, const thr_ctx_t &ctx_init,
            const thr_ctx_t &ctx_exe, int64_t mb = 0)
        : desc_t(desc)
        , dir(dir)
        , dt(dt)
        , tag(tag)
        , alg(alg)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , user_mb(mb) {
        if (mb) this->mb = mb;

        // Broadcast data types if needed
        if (dt.size() == 1) {
            const auto val = dt[0]; // Need a copy here.
            this->dt.assign(2, val);
        }

        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    dir_t dir;
    std::vector<dnnl_data_type_t> dt;
    std::string tag;
    alg_t alg;
    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;
    int64_t user_mb;

    int64_t kernel_size() const { return kd * kh * kw; }
    bool has_ker_in_pad() const {
        bool ker_in_pad_d = pd >= kd || pd_r >= kd;
        bool ker_in_pad_h = ph >= kh || ph_r >= kh;
        bool ker_in_pad_w = pw >= kw || pw_r >= kw;
        return ker_in_pad_d || ker_in_pad_h || ker_in_pad_w;
    }

    dnnl_data_type_t src_dt() const { return dt[0]; }
    dnnl_data_type_t dst_dt() const { return dt[1]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

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
        , tag_(normalize_tag(p_->tag, p_->ndims)) {}

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->mb << ','

          << p_->ic << ',' << p_->id << ',' << p_->ih << ',' << p_->iw << ','

          << p_->od << ',' << p_->oh << ',' << p_->ow << ','

          << p_->kd << ',' << p_->kh << ',' << p_->kw << ','

          << p_->sd << ',' << p_->sh << ',' << p_->sw << ','

          << p_->pd << ',' << p_->ph << ',' << p_->pw << ','

          << p_->dd << ',' << p_->dh << ',' << p_->dw;
    }

    const int64_t *user_mb() const override { return &p_->user_mb; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const std::string *name() const override { return &p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::string *tag() const override { return &tag_; }

private:
    const prb_t *p_;
    std::string tag_;
};

struct cfg_t : public base_cfg_t {
    cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds);

    cfg_entry_t::cfg_map_t get_cfg_map(data_kind_t kind) const override;
};

int handle_legacy_cfg(
        std::vector<dnnl_data_type_t> &dt, const std::string &cfg);

inline int64_t src_off_f(const prb_t *prb, int64_t mb, int64_t ic, int64_t id,
        int64_t ih, int64_t iw) {
    return (((mb * prb->ic + ic) * prb->id + id) * prb->ih + ih) * prb->iw + iw;
}

inline int64_t dst_off_f(const prb_t *prb, int64_t mb, int64_t ic, int64_t od,
        int64_t oh, int64_t ow) {
    return (((mb * prb->ic + ic) * prb->od + od) * prb->oh + oh) * prb->ow + ow;
}

inline int64_t ker_off_f(const prb_t *prb, int64_t kd, int64_t kh, int64_t kw) {
    return (kd * prb->kh + kh) * prb->kw + kw;
}

inline int64_t get_num_summands(
        const prb_t *prb, int64_t d, int64_t h, int64_t w) {
    const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
    const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
    const int64_t DD = prb->dd, DH = prb->dh, DW = prb->dw;
    const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
    const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;

    auto id_start = d * SD - PD;
    auto ih_start = h * SH - PH;
    auto iw_start = w * SW - PW;
    auto id_end = d * SD - PD + (KD - 1) * DD + KD;
    auto ih_end = h * SH - PH + (KH - 1) * DH + KH;
    auto iw_end = w * SW - PW + (KW - 1) * DW + KW;

    auto id_start_excluded
            = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
    auto ih_start_excluded
            = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
    auto iw_start_excluded
            = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
    auto id_end_excluded = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
    auto ih_end_excluded = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
    auto iw_end_excluded = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

    return prb->alg == avg_p ? prb->kernel_size()
                             : (KD - id_start_excluded - id_end_excluded)
                    * (KH - ih_start_excluded - ih_end_excluded)
                    * (KW - iw_start_excluded - iw_end_excluded);
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
int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace pool

#endif
