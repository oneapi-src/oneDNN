/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GNORM_HPP
#define GNORM_HPP

#include <stdint.h>
#include <string>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

#include "bnorm/bnorm.hpp"

namespace gnorm {

using check_alg_t = bnorm::check_alg_t;
using flags_t = bnorm::flags_t;
const flags_t NONE = bnorm::NONE;
const flags_t GLOB_STATS = bnorm::GLOB_STATS;
const flags_t USE_SCALE = bnorm::USE_SCALE;
const flags_t USE_SHIFT = bnorm::USE_SHIFT;
const auto flags2str = bnorm::flags2str;
flags_t str2flags(const char *str);

struct desc_t {
    int64_t g, mb, ic, id, ih, iw;
    float eps;
    std::string name;
    int ndims;

    dims_t data_dims() const;
};
int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    desc_t desc {};

    std::vector<dir_t> dir {FWD_D};
    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::vector<std::string>> tag {{tag::abx, tag::any}};
    std::vector<flags_t> flags {NONE};
    check_alg_t check_alg = bnorm::ALG_AUTO;

    const char *perf_template_csv() const {
        static const std::string args = "%dir%,%dt%,%tag%,%flags%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dir.size() == 1 && dt.size() == 1 && tag.size() == 1
                && flags.size() == 1 && base_settings_t::has_single_setup();
    }
};

struct prb_t : public desc_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.desc, s.mb[0], s.dir[0], s.dt[0], s.tag[0], s.flags[0],
                s.inplace[0],
                settings_t::get_attr(s.scales[0], s.zero_points[0],
                        s.post_ops[0], s.scratchpad_mode[0], s.fpmath_mode[0]),
                s.ctx_init[0], s.ctx_exe[0], s.check_alg) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const desc_t &desc, int64_t mb, dir_t dir,
            const std::vector<dnnl_data_type_t> &dt,
            const std::vector<std::string> &tag, flags_t flags, bool inplace,
            const attr_t &attr, const thr_ctx_t &ctx_init,
            const thr_ctx_t &ctx_exe, check_alg_t check_alg)
        : desc_t(desc)
        , check_alg(check_alg)
        , dir(dir)
        , dt(dt)
        , tag(tag)
        , flags(flags)
        , inplace(inplace)
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

        // Broadcast tags if needed
        if (tag.size() == 1) {
            const auto val = tag[0];
            this->tag.assign(2, val);
        }

        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    check_alg_t check_alg;

    std::string stat_tag;
    dir_t dir;
    std::vector<dnnl_data_type_t> dt;
    std::vector<std::string> tag;
    flags_t flags;
    bool inplace;
    attr_t attr;
    const thr_ctx_t ctx_init, ctx_exe;
    int64_t user_mb;

    bool use_stats() const { return flags & GLOB_STATS; }
    bool use_sc() const { return flags & USE_SCALE; }
    bool use_sh() const { return flags & USE_SHIFT; }

    int64_t get_c_start(int64_t _g) const { return _g * this->ic / this->g; }

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
        : base_perf_report_t(perf_template), p_(prb) {
        for (size_t d = 0; d < p_->tag.size(); d++)
            tag_.push_back(normalize_tag(p_->tag[d], p_->ndims));
    }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->g << ',' << p_->mb << ',' << p_->ic << ',' << p_->id << ','
          << p_->ih << ',' << p_->iw << ',' << p_->eps;
    }

    void dump_flags(std::ostream &s) const override {
        s << flags2str(p_->flags);
    }

    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const int64_t *user_mb() const override { return &p_->user_mb; }
    const std::string *name() const override { return &p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const std::vector<std::string> *stag() const override { return &tag_; }

private:
    const prb_t *p_;
    std::vector<std::string> tag_;
};

/* some extra control parameters which shouldn't be placed in prb_t */

inline size_t data_off(const prb_t *prb, int64_t mb, int64_t c, int64_t d,
        int64_t h, int64_t w) {
    return (((mb * prb->ic + c) * prb->id + d) * prb->ih + h) * prb->iw + w;
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

} // namespace gnorm

#endif
