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

#ifndef BRGEMM_HPP
#define BRGEMM_HPP

#include <algorithm>
#include <iostream>
#include <numeric>

#include "oneapi/dnnl/dnnl.h"

#if defined(DNNL_X64) && DNNL_X64 == 1 \
        && (DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE)
#include "src/cpu/x64/brgemm/brgemm.hpp"
#endif

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/cfg.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace brgemm {

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_vdims_t prb_vdims;

    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::string> stag {tag::abx}, wtag {tag::undef},
            dtag {tag::abx};
    std::vector<std::vector<int64_t>> ld {{}};
    std::vector<dnnl_data_type_t> bia_dt {dnnl_data_type_undef};
    std::vector<int> batch_size {1};
    std::vector<float> alpha {1.f}, beta {0.f};
    std::vector<std::string> brgemm_attr {std::string()};

    const char *perf_template_csv() const {
        static const std::string args = "";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_vdims_t {
    prb_t(const prb_vdims_t &prb_vdims, const std::vector<dnnl_data_type_t> &dt,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, const std::vector<int64_t> &ld,
            dnnl_data_type_t bia_dt, float alpha, float beta, int batch_size,
            const std::string &brgemm_attr, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe)
        : prb_vdims_t(prb_vdims)
        , dt(dt)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , ld(ld)
        , bia_dt(bia_dt)
        , alpha(alpha)
        , beta(beta)
        , batch_size(batch_size)
        , brgemm_attr(brgemm_attr)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , scales(NULL)
        , dst_scales(NULL) {

        // Broadcast data types if needed
        if (dt.size() == 1) {
            const auto val = dt[0]; // Need a copy here.
            this->dt.assign(3, val);
        }

        const auto &srcdims = src_dims();
        const auto &weidims = weights_dims();
        m = srcdims[ndims - 2];
        k = srcdims.back();
        n = weidims.back();
        dst_dims[ndims - 2] = m;
        dst_dims[ndims - 1] = n;

        const auto nelems = std::accumulate(dst_dims.begin(), dst_dims.end(),
                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
        ops = 2. * nelems * k;

        generate_oscales();
        generate_dst_scales();
        src_zp = generate_zero_points(DNNL_ARG_SRC, attr.zero_points, k);
        dst_zp = generate_zero_points(DNNL_ARG_DST, attr.zero_points, n);

        repro = set_repro_line(); // must be last in ctor to collect right info
    }
    ~prb_t() {
        zfree(scales);
        zfree(dst_scales);
        zfree(src_zp);
        zfree(dst_zp);
    }

    int64_t m, n, k;
    // brgemm does not have any propagation kind. Treat as forward inference to
    // have correct check for data type support.
    dir_t dir = FWD_I;
    std::vector<dnnl_data_type_t> dt;
    std::string stag, wtag, dtag;
    std::vector<int64_t> ld;
    dnnl_data_type_t bia_dt;
    float alpha, beta;
    int64_t batch_size;
    std::string brgemm_attr;

    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;

    double ops;
    float *scales, *dst_scales;
    int32_t *src_zp, *dst_zp;

    const dims_t &src_dims() const { return vdims[0]; }
    const dims_t &weights_dims() const { return vdims[1]; }
    // const dims_t &prb_vdims_t::dst_dims() const;

    dnnl_data_type_t src_dt() const { return dt[0]; }
    dnnl_data_type_t wei_dt() const { return dt[1]; }
    dnnl_data_type_t acc_dt() const {
        return is_integral_dt(wei_dt()) ? dnnl_s32 : dnnl_f32;
    }
    dnnl_data_type_t dst_dt() const { return dt[2]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

    int64_t get_lda() const {
        if (!ld.empty() && ld[0] != 0) {
            assert(ld[0] >= batch_size * k);
            return ld[0];
        }
        return batch_size * k;
    }
    int64_t get_ldb() const {
        if (!ld.empty() && ld[1] != 0) {
            assert(ld[1] >= n);
            return ld[1];
        }
        // by default, use blocked weights w/ blocksize 16
        const int64_t ldb = rnd_up(n, 16);
        return ldb;
    }
    int64_t get_ldc(bool use_dst_as_acc) const {
        if (use_dst_as_acc) return get_ldd();
        return n;
    }
    int64_t get_ldd() const {
        if (!ld.empty() && ld[2] != 0) {
            assert(ld[2] >= n);
            return ld[2];
        }
        return n;
    }

    void generate_oscales();
    void generate_dst_scales();
    int32_t *generate_zero_points(
            int arg, const attr_t::zero_points_t &zero_points, int N);

    // Used to construct memory desc when dimensions are runtime since such mds
    // can't be used directly from query and memory objects can't be constructed.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const {
        assert(!"No runtime dimensions support for this driver!");
        return make_benchdnn_dnnl_wrapper<dnnl_memory_desc_t>(nullptr);
    }

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);

    const char *str() const { return repro.c_str(); }

private:
    std::string repro;

    std::string set_repro_line();
};

// TODO: not supported as of now.
struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , wtag_(normalize_tag(p_->wtag, p_->ndims))
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    double ops() const override { return p_->ops; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const std::string *name() const override { return &p_->name; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *wtag() const override { return &wtag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<std::string> stag_;
    std::string wtag_, dtag_;
};

struct cfg_t : public base_cfg_t {
    cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds);

    cfg_entry_t::cfg_map_t get_cfg_map(data_kind_t kind) const override;

    float get_density(const density_args_t &density_args) const override;
};

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int doit(const prb_t *prb, res_t *res);

int bench(int argc, char **argv);

} // namespace brgemm

#endif
