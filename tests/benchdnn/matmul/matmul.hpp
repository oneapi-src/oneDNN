/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <algorithm>
#include <bitset>
#include <iostream>
#include <numeric>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/perf_report.hpp"

namespace matmul {

typedef struct dt_conf_t {
    dnnl_data_type_t dt;
    double min, max; /* representative */
    double f_min, f_max; /* fill range */
    int f_base; /* fill base, use 0 */
    double f_sparsity; /* amount of non-zeros, default 0.25 */
    double f_scale; /* fill scale, scaling factor for integer generated data */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

typedef std::bitset<DNNL_MAX_NDIMS> dims_mask_t;
extern const _dt_conf_t conf_f32;

const int64_t LD_GOOD = INT64_MAX;
const int64_t LD_NONE = INT64_MAX - 1;

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_vdims_t prb_vdims;

    std::vector<const dt_conf_t *> cfg {conf_f32};
    std::vector<std::string> stag {tag::any}, wtag {tag::any}, dtag {tag::any};
    std::vector<vdims_t> strides {vdims_t(STRIDES_SIZE)};
    std::vector<dnnl_data_type_t> bia_dt {dnnl_data_type_undef};
    std::vector<int> bia_mask {2};
    std::vector<std::vector<dims_mask_t>> rt_dims_masks {{}};
    std::vector<attr_t::scale_t> oscale {attr_t::scale_t()};
    std::vector<attr_t::zero_points_t> zero_points {attr_t::zero_points_t()};
    std::vector<attr_t::post_ops_t> post_ops {attr_t::post_ops_t()};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};
    attr_t attr = {};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%name%,%cfg%,%attr%,%DESC%,"
              "%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%name%,%prb%,%Gops%,%Gfreq%,%-time%,%-"
              "Gflops%,%0time%,%0Gflops%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_vdims_t {
    prb_t(const prb_vdims_t &prb_vdims, const dt_conf_t *cfg,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, const vdims_t &strides,
            dnnl_data_type_t bia_dt, int bia_mask,
            const std::vector<dims_mask_t> &rt_dims_masks, const attr_t &attr)
        : prb_vdims_t(prb_vdims)
        , cfg(cfg)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , strides(strides)
        , bia_dt(bia_dt)
        , bia_mask(bia_mask)
        , rt_dims_masks(rt_dims_masks)
        , attr(attr)
        , scales(NULL) {

        this->rt_dims_masks.resize(2);
        const auto &srcdims = src_dims();
        const auto &weidims = weights_dims();
        m = srcdims[ndims - 2];
        k = srcdims.back();
        n = weidims.back();
        dst_dims[ndims - 2] = m;
        dst_dims[ndims - 1] = n;

        init_dst_rt_dims_mask();
        const auto nelems = std::accumulate(dst_dims.begin(), dst_dims.end(),
                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
        ops = 2. * nelems * k;

        generate_oscales();
        src_zp = generate_zero_points(DNNL_ARG_SRC, attr.zero_points, k);
        dst_zp = generate_zero_points(DNNL_ARG_DST, attr.zero_points, n);
    }
    ~prb_t() {
        if (scales) zfree(scales);
        if (src_zp) zfree(src_zp);
        if (dst_zp) zfree(dst_zp);
    }

    int m, n, k;
    const dt_conf_t *cfg;
    std::string stag, wtag, dtag;
    vdims_t strides;
    dnnl_data_type_t bia_dt;
    int bia_mask;
    std::vector<dims_mask_t> rt_dims_masks;

    attr_t attr;

    double ops;
    float *scales;
    int32_t *src_zp, *dst_zp;

    const dims_t &src_dims() const { return vdims[0]; }
    const dims_t &weights_dims() const { return vdims[1]; }

    const dims_mask_t &src_runtime_dim_mask() const { return rt_dims_masks[0]; }
    const dims_mask_t &weights_runtime_dim_mask() const {
        return rt_dims_masks[1];
    }
    const dims_mask_t &dst_runtime_dim_mask() const { return rt_dims_masks[2]; }

    int src_broadcast_mask() const { return get_broadcast_mask(0); }

    int weights_broadcast_mask() const { return get_broadcast_mask(1); }

    int bias_broadcast_mask() const { return bia_mask; }

    void generate_oscales();
    int32_t *generate_zero_points(
            int arg, const attr_t::zero_points_t &zero_points, int N);

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);

private:
    void init_dst_rt_dims_mask() {
        if (rt_dims_masks.size() > 2) return;

        const auto &src_rt_dim_mask = src_runtime_dim_mask();
        const auto &wei_rt_dim_mask = weights_runtime_dim_mask();
        dims_mask_t dst_rt_dim_mask;

        for (int i = 0; i < ndims - 2; ++i) {
            dst_rt_dim_mask[i] = src_rt_dim_mask[i] || wei_rt_dim_mask[i];
        }

        // m, n mask
        dst_rt_dim_mask[ndims - 2] = src_rt_dim_mask[ndims - 2];
        dst_rt_dim_mask[ndims - 1] = wei_rt_dim_mask[ndims - 1];

        rt_dims_masks.push_back(dst_rt_dim_mask);
    }
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

/* some extra control parameters which shouldn't be placed in prb_t */

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
        s << static_cast<const prb_vdims_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    double ops() const override { return p_->ops; }
    const attr_t *attr() const override { return &p_->attr; }
    const char *name() const override { return p_->name.c_str(); }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *wtag() const override { return &wtag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<std::string> stag_;
    std::string wtag_, dtag_;
};

inline int64_t src_off_f(const prb_t *prb, int64_t mb, int64_t m, int64_t k) {
    return (mb * prb->m + m) * prb->k + k;
}

inline int64_t wei_off_f(const prb_t *prb, int64_t mb, int64_t k, int64_t n) {
    return (mb * prb->k + k) * prb->n + n;
}

inline int64_t dst_off_f(const prb_t *prb, int64_t mb, int64_t m, int64_t n) {
    return (mb * prb->m + m) * prb->n + n;
}

void compute_ref(
        const prb_t *prb, dnnl_primitive_t prim_ref, const args_t &args);

int doit(const prb_t *prb, res_t *res);

int bench(int argc, char **argv);

} // namespace matmul

#endif
