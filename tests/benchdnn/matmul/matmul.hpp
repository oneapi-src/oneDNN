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

#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <algorithm>
#include <bitset>
#include <iostream>
#include <map>
#include <numeric>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/cfg.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace matmul {

typedef std::bitset<DNNL_MAX_NDIMS> dims_mask_t;

const int64_t LD_GOOD = INT64_MAX;
const int64_t LD_NONE = INT64_MAX - 1;

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_vdims_t prb_vdims;

    std::vector<std::string> cfg {std::string()};
    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::string> stag {tag::any}, wtag {tag::any}, dtag {tag::any};
    std::vector<vdims_t> strides {vdims_t(STRIDES_SIZE)};
    std::vector<dnnl_data_type_t> bia_dt {dnnl_data_type_undef};
    std::vector<int> bia_mask {2};
    std::vector<std::vector<dims_mask_t>> rt_dims_masks {{}};

    const char *perf_template_csv() const {
        static const std::string args = "%cfg%,%stag%,%wtag%,%dtag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_vdims_t {
    prb_t(const prb_vdims_t &prb_vdims, const std::vector<dnnl_data_type_t> &dt,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, const vdims_t &strides,
            dnnl_data_type_t bia_dt, int bia_mask,
            const std::vector<dims_mask_t> &rt_dims_masks, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe)
        : prb_vdims_t(prb_vdims)
        , dt(dt)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , strides(strides)
        , bia_dt(bia_dt)
        , bia_mask(bia_mask)
        , rt_dims_masks(rt_dims_masks)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , src_scales(NULL)
        , wei_scales(NULL)
        , dst_scales(NULL) {

        // Broadcast data types if needed
        if (dt.size() == 1) {
            const auto val = dt[0]; // Need a copy here.
            this->dt.assign(3, val);
        }

        this->rt_dims_masks.resize(2);
        const auto &srcdims = src_dims();
        const auto &weidims = weights_dims();
        m = srcdims[ndims - 2];
        k = srcdims.back();
        n = weidims.back();
        dst_dims[ndims - 2] = m;
        dst_dims[ndims - 1] = n;

        init_dst_rt_dims_mask();
        mb = std::accumulate(dst_dims.begin(), dst_dims.end() - 2,
                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
        const auto nelems = std::accumulate(dst_dims.begin(), dst_dims.end(),
                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
        ops = 2. * nelems * k;

        src_scales = generate_scales(DNNL_ARG_SRC);
        wei_scales = generate_scales(DNNL_ARG_WEIGHTS);
        dst_scales = generate_scales(DNNL_ARG_DST);
        src_zp = generate_zero_points(DNNL_ARG_SRC, attr.zero_points, k);
        dst_zp = generate_zero_points(DNNL_ARG_DST, attr.zero_points, n);
    }
    ~prb_t() {
        if (src_scales) zfree(src_scales);
        if (wei_scales) zfree(wei_scales);
        if (dst_scales) zfree(dst_scales);
        if (src_zp) zfree(src_zp);
        if (dst_zp) zfree(dst_zp);
    }

    int64_t m, n, k, mb;
    dir_t dir = FLAG_FWD; // Lack of prop_kind, always considered as forward.
    std::vector<dnnl_data_type_t> dt;
    std::string stag, wtag, dtag;
    vdims_t strides;
    dnnl_data_type_t bia_dt;
    int bia_mask;
    std::vector<dims_mask_t> rt_dims_masks;

    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;

    double ops;
    float *src_scales, *wei_scales, *dst_scales;
    int32_t *src_zp, *dst_zp;

    const dims_t &src_dims() const { return vdims[0]; }
    const dims_t &weights_dims() const { return vdims[1]; }
    dims_t bia_dims() const {
        dims_t dims(ndims, 1);
        for (int d = 0; d < ndims; ++d)
            if (bia_mask & (1 << d)) dims[d] = dst_dims[d];
        return dims;
    }
    // dims_t prb_vdims_t::dst_dims; // A member in `prb_vdims_t`.

    const dims_mask_t &src_runtime_dim_mask() const { return rt_dims_masks[0]; }
    const dims_mask_t &weights_runtime_dim_mask() const {
        return rt_dims_masks[1];
    }
    const dims_mask_t &dst_runtime_dim_mask() const { return rt_dims_masks[2]; }

    int src_broadcast_mask() const {
        return prb_vdims_t::get_broadcast_mask(0);
    }
    int weights_broadcast_mask() const {
        return prb_vdims_t::get_broadcast_mask(1);
    }
    int bias_broadcast_mask() const { return bia_mask; }

    dnnl_data_type_t src_dt() const { return dt[0]; }
    dnnl_data_type_t wei_dt() const { return dt[1]; }
    dnnl_data_type_t dst_dt() const { return dt[2]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

    float *generate_scales(int arg) const;
    int32_t *generate_zero_points(
            int arg, const attr_t::zero_points_t &zero_points, int N) const;

    // Used to construct memory desc when dimensions are runtime since such mds
    // can't be used directly from query and memory objects can't be constructed.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const;

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

std::string str2cfg(const char *str);

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
    cfg_t(const prb_t *prb, std::vector<data_kind_t> kinds) {
        for (const auto kind : kinds) {
            auto orig_data_type = prb->get_dt(kind);
            auto data_type
                    = deduce_cfg_data_type(orig_data_type, prb->attr, kind);
            cfg_entry_.push_back(cfg_entry_t(
                    kind, orig_data_type, data_type, get_cfg_map(kind)));
        }
    }

    const cfg_entry_t::cfg_map_t &get_cfg_map(data_kind_t kind) const;

    float get_density(const density_args_t &density_args) const override;
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

void handle_legacy_cfg(
        std::vector<dnnl_data_type_t> &dt, const std::string &cfg);

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int doit(const prb_t *prb, res_t *res);

int bench(int argc, char **argv);

} // namespace matmul

#endif
