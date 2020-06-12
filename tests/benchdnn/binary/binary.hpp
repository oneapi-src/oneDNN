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

#ifndef BINARY_HPP
#define BINARY_HPP

#include <iostream>

#include "dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace binary {

enum alg_t { ADD, MUL, MAX, MIN };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    std::vector<dims_t> sdims;

    std::vector<std::vector<dnnl_data_type_t>> sdt {{dnnl_f32, dnnl_f32}};
    std::vector<dnnl_data_type_t> ddt {dnnl_f32};
    std::vector<std::vector<std::string>> stag {{tag::abx, tag::abx}};
    std::vector<alg_t> alg {ADD};
    std::vector<bool> inplace {false};
    std::vector<attr_t::arg_scales_t> scales {attr_t::arg_scales_t()};
    std::vector<attr_t::post_ops_t> post_ops {attr_t::post_ops_t()};
    attr_t attr = {};

    const char *perf_template_csv
            = "perf,%engine%,%sdt%,%ddt%,%stag%,%alg%,%attr%,%DESC%,%-time%,%"
              "0time%";
    const char *perf_template_def = "perf,%engine%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t {
    prb_t(const std::vector<dims_t> &sdims,
            const std::vector<dnnl_data_type_t> &sdt, dnnl_data_type_t ddt,
            const std::vector<std::string> &stag, alg_t alg, bool inplace,
            const attr_t &attr)
        : sdims(sdims)
        , sdt(sdt)
        , ddt(ddt)
        , stag(stag)
        , alg(alg)
        , inplace(inplace)
        , attr(attr)
        , ndims({(int)sdims[0].size(), (int)sdims[1].size()}) {
        get_broadcast_dims();
    }
    ~prb_t() {}

    std::vector<dims_t> sdims;
    std::vector<dnnl_data_type_t> sdt;
    dnnl_data_type_t ddt;
    std::vector<std::string> stag;
    alg_t alg;
    bool inplace;
    attr_t attr;
    std::vector<int> ndims;

    dims_t broadcast_dims;

    int n_inputs() const { return 2; }

    void get_broadcast_dims() {
        const dims_t &dims_A = this->sdims[0];
        const dims_t &dims_B = this->sdims[1];

        broadcast_dims.resize(ndims[0], 1);
        for (int d = 0; d < ndims[1]; ++d)
            broadcast_dims[d] = dims_A[d] == dims_B[d] ? 0 : 1;
    }
};
std::ostream &operator<<(std::ostream &s, const prb_t &p);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *p, const res_t *r, const char *prb_str) {
        p_ = p;
        base_report(r, prb_str);
    }

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_desc(std::ostream &s) const override { s << p_->sdims; }

    void dump_desc_csv(std::ostream &s) const override { s << p_->sdims; }

    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->sdt;
    }
    const dnnl_data_type_t *ddt() const override { return &p_->ddt; }
    const std::vector<std::string> *stag() const override { return &p_->stag; }

private:
    const prb_t *p_ = NULL;
};

// Returns physical offset for dims_idx based on values from dims.
// E.g. AxBxCxD:Ax1x1x1 problem, for `a:b:c:d` point this should return `a` idx
// for second tensor no matter what `b`, `c` or `d` values are.
inline int64_t dims_off(const dims_t &dims, const dims_t &dims_idx) {
    int64_t nelems = 1;
    for (size_t d = 0; d < dims.size(); ++d)
        nelems *= dims[d];

    int64_t off = 0;
    for (size_t d = 0; d < dims_idx.size(); ++d) {
        if (d < dims.size()) // dims may have less dimensions than dims_idx
            nelems /= dims[d];
        off += (dims_idx[d] * nelems);
    }

    return off;
}

void compute_ref(const prb_t *p, const dnn_mem_t &src0, const dnn_mem_t &src1,
        dnn_mem_t &dst);

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

} // namespace binary

#endif
