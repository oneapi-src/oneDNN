/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#ifndef MLP_HPP
#define MLP_HPP

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "dnnl_graph_common.hpp"
#include "dnnl_graph_common_ext.hpp"
#include "utils/perf_report.hpp"

#include "matmul/matmul.hpp"

namespace mlp {

using namespace matmul;
using namespace benchdnnext;
using graph_dt = dnnl::graph::logical_tensor::data_type;

enum tensor_desc {
    DATA_INT8_, //only for int8
    DATA_,
    DATA_OUT_, //only for int8
    WEI_INT8_,
    WEI_,
    BIA_,
    MATMUL_,
    ACTFUNC_GRAD_, //used for backprop - one per layer
    DATA_TGRAD_, //used for backprop - one per layer
    WEI_TGRAD_, //used for backprop - one per layer
    DATA_GRAD_, //used for backprop - one per layer + 1 input
    WEI_GRAD_, //used for backprop - one per layer
    BIA_GRAD_, //used for backprop - one per layer
};
struct lt_info {
    dnnl::graph::logical_tensor lt;
    data_kind_t data_fill_idx;
    int dt_mem_idx;
    int fp_mem_idx;
};
struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }
    prb_dims_t prb_dims;
    std::vector<dir_t> dir {FWD_I};
    std::vector<std::string> cfg {std::string()};
    std::vector<dnnl_data_type_t> bia_dt {dnnl_data_type_undef};
    std::vector<std::string> stag {tag::abx}, wtag {tag::abx}, dtag {tag::abx};
    std::vector<attr_t::post_ops_t> actfunc;
    std::vector<attr_t::scale_t> scales {attr_t::scale_t()};
    std::vector<attr_t::zero_points_t> zero_points {attr_t::zero_points_t()};

    const char *perf_template_csv
            = "perf,%engine%,%DESC%,"
              "%-time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct mlp_graph_spec_t {
    mlp_graph_spec_t(const prb_dims_t &a_dims, const std::string &wtag,
            const std::string &dtag, const dnnl_data_type_t &bia_dt,
            const std::string cfg, std::vector<attr_t::post_ops_t> actfunc,
            attr_t::scale_t scales, attr_t::zero_points_t zps, dir_t dir)
        : prb_dims(a_dims)
        , cfg(cfg)
        , actfunc(actfunc)
        , raw_data_tag(dtag)
        , raw_wei_tag(wtag)
        , batch_sz(a_dims.dims[0])
        , dir(dir) {
        assert(actfunc[0].entry.size() == prb_dims.ndims - 2);
        num_hidden_layers = prb_dims.ndims - 2;
        for (int i = 1; i < prb_dims.ndims - 1; i++) {
            layer_dims.push_back({batch_sz, prb_dims.dims[i]});
            weight_dims.push_back({prb_dims.dims[i], prb_dims.dims[i + 1]});
            bias_dims.push_back({prb_dims.dims[i + 1]});
            activation_func.push_back(benchdnnext::convert_alg_kind(
                    actfunc[0].entry[i - 1].eltwise.alg));
        }
        layer_dims.push_back({batch_sz, prb_dims.dims[prb_dims.ndims - 1]});
        std::vector<dnnl_data_type_t> dt_vec;
        handle_legacy_cfg(dt_vec, cfg);
        mlp_src_dt = benchdnnext::convert_dt(dt_vec[SRC]);
        mlp_wei_dt = (dt_vec.size() == 1)
                ? mlp_src_dt
                : benchdnnext::convert_dt(dt_vec[WEI]);
        mlp_bias_dt = benchdnnext::convert_dt(bia_dt);
        mlp_dst_dt = (dt_vec.size() == 1)
                ? mlp_src_dt
                : benchdnnext::convert_dt(dt_vec[DST - 1]);
        assert(mlp_src_dt == mlp_dst_dt);
        has_bias = (mlp_bias_dt != graph_dt::undef);
        mlp_layer_dt
                = (mlp_src_dt == graph_dt::bf16) ? mlp_src_dt : graph_dt::f32;
        is_mlp_int8 = mlp_src_dt == graph_dt::s8 || mlp_src_dt == graph_dt::u8;
        attr.insert(actfunc[0]);
        if (is_mlp_int8) {
            attr.insert(scales);
            attr.insert(zps);
        }
        use_dst = (rand() % 2 == 1) ? true : false;
        is_fwd_inference = (dir & FLAG_INF);
        is_fwd_training = (dir & FLAG_FWD) && !(dir & FLAG_INF);
        is_bwd_training = (dir & FLAG_BWD);
    }
    ~mlp_graph_spec_t() {}
    prb_dims_t prb_dims;
    const std::string cfg;
    std::vector<attr_t::post_ops_t> actfunc;
    int num_hidden_layers {0};
    attr_t attr;
    std::string raw_data_tag;
    std::string raw_wei_tag;
    int batch_sz;
    dir_t dir;
    bool use_static_transpose {false};

    bool is_fwd_inference, is_fwd_training, is_bwd_training;
    bool has_bias {false};
    bool use_dst {false};
    std::vector<dnnl::graph::logical_tensor::dims_t> layer_dims, weight_dims,
            bias_dims;
    graph_dt mlp_src_dt, mlp_wei_dt, mlp_bias_dt, mlp_dst_dt, mlp_layer_dt;
    bool is_mlp_int8;
    std::vector<dnnl::graph::op::kind> activation_func;
};

std::ostream &operator<<(std::ostream &s, const mlp_graph_spec_t &spec);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const mlp_graph_spec_t *spec, const char *perf_template)
        : base_perf_report_t(perf_template), spec_(spec) {}
    void dump_desc(std::ostream &s) const override { s << *spec_; }
    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

private:
    const mlp_graph_spec_t *spec_;
};

struct mlp_graph_prb_t : public ::benchdnnext::graph_prb_t {
    mlp_graph_prb_t(const mlp_graph_spec_t &spec) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };
        ctor_status = build_mlp_subgraph(spec);
        if (stop_work(ctor_status)) return;

        ctor_status = fill_status::DONE;
    };
    fill_status_t ctor_status;

    ~mlp_graph_prb_t() {}
    fill_status_t build_mlp_subgraph(const mlp_graph_spec_t &spec);
    std::map<int, struct lt_info> ltid_desc_lut;
    std::map<std::string, int> desc_ltid_lut;

    int get_fp_mem_idx(std::string tensor_name) {
        auto id = desc_ltid_lut[tensor_name];
        return ltid_desc_lut[id].fp_mem_idx;
    }

private:
    void add_quan_dequan_op(const mlp_graph_spec_t &spec, const std::string src,
            const std::string dst, std::vector<float> scales,
            std::vector<int64_t> zps, bool isquanop);
    void add_matmul_op(
            const mlp_graph_spec_t &spec, int layer_num, bool is_fwd_pass);
    void add_actfunc_op(
            const mlp_graph_spec_t &spec, int layer_num, bool is_fwd_pass);
    void add_statictranspose_op(const mlp_graph_spec_t &spec, int layer_num);
    void add_reducesum_op(const mlp_graph_spec_t &spec, int layer_num);
    void add_end_op(const mlp_graph_spec_t &spec, int layer_num);

    void build_tensor_desc_fwd(const mlp_graph_spec_t &spec);
    void build_tensor_desc_bwd(const mlp_graph_spec_t &spec);
};

void compute_ref_mlp(
        const mlp_graph_spec_t *spec, const std::vector<args_t> &args);
int doit(const mlp_graph_spec_t *spec, res_t *res);

int bench(int argc, char **argv);
} // namespace mlp

#endif
