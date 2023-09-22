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

#include <tuple>

#include "common.hpp"

#include "utils/compare.hpp"

#include "self/self.hpp"

#include "binary/binary.hpp"
#include "conv/conv.hpp"
#include "eltwise/eltwise.hpp"

namespace self {

template <typename settings_t>
void fill_conv_desc(settings_t &s) {
    auto &d = s.desc;
    d.g = 1;
    d.mb = 2;
    d.ic = 64;
    d.id = 1;
    d.ih = 7;
    d.iw = 7;
    d.oc = 128;
    d.od = 1;
    d.oh = 3;
    d.ow = 3;
    d.kd = 1;
    d.kh = 3;
    d.kw = 3;
    d.sd = 1;
    d.sh = 2;
    d.sw = 2;
    d.pd = 0;
    d.ph = 0;
    d.pw = 0;
    d.pd_r = 0;
    d.ph_r = 0;
    d.pw_r = 0;
    d.dd = 0;
    d.dh = 0;
    d.dw = 0;
    d.has_groups = false;
    d.ndims = 4;
}

template <typename settings_t>
void fill_eltwise_desc(settings_t &s) {
    prb_dims_t &prb_dims = s.prb_dims;
    prb_dims.dims = {2, 128, 3, 3};
    prb_dims.ndims = 4;
    auto &alg = s.alg;
    alg[0] = eltwise::alg_t::RELU;
}

template <typename settings_t>
void fill_binary_desc(settings_t &s) {
    prb_vdims_t &prb_vdims = s.prb_vdims;
    prb_vdims.vdims = {{2, 128, 3, 3}, {1, 128, 1, 1}};
    prb_vdims.ndims = 4;
    prb_vdims.dst_dims = {2, 128, 3, 3};
}

using graph_link_t = std::tuple<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>,
        dnn_mem_map_t, dnn_mem_map_t, args_t, args_t, compare::compare_t>;

void link_args(std::unordered_map<int, graph_link_t> &op_graph) {
    const int op_idx = static_cast<int>(op_graph.size()) - 1;
    if (op_idx == 0) return;

    auto &args_prev = std::get<3>(op_graph[op_idx - 1]);
    auto &ref_args_prev = std::get<4>(op_graph[op_idx - 1]);

    const auto &dst_m = args_prev.find(DNNL_ARG_DST);
    const auto &ref_dst_m = ref_args_prev.find(DNNL_ARG_DST);
    if (has_bench_mode_bit(mode_bit_t::corr) && (!dst_m || !ref_dst_m)) {
        printf("Failed to find prev dst in ref\n");
        exit(1);
    }

    auto &args_cur = std::get<3>(op_graph[op_idx]);
    auto &ref_args_cur = std::get<4>(op_graph[op_idx]);

    args_cur.replace(DNNL_ARG_SRC, &dst_m);
    ref_args_cur.replace(DNNL_ARG_SRC, &ref_dst_m);
}

template <typename settings_t, typename prb_t, typename init_pd_func_t,
        typename supported_exec_args_func_t, typename setup_cmp_func_t,
        /* settings_t here instead */ typename init_desc_t>
int init_op(std::unordered_map<int, graph_link_t> &op_graph,
        const init_pd_func_t &init_pd,
        const supported_exec_args_func_t &supported_exec_args,
        const setup_cmp_func_t &setup_cmp, res_t *res,
        /* settings_t here instead */ const init_desc_t &init_func) {
    int op_idx = static_cast<int>(op_graph.size());
    op_graph.emplace(op_idx,
            std::make_tuple(benchdnn_dnnl_wrapper_t<dnnl_primitive_t>(),
                    dnn_mem_map_t(), dnn_mem_map_t(), args_t(), args_t(),
                    compare::compare_t()));

    // Initialize problem settings.
    settings_t settings {}; // settings(op); - instead.
    init_func(settings);

    // Initializa a problem.
    prb_t prb_(settings), *prb = &prb_;

    // Initialize a primitive.
    auto &prim = std::get<0>(op_graph[op_idx]);
    const_dnnl_memory_desc_t src_md_hint {};
    if (op_idx > 0) {
        auto &prim_prev = std::get<0>(op_graph[op_idx - 1]);
        const auto pd = query_pd(prim_prev);
        src_md_hint = query_md(pd, dnnl_query_dst_md);
    }
    SAFE(create_primitive(prim, get_test_engine(), init_pd, prb, res, prb->dir,
                 /* hint = */ nullptr,
                 /* is_service_prim = */ false, src_md_hint),
            WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    // Initialize memory for the library from a primitive.
    auto &mems = std::get<1>(op_graph[op_idx]);
    auto &ref_mems = std::get<2>(op_graph[op_idx]);

    init_memory_args<prb_t>(mems, prb, prim, supported_exec_args(prb->dir));

    // Initialize reference memories and fill the library memories.
    TIME_FILL(
            SAFE(init_ref_memory_args(ref_mems, mems, prim, prb, res, prb->dir),
                    WARN));

    // Replace empty arguments in op_graph with one that have all memories.
    auto &args = std::get<3>(op_graph[op_idx]) = args_t(mems);
    auto &ref_args = std::get<4>(op_graph[op_idx]) = args_t(ref_mems);
    link_args(op_graph);

    auto &comparator = std::get<5>(op_graph[op_idx]);
    setup_cmp(comparator, prb, DST, ref_args);

    // Execute a primitive.
    SAFE(execute_and_wait(prim, args, res), WARN);

    // Execute reference.
    if (has_bench_mode_bit(mode_bit_t::corr)) { compute_ref(prb, ref_args); }

    return OK;
}

// Note: kinds could be a part of `op_graph` and mark which kinds from which op
// require correctness validation. This is to be configured during `init_op`
// call.
int check_correctness(std::unordered_map<int, graph_link_t> &op_graph,
        const std::vector<data_kind_t> &kinds, res_t *res) {

    // Note: potentially it is possible to execute primitives/reference here and
    // compare side outputs (if any) right away.
    // E.g. bnorm + eltwise fusion, where mean and var will be compared after
    // bnorm execution and dst will be compared after eltwise.
    // TIME_REF(compute_ref(prb, ref_args, prim_ref));

    int op_idx = static_cast<int>(op_graph.size() - 1);
    auto &cmp = std::get<5>(op_graph[op_idx]);

    // Note: need to traverse whole graph to check if eltwise is present.
    bool graph_has_eltwise = true;
    if (graph_has_eltwise) { cmp.set_has_eltwise_post_op(true); }

    for (const auto &kind : kinds) {
        int arg = 0;
        switch (kind) {
            case DST: arg = DNNL_ARG_DST; break;
            case SRC: arg = DNNL_ARG_DIFF_SRC; break;
            case SRC_1: arg = DNNL_ARG_DIFF_SRC_1; break;
            case WEI: arg = DNNL_ARG_DIFF_WEIGHTS; break;
            case BIA: arg = DNNL_ARG_DIFF_BIAS; break;
            case MEAN: arg = DNNL_ARG_MEAN; break;
            case VAR: arg = DNNL_ARG_VARIANCE; break;
            case SC: arg = DNNL_ARG_DIFF_SCALE; break;
            case SH: arg = DNNL_ARG_DIFF_SHIFT; break;
            case DST_ITER: arg = DNNL_ARG_DST_ITER; break;
            case DST_ITER_C: arg = DNNL_ARG_DST_ITER_C; break;
            case AUGRU_ATTENTION: arg = DNNL_ARG_DIFF_AUGRU_ATTENTION; break;
            case SRC_ITER: arg = DNNL_ARG_DIFF_SRC_ITER; break;
            case SRC_ITER_C: arg = DNNL_ARG_DIFF_SRC_ITER_C; break;
            case WEI_ITER: arg = DNNL_ARG_DIFF_WEIGHTS_ITER; break;
            case WEI_PEEPHOLE: arg = DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE; break;
            case WEI_PROJECTION: arg = DNNL_ARG_DIFF_WEIGHTS_PROJECTION; break;
            default: assert(!"unsupported kind"); SAFE_V(FAIL);
        }

        const dnn_mem_t &mem_dt = std::get<3>(op_graph[op_idx]).find(arg);
        const dnn_mem_t &mem_fp = std::get<4>(op_graph[op_idx]).find(arg);
        assert(mem_dt && mem_fp);

        // `mem_fp` is supposed to be primitive reference part. It may have
        // non-f32 data type and non-abx data format, so we reorder `mem_fp` to
        // a golden standard first.
        const auto &mem_fp_md = mem_fp.md_;
        dnn_mem_t mem_fp_golden(
                mem_fp_md, dnnl_f32, tag::abx, get_cpu_engine());
        SAFE(mem_fp_golden.reorder(mem_fp), WARN);

        SAFE(cmp.compare(mem_fp_golden, mem_dt, attr_t(), res), WARN);
    }

    return OK;
}

static int check_graph() {
    res_t res_ {}, *res = &res_;
    std::unordered_map<int, graph_link_t> op_graph;

#define INIT_OP(driver) \
    init_op<driver::settings_t, driver::prb_t>(op_graph, driver::init_pd, \
            driver::supported_exec_args, driver::setup_cmp, res, \
            fill_##driver##_desc<driver::settings_t>)

    INIT_OP(conv);
    INIT_OP(eltwise);
    INIT_OP(binary);

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_correctness(op_graph, {DST}, res), WARN);
    }

    return res->state == FAILED ? FAIL : OK;
}

void graph() {
    RUN(check_graph());
}

} // namespace self
