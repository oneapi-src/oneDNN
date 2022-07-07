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
#include <assert.h>
#include <atomic>
#include <random>

#include "tests/test_thread.hpp"

#include "dnnl_graph_common_ext.hpp"
#include "utils/compare.hpp"
#include "utils/parser.hpp"

#include "mlp/mlp.hpp"

namespace mlp {

std::ostream &operator<<(std::ostream &s, const mlp_graph_spec_t &spec) {
    dump_global_params(s);
    s << " --cfg=" << spec.cfg;
    s << " --actfunc=" << spec.actfunc;
    s << " --bia_dt=" << benchdnnext::convert_dt(spec.mlp_bias_dt);
    s << " --dir=" << spec.dir;
    s << " ";
    if (spec.is_mlp_int8) s << " --attr-oscale=" << spec.attr.oscale;
    if (spec.is_mlp_int8) s << " --attr-zero-points=" << spec.attr.zero_points;
    s << " " << spec.prb_dims;
    return s;
}

fill_status_t mlp_graph_prb_t::build_mlp_subgraph(
        const mlp_graph_spec_t &spec) {

    const std::string TENSOR_ID = std::to_string(ops_.size());
    tensor_id["main"].push_back(TENSOR_ID);

    build_tensor_desc_fwd(spec);
    if (spec.is_fwd_training || spec.is_fwd_inference) {
        for (int i = 0; i < spec.num_hidden_layers; i++) {
            std::string i_str = std::to_string(i);
            std::string iplus1_str = std::to_string(i + 1);

            add_quan_dequan_op(spec, STRINGIFY(DATA_INT8_) + i_str,
                    STRINGIFY(DATA_) + i_str, {1.f},
                    {spec.attr.zero_points.get(DNNL_ARG_SRC).value}, false);
            add_quan_dequan_op(spec, STRINGIFY(WEI_INT8_) + i_str,
                    STRINGIFY(WEI_) + i_str, {spec.attr.oscale.scale},
                    {spec.attr.zero_points.get(DNNL_ARG_WEIGHTS).value}, false);
            add_matmul_op(spec, i, true);
            add_actfunc_op(spec, i, true);
            if (spec.is_fwd_training) add_end_op(spec, i);
            add_quan_dequan_op(spec, STRINGIFY(DATA_OUT_) + i_str,
                    STRINGIFY(DATA_INT8_) + std::to_string(i + 1), {1.f},
                    {spec.attr.zero_points.get(DNNL_ARG_DST).value}, true);
        }
    } else {
        build_tensor_desc_bwd(spec);
        for (int i = spec.num_hidden_layers - 1; i >= 0; i--) {
            add_actfunc_op(spec, i, false);
            if (spec.use_static_transpose) { add_statictranspose_op(spec, i); }
            add_matmul_op(spec, i, false);
            add_reducesum_op(spec, i);
        }
    }
    curr_out_map_ids_.assign({TENSOR_ID});
    return fill_status::DONE;
}

void check_known_skipped_case(const mlp_graph_spec_t *spec, res_t *res) {
    if (spec->dir == BWD_D || spec->dir == BWD_W || spec->dir == BWD_WB) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
    if ((spec->mlp_src_dt == graph_dt::f32 || spec->mlp_src_dt == graph_dt::u8)
            && spec->mlp_bias_dt == graph_dt::bf16) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
    if (spec->mlp_src_dt == graph_dt::bf16
            && spec->mlp_bias_dt == graph_dt::f32) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
    if ((spec->is_bwd_training || spec->is_fwd_training)
            && (spec->is_mlp_int8 || is_bench_mode(CORR))) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

inline int fill_data(data_kind_t kind, const mlp_graph_spec_t *spec,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res,
        dnnl_data_type_t sum_dt = dnnl_data_type_undef) {

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    vdims_t strides {vdims_t(STRIDES_SIZE)};
    std::vector<::matmul::dims_mask_t> rt_dims_masks {};
    int bias_mask = 2;
    prb_vdims_t prb_vdims;
    std::string dims_str = dims2str(spec->layer_dims[0]);
    dims_str = dims_str + ":" + dims2str(spec->weight_dims[0]);
    //pick only the activation function for that layer.
    attr_t attr;
    attr.post_ops.entry.push_back(spec->attr.post_ops.entry[0]);
    attr.insert(spec->attr.oscale);
    attr.insert(spec->attr.zero_points);
    float *scales = (float *)zmalloc(sizeof(float), 4);
    SAFE_V(scales != nullptr ? OK : FAIL);
    scales[0] = spec->attr.oscale.scale;
    ::parser::parse_prb_vdims(prb_vdims, dims_str);

    std::vector<dnnl_data_type_t> dt_vec;
    handle_legacy_cfg(dt_vec, spec->cfg);

    ::matmul::prb_t matmul_prb(prb_vdims, dt_vec, spec->raw_data_tag,
            spec->raw_wei_tag, spec->raw_data_tag, strides,
            benchdnnext::convert_dt(spec->mlp_bias_dt), bias_mask,
            rt_dims_masks, attr);
    matmul_prb.scales = scales;
    ::matmul::cfg_t cfg(&matmul_prb, {SRC, WEI, BIA, DST});
    const auto density = cfg.get_density({kind, matmul_prb.k});

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    dnnl::impl::parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::bernoulli_distribution b_dist(density);

        // make sure the first element is positive
        if (idx_start == 0) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            mem_fp.set_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = density == 1.f ? true : b_dist(b_seed);
            float val = is_one * gen(int_seed);
            mem_fp.set_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });

    const bool swap_dt
            = kind == DST && cfg.get_orig_dt(kind) != cfg.get_dt(kind);
    if (swap_dt) mem_dt.set_dt(cfg.get_dt(kind));
    SAFE(mem_dt.reorder(mem_fp), WARN);
    if (swap_dt) mem_dt.set_dt(cfg.get_orig_dt(kind));

    return OK;
}

int doit(const mlp_graph_spec_t *spec, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case(spec, res);
    if (res->state == SKIPPED) return OK;

    mlp_graph_prb_t graph_prb(*spec);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, OK;
    }
    auto graph_h = graph_prb.to_graph();
    const auto partitions = graph_h.get_partitions();
    if (partitions.empty()) {
        BENCHDNN_PRINT(0, "FAIL: paritition empty %d.\n", 0);
        return res->state = FAILED, FAIL;
    }
    BENCHDNN_PRINT(1, "paritition size %ld.\n", partitions.size());

    const dnnl::graph::engine &engine = benchdnnext::get_test_engine();
    std::vector<std::vector<dnnl::graph::logical_tensor>> ins_vec, outs_vec;
    std::vector<dnnl::graph::compiled_partition> cp_vec;

    for (int i = 0; i < partitions.size(); i++) {
        const auto par = partitions[i];
        if (!par.is_supported()) continue;

        ins_vec.push_back(par.get_in_ports());
        outs_vec.push_back(par.get_out_ports());
        //sort the in_port based on id
        std::sort(ins_vec.back().begin(), ins_vec.back().end(),
                [](const dnnl::graph::logical_tensor &a,
                        const dnnl::graph::logical_tensor &b) {
                    return a.get_id() < b.get_id();
                });
        std::sort(outs_vec.back().begin(), outs_vec.back().end(),
                [](const dnnl::graph::logical_tensor &a,
                        const dnnl::graph::logical_tensor &b) {
                    return a.get_id() < b.get_id();
                });
        //Compile Partitions.
        cp_vec.push_back(par.compile(ins_vec.back(), outs_vec.back(), engine));
    }

    // prepare memory and physical tensors for each partition
    // tensor descriptors are ordered as follows:
    // input ports : input, weight, bias, weight, bias for single partition
    //               input, weight, bias, input, weight, bias for multi partition
    // output ports: final output for single partition
    //               output, output for multi partition
    std::vector<dnn_mem_t> mem_dt, mem_fp, bias_fp_scaled;
    std::vector<std::vector<dnnl ::graph::tensor>> tensors_in, tensors_out;
    int mem_idx = 0;
    for (auto lt_vec : ins_vec) {
        std::vector<dnnl ::graph::tensor> tensor_in;
        for (auto lt : lt_vec) {
            auto &lut_info = graph_prb.ltid_desc_lut[lt.get_id()];
            if (lut_info.dt_mem_idx == -1) {
                mem_dt.push_back(make_dnn_mem(lut_info.lt, tag::abx));
                mem_fp.push_back(make_dnn_mem(lut_info.lt, dt::f32, tag::abx));
                SAFE(fill_data(lut_info.data_fill_idx, spec, mem_dt.back(),
                             mem_fp.back(), res),
                        WARN);
                lut_info.fp_mem_idx = mem_idx;
                lut_info.dt_mem_idx = mem_idx++;
            }
            tensor_in.emplace_back(dnnl::graph::tensor(lut_info.lt, engine,
                    static_cast<void *>(mem_dt[lut_info.dt_mem_idx])));
        }
        tensors_in.emplace_back(tensor_in);
    }
    for (auto lt_vec : outs_vec) {
        std::vector<dnnl ::graph::tensor> tensor_out;
        for (auto lt : lt_vec) {
            auto &lut_info = graph_prb.ltid_desc_lut[lt.get_id()];
            if (lut_info.dt_mem_idx == -1) {
                mem_dt.push_back(make_dnn_mem(lut_info.lt, tag::abx));
                mem_fp.push_back(make_dnn_mem(lut_info.lt, dt::f32, tag::abx));
                lut_info.fp_mem_idx = mem_idx;
                lut_info.dt_mem_idx = mem_idx++;
            }
            tensor_out.emplace_back(dnnl::graph::tensor(lut_info.lt, engine,
                    static_cast<void *>(mem_dt[lut_info.dt_mem_idx])));
        }
        tensors_out.emplace_back(tensor_out);
    }
    //execute partitions
    for (int i = 0; i < cp_vec.size(); i++) {
        SAFE(execute_and_wait(cp_vec[i], tensors_in[i], tensors_out[i], res),
                WARN);
    }
    if (is_bench_mode(CORR)) {
        std::vector<args_t> ref_args_vec;
        std::string tensor_name;
        int idx;
        //Prepare datalayer for reference implementation
        for (int i = 1; i < spec->num_hidden_layers; i++) {
            std::string tensor_name = (spec->is_mlp_int8)
                    ? STRINGIFY(DATA_INT8_) + std::to_string(i)
                    : STRINGIFY(DATA_) + std::to_string(i);
            auto id = graph_prb.desc_ltid_lut[tensor_name];
            if (graph_prb.ltid_desc_lut[id].fp_mem_idx == -1) {
                graph_prb.ltid_desc_lut[id].fp_mem_idx = mem_fp.size();
                mem_fp.push_back(make_dnn_mem(
                        graph_prb.ltid_desc_lut[id].lt, dt::f32, tag::abx));
            }
        }
        //prepare scaled bias
        if (spec->is_mlp_int8) {
            for (int i = 0; i < spec->num_hidden_layers; i++) {
                auto id = graph_prb.desc_ltid_lut[STRINGIFY(BIA_)
                        + std::to_string(i)];
                idx = graph_prb.get_fp_mem_idx(
                        STRINGIFY(BIA_) + std::to_string(i));
                bias_fp_scaled.emplace_back(make_dnn_mem(
                        graph_prb.ltid_desc_lut[id].lt, dt::f32, tag::abx));
                scale_bia(bias_fp_scaled[i], mem_fp[idx],
                        {spec->attr.oscale.scale});
            }
        }
        //compute reference
        for (int i = 0; i < spec->num_hidden_layers; i++) {
            args_t ref_args;
            tensor_name = (spec->is_mlp_int8) ? STRINGIFY(DATA_INT8_)
                                              : STRINGIFY(DATA_);
            idx = graph_prb.get_fp_mem_idx(tensor_name + std::to_string(i));
            ref_args.set(DNNL_ARG_SRC, mem_fp[idx]);
            tensor_name = (spec->is_mlp_int8) ? STRINGIFY(WEI_INT8_)
                                              : STRINGIFY(WEI_);
            idx = graph_prb.get_fp_mem_idx(tensor_name + std::to_string(i));
            ref_args.set(DNNL_ARG_WEIGHTS, mem_fp[idx]);
            if (spec->is_mlp_int8) {
                ref_args.set(DNNL_ARG_BIAS, bias_fp_scaled[i]);
            } else {
                idx = graph_prb.get_fp_mem_idx(
                        STRINGIFY(BIA_) + std::to_string(i));
                ref_args.set(DNNL_ARG_BIAS, mem_fp[idx]);
            }

            tensor_name = (spec->is_mlp_int8) ? STRINGIFY(DATA_INT8_)
                                              : STRINGIFY(DATA_);
            idx = graph_prb.get_fp_mem_idx(tensor_name + std::to_string(i + 1));
            ref_args.set(DNNL_ARG_DST, mem_fp[idx]);
            ref_args_vec.emplace_back(ref_args);
        }

        compute_ref_mlp(spec, ref_args_vec);
        compare::compare_t cmp;
        cmp.set_data_kind(DST);
        const auto dt = benchdnnext::convert_dt(spec->mlp_dst_dt);
        const float trh = dt == dnnl_f32 ? 1e-6f : epsilon_dt(dt);
        cmp.set_threshold(trh);
        cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?
        tensor_name = (spec->is_mlp_int8)
                ? STRINGIFY(DATA_INT8_)
                        + std::to_string(spec->num_hidden_layers)
                : STRINGIFY(DATA_) + std::to_string(spec->num_hidden_layers);
        idx = graph_prb.get_fp_mem_idx(tensor_name);
        SAFE(cmp.compare(mem_fp[idx], mem_dt[idx], spec->attr, res), WARN);
    }

    if (is_bench_mode(PERF)) {
        SAFE(measure_perf(res->timer_map.perf_timer(), cp_vec, tensors_in,
                     tensors_out, res),
                WARN);
    }

    //Check for NaN
    for (auto lt_vec : outs_vec) {
        for (auto lt : lt_vec) {
            auto &lut_info = graph_prb.ltid_desc_lut[lt.get_id()];
            std::atomic<int64_t> nzeros(0);
            if (lut_info.dt_mem_idx != -1) {
                const auto check_nan_cnt = [&](int64_t i) {
                    if (std::isnan(mem_dt[lut_info.dt_mem_idx].get_elem(i))) {
                        nzeros++;
                    }
                };
                dnnl::impl::parallel_nd(
                        mem_dt[lut_info.dt_mem_idx].nelems(), check_nan_cnt);
                if (nzeros > 0) {
                    BENCHDNN_PRINT(0, "NAN values in tensor_id %ld cnt: %ld\n",
                            lt.get_id(),
                            nzeros.load(std::memory_order_relaxed));
                    res->state = FAILED;
                    return FAIL;
                }
            }
        }
    }
    res->state = PASSED;
    return OK;
}
} // namespace mlp
