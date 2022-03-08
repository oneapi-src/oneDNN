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
#include <random>

#include "tests/test_thread.hpp"

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "mlp/mlp.hpp"

namespace mlp {

std::ostream &operator<<(std::ostream &s, const mlp_graph_spec_t &spec) {
    dump_global_params(s);
    s << " --cfg=" << spec.cfg;
    s << " --actfunc=" << spec.actfunc;
    s << " --bia_dt=" << benchdnnext::convert_dt(spec.mlp_bias_dt);
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

    build_tensor_desc(spec);

    for (int i = 0; i < spec.num_hidden_layers; i++) {
        std::string i_str = std::to_string(i);
        std::string iplus1_str = std::to_string(i + 1);

        addQuanDequanOp(spec, STRINGIFY(DATA_INT8_) + i_str,
                STRINGIFY(DATA_) + i_str, {1.f},
                {spec.attr.zero_points.get(DNNL_ARG_SRC).value}, false);
        addQuanDequanOp(spec, STRINGIFY(WEI_INT8_) + i_str,
                STRINGIFY(WEI_) + i_str, {spec.attr.oscale.scale},
                {spec.attr.zero_points.get(DNNL_ARG_WEIGHTS).value}, false);
        addMatmulActFuncOp(spec, i);
        addQuanDequanOp(spec, STRINGIFY(DATA_OUT_) + i_str,
                STRINGIFY(DATA_INT8_) + std::to_string(i + 1), {1.f},
                {spec.attr.zero_points.get(DNNL_ARG_DST).value}, true);
    }
    curr_out_map_ids_.assign({TENSOR_ID});
    return fill_status::DONE;
}

void check_known_skipped_case(const mlp_graph_spec_t *spec, res_t *res) {
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
}

inline int fill_data(data_kind_t kind, const dt_conf_t *cfg, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res,
        dnnl_data_type_t sum_dt = dnnl_data_type_undef) {

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    const auto &c = cfg[kind];
    float c_f_min = c.f_min, c_f_max = c.f_max, c_f_scale = c.f_scale;

    if (kind == BIA && mem_dt.dt() == dnnl_u8) c_f_min = 0;

    const bool dst_with_diff_sum_dt = kind == DST
            && sum_dt != dnnl_data_type_undef && sum_dt != mem_dt.dt();
    if (dst_with_diff_sum_dt) {
        mem_dt.set_dt(sum_dt);
        if (sum_dt == dnnl_s8 || sum_dt == dnnl_u8) {
            c_f_min = lowest_dt(sum_dt);
            c_f_max = max_dt(sum_dt);
        }
        if (sum_dt == dnnl_s32) c_f_scale = 1;
    }

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
        std::minstd_rand msr(kind * nelems + idx_start + 1);
        msr.discard(1);

        std::uniform_int_distribution<> gen(c_f_min, c_f_max);

        // make sure the first element is not zero
        if (idx_start == 0) {
            float val = 0;
            while (val == 0)
                val = (float)gen(msr) * c_f_scale;
            mem_fp.set_elem(0, val);
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            auto val = (float)gen(msr) * c_f_scale;
            mem_fp.set_elem(idx, val);
        }
    });

    // work-around mistrusted when A > 0 && B < 0  && C.dt = u8 (or relu)
    if (kind == WEI && nelems == 1 && cfg[DST].dt == dnnl_u8) {
        if (c.f_max >= 1) mem_fp.set_elem(0, c_f_scale);
    }

    SAFE(mem_dt.reorder(mem_fp), WARN);
    if (dst_with_diff_sum_dt) mem_dt.set_dt(cfg[DST].dt);
    return OK;
}

int doit(const mlp_graph_spec_t *spec, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case(spec, res);
    if (res->state == SKIPPED) return OK;

    mlp_graph_prb_t graph_prb(*spec);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }
    auto graph_h = graph_prb.to_graph();
    const auto partitions = graph_h.get_partitions();
    if (partitions.empty()) {
        BENCHDNN_PRINT(0, "FAIL: paritition empty %d.\n", 0);
        return res->state = FAILED, FAIL;
    }

    dnnl::graph::engine &engine = benchdnnext::get_test_engine();
    std::vector<std::vector<dnnl::graph::logical_tensor>> ins_vec, outs_vec;
    std::vector<dnnl::graph::compiled_partition> cp_vec;
    for (int i = 0; i < partitions.size(); i++) {
        const auto par = partitions[i];
        if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

        ins_vec.push_back(par.get_in_ports());
        outs_vec.push_back(par.get_out_ports());
        //sort the in_port based on id
        std::sort(ins_vec[i].begin(), ins_vec[i].end(),
                [](const dnnl::graph::logical_tensor &a,
                        const dnnl::graph::logical_tensor &b) {
                    return a.get_id() < b.get_id();
                });
        //Compile Partitions.
        cp_vec.push_back(par.compile(ins_vec[i], outs_vec[i], engine));
    }

    // prepare memory and physical tensors for each partition
    // tensor descriptors are ordered as follows:
    // input ports : input, weight, bias, weight, bias for single partition
    //               input, weight, bias, input, weight, bias for multi partition
    // output ports: final output for single partition
    //               output, output for multi partition
    std::vector<dnn_mem_t> layer_dt, layer_fp, wei_fp, wei_dt, bias_fp, bias_dt,
            bias_fp_scaled;
    std::vector<std::vector<dnnl ::graph::tensor>> tensors_in, tensors_out;
    for (int i = 0; i < partitions.size(); i++) {
        std::vector<dnnl ::graph::tensor> tensor_in, tensor_out;
        layer_dt.push_back(make_dnn_mem(ins_vec[i][0], spec->raw_data_tag));
        layer_fp.push_back(make_dnn_mem(ins_vec[i][0], dt::f32, tag::abx));
        //Fill memory for first partition input.
        //For remaining partition in multi partition it will be output of
        //previous partition.
        if (i == 0) {
            SAFE(fill_data(SRC, spec->cfg, layer_dt[i], layer_fp[i], res),
                    WARN);
        } else {
            tensor_out.emplace_back(dnnl::graph::tensor(ins_vec[i][0], engine,
                    static_cast<void *>(layer_dt.back())));
            tensors_out.emplace_back(tensor_out);
        }
        tensor_in.emplace_back(dnnl::graph::tensor(
                ins_vec[i][0], engine, static_cast<void *>(layer_dt.back())));
        for (int j = 1; j < ins_vec[i].size();) {
            wei_dt.push_back(make_dnn_mem(ins_vec[i][j], spec->raw_wei_tag));
            wei_fp.push_back(make_dnn_mem(ins_vec[i][j], dt::f32, tag::abx));
            SAFE(fill_data(WEI, spec->cfg, wei_dt.back(), wei_fp.back(), res),
                    WARN);
            tensor_in.emplace_back(dnnl::graph::tensor(ins_vec[i][j++], engine,
                    static_cast<void *>(wei_dt.back())));
            if (spec->has_bias) {
                bias_dt.push_back(make_dnn_mem(ins_vec[i][j], tag::abx));
                bias_fp.push_back(
                        make_dnn_mem(ins_vec[i][j], dt::f32, tag::abx));
                SAFE(fill_data(BIA, spec->cfg, bias_dt.back(), bias_fp.back(),
                             res),
                        WARN);
                if (spec->is_mlp_int8) {
                    bias_fp_scaled.push_back(
                            make_dnn_mem(ins_vec[i][j], dt::f32, tag::abx));
                    scale_bia(bias_fp_scaled.back(), bias_fp.back(),
                            {spec->attr.oscale.scale});
                }

                tensor_in.emplace_back(dnnl::graph::tensor(ins_vec[i][j++],
                        engine, static_cast<void *>(bias_dt.back())));
            }
        }
        tensors_in.emplace_back(tensor_in);
    }
    //prepare intermediate layers for reference for single partition.
    if (partitions.size() == 1) {
        for (int i = 1; i < spec->num_hidden_layers; i++) {
            dnnl::graph::logical_tensor lt(static_cast<size_t>(100 + i),
                    graph_dt::f32, spec->layer_dims[i],
                    dnnl_graph_layout_type_undef);
            layer_fp.push_back(make_dnn_mem(lt, dt::f32, tag::abx));
        }
    }
    //prepare memory for final output
    layer_dt.push_back(make_dnn_mem(
            outs_vec[partitions.size() - 1][0], spec->raw_data_tag));
    layer_fp.push_back(make_dnn_mem(
            outs_vec[partitions.size() - 1][0], dt::f32, tag::abx));
    //populate tensor out for the final partition
    std::vector<dnnl ::graph::tensor> tensor_out;
    tensor_out.emplace_back(
            dnnl::graph::tensor(outs_vec[partitions.size() - 1][0], engine,
                    static_cast<void *>(layer_dt.back())));
    tensors_out.emplace_back(tensor_out);

    for (int i = 0; i < partitions.size(); i++) {
        SAFE(execute_and_wait(cp_vec[i], tensors_in[i], tensors_out[i]), WARN);
    }

    std::vector<args_t> ref_args_vec;
    if (is_bench_mode(CORR)) {
        for (int i = 0; i < layer_fp.size() - 1; i++) {
            args_t ref_args;
            ref_args.set(DNNL_ARG_SRC, layer_fp[i]);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp[i]);
            if (spec->is_mlp_int8) {
                ref_args.set(DNNL_ARG_BIAS, bias_fp_scaled[i]);
            } else {
                ref_args.set(DNNL_ARG_BIAS, bias_fp[i]);
            }
            ref_args.set(DNNL_ARG_DST, layer_fp[i + 1]);
            ref_args_vec.emplace_back(ref_args);
        }

        compute_ref_mlp(spec, ref_args_vec);
        compare::compare_t cmp;
        cmp.set_threshold(spec->cfg[DST].eps);
        cmp.set_data_kind(DST);
        cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?

        SAFE(cmp.compare(layer_fp.back(), layer_dt.back(), spec->attr, res),
                WARN);
    }

    if (is_bench_mode(PERF)) {
        SAFE(measure_perf(res->timer_map.perf_timer(), cp_vec[0], tensors_in[0],
                     tensors_out[0]),
                WARN);
        BENCHDNN_PRINT(1, "partition : %d min: %f avg: %f \n", 0,
                res->timer_map.perf_timer().ms(timer::timer_t::min),
                res->timer_map.perf_timer().ms(timer::timer_t::avg));
        double totms = res->timer_map.perf_timer().ms(timer::timer_t::avg);
        for (int i = 1; i < partitions.size(); i++) {
            timer::timer_t perf_timer;
            SAFE(measure_perf(
                         perf_timer, cp_vec[i], tensors_in[i], tensors_out[i]),
                    WARN);
            res->timer_map.perf_timer().times_ = perf_timer.times_;
            for (int tidx = 0; tidx < timer::timer_t::mode_t::n_modes; tidx++) {
                res->timer_map.perf_timer().ms_[tidx] += perf_timer.ms_[tidx];
                res->timer_map.perf_timer().ticks_[tidx]
                        += perf_timer.ticks_[tidx];
            }
            BENCHDNN_PRINT(1, "partition : %d min: %f avg: %f\n", i,
                    perf_timer.ms(timer::timer_t::min),
                    perf_timer.ms(timer::timer_t::avg));
            totms += perf_timer.ms(timer::timer_t::avg);
        }
        BENCHDNN_PRINT(0, "Total timetaken for %ld partitions: %f\n",
                partitions.size(), totms);
    }
    res->state = PASSED;

    return OK;
}
} // namespace mlp
