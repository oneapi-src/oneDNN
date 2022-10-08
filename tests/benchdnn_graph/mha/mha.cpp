/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
#include <atomic>
#include <set>

#include "mha/mha.hpp"

namespace mha {

int dt2cfg(graph_dt dt) {
    switch (dt) {
        case graph_dt::f32: return CFG_F32;
        case graph_dt::s8: return CFG_S8;
        case graph_dt::u8: return CFG_U8;
        case graph_dt::bf16: return CFG_BF16;
        default: assert(!"unsupported datatype"); return CFG_DT_UNSUPPORTED;
    }
}

std::ostream &operator<<(std::ostream &s, const mha_graph_spec_t &spec) {
    dump_global_params(s);
    settings_t def;

    if (spec.is_fwd_inference) s << "--pattern=" << spec.mha_pattern << " ";
    s << "--head=" << spec.head << " ";
    s << "--dt=" << convert_dt(spec.mha_inout_dt) << " ";
    s << "--tag=" << spec.tag << " ";
    s << " --dir=" << spec.dir << " ";
    if (!spec.quan_attr.oscale.is_def())
        s << "--attr-quan-oscale=" << spec.quan_attr.oscale << " ";
    if (!spec.quan_attr.zero_points.is_def())
        s << "--attr-quan-zero-points=" << spec.quan_attr.zero_points << " ";
    if (!spec.dequan_attr.oscale.is_def())
        s << "--attr-dequan-oscale=" << spec.dequan_attr.oscale << " ";
    if (!spec.dequan_attr.zero_points.is_def())
        s << "--attr-dequan-zero-points=" << spec.dequan_attr.zero_points
          << " ";
    s << spec.dims[0] << "x" << spec.dims[1] << "x" << spec.dims[2];

    return s;
}

void mha_graph_spec_t::generate_scales() {
    const attr_t::scale_t &oscale = quan_attr.oscale;
    int mask = attr_t::get_default_mask(oscale.policy);

    int64_t uniq_scales = 1;
    for (int d = 0; d < this->ndims; ++d)
        if (mask & (1 << d)) uniq_scales *= this->dims[d];

    for (int d = 0; d < uniq_scales; ++d)
        quan_scales.emplace_back(oscale.scale);
    if (uniq_scales > 1) quan_scales[uniq_scales - 1] /= 2.f;

    const attr_t::scale_t &dq_oscale = dequan_attr.oscale;
    mask = attr_t::get_default_mask(dq_oscale.policy);

    uniq_scales = 1;
    for (int d = 0; d < this->ndims; ++d)
        if (mask & (1 << d)) uniq_scales *= this->dims[d];

    for (int d = 0; d < uniq_scales; ++d)
        dequan_scales.emplace_back(dq_oscale.scale);
    if (uniq_scales > 1) dequan_scales[uniq_scales - 1] /= 2.f;
}

//TODO: zero_points policy per_tensor only supported.
//No per_channel support.
void mha_graph_spec_t::generate_zero_points() {
    const attr_t::zero_points_t &zero_points = quan_attr.zero_points;
    const auto &q_e = zero_points.get(DNNL_ARG_DST);
    assert(q_e.policy == policy_t::COMMON);
    quan_zps.emplace_back(q_e.value);
    const attr_t::zero_points_t &dq_zero_points = dequan_attr.zero_points;
    const auto &dq_e = dq_zero_points.get(DNNL_ARG_SRC);
    assert(dq_e.policy == policy_t::COMMON);
    dequan_zps.emplace_back(dq_e.value);
}

inline int fill_data(
        graph_dt dt, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    const auto &c = cfg[dt2cfg(dt)];
    float c_f_min = c.f_min, c_f_max = c.f_max, c_f_scale = c.f_scale;

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
        std::minstd_rand msr(nelems + idx_start + 1);
        msr.discard(1);

        std::uniform_int_distribution<> gen(c_f_min, c_f_max);

        // make sure the first element is not zero
        if (idx_start == 0) {
            float val = 0;
            while (val == 0)
                val = static_cast<float>(gen(msr));
            mem_fp.set_elem(0, val * c_f_scale);
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            auto val = static_cast<float>(gen(msr)) * c_f_scale;
            mem_fp.set_elem(idx, val);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

fill_status_t mha_graph_prb_t::build_mha_subgraph_fwd(
        const mha_graph_spec_t &spec) {
    using op = dnnl::graph::op;

    dims_t qkv_shape = {
            spec.dims[0], spec.dims[1], spec.head, (spec.dims[2] / spec.head)};

    //attributes
    dims_t qkv_transpose_axis = {0, 2, 1, 3};
    dims_t k2_transpose_axis = {0, 1, 3, 2};

    size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    build_tensor_desc_fwd(spec);

    if (spec.mha_pattern == "v1") {
        add_dequan_op(spec, STRINGIFY(QINT8_SRC), STRINGIFY(QSRC));
        add_dequan_op(spec, STRINGIFY(KINT8_SRC), STRINGIFY(KSRC));
        add_dequan_op(spec, STRINGIFY(VINT8_SRC), STRINGIFY(VSRC));
        add_staticreshape_op(
                spec, STRINGIFY(QSRC), STRINGIFY(QTSRC), qkv_shape);
        add_staticreshape_op(
                spec, STRINGIFY(KSRC), STRINGIFY(KTSRC), qkv_shape);
        add_staticreshape_op(
                spec, STRINGIFY(VSRC), STRINGIFY(VTSRC), qkv_shape);
        add_statictranspose_op(
                spec, STRINGIFY(QTSRC), STRINGIFY(QDST), qkv_transpose_axis);
        add_statictranspose_op(
                spec, STRINGIFY(KTSRC), STRINGIFY(KT2SRC), qkv_transpose_axis);
        add_statictranspose_op(
                spec, STRINGIFY(KT2SRC), STRINGIFY(KDST), k2_transpose_axis);
        add_statictranspose_op(
                spec, STRINGIFY(VTSRC), STRINGIFY(VDST), qkv_transpose_axis);
    } else if (spec.mha_pattern == "v2") {
        add_dequan_op(spec, STRINGIFY(QINT8_SRC), STRINGIFY(QDST));
        add_dequan_op(spec, STRINGIFY(KINT8_SRC), STRINGIFY(KDST));
        add_dequan_op(spec, STRINGIFY(VINT8_SRC), STRINGIFY(VDST));
        //v3 pattern only s8/u8 so output will be from dequan op
    } else if (spec.mha_pattern == "v3") {
        add_dequan_op(spec, STRINGIFY(QINT8_SRC), STRINGIFY(QSRC));
        add_dequan_op(spec, STRINGIFY(KINT8_SRC), STRINGIFY(KSRC));
        add_dequan_op(spec, STRINGIFY(VINT8_SRC), STRINGIFY(VSRC));
        add_typecast_op(STRINGIFY(QSRC), STRINGIFY(QDST));
        add_typecast_op(STRINGIFY(KSRC), STRINGIFY(KDST));
        add_typecast_op(STRINGIFY(VSRC), STRINGIFY(VDST));
    }

    //matmul Q.KT
    add_matmul_op(false, spec.is_fwd_training, STRINGIFY(QDST), STRINGIFY(KDST),
            STRINGIFY(QKMATMULDST));
    //score
    add_arith_op(true, op::kind::Divide, true, STRINGIFY(QKMATMULDST),
            STRINGIFY(QKDIVSCALE), STRINGIFY(QKDIVDST));
    //add
    add_arith_op(true, op::kind::Add, true, STRINGIFY(QKDIVDST),
            STRINGIFY(QKATTN), STRINGIFY(QKADD));
    //softmax
    new_op_id = ops_.size();
    ops_.emplace_back(
            op(new_op_id, op::kind::SoftMax, {tensor_descs_[STRINGIFY(QKADD)]},
                    {tensor_descs_[STRINGIFY(QKSOFTMAX)]}, "SCORE_softmax"));
    ops_[new_op_id].set_attr("axis", static_cast<int64_t>(3));
    if (spec.is_fwd_training) add_end_op(STRINGIFY(QKSOFTMAX));
    //For INT8 quantize and dequantize the softmax output.
    //For int8-bf16 there are typecast before and after quantize ops
    std::string quan_tensor = STRINGIFY(QKSOFTMAX);
    if (spec.mha_pattern == "v3") {
        add_typecast_op(STRINGIFY(QKSOFTMAX), STRINGIFY(QKSOFTMAXQUANCAST));
        quan_tensor = STRINGIFY(QKSOFTMAXQUANCAST);
    }
    add_quan_op(spec, quan_tensor, STRINGIFY(QKSOFTMAXQUAN));
    add_dequan_op(spec, STRINGIFY(QKSOFTMAXQUAN), STRINGIFY(QKSOFTMAXDEQUAN));
    //dropout for training
    add_arith_op(spec.is_fwd_training, op::kind::Multiply, false, quan_tensor,
            STRINGIFY(QKDROPOUT), STRINGIFY(QKSOFTMAXDEQUAN));
    if (spec.is_fwd_training) add_end_op(STRINGIFY(QKSOFTMAXDEQUAN));
    std::string matmul_qkvtensor = (spec.MHA_int8 || spec.is_fwd_training)
            ? STRINGIFY(QKSOFTMAXDEQUAN)
            : STRINGIFY(QKSOFTMAX);
    if (spec.mha_pattern == "v3") {
        add_typecast_op(matmul_qkvtensor, STRINGIFY(QKSOFTMAXDEQUANCAST));
        matmul_qkvtensor = STRINGIFY(QKSOFTMAXDEQUANCAST);
    }

    //QKV matmul
    add_matmul_op(false, false, matmul_qkvtensor, STRINGIFY(VDST),
            STRINGIFY(QKVMATMUL));
    //QKV transpose
    add_statictranspose_op(
            spec, STRINGIFY(QKVMATMUL), STRINGIFY(QKVTDST), qkv_transpose_axis);

    if (spec.mha_pattern == "v1" || spec.is_fwd_training) {
        add_staticreshape_op(
                spec, STRINGIFY(QKVTDST), STRINGIFY(QKVDST), spec.dims);
    } else if (spec.mha_pattern == "v2") {
        add_reorder_op(STRINGIFY(QKVTDST), STRINGIFY(QKVDST));
    } else if (spec.mha_pattern == "v3") {
        add_reorder_op(STRINGIFY(QKVTDST), STRINGIFY(QKVCASTDST));

        add_typecast_op(STRINGIFY(QKVCASTDST), STRINGIFY(QKVDST));
    } else {
        return fill_status::UNHANDLED_CONFIG_OPTIONS;
    }

    add_quan_op(spec, STRINGIFY(QKVDST), STRINGIFY(QKVINT8DST));
    curr_out_map_ids_.assign({TENSOR_ID});
    return fill_status::DONE;
}

fill_status_t mha_graph_prb_t::build_mha_subgraph_bwd(
        const mha_graph_spec_t &spec) {
    if (!spec.is_bwd_training) return fill_status::DONE;
    dims_t qkv_shape = {
            spec.dims[0], spec.dims[1], spec.head, (spec.dims[2] / spec.head)};
    //attributes
    dims_t qkv_transpose_axis = {0, 2, 1, 3};
    build_tensor_desc_bwd(spec);
    add_staticreshape_op(
            spec, STRINGIFY(GRADIN), STRINGIFY(GRADTIN), qkv_shape);
    add_statictranspose_op(
            spec, STRINGIFY(GRADTIN), STRINGIFY(GRADTOUT), qkv_transpose_axis);
    add_matmul_op(true, false, STRINGIFY(QKSOFTMAXDEQUAN), STRINGIFY(GRADTOUT),
            STRINGIFY(GRADVWEI));
    add_matmul_op(false, true, STRINGIFY(GRADTOUT), STRINGIFY(VDST),
            STRINGIFY(GRADVDATA));
    add_arith_op(spec.is_bwd_training, dnnl::graph::op::kind::Multiply, true,
            STRINGIFY(GRADVDATA), STRINGIFY(QKDROPOUT),
            STRINGIFY(GRADMUL1DATA));
    add_arith_op(spec.is_bwd_training, dnnl::graph::op::kind::Multiply, true,
            STRINGIFY(QKSOFTMAX), STRINGIFY(GRADMUL1DATA),
            STRINGIFY(GRADMUL2DATA));
    add_reducesum_op(spec, STRINGIFY(GRADMUL2DATA), STRINGIFY(GRADSOFTMAXSUM));
    add_arith_op(spec.is_bwd_training, dnnl::graph::op::kind::Subtract, true,
            STRINGIFY(GRADMUL1DATA), STRINGIFY(GRADSOFTMAXSUM),
            STRINGIFY(GRADSOFTMAXSUB));
    add_arith_op(spec.is_bwd_training, dnnl::graph::op::kind::Multiply, true,
            STRINGIFY(QKSOFTMAX), STRINGIFY(GRADSOFTMAXSUB),
            STRINGIFY(GRADMUL3DATA));
    add_arith_op(spec.is_bwd_training, dnnl::graph::op::kind::Divide, true,
            STRINGIFY(GRADMUL3DATA), STRINGIFY(QKDIVSCALE),
            STRINGIFY(GRADDIVDATA));
    add_matmul_op(false, false, STRINGIFY(GRADDIVDATA), STRINGIFY(KDST),
            STRINGIFY(GRADQWEI));
    add_matmul_op(true, false, STRINGIFY(GRADDIVDATA), STRINGIFY(QDST),
            STRINGIFY(GRADKWEI));
    return fill_status::DONE;
}

void check_known_skipped_case(const mha_graph_spec_t *spec, res_t *res) {
    if (spec->dir == BWD_D || spec->dir == BWD_W || spec->dir == BWD_WB) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
    if (is_bench_mode(CORR)) {
        BENCHDNN_PRINT(0, "Please run for performance %d\n", 0);
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
    if (spec->is_fwd_inference) {
        if ((spec->mha_pattern == "")
                || (spec->mha_pattern != "v1" && spec->mha_pattern != "v2"
                        && spec->mha_pattern != "v3")) {
            res->state = SKIPPED, res->reason = INVALID_CASE;
        }
        if ((spec->mha_pattern == "v3")
                && (spec->mha_inout_dt == graph_dt::f32
                        || spec->mha_inout_dt == graph_dt::bf16)) {
            res->state = SKIPPED, res->reason = INVALID_CASE;
        }
    }

    if (spec->head > spec->dims[2]) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
    }
    if ((((int)(spec->dims[2] / spec->head)) * spec->head) != spec->dims[2]) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
    }
    if (spec->quan_attr.oscale.policy != attr_t::policy_t::COMMON
            || spec->dequan_attr.oscale.policy != attr_t::policy_t::COMMON) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
}

int doit(const mha_graph_spec_t *spec, res_t *res) {

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case(spec, res);
    if (res->state == SKIPPED) return OK;

    mha_graph_prb_t graph_prb(*spec);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, OK;
    }
    auto graph_h = graph_prb.to_graph();
    const auto partitions = graph_h.get_partitions();

    if (partitions.empty()) return res->state = FAILED, FAIL;
    //this is done to exit for bf16 cases
    if (spec->mha_inout_dt == graph_dt::bf16) {
        if ((spec->dir == FWD_I || spec->dir == BWD_DW)
                && partitions.size() != 1)
            return res->state = UNIMPLEMENTED, OK;
        if (spec->dir == FWD_D && partitions.size() > 3)
            return res->state = UNIMPLEMENTED, OK;
    }

    const dnnl::graph::engine &engine = benchdnnext::get_test_engine();
    std::vector<std::vector<dnnl::graph::logical_tensor>> ins_vec, outs_vec;
    std::vector<dnnl::graph::compiled_partition> cp_vec;
    for (size_t i = 0; i < partitions.size(); i++) {
        const auto par = partitions[i];
        if (!par.is_supported()) continue;

        ins_vec.push_back(par.get_in_ports());
        outs_vec.push_back(par.get_out_ports());
        //sort the in_port based on id
        std::sort(ins_vec[i].begin(), ins_vec[i].end(),
                [](const dnnl::graph::logical_tensor &a,
                        const dnnl::graph::logical_tensor &b) {
                    return a.get_id() < b.get_id();
                });
        //sort the out_port based on id
        std::sort(outs_vec[i].begin(), outs_vec[i].end(),
                [](const dnnl::graph::logical_tensor &a,
                        const dnnl::graph::logical_tensor &b) {
                    return a.get_id() < b.get_id();
                });
        //Compile Partitions.
        cp_vec.push_back(par.compile(ins_vec[i], outs_vec[i], engine));
    }

    // prepare memory and physical tensors for each partition
    std::vector<dnn_mem_t> mem_dt, mem_fp;
    std::vector<std::vector<dnnl ::graph::tensor>> tensors_in, tensors_out;

    float scale = sqrt(spec->dims[2] / spec->head);
    //number of elements = bs * seq_len
    std::vector<int> attention_mask_filter(spec->dims[0] * spec->dims[1], 1);

    for (auto lt_vec : ins_vec) {
        std::vector<dnnl ::graph::tensor> tensor_in;
        for (auto lt : lt_vec) {
            auto &lut_info = graph_prb.ltid_desc_lut[lt.get_id()];
            std::set<std::string> skipfillmem = {"QKDIVSCALE", "QKATTN"};
            if (lut_info.dt_mem_idx == -1
                    && skipfillmem.find(lut_info.name) == skipfillmem.end()) {
                int mem_idx = mem_dt.size();
                mem_dt.push_back(make_dnn_mem(lut_info.lt, tag::abx));
                mem_fp.push_back(make_dnn_mem(lut_info.lt, dt::f32, tag::abx));
                SAFE(fill_data(spec->mha_dt, mem_dt.back(), mem_fp.back(), res),
                        WARN);
                lut_info.fp_mem_idx = mem_idx;
                lut_info.dt_mem_idx = mem_idx;
            }
            if (lut_info.name == "QKDIVSCALE") {
                tensor_in.emplace_back(dnnl::graph::tensor(
                        lut_info.lt, engine, static_cast<void *>(&scale)));
            } else if (lut_info.name == "QKATTN") {
                tensor_in.emplace_back(dnnl::graph::tensor(lut_info.lt, engine,
                        static_cast<void *>(attention_mask_filter.data())));
            } else {
                tensor_in.emplace_back(dnnl::graph::tensor(lut_info.lt, engine,
                        static_cast<void *>(mem_dt.back())));
            }
        }
        tensors_in.emplace_back(tensor_in);
    }
    for (auto lt_vec : outs_vec) {
        std::vector<dnnl ::graph::tensor> tensor_out;
        for (auto lt : lt_vec) {
            auto &lut_info = graph_prb.ltid_desc_lut[lt.get_id()];
            if (lut_info.dt_mem_idx == -1) {
                int mem_idx = mem_dt.size();
                mem_dt.push_back(make_dnn_mem(lut_info.lt, tag::abx));
                mem_fp.push_back(make_dnn_mem(lut_info.lt, dt::f32, tag::abx));
                lut_info.fp_mem_idx = mem_idx;
                lut_info.dt_mem_idx = mem_idx;
            }
            tensor_out.emplace_back(dnnl::graph::tensor(lut_info.lt, engine,
                    static_cast<void *>(mem_dt[lut_info.dt_mem_idx])));
        }
        tensors_out.emplace_back(tensor_out);
    }
    //execute partitions
    assert(cp_vec.size() == 1);
    SAFE(execute_and_wait(cp_vec[0], tensors_in[0], tensors_out[0], res), WARN);
    SAFE(measure_perf(res->timer_map.perf_timer(), cp_vec[0], tensors_in[0],
                 tensors_out[0]),
            WARN);
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
} // namespace mha
