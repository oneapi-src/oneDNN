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

#include "ref_partition.hpp"
#include "cpu/platform.hpp"
#include "dnnl_common.hpp"
#include "utils/compare.hpp"

namespace graph {

class driver_hash_t {
public:
    std::size_t operator()(const dnnl_driver_t dnnl_driver) const {
        return std::hash<int>()(static_cast<int>(dnnl_driver));
    }
};

#define CASE_INIT_OP(driver) \
    case dnnl_driver_t::driver: { \
        init_op<::driver::settings_t, ::driver::prb_t>(par_op_ref, \
                ::driver::init_pd, ::driver::supported_exec_args, \
                ::driver::setup_cmp, partition_mem_map, ::get_test_engine(), \
                res); \
    } break
#define CASE_EXEC_OP(driver) \
    case dnnl_driver_t::driver: { \
        int op_id = static_cast<int>(par_op_ref.get().id_); \
        const ::driver::prb_t *prb \
                = std::get<5>(ref_prims_[op_id])->get<::driver::prb_t>(); \
        exec_op<::driver::prb_t>(par_op_ref, prb, res); \
    } break

#define CASE_CORRECTNESS_CHECK(driver) \
    case dnnl_driver_t::driver: { \
        const ::driver::prb_t *prb \
                = std::get<5>(ref_prims_[op_id])->get<::driver::prb_t>(); \
        check_correctness<::driver::prb_t>(ref_prims_, op_id, output_args, \
                ref_args, prb, has_eltwise, output_has_nans, res); \
    } break

// ops support bf16 in f32 out.
std::unordered_set<dnnl_driver_t, driver_hash_t> valid_driver_in_bf16_mixed_dt {
        dnnl_driver_t::reorder,
        dnnl_driver_t::binary,
        dnnl_driver_t::conv,
        dnnl_driver_t::deconv,
        dnnl_driver_t::matmul,
        dnnl_driver_t::resampling,
        dnnl_driver_t::reduction,
};

// ops support f32 in bf16 out rewrite.
std::unordered_set<dnnl_driver_t, driver_hash_t>
        valid_driver_out_bf16_mixed_dt {
                dnnl_driver_t::reorder,
                dnnl_driver_t::binary,
                dnnl_driver_t::resampling,
                dnnl_driver_t::reduction,
                dnnl_driver_t::softmax,
                dnnl_driver_t::bnorm,
                dnnl_driver_t::eltwise,
                dnnl_driver_t::lnorm,
        };

ref_partition_t::ref_partition_t(const deserialized_graph &dg,
        const dnnl::graph::partition &par,
        const std::vector<dnnl::graph::logical_tensor> &ins,
        const std::vector<dnnl::graph::logical_tensor> &outs)
    : data_displacer(dg, par) {
    const auto &op_ids = par.get_ops();
    const std::unordered_set<size_t> op_ids_set(op_ids.begin(), op_ids.end());

    // dg.ops_ needs make sure its Topo order to first idx, first executed.
    for (const auto &aop : dg.ops_) {
        if (op_ids_set.find(aop.id_) == op_ids_set.end()) continue;

        auto aop_ref = std::ref(aop);
        partition_ops_ref_.emplace_back(aop_ref);
        for (const auto &in_lt : aop.in_lts_) {
            in_lt_2_ops_[in_lt.id_].emplace_back(aop_ref);
        }
        for (const auto &out_lt : aop.out_lts_) {
            out_lt_2_op_.emplace(out_lt.id_, aop_ref);
        }
    }

    for (const auto &in : ins) {
        partition_in_ids_.emplace_back(in.get_id());
    }
    for (const auto &out : outs) {
        partition_out_ids_.emplace_back(out.get_id());
    }
};

void ref_partition_t::init_ref(const bench_mode_t mode,
        const std::vector<size_t> &graph_in_ports,
        partition_mem_map_t &partition_mem_map, res_t *res) {

    handle_special_case_bf16(res);
    for (const auto &par_op_ref : partition_ops_ref_) {
        // res should be independent from op to op
        res->state = UNTESTED;
        const auto op_driver
                = opkind2driver(opstr2kind(par_op_ref.get().kind_));
        switch (op_driver) {
            case dnnl_driver_t::custom: {
                // since custom driver does not generate real primitive
                // we need to skip the init primitve and execute primitve steps in handle_op
                // we extract some implementation from handle_op which we must do
                int op_id = static_cast<int>(par_op_ref.get().id_);
                auto op_setting = ::graph::custom::get_setting(
                        par_op_ref, bf16_to_f32_rewrite_lt_id_, res);
                if (res->state == INVALID_ARGUMENTS) return;
                auto pprb = std::make_shared<::custom::prb_t>(op_setting);
                ::custom::prb_t *prb = pprb.get();
                ref_prims_[op_id] = std::make_tuple(
                        benchdnn_dnnl_wrapper_t<dnnl_primitive_t>(),
                        dnn_mem_map_t(), dnn_mem_map_t(), args_t(), args_t(),
                        std::make_shared<prb_wrapper_t<::custom::prb_t>>(pprb));
                ::custom::init_memory_args(std::get<1>(ref_prims_[op_id]), prb,
                        ::custom::supported_exec_args(prb));
                ::custom::init_ref_memory_args(std::get<2>(ref_prims_[op_id]),
                        std::get<1>(ref_prims_[op_id]), prb, res);

                ::std::get<3>(ref_prims_[op_id])
                        = args_t(std::get<1>(ref_prims_[op_id]));
                ::std::get<4>(ref_prims_[op_id])
                        = args_t(std::get<2>(ref_prims_[op_id]));

                // displace the input tensor
                for (size_t i = 0; i < par_op_ref.get().in_lts_.size(); i++) {
                    const auto &in = par_op_ref.get().in_lts_[i];
                    int arg = get_prim_arg_name_from_graph_op_input_offset(
                            opstr2kind(par_op_ref.get().kind_),
                            static_cast<int>(i));
                    auto &mem = const_cast<dnn_mem_t &>(
                            ::std::get<3>(ref_prims_[op_id]).find(arg));
                    data_displacer.displace_input_data(in.id_, mem, res);
                    if (res->state == SKIPPED || res->state == UNIMPLEMENTED)
                        return;
                }
                link_args(par_op_ref, res);
                init_graph_memory_args(std::get<1>(ref_prims_[op_id]),
                        partition_mem_map, partition_in_ids_,
                        partition_out_ids_, par_op_ref, res);
                break;
            }
                CASE_INIT_OP(binary);
                CASE_INIT_OP(bnorm);
                CASE_INIT_OP(concat);
                CASE_INIT_OP(conv);
                CASE_INIT_OP(deconv);
                CASE_INIT_OP(eltwise);
                CASE_INIT_OP(lnorm);
                CASE_INIT_OP(matmul);
                CASE_INIT_OP(pool);
                CASE_INIT_OP(prelu);
                CASE_INIT_OP(reduction);
                CASE_INIT_OP(reorder);
                CASE_INIT_OP(resampling);
                CASE_INIT_OP(softmax);
            default: break;
        }
        // Initialze the rest ops if current status is UNTESTED or EXECUTED
        // otherwise there is no need to init memory or the rest ops
        if (res->state != UNTESTED && res->state != EXECUTED) {
            // But for perf mode, when the tensors in the current op is not
            // the graph in/out, continue, otherwise return.
            if (mode == bench_mode_t::perf) {
                for (const auto &d_lt : par_op_ref.get().in_lts_) {
                    auto iter_find = std::find(graph_in_ports.begin(),
                            graph_in_ports.end(), d_lt.id_);
                    if (iter_find != graph_in_ports.end()) { return; }
                }
                // if all op ids are not graph inputs, this op failure wont affect perf mode.
                continue;
            } else {
                return;
            }
        }
    }
}

void ref_partition_t::exec_ops(res_t *res) {
    for (const auto &par_op_ref : partition_ops_ref_) {
        const auto op_driver
                = opkind2driver(opstr2kind(par_op_ref.get().kind_));

        switch (op_driver) {
            case dnnl_driver_t::custom: {
                int op_id = static_cast<int>(par_op_ref.get().id_);
                const ::custom::prb_t *prb = std::get<5>(ref_prims_[op_id])
                                                     ->get<::custom::prb_t>();
                ::custom::execute(prb, ::std::get<3>(ref_prims_[op_id]), res);
                break;
            }
                CASE_EXEC_OP(binary);
                CASE_EXEC_OP(bnorm);
                CASE_EXEC_OP(concat);
                CASE_EXEC_OP(conv);
                CASE_EXEC_OP(deconv);
                CASE_EXEC_OP(eltwise);
                CASE_EXEC_OP(lnorm);
                CASE_EXEC_OP(matmul);
                CASE_EXEC_OP(pool);
                CASE_EXEC_OP(prelu);
                CASE_EXEC_OP(reduction);
                CASE_EXEC_OP(reorder);
                CASE_EXEC_OP(resampling);
                CASE_EXEC_OP(softmax);
            default: break;
        }
    }
}

void ref_partition_t::link_args(const deserialized_op &cur_op, res_t *res) {
    for (int cur_op_in_offset = 0;
            cur_op_in_offset < static_cast<int>(cur_op.in_lts_.size());
            ++cur_op_in_offset) {
        int cur_op_id = static_cast<int>(cur_op.id_);
        const auto &cur_op_in_lt = cur_op.in_lts_[cur_op_in_offset];
        int cur_op_in_arg = get_prim_arg_name_from_graph_op_input_offset(
                opstr2kind(cur_op.kind_), cur_op_in_offset,
                eltwise::get_flag_use_dst_for_bwd_compute(cur_op));

        // find if current logical tensor is produced by
        // other ops inside the partition
        auto iter = out_lt_2_op_.find(cur_op_in_lt.id_);

        int prev_op_id, prev_op_arg;
        if (iter == out_lt_2_op_.end()) {
            // current logical tensor is partition input
            // check whether it is a shared input
            auto prev_op = this->in_lt_2_ops_[cur_op_in_lt.id_].front().get();
            if (prev_op.id_ != cur_op.id_) {
                // first handled op using this logical tensor is not current op
                // need to replace the args from the first args
                prev_op_id = static_cast<int>(prev_op.id_);
                size_t prev_op_in_offset = 0;
                for (; prev_op_in_offset < prev_op.in_lts_.size();
                        prev_op_in_offset++) {
                    if (prev_op.in_lts_[prev_op_in_offset].id_
                            == cur_op_in_lt.id_)
                        break;
                }
                prev_op_arg = get_prim_arg_name_from_graph_op_input_offset(
                        opstr2kind(prev_op.kind_),
                        static_cast<int>(prev_op_in_offset));
            } else {
                continue;
            }
        } else {
            // current logical tensor is produced by
            // other ops inside the partition
            const auto &prev_op = iter->second.get();
            prev_op_id = static_cast<int>(prev_op.id_);
            size_t prev_op_out_offset = 0;
            for (; prev_op_out_offset < prev_op.out_lts_.size();
                    prev_op_out_offset++) {
                if (prev_op.out_lts_[prev_op_out_offset].id_
                        == cur_op_in_lt.id_)
                    break;
            }
            prev_op_arg = get_prim_arg_name_from_graph_op_output_offset(
                    opstr2kind(prev_op.kind_), prev_op_out_offset);
        }
        const dnn_mem_t &prev_op_mem
                = std::get<3>(ref_prims_[prev_op_id]).find(prev_op_arg);
        const dnn_mem_t &prev_op_ref_mem
                = std::get<4>(ref_prims_[prev_op_id]).find(prev_op_arg);
        // link previous op's mem to current op's mem
        std::get<3>(ref_prims_[cur_op_id]).replace(cur_op_in_arg, &prev_op_mem);
        std::get<4>(ref_prims_[cur_op_id])
                .replace(cur_op_in_arg, &prev_op_ref_mem);
    }
}

void ref_partition_t::check_partition_correctness(
        partition_mem_map_t &partition_mem_map, res_t *res) {

    size_t errors = 0, total = 0;
    bool mistrusted = false, has_eltwise = false, output_has_nans = false;
    const auto &map_kind_to_alg = eltwise::get_eltwise_kind_map();

    for (auto op : partition_ops_ref_) {
        int op_id = static_cast<int>(op.get().id_);

        // if there is eltwise post-ops or binary div post-ops (GPU test), need
        // to relax compare critria.
        // Currently, both cases use set_has_eltwise_post_op flag in benchdnn
        // compare function.
        // The flag name is not very accurate, add this note to avoid confusion
        const auto op_driver = opkind2driver(opstr2kind(op.get().kind_));
        has_eltwise = has_eltwise
                || ((op_driver == dnnl_driver_t::eltwise
                        || (opstr2kind(op.get().kind_)
                                        == dnnl::graph::op::kind::Divide
                                && engine_tgt_kind == dnnl_gpu)));
        output_has_nans = output_has_nans
                || ((map_kind_to_alg.find(op.get().kind_)
                            != map_kind_to_alg.end())
                        && ::eltwise::eltwise_alg_returns_nan_or_inf(
                                map_kind_to_alg.at(op.get().kind_)));

        const auto &ref_args = std::get<3>(ref_prims_[op_id]);

        // get the args that need comparing
        args_t output_args;
        for (size_t out_idx = 0; out_idx < op.get().out_lts_.size();
                ++out_idx) {
            int out_arg = get_prim_arg_name_from_graph_op_output_offset(
                    opstr2kind(op.get().kind_), out_idx);
            if (out_arg == 0) continue; // unsupported case
            size_t out_lt_id = op.get().out_lts_[out_idx].id_;
            for (size_t i = 0; i < partition_out_ids_.size(); i++) {
                if (out_lt_id == partition_out_ids_[i]) {
                    auto &graph_mem = partition_mem_map.at(out_lt_id);
                    const auto &par_out_mem = graph_mem.get_mem();
                    output_args.set(out_arg, par_out_mem);
                    break;
                }
            }
        }
        if (output_args.size() == 0) continue;

        // reset the state
        res->state = EXECUTED;

        switch (op_driver) {
            CASE_CORRECTNESS_CHECK(binary);
            CASE_CORRECTNESS_CHECK(bnorm);
            CASE_CORRECTNESS_CHECK(concat);
            CASE_CORRECTNESS_CHECK(conv);
            CASE_CORRECTNESS_CHECK(custom);
            CASE_CORRECTNESS_CHECK(deconv);
            CASE_CORRECTNESS_CHECK(eltwise);
            CASE_CORRECTNESS_CHECK(lnorm);
            CASE_CORRECTNESS_CHECK(matmul);
            CASE_CORRECTNESS_CHECK(pool);
            CASE_CORRECTNESS_CHECK(prelu);
            CASE_CORRECTNESS_CHECK(reduction);
            CASE_CORRECTNESS_CHECK(reorder);
            CASE_CORRECTNESS_CHECK(resampling);
            CASE_CORRECTNESS_CHECK(softmax);
            default: assert(!"Unsupported driver"); break;
        }
        // accumulate error count and reset the counter
        errors += res->errors;
        total += res->total;
        mistrusted = mistrusted || (res->state == MISTRUSTED);
        res->errors = 0;
        res->total = 0;
    }
    res->errors = errors;
    res->total = total;
    if (res->errors > 0) {
        res->state = FAILED;
    } else if (mistrusted) {
        res->state = MISTRUSTED;
    } else {
        res->state = PASSED;
    }
}

bool ref_partition_t::check_valid_bf16_in() const {
    std::unordered_set<size_t> rewritable_in_ops, input_ops;
    for (const auto &lt_id : partition_in_ids_) {
        const auto iter = in_lt_2_ops_.find(lt_id);
        const auto &consumer_ops = iter->second;
        for (const auto &op : consumer_ops) {
            // if the op already meets the requirement, skip the check
            if (rewritable_in_ops.find(op.get().id_) != rewritable_in_ops.end())
                continue;
            // record all input ops for comparison
            if (input_ops.find(op.get().id_) == input_ops.end())
                input_ops.emplace(op.get().id_);

            const dnnl_driver_t driver_kind
                    = opkind2driver(opstr2kind(op.get().kind_));
            // exclude ops which doesn't support bf16 in f32 out
            // if one input op is not supported, this feature will not be enabled
            if (valid_driver_in_bf16_mixed_dt.find(driver_kind)
                    == valid_driver_in_bf16_mixed_dt.end())
                return false;
            for (const auto &lt : op.get().in_lts_) {
                if (lt.id_ == lt_id && lt.data_type_ == "bf16") {
                    // if current op has a bf16 input and support bf16 to f32,
                    // add it to rewritable ops
                    rewritable_in_ops.emplace(op.get().id_);
                }
            }
        }
    }
    // at least one input op support bf16 to f32 transformation
    return !rewritable_in_ops.empty();
}

bool ref_partition_t::check_valid_bf16_out() const {
    for (const auto &lt_id : partition_out_ids_) {
        const auto iter = out_lt_2_op_.find(lt_id);
        const auto &producer_op = iter->second;
        // check if the op which produces the partition output supports f32 in bf16 out.
        const dnnl_driver_t driver_kind
                = opkind2driver(opstr2kind(producer_op.get().kind_));
        if (valid_driver_out_bf16_mixed_dt.find(driver_kind)
                == valid_driver_out_bf16_mixed_dt.end())
            return false;
        for (const auto &lt : producer_op.get().out_lts_) {
            // all output lts should be bf16
            if (lt.id_ == lt_id && lt.data_type_ != "bf16") { return false; }
        }
    }
    return true;
}

// dnnl partition with at least one bf16 in and one bf16 out
bool ref_partition_t::is_bf16_partition_support_f32_intermediate_result()
        const {

    // only rewrite the parition with at least one validate op with bf16 in and
    // all ops with bf16 out
    if (partition_ops_ref_.size() <= 1 || !check_valid_bf16_in()
            || !check_valid_bf16_out()) {
        return false;
    }

    // only works for one leading op situations
    size_t leading_op_num = 0;
    for (const auto &aop : partition_ops_ref_) {
        const dnnl_driver_t driver = opkind2driver(opstr2kind(aop.get().kind_));
        if (driver != dnnl_driver_t::binary && driver != dnnl_driver_t::eltwise
                && driver != dnnl_driver_t::reorder) {
            leading_op_num++;
            // currently only support one leading op
            if (leading_op_num > 1) { return false; }
        }
    }

    return true;
}

void ref_partition_t::handle_special_case_bf16(res_t *res) {
    if (!is_bf16_partition_support_f32_intermediate_result()) return;

    // 1. find lt which is intermediate lt;
    // 2. check producer supports bf16 output and comsumer supports bf16 input
    // 3. set the logical tensor as f32
    for_(const auto &par_op_ref : partition_ops_ref_)
    for (const auto &out_lt : par_op_ref.get().out_lts_) {
        auto iter_consumers = in_lt_2_ops_.find(out_lt.id_);
        if (iter_consumers != in_lt_2_ops_.end()) {
            // record the id of logical tensors that need rewriting
            bf16_to_f32_rewrite_lt_id_.insert(out_lt.id_);
        }
    }
}

} // namespace graph
