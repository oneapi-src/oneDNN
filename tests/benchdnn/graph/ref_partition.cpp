/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

// ops support bf16 in f32 out.
const std::unordered_set<dnnl_driver_t, driver_hash_t> &
valid_driver_in_bf16_mixed_dt() {
    static const std::unordered_set<dnnl_driver_t, driver_hash_t> set = {
            dnnl_driver_t::reorder,
            dnnl_driver_t::binary,
            dnnl_driver_t::conv,
            dnnl_driver_t::deconv,
            dnnl_driver_t::matmul,
            dnnl_driver_t::resampling,
            dnnl_driver_t::reduction,
    };
    return set;
}

// ops support f32 in bf16 out rewrite.
const std::unordered_set<dnnl_driver_t, driver_hash_t> &
valid_driver_out_bf16_mixed_dt() {
    static const std::unordered_set<dnnl_driver_t, driver_hash_t> set {
            dnnl_driver_t::reorder,
            dnnl_driver_t::binary,
            dnnl_driver_t::resampling,
            dnnl_driver_t::reduction,
            dnnl_driver_t::softmax,
            dnnl_driver_t::bnorm,
            dnnl_driver_t::eltwise,
            dnnl_driver_t::lnorm,
    };
    return set;
}

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
            lt_id_2_lt_.emplace(in_lt.id_, in_lt);
        }
        for (const auto &out_lt : aop.out_lts_) {
            out_lt_2_op_.emplace(out_lt.id_, aop_ref);
            lt_id_2_lt_.emplace(out_lt.id_, out_lt);
        }
    }

    for (const auto &in : ins) {
        partition_in_ids_.emplace_back(in.get_id());
    }
    for (const auto &out : outs) {
        partition_out_ids_.emplace_back(out.get_id());
    }
};

int ref_partition_t::init_ref(const std::vector<size_t> &graph_in_ports,
        partition_mem_map_t &partition_mem_map, res_t *res) {
    handle_typecast_x16();
    handle_special_case_bf16(res);
    for (const auto &par_op_ref : partition_ops_ref_) {
        // res should be independent from op to op
        res->state = UNTESTED;

        auto ref_prim = ::std::make_shared<ref_primitive_t>(par_op_ref.get());

        ref_prims_.emplace(par_op_ref.get().id_, ref_prim);
        SAFE(ref_prim->init_prb(bf16_to_f32_rewrite_lt_id_, res), WARN);

        SAFE_V(ref_prim->init_prim(::get_test_engine(), res));
        ref_prim->init_memory_args(::get_test_engine());
        SAFE_V(ref_prim->init_ref_memory_args(::get_test_engine(), res));

        // store the memory for each logical tensor
        // op `emplace` will keep the first memory it met for each id
        bool use_dst = ::graph::eltwise::get_flag_use_dst_for_bwd_compute(
                par_op_ref);
        for (size_t i = 0; i < par_op_ref.get().in_lts_.size(); i++) {
            const auto &lt = par_op_ref.get().in_lts_[i];
            int arg = get_prim_arg_name_from_graph_op_input_offset(
                    ref_prim->get_kind(), i, use_dst);
            lt_id_2_mems_.emplace(lt.id_, ref_prim->get_arg(arg));
        }
        for (size_t i = 0; i < par_op_ref.get().out_lts_.size(); i++) {
            const auto &lt = par_op_ref.get().out_lts_[i];
            int arg = get_prim_arg_name_from_graph_op_output_offset(
                    ref_prim->get_kind(), i);
            if (arg == 0) {
                fake_lt_ids_.insert(lt.id_);
            } else if (arg > 0) {
                lt_id_2_mems_.emplace(lt.id_, ref_prim->get_arg(arg));
            }
        }

        // Initialze the rest ops if current status is UNTESTED or EXECUTED
        // otherwise there is no need to init memory for the rest ops.
        if (res->state != UNTESTED && res->state != EXECUTED) {
            // But for perf mode, when the tensors in the current op is not
            // the graph in/out, continue, otherwise return.
            if (has_bench_mode_bit(mode_bit_t::perf)) {
                for (const auto &d_lt : par_op_ref.get().in_lts_) {
                    auto iter_find = std::find(graph_in_ports.begin(),
                            graph_in_ports.end(), d_lt.id_);
                    if (iter_find != graph_in_ports.end()) { return FAIL; }
                }
                // If all op ids are not graph inputs, the op failure doesn't
                // affect the perf mode.
                continue;
            } else {
                return FAIL;
            }
        }
    }

    // displace data if needed
    for (const auto &entry : lt_id_2_mems_) {
        SAFE_V(data_displacer.displace_input_data(
                entry.first, const_cast<dnn_mem_t &>(entry.second), res));
    }

    // init graph input/oputput memory from lt_id_2_mems_
    for (const auto &id : partition_in_ids_) {
        if (lt_id_2_mems_.find(id) == lt_id_2_mems_.end()) {
            BENCHDNN_PRINT(0, "Fail: cannot find memory for %zu\n", id);
            res->state = FAILED;
            return FAIL;
        }
        partition_mem_map.emplace(id,
                dnn_graph_mem_t(
                        lt_id_2_mems_.at(id), lt_id_2_lt_.at(id), true));
    }
    for (const auto &id : partition_out_ids_) {
        if (fake_lt_ids_.find(id) != fake_lt_ids_.end()) {
            partition_mem_map.emplace(
                    id, dnn_graph_mem_t({}, lt_id_2_lt_.at(id), false, true));
        } else if (lt_id_2_mems_.find(id) == lt_id_2_mems_.end()) {
            BENCHDNN_PRINT(0, "Fail: cannot find memory for %zu\n", id);
            res->state = FAILED;
            return FAIL;
        } else
            partition_mem_map.emplace(id,
                    dnn_graph_mem_t(
                            lt_id_2_mems_.at(id), lt_id_2_lt_.at(id), false));
    }

    return OK;
}

void ref_partition_t::exec_ops(res_t *res) {
    for (const auto &par_op_ref : partition_ops_ref_) {
        const auto &op = par_op_ref.get();
        auto ref_prim = ref_prims_.at(op.id_);

        // link args && replace the memory before execution
        bool use_dst = ::graph::eltwise::get_flag_use_dst_for_bwd_compute(
                par_op_ref);
        for (size_t i = 0; i < op.in_lts_.size(); i++) {
            const auto &lt = op.in_lts_[i];
            int arg = get_prim_arg_name_from_graph_op_input_offset(
                    ref_prim->get_kind(), i, use_dst);
            ref_prim->replace_arg(arg, lt_id_2_mems_.at(lt.id_));
        }
        for (size_t i = 0; i < op.out_lts_.size(); i++) {
            const auto &lt = op.out_lts_[i];
            // skip replace for fake output tensor
            if (fake_lt_ids_.find(lt.id_) != fake_lt_ids_.end()) continue;
            int arg = get_prim_arg_name_from_graph_op_output_offset(
                    ref_prim->get_kind(), i);
            ref_prim->replace_arg(arg, lt_id_2_mems_.at(lt.id_));
        }
        ref_prim->execute_prim(res);
    }
}

int ref_partition_t::check_partition_correctness(
        partition_mem_map_t &partition_mem_map, res_t *res) {

    bool mistrusted = false, has_eltwise = false, output_has_nans = false;
    const auto &map_kind_to_alg = eltwise::get_eltwise_kind_map();

    for (const auto &op : partition_ops_ref_) {
        size_t op_id = op.get().id_;
        const auto op_kind = op.get().kind_;
        const auto ref_prim = ref_prims_.at(op_id);

        // if there is eltwise post-ops or binary div post-ops (GPU test), need
        // to relax compare critria.
        // Currently, both cases use set_has_eltwise_post_op flag in benchdnn
        // compare function.
        // The flag name is not very accurate, add this note to avoid confusion
        const auto op_driver = opkind2driver(ref_prim->get_kind());
        has_eltwise = has_eltwise
                || ((op_driver == dnnl_driver_t::eltwise
                        || (opstr2kind(op_kind) == dnnl::graph::op::kind::Divide
                                && engine_tgt_kind == dnnl_gpu)));
        output_has_nans = output_has_nans
                || ((map_kind_to_alg.find(op_kind) != map_kind_to_alg.end())
                        && ::eltwise::eltwise_alg_returns_nan_or_inf(
                                map_kind_to_alg.at(op_kind)))
                // `f8_e4m3` range is very short which makes inputs convert
                // into NaNs.
                || (op_driver == dnnl_driver_t::reorder
                        && op.get().in_lts_.front().get_data_type()
                                == logical_tensor::data_type::f8_e4m3);

        // get the args that need comparing
        args_t output_args;
        for (size_t out_idx = 0; out_idx < op.get().out_lts_.size();
                ++out_idx) {
            int out_arg = get_prim_arg_name_from_graph_op_output_offset(
                    opstr2kind(op_kind), out_idx);
            if (out_arg == 0) continue; // unsupported case

            size_t out_lt_id = op.get().out_lts_[out_idx].id_;
            for (size_t i = 0; i < partition_out_ids_.size(); i++) {
                if (out_lt_id != partition_out_ids_[i]) continue;

                auto &graph_mem = partition_mem_map.at(out_lt_id);
                const auto &par_out_mem = graph_mem.get_mem();
                output_args.set(out_arg, par_out_mem);
                break;
            }
        }
        if (output_args.size() == 0) continue;

        // reset the state
        res->state = EXECUTED;

        ref_prim->check_correctness(
                output_args, has_eltwise, output_has_nans, res);
        if (res->state == FAILED) {
            BENCHDNN_PRINT(
                    2, "Op failed: {(%zu) %s}\n", op_id, op_kind.c_str());
            return FAIL;
        }

        mistrusted = mistrusted || (res->state == MISTRUSTED);
    }
    if (res->errors > 0) {
        res->state = FAILED;
    } else if (mistrusted) {
        res->state = MISTRUSTED;
    } else {
        res->state = PASSED;
    }

    return OK;
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
            if (valid_driver_in_bf16_mixed_dt().find(driver_kind)
                    == valid_driver_in_bf16_mixed_dt().end())
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
        if (valid_driver_out_bf16_mixed_dt().find(driver_kind)
                == valid_driver_out_bf16_mixed_dt().end())
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

//     TypeCast (f32->x16)
//        |    ->
//       OP     change all tensors to f32
//        |    ->
//     TypeCast (x16->f32) Stop if find x16->f32 Typecast
void ref_partition_t::handle_typecast_x16() {

    for (const auto &par_op_ref : partition_ops_ref_) {
        bool fail = false;
        const auto &op = par_op_ref.get();
        // check all input/output tensor is bf16 & connected with typecast
        for (const auto &in : op.in_lts_) {
            // ensure:
            // 1. the input is bf16
            // 2. the producer op is typecast
            // 3. current op is the only one consumer
            if (in.data_type_ != "bf16"
                    || out_lt_2_op_.find(in.id_) == out_lt_2_op_.end()
                    || out_lt_2_op_.at(in.id_).get().kind_ != "TypeCast"
                    || in_lt_2_ops_.at(in.id_).size() != 1) {
                fail = true;
                break;
            }
        }
        for (const auto &out : op.out_lts_) {
            // ensure:
            // 1. the output is bf16
            // 2. the consumer op is typecast
            // 3. typecast op is the only one consumer
            if (out.data_type_ != "bf16"
                    || in_lt_2_ops_.find(out.id_) == in_lt_2_ops_.end()
                    || in_lt_2_ops_.at(out.id_).size() != 1
                    || in_lt_2_ops_.at(out.id_).front().get().kind_
                            != "TypeCast") {
                fail = true;
                break;
            }
        }
        if (!fail) {
            for (const auto &in : op.in_lts_) {
                bf16_to_f32_rewrite_lt_id_.insert(in.id_);
            }
            for (const auto &out : op.out_lts_) {
                bf16_to_f32_rewrite_lt_id_.insert(out.id_);
            }
        }
    }
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
