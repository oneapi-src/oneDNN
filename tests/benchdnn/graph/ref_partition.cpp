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

#define CASE_HANDLE_LEADING_OP(driver) \
    case dnnl_driver_t::driver: { \
        handle_leading_op<::driver::settings_t, ::driver::prb_t>(leading_op, \
                ::driver::init_pd, ::driver::supported_exec_args, \
                ::driver::setup_cmp, map_off_to_dt, ref_eng, res); \
    } break

#define CASE_HANDLE_OP(driver) \
    case dnnl_driver_t::driver: { \
        handle_op<::driver::settings_t, ::driver::prb_t>(par_op_ref, \
                ::driver::init_pd, ::driver::supported_exec_args, \
                ::driver::setup_cmp, input_ts, output_ts, ref_eng, res); \
    } break

// TODO: re-creating prb object shouldn't happen since somehow it may change,
// and it's additional overhead. It seems reasonable to stash prb as a part
// of `graph_link` object and re-use it here.
#define CASE_CORRECTNESS_CHECK(driver) \
    case dnnl_driver_t::driver: { \
        ::driver::settings_t op_setting = get_setting<::driver::settings_t>( \
                last_op_ref, bf16_to_f32_rewrite_lt_id_, res); \
        ::driver::prb_t prb_(op_setting), *prb = &prb_; \
        check_correctness<::driver::prb_t>(ref_prims_, op_id, \
                partition_output_args_, ref_args, prb, has_eltwise_, \
                output_has_nans_, res); \
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
        const std::vector<dnnl::graph::logical_tensor> &outs) {
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

// find the successive leading op following the dequantize / typecast op
bool get_consumer_leading_op(
        std::reference_wrapper<const deserialized_op> &leading_op,
        int &leading_op_in_offset,
        std::unordered_map<size_t,
                std::list<std::reference_wrapper<const deserialized_op>>>
                &in_lt_2_ops) {
    // find leading OP; dq->[tc]->leading_op
    while (leading_op.get().kind_ == "Dequantize"
            || leading_op.get().kind_ == "TypeCast") {
        // dq/tc only has one output
        const auto &leading_op_out_lt = leading_op.get().out_lts_.front();
        // assume dq only has one consumer
        const auto iter = in_lt_2_ops.find(leading_op_out_lt.id_);
        // only one dequant / dequant+typecast in the partition
        if (iter == in_lt_2_ops.end()) return false;
        leading_op = iter->second.front();

        for (leading_op_in_offset = 0; leading_op_in_offset
                < static_cast<int>(leading_op.get().in_lts_.size());
                leading_op_in_offset++) {
            if (leading_op.get().in_lts_[leading_op_in_offset].id_
                    == leading_op_out_lt.id_)
                break;
        }
    }
    return true;
}

// find the partition leading op based on input lts
bool ref_partition_t::get_leading_op_group(
        std::list<std::reference_wrapper<const deserialized_op>>
                &leading_ops_group) {

    std::unordered_set<size_t> leading_op_ids {};
    const std::unordered_set<std::string> quantized_op {"Convolution",
            "ConvTranspose", "AvgPool", "MaxPool", "MatMul", "Add", "Divide",
            "Maximum", "Minimum", "Multiply", "Substract"};

    for (auto in_id : partition_in_ids_) {
        const auto iter = in_lt_2_ops_.find(in_id);
        auto &leading_op = iter->second.front();

        if (leading_op.get().kind_ != "Dequantize") continue;

        int leading_op_in_offset = 0;
        auto res = get_consumer_leading_op(
                leading_op, leading_op_in_offset, in_lt_2_ops_);
        // only one dequant or dequant+typecast in the partition
        if (!res) return false;

        if (quantized_op.find(leading_op.get().kind_) == quantized_op.end())
            return false;

        if (leading_op_ids.find(leading_op.get().id_) != leading_op_ids.end())
            continue;
        leading_ops_group.emplace_back(std::ref(leading_op));
        leading_op_ids.emplace(leading_op.get().id_);
    }
    return true;
}

// find leading_op's input datatype and save it in map(key:input_offset, val:dt)
void ref_partition_t::get_leading_op_input_offset_to_dt_map(
        const deserialized_op &leading_op,
        std::unordered_map<size_t, const std::string> &map_off_to_dt) {
    size_t offset = 0;
    for (const auto &in_lt_of_leading_op : leading_op.in_lts_) {
        const auto it = out_lt_2_op_.find(in_lt_of_leading_op.id_);
        if (it != out_lt_2_op_.end()) {
            auto &input_op = it->second;
            if (input_op.get().kind_ == "Dequantize") {
                // Dequantize has only one input
                map_off_to_dt.insert(
                        {offset, input_op.get().in_lts_.front().data_type_});
            }
            // TypeCast-->Dequantize
            else if (input_op.get().kind_ == "TypeCast") {
                const auto &in_lc_of_tc = input_op.get().in_lts_.front();
                const auto &in_op_of_tc
                        = out_lt_2_op_.find(in_lc_of_tc.id_)->second;
                map_off_to_dt.insert(
                        {offset, in_op_of_tc.get().in_lts_.front().data_type_});
            }
            // non-int8 input
            else {
                map_off_to_dt.insert({offset, in_lt_of_leading_op.data_type_});
            }
        }
        offset++;
    }
}

void ref_partition_t::run(std::vector<dnnl::graph::tensor> &input_ts,
        std::vector<dnnl::graph::tensor> &output_ts, res_t *res) {

    handle_special_case_bf16(res);
    handle_special_case_int8(res);

    for (const auto &par_op_ref : partition_ops_ref_) {
        const auto op_driver
                = opkind2driver(opstr2kind(par_op_ref.get().kind_));
        const auto &ref_eng = get_ref_engine();

        // if there is eltwise post-ops or binary div post-ops (GPU test), need
        // to relax compare critria.
        // Currently, both cases use set_has_eltwise_post_op flag in benchdnn
        // compare function.
        // The flag name is not very accurate, add this note to avoid confusion
        if (op_driver == dnnl_driver_t::eltwise
                || (opstr2kind(par_op_ref.get().kind_)
                                == dnnl::graph::op::kind::Divide
                        && engine_tgt_kind == dnnl_gpu))
            has_eltwise_ = true;

        const auto &map_kind_to_alg = eltwise::get_eltwise_kind_map();
        if (map_kind_to_alg.find(par_op_ref.get().kind_)
                != map_kind_to_alg.end()) {
            output_has_nans_ = ::eltwise::eltwise_alg_returns_nan_or_inf(
                    map_kind_to_alg.at(par_op_ref.get().kind_));
        }

        switch (op_driver) {
            CASE_HANDLE_OP(binary);
            CASE_HANDLE_OP(bnorm);
            CASE_HANDLE_OP(concat);
            CASE_HANDLE_OP(conv);
            CASE_HANDLE_OP(deconv);
            CASE_HANDLE_OP(eltwise);
            CASE_HANDLE_OP(lnorm);
            CASE_HANDLE_OP(matmul);
            CASE_HANDLE_OP(pool);
            CASE_HANDLE_OP(prelu);
            CASE_HANDLE_OP(reduction);
            CASE_HANDLE_OP(reorder);
            CASE_HANDLE_OP(resampling);
            CASE_HANDLE_OP(softmax);
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

        // find if current logical tensor is the input of the partition
        // and also the input of other preceding ops
        auto iter0 = partition_in_out_mems_.find(cur_op_in_lt.id_);
        if (iter0 != partition_in_out_mems_.end()) {
            std::shared_ptr<dnn_mem_t> in_mem_ptr = iter0->second;
            std::get<3>(ref_prims_[cur_op_id])
                    .replace(cur_op_in_arg, in_mem_ptr.get());
            continue;
        }
        // find if current logical tensor is produced by
        // other ops inside the partition
        auto iter = out_lt_2_op_.find(cur_op_in_lt.id_);
        if (iter == out_lt_2_op_.end()) continue;

        const auto &prev_op = iter->second.get();
        int prev_op_id = static_cast<int>(prev_op.id_);
        size_t prev_op_out_offset = 0;
        for (; prev_op_out_offset < prev_op.out_lts_.size();
                prev_op_out_offset++) {
            if (prev_op.out_lts_[prev_op_out_offset].id_ == cur_op_in_lt.id_)
                break;
        }
        int prev_op_out_arg = get_prim_arg_name_from_graph_op_output_offset(
                opstr2kind(prev_op.kind_), prev_op_out_offset);
        const dnn_mem_t &prev_op_dst_mem
                = std::get<3>(ref_prims_[prev_op_id]).find(prev_op_out_arg);
        const dnn_mem_t &prev_op_ref_dst_mem
                = std::get<4>(ref_prims_[prev_op_id]).find(prev_op_out_arg);

        // link previous op's dst mem to current op's src mem
        std::get<3>(ref_prims_[cur_op_id])
                .replace(cur_op_in_arg, &prev_op_dst_mem);
        std::get<4>(ref_prims_[cur_op_id])
                .replace(cur_op_in_arg, &prev_op_ref_dst_mem);
    }
}

void ref_partition_t::reverse_link_args(
        const deserialized_op &cur_op, res_t *res) {
    std::reference_wrapper<const deserialized_op> leading_op = std::ref(cur_op);
    int leading_op_in_offset = 0;

    auto st = get_consumer_leading_op(
            leading_op, leading_op_in_offset, in_lt_2_ops_);
    if (!st) return;

    int leading_op_id = static_cast<int>(leading_op.get().id_);
    int leading_op_in_arg = get_prim_arg_name_from_graph_op_input_offset(
            opstr2kind(leading_op.get().kind_), leading_op_in_offset,
            eltwise::get_flag_use_dst_for_bwd_compute(cur_op));
    const dnn_mem_t &leading_op_input_mem
            = std::get<3>(ref_prims_[leading_op_id]).find(leading_op_in_arg);
    const dnn_mem_t &leading_op_ref_input_mem
            = std::get<4>(ref_prims_[leading_op_id]).find(leading_op_in_arg);

    // only need to update dq src
    int cur_op_in_arg = DNNL_ARG_SRC;
    int cur_op_id = static_cast<int>(cur_op.id_);

    // move leading op's input mem to current op's input mem
    auto &mems = std::get<1>(ref_prims_[cur_op_id]);
    mems.erase(cur_op_in_arg);
    mems.emplace(cur_op_in_arg,
            std::move(const_cast<dnn_mem_t &>(leading_op_input_mem)));

    // move leading op's input ref mem to current op's input ref mem
    auto &ref_mems = std::get<2>(ref_prims_[cur_op_id]);
    ref_mems.erase(cur_op_in_arg);
    ref_mems.emplace(cur_op_in_arg,
            std::move(const_cast<dnn_mem_t &>(leading_op_ref_input_mem)));

    // Update args for primitive memories
    auto &mem_map = std::get<3>(ref_prims_[cur_op_id]);
    mem_map.clear();
    mem_map = args_t(mems);

    // Update args for reference memories
    auto &ref_mem_map = std::get<4>(ref_prims_[cur_op_id]);
    ref_mem_map.clear();
    ref_mem_map = args_t(ref_mems);
}

void ref_partition_t::init_graph_mem(std::vector<dnnl::graph::tensor> &input_ts,
        std::vector<dnnl::graph::tensor> &output_ts,
        const deserialized_op &cur_op, res_t *res) {
    std::vector<size_t> cur_op_in_ids;
    std::vector<size_t> cur_op_out_ids;
    for (const auto &cur_op_in_lt : cur_op.in_lts_) {
        cur_op_in_ids.emplace_back(cur_op_in_lt.id_);
    }
    for (const auto &cur_op_out_lt : cur_op.out_lts_) {
        cur_op_out_ids.emplace_back(cur_op_out_lt.id_);
    }

    // find if cur_op's input is partition's input
    for (size_t in_idx = 0; in_idx < input_ts.size(); ++in_idx) {
        // current input tensor has already been filled
        if (partition_in_out_mems_.find(partition_in_ids_[in_idx])
                != partition_in_out_mems_.end())
            continue;
        auto iter = std::find(cur_op_in_ids.begin(), cur_op_in_ids.end(),
                partition_in_ids_[in_idx]);
        // partition_in_ids[in_idx] is not the input of cur_op
        if (iter == cur_op_in_ids.end()) continue;
        // partition_in_ids[in_idx] is the input of cur_op
        int cur_op_in_idx = iter - cur_op_in_ids.begin();

        int prim_in_arg_idx = get_prim_arg_name_from_graph_op_input_offset(
                opstr2kind(cur_op.kind_), cur_op_in_idx,
                eltwise::get_flag_use_dst_for_bwd_compute(cur_op));
        dnnl_data_type_t dtype = static_cast<dnnl_data_type_t>(
                cur_op.in_lts_[cur_op_in_idx].get_data_type());
        std::string mtag = strides2memory_tag(
                cur_op.in_lts_[cur_op_in_idx].shape_.size(),
                cur_op.in_lts_[cur_op_in_idx].stride_);

        int cur_op_id = static_cast<int>(cur_op.id_);
        std::shared_ptr<dnn_mem_t> par_in_mem_ptr = std::make_shared<dnn_mem_t>(
                std::get<3>(ref_prims_[cur_op_id]).find(prim_in_arg_idx), dtype,
                mtag, ::get_test_engine());
        input_ts[in_idx].set_data_handle(static_cast<void *>(*par_in_mem_ptr));

        // save the shared_ptr here to avoid early free
        partition_in_out_mems_[partition_in_ids_[in_idx]] = par_in_mem_ptr;
    }

    // find if cur_op's output is partition's output
    for (size_t out_idx = 0; out_idx < output_ts.size(); ++out_idx) {
        auto iter = std::find(cur_op_out_ids.begin(), cur_op_out_ids.end(),
                partition_out_ids_[out_idx]);
        // partition_out_ids[out_idx] is not the output of cur_op
        if (iter == cur_op_out_ids.end()) continue;
        // partition_out_ids[out_idx] is the output of cur_op
        size_t cur_op_out_idx = iter - cur_op_out_ids.begin();

        int prim_out_arg_idx = get_prim_arg_name_from_graph_op_output_offset(
                opstr2kind(cur_op.kind_), cur_op_out_idx);
        if (prim_out_arg_idx == 0) continue; // unsupported case
        // create output dnn_mem using information from out_lt
        dnnl_data_type_t dtype = static_cast<dnnl_data_type_t>(
                cur_op.out_lts_[cur_op_out_idx].get_data_type());
        std::string mtag = strides2memory_tag(
                cur_op.out_lts_[cur_op_out_idx].shape_.size(),
                cur_op.out_lts_[cur_op_out_idx].stride_);
        // dims
        dnnl_dims_t dims;
        std::copy(cur_op.out_lts_[cur_op_out_idx].shape_.begin(),
                cur_op.out_lts_[cur_op_out_idx].shape_.end(), dims);
        int ndims = static_cast<int>(
                cur_op.out_lts_[cur_op_out_idx].shape_.size());
        // create dnn_mem_t
        std::shared_ptr<dnn_mem_t> par_out_mem_ptr
                = std::make_shared<dnn_mem_t>(
                        ndims, dims, dtype, mtag, ::get_test_engine());
        // get data handle and set to partition output tensor
        void *data_handle;
        dnnl_memory_get_data_handle(par_out_mem_ptr->m_, &data_handle);
        output_ts[out_idx].set_data_handle(data_handle);
        // the created dnn_mem_t is mapped (to host), need to unmap (to device)
        partition_output_args_.set(prim_out_arg_idx, *par_out_mem_ptr);
        partition_output_args_.find(prim_out_arg_idx).unmap();
        // save the shared_ptr here to avoid early free
        partition_in_out_mems_[partition_out_ids_[out_idx]] = par_out_mem_ptr;
    }
}

void ref_partition_t::check_partition_correctness(res_t *res) {

    const auto &last_op_ref = partition_ops_ref_.back();
    int op_id = static_cast<int>(last_op_ref.get().id_);
    const auto &ref_args = std::get<3>(ref_prims_[op_id]);
    // map the partition output mem to host
    for (int i = 0; i < partition_output_args_.size(); ++i) {
        partition_output_args_.dnn_mem(i).map();
    }

    const auto op_driver = opkind2driver(opstr2kind(last_op_ref.get().kind_));
    switch (op_driver) {
        CASE_CORRECTNESS_CHECK(binary);
        CASE_CORRECTNESS_CHECK(bnorm);
        CASE_CORRECTNESS_CHECK(concat);
        CASE_CORRECTNESS_CHECK(conv);
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

void ref_partition_t::handle_special_case_int8(res_t *res) {
    std::list<std::reference_wrapper<const deserialized_op>> leading_ops_group;
    is_quantized_ = get_leading_op_group(leading_ops_group);
    if (!is_quantized_) return;

    const auto &ref_eng = get_ref_engine();
    for (const auto &leading_op : leading_ops_group) {
        // deal with single leading op
        std::unordered_map<size_t, const std::string> map_off_to_dt;
        get_leading_op_input_offset_to_dt_map(leading_op.get(), map_off_to_dt);
        const auto op_driver
                = opkind2driver(opstr2kind(leading_op.get().kind_));
        switch (op_driver) {
            CASE_HANDLE_LEADING_OP(conv);
            CASE_HANDLE_LEADING_OP(pool);
            CASE_HANDLE_LEADING_OP(deconv);
            CASE_HANDLE_LEADING_OP(matmul);
            CASE_HANDLE_LEADING_OP(binary);
            default: assert(!"unexpected leading op kind"); break;
        }
    }
}

// Get engine used to run correctness ref path for testing.
// To align with benchdnn, the graph driver always use cpu engine
// for reference except several special cases. For the following 3
// cases we will use gpu engine instead:
// 1. the partition includes reduction op, and the rewrite for intermediate
//    result from bf16 to f32 is not needed
// 2. for logical tensors which use f16 data type
// 3. if the data type is not supported by cpu, the driver will use gpu engine
//    for ref path instead
// TODO:
// > To align with benchdnn, the graph driver always use cpu engine...
// Graph driver is like no other driver. Not all the rules are applicable to it.
// It seems logical to validate GPU Graph against GPU Primitives. The question
// is what prevents to do it in a first place?
const engine_t &ref_partition_t::get_ref_engine() const {

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    const bool has_cpu_bf16_support
            = dnnl::impl::cpu::platform::has_data_type_support(dnnl_bf16);
#else
    const bool has_cpu_bf16_support = false;
#endif
    for (const auto &par_op_ref : partition_ops_ref_) {
        const auto op_driver
                = opkind2driver(opstr2kind(par_op_ref.get().kind_));
        // case
        if (op_driver == dnnl_driver_t::reduction
                && !is_bf16_partition_support_f32_intermediate_result())
            return ::get_test_engine();

        // data type cases:
        // 1. for f16 cases, use gpu engine
        // 2. for platforms that does not support bf16 on cpu, use gpu ref
        // engine instead
        for (auto &in_lt : par_op_ref.get().in_lts_) {
            if (in_lt.data_type_ == "f16"
                    || (in_lt.data_type_ == "bf16" && !has_cpu_bf16_support))
                return ::get_test_engine();
        }
        for (auto &out_lt : par_op_ref.get().out_lts_) {
            if (out_lt.data_type_ == "f16"
                    || (out_lt.data_type_ == "bf16" && !has_cpu_bf16_support))
                return ::get_test_engine();
        }
    }
    return ::get_cpu_engine();
}

} // namespace graph
