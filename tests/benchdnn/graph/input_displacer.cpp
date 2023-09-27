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

#include "input_displacer.hpp"
#include "dnnl_common.hpp"
#include "ref_partition.hpp"

namespace graph {

partition_data_displacer_t::partition_data_displacer_t(
        const deserialized_graph &dg, const dnnl::graph::partition &par) {
    const auto &op_ids = par.get_ops();
    const std::unordered_set<size_t> op_ids_set(op_ids.begin(), op_ids.end());

    static const std::unordered_set<std::string> main_op_kind {"Convolution",
            "ConvTranspose", "AvgPool", "MaxPool", "MatMul", "Add", "Divide",
            "Maximum", "Minimum", "Multiply", "Substract"};

    static const std::unordered_set<std::string> go_through_op_kind {
            "StaticTranspose", "StaticReshape", "TypeCast", "Quantize",
            "Dequantize"};

    // dg.ops_ needs make sure its Topo order to first idx, first executed.
    for (const auto &aop : dg.ops_) {
        // Check whether current op is in the partition
        if (op_ids_set.find(aop.id_) == op_ids_set.end()) continue;

        // maintain a map between output tensor id and op
        auto aop_ref = std::ref(aop);
        ops_ref_.emplace_back(aop_ref);
        for (const auto &out_lt : aop.out_lts_) {
            out_lt_2_op_.emplace(out_lt.id_, aop_ref);
        }

        // Try to address which the tensor need displaced and how it will be displaced.

        // Here is how quantize filling work
        //
        //         partition input (lt)    /|
        //                |                 | reverse op filling
        //         [go through op]*         |
        //                |
        //                | <- quantize filling on this tensor (dq_lt)
        //                |
        //           dequantize <- The first dq we met (dq_found)
        //                |
        // [go through op except dq]*  (same as the first input)
        //                 \          /
        //                    main op

        if (main_op_kind.find(aop.kind_) != main_op_kind.end()) {
            // main op found

            // search along the branch for each input of main op
            for (size_t i = 0; i < aop.in_lts_.size(); i++) {
                ::graph::deserialized_lt lt, dq_lt;
                bool dq_found = false;

                for (lt = aop.in_lts_[i];
                        out_lt_2_op_.find(lt.id_) != out_lt_2_op_.end();
                        lt = out_lt_2_op_.at(lt.id_).get().in_lts_[0]) {
                    auto &op = out_lt_2_op_.at(lt.id_);
                    if (op.get().kind_ == "Dequantize" && !dq_found) {
                        // found the first dq
                        dq_lt = op.get().in_lts_[0];
                        dq_found = true;
                    }
                    if (go_through_op_kind.find(op.get().kind_)
                            == go_through_op_kind.end()) {
                        // blocked by other op and fail to continue the search work
                        break;
                    }
                }

                // the partition input found
                if (dq_found
                        && out_lt_2_op_.find(lt.id_) == out_lt_2_op_.end()) {
                    quantize_displace.emplace(
                            lt.id_, ::std::make_tuple(aop, i, dq_lt));
                }
            }
        }
    }
}

int partition_data_displacer_t::displace_input_data(
        size_t lt_id, dnn_mem_t &mem, res_t *res) {

    if (quantize_displace.find(lt_id) == quantize_displace.end()) {
        // no need to displace the data of this tensor
        return OK;
    }
    displace_t displace = quantize_displace.at(lt_id);

    auto main_op = ::std::get<0>(displace);
    auto main_op_offset = ::std::get<1>(displace);
    auto tensor = ::std::get<2>(displace);

    auto opkind = opstr2kind(main_op.kind_);
    int main_op_arg = get_prim_arg_name_from_graph_op_input_offset(
            opkind, main_op_offset);

    dnn_mem_t mem_replace;
    SAFE(gen_quantize_filling(
                 main_op, main_op_arg, mem_replace, tensor.data_type_, res),
            WARN);

    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    // do the reverse job
    while (out_lt_2_op_.find(tensor.id_) != out_lt_2_op_.end()) {
        // generate the reverse op based on OP kind
        // make a copy of deserialized_op to avoid impact on graph execution
        // Currently, we support the following OPs' reverse execution:
        // All of the execution need to swap the input lt and output lt first

        // 1. StaticTranspose: re-permute the 'order' attr to get an inversed effect
        // 2. TypeCast: Do nothing special because the type is already swapped
        // 3. StaticReshape: Do nothing special because the shape is already swapped
        // 4. Quantize: change opkind to Dequantize and keep scales and zps
        // 5. Dequantize: change opkind to Quantize and keep scales and zps

        ::graph::deserialized_op op = out_lt_2_op_.at(tensor.id_);
        ::std::swap(op.in_lts_, op.out_lts_);

        auto opkind = opstr2kind(op.kind_);

        switch (opkind) {
            case ::graph::op::kind::Quantize: op.kind_ = "Dequantize"; break;
            case ::graph::op::kind::Dequantize: op.kind_ = "Quantize"; break;
            case ::graph::op::kind::StaticTranspose: {
                ::std::vector<int64_t> order;
                op.get_attr_s64_vector(order, "order");
                size_t ndims = order.size();
                ::std::vector<int64_t> new_order(ndims, 0);
                for (size_t i = 0; i < ndims; i++) {
                    new_order[(order[i] + ndims) % ndims] = i;
                }
                op.attrs_["order"].s64_vector_ = new_order;
                break;
            }
            case ::graph::op::kind::TypeCast:
            case ::graph::op::kind::StaticReshape: break;
            default:
                assert(!"not support opkind for reverse execution");
                return FAIL;
        }

        // execute the reverse op

        std::unordered_set<size_t> empty_set;
        res_t res {};

        ref_primitive_t ref_prim(op);
        ref_prim.init_prb(empty_set, &res);
        SAFE_V(ref_prim.init_prim(get_cpu_engine(), &res));
        ref_prim.init_memory_args(get_cpu_engine());
        SAFE_V(ref_prim.init_ref_memory_args(get_cpu_engine(), &res));

        // always use the md generated by current reversed op
        // for example
        // matmul op will unsqeeze 1 to fit the dimension
        // so the md generated by matmul prb_t will not be the same as defined in graph

        dnnl_memory_desc_destroy(mem_replace.md_);
        dnnl_memory_desc_clone(
                &mem_replace.md_, ref_prim.get_arg(DNNL_ARG_SRC).md_);
        ref_prim.replace_arg(DNNL_ARG_SRC, mem_replace);
        SAFE_V(ref_prim.execute_prim(&res));
        mem_replace = ::std::move(
                const_cast<dnn_mem_t &>(ref_prim.get_arg(DNNL_ARG_DST)));

        tensor = op.out_lts_[0];
    }

    dnnl_memory_desc_destroy(mem_replace.md_);
    dnnl_memory_desc_clone(&mem_replace.md_, mem.md_);
    SAFE(mem.reorder(mem_replace), WARN);
    return OK;
}

int partition_data_displacer_t::gen_quantize_filling(
        const ::graph::deserialized_op &main_op, int arg, dnn_mem_t &mem,
        const ::std::string &dt, res_t *res) {
    // clone a deserialized op object and modify to specified data type
    ::graph::deserialized_op op = main_op;
    auto driver = opkind2driver(opstr2kind(op.kind_));
    op.in_lts_[0].data_type_ = dt;
    if (op.in_lts_.size() > 1) {
        // matmul/conv/deconv does not support u8u8, replace it with u8s8
        op.in_lts_[1].data_type_
                = ((op.kind_ == "MatMul" || op.kind_ == "Convolution"
                           || op.kind_ == "ConvTranspose")
                          && dt == "u8")
                ? "s8"
                : dt;
    }
    if (driver == dnnl_driver_t::pool || driver == dnnl_driver_t::binary) {
        // pool does not support x8f32 on cpu
        // binary does not support x8x8bf16 on gpu
        // replace output with x8
        op.out_lts_[0].data_type_ = dt;
    } else if (op.out_lts_[0].data_type_ != "bf16") {
        // set output as f32 to avoid the data type not support problem at this stage
        // x8x8bf16 or x8x8f32 is supported for conv/deconv/matmul driver
        op.out_lts_[0].data_type_ = "f32";
    }
    ::std::unordered_set<size_t> empty_set;

    ref_primitive_t ref_prim(op);
    ref_prim.init_prb(empty_set, res);
    if (res->state == INVALID_ARGUMENTS) return FAIL;
    SAFE_V(ref_prim.init_prim(get_cpu_engine(), res));
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    ref_prim.init_memory_args(get_cpu_engine());
    SAFE_V(ref_prim.init_ref_memory_args(get_cpu_engine(), res));
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    mem = ::std::move(const_cast<dnn_mem_t &>(ref_prim.get_arg(arg)));

    return OK;
}

} // namespace graph
