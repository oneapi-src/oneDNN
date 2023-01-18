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

#ifndef BENCHDNN_GRAPH_BRIDGE_HPP
#define BENCHDNN_GRAPH_BRIDGE_HPP

#include <tuple>
#include "common.hpp"
#include "deserialize.hpp"
#include "dnnl_common.hpp"
#include <type_traits>
#include <unordered_map>

#include "setting_handler.hpp"
#include "utils/compare.hpp"
#include "utils/settings.hpp"
namespace graph {

// TODO: the proposal is to extend the link with tensor memories. There are two
// rationale:
// 1. This will allow to remove input_md_process calls along the way for each
//    test case with the price of some memory increase.
// 2. It reduces ref_partition state by removing `partition_in_out_mems_` and
//    `partition_output_args_`. It also removes a necessity to remember that
//    those objects exist just to prolong memory life time since ref_prims_
//    object will own them till the end.
// 3. This will force to have a separate abstraction that will sort out
//    functionality of reorders from dnn_mem_t with its handle to tensor and
//    back.
using graph_link_t = std::tuple<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>,
        dnn_mem_map_t, dnn_mem_map_t, args_t, args_t>;

#define DECLARE_SET_PRB_CFG(driver) \
    template <typename prb_t, \
            requires<std::is_same<prb_t, ::driver::prb_t>::value> = true> \
    void set_prb_cfg(prb_t *prb, \
            const std::unordered_map<size_t, const std::string> \
                    &map_off_to_dt, \
            res_t *res) { \
        driver::set_s8u8_for_prb(prb, map_off_to_dt, res); \
    }

#define DECLARE_TEMPLATE_GET_SETTING(driver) \
    template <typename setting_t, \
            requires<std::is_same<setting_t, \
                    ::driver::settings_t>::value> = true> \
    setting_t get_setting(const deserialized_op &base_op_ref, \
            const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) { \
        return driver::get_setting(base_op_ref, rewrite_lt_ids, res); \
    }

DECLARE_SET_PRB_CFG(conv);
DECLARE_SET_PRB_CFG(deconv);
DECLARE_SET_PRB_CFG(matmul);
DECLARE_SET_PRB_CFG(binary);
DECLARE_SET_PRB_CFG(pool);

// template to generate driver settings
DECLARE_TEMPLATE_GET_SETTING(binary);
DECLARE_TEMPLATE_GET_SETTING(bnorm);
DECLARE_TEMPLATE_GET_SETTING(concat);
DECLARE_TEMPLATE_GET_SETTING(conv);
DECLARE_TEMPLATE_GET_SETTING(deconv);
DECLARE_TEMPLATE_GET_SETTING(eltwise);
DECLARE_TEMPLATE_GET_SETTING(lnorm);
DECLARE_TEMPLATE_GET_SETTING(matmul);
DECLARE_TEMPLATE_GET_SETTING(pool);
DECLARE_TEMPLATE_GET_SETTING(prelu);
DECLARE_TEMPLATE_GET_SETTING(reduction);
DECLARE_TEMPLATE_GET_SETTING(reorder);
DECLARE_TEMPLATE_GET_SETTING(resampling);
DECLARE_TEMPLATE_GET_SETTING(softmax);

// Special backward ops which need forward data when execute
const std::unordered_set<std::string> &get_special_backward_op_kind_set();

// Get map from DNNL_ARG to data_kind
const std::unordered_map<size_t, data_kind_t> &get_dnnl_arg_2_data_kind_map();

// Apart from oneDNN memory tag, oneDNN Graph has op attributes `data_format`
// (NXC/NCX) and `weights_format`(OIX/IOX/XOI/XIO) to indicate the order of
// dimensions.
//
// For example, tensor with shape [1,4,4,3] and NXC data_format means
// batch_size=1, spacial_dims=4x4, channel_dim=3. This semantic info is
// necessary for some scenarios, i.e. per_channel quantization.
//
// As the order of dimensions in oneDNN memory descriptor is always NC[D[H[W]]],
// to align with oneDNN, the memory descriptor of oneDNN Graph should be
// permuted to NCX for primitive creating and executing.
//
// In this case, memory permutation is introduced to avoid memory copy between
// graph and primitive paths, and ensure the memory is in NCX format for
// primitive creating and executing.
//
// `input_md_process` function is designed to be called after the
// initialization and before the execution of primitives to permute the
// memories of input logical tensors. It takes:
// * Memory references, will permute the desciptors based on op kind and format
// * Reference of the deseralized graph op
// * Flag that indicates whether it's in init stage or not
// * A pointer to a `res_t` structure to update validation status.
//
// For op with NXC format, the function does:
// 1. is_init = true:
//      1.1 for NXC ops, permute the input memory descriptor after primitive
//      init for that graph can acquire the memory in proper format
//      1.2 for some op, first reshape weight from GOIX to OIX, then permute
//      weight for XIO/XOI/IOX cases
//      1.3 exchange last two dims for matmul if the transpose attribute is true
// 2. is_init = false:
//      2.1 restore the input memory descriptor from NXC to NCX before primitive
//      execution for correctness
//      2.2 for some op, first permute weight to OIX for XIO/XOI/IOX cases,
//      then reshape weight format to GOIX
//      2.3 exchange last two dims for matmul if the transpose attribute is true
int input_md_process(dnn_mem_map_t &mems, const deserialized_op &base_op_ref,
        const bool is_init, res_t *res);

// `output_md_process` function acts same as `input_md_process` but for output
// tensors.
//
// For op with NXC format, the function does:
// 1. permute the output memory descriptor to NXC after primitive execution, in
// order that following ops in graph can acquire the memory in correct format.
// 2. for grouped ops, reshape output logical tesnor from GOIX to OIX
// 3. permute weight for XIO/XOI/IOX cases
// 4. for staticTranspose cases, reshape output according to order attribute
int output_md_process(
        dnn_mem_map_t &mems, const deserialized_op &base_op_ref, res_t *res);

template <typename prb_t, typename init_pd_func_t,
        typename supported_exec_args_func_t, typename setup_cmp_func_t>
int init_prim(std::unordered_map<int, graph_link_t> &ref_prims,
        const deserialized_op &base_op_ref, const init_pd_func_t &init_pd,
        const supported_exec_args_func_t &supported_exec_args,
        const setup_cmp_func_t &setup_cmp, const prb_t *prb,
        const engine_t &ref_eng, res_t *res) {
    int op_id = static_cast<int>(base_op_ref.id_);
    ref_prims[op_id]
            = std::make_tuple(benchdnn_dnnl_wrapper_t<dnnl_primitive_t>(),
                    dnn_mem_map_t(), dnn_mem_map_t(), args_t(), args_t());

    // Initialize a primitive
    auto &prim = std::get<0>(ref_prims[op_id]);
    auto &mems = std::get<1>(ref_prims[op_id]);
    auto &ref_mems = std::get<2>(ref_prims[op_id]);

    bool is_service_prim = prb->dir & FLAG_BWD;
    const auto &set_special_backward_op_kind
            = get_special_backward_op_kind_set();
    bool is_special_bwd_ops
            = set_special_backward_op_kind.find(base_op_ref.kind_)
            != set_special_backward_op_kind.end();

    // Special backward ops should do forward execution first
    if (is_special_bwd_ops) {
        SAFE(create_primitive(prim, ref_eng, init_pd, prb, res, FLAG_FWD,
                     /* hint = */ nullptr, is_service_prim,
                     /* src_md_hint = */ nullptr),
                WARN);
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

        // Initialize memory for the library from a primitive.
        init_memory_args<prb_t>(
                mems, prb, prim, supported_exec_args(FLAG_FWD), ref_eng);

        // Initialize reference memories and fill the library memories.
        SAFE(init_ref_memory_args(ref_mems, mems, prim, prb, res, FLAG_FWD),
                WARN);

        std::get<3>(ref_prims[op_id]) = args_t(mems);
        std::get<4>(ref_prims[op_id]) = args_t(ref_mems);
        SAFE(execute_and_wait(prim, std::get<3>(ref_prims[op_id]), res), WARN);
    }
    // General init flow
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> cur_prim;
    SAFE(create_primitive(cur_prim, ref_eng, init_pd, prb, res, prb->dir,
                 /* hint = */ is_special_bwd_ops ? query_pd(prim) : nullptr,
                 /* is_service_prim = */ false, /* src_md_hint = */ nullptr),
            WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    prim.reset(cur_prim.release());

    // Pass same memory map as we need data from forward on backward.
    init_memory_args<prb_t>(
            mems, prb, prim, supported_exec_args(prb->dir), ref_eng);

    // Initialize reference memories and fill the library memories.
    SAFE(init_ref_memory_args(ref_mems, mems, prim, prb, res, prb->dir), WARN);

    // Permute memory descriptors of input logical tensors to NCX format
    SAFE(input_md_process(mems, base_op_ref, /* is_init_stage = */ true, res),
            WARN);

    std::get<3>(ref_prims[op_id]) = args_t(mems);
    std::get<4>(ref_prims[op_id]) = args_t(ref_mems);

    return OK;
}

template <typename prb_t>
int execute_prim(std::unordered_map<int, graph_link_t> &ref_prims,
        const deserialized_op &base_op_ref, const prb_t *prb, res_t *res) {
    int op_id = static_cast<int>(base_op_ref.id_);
    auto &prim = std::get<0>(ref_prims[op_id]);
    auto &mems = std::get<1>(ref_prims[op_id]);
    auto &args = std::get<3>(ref_prims[op_id]);

    // restore the memory for NXC cases
    SAFE(input_md_process(mems, base_op_ref, /* is_init_stage = */ false, res),
            WARN);

    // Execute a primitive.
    SAFE(execute_and_wait(prim, args, res), WARN);

    // process output logical tensor md for the following ops
    SAFE(output_md_process(mems, base_op_ref, res), WARN);

    return OK;
}

template <typename prb_t>
void check_correctness(std::unordered_map<int, graph_link_t> &ref_prims,
        size_t op_id, const args_t &args, const args_t &ref_args,
        const prb_t *prb, bool has_eltwise, bool output_has_nans, res_t *res) {
    // `args` contain only output tensors from graph execution.
    for (int idx = 0; idx < args.size(); ++idx) {
        check_zero_padding(args.dnn_mem(idx), args.arg(idx), res);
        check_buffer_overwrite(args.dnn_mem(idx), args.arg(idx), res);

        const auto arg = args.arg(idx);
        const auto &mem_dt = args.find(arg);
        const auto &mem_fp = ref_args.find(arg);
        const auto &dnnl_arg_2_data_kind = get_dnnl_arg_2_data_kind_map();

        compare::compare_t cmp;
        if (dnnl_arg_2_data_kind.find(arg) != dnnl_arg_2_data_kind.end())
            setup_cmp(cmp, prb, dnnl_arg_2_data_kind.at(arg), ref_args);
        else {
            BENCHDNN_PRINT(1, "Output arg %d is unsupported!\n", arg);
            res->state = UNIMPLEMENTED;
            return;
        }

        // TODO: these two conditions are now part of ref_partition abstraction.
        // The proposal is to create cmp objects at init_prim stage for each op
        // and update correspondent values there. Traverse through all of them
        // here and prepare a final cmp object based on flags from all of them.
        // This allows to extend checks scalable and reduce the state of
        // ref_partition by extending graph_link.
        if (has_eltwise) { cmp.set_has_eltwise_post_op(true); }
        if (output_has_nans) { cmp.set_op_output_has_nans(true); }

        dnn_mem_t mem_fp_abx(mem_fp, dnnl_f32, tag::abx, ::get_cpu_engine());
        cmp.compare(mem_fp_abx, mem_dt, prb->attr, res);
    }
}

} // namespace graph
#endif
