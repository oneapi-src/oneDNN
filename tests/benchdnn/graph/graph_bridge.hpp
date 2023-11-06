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

#include "graph_memory.hpp"
#include "setting_handler.hpp"
#include "utils/compare.hpp"
#include "utils/settings.hpp"
namespace graph {

// prb_wrapper_base_t & prb_wrapper_t are defined to wrap prb objection because
// C++ 11 does not support template member variable and there is no common base type
// for all prb_t types, thus we cannot put shared ptr of prb or its base object
// directly into ref_prims_ member of ref_partition_t object. now shared pointer of
// wrapper base object will be put into ref_prims_.
// These wrappers could be removed after moving to C++ 14
class prb_wrapper_base_t {
public:
    virtual ~prb_wrapper_base_t() = default;
    template <typename prb_t>
    const prb_t *get() const;
};

// A template class to wrap shared pointer of prb obj
template <typename prb_t>
class prb_wrapper_t : public prb_wrapper_base_t {
public:
    prb_wrapper_t(const std::shared_ptr<prb_t> prb) { prb_ = prb; }
    // get raw pointer of prb object
    const prb_t *get() const { return prb_.get(); }

private:
    std::shared_ptr<prb_t> prb_;
};

// A template function in base wrapper, which dynamic cast from base object to
// derived object and return raw pointer of prb obj
template <typename prb_t>
inline const prb_t *prb_wrapper_base_t::get() const {
    return dynamic_cast<const prb_wrapper_t<prb_t> &>(*this).get();
}

using graph_link_t = std::tuple<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>,
        dnn_mem_map_t, dnn_mem_map_t, args_t, args_t,
        std::shared_ptr<prb_wrapper_base_t>>;
using ref_prims_t = std::unordered_map<int, graph_link_t>;
using op_ref_list_t = std::list<std::reference_wrapper<const deserialized_op>>;

template <bool B>
using req = typename std::enable_if<B, bool>::type;

#define DECLARE_TEMPLATE_GET_SETTING(driver) \
    template <typename setting_t, \
            req<std::is_same<setting_t, ::driver::settings_t>::value> = true> \
    setting_t get_setting(const deserialized_op &base_op_ref, \
            const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) { \
        deserialized_op base_op = base_op_ref; \
        for (size_t i = 0; i < base_op.in_lts_.size(); i++) { \
            if (base_op.in_lts_[i].shape_.size() == 0) \
                base_op.in_lts_[i].shape_.emplace_back(1); \
            if (base_op.in_lts_[i].stride_.size() == 0) \
                base_op.in_lts_[i].stride_.emplace_back(1); \
        } \
        for (size_t i = 0; i < base_op.out_lts_.size(); i++) { \
            if (base_op.out_lts_[i].shape_.size() == 0) \
                base_op.out_lts_[i].shape_.emplace_back(1); \
            if (base_op.out_lts_[i].stride_.size() == 0) \
                base_op.out_lts_[i].stride_.emplace_back(1); \
        } \
        return driver::get_setting(base_op, rewrite_lt_ids, res); \
    }

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

int init_graph_memory_args(const dnn_mem_map_t &mems,
        partition_mem_map_t &graph_mem_map,
        const std::vector<size_t> &partition_in_ids,
        const std::vector<size_t> &partition_out_ids,
        const deserialized_op &base_op_ref, res_t *res);

template <typename prb_t, typename init_pd_func_t,
        typename supported_exec_args_func_t, typename setup_cmp_func_t>
int init_prim(ref_prims_t &ref_prims, const deserialized_op &base_op_ref,
        const init_pd_func_t &init_pd,
        const supported_exec_args_func_t &supported_exec_args,
        const setup_cmp_func_t &setup_cmp, const std::shared_ptr<prb_t> pprb,
        const engine_t &ref_eng, res_t *res) {
    int op_id = static_cast<int>(base_op_ref.id_);
    const prb_t *prb = pprb.get();
    auto prb_wrp = std::make_shared<prb_wrapper_t<prb_t>>(pprb);
    ref_prims[op_id] = std::make_tuple(
            benchdnn_dnnl_wrapper_t<dnnl_primitive_t>(), dnn_mem_map_t(),
            dnn_mem_map_t(), args_t(), args_t(), prb_wrp);

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

    std::get<3>(ref_prims[op_id]) = args_t(mems);
    std::get<4>(ref_prims[op_id]) = args_t(ref_mems);

    return OK;
}

template <typename prb_t>
int execute_prim(const ref_prims_t &ref_prims,
        const deserialized_op &base_op_ref, const prb_t *prb, res_t *res) {
    int op_id = static_cast<int>(base_op_ref.id_);
    auto &prim = std::get<0>(ref_prims.at(op_id));
    auto &args = std::get<3>(ref_prims.at(op_id));

    // Execute a primitive.
    SAFE(execute_and_wait(prim, args, res), WARN);

    return OK;
}

template <typename prb_t>
void check_correctness(ref_prims_t &ref_prims, size_t op_id, const args_t &args,
        const args_t &ref_args, const prb_t *prb, bool has_eltwise,
        bool output_has_nans, res_t *res) {
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

        if (has_eltwise) { cmp.set_has_eltwise_post_op(true); }
        if (output_has_nans) { cmp.set_op_output_has_nans(true); }
        dnn_mem_t mem_fp_abx(mem_fp, dnnl_f32, tag::abx, ::get_cpu_engine());
        if (cmp.compare(mem_fp_abx, mem_dt, prb->attr, res) == FAIL) {
            const std::string p2p_check_fail
                    = "P2P check failed, fall back to use norm check!";
            BENCHDNN_PRINT(0, "%s\n", p2p_check_fail.c_str());

            // TODO: we need a reasonable threshold here for GC Backend cases
            // once the complex fusion validation is enabled.

            // Fall back to norm check if P2P check failed.
            res->state = EXECUTED;
            res->errors = 0;
            cmp.set_norm_validation_mode(true);
            if (cmp.compare(mem_fp_abx, mem_dt, prb->attr, res) == FAIL) {
                const std::string norm_check_fail = "Norm check failed, quit!";
                BENCHDNN_PRINT(
                        0, "Output arg %d: %s\n", arg, norm_check_fail.c_str());
                break;
            }
        }
    }
}

} // namespace graph
#endif
