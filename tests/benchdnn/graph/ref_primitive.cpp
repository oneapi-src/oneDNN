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

#include "ref_primitive.hpp"
#include "setting_handler.hpp"

namespace graph {

ref_primitive_t::ref_primitive_t(const deserialized_op &op) {
    op_ = op;
    kind_ = opstr2kind(op_.kind_);
    driver_ = opkind2driver(kind_);

    static const ::std::unordered_set<::std::string> special_backward_op = {
            // bnorm backward
            "BatchNormTrainingBackward",
            // eltwise backward
            "AbsBackward",
            "ClampBackward",
            "EluBackward",
            "GELUBackward",
            "HardSigmoidBackward",
            "HardSwishBackward",
            "MishBackward",
            "ReLUBackward",
            "SigmoidBackward",
            "SoftPlusBackward",
            "SqrtBackward",
            "TanhBackward",
            // pool backward
            "AvgPoolBackward",
            "MaxPoolBackward",
    };
    is_special_backward_op_
            = special_backward_op.find(op_.kind_) != special_backward_op.end();
}

// a switch skeleton to handle thing for each driver and template code
// CASE_DRIVER is for primitive driver and CASE_CUSTOM is for custom driver
// caller should define the behavior for both situation
#define SWITCH_DRIVER(CASE_DRIVER, CASE_CUSTOM) \
    switch (driver_) { \
        CASE_CUSTOM; \
        CASE_DRIVER(binary); \
        CASE_DRIVER(bnorm); \
        CASE_DRIVER(concat); \
        CASE_DRIVER(conv); \
        CASE_DRIVER(deconv); \
        CASE_DRIVER(eltwise); \
        CASE_DRIVER(lnorm); \
        CASE_DRIVER(matmul); \
        CASE_DRIVER(pool); \
        CASE_DRIVER(prelu); \
        CASE_DRIVER(reduction); \
        CASE_DRIVER(reorder); \
        CASE_DRIVER(resampling); \
        CASE_DRIVER(softmax); \
        default: { \
            SAFE_V(FAIL); \
            break; \
        } \
    }

int ref_primitive_t::init_prb(
        ::std::unordered_set<size_t> &bf16_rewrite, res_t *res) {
#define CASE_INIT_PRB(driver) \
    case dnnl_driver_t::driver: { \
        ::driver::settings_t setting \
                = get_setting<::driver::settings_t>(op_, bf16_rewrite, res); \
        if (res->state == INVALID_ARGUMENTS) return FAIL; \
        setting.finalize(); \
        auto pprb = ::std::make_shared<::driver::prb_t>(setting); \
        prb_wrapper_ \
                = ::std::make_shared<prb_wrapper_t<::driver::prb_t>>(pprb); \
        break; \
    }

#define CASE_INIT_CUSTOM_PRB CASE_INIT_PRB(custom);

    SWITCH_DRIVER(CASE_INIT_PRB, CASE_INIT_CUSTOM_PRB);

    return OK;
}

int ref_primitive_t::init_prim(const engine_t &ref_eng, res_t *res) {
#define CASE_INIT_PRIM(driver) \
    case dnnl_driver_t::driver: { \
        const ::driver::prb_t *prb = prb_wrapper_->get<::driver::prb_t>(); \
        dnn_mem_map_t ref_mems; \
        if (is_special_backward_op_) { \
            SAFE(create_primitive(fwd_prim_, ref_eng, ::driver::init_pd, prb, \
                         res, FLAG_FWD, nullptr, prb->dir &FLAG_BWD, nullptr, \
                         false), \
                    WARN); \
            if (res->state == SKIPPED || res->state == UNIMPLEMENTED) \
                return OK; \
            ::init_memory_args(mems_, prb, fwd_prim_, \
                    ::driver::supported_exec_args(FLAG_FWD), ref_eng); \
            SAFE(::driver::init_ref_memory_args( \
                         ref_mems, mems_, fwd_prim_, prb, res), \
                    WARN); \
            args_ = args_t(mems_); \
            SAFE(execute_and_wait(fwd_prim_, args_, res), WARN); \
        } \
        SAFE(create_primitive(prim_, ref_eng, ::driver::init_pd, prb, res, \
                     prb->dir, \
                     is_special_backward_op_ ? query_pd(fwd_prim_) : nullptr, \
                     false, nullptr, false), \
                WARN); \
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK; \
        break; \
    }
// custom driver does not contain primitive, skip this step
#define CASE_INIT_CUSTOM_PRIM \
    case dnnl_driver_t::custom: break;

    SWITCH_DRIVER(CASE_INIT_PRIM, CASE_INIT_CUSTOM_PRIM);
    return OK;
}

void ref_primitive_t::init_memory_args(const engine_t &ref_eng) {
#define CASE_INIT_MEMORY_ARGS(driver) \
    case dnnl_driver_t::driver: { \
        if (prb_wrapper_) { \
            const ::driver::prb_t *prb = prb_wrapper_->get<::driver::prb_t>(); \
            ::init_memory_args(mems_, prb, prim_, \
                    ::driver::supported_exec_args(prb->dir), ref_eng); \
        } \
        break; \
    }

#define CASE_INIT_CUSTOM_MEMORY_ARGS \
    case dnnl_driver_t::custom: { \
        if (prb_wrapper_) { \
            const ::custom::prb_t *prb = prb_wrapper_->get<::custom::prb_t>(); \
            ::custom::init_memory_args( \
                    mems_, prb, ::custom::supported_exec_args(prb), ref_eng); \
        } \
        break; \
    }

    SWITCH_DRIVER(CASE_INIT_MEMORY_ARGS, CASE_INIT_CUSTOM_MEMORY_ARGS);
}

int ref_primitive_t::init_ref_memory_args(const engine_t &ref_eng, res_t *res) {
#define CASE_INIT_REF_MEMORY_ARGS(driver) \
    case dnnl_driver_t::driver: { \
        dnn_mem_map_t ref_mems; \
        if (prb_wrapper_) { \
            const ::driver::prb_t *prb = prb_wrapper_->get<::driver::prb_t>(); \
            SAFE(::driver::init_ref_memory_args( \
                         ref_mems, mems_, prim_, prb, res), \
                    WARN); \
            args_ = args_t(mems_); \
        } \
        break; \
    }

#define CASE_INIT_CUSTOM_REF_MEMORY_ARGS \
    case dnnl_driver_t::custom: { \
        dnn_mem_map_t ref_mems; \
        if (prb_wrapper_) { \
            const ::custom::prb_t *prb = prb_wrapper_->get<::custom::prb_t>(); \
            SAFE(::custom::init_ref_memory_args(ref_mems, mems_, prb, res), \
                    WARN); \
            args_ = args_t(mems_); \
        } \
        break; \
    }

    SWITCH_DRIVER(CASE_INIT_REF_MEMORY_ARGS, CASE_INIT_CUSTOM_REF_MEMORY_ARGS);
    return OK;
}

int ref_primitive_t::execute_prim(res_t *res) const {
    if (driver_ == dnnl_driver_t::custom) {
        const ::custom::prb_t *prb = prb_wrapper_->get<::custom::prb_t>();
        SAFE(::custom::execute(prb, args_, res), WARN);
    } else {
        SAFE(execute_and_wait(prim_, args_, res), WARN);
    }
    return OK;
}

void ref_primitive_t::check_correctness(
        const args_t &args, bool has_eltwise, bool has_nans, res_t *res) const {

    static const std::unordered_map<size_t, data_kind_t>
            dnnl_arg_2_data_kind_map {
                    {DNNL_ARG_SRC, SRC},
                    {DNNL_ARG_WEIGHTS_0, WEI},
                    {DNNL_ARG_DIFF_WEIGHTS_0, WEI},
                    {DNNL_ARG_BIAS, BIA},
                    {DNNL_ARG_DIFF_BIAS, BIA},
                    {DNNL_ARG_DST, DST},
                    {DNNL_ARG_DIFF_SRC_0, DST},
                    {DNNL_ARG_SRC_1, SRC_1},
                    {DNNL_ARG_MEAN, MEAN},
                    {DNNL_ARG_VARIANCE, VAR},
                    {DNNL_ARG_SCALE, SC},
                    {DNNL_ARG_DIFF_SCALE, SC},
                    {DNNL_ARG_SHIFT, SH},
                    {DNNL_ARG_DIFF_SHIFT, SH},
            };

#define CASE_CHECK_CORRECTNESS(driver) \
    case dnnl_driver_t::driver: { \
        const ::driver::prb_t *prb = prb_wrapper_->get<::driver::prb_t>(); \
        setup_cmp(cmp, prb, dnnl_arg_2_data_kind_map.at(arg), args_); \
        attr = prb->attr; \
        break; \
    }

#define CASE_CUSTOM_CHECK_CORRECTNESS CASE_CHECK_CORRECTNESS(custom)

    // args is the result from graph side
    // args_ is the reference result under this context
    // only check the arg contained in args, compare args with args_
    for (int i = 0; i < args.size(); i++) {
        check_zero_padding(args.dnn_mem(i), args.arg(i), res);
        check_buffer_overwrite(args.dnn_mem(i), args.arg(i), res);

        const auto arg = args.arg(i);
        const auto &mem_dt = args.find(arg);
        const auto &mem_fp = args_.find(arg);

        if (dnnl_arg_2_data_kind_map.find(arg)
                == dnnl_arg_2_data_kind_map.end()) {
            BENCHDNN_PRINT(1, "Output arg %d is unsupported!\n", arg);
            res->state = UNIMPLEMENTED;
            return;
        }
        compare::compare_t cmp;
        attr_t attr;
        SWITCH_DRIVER(CASE_CHECK_CORRECTNESS, CASE_CUSTOM_CHECK_CORRECTNESS);

        cmp.set_has_eltwise_post_op(has_eltwise);
        cmp.set_op_output_has_nans(has_nans);
        dnn_mem_t mem_fp_abx(mem_fp, dnnl_f32, tag::abx, ::get_cpu_engine());
        if (cmp.compare(mem_fp_abx, mem_dt, attr, res) == FAIL) {
            const std::string p2p_check_fail
                    = "P2P check failed, fall back to use norm check!";
            BENCHDNN_PRINT(0, "%s\n", p2p_check_fail.c_str());

            // TODO: we need a reasonable threshold here for GC Backend cases
            // once the complex fusion validation is enabled.

            // Fall back to norm check if P2P check failed.
            res->state = EXECUTED;
            res->errors = 0;
            cmp.set_norm_validation_mode(true);
            if (cmp.compare(mem_fp_abx, mem_dt, attr, res) == FAIL) {
                const std::string norm_check_fail = "Norm check failed, quit!";
                BENCHDNN_PRINT(
                        0, "Output arg %d: %s\n", arg, norm_check_fail.c_str());
                break;
            }
        }
    }
}

int ref_primitive_t::displace_scales() const {
    // Runtime data for scales attribute is supported for quantization ops only.
    if (op_.kind_ != "Dequantize" && op_.kind_ != "Quantize") return OK;

    const auto it_attr_scales = op_.attrs_.find("scales");
    const bool has_scales = it_attr_scales != op_.attrs_.end();
    if (!has_scales) return OK;

    int arg = BENCHDNN_DNNL_ARG_UNDEF;
    bool scales_found = false;
    for (auto it = mems_.begin(); it != mems_.end(); it++) {
        const int cur_arg = (*it).first;
        const bool is_scales_arg = (cur_arg & DNNL_ARG_ATTR_SCALES);
        if (!is_scales_arg) continue;
        // Protection from the cases when somehow scales are applied to more
        // than a single argument (which is unexpected).
        if (scales_found) {
            assert(!"scales are applied to more than a single arg");
            return FAIL;
        }
        scales_found = true;
        arg = cur_arg;
    }

    // No correspondent memory was found. Nothing to update.
    if (arg == BENCHDNN_DNNL_ARG_UNDEF) return OK;

    // Updating values.
    const auto &mem = mems_.at(arg);
    const auto &f32_vector = it_attr_scales->second.f32_vector_;
    for (size_t i = 0; i < f32_vector.size(); i++) {
        mem.set_elem(i, f32_vector[i]);
    }

    return OK;
}

} // namespace graph
