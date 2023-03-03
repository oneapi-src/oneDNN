/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
* Copyright 2022 Arm Ltd. and affiliates
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

#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "pool/pool.hpp"

namespace pool {

int fill_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const int64_t MB {prb->mb};
    const int64_t IC {prb->ic};
    const int64_t D {kind == SRC ? prb->id : prb->od};
    const int64_t H {kind == SRC ? prb->ih : prb->oh};
    const int64_t W {kind == SRC ? prb->iw : prb->ow};
    const int64_t ker_size {prb->kd * prb->kh * prb->kw};
    const auto &c = prb->cfg[kind];
    // For huge kernels to get different output values filling should be very
    // variative, thus, use a factor of 1.
    const bool has_huge_kernel = ker_size >= c.f_max;

    benchdnn_parallel_nd(MB, IC, D, H, W,
            [&](int64_t mb, int64_t ic, int64_t d, int64_t h, int64_t w) {
                const int64_t factor
                        = prb->alg == max || has_huge_kernel ? 1 : ker_size;
                // keep values for avg_exclude_pad positive to prevent cancellation err
                const int64_t f_min = prb->alg == max ? c.f_min / factor : 0;
                // divide on factor to keep value in the range
                const int64_t range = c.f_max / factor - f_min + 1;
                const int64_t gen
                        = 5 * d + 17 * h + 13 * w + 13 * mb + 19 * ic + 1637;
                const float value = (f_min + gen % range) * factor;

                const size_t off = kind == SRC
                        ? src_off_f(prb, mb, ic, d, h, w)
                        : dst_off_f(prb, mb, ic, d, h, w);
                ((float *)mem_fp)[off] = value;
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

// fill ws with big numbers to reliably cause a correctness issue (and not
// anything else) in case of a bug in the library
int fill_ws(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    benchdnn_parallel_nd(
            nelems, [&](int64_t i) { mem_fp.set_elem(i, (1 << 24) - 1); });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    const dir_t dir = init_pd_args.dir;

    const auto src_tag = (dir & FLAG_FWD) ? prb->tag : tag::any;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->src_dims().data(), prb->cfg[SRC].dt, src_tag);
    auto dst_d = dnn_mem_t::init_md(
            prb->ndims, prb->dst_dims().data(), prb->cfg[DST].dt, tag::any);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->dst_dims().data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);

    if (dir & FLAG_FWD) {
        auto prop_kind = prb->dir & FLAG_INF ? dnnl_forward_inference
                                             : dnnl_forward_training;
        DNN_SAFE_STATUS(dnnl_pooling_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop_kind, alg,
                init_pd_args.src_md ? init_pd_args.src_md : src_d, dst_d,
                prb->strides().data(), prb->kernel().data(),
                prb->dilations().data(), prb->padding().data(),
                prb->padding_r().data(), dnnl_attr));
    } else {
        DNN_SAFE_STATUS(dnnl_pooling_backward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, alg, src_d, dst_d,
                prb->strides().data(), prb->kernel().data(),
                prb->dilations().data(), prb->padding().data(),
                prb->padding_r().data(), init_pd_args.hint, dnnl_attr));
    }
    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->cfg[SRC].dt, prb->cfg[DST].dt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);

    if (is_cpu() && prb->cfg[SRC].dt != prb->cfg[DST].dt) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

#if DNNL_AARCH64_USE_ACL
    // Since ACL supports only forward pass.
    // Ref: https://github.com/oneapi-src/oneDNN/issues/1205
    if (prb->dir & FLAG_BWD) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
#endif
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // Average pooling without padding can't handle cases when kernel window is
    // applied to padded area only.
    if (prb->alg == avg_np && prb->has_ker_in_pad()) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    cmp.set_threshold(prb->cfg[kind].eps);
    // Backward may have most zeroes for ker_in_pad with huge kernels problems.
    const float zero_percent = (prb->dir & FLAG_FWD) ? 99.f : 100.f;
    cmp.set_zero_trust_percent(zero_percent); // TODO: consider enabling

    const bool has_inf_output = prb->alg == alg_t::max && prb->has_ker_in_pad()
            && prb->attr.post_ops.len();
    cmp.set_op_output_has_nans(has_inf_output);

    const auto pooling_add_check
            = [&](const compare::compare_t::driver_check_func_args_t &args) {
                  // cuDNN bug: it spits fp16 min value as -inf,
                  // not -65504.
                  if (is_nvidia_gpu() && args.dt == dnnl_f16) {
                      return args.exp == lowest_dt(args.dt)
                              && std::isinf(args.got) && std::signbit(args.got);
                  }
                  if (is_nvidia_gpu() && args.dt == dnnl_bf16) {
                      return args.exp == lowest_dt(args.dt)
                              && std::isinf(args.got) && std::signbit(args.got);
                  }
                  return false;
              };
    cmp.set_driver_check_function(pooling_add_check);
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DST,
            DNNL_ARG_WORKSPACE,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_SRC, // For Graph to compute ws on backward
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_WORKSPACE,
    };
    return (dir & FLAG_FWD) ? exec_fwd_args : exec_bwd_args;
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref) {
    if (has_bench_mode_modifier(mode_modifier_t::no_host_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        ref_mem_map.emplace(
                exec_arg, dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_dat(prb, SRC, mem, ref_mem), WARN);
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_dat(prb, DST, mem, ref_mem), WARN);
                break;
            case DNNL_ARG_WORKSPACE:
                ref_mem_map[exec_arg]
                        = dnn_mem_t(mem.md_, dnnl_s32, tag::abx, ref_engine);
                if (prb->dir & FLAG_FWD && !is_nvidia_gpu())
                    SAFE(fill_ws(prb, mem, ref_mem), WARN);
                break;
            case DNNL_ARG_DST:
                SAFE(!check_md_consistency_with_tag(mem.md_, prb->tag), WARN);
                break;
            case DNNL_ARG_SCRATCHPAD: break;
            default: { // Process all attributes here
                int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                        - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
                bool is_post_ops_arg = (exec_arg & post_ops_range);
                if (is_post_ops_arg) {
                    SAFE(binary::fill_mem(exec_arg, mem, ref_mem), WARN);
                }
            } break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(2); // just fwd or fwd + bwd.
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res, FLAG_FWD,
                 nullptr, /* is_service_prim = */ prb->dir & FLAG_BWD),
            WARN);
    if (prb->dir & FLAG_BWD) {
        SAFE(init_prim(prb->ctx_init, v_prim[1], init_pd, prb, res, FLAG_BWD,
                     query_pd(v_prim[0])),
                WARN);
    }
    return OK;
}

int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        res_t *res) {
    SAFE(check_caches(v_prim[0], res), WARN);
    if (v_prim[1]) { SAFE(check_caches(v_prim[1], res), WARN); }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = prb->dir & FLAG_FWD ? v_prim[0] : v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(
            mem_map, prb, v_prim[0], supported_exec_args(FLAG_FWD));
    SAFE(init_ref_memory_args(
                 ref_mem_map, mem_map, v_prim[0], prb, res, FLAG_FWD),
            WARN);

    args_t args(mem_map), ref_args(ref_mem_map);

    if (bench_mode != bench_mode_t::init)
        SAFE(execute_and_wait(v_prim[0], args, res), WARN);

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        if (prb->dir & FLAG_FWD) {
            check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
        }
    }

    if (prb->dir & FLAG_BWD) {
        // Pass same memory map as we need data from forward on backward.
        init_memory_args<prb_t>(
                mem_map, prb, v_prim[1], supported_exec_args(FLAG_BWD));
        SAFE(init_ref_memory_args(
                     ref_mem_map, mem_map, v_prim[1], prb, res, FLAG_BWD),
                WARN);

        args = args_t(mem_map);
        ref_args = args_t(ref_mem_map);

        SAFE(execute_and_wait(v_prim[1], args, res), WARN);

        if (has_bench_mode_bit(mode_bit_t::corr)) {
            check_correctness(prb, {SRC}, args, ref_args, setup_cmp, res);
        }
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace pool
