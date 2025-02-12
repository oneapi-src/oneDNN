/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
* Copyright 2022-2023 Arm Ltd. and affiliates
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

#include <algorithm>
#include <cstring>
#include <random>
#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "pool/pool.hpp"

namespace pool {

int fill_data(data_kind_t kind, const prb_t *prb, const cfg_t &cfg,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, res);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, res, get_perf_fill_cfg(mem_dt.dt()));
    }

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));

        // make sure the first element is positive
        if (idx_start == 0) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            mem_fp.set_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float val = gen(int_seed);
            mem_fp.set_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
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
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    const auto src_tag = (dir & FLAG_FWD) ? prb->tag : tag::any;

    auto src_d = dnn_mem_t::init_md(prb->ndims, prb->src_dims().data(),
            force_f32_dt ? dnnl_f32 : prb->src_dt(), src_tag);
    auto dst_d = dnn_mem_t::init_md(prb->ndims, prb->dst_dims().data(),
            force_f32_dt ? dnnl_f32 : prb->dst_dt(), tag::any);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->dst_dims().data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);

    if (dir & FLAG_FWD) {
        auto prop_kind = prb->dir & FLAG_INF ? dnnl_forward_inference
                                             : dnnl_forward_training;
        TIME_C_PD(DNN_SAFE_STATUS(dnnl_pooling_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop_kind, alg,
                init_pd_args.src_md ? init_pd_args.src_md : src_d, dst_d,
                prb->strides().data(), prb->kernel().data(),
                prb->dilations().data(), prb->padding().data(),
                prb->padding_r().data(), dnnl_attr)));
    } else {
        TIME_C_PD(DNN_SAFE_STATUS(dnnl_pooling_backward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, alg, src_d, dst_d,
                prb->strides().data(), prb->kernel().data(),
                prb->dilations().data(), prb->padding().data(),
                prb->padding_r().data(), init_pd_args.hint, dnnl_attr)));
    }
    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->src_dt(), prb->dst_dt()}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_pooling, prb->src_dt());
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_pooling);

    if (is_cpu() && prb->src_dt() != prb->dst_dt()) {
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
        return;
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // Average pooling without padding can't handle cases when kernel window is
    // applied to padded area only.
    if (prb->alg == avg_np && prb->has_ker_in_pad()) {
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }
}

// Special function to handle Nvidia libraries issues showing up through the
// timeline. Not recommended to remove instances to keep working state of any
// cuda/cuDNN/cuBLAS versions.
bool cuda_check_correctness(const prb_t *prb,
        const compare::compare_t::driver_check_func_args_t &args) {
    if (!is_nvidia_gpu()) return false;

    if (args.dt == dnnl_f16) {
        // cuDNN bug: it spits f16 min value as -inf, not -65504.
        return args.exp == lowest_dt(args.dt) && std::isinf(args.got)
                && std::signbit(args.got);
    } else if (args.dt == dnnl_s8) {
        // cuDNN bug: ... and s8 min value as -127 (-INT8_MAX?), not -128.
        return args.exp == lowest_dt(args.dt) && args.got == -127;
    } else if (prb->alg == alg_t::max && (prb->dir & FLAG_BWD)
            && prb->dt[1] == dnnl_bf16) {
        // cuDNN seems to accumulate bf16 inputs into bf16 accumulator which
        // leads to rounding errors while f32 would accumulate precisely.
        // Adjusting filling not to overflow bf16 exact value solves the problem
        // but will affect the validation of rest of the library.
        // Just ignore errors if they are not extreme - `32` is just a random
        // number to speculate on potential rounding error.
        return args.diff < 32.f;
    }
    return false;
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    // Threshold to compensate division error. CPU could live with 6.f coeff.
    const float trh = 10.f * epsilon_dt(prb->dt[1]);
    cmp.set_threshold(trh);
    // Backward may have most zeroes for ker_in_pad with huge kernels problems.
    const float zero_percent = (prb->dir & FLAG_FWD) ? 99.f : 100.f;
    cmp.set_zero_trust_percent(zero_percent); // TODO: consider enabling

    const bool has_inf_output = prb->alg == alg_t::max && prb->has_ker_in_pad()
            && prb->attr.post_ops.len();
    cmp.set_op_output_has_nans(has_inf_output);

    // Since lambda is called when stack is unavailable, need to capture `prb`
    // and `kind` by value to avoid using dangling references.
    const auto pooling_add_check =
            [&, prb](const compare::compare_t::driver_check_func_args_t &args) {
                return cuda_check_correctness(prb, args);
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
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_WORKSPACE,
    };
    static const std::vector<int> exec_bwd_args_graph = {
            DNNL_ARG_SRC, // For Graph to compute ws on backward
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_WORKSPACE,
    };
    return (dir & FLAG_FWD)            ? exec_fwd_args
            : (driver_name == "graph") ? exec_bwd_args_graph
                                       : exec_bwd_args;
};

fill_cfg_t binary_po_fill_cfg(
        int exec_arg, const dnn_mem_t &mem, const attr_t &attr) {
    fill_cfg_t cfg;
    const int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
            - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
    const bool is_post_ops_arg = (exec_arg & post_ops_range);
    if (is_post_ops_arg) {
        // Config secures only positive values since average pooling output is
        // positive, and using negative values leads to the cancellation
        // effect.
        const int bin_po_idx
                = exec_arg / DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE - 1;
        assert(bin_po_idx < attr.post_ops.len());
        const auto alg = attr.post_ops.entry[bin_po_idx].kind;
        cfg = fill_cfg_t(mem.dt(), 0.f, 16.f, /* int = */ true, alg,
                "pooling_binary_post_op");
    }
    return cfg;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, DST});

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg != DNNL_ARG_SCRATCHPAD && exec_arg != DNNL_ARG_WORKSPACE) {
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data(SRC, prb, cfg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_data(DST, prb, cfg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_WORKSPACE: {
                const auto ws_dt
                        = is_integral_dt(mem.dt()) ? dnnl_s32 : dnnl_f32;
                ref_mem_map[exec_arg]
                        = dnn_mem_t(mem.md_, ws_dt, tag::abx, ref_engine);
                if (prb->dir & FLAG_FWD) SAFE(fill_ws(prb, mem, ref_mem), WARN);
                break;
            }
            case DNNL_ARG_DST:
                SAFE(!check_md_consistency_with_tag(mem.md_, prb->tag), WARN);
                break;
            default:
                const auto &binary_fill_cfg
                        = binary_po_fill_cfg(exec_arg, mem, prb->attr);
                std::unordered_map<int, fill_cfg_t> fill_cfg_map {
                        {DNNL_ARG_SRC_1, binary_fill_cfg}};
                SAFE(init_ref_memory_args_default_case(exec_arg, mem, ref_mem,
                             prb->attr, res, fill_cfg_map),
                        WARN);
                break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb, dir_t dir) {
    std::vector<data_kind_t> check_kinds;
    if ((prb->dir & FLAG_FWD) && (dir & FLAG_FWD)) {
        check_kinds = {DST};
    } else if ((prb->dir & FLAG_BWD) && (dir & FLAG_BWD)) {
        check_kinds = {SRC};
    }
    // `check_kinds` is empty for `(prb->dir & FLAG_BWD) && (dir & FLAG_FWD)`.
    return check_kinds;
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

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb, res), WARN);
        if (v_prim[1]) { SAFE(check_caches(v_prim[1], prb, res), WARN); }
    }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = prb->dir & FLAG_FWD ? v_prim[0] : v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(
            mem_map, prb, v_prim[0], supported_exec_args(FLAG_FWD));
    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, v_prim[0], prb, res),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    if (bench_mode != bench_mode_t::init)
        SAFE(execute_and_wait(v_prim[0], args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb, FLAG_FWD), args, ref_args,
            setup_cmp, res);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb, FLAG_FWD), args, prb->attr,
                 prb->inplace, res),
            WARN);

    if (prb->dir & FLAG_BWD) {
        // Pass same memory map as we need data from forward on backward.
        init_memory_args<prb_t>(
                mem_map, prb, v_prim[1], supported_exec_args(FLAG_BWD));
        TIME_FILL(SAFE(
                init_ref_memory_args(ref_mem_map, mem_map, v_prim[1], prb, res),
                WARN));

        args = args_t(mem_map);
        ref_args = args_t(ref_mem_map);

        SAFE(execute_and_wait(v_prim[1], args, res), WARN);

        check_correctness(prb, get_kinds_to_check(prb, FLAG_BWD), args,
                ref_args, setup_cmp, res);
        SAFE(check_bitwise(prim, get_kinds_to_check(prb, FLAG_BWD), args,
                     prb->attr, prb->inplace, res),
                WARN);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace pool
