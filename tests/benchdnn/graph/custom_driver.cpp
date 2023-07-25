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

#include <random>
#include "common.hpp"
#include "deserialize.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "utils/parallel.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

#include "custom_driver.hpp"

namespace custom {

namespace select {
// SELECT OP
// DNNL_ARG_WEIGHTS: cond
// DNNL_ARG_SRC_0: src_0
// DNNL_ARG_SRC_1: src_1
// DNNL_ARG_DST: dst
// dst[i] = cond[i] ? src_0[i] : src_1[i]

std::vector<int> exec_args = {
        DNNL_ARG_WEIGHTS,
        DNNL_ARG_SRC_0,
        DNNL_ARG_SRC_1,
        DNNL_ARG_DST,
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        const prb_t *prb, res_t *res) {

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second;

        ref_mem_map.emplace(
                exec_arg, dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC_0:
                SAFE(::custom::fill_mem(mem, ref_mem, 0, 1), WARN);
                break;
            case DNNL_ARG_SRC_1:
                SAFE(::custom::fill_mem(mem, ref_mem, 0, 1), WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(::custom::fill_mem(mem, ref_mem, 0, 2), WARN);
                break;
            default: break;
        }
    }
    return OK;
}

int execute(const prb_t *prb, args_t args, res_t *res) {
    const dnn_mem_t &src0 = args.find(DNNL_ARG_SRC_0);
    const dnn_mem_t &src1 = args.find(DNNL_ARG_SRC_1);
    const dnn_mem_t &wei = args.find(DNNL_ARG_WEIGHTS);
    dnn_mem_t &dst = const_cast<dnn_mem_t &>(args.find(DNNL_ARG_DST));

    benchdnn_parallel_nd(dst.nelems(), [&](int64_t index) {
        size_t offsrc0 = 0, offsrc1 = 0, offwei = 0, offdst = 0;
        for (int i = 0; i < dst.ndims(); i++) {
            // calculate the idx on each dimension
            int idx = index % dst.dims()[i];
            index /= dst.dims()[i];
            // accumulate offset for each arg
            offwei += wei.strides()[i] * (wei.dims()[i] < 2 ? 0 : idx);
            offsrc0 += src0.strides()[i] * (src0.dims()[i] < 2 ? 0 : idx);
            offsrc1 += src1.strides()[i] * (src1.dims()[i] < 2 ? 0 : idx);
            offdst += dst.strides()[i] * (dst.dims()[i] < 2 ? 0 : idx);
        }
        dst.set_elem(offdst,
                wei.get_elem(offwei) ? src0.get_elem(offsrc0)
                                     : src1.get_elem(offsrc1));
    });
    return OK;
}
} // namespace select

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    return dnnl_success;
}

std::vector<int> supported_exec_args(const prb_t *prb) {
    std::vector<int> exec_args;
    switch (prb->alg) {
        case SELECT: return ::custom::select::exec_args;
        default: assert(!"unknown alg"); break;
    }
    return exec_args;
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    switch (prb->alg) {
        case SELECT: cmp.set_zero_trust_percent(100.f); break;
        default: assert(!"unknown alg"); break;
    }
    return;
}

int fill_mem(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, int f_min, int f_max) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto dt = mem_dt.dt();
    f_min = (dt == dnnl_u8 && f_min < 0) ? 0 : f_min;
    std::minstd_rand int_seed(mem_dt.dt() * nelems + 1);
    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        std::uniform_int_distribution<> gen(f_min, f_max);
        float value = gen(int_seed);
        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void init_memory_args(dnn_mem_map_t &mem_map, const prb_t *prb,
        const std::vector<int> &supported_exec_args,
        const engine_t &test_engine) {
    for (const auto &exec_arg : supported_exec_args) {
        if (prb->arg_mds_.find(exec_arg) == prb->arg_mds_.end()) {
            assert(!"missing required args");
            continue;
        };
        auto arg_mds_ = prb->arg_mds_.find(exec_arg)->second;
        dnnl_dims_t dnnl_dims;
        auto dim = ::std::get<1>(arg_mds_);
        for (size_t i = 0; i < dim.size(); i++) {
            dnnl_dims[i] = dim[i];
        }
        mem_map.emplace(exec_arg,
                dnn_mem_t(static_cast<int>(dim.size()), dnnl_dims,
                        std::get<2>(arg_mds_), ::std::get<0>(arg_mds_),
                        test_engine));
    }
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        const prb_t *prb, res_t *res) {
    if (has_bench_mode_modifier(mode_modifier_t::no_host_memory)) return OK;

    switch (prb->alg) {
        case SELECT:
            SAFE(::custom::select::init_ref_memory_args(
                         ref_mem_map, mem_map, prb, res),
                    WARN);
            break;
        default: assert(!"unknown alg"); break;
    }
    // Don't keep reference memory if it is not used further.
    if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {}

int execute(const prb_t *prb, args_t args, res_t *res) {
    switch (prb->alg) {
        case SELECT: ::custom::select::execute(prb, args, res); break;
        default: assert(!"unknown alg"); break;
    }
    return OK;
}

} // namespace custom
