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

#ifndef BENCHDNN_GRAPH_CUSTOM_DRIVER_H
#define BENCHDNN_GRAPH_CUSTOM_DRIVER_H

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "deserialize.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/settings.hpp"

namespace custom {

enum alg_t {
    SELECT,
    TRANSPOSE,
    RESHAPE,
    ALG_UNKNOWN,
};

using arg_md_t = ::std::tuple<::std::string, dims_t, dnnl_data_type_t>;

struct settings_t {

    settings_t() = default;

    ::std::unordered_map<int, arg_md_t> arg_mds_;
    ::std::vector<int64_t> order;
    alg_t alg;
};

struct prb_t {
    prb_t(const settings_t &s) {
        arg_mds_ = s.arg_mds_;
        alg = s.alg;
        switch (alg) {
            case TRANSPOSE: order = s.order; break;
            default: break;
        }
    }
    ::std::unordered_map<int, arg_md_t> arg_mds_;
    ::std::vector<int64_t> order;
    attr_t attr;
    alg_t alg;
};

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args);
std::vector<int> supported_exec_args(const prb_t *prb);

int fill_mem(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, int f_min, int f_max);
void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args);

void init_memory_args(dnn_mem_map_t &mem_map, const prb_t *prb,
        const std::vector<int> &supported_exec_args,
        const engine_t &test_engine = get_test_engine());

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        const prb_t *prb, res_t *res);

void skip_unimplemented_prb(const prb_t *prb, res_t *res);

int execute(const prb_t *prb, const args_t &args, res_t *res);

} // namespace custom
#endif
