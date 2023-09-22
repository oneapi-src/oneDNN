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

#ifndef UTILS_TASK_HPP
#define UTILS_TASK_HPP

#include <memory>
#include <sstream>
#include <string>
#include <vector>

template <typename prb_t, typename perf_report_t, typename create_func_t,
        typename check_cache_func_t, typename do_func_t>
struct rnn_task_t {
    rnn_task_t(std::shared_ptr<const prb_t> prb,
            const std::string &perf_template, const create_func_t &create_func,
            const check_cache_func_t &check_cache_func,
            const do_func_t &do_func)
        : prb_(std::move(prb))
        , create_func_(create_func)
        , check_cache_func_(check_cache_func)
        , do_func_(do_func)
        , perf_template_(perf_template) {}

    int create() {
        BENCHDNN_PRINT(1, "create: %s\n", prb_.get()->str());
        if (bench_mode == bench_mode_t::list) return res_.state = LISTED, OK;

        v_prim_ = std::make_shared<
                std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>>>();
        const prb_t *prb = prb_.get();
        SAFE(create_func_(*v_prim_, *prb, &res_), WARN);
        return OK;
    }

    // Since rnn_task_t doesn't have a control over primitives, it has to pass
    // this control to a driver which is aware of what primitives should be
    // checked for being in the cache.
    int check_cache() {
        if (!has_bench_mode_bit(mode_bit_t::corr)) return OK;

        const prb_t *prb = prb_.get();
        return check_cache_func_(*v_prim_, prb, &res_);
    }

    int exec() {
        BENCHDNN_PRINT(1, "run: %s\n", prb_.get()->str());
        if (res_.state == INITIALIZED && bench_mode != bench_mode_t::init) {
            const prb_t *prb = prb_.get();
            do_func_(*v_prim_, *prb, &res_);
        }

        return report();
    }

private:
    std::shared_ptr<const prb_t> prb_;
    create_func_t create_func_;
    check_cache_func_t check_cache_func_;
    do_func_t do_func_;
    std::string perf_template_;
    res_t res_ {};
    // Use vector to handle any number of primitives needed for a driver.
    // It's a driver responsibility to initialize vector at `create()` stage
    // and it utilizes the knowledge of primitive order inside the vector.
    //
    // Note: shared_ptr here is to work around old compiler issue that triggers
    // `vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>>` copy constructor,
    // which is deleted for `benchdnn_dnnl_wrapper_t`. New compilers don't do
    // that since `v_prim_` is not initialized at constructor time.
    //
    // TODO: remove shared_ptr work around when migrate to newer miminal default
    // compiler.
    std::shared_ptr<std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>>>
            v_prim_;

    // Note: can't be `const` because of `parse_result`.
    int report() {
        const prb_t *prb = prb_.get();
        parse_result(res_, prb->str());
        if (has_bench_mode_bit(mode_bit_t::perf)) {
            perf_report_t pr(prb, perf_template_.c_str());
            pr.report(&res_, prb->str());
        }
        return OK;
    }
};

#endif
