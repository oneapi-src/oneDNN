/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef DNNL_GRAPH_COMMON_HPP
#define DNNL_GRAPH_COMMON_HPP

#include <cstddef>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

namespace benchdnnext {

using dims_t = dnnl::graph::logical_tensor::dims_t;
using dim_t = dims_t::value_type;

using input_list_t = std::vector<dnnl::graph::logical_tensor>;
using output_list_t = std::vector<dnnl::graph::logical_tensor>;

enum class fill_status {
    DONE, // everything was fine
    UNHANDLED_CONFIG_OPTIONS, // conversion done, but too much options was provided
    UNSUPPORTED_OP,
    UNKNOWN_ERROR
};
typedef fill_status fill_status_t;

dnnl::graph::logical_tensor::data_type convert_dt(const dnnl_data_type_t dt);
dnnl::graph::op::kind convert_alg_kind(const dnnl_alg_kind_t kind);
std::string convert_tag(const std::string &tag, bool activation_tag = true);
dims_t convert_bin_policy(const dims_t &lhs_dims, const attr_t::policy_t policy,
        const std::string &data_format);
std::map<std::string, float> convert_eltw_entry(
        const dnnl::graph::op::kind kind,
        const attr_t::post_ops_t::entry_t &entry);

#define BENCHDNNEXT_SAFE(f, s) \
    do { \
        try { \
            f; \
        } catch (const dnnl::graph::error &e) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, e.what()); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            return FAIL; \
        } \
    } while (0)

struct id_manager_t {
    id_manager_t() : frozen_(false) {};

    size_t operator[](const std::string &arg) {
        const auto &it = knots_.find(arg);
        if (it != knots_.end()) return it->second;
        if (frozen_) {
            std::cout << "Unrecognized argument [" << arg << "]!\n";
            std::abort();
        }
        const auto &new_it = knots_.emplace(arg, knots_.size());
        if (new_it.second) {
            return new_it.first->second;
        } else {
            std::cout << "New argument [" << arg
                      << "] is failed to be added to knots.\n";
            std::abort();
        }
    }

    void freeze() { frozen_ = true; }

private:
    std::map<std::string, size_t> knots_;
    // indicates that the graph is frozen
    bool frozen_;
};

struct tensor_descs_t {
    tensor_descs_t() = default;

    template <typename... Args>
    void emplace(std::string str, Args... args) {
        dnnl::graph::logical_tensor t(idmgr_[str], std::forward<Args>(args)...);
        map_.emplace(str, t);
    }

    dnnl::graph::logical_tensor operator[](const std::string &str) {
        return map_.at(str);
    }

private:
    std::map<std::string, dnnl::graph::logical_tensor> map_;
    id_manager_t idmgr_;
};

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt,
        const dnnl::graph::logical_tensor::data_type &graph_dt,
        const char *tag = nullptr);

dnn_mem_t make_dnn_mem(
        const dnnl::graph::logical_tensor &lt, const char *tag = nullptr);

template <typename T, std::size_t N>
constexpr T *end(T (&arr)[N]) noexcept {
    return arr + N;
}

typedef std::function<void(dnnl::graph::stream &,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs)>
        perf_function_t;

// Engine used to run oneDNN fusion patterns for testing.
inline dnnl::graph::engine &get_test_engine() {
    dnnl::graph::engine::kind graph_engine_kind = engine_tgt_kind == dnnl_cpu
            ? dnnl::graph::engine::kind::cpu
            : dnnl::graph::engine::kind::gpu;
    static dnnl::graph::engine instance(
            graph_engine_kind, static_cast<int>(engine_index));
    return instance;
}

int execute_and_wait(perf_function_t &exec_func,
        const dnnl::graph::engine &engine,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int execute_and_wait(dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int measure_perf(benchdnn_timer_t &t, perf_function_t &perf_func,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int measure_perf(benchdnn_timer_t &t, dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

struct graph_prb_t {
    using dt = dnnl::graph::logical_tensor::data_type;
    using lt = dnnl::graph::logical_tensor::layout_type;

    dnnl::graph::graph to_graph() {
        dnnl::graph::engine &engine = benchdnnext::get_test_engine();
        dnnl::graph::graph graph(engine.get_kind());
        for (auto &&op : ops_)
            graph.add_op(op);
        return graph;
    }

protected:
    std::vector<dnnl::graph::op> ops_;
    tensor_descs_t tensor_descs_;

    std::vector<std::string> curr_out_map_ids_;

    friend struct po_handlers_t;
};

struct po_handlers_t {
    using dt = dnnl::graph::logical_tensor::data_type;
    using lt = dnnl::graph::logical_tensor::layout_type;

private:
    struct bias_po_handler_t {
        fill_status_t operator()(graph_prb_t &p, const std::string &dst_dataf,
                const dnnl::graph::logical_tensor::data_type bia_dt);
    };

    struct eltwise_po_handler_t {
        fill_status_t operator()(
                graph_prb_t &p, const attr_t::post_ops_t::entry_t &po_entry);
    };

    struct binary_po_handler_t {
        fill_status_t operator()(graph_prb_t &p, const std::string &dst_dataf,
                const attr_t::post_ops_t::entry_t &po_entry);
    };

    struct sum_po_handler_t {
        fill_status_t operator()(graph_prb_t &p);
    };

public:
    union {
        struct {
            bias_po_handler_t bias_handler;
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
        } matmul;

        struct {
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
        } binary;
    };
};

} // namespace benchdnnext

#endif
