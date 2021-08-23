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

#include <algorithm>
#include <cstddef>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "dnn_graph_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

namespace benchdnnext {

using dims_t = dnnl::graph::logical_tensor::dims_t;
using dim_t = dims_t::value_type;

using dt = dnnl::graph::logical_tensor::data_type;
using lt = dnnl::graph::logical_tensor::layout_type;

enum class fill_status {
    DONE, // everything was fine
    UNHANDLED_CONFIG_OPTIONS, // conversion done, but too much options was provided
    UNSUPPORTED_OP,
    UNKNOWN_ERROR
};
typedef fill_status fill_status_t;

void check_known_skipped_case_graph_common(
        const std::vector<dnnl_data_type_t> &v_dt, const std::string &tag,
        const dir_t &dir, res_t *res);
dnnl::graph::logical_tensor::data_type convert_dt(
        const dnnl_data_type_t dt) noexcept;
dnnl::graph::op::kind convert_alg_kind(const dnnl_alg_kind_t kind) noexcept;
std::string convert_tag(
        const std::string &tag, bool activation_tag = true) noexcept;
dims_t convert_bin_policy(const dims_t &lhs_dims, const attr_t::policy_t policy,
        const std::string &data_format) noexcept;
std::map<std::string, float> convert_eltw_entry(
        const dnnl::graph::op::kind op_kind,
        const attr_t::post_ops_t::entry_t &entry) noexcept;
bool should_handle_swish(
        struct graph_prb_t &p, const dnnl_alg_kind_t kind) noexcept;

int scale_bia(dnn_mem_t &dst, dnn_mem_t &src, const std::vector<float> &scales);

dnnl_format_tag_t dnnl_fmt_str2tag(const std::string &fmt_str);

/**  Get the outer format string without blocking expression.
 *     For example, AcdB16a8b -> acdb
 */
inline std::string get_ou_format(const std::string &fmt_tag) {
    size_t pos = 0;
    while (pos < fmt_tag.size() && !std::isdigit(fmt_tag[pos]))
        pos++;
    std::string fmt(fmt_tag.begin(), fmt_tag.begin() + pos);
    // convert the string to lower case for the convenient of deriving
    // permutation keys.
    std::transform(fmt.begin(), fmt.end(), fmt.begin(),
            [](char c) { return static_cast<char>(std::tolower(c)); });
    return fmt;
}

inline std::string tag2data_format(const std::string &tag_) {
    if (tag_ == tag::any) return "NCX";
    std::string tag = normalize_tag(tag_);
    tag = get_ou_format(tag);
    std::string ret;
    for (size_t i = 0; i < tag.size(); ++i) {
        if (tag[i] == 'a') {
            ret += "N";
        } else if (tag[i] == 'b') {
            ret += "C";
        } else {
            if (ret.back() == 'X') continue;
            ret += "X";
        }
    }
    return ret;
}

inline void validate_data_format(const std::string &format_) {
    if (!(format_ == "NCX" || format_ == "NXC")) {
        []() {
            SAFE(FAIL, CRIT);
            return 0;
        }();
    }
}

inline std::string tag2filter_format(
        const std::string &tag_, bool with_groups = false) {
    if (tag_ == tag::any) return "OIX";
    std::string tag = normalize_tag(tag_);
    tag = get_ou_format(tag);
    std::string ret;

    char out_c_repr = 'a';
    char in_c_repr = 'b';
    if (with_groups) {
        // groups becomes 'a', OC and IC take next values
        ++out_c_repr;
        ++in_c_repr;
    }
    for (size_t i = 0; i < tag.size(); ++i) {
        if (tag[i] == out_c_repr) {
            ret += "O";
        } else if (tag[i] == in_c_repr) {
            ret += "I";
        } else if (with_groups && tag[i] == 'a') {
            continue;
        } else {
            if (ret.back() == 'X') continue;
            ret += "X";
        }
    }
    return ret;
}

inline void validate_filter_format(const std::string &format_) {
    if (!(format_ == "XIO" || format_ == "OIX")) {
        []() {
            SAFE(FAIL, CRIT);
            return 0;
        }();
    }
}

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

inline size_t encode_dnnl_layout(size_t layout_idx) noexcept {
    // NOTE: Follows the definitions in oneDNN Graph implementation. And there
    //   is a assumption where the dnnl backend is registered in the first
    //   place. Can we expose any API to fetch the information of backend
    //   registry?
    static constexpr int backend_id_length = 4;
    static constexpr int dnnl_id = 1;

    size_t layout_id = (layout_idx << backend_id_length)
            | (dnnl_id & (size_t)((1 << backend_id_length) - 1));
    return layout_id;
}

struct tensor_descs_t {
    tensor_descs_t() = default;

    template <typename... Args>
    void emplace(std::string str, Args... args) {
        dnnl::graph::logical_tensor t(idmgr_[str], std::forward<Args>(args)...);
        map_.emplace(str, t);
    }

    void emplace(std::string str, dt dtype, const dims_t &adims,
            const std::string &atag) {
        size_t ndims = adims.size();
        const std::string dnnl_fmt_tag_str
                = normalize_tag(atag, static_cast<int>(ndims));
        const dnnl_format_tag_t fmt_tag = dnnl_fmt_str2tag(dnnl_fmt_tag_str);
        if (fmt_tag == dnnl_format_tag_undef) {
            []() {
                SAFE(FAIL, CRIT);
                return 0;
            }();
            return;
        }

        if (fmt_tag == dnnl_format_tag_any) {
            emplace(str, dtype, adims, lt::strided);
            return;
        }

#ifndef DNNL_GRAPH_LAYOUT_DEBUG
        if (fmt_tag > dnnl_abcdefghijlk) {
            []() {
                BENCHDNN_PRINT(0, "error: %s\n",
                        "to use dnnl opaque blocked formats, please build the "
                        "library with \"DNNL_GRAPH_LAYOUT_DEBUG=ON\"");
                SAFE(FAIL, CRIT);
                return 0;
            }();
        }
#endif // DNNL_GRAPH_LAYOUT_DEBUG

        const std::string ou_fmt_str = get_ou_format(dnnl_fmt_tag_str);

        static_assert(DNNL_GRAPH_MAX_NDIMS == DNNL_MAX_NDIMS,
                "Maximum number of dimensions of primitive and graph is not "
                "the same.");
        // The permutation keys is used to sort the order of graph dimensions
        // according to the format tag specified by benchdnn command line in
        // tag knob. If user makes a dnnl memory desc with a format tag, the
        // order of dimensions is always `abcdefg...`. Different tag, for
        // instance with `acdb` (comparing to `abcd`), logical tensor could be
        // created with either
        //     i) `acdb` order dimensions plus strided/opaque layout type, or
        //    ii) `abcd` order with strides [b*c*d, 1, d*b, b].
        //
        // NOTE: Currently, only the first method is used.
        dims_t permuted_dims(ndims, 0);
        for (size_t d = 0; d < ndims; ++d) {
            const size_t coord = static_cast<size_t>(ou_fmt_str[d] - 'a');
            permuted_dims[d] = adims[coord];
        }

        if (fmt_tag <= dnnl_abcdefghijlk) {
            emplace(str, dtype, permuted_dims, lt::strided);
        } else {
            const int64_t dnnl_layout_id
                    = encode_dnnl_layout(static_cast<size_t>(fmt_tag));
            dnnl::graph::logical_tensor t(
                    idmgr_[str], dtype, permuted_dims, dnnl_layout_id);
            map_.emplace(str, t);
        }
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

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt,
        const dims_t &dims,
        const dnnl::graph::logical_tensor::data_type &graph_dt,
        const char *atag);

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt,
        const dims_t &dims, const std::string &atag);

dnn_mem_t make_dnn_mem(
        const dnnl::graph::logical_tensor &lt, const char *tag = nullptr);

dnn_mem_t make_dnn_mem(
        const dnnl::graph::logical_tensor &lt, const std::string &tag);

dims_t calculate_strides(dims_t dims, dt dtype, const std::string &tag);

template <typename T, std::size_t N>
constexpr T *end(T (&arr)[N]) noexcept {
    return arr + N;
}

typedef std::function<void(const dnnl::graph::engine &,
        const std::vector<dnnl::graph::logical_tensor> &,
        const std::vector<dnnl::graph::logical_tensor> &)>
        cmpl_function_t;

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

int measure_cmpl(benchdnn_timer_t &t, const dnnl::graph::engine &engine,
        const dnnl::graph::partition &par,
        const std::vector<dnnl::graph::logical_tensor> &inputs,
        const std::vector<dnnl::graph::logical_tensor> &outputs);

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

int measure_partition_compl(benchdnn_timer_t &t,
        const dnnl::graph::partition &par,
        const std::vector<dnnl::graph::logical_tensor> &inputs,
        const std::vector<dnnl::graph::logical_tensor> &outputs,
        dnnl::graph::engine &engine);

template <typename func_t, typename prb_t>
dnnl::graph::compiled_partition compile_partition(const func_t &init_pd_func,
        prb_t *prb, res_t *res, const dnnl::graph::partition &par,
        const std::vector<dnnl::graph::logical_tensor> &inputs,
        const std::vector<dnnl::graph::logical_tensor> &outputs) {
    dnnl::graph::engine &engine = get_test_engine();
    auto cp = par.compile(inputs, outputs, engine);

    if (is_bench_mode(PERF)) {
        // Measure parititon compilation perf.
        measure_partition_compl(
                res->par_compl_timer, par, inputs, outputs, engine);

        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
        // This function call here is needed
        // to measure primitive creation time.
        init_prim(prim, init_pd_func, prb, res);
    }

    return cp;
}
struct graph_prb_t {
    using dt = dnnl::graph::logical_tensor::data_type;
    using lt = dnnl::graph::logical_tensor::layout_type;

    dnnl::graph::graph to_graph() {
        const dnnl::graph::engine &engine = benchdnnext::get_test_engine();
        dnnl::graph::graph graph(engine.get_kind());
        for (auto &&op : ops_)
            graph.add_op(op);
        return graph;
    }

    virtual dnnl::graph::op::kind get_main_op_kind() const = 0;

    bool has_post_bia() const noexcept { return has_post_bia_; }
    bool has_post_bin() const noexcept { return has_post_bin_; }
    bool has_post_sum() const noexcept { return has_post_sum_; }

protected:
    std::vector<dnnl::graph::op> ops_;
    tensor_descs_t tensor_descs_;
    std::unordered_map<std::string, std::vector<std::string>> tensor_id;

    std::vector<std::string> curr_out_map_ids_;

    bool has_post_bia_ {false};
    bool has_post_bin_ {false};
    bool has_post_sum_ {false};

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
            binary_po_handler_t bin_handler;
        } conv;

        struct {
            bias_po_handler_t bias_handler;
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
            binary_po_handler_t bin_handler;
        } matmul;

        struct {
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
            binary_po_handler_t bin_handler;
        } binary;

        struct {
            binary_po_handler_t bin_handler;
        } eltwise;

        struct {
            eltwise_po_handler_t eltw_handler;
        } bnorm;
    };
};

} // namespace benchdnnext

#endif
