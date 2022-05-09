/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
#include <cctype>
#include <cstddef>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_graph_threadpool.hpp"
#include "test_thread.hpp"
#endif

#if DNNL_GRAPH_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif

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
    UNSUPPORTED_CONFIG,
    UNKNOWN_ERROR
};
typedef fill_status fill_status_t;

bool check_graph_creation_status(const struct graph_prb_t *prb, res_t *res);
void check_known_skipped_case_graph_common(
        const std::vector<dnnl_data_type_t> &v_dt, const std::string &tag,
        const dir_t &dir, res_t *res);
void check_graph_eltwise_post_ops(const attr_t &attr, res_t *res);
void check_graph_eltwise_params(res_t *res,
        const attr_t::post_ops_t::kind_t alg, const float alpha,
        const float beta);
float get_post_eltwise_scale(
        const std::vector<attr_t::post_ops_t::entry_t> &post_ops) noexcept;
dnnl::graph::logical_tensor::data_type convert_dt(
        const dnnl_data_type_t dt) noexcept;
dnnl_data_type_t convert_dt(
        const dnnl::graph::logical_tensor::data_type dt) noexcept;
dnnl::graph::op::kind convert_alg_kind(
        const dnnl_alg_kind_t kind, bool is_fwd = true) noexcept;
std::string convert_attr_policy(const attr_t::policy_t policy) noexcept;
dims_t convert_bin_policy(
        const dims_t &lhs_dims, const attr_t::policy_t policy) noexcept;
std::map<std::string, float> convert_eltw_entry(
        const dnnl::graph::op::kind op_kind,
        const attr_t::post_ops_t::entry_t &entry) noexcept;

int scale_bia(dnn_mem_t &dst, dnn_mem_t &src, const std::vector<float> &scales);

dnnl_format_tag_t dnnl_fmt_str2tag(const std::string &fmt_str);

std::vector<size_t> get_post_bin_indices(
        const std::vector<attr_t::post_ops_t::entry_t> &po_entry);

template <typename prb_t>
void skip_invalid_and_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_invalid_prb(prb, res);
    if (res->state == SKIPPED) return;
    skip_unimplemented_prb(prb, res);
}

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

#define BENCHDNNEXT_VERIFY(status) \
    if ((status) != fill_status::DONE) return status

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
    using property_type = dnnl::graph::logical_tensor::property_type;
    tensor_descs_t() = default;

    template <typename... Args>
    void emplace(std::string str, Args... args) {
        dnnl::graph::logical_tensor t(idmgr_[str], std::forward<Args>(args)...);
        map_.emplace(str, t);
    }

    void emplace(std::string str, dt dtype, const dims_t &adims,
            const std::string &atag,
            property_type ptype = property_type::undef) {
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
            emplace(str, dtype, adims, lt::strided, ptype);
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

        if (fmt_tag <= dnnl_abcdefghijlk) {
            dims_t strides(ndims, 0);
            dim_t acc = 1;
            // if the layout can be described with strides, let's calculate
            // the strides according to the given tag.
            for (int d = static_cast<int>(ndims) - 1; d >= 0; --d) {
                const size_t coord = static_cast<size_t>(ou_fmt_str[d] - 'a');
                strides[coord] = acc;
                acc *= adims[coord];
            }
            emplace(str, dtype, adims, strides, ptype);
        } else {
            const size_t dnnl_layout_id
                    = encode_dnnl_layout(static_cast<size_t>(fmt_tag));
            dnnl::graph::logical_tensor t(
                    idmgr_[str], dtype, adims, dnnl_layout_id, ptype);
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
const dnnl::graph::engine &get_test_engine();
const dnnl::graph::stream &get_test_stream();

#if DNNL_GRAPH_WITH_SYCL
void *sycl_alloc(size_t n, const void *dev, const void *ctx,
        dnnl::graph::allocator::attribute attr);
void sycl_free(void *ptr, const void *ctx);
dnnl::graph::engine &get_engine();
#endif // DNNL_GRAPH_WITH_SYCL

// Engine used to run oneDNN fusion patterns for testing.
inline dnnl_engine_kind_t &get_test_engine_kind() {
    return engine_tgt_kind;
}

void compiled_partition_executor(dnnl::graph::compiled_partition &cp,
        dnnl::graph::stream &stream,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int measure_cmpl(timer::timer_t &t, const dnnl::graph::engine &engine,
        const dnnl::graph::partition &par,
        const std::vector<dnnl::graph::logical_tensor> &inputs,
        const std::vector<dnnl::graph::logical_tensor> &outputs);

int execute_and_wait(perf_function_t &exec_func,
        const dnnl::graph::engine &engine,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int execute_and_wait(dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs, res_t *res);

int measure_perf(timer::timer_t &t, perf_function_t &perf_func,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int measure_perf(timer::timer_t &t, dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int measure_perf(timer::timer_t &t, dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs, res_t *res);

int measure_partition_compl(timer::timer_t &t,
        const dnnl::graph::partition &par,
        const std::vector<dnnl::graph::logical_tensor> &inputs,
        const std::vector<dnnl::graph::logical_tensor> &outputs,
        const dnnl::graph::engine &engine);

template <typename func_t, typename prb_t>
dnnl::graph::compiled_partition compile_partition(const func_t &init_pd_func,
        prb_t *prb, res_t *res, const dnnl::graph::partition &par,
        const std::vector<dnnl::graph::logical_tensor> &inputs,
        const std::vector<dnnl::graph::logical_tensor> &outputs) {
    const dnnl::graph::engine &engine = get_test_engine();
    auto cp = par.compile(inputs, outputs, engine);

    if (is_bench_mode(PERF)) {
        // Measure parititon compilation perf.
        measure_partition_compl(
                res->timer_map.par_compl_timer(), par, inputs, outputs, engine);

        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> user_prim;
        measure_prim_create(res->timer_map.prim_create_timer(), user_prim,
                init_pd_func, prb, res);
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

    bool has_post_bia() const noexcept { return has_post_bia_; }
    bool has_post_bin() const noexcept { return has_post_bin_; }
    bool has_post_dw() const noexcept { return has_post_dw_; }
    bool has_post_sum() const noexcept { return has_post_sum_; }
    bool has_post_eltwise() const noexcept { return has_post_eltwise_; }
    bool with_quantization() const noexcept { return with_quantization_; }

    fill_status_t ctor_status;

protected:
    std::vector<dnnl::graph::op> ops_;
    tensor_descs_t tensor_descs_;
    std::unordered_map<std::string, std::vector<std::string>> tensor_id;

    std::vector<std::string> curr_out_map_ids_;

    bool has_post_bia_ {false};
    bool has_post_bin_ {false};
    bool has_post_dw_ {false};
    bool has_post_sum_ {false};
    bool has_post_eltwise_ {false};
    bool with_quantization_ {false};

    friend struct po_handlers_t;
};

struct low_precision_attr {
    const dt src_dt;
    const dt wei_dt;
    const dt dst_dt;
    const std::string &stag;
    const std::string &wtag;
    const std::string &dtag;
    dims_t wei_strides;
    const attr_t::policy_t &oscale_policy;
    std::vector<float> *oscales;
    const std::vector<int64_t> *src_zp;
    const std::vector<int64_t> *wei_zp;
    const std::vector<int64_t> *dst_zp;

    const float *scales;
    const int n_oc;
    const bool with_typecast;
    const bool def_oscales;
    const float dst_scale;

    // For matmul, conv and deconv
    static low_precision_attr lp_attr(const dt src_dt, const dt wei_dt,
            const dt dst_dt, const std::string &stag, const std::string &wtag,
            const std::string &dtag, const attr_t::policy_t &oscale_policy,
            std::vector<float> *oscales, const float &dst_scale,
            const std::vector<int64_t> *src_zp,
            const std::vector<int64_t> *wei_zp,
            const std::vector<int64_t> *dst_zp, const float *scales,
            const int &n_oc, const bool &def_oscales,
            const bool &with_typecast = false) {
        return low_precision_attr(src_dt, wei_dt, dst_dt, stag, wtag, dtag,
                oscale_policy, oscales, dst_scale, src_zp, wei_zp, dst_zp,
                scales, n_oc, def_oscales, with_typecast);
    };

    // For op with no additional attributes e.g. pool
    static low_precision_attr lp_attr(
            const dt src_dt, const dt dst_dt, const std::string &common_tag) {
        return low_precision_attr(
                src_dt, dt::undef, dst_dt, common_tag, common_tag, common_tag);
    };

    // For op with src and dst data types and src and data formats e.g. concat
    static low_precision_attr lp_attr(const dt src_dt, const dt dst_dt,
            const std::string &stag, const std::string &dtag) {
        return low_precision_attr(src_dt, dt::undef, dst_dt, stag, "", dtag);
    };

    // For op with only one data type e.g. eltwise
    static low_precision_attr lp_attr(
            const dt data_type, const std::string &common_tag) {
        return low_precision_attr(data_type, dt::undef, data_type, common_tag,
                common_tag, common_tag);
    };

    void set_wei_strides(const dims_t &wei_dims) {
        this->wei_strides = wei_dims;
    }

private:
    low_precision_attr(const dt src_dt, const dt wei_dt, const dt dst_dt,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag,
            const attr_t::policy_t &oscale_policy = attr_t::policy_t::COMMON,
            std::vector<float> *oscales = nullptr, const float dst_scale = 1.f,
            const std::vector<int64_t> *src_zp = nullptr,
            const std::vector<int64_t> *wei_zp = nullptr,
            const std::vector<int64_t> *dst_zp = nullptr,
            const float *scales = nullptr, const int n_oc = 0,
            const bool def_oscales = false, const bool with_typecast = false)
        : src_dt(src_dt)
        , wei_dt(wei_dt)
        , dst_dt(dst_dt)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , wei_strides(0)
        , oscale_policy(oscale_policy)
        , oscales(oscales)
        , src_zp(src_zp)
        , wei_zp(wei_zp)
        , dst_zp(dst_zp)
        , scales(scales)
        , n_oc(n_oc)
        , with_typecast(with_typecast)
        , def_oscales(def_oscales)
        , dst_scale(dst_scale) {};
};

struct po_handlers_t {
    using dt = dnnl::graph::logical_tensor::data_type;
    using lt = dnnl::graph::logical_tensor::layout_type;

private:
    struct eltwise_po_handler_t {
        fill_status_t operator()(graph_prb_t &p,
                const attr_t::post_ops_t::entry_t &po_entry,
                bool allow_swish_fuse = false);
    };

    struct binary_po_handler_t {
        fill_status_t operator()(
                graph_prb_t &p, const attr_t::post_ops_t::entry_t &po_entry);
    };

    struct sum_po_handler_t {
        fill_status_t operator()(graph_prb_t &p);
    };

    struct low_precision_handler_t {
        fill_status handle_low_precision_src(
                graph_prb_t &p, const low_precision_attr &lp_attr);
        fill_status handle_low_precision_srcs(graph_prb_t &p,
                const low_precision_attr &lp_attr, const size_t num_srcs);
        fill_status handle_low_precision_wei(
                graph_prb_t &p, const low_precision_attr &lp_attr);
        fill_status handle_low_precision_dst(
                graph_prb_t &p, const low_precision_attr &lp_attr);
        fill_status handle_low_precision_post_sum(graph_prb_t &p,
                const low_precision_attr &lp_attr,
                const std::vector<attr_t::post_ops_t::entry_t> &po_entry);
        fill_status handle_low_precision_post_bin(graph_prb_t &p,
                const low_precision_attr &lp_attr,
                const std::vector<attr_t::post_ops_t::entry_t> &po_entry);
    };

public:
    union {
        struct {
            low_precision_handler_t low_precision_handler;
        } concat;

        struct {
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
            binary_po_handler_t bin_handler;
            low_precision_handler_t low_precision_handler;
        } conv;

        struct {
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
            binary_po_handler_t bin_handler;
            low_precision_handler_t low_precision_handler;
        } deconv;

        struct {
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
            binary_po_handler_t bin_handler;
            low_precision_handler_t low_precision_handler;
        } matmul;

        struct {
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
            binary_po_handler_t bin_handler;
        } binary;

        struct {
            binary_po_handler_t bin_handler;
            low_precision_handler_t low_precision_handler;
        } eltwise;

        struct {
            eltwise_po_handler_t eltw_handler;
        } bnorm;

        struct {
            binary_po_handler_t bin_handler;
            low_precision_handler_t low_precision_handler;
        } pool;

        struct {
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
            binary_po_handler_t bin_handler;
        } reduction;

        struct {
            sum_po_handler_t sum_handler;
        } reorder;

        struct {
            eltwise_po_handler_t eltw_handler;
            sum_po_handler_t sum_handler;
            binary_po_handler_t bin_handler;
        } resampling;
    };
};

} // namespace benchdnnext

#endif
