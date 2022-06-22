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
#include <utility>
#include <vector>
#include <unordered_map>

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
#include "test_allocator.hpp"

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
void check_graph_scales_and_zps_support(const attr_t &attr, res_t *res);
void check_graph_eltwise_params(res_t *res,
        const attr_t::post_ops_t::kind_t alg, const float alpha,
        const float beta);
std::vector<float> get_scales(const attr_t::scale_t &scales_info,
        const float *raw_scales, int64_t channel_size);
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
dnnl::graph::graph::fpmath_mode convert_fpmath_mode(
        const dnnl_fpmath_mode_t mode) noexcept;

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
    using fpmath_mode = dnnl::graph::graph::fpmath_mode;

    dnnl::graph::graph to_graph(fpmath_mode mode = fpmath_mode::strict) {
        const dnnl::graph::engine &engine = benchdnnext::get_test_engine();
        dnnl::graph::graph graph(engine.get_kind(), mode);
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

struct quant_data_t {
    quant_data_t() = default;
    quant_data_t(dnnl::graph::logical_tensor::data_type q_dt,
            const std::string &q_tag)
        : dt(q_dt), tag(q_tag) {}
    quant_data_t(dnnl::graph::logical_tensor::data_type q_dt,
            const std::vector<float> &q_scales,
            const std::vector<int64_t> &q_zps, const std::string &q_tag)
        : dt(q_dt), scales(q_scales), zps(q_zps), tag(q_tag) {}
    quant_data_t(dnnl::graph::logical_tensor::data_type q_dt,
            const std::vector<float> &q_scales,
            const std::vector<int64_t> &q_zps, const std::string &q_type,
            int64_t q_axis, const std::string &q_tag)
        : dt(q_dt)
        , scales(q_scales)
        , zps(q_zps)
        , qtype(q_type)
        , axis(q_axis)
        , tag(q_tag) {}
    quant_data_t(dnnl::graph::logical_tensor::data_type q_dt,
            const std::vector<float> &q_scales,
            const std::vector<int64_t> &q_zps, const std::string &q_type,
            int64_t q_axis, const std::vector<int64_t> &q_strides)
        : dt(q_dt)
        , scales(q_scales)
        , zps(q_zps)
        , qtype(q_type)
        , axis(q_axis)
        , strides(q_strides) {}

    dnnl::graph::logical_tensor::data_type dt {
            dnnl::graph::logical_tensor::data_type::s8};
    std::vector<float> scales {1.f};
    std::vector<int64_t> zps {0L};
    std::string qtype {"per_tensor"};
    int64_t axis {0};

    std::vector<int64_t> strides {};
    std::string tag {""};
};

quant_data_t sum_po_entry2quant_data(const attr_t::post_ops_t::entry_t &e,
        const std::string &tag,
        dnnl::graph::logical_tensor::data_type default_dt);
quant_data_t bin_po_entry2quant_data(const attr_t::post_ops_t::entry_t &e,
        const std::string &tag,
        dnnl::graph::logical_tensor::data_type default_dt);

bool is_dequantize_required_for(const attr_t::post_ops_t::entry_t &e);

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
        fill_status_t insert_dequant_before(const std::string &lt_id,
                const quant_data_t &qdata, graph_prb_t &p,
                bool as_constant = false);
        fill_status_t insert_quant_after(const std::string &lt_id,
                const quant_data_t &qdata, graph_prb_t &p);
        fill_status_t handle_quant_dequant_(const std::string &lt_id,
                const quant_data_t &qdata, graph_prb_t &p, bool as_constant,
                dnnl::graph::op::kind op_kind);
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

fill_status_t append_graph_with_eltwise(
        const attr_t::post_ops_t::entry_t &eltw_entry);
std::pair<fill_status_t, size_t> append_graph_with_binary(
        const attr_t::post_ops_t::entry_t &bin_entry);
std::pair<fill_status_t, size_t> append_graph_with_sum(
        const attr_t::post_ops_t::entry_t &bin_entry);

fill_status_t append_graph_with_swish(
        const attr_t::post_ops_t::entry_t &swish_entry, size_t src1_id);

std::pair<fill_status_t, size_t> insert_typecast_before(
        size_t dst_id, bool as_constant = false);

fill_status_t insert_dequant_before(
        size_t lt_id, const quant_data_t &qdata, bool as_constant = false);
fill_status_t insert_quant_after(size_t lt_id, const quant_data_t &qdata);
fill_status_t handle_quant_dequant(size_t lt_id, const quant_data_t &qdata,
        bool as_constant, dnnl::graph::op::kind op_kind);

inline bool is_plain(dnnl_format_tag_t fmt_tag) {
    return fmt_tag >= dnnl_a && fmt_tag <= dnnl_abcdefghijlk;
}

enum class entry_kind {
    NONE = 0,
    BINARY,
    BNORM,
    CONCAT,
    CONV,
    DECONV,
    ELTWISE,
    LNORM,
    MATMUL,
    POOL,
    PRELU,
    REDUCTION,
    REORDER,
    RESAMPLING,
    SOFTMAX,
    SUM,
    // handling data other than f32
    QUANTIZE,
    DEQUANTIZE,
    TYPECAST,
    // handling special case - shuffle
    RESHAPE,
    TRANSPOSE
};
typedef entry_kind entry_kind_t;
std::string entry_kind2str(entry_kind_t ekind);

enum class lt_kind {
    NONE = 0,
    SRC,
    SRC1,
    WEI,
    BIA,
    DST,
    MEAN,
    RUN_MEAN,
    BATCH_MEAN,
    VAR,
    RUN_VAR,
    BATCH_VAR,
    SC,
    SH,
    DIFF_SRC,
    DIFF_WEI,
    DIFF_DST,
    DIFF_SC,
    DIFF_SH,
    SRC_I
};
typedef lt_kind lt_kind_t;
std::string lt_kind2str(lt_kind_t lkind);

struct id_mgr_t {
    size_t hash(entry_kind_t ekind, size_t graph_pos) const noexcept {
        size_t graph_pos_part = graph_pos << graph_pos_offset_;
        return static_cast<size_t>(ekind) + graph_pos_part;
    }

    size_t hash(size_t op_id, lt_kind_t ltkind) const noexcept {
        size_t lt_kind_part = static_cast<size_t>(ltkind) << lt_kind_offset_;
        return op_id + lt_kind_part;
    }

    size_t increment_lt_part(size_t aid, size_t val) const noexcept {
        return aid + (val << lt_kind_offset_);
    }

    size_t retrieve_entry_kind_part(size_t aid) const noexcept {
        return aid & entry_kind_mask_;
    }

    size_t retrieve_lt_kind_part(size_t aid) const noexcept {
        return (aid & lt_kind_mask_) >> lt_kind_offset_;
    }

    size_t retrieve_graph_pos_part(size_t aid) const noexcept {
        return (aid & graph_pos_mask_) >> graph_pos_offset_;
    }

    std::string stringify_id(size_t aid) const noexcept {
        size_t entry_kind_part = retrieve_entry_kind_part(aid);
        size_t lt_kind_part = retrieve_lt_kind_part(aid);
        size_t graph_pos_part = retrieve_graph_pos_part(aid);
        std::string id_as_str;
        if (entry_kind_part)
            id_as_str += entry_kind2str(
                                 static_cast<entry_kind_t>(entry_kind_part))
                    + "_";
        if (lt_kind_part)
            id_as_str
                    += lt_kind2str(static_cast<lt_kind_t>(lt_kind_part)) + "_";
        if (!id_as_str.empty()) id_as_str += std::to_string(graph_pos_part);
        return id_as_str;
    }

private:
    // entry_kind occupies first 5-bits
    const size_t lt_kind_offset_ {0x5};
    // 5 bits for entry_kind + 6 bits for lt_kind
    const size_t graph_pos_offset_ {0xb};
    //              11111 <- LSB
    const size_t entry_kind_mask_ {0x1f};
    //        11111100000 <- LSB
    const size_t lt_kind_mask_ {0x7e0};
    // 111111100000000000 <- LSB
    const size_t graph_pos_mask_ {0x3f800};
};

struct graph_t {
    static graph_t &get() {
        static graph_t g;
        return g;
    }

    size_t generate_id_for(entry_kind_t ekind) const {
        return id_mgr_.hash(ekind, ops_.size());
    }

    size_t generate_id_for(size_t op_id, lt_kind_t lkind,
            bool prefer_append = false) const noexcept {
        if (prefer_append)
            return cur_block_out_id_;
        else
            return id_mgr_.hash(op_id, lkind);
    }

    size_t generate_id_for(size_t op_id, lt_kind_t lkind, size_t occurrence,
            bool prefer_append = false) const noexcept {
        if (occurrence == 0 || lkind != lt_kind::SRC_I)
            return generate_id_for(op_id, lkind);
        return id_mgr_.increment_lt_part(
                id_mgr_.hash(op_id, lkind), occurrence);
    }

    std::string stringify_id(size_t aid) const {
        return id_mgr_.stringify_id(aid);
    }

    template <typename... Args>
    void create_lt(size_t aid, Args... args) {
        dnnl::graph::logical_tensor lt(aid, std::forward<Args>(args)...);
        lts_.emplace(aid, lt);
    }

    void create_lt(size_t aid, const dnnl::graph::logical_tensor &lt) {
        if (lt.get_layout_type()
                == dnnl::graph::logical_tensor::layout_type::opaque)
            create_lt(aid, lt.get_data_type(), lt.get_dims(),
                    lt.get_layout_id(), lt.get_property_type());
        else
            create_lt(aid, lt.get_data_type(), lt.get_dims(), lt.get_strides(),
                    lt.get_property_type());
    }

    void create_lt(size_t aid, dt dtype, const dims_t &adims,
            const std::string &atag,
            dnnl::graph::logical_tensor::property_type ptype
            = dnnl::graph::logical_tensor::property_type::undef);

    void append(size_t op_id, dnnl::graph::op &aop,
            const std::vector<size_t> &src_ids,
            const std::vector<size_t> &dst_ids, bool is_after = true) {
        for (auto src_id : src_ids)
            aop.add_input(get_lt(src_id));
        for (auto dst_id : dst_ids)
            aop.add_output(get_lt(dst_id));
        ops_.emplace_back(aop);

        if (is_after) cur_block_out_id_ = dst_ids.front();
    }

    dnnl::graph::logical_tensor get_lt(size_t lt_id) const {
        if (lts_.count(lt_id))
            return lts_.at(lt_id);
        else
            return get_empty_lt(lt_id);
    }

    bool has_blocks() const noexcept { return last_block_out_id_ > 0; }

    size_t get_last_block_out_id() const noexcept { return last_block_out_id_; }

    size_t get_cur_block_out_id() const noexcept { return cur_block_out_id_; }

    std::vector<dnnl::graph::partition> get_partitions(
            dnnl::graph::graph::fpmath_mode mode
            = dnnl::graph::graph::fpmath_mode::strict,
            dnnl::graph::partition::policy p
            = dnnl::graph::partition::policy::fusion) const {
        const dnnl::graph::engine &engine = benchdnnext::get_test_engine();
        dnnl::graph::graph graph(engine.get_kind(), mode);
        for (auto &&op : ops_)
            graph.add_op(op);
        return graph.get_partitions(p);
    }

    void update_lts(const std::vector<dnnl::graph::logical_tensor> &lts) {
        for (const auto &lt : lts)
            if (lts_.count(lt.get_id())) lts_[lt.get_id()] = lt;
    }

    void close_block() noexcept { last_block_out_id_ = cur_block_out_id_; }

    void clear() noexcept {
        ops_.clear();
        lts_.clear();
        cur_block_out_id_ = 0;
        last_block_out_id_ = 0;
    };

private:
    graph_t() = default;
    ~graph_t() = default;
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(graph_t);

    dnnl::graph::logical_tensor get_empty_lt(size_t aid) const {
        BENCHDNN_PRINT(0,
                "warning: empty logical tensors was returned for id: %s\n",
                stringify_id(aid).c_str());
        return dnnl::graph::logical_tensor();
    }

    std::vector<dnnl::graph::op> ops_;
    std::unordered_map<size_t, dnnl::graph::logical_tensor> lts_;

    id_mgr_t id_mgr_;

    size_t cur_block_out_id_ {0};
    size_t last_block_out_id_ {0};
};

inline void cleanup() {
    graph_t::get().clear();
}

} // namespace benchdnnext

#endif
