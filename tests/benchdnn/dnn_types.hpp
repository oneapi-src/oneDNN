/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
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

#ifndef DNN_TYPES_HPP
#define DNN_TYPES_HPP

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "common.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "utils/data_kind.hpp"
#include "utils/wrapper.hpp"

namespace tag {
extern const char *x;
extern const char *abx;
extern const char *axb;
extern const char *any;
extern const char *undef;
} // namespace tag

/* TODO: merge prop and dir_t (in favor of prop) */
const char *prop2str(dnnl_prop_kind_t prop);
dnnl_prop_kind_t prop2prop_kind(dir_t dir);
int str2arg(const std::string &str);
std::string arg2str(int arg);

std::ostream &operator<<(std::ostream &s, dir_t dir);
std::ostream &operator<<(std::ostream &s, dnnl_data_type_t dt);
std::ostream &operator<<(std::ostream &s, dnnl_engine_kind_t ek);
template <typename T>
std::ostream &operator<<(std::ostream &s, const std::vector<T> &v) {
    s << v[0];
    for (size_t d = 1; d < v.size(); ++d)
        s << ":" << v[d];
    return s;
}

struct attr_t {
    // policy_t defines the way entity values will be applied to a tensor
    enum policy_t {
        COMMON = 0, // single value for each point in a tensor
        // apply a single value per...
        PER_OC, // channel (dims[1]) point
        PER_OCIC, // channel (dims[0] and dims[1]) point
        PER_DIM_0, // ... dims[0] point.
        PER_DIM_1, // ... dims[1] point.
        PER_DIM_01, // ... unique combination of dims[0] and dims[1] points.
        PER_DIM_2, // ... dims[2] point.
        PER_DIM_3, // ... dims[3] point.
        PER_TENSOR, // ... point in the tensor.
        POLICY_TOTAL // guard
    };

    static policy_t str2policy(const std::string &str);
    static const char *policy2str(policy_t policy);
    static int get_default_mask(policy_t policy);
    static int policy2mask(int arg, policy_t policy,
            dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive,
            int ndims = -1, bool has_groups = false);

    struct zero_points_t {
        struct entry_t {
            entry_t(policy_t apolicy = COMMON, int avalue = 0,
                    dnnl_data_type_t adt = dnnl_s32,
                    const std::vector<dnnl_dim_t> &agroups = {})
                : policy(apolicy), value(avalue), dt(adt), groups(agroups) {}

            int from_str(const std::string &s);

            bool is_def() const {
                return policy == COMMON && value == 0 && dt == dnnl_s32
                        && groups.size() == 0;
            }

            policy_t policy = COMMON;
            int value = 0;
            dnnl_data_type_t dt = dnnl_s32;
            std::vector<dnnl_dim_t> groups;
        };

        int from_str(const std::string &s);

        int operator[](int arg) const { return get(arg).value; }

        bool is_def(int arg) const {
            return points.empty() || get(arg).is_def();
        }
        bool is_def() const {
            if (points.empty()) return true;

            bool def = true;
            for (const auto &e : points) {
                def = def && is_def(e.first);
            }
            return def;
        }

        void set(int arg, policy_t policy, int value) {
            set(arg, entry_t(policy, value));
        }
        void set(int arg, const entry_t &entry) {
            if (!entry.is_def()) points[arg] = entry;
        }
        void set(int arg, policy_t policy, int value,
                dnnl_data_type_t data_type, std::vector<dnnl_dim_t> &groups) {
            set(arg, entry_t(policy, value, data_type, groups));
        }
        entry_t get(int arg) const {
            const auto it = points.find(arg);
            return it == points.end() ? entry_t() : it->second;
        }

        int get_mask(int arg,
                dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive,
                int ndims = -1, bool has_groups = false) const {
            const auto &e = get(arg);
            return attr_t::policy2mask(
                    arg, e.policy, prim_kind, ndims, has_groups);
        }

        zero_points_t() : points() {} // needed for debug icc190 build;
        std::map<int, entry_t> points;
    };

    struct arg_scales_t {
        struct entry_t {
            entry_t(policy_t apolicy = COMMON, float ascale = 1.f,
                    dnnl_data_type_t adt = dnnl_f32,
                    const std::vector<dnnl_dim_t> &agroups = {})
                : policy(apolicy), scale(ascale), dt(adt), groups(agroups) {}

            int from_str(const std::string &s);

            bool is_def() const {
                return policy == COMMON && scale == 1.f && dt == dnnl_f32
                        && groups.size() == 0;
            }

            policy_t policy = COMMON;
            float scale = 1.f;
            dnnl_data_type_t dt = dnnl_f32;
            std::vector<dnnl_dim_t> groups;
        };

        void set(int arg, entry_t scale) { scales[arg] = scale; }

        entry_t get(int arg) const {
            const auto &s = scales.find(arg);
            return s == scales.end() ? entry_t() : s->second;
        }

        int get_mask(int arg,
                dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive,
                int ndims = -1, bool has_groups = false) const {
            const auto &e = get(arg);
            return attr_t::policy2mask(
                    arg, e.policy, prim_kind, ndims, has_groups);
        }

        bool is_def(int arg) const {
            return scales.empty() || get(arg).is_def();
        }
        bool is_def() const {
            if (scales.empty()) return true;

            bool def = true;
            for (const auto &e : scales) {
                def = def && is_def(e.first);
            }
            return def;
        }
        int from_str(const std::string &s);

        arg_scales_t() : scales() {} // needed for debug icc190 build;

        std::map<int, entry_t> scales;
    };

    struct rounding_mode_t {
        dnnl_rounding_mode_t get(int arg) const {
            const auto &r = rounding_modes_.find(arg);
            return r == rounding_modes_.end() ? dnnl_rounding_mode_environment
                                              : r->second;
        }
        void set(int arg, dnnl_rounding_mode_t rm) {
            rounding_modes_[arg] = rm;
        }
        void set_seed(uint32_t s) {
            if (is_set_seed && seed != s)
                BENCHDNN_PRINT(0, "%s\n",
                        "WARN: rounding seed has to be the same for all "
                        "arguments");
            is_set_seed = true;
            seed = s;
        }
        bool is_def() const { return rounding_modes_.empty(); }

        std::map<int, dnnl_rounding_mode_t> rounding_modes_;
        // Setting seed to all ones for better entropy
        uint32_t seed = -1;
        bool is_set_seed = false;
    };

    struct post_ops_t {
        enum kind_t {
            // sum
            SUM,
            // depthwise convolution
            DW,
            // eltwise
            ELTWISE_START, // a guard to check kind is eltwise
            ABS,
            CLIP,
            CLIP_V2,
            CLIP_V2_DST,
            ELU,
            ELU_DST,
            EXP,
            EXP_DST,
            GELU_ERF,
            GELU_TANH,
            HARDSIGMOID,
            HARDSWISH,
            LINEAR,
            LOG,
            LOGISTIC,
            LOGISTIC_DST,
            MISH,
            POW,
            RELU,
            RELU_DST,
            ROUND,
            SQRT,
            SQRT_DST,
            SQUARE,
            SRELU,
            SWISH,
            TANH,
            TANH_DST,
            ELTWISE_END, // a guard to check kind is eltwise
            // binary
            BINARY_START, // a guard to check kind is binary
            ADD,
            DIV,
            EQ,
            GE,
            GT,
            LE,
            LT,
            MAX,
            MIN,
            MUL,
            NE,
            SELECT,
            SUB,
            BINARY_END, // a guard to check kind is binary
            // prelu
            PRELU,
            // guard entry
            KIND_TOTAL
        };
        static kind_t str2kind(const std::string &str);
        static const char *kind2str(kind_t kind);
        static dnnl_alg_kind_t kind2dnnl_kind(kind_t kind);

        struct entry_t {
            entry_t(kind_t akind) : kind(akind) {
                if (is_sum_kind()) {
                } else if (is_eltwise_kind()) {
                    eltwise.alg = kind2dnnl_kind(kind);
                } else if (is_convolution_kind()) {
                } else if (is_binary_kind()) {
                    binary.alg = kind2dnnl_kind(kind);
                }
            }

            kind_t kind;
            struct {
                float scale = 1.f;
                int32_t zero_point = 0;
                dnnl_data_type_t dt = dnnl_data_type_undef;
            } sum;
            struct {
                dnnl_alg_kind_t alg = dnnl_alg_kind_undef;
                float alpha = 0.f;
                float beta = 0.f;
            } eltwise;
            struct {
                int kernel = 0;
                int stride = 0;
                int padding = 0;
                dnnl_data_type_t dst_dt = dnnl_f32;
            } convolution;
            struct binary_t {
                enum class mask_input_t {
                    none,
                    mask,
                    policy,
                };

                dnnl_alg_kind_t alg = dnnl_alg_kind_undef;
                dnnl_data_type_t src1_dt = dnnl_data_type_undef;
                policy_t policy = policy_t::COMMON;
                int64_t mask = -1;
                mask_input_t mask_input = mask_input_t::none;
                std::string tag = tag::any;
            } binary;
            struct {
                policy_t policy = policy_t::COMMON;
            } prelu;

            bool is_sum_kind() const;
            bool is_convolution_kind() const;
            bool is_eltwise_kind() const;
            bool is_binary_kind() const;
            bool is_prelu_kind() const;
        };

        post_ops_t() : entry() {}

        int len() const { return (int)entry.size(); }
        bool is_def() const { return len() == 0; }

        int find(kind_t kind, int start = 0, int stop = -1) const;
        int eltwise_index() const;
        int convolution_index() const;
        int binary_index() const;
        int prelu_index() const;

        // ndims must be provided for primitives that have po mask that depends
        // on the ndims. Currently this includes only lnorm with binary post-op
        // with policy PER_OC.
        // Some primitives might have a special handling for a policy provided.
        // For such primitives prim_kind must be set so get_po_masks generates
        // a correct mask. Currently this behavior depends on policy2mask().
        std::vector<std::pair<int, int>> get_po_masks(int ndims = -1,
                dnnl_primitive_kind_t prim_kind
                = dnnl_undefined_primitive) const;

        std::vector<entry_t> entry;
    };

    struct deterministic_t {
        // The default value is changed for a bitwise mode. To properly work,
        // `--mode=B` must be specified before the driver name, otherwise,
        // the driver settings will use `false` as the mode bit was not set
        // before entering the driver parsing section.
        deterministic_t() : enabled(has_bench_mode_bit(mode_bit_t::bitwise)) {}

        bool is_def() const {
            static deterministic_t def;
            return enabled == def.enabled;
        }

        bool enabled;
    };

    struct fpmath_mode_t {
        fpmath_mode_t() = default;

        void set(dnnl_fpmath_mode_t mode, bool apply_to_int = false) {
            this->mode = mode;
            this->apply_to_int = apply_to_int;
        }
        bool is_def() const {
            return mode == dnnl_fpmath_mode_strict && apply_to_int == false;
        }

        dnnl_fpmath_mode_t mode = dnnl_fpmath_mode_strict;
        bool apply_to_int = false;
    };

    struct dropout_t {
        float p = 0.f;
        uint32_t seed = 0;
        std::string tag = tag::any;
        bool is_def() const { return p == 0.f; }
    };

    attr_t()
        : scratchpad_mode(get_default_scratchpad_mode())
        , acc_mode(dnnl_accumulation_mode_strict) {}

    template <typename First, typename... Rest>
    void insert(const First &first, const Rest &...rest) {
        this->insert(first);
        if (sizeof...(rest) > 0) this->insert(rest...);
    }

    void insert(const arg_scales_t &as) { this->scales = as; }
    void insert(const zero_points_t &zp) { this->zero_points = zp; }
    void insert(const post_ops_t &po) { this->post_ops = po; }
    void insert(dnnl_scratchpad_mode_t sm) { this->scratchpad_mode = sm; }
    void insert(const fpmath_mode_t &fpm) { this->fpmath_mode = fpm; }
    void insert(dnnl_accumulation_mode_t am) { this->acc_mode = am; }
    void insert(const deterministic_t &d) { this->deterministic = d; }
    void insert(const dropout_t &d) { this->dropout = d; }
    void insert(const rounding_mode_t &rm) { this->rounding_mode = rm; }

    // When parallel creation modifier is enabled, the library scratchpad mode
    // can't be used unless "-DDNNL_ENABLE_CONCURRENT_EXEC=ON" is enabled at the
    // build time, otherwise scratchpad pointers are invalidated (as were
    // created inside threads that no longer exist when execution time comes).
    // Relevant for both engines since GPU uses CPU for faster validation.
    static dnnl_scratchpad_mode_t get_default_scratchpad_mode() {
        return has_bench_mode_modifier(mode_modifier_t::par_create)
                ? dnnl_scratchpad_mode_user
                : dnnl_scratchpad_mode_library;
    }

    arg_scales_t scales;
    zero_points_t zero_points;
    post_ops_t post_ops;
    dnnl_scratchpad_mode_t scratchpad_mode;
    fpmath_mode_t fpmath_mode;
    dnnl_accumulation_mode_t acc_mode;
    deterministic_t deterministic;
    dropout_t dropout;
    rounding_mode_t rounding_mode;

    bool is_def(bool skip_fpmath = false) const;
};

struct isa_hints_t {
    enum cpu_hints_t {
        // If DNNL_CPU_ISA_HINTS is set then use hints from there
        // Otherwise no hints
        none = 0x0,
        // No CPU ISA specific hints
        // Will override DNNL_CPU_ISA_HINTS if that is available too
        no_hints = 0x1,
        // Use prefer_ymm CPU ISA hint
        // Will override DNNL_CPU_ISA_HINTS if that is available too
        prefer_ymm = 0x2,
    };

    cpu_hints_t hints_;
    isa_hints_t(cpu_hints_t hints) : hints_(hints) {}

    cpu_hints_t get() { return hints_; }

    static std::string hints2str(const isa_hints_t &isa_hints) {
        switch (isa_hints.hints_) {
            case none: return "none";
            case no_hints: return "no_hints";
            case prefer_ymm: return "prefer_ymm";
            default: assert(!"unknown hint"); return "unknown_hint";
        }
    }

    static isa_hints_t str2hints(const char *str) {
        cpu_hints_t hints = none;

        if (strcasecmp(str, "prefer_ymm") == 0)
            hints = prefer_ymm;
        else if (strcasecmp(str, "no_hints") == 0)
            hints = no_hints;

        return isa_hints_t(hints);
    }
};

using policy_t = attr_t::policy_t;

struct sparse_options_t {
    static constexpr dnnl_sparse_encoding_t def_encoding
            = dnnl_sparse_encoding_undef;
    static constexpr float def_sparsity = 0.9f;

    sparse_options_t() = default;
    sparse_options_t(int arg, dnnl_sparse_encoding_t encoding, float sparsity) {
        add(arg, encoding, sparsity);
    }

    void add(int arg, dnnl_sparse_encoding_t encoding, float sparsity) {
        options_.insert({arg, {encoding, sparsity}});
    }

    dnnl_sparse_encoding_t get_encoding(int arg) const {
        if (options_.count(arg) == 0) return dnnl_sparse_encoding_undef;
        return options_.at(arg).first;
    }
    dnnl_sparse_encoding_t get_encoding(data_kind_t kind) const {
        // Note: the commented code doesn't work as `arg` returned is a
        // backward exec_arg. See the function comment.
        // const auto arg = data_kind2exec_arg(kind);
        // return get_encoding(arg);
        switch (kind) {
            case SRC: return get_encoding(DNNL_ARG_SRC);
            case WEI: return get_encoding(DNNL_ARG_WEIGHTS);
            default: return def_encoding;
        }
    }

    float get_sparsity(int arg) const {
        if (options_.count(arg) == 0) return 0.0f;
        return options_.at(arg).second;
    }

    bool is_encoding_def(int arg) const {
        return get_encoding(arg) == def_encoding;
    }

    bool is_sparsity_def(int arg) const {
        return get_sparsity(arg) == def_sparsity;
    }

    bool is_def() const {
        bool ret = true;
        for (const auto &opt : options_) {
            ret = ret && is_encoding_def(opt.first)
                    && is_sparsity_def(opt.first);
        }
        return ret;
    }

    std::vector<int> get_args() const {
        std::vector<int> args;
        for (const auto &opt : options_) {
            args.push_back(opt.first);
        }
        return args;
    }

    int from_str(const std::string &s);

private:
    std::unordered_map<int, std::pair<dnnl_sparse_encoding_t, float>> options_;
};

std::ostream &operator<<(
        std::ostream &s, const sparse_options_t &sparse_options);

std::ostream &operator<<(std::ostream &s, const policy_t &policy);
std::ostream &operator<<(
        std::ostream &s, const attr_t::zero_points_t &zero_points);
std::ostream &operator<<(std::ostream &s, const attr_t::arg_scales_t &scales);
std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t::kind_t &k);
std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t &post_ops);
std::ostream &operator<<(std::ostream &s, dnnl_scratchpad_mode_t sm);
std::ostream &operator<<(std::ostream &s, const attr_t::fpmath_mode_t &fm);
std::ostream &operator<<(std::ostream &s, dnnl_accumulation_mode_t am);
std::ostream &operator<<(std::ostream &s, dnnl_rounding_mode_t rm);
std::ostream &operator<<(std::ostream &s, const attr_t::dropout_t &drop);
std::ostream &operator<<(std::ostream &s, const attr_t &attr);

// A container for additional data and info, not available from user's input at
// parse time, but which are required to create the library attributes.
struct attr_args_t {
    struct dw_t {
        dnnl_data_type_t wei_dt = dnnl_data_type_undef;
        dnnl_data_type_t bia_dt = dnnl_data_type_undef;
    };

    attr_args_t() = default;

    void prepare_quant(const attr_t &attr, int arg, int mask = -1) {
        entries.insert(std::make_pair(arg, mask));
    };

    int prepare_post_ops_mds(const attr_t &attr, int ndims,
            const dnnl_dims_t prb_dims,
            dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive);

    void prepare_dw_post_op(const attr_t &attr, dnnl_data_type_t wei_dt,
            dnnl_data_type_t bia_dt);

    // Returns mask set for correspondent `arg`. The default value is `-1`.
    int get_mask(int arg) const {
        const auto it = entries.find(arg);
        return it == entries.end() ? undefined_mask : it->second;
    }

    dnnl_memory_desc_t get_md(int arg) const {
        const auto it = mds.find(arg);
        return it == mds.end() ? nullptr : (dnnl_memory_desc_t)it->second;
    }

    dnnl_data_type_t get_dw_arg(int arg) const {
        if (arg == DNNL_ARG_WEIGHTS)
            return dw_entry.wei_dt;
        else if (arg == DNNL_ARG_BIAS)
            return dw_entry.bia_dt;
        else {
            assert(!"unsupported_argument");
            return dnnl_data_type_undef;
        }
    }

    static constexpr int undefined_mask = -1;

private:
    std::map<int, int /* mask*/> entries;
    std::map<int, benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t>> mds;
    dw_t dw_entry; // only single dw fusion is supported
};

std::ostream &dump_global_params(std::ostream &s);

// Validates a tag/meta-tag.
int check_tag(const std::string &tag_, bool check_enum_tags_only = false);

// Validates a tag in abc notation.
int check_abc_tag(const std::string &tag, bool check_enum_tags_only = false);

// Removes extra dimensions from a tag according to ndims.
std::string trim_tag(const std::string &tag, int ndims);
// Removes extra dimensions from a tag according to mask. `ndims` version is a
// custom case of `mask` version, assuming that first `ndims` dimensions of mask
// are non-zero.
std::string trim_tag_by_mask(const std::string &tag, int mask);

// Converts a tag/meta-tag to abc notation.
std::string normalize_tag(const std::string &tag, int ndims = -1);

dnnl_primitive_attr_t create_dnnl_attr(
        const attr_t &attr, const attr_args_t &attr_args);

dnnl_engine_kind_t str2engine_kind(const char *str);
dnnl_scratchpad_mode_t str2scratchpad_mode(const char *str);
dnnl_fpmath_mode_t str2fpmath_mode(const char *str);
dnnl_accumulation_mode_t str2accumulation_mode(const char *str);
dnnl_rounding_mode_t str2rounding_mode(const std::string &str);

struct dnn_mem_t;

void maybe_scale(const attr_t &attr, float &d, const float *scales, int64_t c,
        int arg, bool opposite_scale = false);
void maybe_zero_point(const attr_t &attr, float &d, const int32_t *zero_points,
        int64_t c, int arg, bool opposite_zero_point = false);
float compute_eltwise_fwd(
        attr_t::post_ops_t::kind_t kind, float src, float alpha, float beta);
float compute_eltwise_bwd(attr_t::post_ops_t::kind_t kind, float d_dst,
        float src, float alpha, float beta);
float compute_binary(
        attr_t::post_ops_t::kind_t kind, float src0, float src1, bool src2);
void maybe_dropout(const attr_t &attr, float &val, int64_t offset,
        const dnn_mem_t &dropout);
void maybe_round(const attr_t &attr, int arg, float &val, int64_t offset,
        dnnl_data_type_t dst_dt);
void maybe_post_ops(const attr_t &attr, float &val, float sum_val,
        const std::vector<float> &v_po_vals);
inline void maybe_post_ops(
        const attr_t &attr, float &val, float sum_val = 0.f) {
    maybe_post_ops(attr, val, sum_val, std::vector<float>());
}

// When using fast-ref option, reference expects everything to be in f32
// data type and also no additional memories coming from runtime attributes.
// That's why we update all data types to f32 and remove all runtime arguments
// to makes them constant when possible.
void update_cpu_ref_attrs(attr_t &attr, dnnl_data_type_t new_dt = dnnl_f32);
#endif
