/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_BRGEMM_COMMON_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_BRGEMM_COMMON_HPP

#include <assert.h>
#include <map>
#include <utility>
#include <vector>
#include "brgemm_alg_kind.hpp"
#include <runtime/context.hpp>
#include <runtime/data_type.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace brgemm {

enum attr_key {
    // if unrollaed kernel is used (use_uker == true)
    // then "max_bs" is the the only batch size that can be used on
    // kernel call else "max_bs" is the maximum batch size that can be
    // used
    max_bs = 0, // int
    max_top_vpad, // int
    max_bottom_vpad, // int
    hint_expected_A_size, // int64_t
    hint_expected_B_size, // int64_t
    hint_expected_C_size, // int64_t
    hint_innermost_loop, // bool
    hint_loop_order, // enum, not use for now
    hint_prefetching, // default true, not use for now
    wary_tail_read, // bool
    generate_skip_accumulation, // bool
    // Value of bd_mask_level specifies how bd_mask is used in brgemm kernel
    // 0 - bd_mask is not used
    // 1 - bd_mask is used on storing stage only
    // 2 - bd_mask used both on reading and storing stages
    bd_mask_level, // int
    // use_uker is a boolean value that determines whether to use the unrolled
    // kernel or not
    use_uker, // bool
    // use_interleave_stores is a value that determines whether to use the
    // interleave stores or not
    use_interleave_stores, // bool
    // hint of prefetching distance for A, B and C
    // value should be int
    hint_prfA_dist1,
    hint_prfA_dist2,
    hint_prfB_dist1,
    hint_prfB_dist2,
    hint_prfC_dist1,
    hint_prfC_dist2,
    var_bs, // bool, enable variable batch size for uker
    bs_group, // int, grouping in bs, used by brdgmm
    dispatch_avx, // use avx instead of amx isa
    nkeys, // brgemm internal attribute nkeys
    // extra attribute for range brgemm cache.
    M_range_upper_bound, // generate brg with M from 1 to M_range_upper_bound
    N_range_upper_bound, // generate brg with N from 1 to N_range_upper_bound
    K_range_upper_bound, // generate brg with K from 1 to K_range_upper_bound
    M_range_tail_value, // could be dyn_tail(-1), no_tail(0), static tail(>0)
    N_range_tail_value, // could be dyn_tail(-1), no_tail(0), static tail(>0)
    K_range_tail_value, // could be dyn_tail(-1), no_tail(0), static tail(>0)
};

// enumerate of buffer type in post op calculation
enum postop_data_kind : int {
    bias = 0,
    scales,
    binary_post_ops_rhs,
    oc_logical_off,
    dst_row_logical_off,
    data_C_ptr,
    first_mb_matrix_addr_off,
    a_zp_compensations,
    b_zp_compensations,
    c_zp_values,
    skip_accumulation,
    zp_a_val,
    do_only_comp,
    do_only_zp_a_val,
};

struct attrs_setting_t {
    static const int max_attrs_num = attr_key::nkeys; // without bd_mask
    typedef std::pair<attr_key, int64_t> attrs_map_t;
    int num_ = 0;
    attrs_map_t map_[0];
};

// Todo: currently we don't support sum post op(inplace add)

// elementwise post op define
struct elt_op_t {
    elt_op_t() : elt_op_t(alg_kind_t::alg_kind_undef) {}
    elt_op_t(alg_kind_t alg, float scale = 1.f, float alpha = 1.f,
            float beta = 0.f)
        : alg_(alg), scale_(scale), alpha_(alpha), beta_(beta) {}
    alg_kind_t alg_;
    float scale_;
    float alpha_; // 0.f for general relu.
    float beta_;
};

// binary post op define
struct bin_op_t {
    bin_op_t(alg_kind_t alg, const int *shape, sc_data_etype dtype)
        : alg_(alg) {
        shape_[0] = shape[0];
        shape_[1] = shape[1];
        assert(shape_[0] > 0 && shape_[1] > 0);
        dtype_ = dtype;
    }
    alg_kind_t alg_ = alg_kind_t::alg_kind_undef;
    int shape_[2] = {0};
    sc_data_etype dtype_ = sc_data_etype::F32;
};

// customize bias op, align onednn sematic
// bias add occured before zp/scale calculation in onednn.
struct bias_op_t {
    bias_op_t(sc_data_etype dtype)
        : alg_(alg_kind_t::bias_add), dtype_(dtype) {}
    alg_kind_t alg_ = alg_kind_t::bias_add;
    sc_data_etype dtype_ = sc_data_etype::F32;
};

// Currently we only support single scale, but onednn need a vector of scales,
// even for `per_tensor`.
struct scale_op_t {
    scale_op_t() = default;
    alg_kind_t alg_ = alg_kind_t::out_scales;
    // the scale is fake, only need to tell brgemm creator that scales exist.
    float scale_ = 1.1f;
};

// currently not support zp because of brgemm interface.
// But it is effective.
struct zp_op_t {
    zp_op_t(alg_kind_t alg) : alg_(alg) {}
    alg_kind_t alg_ = alg_kind_t::b_zp;
    // the zp is fake, only need to tell brgemm creator that zp exist.
    int zp_ = 2;
};

struct out_op_t {
    out_op_t(sc_data_etype dtype) : dtype_(dtype) {}
    alg_kind_t alg_ = alg_kind_t::out_dtype;
    sc_data_etype dtype_;
};

struct empty_op_t {
    alg_kind_t alg_ = alg_kind_t::alg_kind_undef;
};

#define DECLARE_POSTOP_CONSTRUCTOR(kind) \
    postop_setting_t(const kind##_op_t &op) { \
        reset(); \
        kind##_op_ = op; \
    }
union postop_setting_t {
    void reset() {
        pack_info_[0] = 0;
        pack_info_[1] = 0;
    }
    postop_setting_t() {
        static_assert(sizeof(postop_setting_t) == sizeof(int64_t) * 2,
                "postop setting size is bigger than 16 bytes.");
        reset();
        empty_op_ = empty_op_t();
    }
    DECLARE_POSTOP_CONSTRUCTOR(elt);
    DECLARE_POSTOP_CONSTRUCTOR(bin);
    DECLARE_POSTOP_CONSTRUCTOR(bias);
    DECLARE_POSTOP_CONSTRUCTOR(scale);
    DECLARE_POSTOP_CONSTRUCTOR(zp);
    DECLARE_POSTOP_CONSTRUCTOR(out);

    bool operator==(const postop_setting_t &other) const {
        return pack_info_[0] == other.pack_info_[0]
                && pack_info_[1] == other.pack_info_[1];
    }

    bool operator!=(const postop_setting_t &other) const {
        return !(*this == other);
    }

    empty_op_t empty_op_;
    elt_op_t elt_op_;
    bin_op_t bin_op_;
    bias_op_t bias_op_;
    scale_op_t scale_op_;
    zp_op_t zp_op_;
    out_op_t out_op_;
    int64_t pack_info_[2];
};

// allow multiple post ops.
struct postops_setting_t {
    // currently we support maximum 9 postops because of alignment of brgemm
    // cache `brg_arg` in runtime.
    static const int max_postops_num = 9;
    static const int op_size = sizeof(postop_setting_t);
    // number of post ops;
    int num_ = 0;
    postop_setting_t ops_[];
};

// nargs inherited from `brgemm_post_ops_data_t` in onednn backend.
static const int postops_data_init_func_nargs = 14;
static const int postops_data_size = postops_data_init_func_nargs * 8; // bytes
} // namespace brgemm

using sc_brgemm_attrs_t = std::map<brgemm::attr_key, int64_t>;
// to use bd_mask, we need to set brgemm kind to list_addr, use amx, max_bs>=1,
// bd_mask_level>=0 and use_uker=true
using sc_brgemm_bd_mask_t = std::vector<char>;
using sc_brgemm_postops_setting_t = std::vector<brgemm::postop_setting_t>;

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

static constexpr int PALETTE_SIZE = 64;

// insert palette ptr to global map, return the inserted dnnl palette ptr
// will return existing one if there's same one
char *insert_global_palette(char *palette);

void *do_get_amx_tile_buf(const char *palette,
        dnnl::impl::graph::gc::runtime::stream_t *stream, bool &amx_exclusive,
        bool &need_config_amx);

#endif
