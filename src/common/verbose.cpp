/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
* Copyright 2023 Arm Ltd. and affiliates
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

#include <atomic>
#include <regex>
#include <sstream>
#include <type_traits>

#include <stdlib.h>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_version.h"

#include "c_types_map.hpp"
#include "verbose.hpp"

#include "batch_normalization_pd.hpp"
#include "binary_pd.hpp"
#include "concat_pd.hpp"
#include "convolution_pd.hpp"
#include "deconvolution_pd.hpp"
#include "eltwise_pd.hpp"
#include "gemm_pd.hpp"
#include "group_normalization_pd.hpp"
#include "inner_product_pd.hpp"
#include "layer_normalization_pd.hpp"
#include "lrn_pd.hpp"
#include "matmul_pd.hpp"
#include "pooling_pd.hpp"
#include "prelu_pd.hpp"
#include "reduction_pd.hpp"
#include "reorder_pd.hpp"
#include "resampling_pd.hpp"
#include "rnn_pd.hpp"
#include "shuffle_pd.hpp"
#include "softmax_pd.hpp"
#include "sum_pd.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "common/dnnl_thread.hpp"
#include "cpu/platform.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/verbose.hpp"
#endif

#ifdef DNNL_WITH_SYCL
#include "sycl/verbose.hpp"
#endif

#ifdef DNNL_EXPERIMENTAL
#include "common/experimental.hpp"
#endif

namespace dnnl {
namespace impl {

static setting_t<uint32_t> verbose {0};

void print_header(const filter_status_t &filter_status) noexcept {
    static std::atomic_flag version_printed = ATOMIC_FLAG_INIT;
    if (!version_printed.test_and_set()) {
        printf("onednn_verbose,info,oneDNN v%d.%d.%d (commit %s)\n",
                dnnl_version()->major, dnnl_version()->minor,
                dnnl_version()->patch, dnnl_version()->hash);
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
        printf("onednn_verbose,info,cpu,runtime:%s,nthr:%d\n",
                dnnl_runtime2str(dnnl_version()->cpu_runtime),
                dnnl_get_max_threads());
        printf("onednn_verbose,info,cpu,isa:%s\n",
                cpu::platform::get_isa_info());
#endif
        printf("onednn_verbose,info,gpu,runtime:%s\n",
                dnnl_runtime2str(dnnl_version()->gpu_runtime));
        // Printing the header generally requires iterating over devices/backends,
        // which may involve an allocation. Use a try/catch block in case
        // these fail (not printing a header is reasonable in this case)
        try {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            gpu::intel::ocl::print_verbose_header();
#endif
#ifdef DNNL_WITH_SYCL
            sycl::print_verbose_header();
#endif
#ifdef ONEDNN_BUILD_GRAPH
            graph::utils::print_verbose_header();
#endif
        } catch (...) {
            printf("onednn_verbose,info,exception while printing verbose "
                   "header\n");
        }
#ifdef DNNL_EXPERIMENTAL
        printf("onednn_verbose,info,experimental features are enabled\n");
        printf("onednn_verbose,info,use batch_normalization stats one pass is "
               "%s\n",
                experimental::use_bnorm_stats_one_pass() ? "enabled"
                                                         : "disabled");
#endif

#ifdef DNNL_EXPERIMENTAL_SPARSE
        printf("onednn_verbose,info,experimental functionality for sparse "
               "domain is enabled\n");
#endif

        printf("onednn_verbose,primitive,info,template:");
        printf("%soperation,engine,primitive,implementation,prop_"
               "kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_"
               "time\n",
                get_verbose_timestamp() ? "timestamp," : "");

#ifdef ONEDNN_BUILD_GRAPH
        printf("onednn_verbose,graph,info,template:");
        printf("%soperation,engine,partition_id,partition_kind,op_names,"
               "data_formats,logical_tensors,fpmath_mode,backend,exec_"
               "time\n",
                get_verbose_timestamp() ? "timestamp," : "");
#endif
        if (filter_status.status == filter_status_t::flags::valid)
            printf("onednn_verbose,common,info,filter format is enabled, "
                   "hit components: %s\n",
                    filter_status.components.c_str());
        else if (filter_status.status == filter_status_t::flags::invalid)
            printf("onednn_verbose,common,error,filter format is ill-formed "
                   "and is not applied, error: %s\n",
                    filter_status.err_msg.c_str());
    }
}

// hint parameter is the kind of verbose we are querying for
uint32_t get_verbose(verbose_t::flag_kind verbosity_kind,
        component_t::flag_kind filter_kind) noexcept {
#if defined(DISABLE_VERBOSE)
    return verbose_t::none;
#else
    // we print all verbose by default
    static int flags = component_t::all;
    // record filter parsing result to instruct verbose printing
    static filter_status_t filter_status;

    if (!verbose.initialized()) {
        // Assumes that all threads see the same environment
        static std::string user_opt = getenv_string_user("VERBOSE");
        auto update_kind = [&](const std::string &s, int &k) {
            // Legacy: we accept values 0,1,2
            // 0 and none erase previously set flags, including error
            if (s == "0" || s == "none") k = verbose_t::none;
            if (s == "1") k |= verbose_t::exec_profile;
            if (s == "2")
                k |= verbose_t::exec_profile | verbose_t::create_profile;
            if (s == "all" || s == "-1") k |= verbose_t::all;
            if (s == "error") k |= verbose_t::error;
            if (s == "check")
                k |= verbose_t::create_check | verbose_t::exec_check;
            if (s == "dispatch") k |= verbose_t::create_dispatch;
            if (s == "profile")
                k |= verbose_t::create_profile | verbose_t::exec_profile;
            if (s == "profile_create") k |= verbose_t::create_profile;
            if (s == "profile_exec") k |= verbose_t::exec_profile;
            // Enable profiling to external libraries
            if (s == "profile_externals") k |= verbose_t::profile_externals;
            // we extract debug info debuginfo=XX. ignore if debuginfo is invalid.
            if (s.rfind("debuginfo=", 0) == 0)
                k |= verbose_t::make_debuginfo(
                        std::strtol(s.c_str() + 10, nullptr, 10));
        };

        auto update_filter = [&](const std::string &s,
                                     filter_status_t &filter_status) -> int {
            int k = component_t::none;
            try {
                std::regex regexp = std::regex(s);

#define REGEX_SEARCH(k, component, regexp, filter_status) \
    if (std::regex_search("" #component "", regexp)) { \
        (k) |= component_t::component; \
        (filter_status).components += "" #component ","; \
    }
                REGEX_SEARCH(k, primitive, regexp, filter_status);
                REGEX_SEARCH(k, reorder, regexp, filter_status);
                REGEX_SEARCH(k, shuffle, regexp, filter_status);
                REGEX_SEARCH(k, concat, regexp, filter_status);
                REGEX_SEARCH(k, sum, regexp, filter_status);
                REGEX_SEARCH(k, convolution, regexp, filter_status);
                REGEX_SEARCH(k, deconvolution, regexp, filter_status);
                REGEX_SEARCH(k, eltwise, regexp, filter_status);
                REGEX_SEARCH(k, lrn, regexp, filter_status);
                REGEX_SEARCH(k, batch_normalization, regexp, filter_status);
                REGEX_SEARCH(k, inner_product, regexp, filter_status);
                REGEX_SEARCH(k, rnn, regexp, filter_status);
                REGEX_SEARCH(k, binary, regexp, filter_status);
                REGEX_SEARCH(k, matmul, regexp, filter_status);
                REGEX_SEARCH(k, resampling, regexp, filter_status);
                REGEX_SEARCH(k, pooling, regexp, filter_status);
                REGEX_SEARCH(k, reduction, regexp, filter_status);
                REGEX_SEARCH(k, prelu, regexp, filter_status);
                REGEX_SEARCH(k, softmax, regexp, filter_status);
                REGEX_SEARCH(k, layer_normalization, regexp, filter_status);
                REGEX_SEARCH(k, group_normalization, regexp, filter_status);
                REGEX_SEARCH(k, graph, regexp, filter_status);
                REGEX_SEARCH(k, gemm_api, regexp, filter_status);
#undef REGEX_SEARCH
            } catch (const std::exception &e) {
                filter_status.status = filter_status_t::flags::invalid;
                filter_status.err_msg = e.what();
                return component_t::all;
            }

            // filter enabled and at least one component is hit
            if (filter_status.components.length() != 0) {
                // pop out the last comma
                filter_status.components.pop_back();
                filter_status.status = filter_status_t::flags::valid;
            } else {
                filter_status.status = filter_status_t::flags::invalid;
                filter_status.err_msg
                        = "component with name \'" + s + "\' not found";
            }
            return k;
        };

        // we always enable error by default
        int val = verbose_t::error;
        for (size_t pos_st = 0, pos_en = user_opt.find_first_of(',', pos_st);
                true; pos_st = pos_en + 1,
                    pos_en = user_opt.find_first_of(',', pos_st)) {
            std::string tok = user_opt.substr(pos_st, pos_en - pos_st);
            // update verbose flags
            update_kind(tok, val);
            // update filter flags
            if (tok.rfind("filter=", 0) == 0) {
                auto filter_str = tok.substr(7);
                if (!filter_str.empty()) {
                    flags = update_filter(filter_str, filter_status);
                }
            }
            if (pos_en == std::string::npos) break;
        }

        // We parse for explicit flags
        verbose.set(val);
    }

    int result = verbose.get() & verbosity_kind;
    if (verbosity_kind == verbose_t::debuginfo)
        result = verbose_t::get_debuginfo(verbose.get());
    if (result) print_header(filter_status);
    bool filter_result = flags & filter_kind;
    return filter_result ? result : 0;
#endif
}

static setting_t<bool> verbose_timestamp {false};
bool get_verbose_timestamp() {
#if defined(DISABLE_VERBOSE)
    return false;
#else
    if (verbose.get() == 0) return false;

    if (!verbose_timestamp.initialized()) {
        // Assumes that all threads see the same environment
        static bool val
                = getenv_int_user("VERBOSE_TIMESTAMP", verbose_timestamp.get());
        verbose_timestamp.set(val);
    }
    return verbose_timestamp.get();
#endif
}

#if defined(DISABLE_VERBOSE)
void pd_info_t::init(
        dnnl::impl::engine_t *, const dnnl::impl::primitive_desc_t *) {}

std::string rt_mds2str(primitive_kind_t prim_kind, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *bia_md,
        const memory_desc_t *dst_md) {
    return std::string();
}

std::string rt_dims2fmt_str(primitive_kind_t prim_kind,
        const memory_desc_t *src_md, const memory_desc_t *wei_md,
        const memory_desc_t *dst_md) {
    return std::string();
}

#else

std::ostream &operator<<(std::ostream &ss, engine_kind_t eng_kind) {
    ss << dnnl_engine_kind2str(eng_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const engine_t *engine) {
    ss << dnnl_engine_kind2str(engine->kind());
    if (dnnl_engine_get_count(engine->kind()) > 1)
        ss << ":" + std::to_string(engine->index());
    return ss;
}

const char *prim_kind2str(primitive_kind_t prim_kind) {
    switch ((int)prim_kind) {
        case primitive_kind::zero_pad: return "zero_pad";
        default: return dnnl_prim_kind2str(prim_kind);
    }
}

std::ostream &operator<<(std::ostream &ss, primitive_kind_t prim_kind) {
    ss << prim_kind2str(prim_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, prop_kind_t prop_kind) {
    ss << dnnl_prop_kind2str(prop_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, data_type_t data_type) {
    ss << dnnl_dt2str(data_type);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, alg_kind_t alg) {
    ss << dnnl_alg_kind2str(alg);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, format_kind_t format_kind) {
    ss << dnnl_fmt_kind2str(format_kind);
    return ss;
}

#ifdef DNNL_EXPERIMENTAL_SPARSE
std::ostream &operator<<(std::ostream &ss, sparse_encoding_t encoding) {
    ss << dnnl_sparse_encoding2str(encoding);
    return ss;
}
#endif

std::string normalization_flags2str(unsigned flags) {
    std::string s;
    if (flags & normalization_flags::use_global_stats) s += "G";
    if (flags & normalization_flags::use_scale) s += "C";
    if (flags & normalization_flags::use_shift) s += "H";
    if (flags & normalization_flags::fuse_norm_relu) s += "R";
    if (flags & normalization_flags::fuse_norm_add_relu) s += "A";
    return s;
}

std::string rnn_flags2str(unsigned flags) {
    std::string s;
    if (flags & rnn_flags::diff_weights_overwrite) s += "O";
    return s;
}

std::ostream &operator<<(std::ostream &ss, const memory_extra_desc_t &extra) {
    using namespace memory_extra_flags;

    ss << ":f" << extra.flags;
    if (extra.flags & compensation_conv_s8s8)
        ss << ":s8m" << extra.compensation_mask;
    if (extra.flags & compensation_conv_asymmetric_src)
        ss << ":zpm" << extra.asymm_compensation_mask;
    if (extra.flags & scale_adjust && extra.scale_adjust != 1.f)
        ss << ":sa" << extra.scale_adjust;
    return ss;
}

std::string md2fmt_tag_str(const memory_desc_t *md) {
    memory_desc_wrapper mdw(md);

    dims_t blocks = {0};
    mdw.compute_blocks(blocks);

    char dim_chars[DNNL_MAX_NDIMS + 1];
    dims_t ou_blocks = {0};
    utils::array_copy(ou_blocks, mdw.padded_dims(), mdw.ndims());

    for (int d = 0; d < mdw.ndims(); ++d) {
        dim_chars[d] = (blocks[d] == 1 ? 'a' : 'A') + (char)d;
        ou_blocks[d] /= blocks[d];
    }

    // Can't report meaningful tag for runtime dimensions.
    if (mdw.has_runtime_strides()) return "*";

    dims_t strides;
    const auto &blk = mdw.blocking_desc();
    utils::array_copy(strides, blk.strides, mdw.ndims());

    utils::simultaneous_sort(strides, ou_blocks, dim_chars, mdw.ndims(),
            [](dim_t a, dim_t b) { return b - a; });

    dim_chars[mdw.ndims()] = '\0';

    std::string s(dim_chars);

    if (!mdw.is_plain()) {
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            char c = ('a' + (char)blk.inner_idxs[iblk]);
            s += (std::to_string(blk.inner_blks[iblk]) + c);
        }
    }

    return s;
}

std::string md2fmt_strides_str(const memory_desc_t *md) {
    memory_desc_wrapper mdw(md);
    std::string s;

    // Print strides if non-dense descriptor with defined dims/strides was
    // provided.
    if (mdw.has_runtime_dims_or_strides() || mdw.is_dense(true)) return s;

    // Note: there's no API to create a memory desc with strides and blocks
    // together, thus, this non-plain md will dump strides for info purpose.
    s += md2dim_str(md, dims_type_t::strides);
    return s;
}

// Forms a format string for a given memory descriptor.
//
// There are two formats:
// - dense: defined as: 'dt:[a|p|o|0]:fmt_kind:fmt:strides:extra'.
// - sparse: defined as: 'dt:[a|p|o|0]:fmt_kind:encoding:extra'.
// Here:
//  - dt       -- data type
//  - a        -- indicates memory desc was created with fmt_kind `any`.
//  - p        -- indicates there is non-trivial padding
//  - o        -- indicates there is non-trivial padding offset
//  - 0        -- indicates there is non-trivial offset0
//  - fmt_kind -- format kind (blocked, wino, etc...)
//  - encoding -- [sparse_desc only] sparse encoding (csr, etc...)
//  - fmt      -- [blocking_desc only] extended format string
//  - strides  -- [blocking_desc only] non-dense strides string (dims style)
//  - extra    -- shows extra fields (underspecified)
//
// Note: `user_format` is an information that is not available in memory descs
// from `pd` since those are initialized by implementations. The knowledge about
// original user format specified is kept in pd->desc()->xxx_desc and this is
// the info provided to this call.
// On the other hand, just a user memory descriptor can't be passed because it
// is not initialized b the library, and format information will be missed.
std::string md2fmt_str(const memory_desc_t *md, format_kind_t user_format) {
    std::stringstream ss;
    if (!md || types::is_zero_md(md)) {
        ss << data_type::undef << "::" << format_kind::undef << ":::";
        return ss.str();
    }

    memory_desc_wrapper mdw(md);
    ss << mdw.data_type() << ":";

    bool padded_dims = false, padded_offsets = false;
    for (int d = 0; d < mdw.ndims(); ++d) {
        if (mdw.dims()[d] != mdw.padded_dims()[d]) padded_dims = true;
        if (mdw.padded_offsets()[d] != 0) padded_offsets = true;
    }
    bool offset0 = mdw.offset0();
    ss << (user_format == format_kind::any ? "a" : "");
    ss << (padded_dims ? "p" : "");
    ss << (padded_offsets ? "o" : "");
    ss << (offset0 ? "0" : "");
    ss << ":" << mdw.format_kind();

    // Cast is required to pass through compiler error:
    // error: case value ‘256’ not in enumerated type
    // ‘dnnl::impl::format_kind_t’ {aka ‘dnnl_format_kind_t’}
    switch (static_cast<int>(mdw.format_kind())) {
        case format_kind::blocked:
            ss << ":" << md2fmt_tag_str(md) << ":" << md2fmt_strides_str(md);
            break;
        case format_kind::wino:
        case format_kind::rnn_packed:
        case format_kind::opaque: ss << "::"; break;
        case format_kind::sparse: ss << ":" << mdw.encoding() << ":"; break;
        case format_kind::any: ss << ":any:"; break;
        default:
            assert(!"unsupported format_kind");
            ss << "::";
            break;
    }

    ss << mdw.extra();

    return ss.str();
}

// Puts memory_desc information into stream without dimensions
std::ostream &operator<<(std::ostream &ss, const memory_desc_t *md) {
    ss << md2fmt_str(md, format_kind::undef);
    return ss;
}

template <typename T>
static std::string get_val_str(T val) {
    static_assert(
            std::is_arithmetic<T>::value, "T must be an arithmetic type.");
    if (is_runtime_value(val)) return std::string("*");
    return std::to_string(val);
}

// Returns string with dimensions from a given memory descriptor.
// The format is defined as: dim0xdim1x...xdimN, with RT values signed as `*`.
std::string md2dim_str(const memory_desc_t *md, dims_type_t dims_type) {
    if (md == nullptr || md->ndims == 0) return "";

    memory_desc_wrapper mdw(md);
    std::string s;

    assert(dims_type == dims_type_t::dims || dims_type == dims_type_t::strides);
    const auto &dims_obj
            = dims_type == dims_type_t::dims ? mdw.dims() : mdw.strides();

    s += get_val_str(dims_obj[0]);
    for (int d = 1; d < mdw.ndims(); ++d)
        s += ("x" + get_val_str(dims_obj[d]));

    return s;
}

// Returns string with descriptor style from memory_desc since there's an
// operator<< for memory_desc.
std::string md2desc_str(const memory_desc_t *md) {
    const auto dims = md->dims;
    std::string s;
    if (md->ndims >= 6) return md2dim_str(md);

    if (md->ndims == 1) {
        s += "x" + std::to_string(dims[0]);
        return s;
    }

    s += "mb" + std::to_string(dims[0]) + "ic" + std::to_string(dims[1]);
    if (md->ndims >= 5) s += "id" + std::to_string(dims[md->ndims - 3]);
    if (md->ndims >= 4) s += "ih" + std::to_string(dims[md->ndims - 2]);
    if (md->ndims >= 3) s += "iw" + std::to_string(dims[md->ndims - 1]);
    return s;
}

std::ostream &operator<<(std::ostream &ss, const runtime_scales_t &scale) {
    ss << scale.mask_;
    ss << ":" << scale.data_type_;
    if (scale.ndims_) {
        ss << ":";
        for (int i = 0; i < scale.ndims_ - 1; ++i)
            ss << scale.group_dims_[i] << 'x';
        ss << scale.group_dims_[scale.ndims_ - 1];
    }
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const scales_t &oscale) {
    ss << oscale.mask_;
    const float val = oscale.scales_[0];
    // Can't use scientific flags since it breaks parsing on converter and
    // benchdnn side.
    if (oscale.mask_ == 0 || is_runtime_value(val))
        ss << ":" << get_val_str(val);
    return ss;
}

namespace {
int get_runtime_mask(const memory_desc_t *md) {
    int mask = 0;
    for (int d = md->ndims - 1; d >= 0; --d) {
        mask += md->dims[d] == DNNL_RUNTIME_DIM_VAL ? 1 << d : 0;
    }
    return mask;
}

int get_arg_index(int arg) {
    if (arg & DNNL_ARG_MULTIPLE_SRC) return arg - DNNL_ARG_MULTIPLE_SRC;
    switch (arg) {
        case DNNL_ARG_SRC_0: return 0;
        case DNNL_ARG_SRC_1: return 1;
        default: return -1;
    }
    return -1;
}

std::string get_arg(int arg) {
    if (arg & DNNL_ARG_MULTIPLE_SRC) return "msrc";

    std::string s;
    switch (arg) {
        case DNNL_ARG_SRC: // DNNL_ARG_SRC_0
        case DNNL_ARG_SRC_1: s = "src"; break;
        case DNNL_ARG_DST: s = "dst"; break;
        case DNNL_ARG_WEIGHTS: s = "wei"; break;
        case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DST:
            s = "attr_post_op_dw_dst";
            break;
        case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS:
            s = "attr_post_op_dw_wei";
            break;
        default: assert(!"unsupported arg"); s = "unsupported arg";
    }
    return s;
}
} // namespace

std::string arg2str(int arg) {
    std::string s = get_arg(arg);
    const int idx = get_arg_index(arg);
    if (idx != -1) s += std::to_string(idx);
    return s;
}

std::ostream &operator<<(std::ostream &ss, const primitive_attr_t *attr) {
    struct {
        const char *operator()() {
            current[0] = next;
            next = ' ';
            return current;
        }

    private:
        char current[2] = {};
        char next = 0;
    } field_delim;

    // scratchpad and fpmath mode are not a part of
    // has_default_values(). Check them first.
    const scratchpad_mode_t &spm = attr->scratchpad_mode_;
    if (spm != scratchpad_mode_t::dnnl_scratchpad_mode_library) {
        ss << field_delim()
           << "attr-scratchpad:" << dnnl_scratchpad_mode2str(spm);
    }
    const fpmath_t &fpm = attr->fpmath_;
    if (fpm.mode_ != fpmath_mode_t::dnnl_fpmath_mode_strict
            || fpm.apply_to_int_) {
        ss << field_delim()
           << "attr-fpmath:" << dnnl_fpmath_mode2str(fpm.mode_);
        if (fpm.apply_to_int_) ss << ":true";
    }

    const accumulation_mode_t &am = attr->acc_mode_;
    if (am != accumulation_mode::strict) {
        ss << field_delim() << "attr-acc:" << dnnl_accumulation_mode2str(am);
    }

    const bool deterministic = attr->deterministic_;
    if (deterministic) {
        ss << field_delim() << "attr-deterministic:" << deterministic;
    }
    if (attr->has_default_values()) return ss;

    const runtime_scales_t &os = attr->output_scales_;
    if (!os.has_default_values()) {
        ss << field_delim() << "attr-oscale:" << os;
    }

    std::string empty_delim, attr_delim = "+";

    const arg_scales_t &as = attr->scales_;
    if (!as.has_default_values()) {
        std::string delim = empty_delim;
        ss << field_delim() << "attr-scales:";
        for (const auto &map_entry : as.scales_) {
            const auto &val = map_entry.second;
            if (val.has_default_values()) continue;

            int arg = map_entry.first;
            ss << delim << arg2str(arg) << ":" << val;
            delim = attr_delim;
        }
    }

    const zero_points_t &zp = attr->zero_points_;
    if (!zp.has_default_values()) {
        std::string delim = empty_delim;
        ss << field_delim() << "attr-zero-points:";
        for (const auto &arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (zp.has_default_values(arg)) continue;

            int mask = 0;
            zp.get(arg, &mask);
            const auto dt = zp.get_data_type(arg);

            ss << delim << arg2str(arg) << ":" << mask << ":" << dt;

            const auto &g_ndim = zp.get_groups_ndims(arg);
            if (g_ndim) {
                const auto &g_dims = zp.get_groups(arg);
                ss << ":";
                for (int i = 0; i < g_ndim - 1; ++i)
                    ss << g_dims[i] << 'x';
                ss << g_dims[g_ndim - 1];
            }

            delim = attr_delim;
        }
    }

    const post_ops_t &po = attr->post_ops_;
    if (!po.has_default_values()) {
        std::string delim = empty_delim;
        ss << field_delim() << "attr-post-ops:";
        for (int i = 0; i < po.len(); ++i) {
            const post_ops_t::entry_t &e = po.entry_[i];
            switch (e.kind) {
                case primitive_kind::sum: {
                    const auto &s = e.sum;
                    ss << delim << "sum";
                    if (s.scale != 1.f || s.zero_point != 0
                            || s.dt != data_type::undef)
                        ss << ":" << s.scale;
                    if (s.zero_point != 0 || s.dt != data_type::undef)
                        ss << ":" << s.zero_point;
                    if (s.dt != data_type::undef) ss << ":" << s.dt;
                } break;
                case primitive_kind::convolution: {
                    using namespace data_type;
                    const auto &c = e.depthwise_conv;
                    ss << delim << "dw:k" << c.kernel << "s" << c.stride << "p"
                       << c.padding;
                    if (c.wei_dt == s8 || c.dst_dt != f32)
                        ss << ":" << c.dst_dt;
                } break;
                case primitive_kind::eltwise: {
                    const post_ops_t::entry_t::eltwise_t &ew = e.eltwise;
                    ss << delim << ew.alg;
                    if (ew.alpha != 0.f || ew.beta != 0.f || ew.scale != 1.f)
                        ss << ":" << ew.alpha;
                    if (ew.beta != 0.f || ew.scale != 1.f) ss << ":" << ew.beta;
                    if (ew.scale != 1.f) ss << ":" << ew.scale;
                } break;
                case primitive_kind::binary: {
                    const post_ops_t::entry_t::binary_t &eb = e.binary;
                    const auto &md = eb.src1_desc;
                    int mask = 0;
                    for (int d = 0; d < md.ndims; ++d)
                        mask += md.dims[d] != 1 ? (1 << d) : 0;
                    ss << delim << eb.alg << ":" << md.data_type << ":" << mask;
                    const memory_desc_wrapper mdw(md);
                    switch (mdw.format_kind()) {
                        case format_kind::blocked:
                            if (!mdw.count_non_unit_dims(1))
                                ss << ":" << md2fmt_tag_str(&md);
                            break;
                        case format_kind::any: ss << ":any"; break;
                        default: assert(!"unsupported format_kind");
                    }
                } break;
                case primitive_kind::prelu: {
                    const auto &ep = e.prelu;
                    ss << delim << "prelu"
                       << ":" << ep.mask;
                } break;
                default: assert(!"unsupported post op primitive kind!"); break;
            }
            delim = attr_delim;
        }
    }

    const rnn_data_qparams_t &rnn_qp = attr->rnn_data_qparams_;
    if (!rnn_qp.has_default_values()) {
        ss << field_delim() << "rnn_data_qparams:" << rnn_qp.scale_ << ":"
           << rnn_qp.shift_ << ";";
    }

    return ss;
}

/* init_info section */
namespace {

template <typename pd_t>
std::string init_info_batch_normalization(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();
    auto diff_src_md = pd->diff_src_md();

    ss << "data_" << src_md;
    if (diff_src_md) ss << " diff_" << diff_src_md;

    ss << "," << pd->attr() << ",";
    ss << "flags:" << normalization_flags2str(pd->desc()->flags) << ",";
    ss << md2desc_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_binary(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src0_md = pd->invariant_src_md(0);
    auto src1_md = pd->invariant_src_md(1);
    auto dst_md = pd->invariant_dst_md();

    ss << "src_" << md2fmt_str(src0_md, pd->invariant_src_user_format_kind(0));
    ss << " src_" << md2fmt_str(src1_md, pd->invariant_src_user_format_kind(1));
    ss << " dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";
    ss << md2dim_str(src0_md) << ":" << md2dim_str(src1_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_concat(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->invariant_src_md(i);
        ss << "src_"
           << md2fmt_str(src_i_md, pd->invariant_src_user_format_kind(i))
           << " ";
    }
    auto dst_md = pd->invariant_dst_md();
    ss << "dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "axis:" << pd->desc()->concat_dimension << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->src_md(i);
        ss << md2dim_str(src_i_md);
        if (i < pd->n_inputs() - 1) ss << ":";
    }

    return ss.str();
}

template <typename pd_t>
std::string init_info_convolution(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto wei_md = pd->invariant_wei_md();
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    ss << "src_" << md2fmt_str(src_md, pd->invariant_src_user_format_kind());
    ss << " wei_" << md2fmt_str(wei_md, pd->invariant_wei_user_format_kind());
    if (bia_md)
        ss << " bia_"
           << md2fmt_str(bia_md, pd->invariant_bia_user_format_kind());

    // `has_fused_dw` modifies the convolution output in the following way:
    // * It provides additional src, wei and bia md to show their presence and
    //   wei format (for convenience, it can't be used directly).
    // * It makes output spatial dimensions same as input since first conv is
    //   always 1x1 for such fusion. This is to make a problem benchdnn
    //   compatible.
    // * Note: Queried `dst_md` with final dimensions after fusion will reside
    //   in fused conv pd. The op_desc for it is created on the library side and
    //   filled with already blocked formats compatible with precedeing 1x1
    //   convolution. It means that it can't identify if original dst_md was
    //   created with `format_kind::any` or not. For purposes of re-construction
    //   of benchdnn line, intermediate "src_fused" is required - it gives an
    //   info about data type to pass into benchdnn and also whether 1x1 conv
    //   dst was created with `format_kind::any`.
    // * Note: DW-post op is the only reason why `arg_md` got `user_input`
    //   argument and also takes argument. This is due to dw post-op mds can
    //   be queried only through `arg_md` interface.
    const bool has_fused_dw
            = pd->attr()->post_ops_.find(primitive_kind::convolution) >= 0;
    if (has_fused_dw) {
        auto src_fused_md = pd->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_SRC);
        auto wei_fused_md
                = pd->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
        auto bia_fused_md
                = pd->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
        // User-provided dst memory descriptor.
        ss << " src_fused_"
           << md2fmt_str(src_fused_md,
                      pd->invariant_dst_user_format_kind(
                              DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_SRC));
        // Not user-provided memory descriptors.
        ss << " wei_fused_" << wei_fused_md;
        ss << " bia_fused_" << bia_fused_md;
        ss << " dst_" << dst_md;
    } else {
        ss << " dst_"
           << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    if (pd->with_groups()) ss << "g" << pd->G();
    ss << "mb" << pd->MB() << "_"
       << "ic" << pd->IC() << "oc" << pd->OC() << "_";
    if (pd->ndims() >= 5)
        ss << "id" << pd->ID() << "od" << (has_fused_dw ? pd->ID() : pd->OD())
           << "kd" << pd->KD() << "sd" << pd->KSD() << "dd" << pd->KDD() << "pd"
           << pd->padFront() << "_";
    if (pd->ndims() >= 4)
        ss << "ih" << pd->IH() << "oh" << (has_fused_dw ? pd->IH() : pd->OH())
           << "kh" << pd->KH() << "sh" << pd->KSH() << "dh" << pd->KDH() << "ph"
           << pd->padT() << "_";
    ss << "iw" << pd->IW() << "ow" << (has_fused_dw ? pd->IW() : pd->OW())
       << "kw" << pd->KW() << "sw" << pd->KSW() << "dw" << pd->KDW() << "pw"
       << pd->padL();

    return ss.str();
}

template <typename pd_t>
std::string init_info_deconvolution(const engine_t *e, const pd_t *pd) {
    return init_info_convolution(e, pd);
}

template <typename pd_t>
std::string init_info_eltwise(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->use_dst() ? pd->dst_md() : pd->src_md();
    auto diff_src_md = pd->diff_src_md();

    ss << "data_" << data_md;
    if (diff_src_md) ss << " diff_" << diff_src_md;

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << " alpha:" << pd->desc()->alpha
       << " beta:" << pd->desc()->beta << ",";
    ss << md2dim_str(data_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_gemm(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_a_md = pd->invariant_src_md(0);
    auto src_b_md = pd->invariant_src_md(1);
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    auto get_bia_mask = [&bia_md]() {
        auto bia_ndims = bia_md->ndims;
        auto bia_dims = bia_md->dims;
        int mask = 0;
        for (int d = bia_ndims - 1; d >= 0; --d) {
            mask += bia_dims[d] != 1 ? 1 << d : 0;
        }
        return mask;
    };

    ss << "src_a_"
       << md2fmt_str(src_a_md, pd->invariant_src_user_format_kind(0));
    ss << " src_b_"
       << md2fmt_str(src_b_md, pd->invariant_src_user_format_kind(1));
    if (pd->with_bias()) {
        ss << " bia_"
           << md2fmt_str(bia_md, pd->invariant_bia_user_format_kind());
        ss << "_mask" << get_bia_mask();
    }
    ss << " dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",,";

    ss << md2dim_str(src_a_md) << ":" << md2dim_str(src_b_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_group_normalization(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();
    auto dst_md = pd->invariant_dst_md();
    ss << "src_" << src_md;
    ss << " dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());
    if (!pd->is_fwd()) ss << " diff_src_" << pd->diff_src_md();
    ss << ",";

    ss << pd->attr() << ",";
    ss << "flags:" << normalization_flags2str(pd->desc()->flags) << ",";
    ss << "g" << pd->desc()->groups << md2desc_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_inner_product(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto wei_md = pd->invariant_wei_md();
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    ss << "src_" << md2fmt_str(src_md, pd->invariant_src_user_format_kind());
    ss << " wei_" << md2fmt_str(wei_md, pd->invariant_wei_user_format_kind());
    if (bia_md)
        ss << " bia_"
           << md2fmt_str(bia_md, pd->invariant_bia_user_format_kind());
    ss << " dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",,";

    ss << md2desc_str(src_md);
    ss << "oc" << pd->OC();

    return ss.str();
}

template <typename pd_t>
std::string init_info_layer_normalization(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();
    auto dst_md = pd->invariant_dst_md();
    auto stats_md = pd->is_fwd() && !pd->stats_are_src() ? pd->dst_md(1)
                                                         : pd->src_md(1);
    auto scaleshift_md = pd->weights_md(0);
    auto diff_scaleshift_md = pd->diff_weights_md(0);

    ss << "src_" << src_md;
    ss << " dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());
    if (stats_md) ss << " stats_" << stats_md;
    if (pd->use_scale()) ss << " scale_" << scaleshift_md;
    if (pd->use_shift()) ss << " shift_" << scaleshift_md;

    if (!pd->is_fwd()) ss << " diff_src_" << pd->diff_src_md();

    if (!pd->is_fwd() && pd->use_scale())
        ss << " diff_scale_" << diff_scaleshift_md;
    if (!pd->is_fwd() && pd->use_shift())
        ss << " diff_shift_" << diff_scaleshift_md;

    ss << "," << pd->attr() << ",";
    ss << "flags:" << normalization_flags2str(pd->desc()->flags) << ",";
    ss << md2dim_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_lrn(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->src_md();
    auto diff_src_md = pd->diff_src_md();

    ss << "data_" << data_md;
    if (diff_src_md) ss << " diff_" << diff_src_md;

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";
    ss << md2desc_str(data_md);
    ss << "ls" << pd->desc()->local_size << "beta" << pd->desc()->lrn_beta;

    return ss.str();
}

std::string mds2str_matmul(const memory_desc_t *src_md,
        format_kind_t src_user_format_kind, const memory_desc_t *wei_md,
        format_kind_t wei_user_format_kind, const memory_desc_t *bia_md,
        format_kind_t bia_user_format_kind, const memory_desc_t *dst_md,
        format_kind_t dst_user_format_kind) {
    std::string s;
    auto get_bia_mask = [&bia_md]() {
        auto bia_ndims = bia_md->ndims;
        auto bia_dims = bia_md->dims;
        int mask = 0;
        for (int d = bia_ndims - 1; d >= 0; --d) {
            mask += bia_dims[d] != 1 ? 1 << d : 0;
        }
        return mask;
    };

    std::stringstream ss;
    ss << "src_" << md2fmt_str(src_md, src_user_format_kind);
    ss << " wei_" << md2fmt_str(wei_md, wei_user_format_kind);
    if (!memory_desc_wrapper(bia_md).is_zero()) {
        ss << " bia_" << md2fmt_str(bia_md, bia_user_format_kind);
        ss << "_mask" << get_bia_mask();
    }
    ss << " dst_" << md2fmt_str(dst_md, dst_user_format_kind);
    s = ss.str();
    return s;
}

std::string dims2fmt_str_matmul(
        const memory_desc_t *src_md, const memory_desc_t *wei_md) {
    return md2dim_str(src_md) + ":" + md2dim_str(wei_md);
}

template <typename pd_t>
std::string init_info_matmul(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->invariant_src_md();
    auto wei_md = pd->invariant_wei_md();
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    ss << mds2str_matmul(src_md, pd->invariant_src_user_format_kind(), wei_md,
            pd->invariant_wei_user_format_kind(), bia_md,
            pd->invariant_bia_user_format_kind(), dst_md,
            pd->invariant_dst_user_format_kind());
    ss << "," << pd->attr() << ",";

    if (pd->has_runtime_dims_or_strides()) {
        ss << "runtime_dims_masks:" << get_runtime_mask(src_md) << ":"
           << get_runtime_mask(wei_md);
    }
    ss << "," << dims2fmt_str_matmul(src_md, wei_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_pooling(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();
    auto ws_md = pd->workspace_md();

    ss << "src_" << md2fmt_str(src_md, pd->invariant_src_user_format_kind());
    ss << " dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());
    if (ws_md) ss << " ws_" << ws_md;

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    ss << "mb" << pd->MB() << "ic" << pd->IC() << "_";
    if (pd->ndims() >= 5)
        ss << "id" << pd->ID() << "od" << pd->OD() << "kd" << pd->KD() << "sd"
           << pd->KSD() << "dd" << pd->KDD() << "pd" << pd->padFront() << "_";
    if (pd->ndims() >= 4)
        ss << "ih" << pd->IH() << "oh" << pd->OH() << "kh" << pd->KH() << "sh"
           << pd->KSH() << "dh" << pd->KDH() << "ph" << pd->padT() << "_";
    ss << "iw" << pd->IW() << "ow" << pd->OW() << "kw" << pd->KW() << "sw"
       << pd->KSW() << "dw" << pd->KDW() << "pw" << pd->padL();

    return ss.str();
}

template <typename pd_t>
std::string init_info_prelu(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->src_md(0);
    auto wei_md = pd->weights_md(0);
    auto diff_data_md = pd->diff_src_md(0);
    auto diff_wei_md = pd->diff_weights_md(0);

    ss << "data_" << data_md;
    ss << " wei_" << wei_md;
    if (diff_data_md) ss << " diff_" << diff_data_md;
    if (diff_wei_md) ss << " diff_wei_" << diff_wei_md;

    ss << "," << pd->attr() << ",,";
    ss << md2dim_str(data_md) << ":" << md2dim_str(wei_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_reduction(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();

    ss << "src_" << md2fmt_str(src_md, pd->invariant_src_user_format_kind());
    ss << " dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << " p:" << pd->desc()->p
       << " eps:" << pd->desc()->eps << ",";
    ss << md2dim_str(src_md) << ":" << md2dim_str(dst_md);

    return ss.str();
}

std::string mds2str_reorder(const memory_desc_t *src_md,
        format_kind_t src_user_format_kind, const memory_desc_t *dst_md,
        format_kind_t dst_user_format_kind) {
    std::string s;
    s += "src_" + md2fmt_str(src_md, src_user_format_kind);
    s += " dst_" + md2fmt_str(dst_md, dst_user_format_kind);
    return s;
}

std::string dims2fmt_str_reorder(const memory_desc_t *src_md) {
    return md2dim_str(src_md);
}

template <typename pd_t>
std::string init_info_reorder(const engine_t *e, pd_t *pd) {
    std::stringstream ss;

    const auto src_ek = pd->desc()->src_engine_kind;
    const auto dst_ek = pd->desc()->dst_engine_kind;

    if (src_ek != dst_ek)
        ss << src_ek << "2" << dst_ek;
    else
        ss << e;

    ss << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();

    ss << mds2str_reorder(src_md, pd->invariant_src_user_format_kind(), dst_md,
            pd->invariant_dst_user_format_kind());
    ss << "," << pd->attr() << ",";

    if (pd->has_runtime_dims_or_strides()) {
        ss << "runtime-dim-mask:" << get_runtime_mask(src_md);
    }
    ss << "," << dims2fmt_str_reorder(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_resampling(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();

    ss << "src_" << md2fmt_str(src_md, pd->invariant_src_user_format_kind());
    ss << " dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    ss << "mb" << pd->MB() << "ic" << pd->C() << "_";
    if (pd->ndims() >= 5) ss << "id" << pd->ID() << "od" << pd->OD() << "_";
    if (pd->ndims() >= 4) ss << "ih" << pd->IH() << "oh" << pd->OH() << "_";
    ss << "iw" << pd->IW() << "ow" << pd->OW();

    return ss.str();
}

template <typename pd_t>
std::string init_info_rnn(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    // TODO: shorten the names to consume fewer characters on verbose output.
    ss << "src_layer_"
       << md2fmt_str(pd->src_md(0), pd->src_md(0, true)->format_kind);
    if (pd->with_src_iter())
        ss << " src_iter_"
           << md2fmt_str(pd->src_md(1), pd->src_md(1, true)->format_kind);
    ss << " wei_layer_"
       << md2fmt_str(pd->weights_md(0), pd->weights_md(0, true)->format_kind);
    ss << " wei_iter_"
       << md2fmt_str(pd->weights_md(1), pd->weights_md(1, true)->format_kind);
    if (pd->is_lstm_peephole())
        ss << " wei_peephole_"
           << md2fmt_str(
                      pd->weights_md(2), pd->weights_md(2, true)->format_kind);
    // TODO: separate methods for aux weights?
    if (pd->is_lstm_projection()) {
        auto proj_idx = 2 + pd->is_lstm_peephole();
        ss << " wei_proj_"
           << md2fmt_str(pd->weights_md(proj_idx),
                      pd->weights_md(proj_idx, true)->format_kind);
    }
    if (pd->with_bias()) {
        auto bias_idx = 2 + pd->is_lstm_peephole() + pd->is_lstm_projection();
        ss << " bias_"
           << md2fmt_str(pd->weights_md(bias_idx),
                      pd->weights_md(bias_idx, true)->format_kind);
    }
    ss << " dst_layer_"
       << md2fmt_str(pd->dst_md(0), pd->dst_md(0, true)->format_kind);
    if (pd->with_dst_iter())
        ss << " dst_iter_"
           << md2fmt_str(pd->dst_md(1), pd->dst_md(1, true)->format_kind);

    if (!pd->is_fwd()) {
        ss << " diff_src_layer_"
           << md2fmt_str(pd->diff_src_md(0),
                      pd->invariant_src_user_format_kind(0));
        if (pd->with_src_iter())
            ss << " diff_src_iter_"
               << md2fmt_str(pd->diff_src_md(1),
                          pd->invariant_src_user_format_kind(1));
        ss << " diff_wei_layer_"
           << md2fmt_str(pd->diff_weights_md(0),
                      pd->invariant_wei_user_format_kind(0));
        ss << " diff_wei_iter_"
           << md2fmt_str(pd->diff_weights_md(1),
                      pd->invariant_wei_user_format_kind(1));
        if (pd->is_lstm_peephole())
            ss << " diff_wei_peephole_"
               << md2fmt_str(pd->diff_weights_md(2),
                          pd->invariant_wei_user_format_kind(2));
        if (pd->is_lstm_projection()) {
            auto proj_idx = 2 + pd->is_lstm_peephole();
            ss << " diff_wei_proj_"
               << md2fmt_str(pd->weights_md(proj_idx),
                          pd->invariant_wei_user_format_kind(proj_idx));
        }
        if (pd->with_bias()) {
            auto bias_idx
                    = 2 + pd->is_lstm_peephole() + pd->is_lstm_projection();
            ss << " diff_bias_"
               << md2fmt_str(pd->weights_md(bias_idx),
                          pd->invariant_wei_user_format_kind(bias_idx));
        }
        ss << " diff_dst_layer_"
           << md2fmt_str(pd->diff_dst_md(0),
                      pd->invariant_dst_user_format_kind(0));
        if (pd->with_dst_iter())
            ss << " diff_dst_iter_"
               << md2fmt_str(pd->diff_dst_md(1),
                          pd->invariant_dst_user_format_kind(1));
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->cell_kind()
       << " direction:" << dnnl_rnn_direction2str(pd->direction())
       << " activation:" << pd->activation_kind()
       << " flags:" << rnn_flags2str(pd->desc()->flags) << ",";

    ss << "l" << pd->L() << "t" << pd->T() << "mb" << pd->MB() << "sic"
       << pd->SIC() << "slc" << pd->SLC() << "dhc" << pd->DHC() << "dic"
       << pd->DIC();

    return ss.str();
}

template <typename pd_t>
std::string init_info_shuffle(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->invariant_src_md();

    ss << "data_" << data_md;

    ss << "," << pd->attr() << ",";
    ss << "axis:" << pd->axis() << " group:" << pd->group_size() << ",";
    ss << md2dim_str(data_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_softmax(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->dst_md();
    auto diff_dst_md = pd->diff_dst_md();

    ss << "src_" << md2fmt_str(src_md, pd->invariant_src_user_format_kind());
    ss << " dst_" << dst_md;
    if (!types::is_zero_md(diff_dst_md)) ss << " diff_dst_" << diff_dst_md;

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->alg_kind() << " axis:" << pd->axis() << ",";
    ss << md2dim_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_sum(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->invariant_src_md(i);
        ss << "src_"
           << md2fmt_str(src_i_md, pd->invariant_src_user_format_kind(i))
           << " ";
    }
    auto dst_md = pd->invariant_dst_md();
    ss << "dst_" << md2fmt_str(dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",,";
    ss << md2dim_str(dst_md);

    return ss.str();
}

} // namespace

std::string rt_mds2str(primitive_kind_t prim_kind, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *bia_md,
        const memory_desc_t *dst_md) {
    // Note: pass format_kind::undef since runtime dims-ed mds can't have
    // format_kind::any at any stage.
    std::string s;
    switch ((int)prim_kind) {
        case primitive_kind::matmul:
            s = mds2str_matmul(src_md, format_kind::undef, wei_md,
                    format_kind::undef, bia_md, format_kind::undef, dst_md,
                    format_kind::undef);
            break;
        case primitive_kind::reorder:
            s = mds2str_reorder(
                    src_md, format_kind::undef, dst_md, format_kind::undef);
            break;

        case primitive_kind::batch_normalization:
        case primitive_kind::binary:
        case primitive_kind::concat:
        case primitive_kind::convolution:
        case primitive_kind::deconvolution:
        case primitive_kind::eltwise:
        case primitive_kind::inner_product:
        case primitive_kind::layer_normalization:
        case primitive_kind::lrn:
        case primitive_kind::pooling:
        case primitive_kind::prelu:
        case primitive_kind::reduction:
        case primitive_kind::resampling:
        case primitive_kind::rnn:
        case primitive_kind::shuffle:
        case primitive_kind::softmax:
        case primitive_kind::sum: assert(!"unsupported primitive kind"); break;
        default: assert(!"unknown primitive kind");
    }
    return s;
}

std::string rt_dims2fmt_str(primitive_kind_t prim_kind,
        const memory_desc_t *src_md, const memory_desc_t *wei_md,
        const memory_desc_t *dst_md) {
    std::string s;
    switch ((int)prim_kind) {
        case primitive_kind::matmul:
            s = dims2fmt_str_matmul(src_md, wei_md);
            break;
        case primitive_kind::reorder: s = dims2fmt_str_reorder(src_md); break;

        case primitive_kind::batch_normalization:
        case primitive_kind::binary:
        case primitive_kind::concat:
        case primitive_kind::convolution:
        case primitive_kind::deconvolution:
        case primitive_kind::eltwise:
        case primitive_kind::inner_product:
        case primitive_kind::layer_normalization:
        case primitive_kind::lrn:
        case primitive_kind::pooling:
        case primitive_kind::prelu:
        case primitive_kind::reduction:
        case primitive_kind::resampling:
        case primitive_kind::rnn:
        case primitive_kind::shuffle:
        case primitive_kind::softmax:
        case primitive_kind::sum: assert(!"unsupported primitive kind"); break;
        default: assert(!"unknown primitive kind");
    }
    return s;
}

void pd_info_t::init(engine_t *engine, const primitive_desc_t *pd) {
    if (is_initialized_) return;

    std::call_once(initialization_flag_, [&] {
    // clang-format off
#define CASE(kind) \
    case primitive_kind::kind: \
        str_ = init_info_##kind(engine, (const kind##_pd_t *)pd); \
        break

        switch ((int)pd->kind()) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(concat);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(gemm);
            CASE(group_normalization);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(matmul);
            CASE(pooling);
            CASE(prelu);
            CASE(reduction);
            CASE(reorder);
            CASE(resampling);
            CASE(rnn);
            CASE(shuffle);
            CASE(softmax);
            CASE(sum);
            case primitive_kind::sdpa:
              str_ = "sdpa, unknown info";
              break;
            case primitive_kind::zero_pad:
              str_ = "zero_pad, unknown info";
              break;
            default:
              str_ = "unknown primitive info";
              assert(!"unknown primitive kind");
        }
#undef CASE
        // clang-format on

        is_initialized_ = true;
    });
}
#endif

} // namespace impl
} // namespace dnnl

dnnl_status_t dnnl_set_verbose(int level) {
    using namespace dnnl::impl::status;
    using namespace dnnl::impl;
    if (level < 0 || level > 2) return invalid_arguments;

    uint32_t verbose_level = verbose_t::none;
    if (level == 1) verbose_level = verbose_t::error | verbose_t::exec_profile;
    if (level == 2)
        verbose_level = verbose_t::error | verbose_t::exec_profile
                | verbose_t::create_profile;
    // we put the lower byte of level as devinfo to preserve backward
    // compatibility with historical VERBOSE={1,2}
    if (level == 1 || level == 2) verbose_level |= (level << 24);
    verbose.set(verbose_level);
    return success;
}

const dnnl_version_t *dnnl_version(void) {
    static const dnnl_version_t ver
            = {DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR, DNNL_VERSION_PATCH,
                    DNNL_VERSION_HASH, DNNL_CPU_RUNTIME, DNNL_GPU_RUNTIME};
    return &ver;
}
