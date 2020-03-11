/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include <stdlib.h>
#ifndef _WIN32
#include <sys/time.h>
#endif

#include "dnnl.h"
#include "dnnl_debug.h"
#include "dnnl_version.h"

#include "c_types_map.hpp"
#include "cpu/cpu_isa_traits.hpp"
#include "verbose.hpp"

#include "batch_normalization_pd.hpp"
#include "binary_pd.hpp"
#include "concat_pd.hpp"
#include "convolution_pd.hpp"
#include "deconvolution_pd.hpp"
#include "eltwise_pd.hpp"
#include "gemm_pd.hpp"
#include "inner_product_pd.hpp"
#include "layer_normalization_pd.hpp"
#include "lrn_pd.hpp"
#include "matmul_pd.hpp"
#include "pooling_pd.hpp"
#include "reorder_pd.hpp"
#include "resampling_pd.hpp"
#include "rnn_pd.hpp"
#include "shuffle_pd.hpp"
#include "softmax_pd.hpp"
#include "sum_pd.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/ocl/verbose.hpp"
#endif

/* oneDNN CPU ISA info */
#define ISA_ANY "Intel 64"
#define SSE41 "Intel SSE4.1"
#define AVX "Intel AVX"
#define AVX2 "Intel AVX2"
#define AVX512_COMMON "Intel AVX-512"
#define AVX512_CORE \
    "Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions"
#define AVX512_CORE_VNNI "Intel AVX-512 with Intel DL Boost"
#define AVX512_MIC \
    "Intel AVX-512 with AVX512CD, AVX512ER, and AVX512PF extensions"
#define AVX512_MIC_4OPS \
    "Intel AVX-512 with AVX512_4FMAPS and AVX512_4VNNIW extensions"
#define AVX512_CORE_BF16 \
    "Intel AVX-512 with Intel DL Boost and bfloat16 support"

namespace dnnl {
namespace impl {

static setting_t<int> verbose {0};
int get_verbose() {
#if !defined(DISABLE_VERBOSE)
    if (!verbose.initialized()) {
        // Assumes that all threads see the same environment
        const int len = 2;
        char val[len] = {0};
        if (getenv("MKLDNN_VERBOSE", val, len) == 1) verbose.set(atoi(val));
        if (getenv("DNNL_VERBOSE", val, len) == 1) verbose.set(atoi(val));
        if (!verbose.initialized()) verbose.set(0);
    }
    static bool version_printed = false;
    if (!version_printed && verbose.get() > 0) {
        printf("dnnl_verbose,info,oneDNN v%d.%d.%d (commit %s)\n",
                dnnl_version()->major, dnnl_version()->minor,
                dnnl_version()->patch, dnnl_version()->hash);
        printf("dnnl_verbose,info,cpu,runtime:%s\n",
                dnnl_runtime2str(dnnl_version()->cpu_runtime));
        printf("dnnl_verbose,info,cpu,isa:%s\n", get_isa_info());
        printf("dnnl_verbose,info,gpu,runtime:%s\n",
                dnnl_runtime2str(dnnl_version()->gpu_runtime));
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        gpu::ocl::print_verbose_header();
#endif
        version_printed = true;
    }
#endif
    return verbose.get();
}

double get_msec() {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return 1e+3 * now.QuadPart / frequency.QuadPart;
#else
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
#endif
}

const char *get_isa_info() {
    using namespace dnnl::impl::cpu;
    if (mayiuse(avx512_core_bf16)) return AVX512_CORE_BF16;
    if (mayiuse(avx512_mic_4ops)) return AVX512_MIC_4OPS;
    if (mayiuse(avx512_mic)) return AVX512_MIC;
    if (mayiuse(avx512_core_vnni)) return AVX512_CORE_VNNI;
    if (mayiuse(avx512_core)) return AVX512_CORE;
    if (mayiuse(avx512_common)) return AVX512_COMMON;
    if (mayiuse(avx2)) return AVX2;
    if (mayiuse(avx)) return AVX;
    if (mayiuse(sse41)) return SSE41;
    return ISA_ANY;
}

#if defined(DISABLE_VERBOSE)
void pd_info_t::init(const primitive_desc_t *) {}

#else

/* init_info section */
namespace {
#define DNNL_VERBOSE_DAT_LEN 256
#define DNNL_VERBOSE_ATTR_LEN 128
#define DNNL_VERBOSE_AUX_LEN 384
#define DNNL_VERBOSE_PRB_LEN 384

#define DECL_DAT_AUX_PRB_STRS() \
    int dat_written = 0, aux_written = 0, prb_written = 0, attr_written = 0; \
    MAYBE_UNUSED((dat_written * aux_written * prb_written * attr_written)); \
    char dat_str[DNNL_VERBOSE_DAT_LEN] = {'\0'}; \
    MAYBE_UNUSED(dat_str); \
    char attr_str[DNNL_VERBOSE_ATTR_LEN] = {'\0'}; \
    MAYBE_UNUSED(attr_str); \
    char aux_str[DNNL_VERBOSE_AUX_LEN] = {'\0'}; \
    MAYBE_UNUSED(aux_str); \
    char prb_str[DNNL_VERBOSE_PRB_LEN] = {'\0'}; \
    MAYBE_UNUSED(prb_str)

#define DFMT "%" PRId64

void clear_buf(char *buf, int &written) {
    /* TODO: do it better */
    buf[0] = '#';
    buf[1] = '\0';
    written = 1;
}

#define CHECK_WRITTEN(buf, buf_len, written_now, written_total) \
    do { \
        if (written_now < 0 || written_total + written_now > buf_len) { \
            clear_buf(buf, written_total); \
        } else { \
            written_total += written_now; \
        } \
    } while (0)

#define DPRINT(buf, buf_len, written, ...) \
    do { \
        int l = snprintf(buf + written, buf_len - written, __VA_ARGS__); \
        CHECK_WRITTEN(buf, buf_len, l, written); \
    } while (0)

#define MD2STR(buf, buf_len, written, md) \
    do { \
        int l = dnnl_md2fmt_str(buf + written, buf_len - written, md); \
        CHECK_WRITTEN(buf, buf_len, l, written); \
    } while (0)

#define DIM2STR(buf, buf_len, written, md) \
    do { \
        int l = dnnl_md2dim_str(buf + written, buf_len - written, md); \
        CHECK_WRITTEN(buf, buf_len, l, written); \
    } while (0)

// XXX: Outputs strings corresponding to memory formats used for data tensors.
void format_prb_desc_str(
        char *str, int len, int &written, const memory_desc_t *md) {
    const auto dims = md->dims;
    if (md->ndims == 1)
        DPRINT(str, len, written, "x" DFMT, dims[0]);
    else if (md->ndims == 2)
        DPRINT(str, len, written, "mb" DFMT "ic" DFMT, dims[0], dims[1]);
    else if (md->ndims == 3)
        DPRINT(str, len, written, "mb" DFMT "ic" DFMT "iw" DFMT, dims[0],
                dims[1], dims[2]);
    else if (md->ndims == 4)
        DPRINT(str, len, written, "mb" DFMT "ic" DFMT "ih" DFMT "iw" DFMT,
                dims[0], dims[1], dims[2], dims[3]);
    else if (md->ndims == 5)
        DPRINT(str, len, written,
                "mb" DFMT "ic" DFMT "id" DFMT "ih" DFMT "iw" DFMT, dims[0],
                dims[1], dims[2], dims[3], dims[4]);
    else
        DIM2STR(str, len, written, md);
}

void attr2str(char *str, int len, int written, const primitive_attr_t *attr) {
    // scratchpad mode is not a part of has_default_values(). Check it first.
    const scratchpad_mode_t &spm = attr->scratchpad_mode_;
    if (spm != scratchpad_mode_t::dnnl_scratchpad_mode_library) {
        DPRINT(str, len, written, "scratchpad_mode:%s;",
                dnnl_scratchpad_mode2str(spm));
    }

    if (attr->has_default_values()) return;

    const scales_t &os = attr->output_scales_;
    if (!os.has_default_values()) {
        DPRINT(str, len, written, "oscale:%d", os.mask_);
        if (os.mask_ == 0) DPRINT(str, len, written, ":%g", os.scales_[0]);
        DPRINT(str, len, written, ";");
    }

    const arg_scales_t &as = attr->scales_;
    if (!as.has_default_values()) {
        const char *delim = "";
        DPRINT(str, len, written, "scales:'");
        for (const auto &map_entry : as.scales_) {
            const auto &val = map_entry.second;
            if (val.has_default_values()) continue;

            DPRINT(str, len, written, "%ssrc:%d", delim, val.mask_);
            if (val.mask_ == 0)
                DPRINT(str, len, written, ":%g", val.scales_[0]);
            delim = "_";
        }
        DPRINT(str, len, written, "';");
    }

    const zero_points_t &zp = attr->zero_points_;
    if (!zp.has_default_values()) {
        const char *delim = "";
        DPRINT(str, len, written, "zero_points:'");
        for (const auto &arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (zp.has_default_values(arg)) continue;

            int mask = 0;
            const int *zpp = nullptr;
            zp.get(arg, nullptr, &mask, &zpp);
            const char *arg_name = arg == DNNL_ARG_SRC
                    ? "src"
                    : arg == DNNL_ARG_WEIGHTS ? "wei" : "dst";
            DPRINT(str, len, written, "%s%s:%d", delim, arg_name, mask);
            if (mask == 0) {
                if (is_runtime_value(*zpp))
                    DPRINT(str, len, written, ":*");
                else
                    DPRINT(str, len, written, ":%d", *zpp);
            }
            delim = "_";
        }
        DPRINT(str, len, written, "';");
    }

    const post_ops_t &po = attr->post_ops_;
    if (!po.has_default_values()) {
        DPRINT(str, len, written, "post_ops:'");
        for (int i = 0; i < po.len_; ++i) {
            const post_ops_t::entry_t &e = po.entry_[i];
            if (e.is_sum()) {
                DPRINT(str, len, written, "sum;");
            } else if (e.is_sum(false)) {
                DPRINT(str, len, written, "sum:%g;", e.sum.scale);
            } else if (e.is_eltwise(true)) {
                const post_ops_t::entry_t::eltwise_t &ew = e.eltwise;
                if (ew.beta == 0) {
                    if (ew.alpha == 0) {
                        DPRINT(str, len, written, "%s;",
                                dnnl_alg_kind2str(ew.alg));
                    } else {
                        DPRINT(str, len, written, "%s:%g;",
                                dnnl_alg_kind2str(ew.alg), ew.alpha);
                    }
                } else {
                    DPRINT(str, len, written, "%s:%g:%g;",
                            dnnl_alg_kind2str(ew.alg), ew.alpha, ew.beta);
                }
            } else if (e.is_eltwise(false)) {
                const post_ops_t::entry_t::eltwise_t &ew = e.eltwise;
                DPRINT(str, len, written, "%s:%g:%g:%g;",
                        dnnl_alg_kind2str(ew.alg), ew.alpha, ew.beta, ew.scale);
            }
        }
        DPRINT(str, len, written, "';");
    }

    const rnn_data_qparams_t &rnn_qp = attr->rnn_data_qparams_;
    if (!rnn_qp.has_default_values()) {
        DPRINT(str, len, written, "rnn_data_qparams:%g:%g;", rnn_qp.scale_,
                rnn_qp.shift_);
    }
}

void flags2str(char *str, int len, int written, unsigned flags) {
    std::string s;
    if (flags & dnnl_use_global_stats) s += "G";
    if (flags & dnnl_use_scaleshift) s += "S";
    if (flags & dnnl_fuse_norm_relu) s += "R";
    DPRINT(str, len, written, "flags:%s", s.c_str());
}

void verbose_templ(char *buffer, dnnl_engine_t engine,
        dnnl_primitive_kind_t prim_kind, const char *impl_str,
        dnnl_prop_kind_t prop_kind, const char *data_str, const char *attr_str,
        const char *aux_str, const char *prb_str) {
    MAYBE_UNUSED(verbose_templ);
    int written = 0;
    dnnl_engine_kind_t engine_kind;
    dnnl_engine_get_kind(engine, &engine_kind);
    DPRINT(buffer, DNNL_VERBOSE_BUF_LEN, written, "%s,%s,%s,%s,%s,%s,%s,%s",
            dnnl_engine_kind2str(engine_kind), dnnl_prim_kind2str(prim_kind),
            impl_str, dnnl_prop_kind2str(prop_kind), data_str, attr_str,
            aux_str, prb_str);
}

template <typename pd_t>
static void init_info_batch_normalization(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // data
        auto md = s->src_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "data_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // diff data
        auto md = s->diff_src_md();
        if (md) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " diff_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    flags2str(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, s->desc()->flags);

    format_prb_desc_str(
            prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, s->src_md());

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_concat(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // src
        for (int i = 0; i < s->n_inputs(); ++i) {
            auto md = s->src_md(i);
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "src_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " ");

            DIM2STR(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, md);
            if (i != s->n_inputs() - 1)
                DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, ":");
        }
    }
    { // dst
        auto md = s->dst_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "dst_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);

        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, " ");
        DIM2STR(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, md);
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, "axis:" DFMT,
            s->desc()->concat_dimension);

    verbose_templ(buffer, s->engine(), s->kind(), s->name(), prop_kind::undef,
            dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_convolution(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // src
        auto md = s->desc()->prop_kind == prop_kind::backward_data
                ? s->diff_src_md()
                : s->src_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "src_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // wei
        auto md = s->desc()->prop_kind == prop_kind::backward_weights
                ? s->diff_weights_md()
                : s->weights_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " wei_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // bia
        auto md = s->desc()->prop_kind == prop_kind::backward_weights
                ? s->diff_weights_md(1)
                : s->weights_md(1);
        if (md) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " bia_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }
    { // dst
        auto md = !s->is_fwd() ? s->diff_dst_md() : s->dst_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " dst_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, "alg:%s",
            dnnl_alg_kind2str(s->desc()->alg_kind));

    if (s->ndims() == 5) {
        if (s->with_groups())
            DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                    "mb" DFMT "_g" DFMT "ic" DFMT "oc" DFMT "_id" DFMT "od" DFMT
                    "kd" DFMT "sd" DFMT "dd" DFMT "pd" DFMT "_ih" DFMT "oh" DFMT
                    "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT "_iw" DFMT "ow" DFMT
                    "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                    s->MB(), s->G(), s->IC(), s->OC(), s->ID(), s->OD(),
                    s->KD(), s->KSD(), s->KDD(), s->padFront(), s->IH(),
                    s->OH(), s->KH(), s->KSH(), s->KDH(), s->padT(), s->IW(),
                    s->OW(), s->KW(), s->KSW(), s->KDW(), s->padL());
        else
            DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                    "mb" DFMT "_ic" DFMT "oc" DFMT "_id" DFMT "od" DFMT
                    "kd" DFMT "sd" DFMT "dd" DFMT "pd" DFMT "_ih" DFMT "oh" DFMT
                    "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT "_iw" DFMT "ow" DFMT
                    "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                    s->MB(), s->IC(), s->OC(), s->ID(), s->OD(), s->KD(),
                    s->KSD(), s->KDD(), s->padFront(), s->IH(), s->OH(),
                    s->KH(), s->KSH(), s->KDH(), s->padT(), s->IW(), s->OW(),
                    s->KW(), s->KSW(), s->KDW(), s->padL());
    } else {
        if (s->with_groups())
            DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                    "mb" DFMT "_g" DFMT "ic" DFMT "oc" DFMT "_ih" DFMT "oh" DFMT
                    "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT "_iw" DFMT "ow" DFMT
                    "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                    s->MB(), s->G(), s->IC(), s->OC(), s->IH(), s->OH(),
                    s->KH(), s->KSH(), s->KDH(), s->padT(), s->IW(), s->OW(),
                    s->KW(), s->KSW(), s->KDW(), s->padL());
        else
            DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                    "mb" DFMT "_ic" DFMT "oc" DFMT "_ih" DFMT "oh" DFMT
                    "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT "_iw" DFMT "ow" DFMT
                    "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                    s->MB(), s->IC(), s->OC(), s->IH(), s->OH(), s->KH(),
                    s->KSH(), s->KDH(), s->padT(), s->IW(), s->OW(), s->KW(),
                    s->KSW(), s->KDW(), s->padL());
    }

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_deconvolution(pd_t *s, char *buffer) {
    init_info_convolution(s, buffer);
}

template <typename pd_t>
static void init_info_shuffle(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    auto md = s->is_fwd() ? s->src_md() : s->diff_dst_md();

    { // data
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "data_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, "axis:%d group:" DFMT,
            s->axis(), s->group_size());

    dnnl_md2dim_str(prb_str, DNNL_VERBOSE_PRB_LEN, md);

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_eltwise(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // data
        auto md = s->src_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "data_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // diff data
        auto md = s->diff_src_md();
        if (md) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " diff_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written,
            "alg:%s alpha:%g beta:%g", dnnl_alg_kind2str(s->desc()->alg_kind),
            s->desc()->alpha, s->desc()->beta);

    dnnl_md2dim_str(prb_str, DNNL_VERBOSE_PRB_LEN, s->src_md());

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_gemm(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, dat_written,
            "m" DFMT "n" DFMT "k" DFMT "a_dt:%s b_dt:%s c_dt:%s acc_dt:%s",
            s->desc()->m, s->desc()->n, s->desc()->k,
            dnnl_dt2str(s->desc()->a_type), dnnl_dt2str(s->desc()->b_type),
            dnnl_dt2str(s->desc()->c_type), dnnl_dt2str(s->desc()->acc_type));

    verbose_templ(buffer, s->engine(), s->kind(), s->name(), prop_kind::undef,
            dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_inner_product(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // src
        auto md = s->desc()->prop_kind == prop_kind::backward_data
                ? s->diff_src_md()
                : s->src_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "src_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // wei
        auto md = s->desc()->prop_kind == prop_kind::backward_weights
                ? s->diff_weights_md()
                : s->weights_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " wei_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // bia
        auto md = s->desc()->prop_kind == prop_kind::backward_weights
                ? s->diff_weights_md(1)
                : s->weights_md(1);
        if (md) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " bia_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }
    { // dst
        auto md = !s->is_fwd() ? s->diff_dst_md() : s->dst_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " dst_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    if (s->ndims() == 5) {
        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "ic" DFMT "id" DFMT "ih" DFMT "iw" DFMT "oc" DFMT,
                s->MB(), s->IC(), s->ID(), s->IH(), s->IW(), s->OC());
    } else if (s->ndims() == 4) {
        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "ic" DFMT "ih" DFMT "iw" DFMT "oc" DFMT, s->MB(),
                s->IC(), s->IH(), s->IW(), s->OC());
    } else if (s->ndims() == 3) {
        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "ic" DFMT "iw" DFMT "oc" DFMT, s->MB(), s->IC(),
                s->IW(), s->OC());
    } else if (s->ndims() == 2) {
        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "ic" DFMT "oc" DFMT, s->MB(), s->IC(), s->OC());
    }

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_layer_normalization(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // data
        auto md = s->src_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "data_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // stats
        auto md = s->is_fwd() && !s->stats_are_src() ? s->dst_md(1)
                                                     : s->src_md(1);
        if (md) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " stats_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }
    { // diff data
        auto md = s->diff_src_md();
        if (md) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " diff_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    flags2str(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, s->desc()->flags);

    dnnl_md2dim_str(prb_str, DNNL_VERBOSE_PRB_LEN, s->dst_md());

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_lrn(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // data
        auto md = s->src_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "data_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // diff data
        auto md = s->diff_src_md();
        if (md) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " diff_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, "alg:%s",
            dnnl_alg_kind2str(s->desc()->alg_kind));

    format_prb_desc_str(
            prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, s->src_md());
    DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, "ls" DFMT "beta%g",
            s->desc()->local_size, s->desc()->lrn_beta);

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_mem(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // src
        for (int i = 0; i < s->n_inputs(); ++i) {
            auto md = s->src_md(i);
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "src_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " ");
        }
    }
    { // dst
        auto md = s->dst_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "dst_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    dnnl_md2dim_str(prb_str, DNNL_VERBOSE_PRB_LEN, s->dst_md());

    verbose_templ(buffer, s->engine(), s->kind(), s->name(), prop_kind::undef,
            dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_reorder(pd_t *s, char *buffer) {
    init_info_mem(s, buffer);
}

template <typename pd_t>
static void init_info_sum(pd_t *s, char *buffer) {
    init_info_mem(s, buffer);
}

template <typename pd_t>
static void init_info_pooling(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // src
        auto md = s->is_fwd() ? s->src_md() : s->diff_src_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "src_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // dst
        auto md = s->is_fwd() ? s->dst_md() : s->diff_dst_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " dst_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // ws
        auto md = s->workspace_md();
        if (md) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " ws_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, "alg:%s",
            dnnl_alg_kind2str(s->desc()->alg_kind));

    if (s->is_3d()) {
        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "ic" DFMT
                "_"
                "id" DFMT "od" DFMT "kd" DFMT "sd" DFMT "pd" DFMT
                "_"
                "ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "ph" DFMT
                "_"
                "iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "pw" DFMT "",
                s->MB(), s->C(), s->ID(), s->OD(), s->KD(), s->KSD(),
                s->padFront(), s->IH(), s->OH(), s->KH(), s->KSH(), s->padT(),
                s->IW(), s->OW(), s->KW(), s->KSW(), s->padL());
    } else {
        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "ic" DFMT
                "_"
                "ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "ph" DFMT
                "_"
                "iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "pw" DFMT,
                s->MB(), s->C(), s->IH(), s->OH(), s->KH(), s->KSH(), s->padT(),
                s->IW(), s->OW(), s->KW(), s->KSW(), s->padL());
    }

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_softmax(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // data
        auto md = s->dst_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "data_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // diff data
        auto md = s->diff_src_md();
        if (md) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " diff_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, "alg:%s ",
            s->is_softmax() ? "softmax" : "logsoftmax");
    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, "axis:%d", s->axis());

    dnnl_md2dim_str(prb_str, DNNL_VERBOSE_PRB_LEN, s->dst_md());

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_logsoftmax(pd_t *s, char *buffer) {
    init_info_softmax(s, buffer);
}

template <typename pd_t>
static void init_info_rnn(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // src layer
        auto md = s->is_fwd() ? s->src_md(0) : s->diff_src_md(0);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "src_layer_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // src iter
        auto md = s->is_fwd() ? s->src_md(1) : s->diff_src_md(1);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " src_iter_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // wei_layer
        auto md = s->is_fwd() ? s->weights_md(0) : s->diff_weights_md(0);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " wei_layer_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // wei_iter
        auto md = s->is_fwd() ? s->weights_md(1) : s->diff_weights_md(1);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " wei_layer_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    if (s->is_lstm_peephole()) { // wei_peephole
        auto md = s->arg_md(s->is_fwd() ? DNNL_ARG_WEIGHTS_PEEPHOLE
                                        : DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " wei_peephole_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // bias
        auto md = s->arg_md(s->is_fwd() ? DNNL_ARG_BIAS : DNNL_ARG_DIFF_BIAS);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " bias_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // dst layer
        auto md = s->is_fwd() ? s->dst_md(0) : s->diff_dst_md(0);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " dst_layer_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // dst iter
        auto md = s->is_fwd() ? s->dst_md(1) : s->diff_dst_md(1);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " dst_iter_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written,
            "alg:%s direction:%s activation:%s",
            dnnl_alg_kind2str(s->cell_kind()),
            dnnl_rnn_direction2str(s->direction()),
            dnnl_alg_kind2str(s->activation_kind()));

    DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
            "l" DFMT "t" DFMT "mb" DFMT "sic" DFMT "slc" DFMT "dhc" DFMT
            "dlc" DFMT,
            s->L(), s->T(), s->MB(), s->SIC(), s->SLC(), s->DHC(), s->DLC());

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_binary(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // src0
        auto md = s->src_md(0);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "src_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);

        DIM2STR(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, md);
        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, ":");
    }
    { // src1
        auto md = s->src_md(1);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " src_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);

        DIM2STR(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, md);
    }
    { // dst
        auto md = s->dst_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " dst_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);

        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, " ");
        DIM2STR(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, md);
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, "alg:%s",
            dnnl_alg_kind2str(s->desc()->alg_kind));

    verbose_templ(buffer, s->engine(), s->kind(), s->name(), prop_kind::undef,
            dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_matmul(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // src
        auto md = s->src_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "src_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // src1
        auto md = s->weights_md(0);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " wei_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }
    { // bia
        auto md = s->weights_md(1);
        if (md->ndims != 0) {
            DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " bia_");
            MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        }
    }
    { // dst
        auto md = s->dst_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " dst_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    if (s->batched())
        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, "b" DFMT,
                s->batch());
    DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written,
            "m" DFMT "n" DFMT "k" DFMT, s->M(), s->N(), s->K());

    verbose_templ(buffer, s->engine(), s->kind(), s->name(), prop_kind::undef,
            dat_str, attr_str, aux_str, prb_str);
}

template <typename pd_t>
static void init_info_resampling(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    { // src
        auto md = !s->is_fwd() ? s->diff_src_md() : s->src_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, "src_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " ");
        DIM2STR(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, md);
    }
    { // dst
        auto md = !s->is_fwd() ? s->diff_dst_md() : s->dst_md();
        DPRINT(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, " dst_");
        MD2STR(dat_str, DNNL_VERBOSE_DAT_LEN, dat_written, md);
        DPRINT(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, " ");
        DIM2STR(prb_str, DNNL_VERBOSE_PRB_LEN, prb_written, md);
    }

    attr2str(attr_str, DNNL_VERBOSE_ATTR_LEN, attr_written, s->attr());

    DPRINT(aux_str, DNNL_VERBOSE_AUX_LEN, aux_written, "alg:%s",
            dnnl_alg_kind2str(s->desc()->alg_kind));

    verbose_templ(buffer, s->engine(), s->kind(), s->name(),
            s->desc()->prop_kind, dat_str, attr_str, aux_str, prb_str);
}

#undef DPRINT
} // namespace

void pd_info_t::init(const primitive_desc_t *pd) {
    if (is_initialized_) return;

    std::call_once(initialization_flag_, [&] {
        str_.resize(DNNL_VERBOSE_BUF_LEN, '\0');

        using logsoftmax_pd_t = softmax_pd_t;
#define CASE(kind) \
    case primitive_kind::kind: \
        init_info_##kind((const kind##_pd_t *)pd, &str_[0]); \
        break

        switch (pd->kind()) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(concat);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(gemm);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(logsoftmax);
            CASE(matmul);
            CASE(pooling);
            CASE(reorder);
            CASE(resampling);
            CASE(rnn);
            CASE(shuffle);
            CASE(softmax);
            CASE(sum);
            default: assert(!"unknown primitive kind");
        }
#undef CASE

        is_initialized_ = true;
    });
}
#endif

} // namespace impl
} // namespace dnnl

dnnl_status_t dnnl_set_verbose(int level) {
    using namespace dnnl::impl::status;
    if (level < 0 || level > 2) return invalid_arguments;
    dnnl::impl::verbose.set(level);
    return success;
}

const dnnl_version_t *dnnl_version() {
    static const dnnl_version_t ver
            = {DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR, DNNL_VERSION_PATCH,
                    DNNL_VERSION_HASH, DNNL_CPU_RUNTIME, DNNL_GPU_RUNTIME};
    return &ver;
}
