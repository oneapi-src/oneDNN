/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef VERBOSE_HPP
#define VERBOSE_HPP

#include <cinttypes>

#include "mkldnn_debug.h"
#include "c_types_map.hpp"
#include "utils.hpp"
#include "z_magic.hpp"

namespace mkldnn {
namespace impl {

struct verbose_t {
    int level;
};

const verbose_t *mkldnn_verbose();
double get_msec();
const char *get_isa_info();

#if !defined(DISABLE_VERBOSE)
#include <stdio.h>

#define MKLDNN_VERBOSE_BUF_LEN 1024

#define MKLDNN_VERBOSE_DAT_LEN 256
#define MKLDNN_VERBOSE_AUX_LEN 384
#define MKLDNN_VERBOSE_PRB_LEN 384

#define DECL_DAT_AUX_PRB_STRS() \
    int dat_written = 0, aux_written = 0, prb_written = 0; \
    MAYBE_UNUSED((dat_written * aux_written * prb_written)); \
    char dat_str[MKLDNN_VERBOSE_DAT_LEN] = {'\0'}; MAYBE_UNUSED(dat_str); \
    char aux_str[MKLDNN_VERBOSE_AUX_LEN] = {'\0'}; MAYBE_UNUSED(aux_str); \
    char prb_str[MKLDNN_VERBOSE_PRB_LEN] = {'\0'}; MAYBE_UNUSED(prb_str)

#define DFMT "%" PRId64

namespace {
inline void clear_buf(char *buf, int &written) {
    /* TODO: do it better */
    buf[0] = '#';
    buf[1] = '\0';
    written = 1;
}
}

#define DPRINT(buf, buf_len, written, ...) do { \
    int l = snprintf(buf + written, buf_len - written, __VA_ARGS__); \
    if (l < 0 || written + l > buf_len) { \
        clear_buf(buf, written); \
    } else { \
        written += l; \
    } \
} while(0)

// XXX: Outputs strings corresponding to memory formats used for data tensors.
inline void format_prb_desc_str(char *str, int len, const memory_desc_t *md) {
    const auto dims = md->dims;
	int written = 0;
    if (md->ndims == 1)
        DPRINT(str, len, written,
                "x" DFMT, dims[0]);
    else if (md->ndims == 2)
        DPRINT(str, len, written,
                "mb" DFMT "ic" DFMT, dims[0], dims[1]);
    else if (md->ndims == 3)
        DPRINT(str, len, written,
                "mb" DFMT "ic" DFMT "iw" DFMT,
                dims[0], dims[1], dims[2]);
    else if (md->ndims == 4)
        DPRINT(str, len, written,
                "mb" DFMT "ic" DFMT "ih" DFMT "iw" DFMT,
                dims[0], dims[1], dims[2], dims[3]);
    else if (md->ndims == 5)
        DPRINT(str, len, written,
                "mb" DFMT "ic" DFMT "id" DFMT "ih" DFMT "iw" DFMT,
                dims[0], dims[1], dims[2], dims[3], dims[4]);
    else
        mkldnn_md2dim_str(str, len, md);
}

inline void verbose_templ(char *buffer, mkldnn_primitive_kind_t prim_kind,
        const char *impl_str, mkldnn_prop_kind_t prop_kind,
        const char *data_str, const char *aux_str, const char *prb_str) {
    MAYBE_UNUSED(verbose_templ);
    int written = 0;
    DPRINT(buffer, MKLDNN_VERBOSE_BUF_LEN, written, "%s,%s,%s,%s,%s,%s",
            mkldnn_prim_kind2str(prim_kind), impl_str,
            mkldnn_prop_kind2str(prop_kind), data_str, aux_str, prb_str);
}

template <typename pd_t> static void init_info_bnorm(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // data
        auto md = s->src_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "data_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // diff data
        auto md = s->diff_src_md();
        if (md) {
            DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " diff_");
            int l = mkldnn_md2fmt_str(dat_str + dat_written,
                    MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
            if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
        }
    }

    DPRINT(aux_str, MKLDNN_VERBOSE_AUX_LEN, aux_written,
            "flags:%u", s->desc()->flags);

    format_prb_desc_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, s->src_md());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_conv(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // src
        auto md = s->desc()->prop_kind == prop_kind::backward_data
            ? s->diff_src_md() : s->src_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "src_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // wei
        auto md = s->desc()->prop_kind == prop_kind::backward_weights
            ? s->diff_weights_md() : s->weights_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " wei_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // bia
        auto md = s->desc()->prop_kind == prop_kind::backward_weights
            ? s->diff_weights_md(1) : s->weights_md(1);
        if (md) {
            DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " bia_");
            int l = mkldnn_md2fmt_str(dat_str + dat_written,
                    MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
            if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
        }
    }
    if (1) { // dst
        auto md = !s->is_fwd() ? s->diff_dst_md() : s->dst_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " dst_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }

    DPRINT(aux_str, MKLDNN_VERBOSE_AUX_LEN, aux_written,
            "alg:%s", mkldnn_alg_kind2str(s->desc()->alg_kind));

    if (s->ndims() == 5) {
        if (s->with_groups())
            DPRINT(prb_str, MKLDNN_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "_g" DFMT "ic" DFMT "oc" DFMT
                "_id" DFMT "od" DFMT "kd" DFMT "sd" DFMT "dd" DFMT "pd" DFMT
                "_ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT
                "_iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                s->MB(), s->G(), s->IC(), s->OC(),
                s->ID(), s->OD(), s->KD(), s->KSD(), s->KDD(), s->padFront(),
                s->IH(), s->OH(), s->KH(), s->KSH(), s->KDH(), s->padT(),
                s->IW(), s->OW(), s->KW(), s->KSW(), s->KDW(), s->padL());
        else
            DPRINT(prb_str, MKLDNN_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "_ic" DFMT "oc" DFMT
                "_id" DFMT "od" DFMT "kd" DFMT "sd" DFMT "dd" DFMT "pd" DFMT
                "_ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT
                "_iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                s->MB(), s->IC(), s->OC(),
                s->ID(), s->OD(), s->KD(), s->KSD(), s->KDD(), s->padFront(),
                s->IH(), s->OH(), s->KH(), s->KSH(), s->KDH(), s->padT(),
                s->IW(), s->OW(), s->KW(), s->KSW(), s->KDW(), s->padL());
    } else {
        if (s->with_groups())
            DPRINT(prb_str, MKLDNN_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "_g" DFMT "ic" DFMT "oc" DFMT
                "_ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT
                "_iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                s->MB(), s->G(), s->IC(), s->OC(),
                s->IH(), s->OH(), s->KH(), s->KSH(), s->KDH(), s->padT(),
                s->IW(), s->OW(), s->KW(), s->KSW(), s->KDW(), s->padL());
        else
            DPRINT(prb_str, MKLDNN_VERBOSE_PRB_LEN, prb_written,
                "mb" DFMT "_ic" DFMT "oc" DFMT
                "_ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT
                "_iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                s->MB(), s->IC(), s->OC(),
                s->IH(), s->OH(), s->KH(), s->KSH(), s->KDH(), s->padT(),
                s->IW(), s->OW(), s->KW(), s->KSW(), s->KDW(), s->padL());
    }

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_shuffle(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    auto md = s->is_fwd() ? s->src_md() : s->diff_dst_md();

    if (1) { // data
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "data_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }

    DPRINT(aux_str, MKLDNN_VERBOSE_AUX_LEN, aux_written,
            "axis:%d group_size:" DFMT, s->axis(), s->group_size());

    mkldnn_md2dim_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, md);

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_eltwise(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // data
        auto md = s->src_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "data_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // diff data
        auto md = s->diff_src_md();
        if (md) {
            DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " diff_");
            int l = mkldnn_md2fmt_str(dat_str + dat_written,
                    MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
            if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
        }
    }

    DPRINT(aux_str, MKLDNN_VERBOSE_AUX_LEN, aux_written,
            "alg:%s", mkldnn_alg_kind2str(s->desc()->alg_kind));

    mkldnn_md2dim_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, s->src_md());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_iprod(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // src
        auto md = s->desc()->prop_kind == prop_kind::backward_data
            ? s->diff_src_md() : s->src_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "src_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // wei
        auto md = s->desc()->prop_kind == prop_kind::backward_weights
            ? s->diff_weights_md() : s->weights_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " wei_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // bia
        auto md = s->desc()->prop_kind == prop_kind::backward_weights
            ? s->diff_weights_md(1) : s->weights_md(1);
        if (md) {
            DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " bia_");
            int l = mkldnn_md2fmt_str(dat_str + dat_written,
                    MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
            if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
        }
    }
    if (1) { // dst
        auto md = !s->is_fwd() ? s->diff_dst_md() : s->dst_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " dst_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }

    DPRINT(prb_str, MKLDNN_VERBOSE_PRB_LEN, prb_written,
            "mb" DFMT "ic" DFMT "oc" DFMT, s->MB(), s->IC_total(), s->OC());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_lrn(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // data
        auto md = s->src_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "data_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // diff data
        auto md = s->diff_src_md();
        if (md) {
            DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " diff_");
            int l = mkldnn_md2fmt_str(dat_str + dat_written,
                    MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
            if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
        }
    }

    DPRINT(aux_str, MKLDNN_VERBOSE_AUX_LEN, aux_written,
            "alg:%s", mkldnn_alg_kind2str(s->desc()->alg_kind));

    format_prb_desc_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, s->src_md());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_mem(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // src
        auto md = s->src_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "src_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // dst
        auto md = s->dst_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " dst_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }

    DPRINT(aux_str, MKLDNN_VERBOSE_AUX_LEN, aux_written,
            "num:%d", s->n_inputs());

    mkldnn_md2dim_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, s->dst_md());

    verbose_templ(buffer, s->kind(), s->name(), prop_kind::undef, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_pool(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // src
        auto md = s->is_fwd() ? s->src_md() : s->diff_src_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "src_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // dst
        auto md = s->is_fwd() ? s->dst_md() : s->diff_dst_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " dst_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // ws
        auto md = s->workspace_md();
        if (md) {
            DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " ws_");
            int l = mkldnn_md2fmt_str(dat_str + dat_written,
                    MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
            if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
        }
    }

    DPRINT(aux_str, MKLDNN_VERBOSE_AUX_LEN, aux_written,
            "alg:%s", mkldnn_alg_kind2str(s->desc()->alg_kind));

    if (s->is_3d()) {
        DPRINT(prb_str, MKLDNN_VERBOSE_PRB_LEN, prb_written,
            "mb" DFMT "ic" DFMT "_"
            "id" DFMT "od" DFMT "kd" DFMT "sd" DFMT "pd" DFMT "_"
            "ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "ph" DFMT "_"
            "iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "pw" DFMT "",
            s->MB(), s->C(),
            s->ID(), s->OD(), s->KD(), s->KSD(), s->padFront(),
            s->IH(), s->OH(), s->KH(), s->KSH(), s->padT(),
            s->IW(), s->OW(), s->KW(), s->KSW(), s->padL());
    } else {
        DPRINT(prb_str, MKLDNN_VERBOSE_PRB_LEN, prb_written,
            "mb" DFMT "ic" DFMT "_"
            "ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "ph" DFMT "_"
            "iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "pw" DFMT,
            s->MB(), s->C(),
            s->IH(), s->OH(), s->KH(), s->KSH(), s->padT(),
            s->IW(), s->OW(), s->KW(), s->KSW(), s->padL());
    }

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_softmax(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // data
        auto md = s->dst_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "data_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // diff data
        auto md = s->diff_src_md();
        if (md) {
            DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " diff_");
            int l = mkldnn_md2fmt_str(dat_str + dat_written,
                    MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
            if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
        }
    }

    mkldnn_md2dim_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, s->dst_md());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

/// @todo print meaningful data
template <typename pd_t> static void init_info_rnn(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // src layer
        auto md = s->is_fwd() ? s->src_md(0) : s->diff_src_md(0);
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "src_layer_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // src iter
        auto md = s->is_fwd() ? s->src_md(1) : s->diff_src_md(1);
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "src_iter_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // wei_layer
        auto md = s->is_fwd() ? s->weights_md(0) : s->diff_weights_md(0);
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " wei_layer_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // wei_iter
        auto md = s->is_fwd() ? s->weights_md(1) : s->diff_weights_md(1);
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " wei_layer_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // bias
        auto md = s->is_fwd() ? s->weights_md(2) : s->diff_weights_md(2);
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " bias_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // dst layer
        auto md = s->is_fwd() ? s->dst_md(0) : s->diff_dst_md(0);
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "dst_layer_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // dst iter
        auto md = s->is_fwd() ? s->dst_md(1) : s->diff_dst_md(1);
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "dst_iter_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }

    alg_kind_t alg_kind = s->cell_kind();
    rnn_direction_t rnn_dir = s->direction();
    DPRINT(aux_str, MKLDNN_VERBOSE_AUX_LEN, aux_written,
            "alg:%s_%s", mkldnn_alg_kind2str(alg_kind),
            mkldnn_rnn_direction2str(rnn_dir));

    DPRINT(prb_str, MKLDNN_VERBOSE_PRB_LEN, prb_written,
            "l" DFMT "t" DFMT "mb" DFMT
            "sic" DFMT "slc" DFMT "dic" DFMT "dlc" DFMT,
             s->L(), s->T(), s->MB(),
             s->SIC(), s->SLC(), s->DIC(), s->DLC());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

#undef DPRINT

#else /* !defined(DISABLE_VERBOSE) */
#define MKLDNN_VERBOSE_BUF_LEN 1

#define DEFINE_STUB(name) \
    template <typename pd_t> \
    static void CONCAT2(init_info_, name)(pd_t *s, char *buffer) \
    { UNUSED(s); UNUSED(buffer); }

DEFINE_STUB(bnorm);
DEFINE_STUB(conv);
DEFINE_STUB(eltwise);
DEFINE_STUB(iprod);
DEFINE_STUB(lrn);
DEFINE_STUB(mem);
DEFINE_STUB(pool);
DEFINE_STUB(softmax);
DEFINE_STUB(rnn);
DEFINE_STUB(shuffle);
#undef DEFINE_STUB
#endif /* !defined(DISABLE_VERBOSE) */

}
}

#endif
