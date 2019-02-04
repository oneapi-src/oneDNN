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

#define MKLDNN_VERBOSE_DAT_LEN 128
#define MKLDNN_VERBOSE_AUX_LEN 384
#define MKLDNN_VERBOSE_PRB_LEN 384

#define DECL_DAT_AUX_PRB_STRS() \
    char dat_str[MKLDNN_VERBOSE_DAT_LEN] = {'\0'}; MAYBE_UNUSED(dat_str); \
    char aux_str[MKLDNN_VERBOSE_AUX_LEN] = {'\0'}; MAYBE_UNUSED(aux_str); \
    char prb_str[MKLDNN_VERBOSE_PRB_LEN] = {'\0'}; MAYBE_UNUSED(prb_str)

#define DFMT "%" PRId64

inline void verbose_templ(char *buffer, mkldnn_primitive_kind_t prim_kind,
        const char *impl_str, mkldnn_prop_kind_t prop_kind,
        const char *data_str, const char *aux_str, const char *prb_str) {
    MAYBE_UNUSED(verbose_templ);
    snprintf(buffer, MKLDNN_VERBOSE_BUF_LEN, "%s,%s,%s,%s,%s,%s",
            mkldnn_prim_kind2str(prim_kind), impl_str,
            mkldnn_prop_kind2str(prop_kind), data_str, aux_str, prb_str);
}

inline void format_mem_desc_str_generic(char *str, int len,
        const memory_desc_t *md) {
    auto ndims = md->ndims;
    auto dims = md->dims;
    int l = 0;
    for (int d = 0; d < ndims - 1; ++d)
        l += snprintf(str + l, len - l, DFMT "x", dims[d]);
    snprintf(str + l, len - l, DFMT, dims[ndims - 1]);
}

// XXX: Outputs strings corresponding to memory formats used for data tensors.
inline void format_mem_desc_str(char *str, int len, const memory_desc_t *md) {
    auto ndims = md->ndims;
    auto dims = md->dims;
    if (ndims == 1)
        snprintf(str, len, "x" DFMT, dims[0]);
    else if (ndims == 2)
        snprintf(str, len, "mb" DFMT "ic" DFMT, dims[0], dims[1]);
    else if (ndims == 3)
        snprintf(str, len, "mb" DFMT "ic" DFMT "iw" DFMT,
                dims[0], dims[1], dims[2]);
    else if (ndims == 4)
        snprintf(str, len, "mb" DFMT "ic" DFMT "ih" DFMT "iw" DFMT,
                dims[0], dims[1], dims[2], dims[3]);
    else if (ndims == 5)
        snprintf(str, len, "mb" DFMT "ic" DFMT "id" DFMT "ih" DFMT "iw" DFMT,
                dims[0], dims[1], dims[2], dims[3], dims[4]);
    else
        format_mem_desc_str_generic(str, len, md);
}

template <typename pd_t> static void init_info_bnorm(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    auto fmt_data = s->src_md()->format_kind;
    auto fmt_diff = s->is_bwd()
        ? s->diff_src_md()->format_kind : format_kind::undef;
    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN, "fdata:%s fdiff:%s",
            mkldnn_fmt_kind2str(fmt_data), mkldnn_fmt_kind2str(fmt_diff));

    snprintf(aux_str, MKLDNN_VERBOSE_AUX_LEN, "flags:%u", s->desc()->flags);

    format_mem_desc_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, s->src_md());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_conv(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    auto fmt_src = (s->desc()->prop_kind == prop_kind::backward_data
            ? s->diff_src_md() : s->src_md())->format_kind;
    auto fmt_wei = (s->desc()->prop_kind == prop_kind::backward_weights
            ? s->diff_weights_md(0) : s->weights_md(0))->format_kind;
    auto fmt_bia = s->with_bias()
        ? (s->desc()->prop_kind == prop_kind::backward_weights
                ? s->diff_weights_md(1) : s->weights_md(1))->format_kind
        : format_kind::undef;
    auto fmt_dst = (s->desc()->prop_kind == prop_kind::backward_data
            || s->desc()->prop_kind == prop_kind::backward_weights
        ? s->diff_dst_md() : s->dst_md())->format_kind;
    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN,
            "fsrc:%s fwei:%s fbia:%s fdst:%s",
            mkldnn_fmt_kind2str(fmt_src), mkldnn_fmt_kind2str(fmt_wei),
            mkldnn_fmt_kind2str(fmt_bia), mkldnn_fmt_kind2str(fmt_dst));

    snprintf(aux_str, MKLDNN_VERBOSE_AUX_LEN,
            "alg:%s", mkldnn_alg_kind2str(s->desc()->alg_kind));

    if (s->ndims() == 5) {
        if (s->with_groups())
            snprintf(prb_str, MKLDNN_VERBOSE_PRB_LEN,
                "mb" DFMT "_g" DFMT "ic" DFMT "oc" DFMT
                "_id" DFMT "od" DFMT "kd" DFMT "sd" DFMT "dd" DFMT "pd" DFMT
                "_ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT
                "_iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                s->MB(), s->G(), s->IC(), s->OC(),
                s->ID(), s->OD(), s->KD(), s->KSD(), s->KDD(), s->padFront(),
                s->IH(), s->OH(), s->KH(), s->KSH(), s->KDH(), s->padT(),
                s->IW(), s->OW(), s->KW(), s->KSW(), s->KDW(), s->padL());
        else
            snprintf(prb_str, MKLDNN_VERBOSE_PRB_LEN,
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
            snprintf(prb_str, MKLDNN_VERBOSE_PRB_LEN,
                "mb" DFMT "_g" DFMT "ic" DFMT "oc" DFMT
                "_ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "dh" DFMT "ph" DFMT
                "_iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "dw" DFMT "pw" DFMT,
                s->MB(), s->G(), s->IC(), s->OC(),
                s->IH(), s->OH(), s->KH(), s->KSH(), s->KDH(), s->padT(),
                s->IW(), s->OW(), s->KW(), s->KSW(), s->KDW(), s->padL());
        else
            snprintf(prb_str, MKLDNN_VERBOSE_PRB_LEN,
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

    const auto md = (s->desc()->prop_kind == prop_kind::backward_data
            ? s->diff_dst_md() : s->src_md());

    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN, "dt:%s fmt:%s",
            mkldnn_dt2str(md->data_type), mkldnn_fmt_kind2str(md->format_kind));

    snprintf(aux_str, MKLDNN_VERBOSE_AUX_LEN, "axis:%d group_size:" DFMT,
            s->axis(), s->group_size());

    format_mem_desc_str_generic(prb_str, MKLDNN_VERBOSE_PRB_LEN, md);

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_eltwise(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    auto fmt_data = s->src_md()->format_kind;
    auto fmt_diff = s->desc()->prop_kind == prop_kind::backward_data
        ? s->diff_src_md()->format_kind : format_kind::undef;
    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN, "fdata:%s fdiff:%s",
            mkldnn_fmt_kind2str(fmt_data), mkldnn_fmt_kind2str(fmt_diff));

    snprintf(aux_str, MKLDNN_VERBOSE_AUX_LEN,
            "alg:%s", mkldnn_alg_kind2str(s->desc()->alg_kind));

    format_mem_desc_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, s->src_md());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_iprod(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    auto fmt_src = (s->desc()->prop_kind == prop_kind::backward_data
            ? s->diff_src_md() : s->src_md())->format_kind;
    auto fmt_wei = (s->desc()->prop_kind == prop_kind::backward_weights
            ? s->diff_weights_md(0) : s->weights_md(0))->format_kind;
    auto fmt_bia = s->with_bias()
        ? (s->desc()->prop_kind == prop_kind::backward_weights
                ? s->diff_weights_md(1) : s->weights_md(1))->format_kind
        : format_kind::undef;
    auto fmt_dst = (s->desc()->prop_kind == prop_kind::backward_data
            || s->desc()->prop_kind == prop_kind::backward_weights
        ? s->diff_dst_md() : s->dst_md())->format_kind;
    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN,
            "fsrc:%s fwei:%s fbia:%s fdst:%s",
            mkldnn_fmt_kind2str(fmt_src), mkldnn_fmt_kind2str(fmt_wei),
            mkldnn_fmt_kind2str(fmt_bia), mkldnn_fmt_kind2str(fmt_dst));

    snprintf(prb_str, MKLDNN_VERBOSE_PRB_LEN,
            "mb" DFMT "ic" DFMT "oc" DFMT, s->MB(), s->IC_total(), s->OC());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_lrn(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    auto fmt_data = s->src_md()->format_kind;
    auto fmt_diff = s->desc()->prop_kind == prop_kind::backward_data
        ? s->diff_src_md()->format_kind : format_kind::undef;
    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN, "fdata:%s fdiff:%s",
            mkldnn_fmt_kind2str(fmt_data), mkldnn_fmt_kind2str(fmt_diff));

    snprintf(aux_str, MKLDNN_VERBOSE_AUX_LEN,
            "alg:%s", mkldnn_alg_kind2str(s->desc()->alg_kind));

    format_mem_desc_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, s->src_md());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_mem(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    const auto i_md = s->src_md();
    const auto o_md = s->dst_md();
    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN,
            "in:%s_%s out:%s_%s",
            mkldnn_dt2str(i_md->data_type), mkldnn_fmt_kind2str(i_md->format_kind),
            mkldnn_dt2str(o_md->data_type), mkldnn_fmt_kind2str(o_md->format_kind));

    snprintf(aux_str, MKLDNN_VERBOSE_AUX_LEN, "num:%d", s->n_inputs());

    format_mem_desc_str_generic(prb_str, MKLDNN_VERBOSE_PRB_LEN, o_md);

    verbose_templ(buffer, s->kind(), s->name(), prop_kind::undef, dat_str,
            aux_str, prb_str);
}

template <typename pd_t> static void init_info_pool(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    auto fmt_data = (s->desc()->prop_kind == prop_kind::backward_data
            ? s->diff_src_md() : s->src_md())->format_kind;
    auto fmt_ws = s->workspace_md()
        ? s->workspace_md()->format_kind : format_kind::undef;
    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN, "fdata:%s fws:%s",
            mkldnn_fmt_kind2str(fmt_data), mkldnn_fmt_kind2str(fmt_ws));

    snprintf(aux_str, MKLDNN_VERBOSE_AUX_LEN,
            "alg:%s", mkldnn_alg_kind2str(s->desc()->alg_kind));

    if (s->is_3d())
    {
        snprintf(prb_str, MKLDNN_VERBOSE_PRB_LEN,
            "mb" DFMT "ic" DFMT "_"
            "id" DFMT "od" DFMT "kd" DFMT "sd" DFMT "pd" DFMT "_"
            "ih" DFMT "oh" DFMT "kh" DFMT "sh" DFMT "ph" DFMT "_"
            "iw" DFMT "ow" DFMT "kw" DFMT "sw" DFMT "pw" DFMT "",
            s->MB(), s->C(),
            s->ID(), s->OD(), s->KD(), s->KSD(), s->padFront(),
            s->IH(), s->OH(), s->KH(), s->KSH(), s->padT(),
            s->IW(), s->OW(), s->KW(), s->KSW(), s->padL());
    } else {
        snprintf(prb_str, MKLDNN_VERBOSE_PRB_LEN,
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

    auto md = (s->desc()->prop_kind == prop_kind::backward_data
        ? s->diff_src_md() : s->src_md());
    auto fmt_data = md->format_kind;
    auto fmt_diff = s->desc()->prop_kind == prop_kind::backward_data
        ? s->diff_src_md()->format_kind : format_kind::undef;
    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN, "fdata:%s fdiff:%s",
            mkldnn_fmt_kind2str(fmt_data), mkldnn_fmt_kind2str(fmt_diff));

    format_mem_desc_str(prb_str, MKLDNN_VERBOSE_PRB_LEN, md);

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

/// @todo print meaningful data
template <typename pd_t> static void init_info_rnn(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    const mkldnn::impl::memory_desc_t *src_lay_md, *src_iter_md, *wei_lay_md,
            *wei_iter_md, *bias_md, *dst_lay_md, *dst_iter_md;
    if (s->desc()->prop_kind != prop_kind::backward_data) {
        src_lay_md = s->src_md(0);
        src_iter_md = s->src_md(1) ? s->src_md(1) : nullptr;
        wei_lay_md = s->weights_md(0);
        wei_iter_md = s->weights_md(1);
        bias_md = s->weights_md(2);
        dst_lay_md = s->dst_md(0);
        dst_iter_md = s->dst_md(1) ? s->dst_md(1) : nullptr;
    } else {
        src_lay_md = s->diff_src_md(0);
        src_iter_md = s->diff_src_md(1) ? s->diff_src_md(1) : nullptr;
        wei_lay_md = s->diff_weights_md(0);
        wei_iter_md = s->diff_weights_md(1);
        bias_md = s->diff_weights_md(2);
        dst_lay_md = s->diff_dst_md(0);
        dst_iter_md = s->diff_dst_md(1) ? s->diff_dst_md(1) : nullptr;
    }

    alg_kind_t alg_kind = s->cell_kind();
    rnn_direction_t rnn_dir = s->direction();
    snprintf(aux_str, MKLDNN_VERBOSE_AUX_LEN,
            "alg:%s_%s", mkldnn_alg_kind2str(alg_kind), mkldnn_rnn_direction2str(rnn_dir));
    snprintf(dat_str, MKLDNN_VERBOSE_DAT_LEN, "fdata:%s-%s-%s-%s fwei:%s-%s-%s ddata:%s%s-%s%s dwei:%s%s%s",
             mkldnn_fmt_kind2str(src_lay_md->format_kind),
             mkldnn_fmt_kind2str(src_iter_md ? src_iter_md->format_kind : format_kind::undef),
             mkldnn_fmt_kind2str(dst_lay_md->format_kind),
             mkldnn_fmt_kind2str(dst_iter_md ? dst_iter_md->format_kind : format_kind::undef),
             mkldnn_fmt_kind2str(wei_lay_md->format_kind),
             mkldnn_fmt_kind2str(wei_iter_md->format_kind),
             mkldnn_fmt_kind2str(bias_md->format_kind),
             mkldnn_dt2str(src_lay_md->data_type),
             mkldnn_dt2str(src_iter_md ? src_iter_md->data_type : data_type::undef),
             mkldnn_dt2str(dst_lay_md->data_type),
             mkldnn_dt2str(dst_iter_md ? dst_iter_md->data_type : data_type::undef),
             mkldnn_dt2str(wei_lay_md->data_type),
             mkldnn_dt2str(wei_iter_md->data_type),
             mkldnn_dt2str(bias_md->data_type));

    snprintf(prb_str, MKLDNN_VERBOSE_PRB_LEN,
            "l" DFMT "t" DFMT "mb" DFMT
            "sic" DFMT "slc" DFMT "dic" DFMT "dlc" DFMT,
             s->L(), s->T(), s->MB(),
             s->SIC(), s->SLC(), s->DIC(), s->DLC());

    verbose_templ(buffer, s->kind(), s->name(), s->desc()->prop_kind, dat_str,
            aux_str, prb_str);
}

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
