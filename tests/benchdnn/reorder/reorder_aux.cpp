/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "mkldnn_debug.hpp"
#include "reorder/reorder.hpp"

namespace reorder {

alg_t str2alg(const char *str) {
    if (!strcasecmp("bootstrap", str))
        return ALG_BOOT;
    if (!strcasecmp("reference", str))
        return ALG_REF;
    assert(!"unknown algorithm");
    return ALG_REF;
}

const char *alg2str(alg_t alg) {
    switch (alg) {
    case ALG_REF: return "reference";
    case ALG_BOOT: return "bootstrap";
    default: assert(!"unknown algorithm"); return "unknown algorithm";
    }
}

flag_t str2flag(const char *str) {
    if (!strcasecmp("conv_s8s8", str))
        return FLAG_CONV_S8S8;
    else if (!strcasecmp("gconv_s8s8", str))
        return FLAG_GCONV_S8S8;
    assert(!"unknown flag");
    return FLAG_NONE;
}

const char *flag2str(flag_t flag) {
    switch (flag) {
    case FLAG_NONE: return "";
    case FLAG_CONV_S8S8: return "conv_s8s8";
    case FLAG_GCONV_S8S8: return "gconv_s8s8";
    default: assert(!"Invalid flag"); return "";
    }
}

void prb2str(const prb_t *p, char *buffer) {
    char dt_str[32] = "", tag_str[32] = "", alg_str[32] = "",
         oflag_str[32] = "", attr_str[max_attr_len] = "",
         dims_str[max_desc_len] = "";

    snprintf(dt_str, sizeof(dt_str), "--idt=%s --odt=%s ",
            dt2str(cfg2dt(p->conf_in)), dt2str(cfg2dt(p->conf_out)));
    snprintf(tag_str, sizeof(tag_str), "--itag=%s --otag=%s ",
            tag2str(p->reorder.tag_in), tag2str(p->reorder.tag_out));
    if (p->alg != ALG_REF)
        snprintf(alg_str, sizeof(alg_str), "--alg=%s ", alg2str(p->alg));
    if (p->oflag != FLAG_NONE)
        snprintf(oflag_str, sizeof(oflag_str), "--oflag=%s ",
                flag2str(p->oflag));
    if (!p->attr.is_def()) {
        int len = snprintf(attr_str, max_attr_len, "--attr=\"");
        SAFE_V(len >= 0 ? OK : FAIL);
        attr2str(&p->attr, attr_str + len);
        len = (int)strnlen(attr_str, max_attr_len);
        snprintf(attr_str + len, max_attr_len - len, "\" ");
    }
    dims2str(p->reorder.dims, dims_str);

    snprintf(buffer, max_prb_len, "%s%s%s%s%s%s", dt_str, tag_str, alg_str,
            oflag_str, attr_str, dims_str);
}

}
