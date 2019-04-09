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

#define DPRINT(...) do { \
    int l = snprintf(buffer, rem_len, __VA_ARGS__); \
    buffer += l; rem_len -= l; \
} while(0)

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

dims_t str2dims(const char *str) {
    dims_t dims;
    do {
        int len;
        int64_t dim;
        int scan = sscanf(str, IFMT "%n", &dim, &len);
        SAFE_V(scan == 1 ? OK : FAIL);
        dims.push_back(dim);
        str += len;
        SAFE_V(*str == 'x' || *str == '\0' ? OK : FAIL);
    } while (*str++ != '\0');
    return dims;
}

void dims2str(const dims_t &dims, char *buffer) {
    int rem_len = max_dims_len;
    for (size_t d = 0; d < dims.size() - 1; ++d)
        DPRINT(IFMT "x", dims[d]);
    DPRINT(IFMT, dims[dims.size() - 1]);
}

void prb2str(const prb_t *p, const res_t *res, char *buffer) {
    char dims_buf[max_dims_len] = {0};
    dims2str(p->reorder.dims, dims_buf);

    char alg_str[32] = "";
    if (p->alg != ALG_REF)
        snprintf(alg_str, sizeof(alg_str), "--alg=%s ", alg2str(p->alg));

    char oflag_str[32] = "";
    if (p->oflag != FLAG_NONE)
        snprintf(oflag_str, sizeof(oflag_str), "--oflag=%s ",
                flag2str(p->oflag));

    char attr_buf[max_attr_len] = {0};
    bool is_attr_def = p->attr.is_def();
    if (!is_attr_def) {
        int len = snprintf(attr_buf, max_attr_len, "--attr=\"");
        SAFE_V(len >= 0 ? OK : FAIL);
        attr2str(&p->attr, attr_buf + len);
        len = (int)strnlen(attr_buf, max_attr_len);
        snprintf(attr_buf + len, max_attr_len - len, "\" ");
    }

    int rem_len = max_prb_len;
    DPRINT("--idt=%s --odt=%s --itag=%s --otag=%s %s%s%s%s",
            dt2str(cfg2dt(p->conf_in)), dt2str(cfg2dt(p->conf_out)),
            tag2str(p->reorder.tag_in), tag2str(p->reorder.tag_out),
            alg_str, oflag_str, attr_buf, dims_buf);
}

}
