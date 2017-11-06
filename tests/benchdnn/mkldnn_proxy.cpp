/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include <stdint.h>
#include <limits.h>
#include <assert.h>

#include "mkldnn.h"

#include "common.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_proxy.hpp"

attr_t::scale_t::policy_t attr_t::scale_t::str2policy(const char *str) {
#define CASE(_plc) if (!strcasecmp(STRINGIFY(_plc), str)) return _plc
    CASE(NONE);
    CASE(COMMON);
    CASE(PER_OC);
#undef CASE
    assert(!"unknown attr::scale::policy");
    return NONE;
}

const char *attr_t::scale_t::policy2str(attr_t::scale_t::policy_t policy) {
    if (policy == NONE) return "none";
    if (policy == COMMON) return "common";
    if (policy == PER_OC) return "per_oc";
    assert(!"unknown attr::scale::policy");
    return "unknown attr::scale::policy";
}

int attr_t::scale_t::str2scale(const char *str, const char **end_s) {
    *this = attr_t::scale_t();

    if (str == NULL) return FAIL;

    const char *s_;
    const char * &s = end_s ? *end_s : s_;
    s = str;

    for (policy_t p = NONE; true; p = (policy_t)((int)p + 1)) {
        if (p == POLICY_TOTAL) return FAIL;

        const char *ps = policy2str(p);
        if (!strncasecmp(ps, s, strlen(ps))) {
            this->policy = p;
            s += strlen(ps);
            break;
        }
    }

    if (*s != ':') return OK;
    s++;

    char *end;
    this->scale = strtof(s, &end);
    if (this->scale <= 0 || end == s) return FAIL;

    s = end;
    assert(*s == '\0' || *s == ';');

    return OK;
}

void attr_t::scale_t::scale2str(char *buffer, char **end_b) const {
    assert(buffer);
    buffer += sprintf(buffer, "%s:%g", policy2str(this->policy), this->scale);
    if (end_b) *end_b = buffer;
}

bool attr_t::is_def() const {
    return true
        && irmode == round_mode_t::NEAREST
        && oscale.is_def();
}

int attr_t::mkldnn_attr_recreate() {
    if (mkldnn_attr) mkldnn_primitive_attr_destroy(mkldnn_attr);
    DNN_SAFE(mkldnn_primitive_attr_create(&mkldnn_attr), CRIT);

    if (irmode != round_mode_t::NEAREST)
        DNN_SAFE(mkldnn_primitive_attr_set_int_output_round_mode(mkldnn_attr,
                    (mkldnn_round_mode_t)irmode), CRIT);

    if (!oscale.is_def()) {
        int count = oscale.policy == scale_t::policy_t::COMMON ? 1 : 4096;
        int mask = oscale.policy == scale_t::policy_t::PER_OC ? 1 << 1 : 0;
        float *scales = (float *)zmalloc(count * sizeof(float), 64);
        SAFE(scales != NULL ? OK : FAIL, CRIT);
        for (int i = 0; i < count; ++i)
            scales[i] = oscale.scale; /* TODO: extend for many cases */
        DNN_SAFE(mkldnn_primitive_attr_set_output_scales(mkldnn_attr, count,
                    mask, scales), CRIT);
        zfree(scales);
    }

    return OK;
}

int str2attr(attr_t *attr, const char *str) {
    if (attr == NULL || str == NULL) return FAIL;
    *attr = attr_t();

    const char *s = str;

    while (*s != '\0') {
        int rc = FAIL;
        const char *param;

        param = "irmode=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            attr->irmode = (attr_t::round_mode_t)str2rmode(s);
            s += strlen(rmode2str((mkldnn_round_mode_t)attr->irmode));
            rc = OK;
        }

        param = "oscale=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            rc = attr->oscale.str2scale(s, &s);
            if (rc != OK) return rc;
        }

        if (rc != OK) return FAIL;
        if (*s == ';') ++s;
    }

    attr->mkldnn_attr_recreate();

    return OK;
}

void attr2str(const attr_t *attr, char *buffer) {
    buffer += sprintf(buffer, "irmode=%s",
            rmode2str((mkldnn_round_mode_t)attr->irmode));
    buffer += sprintf(buffer, ";oscale=");
    attr->oscale.scale2str(buffer, &buffer);
}
