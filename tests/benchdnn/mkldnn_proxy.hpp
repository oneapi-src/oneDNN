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

#ifndef _MKLDNN_PROXY_HPP
#define _MKLDNN_PROXY_HPP

#include "mkldnn_types.h"

struct attr_t {
    enum round_mode_t {
        NEAREST = (int)mkldnn_round_nearest,
        DOWN = (int)mkldnn_round_down,
    };

    struct scale_t {
        enum policy_t { NONE = 0, COMMON, PER_OC, POLICY_TOTAL };
        static policy_t str2policy(const char *str);
        static const char *policy2str(policy_t policy);

        policy_t policy = NONE;
        float scale = 1.;

        int str2scale(const char *str, const char **end_s);
        void scale2str(char *buffer, char **end_b) const;

        bool is_def() const { return this->policy == NONE; }
    };

    round_mode_t irmode = NEAREST;
    scale_t oscale;
    mkldnn_primitive_attr_t mkldnn_attr = NULL;

    bool is_def() const;
    int mkldnn_attr_recreate();
};

const size_t max_attr_len = 128;
int str2attr(attr_t *attr, const char *str);
void attr2str(const attr_t *attr, char *buffer);

#endif
