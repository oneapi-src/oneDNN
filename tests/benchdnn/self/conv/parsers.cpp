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
#include <string.h>
#include <stdlib.h>

#include "self/self.hpp"
#include "conv/conv.hpp"

using namespace conv;

namespace self {

int check_simple_enums() {
    /* alg */
    CHECK_CASE_STR_EQ(alg2str(alg_t::DIRECT), "direct");
    CHECK_CASE_STR_NE(alg2str(alg_t::DIRECT), "directx");

    CHECK_CASE_STR_EQ(alg2str(alg_t::WINO), "wino");
    CHECK_CASE_STR_NE(alg2str(alg_t::WINO), "winox");

    CHECK_EQ(str2alg("direct"), alg_t::DIRECT);
    CHECK_EQ(str2alg("DIRECT"), alg_t::DIRECT);

    CHECK_EQ(str2alg("wino"), alg_t::WINO);
    CHECK_EQ(str2alg("WINO"), alg_t::WINO);

    /* merge */
    CHECK_CASE_STR_EQ(merge2str(merge_t::NONE), "none");
    CHECK_CASE_STR_NE(merge2str(merge_t::NONE), "nonex");

    CHECK_CASE_STR_EQ(merge2str(merge_t::RELU), "relu");
    CHECK_CASE_STR_NE(merge2str(merge_t::RELU), "relux");

    CHECK_EQ(str2merge("none"), merge_t::NONE);
    CHECK_EQ(str2merge("NONE"), merge_t::NONE);

    CHECK_EQ(str2merge("relu"), merge_t::RELU);
    CHECK_EQ(str2merge("RELU"), merge_t::RELU);

    return OK;
}

int check_attr2str() {
    char str[max_attr_len] = {'\0'};

    attr_t attr;
    CHECK_EQ(attr.is_def(), true);

#   define CHECK_ATTR_STR_EQ(attr, s) \
    do { \
        attr2str(&attr, str); \
        CHECK_CASE_STR_EQ(str, s); \
    } while(0)

    CHECK_ATTR_STR_EQ(attr, "irmode=nearest;oscale=none:1");

    attr.irmode = attr_t::round_mode_t::DOWN;
    attr.oscale.policy = attr_t::scale_t::policy_t::COMMON;
    attr.oscale.scale = 2.4;
    CHECK_ATTR_STR_EQ(attr, "irmode=down;oscale=common:2.4");

    attr.irmode = attr_t::round_mode_t::NEAREST;
    attr.oscale.policy = attr_t::scale_t::policy_t::PER_OC;
    attr.oscale.scale = 3.2;
    CHECK_ATTR_STR_EQ(attr, "irmode=nearest;oscale=per_oc:3.2");

#   undef CHECK_ATTR_STR_EQ

    return OK;
}

int check_str2attr() {
    attr_t attr;

#   define CHECK_ATTR(str, irmode, os_policy, os_scale) \
    do { \
        CHECK_EQ(str2attr(&attr, str), OK); \
        CHECK_EQ(attr.irmode, attr_t::round_mode_t:: irmode); \
        CHECK_EQ(attr.oscale.policy, attr_t::scale_t::policy_t:: os_policy); \
        CHECK_EQ(attr.oscale.scale, os_scale); \
    } while(0)

    CHECK_ATTR("", NEAREST, NONE, 1.);
    CHECK_EQ(attr.is_def(), true);

    CHECK_ATTR("irmode=nearest;oscale=none:1.0", DOWN, NONE, 1.);
    CHECK_EQ(attr.is_def(), true);

    CHECK_ATTR("irmode=down;oscale=none:2.0", DOWN, NONE, 2.);
    CHECK_ATTR("irmode=nearest", NEAREST, NONE, 1.);
    CHECK_ATTR("oscale=common:2.0", DOWN, COMMON, 2.);
    CHECK_ATTR("oscale=per_oc:.5;irmode=nearest;", NEAREST, PER_OC, .5);
    CHECK_ATTR("oscale=none:.5;oscale=common:1.5", DOWN, COMMON, 1.5);

#   undef CHECK_ATTR

    return OK;
}

void conv() {
    RUN(check_simple_enums());
    RUN(check_attr2str());
    RUN(check_str2attr());
}

}
