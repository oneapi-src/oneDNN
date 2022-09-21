/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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
#include <string.h>

#include "bnorm/bnorm.hpp"
#include "self/self.hpp"

using namespace bnorm;

namespace self {

static int check_flags() {
    SELF_CHECK_CASE_CPP_STR_EQ(flags2str(NONE), "");
    SELF_CHECK_CASE_CPP_STR_EQ(flags2str(GLOB_STATS), "G");
    SELF_CHECK_CASE_CPP_STR_EQ(flags2str(USE_SCALE), "C");
    SELF_CHECK_CASE_CPP_STR_EQ(flags2str(USE_SHIFT), "H");
    SELF_CHECK_CASE_CPP_STR_EQ(flags2str(USE_SCALE | USE_SHIFT), "CH");
    SELF_CHECK_CASE_CPP_STR_EQ(flags2str(FUSE_NORM_RELU), "R");
    SELF_CHECK_CASE_CPP_STR_EQ(flags2str(GLOB_STATS | FUSE_NORM_RELU), "GR");
    SELF_CHECK_CASE_CPP_STR_EQ(flags2str(FUSE_NORM_ADD_RELU), "A");
    SELF_CHECK_CASE_CPP_STR_EQ(
            flags2str(GLOB_STATS | FUSE_NORM_ADD_RELU), "GA");

    SELF_CHECK_EQ(str2flags(""), NONE);
    SELF_CHECK_EQ(str2flags("G"), GLOB_STATS);
    SELF_CHECK_EQ(str2flags("C"), USE_SCALE);
    SELF_CHECK_EQ(str2flags("H"), USE_SHIFT);
    SELF_CHECK_EQ(str2flags("CH"), USE_SCALE | USE_SHIFT);
    SELF_CHECK_EQ(str2flags("R"), FUSE_NORM_RELU);
    SELF_CHECK_EQ(str2flags("GR"), GLOB_STATS | FUSE_NORM_RELU);
    SELF_CHECK_EQ(str2flags("A"), FUSE_NORM_ADD_RELU);
    SELF_CHECK_EQ(str2flags("GA"), GLOB_STATS | FUSE_NORM_ADD_RELU);
    return OK;
}

static int check_desc() {
    desc_t d {0};
    d.ndims = 4;
    d.mb = 3;
    d.ic = 4;
    d.ih = 5;
    d.iw = 6;
    d.eps = 7.;
    d.name = "test";

    SELF_CHECK_PRINT_EQ(d, "mb3ic4ih5iw6eps7ntest");

    d.ndims = 4;
    d.mb = 2;
    d.iw = d.ih;
    d.eps = 1.f / 16;
    SELF_CHECK_PRINT_EQ(d, "ic4ih5ntest");

    canonical = true;
    SELF_CHECK_PRINT_EQ(d, "mb2ic4ih5iw5eps0.0625ntest");

#define SELF_CHECK_D(_mb, _ic, _ih, _iw, _eps, _name) \
    SELF_CHECK_EQ(d.mb, _mb); \
    SELF_CHECK_EQ(d.ic, _ic); \
    SELF_CHECK_EQ(d.ih, _ih); \
    SELF_CHECK_EQ(d.iw, _iw); \
    SELF_CHECK_EQ(d.eps, _eps); \
    if (d.name != (_name)) return FAIL;
    SELF_CHECK_EQ(str2desc(&d, "mb1ic2ih3iw4eps5ntest2"), OK);
    SELF_CHECK_D(1, 2, 3, 4, 5.f, "test2");
    SELF_CHECK_EQ(str2desc(&d, "ic8ih9ntest3"), OK);
    SELF_CHECK_D(2, 8, 9, 9, 1.f / 16, "test3");
    SELF_CHECK_EQ(str2desc(&d, "ic8iw9ntest3"), OK);
    SELF_CHECK_D(2, 8, 1, 9, 1.f / 16, "test3");
#undef SELF_CHECK_D
    return OK;
}

void bnorm() {
    RUN(check_flags());
    RUN(check_desc());
}

} // namespace self
