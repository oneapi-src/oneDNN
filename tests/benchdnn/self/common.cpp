/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "common.hpp"
#include "dnnl_common.hpp"

#include "self/self.hpp"

namespace self {

static int check_simple_enums() {
    /* attr::post_ops::kind */
    using p = attr_t::post_ops_t;
    CHECK_CASE_STR_EQ(p::kind2str(p::kind_t::SUM), "sum");
    CHECK_CASE_STR_EQ(p::kind2str(p::kind_t::RELU), "relu");

    CHECK_EQ(p::str2kind("sum"), p::kind_t::SUM);
    CHECK_EQ(p::str2kind("SuM"), p::kind_t::SUM);

    CHECK_EQ(p::str2kind("relu"), p::kind_t::RELU);
    CHECK_EQ(p::str2kind("ReLU"), p::kind_t::RELU);

    return OK;
}

static int check_attr2str() {
    attr_t attr;
    CHECK_EQ(attr.is_def(), true);

    CHECK_PRINT_EQ(attr, "");

    attr.oscale.policy = attr_t::scale_t::policy_t::COMMON;
    attr.oscale.scale = 2.4;
    CHECK_PRINT_EQ(attr, "oscale=common:2.4;");

    attr.oscale.policy = attr_t::scale_t::policy_t::PER_OC;
    attr.oscale.scale = 3.2;
    attr.oscale.runtime = true;
    CHECK_PRINT_EQ(attr, "oscale=per_oc:3.2*;");

    attr.oscale.policy = attr_t::scale_t::policy_t::PER_DIM_01;
    attr.oscale.scale = 3.2;
    attr.oscale.runtime = false;
    CHECK_PRINT_EQ(attr, "oscale=per_dim_01:3.2;");

    attr.zero_points.set(DNNL_ARG_SRC, {1, false});
    CHECK_PRINT_EQ(attr, "oscale=per_dim_01:3.2;zero_points=src:1;");

    attr.zero_points.set(DNNL_ARG_WEIGHTS, {2, true});
    CHECK_PRINT_EQ2(attr, "oscale=per_dim_01:3.2;zero_points=src:1_wei:2*;",
            "oscale=per_dim_01:3.2;zero_points=wei:2*_src:1;");

    attr = attr_t();
    attr.scales.set(DNNL_ARG_SRC_0, attr_t::scale_t::policy_t::COMMON, 2.3);
    CHECK_PRINT_EQ(attr, "scales='src:common:2.3';");

    attr.scales.set(DNNL_ARG_SRC_0, attr_t::scale_t::policy_t::COMMON, 2.3);
    attr.scales.set(DNNL_ARG_SRC_1, attr_t::scale_t::policy_t::COMMON, 3);
    CHECK_PRINT_EQ(attr, "scales='src:common:2.3_src1:common:3';");

    return OK;
}

static int check_str2attr() {
    attr_t attr;

#define CHECK_ATTR(str, os_policy, os_scale, os_runtime) \
    do { \
        CHECK_EQ(str2attr(&attr, str), OK); \
        CHECK_EQ(attr.oscale.policy, attr_t::scale_t::policy_t::os_policy); \
        CHECK_EQ(attr.oscale.scale, os_scale); \
        CHECK_EQ(attr.oscale.runtime, os_runtime); \
    } while (0)
#define CHECK_ATTR_ZP(arg, zero_points_value, zero_points_runtime) \
    do { \
        const auto entry = attr.zero_points.get(arg); \
        CHECK_EQ(entry.value, zero_points_value); \
        CHECK_EQ(entry.runtime, zero_points_runtime); \
    } while (0)

    CHECK_ATTR("", NONE, 1., false);
    CHECK_EQ(attr.is_def(), true);

    CHECK_ATTR("oscale=none:1.0", NONE, 1., false);
    CHECK_EQ(attr.is_def(), true);

    CHECK_ATTR(
            "oscale=none:1.0;zero_points=src:0_wei:0_dst:0", NONE, 1., false);
    CHECK_EQ(attr.is_def(), true);

    CHECK_ATTR("oscale=none:2.0", NONE, 2., false);
    CHECK_ATTR("oscale=none:2.0*", NONE, 2., false);
    CHECK_ATTR("oscale=common:2.0*", COMMON, 2., true);
    CHECK_ATTR("oscale=per_oc:.5*;", PER_OC, .5, true);
    CHECK_ATTR("oscale=none:.5*;oscale=common:1.5", COMMON, 1.5, false);

    CHECK_ATTR(
            "oscale=common:2.0*;zero_points=src:0_dst:-2*", COMMON, 2., true);
    CHECK_ATTR_ZP(DNNL_ARG_SRC, 0, false);
    CHECK_ATTR_ZP(DNNL_ARG_WEIGHTS, 0, false);
    CHECK_ATTR_ZP(DNNL_ARG_DST, -2, true);

    CHECK_EQ(str2attr(&attr, "scales='src1:common:1.5';"), OK);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_1).policy,
            attr_t::scale_t::policy_t::COMMON);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_1).scale, 1.5);

    CHECK_EQ(str2attr(&attr, "scales='src:common:2.5_src1:common:1.5';"), OK);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_0).policy,
            attr_t::scale_t::policy_t::COMMON);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_0).scale, 2.5);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_1).policy,
            attr_t::scale_t::policy_t::COMMON);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_1).scale, 1.5);

#undef CHECK_ATTR
#undef CHECK_ATTR_ZP

    return OK;
}

static int check_post_ops2str() {
    attr_t::post_ops_t ops;
    CHECK_EQ(ops.is_def(), true);

    CHECK_PRINT_EQ(ops, "''");

    ops.len = 4;
    for (int i = 0; i < 2; ++i) {
        ops.entry[2 * i + 0].kind = attr_t::post_ops_t::SUM;
        ops.entry[2 * i + 0].sum.scale = 2. + i;
        ops.entry[2 * i + 1].kind = attr_t::post_ops_t::RELU;
        ops.entry[2 * i + 1].eltwise.scale = 1.;
        ops.entry[2 * i + 1].eltwise.alpha = (i == 0 ? 0. : 5.);
        ops.entry[2 * i + 1].eltwise.beta = 0.;
    }
    CHECK_PRINT_EQ(ops, "'sum:2;relu;sum:3;relu:5'");

    ops.len = 3;
    CHECK_PRINT_EQ(ops, "'sum:2;relu;sum:3'");

    ops.len = 2;
    CHECK_PRINT_EQ(ops, "'sum:2;relu'");

    ops.len = 1;
    CHECK_PRINT_EQ(ops, "'sum:2'");

    return OK;
}

static int check_str2post_ops() {
    attr_t::post_ops_t ops;

    CHECK_EQ(ops.is_def(), true);

    auto quick = [&](int len) -> int {
        for (int i = 0; i < 2; ++i) {
            if (2 * i + 0 >= len) return OK;
            CHECK_EQ(ops.entry[2 * i + 0].kind, attr_t::post_ops_t::SUM);
            CHECK_EQ(ops.entry[2 * i + 0].sum.scale, 2. + i);
            if (2 * i + 1 >= len) return OK;
            CHECK_EQ(ops.entry[2 * i + 1].kind, attr_t::post_ops_t::RELU);
            CHECK_EQ(ops.entry[2 * i + 1].eltwise.scale, 1.);
            CHECK_EQ(ops.entry[2 * i + 1].eltwise.alpha, 0.);
            CHECK_EQ(ops.entry[2 * i + 1].eltwise.beta, 0.);
        }
        return OK;
    };

    ops.from_str("''", NULL);
    CHECK_EQ(ops.is_def(), true);

    ops.from_str("'sum:2;'", NULL);
    CHECK_EQ(quick(1), OK);

    ops.from_str("'sum:2;relu'", NULL);
    CHECK_EQ(quick(2), OK);

    ops.from_str("'sum:2;relu;sum:3'", NULL);
    CHECK_EQ(quick(3), OK);

    ops.from_str("'sum:2;relu;sum:3;relu;'", NULL);
    CHECK_EQ(quick(4), OK);

    return OK;
}

void common() {
    RUN(check_simple_enums());
    RUN(check_attr2str());
    RUN(check_str2attr());
    RUN(check_post_ops2str());
    RUN(check_str2post_ops());
}

} // namespace self
