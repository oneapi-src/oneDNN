/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include <functional>
#include <iostream>
#include <vector>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/passlet/ssa_value_hash.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <gtest/gtest.h>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc;

using namespace passlet;
namespace ssa_hash {

struct viewer_t : public ssa_viewer_t {
    using ssa_viewer_t::dispatch;
    using ssa_viewer_t::view;

    ssa_value_hash_t hasher_;
    struct result_t {
        size_t val_ = 0;
    };
    viewer_t()
        : hasher_ {temp_data_addresser<result_t, size_t, &result_t::val_>()} {}
    void view(define_c v) override {
        hasher_.view(v, passlet::PRE_VISIT);
        ssa_viewer_t::view(v);
        hasher_.view(v, passlet::POST_VISIT);
    };
};

} // namespace ssa_hash

static constexpr auto s32 = datatypes::s32;

static size_t get_status(const stmt &v) {
    return v->temp_data().get<ssa_hash::viewer_t::result_t>().val_;
}

TEST(GCCore_CPU_ssa_value_hash, TestSSAValueHash) {
    builder::ir_builder_t bld;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        _var_ex_(v0, s32, linkage::local, 2);
        A[a] = a + 2;
        A[a] = b + 2;
        A[a] = a + g;
        A[a] = a + g;
        A[a] = (a + 2) + A[0];
        A[a] = (a + 2) + A[0];
        A[a] = (a + 2) + A[0] + 2;
        A[a] = (a + 2) + A[0] + 2;
        _var_ex_(v1, s32, linkage::local, 2);
        _var_ex_(v2, s32, linkage::local, v0);
        _return_(v1 + v2);
    }

    auto ssa_ccc = ssa_transform_t()(ccc);
    ssa_hash::viewer_t().dispatch(ssa_ccc);
    auto &seq = ssa_ccc->body_.checked_as<stmts>()->seq_;

    /* Reference hash value (may change because of change of address)
0 var g: s32 0
1 var v0: s32 = 2 2654436219
2 var __tmp0: s32 = 2 2654436219
3 var __tmp1: s32 = (a + __tmp0) 179049339825
5 var __tmp2: s32 = 2 2654436219
6 var __tmp3: s32 = (b + __tmp2) 179049350365
8 var __tmp4: s32 = g 15973760
9 var __tmp5: s32 = (a + __tmp4) 172048795018
11 var __tmp6: s32 = g 15973760
12 var __tmp7: s32 = (a + __tmp6) 172048795018
14 var __tmp8: s32 = 2 2654436219
15 var __tmp9: s32 = (a + __tmp8) 179049339825
16 var __tmp10: s32 = 0 2654436221
17 var __tmp11: s32 = A[__tmp10] 179049365863
18 var __tmp12: s32 = (__tmp9 + __tmp11) 11966659815956
20 var __tmp13: s32 = 2 2654436219
21 var __tmp14: s32 = (a + __tmp13) 179049339825
22 var __tmp15: s32 = 0 2654436221
23 var __tmp16: s32 = A[__tmp15] 179049365863
24 var __tmp17: s32 = (__tmp14 + __tmp16) 11966659815956
26 var __tmp18: s32 = 2 2654436219
27 var __tmp19: s32 = (a + __tmp18) 179049339825
28 var __tmp20: s32 = 0 2654436221
29 var __tmp21: s32 = A[__tmp20] 179049365863
30 var __tmp22: s32 = (__tmp19 + __tmp21) 11966659815956
31 var __tmp23: s32 = 2 2654436219
32 var __tmp24: s32 = (__tmp22 + __tmp23) 758166112429100
34 var __tmp25: s32 = 2 2654436219
35 var __tmp26: s32 = (a + __tmp25) 179049339825
36 var __tmp27: s32 = 0 2654436221
37 var __tmp28: s32 = A[__tmp27] 179049365863
38 var __tmp29: s32 = (__tmp26 + __tmp28) 11966659815956
39 var __tmp30: s32 = 2 2654436219
40 var __tmp31: s32 = (__tmp29 + __tmp30) 758166112429100
42 var v1: s32 = 2 2654436219
43 var v2: s32 = v0 2654436219
44 var __tmp32: s32 = (v1 + v2) 350495361329
    */
    auto hash_2 = get_status(seq.at(1)); // hash of constant 2
    ASSERT_EQ(get_status(seq.at(5)), hash_2);
    // test of copy propagation: v2=v0, and v2
    // and v0 share the same hash value
    ASSERT_EQ(get_status(seq.at(43)), hash_2);

    auto hash_a_plus_2 = get_status(seq.at(3)); // hash of a+2
    ASSERT_EQ(get_status(seq.at(15)), hash_a_plus_2);
    ASSERT_EQ(get_status(seq.at(21)), hash_a_plus_2);
    ASSERT_EQ(get_status(seq.at(27)), hash_a_plus_2);
    ASSERT_EQ(get_status(seq.at(35)), hash_a_plus_2);

    auto hash_g_plus_2 = get_status(seq.at(9)); // hash of g+2
    ASSERT_EQ(get_status(seq.at(12)), hash_g_plus_2);

    auto hash_a_plus_2_plus_a0 = get_status(seq.at(18)); // a+2+A[0]
    ASSERT_EQ(get_status(seq.at(24)), hash_a_plus_2_plus_a0);
    ASSERT_EQ(get_status(seq.at(30)), hash_a_plus_2_plus_a0);
    ASSERT_EQ(get_status(seq.at(38)), hash_a_plus_2_plus_a0);

    auto last_level = get_status(seq.at(32)); // a+2+A[0]+2
    ASSERT_EQ(get_status(seq.at(40)), last_level);
}

TEST(GCCore_CPU_ssa_value_hash, TestSSAValueHashAdd) {
    builder::ir_builder_t bld;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        _return_((a + b) + (b + a));
    }
    auto ssa_ccc = ssa_transform_t()(ccc);
    ssa_hash::viewer_t().dispatch(ssa_ccc);
    auto &seq = ssa_ccc->body_.checked_as<stmts>()->seq_;
    ASSERT_EQ(get_status(seq.at(0)), get_status(seq.at(1)));
}
