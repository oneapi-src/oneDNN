/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include <iostream>
#include "exception_util.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/transform/loop_merge.hpp>
#include <compiler/ir/transform/loop_unroll.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_loop_transform_cpp, TestLoopTransform) {
    builder::ir_builder_t builder;

    for_loop li, lj, lk, lp;
    _function_(datatypes::f32, aaa, _arg_("ii", datatypes::s32),
            _arg_("jj", datatypes::s32),
            _arg_("buf", datatypes::s32, {100, 200})) {
        _bind_(ii, jj, buf);
        _tensor_(buf2, datatypes::s32, {100, 200});
        _tensor_(buf3, datatypes::s32, {100, 200});
        _var_(v1, datatypes::f32);
        _var_(v2, datatypes::f32);
        _named_for_(li, i, 0, 20, 1, for_type::PARALLEL) {
            _named_for_(lj, j, 0, 50) {
                _named_for_(lk, k, 0, 30) {
                    buf3[{i, j}] = buf3[{i, j}] + buf[{i, k}] * buf2[{k, j}];
                    _named_for_(lp, p, 0, 20) { buf3[{i, p}] = 3; }
                }
            }
        }
    }

    auto lk_i = lk->split(5); //[li,lj,lk,lk_i]
    auto li_i = li->split(5); //[li,li_i,lj,lk,lk_i]
    auto li_i_j = li_i->fuse(lj);
    li->reorder(aaa->body_, {li, lk, lk_i, li_i_j});
    lp->split(5);

    _function_(datatypes::f32, bbb, _arg_("ii", datatypes::s32),
            _arg_("jj", datatypes::s32),
            _arg_("buf", datatypes::s32, {100, 200})) {
        _bind_(ii, jj, buf);
        _tensor_(buf2, datatypes::s32, {100, 200});
        _tensor_(buf3, datatypes::s32, {100, 200});
        _var_(v1, datatypes::f32);
        _var_(v2, datatypes::f32);
        _for_(io, 0UL, 4UL, 1, for_type::PARALLEL) {
            _for_(ko, 0UL, 6UL) {
                _for_(ki, 0UL, 5UL, 1UL) {
                    _for_(ii_j, 0UL, 250UL, 1UL) {
                        expr j = (ii_j % UINT64_C(50));
                        expr i = io * UINT64_C(5) + ii_j / UINT64_C(50);
                        expr k = (ko * UINT64_C(5)) + ki;
                        buf3[{i, j}]
                                = buf3[{i, j}] + buf[{i, k}] * buf2[{k, j}];
                        _for_(po, 0UL, 4UL, 1) {
                            _for_(pi, 0UL, 5UL, 1UL) {
                                expr p = po * UINT64_C(5) + pi;
                                buf3[{i, p}] = 3;
                            }
                        }
                    }
                }
            }
        }
    }
    ir_comparer cmper;
    EXPECT_TRUE(aaa->equals(bbb, cmper));
}

TEST(GCCore_CPU_loop_transform_cpp, LoopTransformSplit) {
    builder::ir_builder_t builder;
    builder.push_scope();
    for_loop li, lj;
    _tensor_(buf, datatypes::f32, {100});
    _named_for_(li, i, 0, 20) { buf[i] = 4.0f; }
    expr idx;
    _named_for_(lj, i, 0, 20) {
        idx = i + 2;
        buf[idx] = buf[idx + idx];
    }
    builder.pop_scope();
    // infinite loop check
    lj->split(2);
    EXPECT_SC_ERROR(li->split(6),
            "The loop length 20 should be divisible of and larger than "
            "the block size 6");

    builder.push_scope();
    _var_(len, datatypes::s32);
    _named_for_(li, i, len, 20) { buf[i] = 4.0f; }
    builder.pop_scope();
    EXPECT_SC_ERROR(li->split(4),
            "Only support constant for loops for for-loop-transforms: ");

    builder.push_scope();
    _named_for_(li, i, 1, 20, 2) { buf[i] = 4.0f; }
    builder.pop_scope();
    EXPECT_SC_ERROR(li->split(4),
            "for-loop-transforms only support step=1: for i in (1, 20, 2)");

    builder.push_scope();
    _named_for_(li, i, 100, 20, 1) { buf[i] = 4.0f; }
    builder.pop_scope();
    EXPECT_SC_ERROR(li->split(4),
            "for-loop-transforms: the begin should be less than or eq to end: "
            "for i ");
}

TEST(GCCore_CPU_loop_transform_cpp, LoopTransformFuse) {
    builder::ir_builder_t builder;
    builder.push_scope();
    for_loop li, lj, lk, lp;
    _tensor_(buf, datatypes::f32, {100});
    _named_for_(li, i, 0, 20) {
        _named_for_(lj, j, 0, 20) {
            _named_for_(lk, k, 0, 20) {
                _named_for_(lp, p, 0, 20) { buf[i + j + k + p] = 4.0f; }
            }
        }
    }

    builder.pop_scope();
    lk->fuse(lp);
    li->fuse(lj);
    EXPECT_SC_ERROR(lj->fuse(lk), "Transforming an invalid for-loop: this");
    EXPECT_SC_ERROR(li->fuse(lj), "Transforming an invalid for-loop: ax");
}

TEST(GCCore_CPU_loop_transform_cpp, LoopTransformReorder) {
    builder::ir_builder_t builder;
    builder.push_scope();
    for_loop li, lj, lk, lp, lg;
    _tensor_(buf, datatypes::f32, {100});
    _named_for_(li, i, 0, 20) {
        _named_for_(lj, j, 0, 20) {
            _named_for_(lk, k, 0, 20) {
                _named_for_(lp, p, 0, 20) {
                    buf[i + j + k + p] = 4.0f;
                    _named_for_(lg, g, 0, 20) { buf[g] = 3.0f; }
                }
            }
        }
    }

    auto body = builder.pop_scope();
    EXPECT_SC_ERROR(li->reorder(body, {}),
            "The number of axises to reorder should > 0");
    EXPECT_SC_ERROR(li->reorder(body, {li, lj, lk, lp, lg}),
            "Bad number of axises to reorder. Got 5 to reorder, but only have "
            "4 nested for-loops");
    EXPECT_SC_ERROR(li->reorder(body, {li, lk}),
            "Cannot find axis j in the given axises to reorder");
    EXPECT_SC_ERROR(li->reorder(body, {li, lj, lj}),
            "Cannot find axis k in the given axises to reorder");
    EXPECT_SC_ERROR(li->reorder(make_stmt<stmts_node_t>(std::vector<stmt> {}),
                            {li, lj, lk}),
            "Cannot find the for-loop to replace in the parent stmt");
}

TEST(GCCore_CPU_loop_transform_cpp, LoopTransformReorderTwoLoops) {
    builder::ir_builder_t builder;
    builder.push_scope();
    for_loop li, lj;
    _tensor_(buf, datatypes::f32, {100});
    _named_for_(li, i, 1, 30) {
        _named_for_(lj, j, 0, 20) { buf[i + j] = 3; }
    }

    auto body = builder.pop_scope();
    li->reorder(body, {lj, li});

    builder.push_scope();
    _tensor_(buf2, datatypes::f32, {100});
    _for_(j, 0, 20) {
        _for_(i, 1, 30) { buf2[i + j] = 3; }
    }

    ir_comparer cmper;
    EXPECT_TRUE(body->equals(builder.pop_scope(), cmper));
}

TEST(GCCore_CPU_loop_transform_cpp, LoopTransformMergeGood) {
    builder::ir_builder_t builder;
    builder.push_scope();
    for_loop li, lj;
    _tensor_(buf, datatypes::f32, {100});
    _var_(len, datatypes::s32);
    _named_for_(li, i, 0, len, len / 20) { buf[i] = 3; }
    _named_for_(lj, j, 0, len, len / 20) { buf[j] = buf[j] + 1; }
    auto body = builder.pop_scope();
    li->merge(body, lj);
    EXPECT_FALSE(lj->isvalid());

    builder.push_scope();
    _tensor_(buf2, datatypes::f32, {100});
    _var_(len2, datatypes::s32);
    _for_(i, 0, len2, len2 / 20) {
        buf2[i] = 3;
        buf2[i] = buf2[i] + 1;
    }
    ir_comparer cmper;
    EXPECT_TRUE(body->equals(builder.pop_scope(), cmper));
}

TEST(GCCore_CPU_loop_transform_cpp, LoopTransformMergeDeath) {
    builder::ir_builder_t builder;
    for_loop li, lj;
    stmt body;
    expr len_v;
    auto make = [&]() {
        builder.push_scope();
        _tensor_(buf, datatypes::f32, {100});
        _var_(len, datatypes::s32);
        len_v = len.get();
        _named_for_(li, i, 0, len, len / 20) { buf[i] = 3; }
        _named_for_(lj, j, 0, len, len / 20) { buf[j] = buf[j] + 1; }
        body = builder.pop_scope();
    };

    make();
    stmt s = make_stmt<evaluate_node_t>(expr(0));
    EXPECT_SC_ERROR(lj->merge(s, li), "The parent should be an stmts_node_t");

    make();
    s = make_stmt<stmts_node_t>(std::vector<stmt>());
    EXPECT_SC_ERROR(lj->merge(s, li), "Cannot find the axises in the parent");

    make();
    EXPECT_SC_ERROR(
            lj->merge(body, lj), "The axis to merge should not be \'this\'");
    make();
    lj->step_ = expr(20);
    EXPECT_SC_ERROR(lj->merge(body, li),
            "The ranges of the merged for-loops should be the same");

    make();
    lj->step_ = len_v / 20;
    li->merge(body, lj);
    EXPECT_SC_ERROR(lj->merge(body, li),
            "Invalid for-loop. It has been fused or merged");
}

TEST(GCCore_CPU_loop_transform_cpp, LoopTransformMergeMultiGood) {
    builder::ir_builder_t builder;
    for_loop li, lj;
    stmt body;
    auto make = [&]() {
        builder.push_scope();
        _tensor_(buf, datatypes::f32, {100});
        _var_(len, datatypes::s32);
        _named_for_(li, i, 0, len, 1) {
            _for_(j, 0, i, 1) {
                _for_(k, 0, j, 1) { buf[i + j + k] = 3; }
            }
        }
        _named_for_(lj, i, 0, len, 1) {
            _for_(j, 0, i, 1) {
                _for_(k, 0, j, 1) { buf[i + j * k] = 1; }
            }
        }
        body = builder.pop_scope();
    };
    make();
    EXPECT_SC_ERROR(li->merge(body, lj, 4),
            "Merging 4 inner loops, but have only 3 loops in the IR");

    make();
    li->merge(body, lj, 3);
    EXPECT_FALSE(lj->isvalid());

    builder.push_scope();
    _tensor_(buf2, datatypes::f32, {100});
    _var_(len2, datatypes::s32);
    _named_for_(li, i, 0, len2, 1) {
        _for_(j, 0, i, 1) {
            _for_(k, 0, j, 1) {
                buf2[i + j + k] = 3;
                buf2[i + j * k] = 1;
            }
        }
    }
    ir_comparer cmper;
    EXPECT_TRUE(body->equals(builder.pop_scope(), cmper));
}

TEST(GCCore_CPU_loop_transform_cpp, LoopTransformMergeAll) {
    builder::ir_builder_t builder;
    builder.push_scope();
    for_loop li, lj;
    _tensor_(buf, datatypes::f32, {100});
    _var_(len, datatypes::s32);
    _named_for_(li, i, 0, len, 1) {
        _for_(j, 0, i, 1) {
            _for_(k, 0, j, 1) {
                _for_(t, 0, j, 1) { buf[i + j + k] = 3; }
            }
        }
    }
    _named_for_(lj, i, 0, len, 1) {
        _for_(j, 0, i, 1) {
            _for_(k, 0, j, 1) {
                _for_(t, 0, i, 1) { buf[i + j + k] = 1; }
            }
        }
    }
    auto body = builder.pop_scope();
    EXPECT_EQ(li->merge_all(body, lj), 3);

    builder.push_scope();
    _tensor_(buf2, datatypes::f32, {100});
    _var_(len2, datatypes::s32);
    _named_for_(li, i, 0, len2, 1) {
        _for_(j, 0, i, 1) {
            _for_(k, 0, j, 1) {
                _for_(t, 0, j, 1) { buf2[i + j + k] = 3; }
                _for_(t, 0, i, 1) { buf2[i + j + k] = 1; }
            }
        }
    }
    ir_comparer cmper(true);
    EXPECT_TRUE(body->equals(builder.pop_scope(), cmper));
}

TEST(GCCore_CPU_loop_transform_cpp, LoopMergeRecursivePass) {
    builder::ir_builder_t builder;
    for_loop li[3];
    builder.push_scope();
    _tensor_(buf, datatypes::f32, {100, 100});
    _var_(len, datatypes::s32);
    builder.push_scope();
    _named_for_(li[0], i, 0, len) {
        _for_(j, 0, len) { buf[i][j] = 3; }
    }
    _named_for_(li[1], i, 0, len) {
        // the inner loop is not mergable with first loop, but can be merged
        // with last loop
        _for_(j, 1, len) { buf[i][j] = buf[i][j] * 1; }
    }
    _named_for_(li[2], i, 0, len) {
        _for_(j, 1, len) { buf[i][j] = buf[i][j] + 1; }
    }
    auto body = builder.pop_scope().static_as<stmts>();

    li[0]->attr()[stmt_attr_key::merge_loop] = true;
    li[1]->attr()[stmt_attr_key::merge_loop] = true;
    li[2]->attr()[stmt_attr_key::merge_loop] = true;

    auto out = loop_merger_t()(body);
    // std::cout << out << std::endl;

    builder.push_scope();
    _for_(i, 0, len) {
        _for_(j, 0, len) { buf[i][j] = 3; }
        _for_(j, 1, len) {
            buf[i][j] = buf[i][j] * 1;
            buf[i][j] = buf[i][j] + 1;
        }
    }
    ir_comparer cmper(true);
    auto expected = builder.pop_scope();
    // std::cout << expected << std::endl;
    EXPECT_TRUE(cmper.compare(expected, out, false));
    // std::cout << cmper << std::endl;
}

TEST(GCCore_CPU_loop_transform_cpp, LoopMergePass) {
    builder::ir_builder_t builder;
    for_loop li[3];
    builder.push_scope();
    _tensor_(buf, datatypes::f32, {100});
    _var_(len, datatypes::s32);
    builder.push_scope();
    _named_for_(li[0], i, 0, len) { buf[i] = 3; }
    _named_for_(li[1], j, 0, len) { buf[j] = buf[j] + 1; }
    // the third loop is not mergable
    _named_for_(li[2], j, 1, len) { buf[j] = buf[j] + 1; }
    auto body = builder.pop_scope().static_as<stmts>();

    // make a copy of the original body
    builder.push_scope();
    _for_(i, 0, len) { buf[i] = 3; }
    _for_(j, 0, len) { buf[j] = buf[j] + 1; }
    // the third loop is not mergable
    _for_(j, 1, len) { buf[j] = buf[j] + 1; }
    auto body_copy = builder.pop_scope();

    li[0]->attr()[stmt_attr_key::merge_loop] = true;
    li[1]->attr()[stmt_attr_key::merge_loop] = true;
    li[2]->attr()[stmt_attr_key::merge_loop] = true;

    auto out = loop_merger_t()(body);

    builder.push_scope();
    _for_(i, 0, len) {
        buf[i] = 3;
        buf[i] = buf[i] + 1;
    }
    _for_(j, 1, len) { buf[j] = buf[j] + 1; }
    ir_comparer cmper(true);
    auto expected = builder.pop_scope();
    EXPECT_TRUE(cmper.compare(expected, out));
    // make sure original body is unchanged
    EXPECT_TRUE(cmper.compare(body, body_copy));

    // simulate inlined function bodies (stmts node with only one for-node)
    auto mk_stmts = [](const std::vector<stmt> &arr) {
        return make_stmt<stmts_node_t>(std::vector<stmt>(arr));
    };
    stmts inlined_body = mk_stmts({
            mk_stmts({body->seq_[0]}),
            mk_stmts({body->seq_[1]}),
            mk_stmts({body->seq_[2]}),
    });
    EXPECT_TRUE(cmper.compare(expected, loop_merger_t()(inlined_body)));

    // loops without the attr should not be merged
    li[0]->attr().remove(stmt_attr_key::merge_loop);
    out = loop_merger_t()(body);
    EXPECT_TRUE(out->equals(body, cmper));
}

TEST(GCCore_CPU_loop_transform_cpp, LoopMergeWithExprPass) {
    builder::ir_builder_t builder;
    for_loop li[4];
    builder.push_scope();
    _tensor_(buf, datatypes::f32, {100});
    _var_(len, datatypes::s32);
    builder.push_scope();
    _var_(test_1, datatypes::s32);
    _named_for_(li[0], i, 0, len) { buf[i] = test_1; }
    _var_(test_2, datatypes::s32);
    _named_for_(li[1], j, 0, len) { buf[j] = buf[j] + test_2; }
    _var_(test_3, datatypes::s32);
    _named_for_(li[2], k, 0, len) { buf[k] = buf[k] * test_3; }
    // the forth loop is not mergable
    _named_for_(li[3], m, 1, len) { buf[m] = buf[m] + 1; }
    auto body = builder.pop_scope().static_as<stmts>();

    li[0]->attr()[stmt_attr_key::merge_loop] = true;
    li[1]->attr()[stmt_attr_key::merge_loop] = true;
    li[2]->attr()[stmt_attr_key::merge_loop] = true;
    li[3]->attr()[stmt_attr_key::merge_loop] = true;

    auto out = loop_merger_t()(body);
    // std::cout << out << std::endl;

    builder.push_scope();
    _var_(test_exp_1, datatypes::s32);
    _var_(test_exp_2, datatypes::s32);
    _var_(test_exp_3, datatypes::s32);
    _for_(i, 0, len) {
        buf[i] = test_exp_1;
        buf[i] = buf[i] + test_exp_2;
        buf[i] = buf[i] * test_exp_3;
    }
    _for_(j, 1, len) { buf[j] = buf[j] + 1; }
    auto expected = builder.pop_scope();
    // std::cout << expected << std::endl;
    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(expected, out, false));
    // std::cout << cmper << std::endl;
}

TEST(GCCore_CPU_loop_transform_cpp, LoopUnrollPass) {
    builder::ir_builder_t builder;
    for_loop li, li2, lj, lj1;
    builder.push_scope();
    _tensor_(A, datatypes::f32, 100, 100);
    _tensor_(B, datatypes::f32, 100, 100);
    _named_for_(li, i, 0, 100, 2) {
        _for_(k, 0, 100, 1) {
            _named_for_(lj1, j, 0, 4, 2) {
                A[{i, j}] = A[{j, i}];
                A[{i, j}] = i + 1;
            }
        }
    }
    _named_for_(li2, i, 0, 100, 2) {
        _for_(k, 0, 100, 1) {
            _named_for_(lj, j, 0, 4, 2) {
                B[{i, j}] = A[{j, i}];
                B[{i, j}] = i + 1;
            }
        }
    }
    auto body = builder.pop_scope().static_as<stmts>();

    li->attr()[stmt_attr_key::merge_loop] = true;
    li2->attr()[stmt_attr_key::merge_loop] = true;
    lj->attr()[stmt_attr_key::unroll_loop] = 0;

    auto out = loop_unroller_t()(loop_merger_t()(body));

    builder.push_scope();
    {
        _tensor_(A, datatypes::f32, 100, 100);
        _tensor_(B, datatypes::f32, 100, 100);
        _named_for_(li, i, 0, 100, 2) {
            _for_(k, 0, 100, 1) {
                {
                    uint64_t j = 0;
                    builder.push_scope();
                    A[{i, j}] = A[{j, i}];
                    A[{i, j}] = i + 1;
                    B[{i, j}] = A[{j, i}];
                    B[{i, j}] = i + 1;
                    builder.emit(builder.pop_scope());
                }
                {
                    uint64_t j = 2;
                    builder.push_scope();
                    A[{i, j}] = A[{j, i}];
                    A[{i, j}] = i + 1;
                    B[{i, j}] = A[{j, i}];
                    B[{i, j}] = i + 1;
                    builder.emit(builder.pop_scope());
                }
            }
        }
    }
    auto expected = builder.pop_scope();
    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(expected, out, false));
}

TEST(GCCore_CPU_loop_transform_cpp, Unroll) {
    builder::ir_builder_t builder;
    for_loop lo1, lo2, lo3, lo4;
    for_loop lr1, lr2, lr3, lr4;
    builder.push_scope();
    _tensor_(buf, datatypes::f32, {100});
    _var_(len, datatypes::s32);
    builder.push_scope();
    _named_for_(lo1, i, 3, 300) { buf[i] = i; }
    _named_for_(lo2, j, 0, 20, 2) {
        buf[j] = buf[j] + 1;
        buf[j + len] = 0;
    }

    _named_for_(lr1, j, 1, len) { buf[j] = 0; }
    _named_for_(lr2, j, 1, 4) { buf[j] = 0; }
    _named_for_(lr3, j, 1, 3) {
        buf[j] = 0;
        buf[j + UINT64_C(3)] = j * j;
    }
    stmt body = builder.pop_scope();

    /////// expected
    builder.push_scope();
    _for_(i_u, 0, 99UL, 1) {
        _var_ex_(i, datatypes::index, linkage::local, i_u * UINT64_C(3) + 3);
        builder.push_scope();
        buf[i + UINT64_C(0)] = i + UINT64_C(0);
        builder.emit(builder.pop_scope());
        builder.push_scope();
        buf[i + UINT64_C(1)] = i + UINT64_C(1);
        builder.emit(builder.pop_scope());
        builder.push_scope();
        buf[i + UINT64_C(2)] = i + UINT64_C(2);
        builder.emit(builder.pop_scope());
    }
    _for_(j_u, 0, 5UL, 1) {
        _var_ex_(j, datatypes::index, linkage::local, j_u * UINT64_C(4) + 0);
        builder.push_scope();
        buf[j + UINT64_C(0)] = buf[j + UINT64_C(0)] + 1;
        buf[j + UINT64_C(0) + len] = 0;
        builder.emit(builder.pop_scope());
        builder.push_scope();
        buf[j + UINT64_C(2)] = buf[j + UINT64_C(2)] + 1;
        buf[j + UINT64_C(2) + len] = 0;
        builder.emit(builder.pop_scope());
    }

    _for_(i_u, 0, (((len - 1) / (expr(1) * UINT64_C(2))) + 1), 1) {
        _if_(i_u < (len - 1) / (expr(1) * UINT64_C(2))) {
            _var_ex_(i, datatypes::index, linkage::local,
                    i_u * (expr(1) * UINT64_C(2)) + 1);
            builder.push_scope();
            buf[i + expr(0UL) * 1] = 0;
            builder.emit(builder.pop_scope());
            builder.push_scope();
            buf[i + expr(1UL) * 1] = 0;
            builder.emit(builder.pop_scope());
        }
        else {
            _for_(i,
                    (len - 1) / (expr(1) * UINT64_C(2))
                                    * (expr(1) * UINT64_C(2))
                            + 1,
                    len) {
                buf[i] = 0;
            }
        }
    }

    _for_(i_u, 0, 2UL, 1) {
        _if_(i_u < UINT64_C(1)) {
            _var_ex_(
                    i, datatypes::index, linkage::local, i_u * UINT64_C(2) + 1);
            builder.push_scope();
            buf[i + UINT64_C(0)] = 0;
            builder.emit(builder.pop_scope());
            builder.push_scope();
            buf[i + UINT64_C(1)] = 0;
            builder.emit(builder.pop_scope());
        }
        else {
            _for_(i, 3UL, 4) { buf[i] = 0; }
        }
    }
    expr a = make_expr<constant_node>((uint64_t)1, datatypes::index);
    builder.push_scope();
    buf[a + UINT64_C(0)] = 0;
    buf[a + UINT64_C(0) + UINT64_C(3)] = (a + UINT64_C(0)) * (a + UINT64_C(0));
    builder.emit(builder.pop_scope());
    builder.push_scope();
    buf[a + UINT64_C(1)] = 0;
    buf[a + UINT64_C(1) + UINT64_C(3)] = (a + UINT64_C(1)) * (a + UINT64_C(1));
    builder.emit(builder.pop_scope());
    auto expected = builder.pop_scope();

    lo1->unroll(3);
    lo2->unroll(2);
    lr1->unroll(2);
    lr2->unroll(2);
    lr3->unroll(0, body);

    ir_comparer cmp(true);
    cmp.compare(body, expected, false);
    EXPECT_TRUE(cmp.compare(body, expected, false));
}

TEST(GCCore_CPU_loop_transform_cpp, UnrollBad) {
    builder::ir_builder_t builder;
    builder.push_scope();
    for_loop lo1, lo2;
    _var_(a, datatypes::f32);
    _named_for_(lo1, i, 3, 300) {
        a = i;
        _var_ex_(a2, datatypes::f32, linkage::static_local);
    }
    _named_for_(lo2, i, 3, 300) {
        a = i;
        _var_(a2, datatypes::f32);
    }

    // good unroll with local var
    lo2->unroll(2);
    // bad unroll with static var
    EXPECT_SC_ERROR(
            lo1->unroll(2), "Only allow local variables in unroll, got:");
}

TEST(GCCore_CPU_loop_transform_cpp, SerialMerge) {
    builder::ir_builder_t builder;
    for_loop lo1, lo2;
    builder.push_scope();
    _tensor_(buf, datatypes::f32, {100});
    _var_(len, datatypes::s32);
    builder.push_scope();
    _named_for_(lo1, i, 3, 300) { buf[i] = i; }
    _named_for_(lo2, j, 1, 20) { buf[j + len] = 0; }
    stmt body = builder.pop_scope();

    /////// expected
    builder.push_scope();
    _for_(i, 3, expr(300) + 20 - 1) {
        _if_(i < 300) { buf[i] = i; }
        _else_ {
            _var_ex_(j, datatypes::index, linkage::local, i - 300 + 1);
            buf[j + len] = 0;
        }
    }
    auto expected = builder.pop_scope();

    lo1->parallel_merge(body, lo2);

    ir_comparer cmp(true);
    EXPECT_TRUE(cmp.compare(body, expected, false));
}
