/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <memory>
#include <thread>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"

#include "backend/dnnl/resource.hpp"
#include "backend/dnnl/subgraph/memory_binding.hpp"

#include <dnnl.hpp>

using dnnl::graph::impl::dnnl_impl::resource_cache_t;
using dnnl::graph::impl::dnnl_impl::resource_t;

struct test_resource_t : public resource_t {
    test_resource_t(size_t data) : data_(data) {}
    size_t data_;
};

TEST(resource, resource_cache) {
    resource_cache_t resource_cache;
    resource_cache.clear();

    resource_cache_t::key_t key1 = (resource_cache_t::key_t)1;
    std::unique_ptr<resource_t> resource1(new test_resource_t(10));
    resource_cache.add<test_resource_t>(key1, std::move(resource1));

    resource_cache_t::key_t key2 = (resource_cache_t::key_t)2;
    std::unique_ptr<resource_t> resource2(new test_resource_t(20));
    resource_cache.add<test_resource_t>(key2, std::move(resource2));

    ASSERT_TRUE(resource_cache.has_resource(key1));
    ASSERT_TRUE(resource_cache.has_resource(key2));

    test_resource_t *resource_ptr1 = resource_cache.get<test_resource_t>(key1);
    test_resource_t *resource_ptr2 = resource_cache.get<test_resource_t>(key2);

    ASSERT_EQ(resource_ptr1->data_, 10);
    ASSERT_EQ(resource_ptr2->data_, 20);

    resource_ptr1->data_ = 30;
    resource_ptr1 = resource_cache.get<test_resource_t>(key1);
    ASSERT_EQ(resource_ptr1->data_, 30);

    // the given creator will not take effect since the key1 is already in the
    // cache
    resource_ptr1 = resource_cache.get<test_resource_t>(key1, []() {
        return std::unique_ptr<resource_t>(new test_resource_t(100));
    });
    ASSERT_EQ(resource_ptr1->data_, 30);
    ASSERT_EQ(resource_cache.size(), 2);

    // the given creator will take effect since the key3 is not in the cache
    resource_cache_t::key_t key3 = (resource_cache_t::key_t)3;
    test_resource_t *resource_ptr3
            = resource_cache.get<test_resource_t>(key3, []() {
                  return std::unique_ptr<resource_t>(new test_resource_t(100));
              });
    ASSERT_EQ(resource_ptr3->data_, 100);
    ASSERT_EQ(resource_cache.size(), 3);
}

TEST(resource, resource_cache_multithreading) {
    auto func = []() {
        resource_cache_t resource_cache;
        resource_cache.clear();

        ASSERT_EQ(resource_cache.size(), 0);

        resource_cache_t::key_t key1 = (resource_cache_t::key_t)1;
        std::unique_ptr<resource_t> resource1(new test_resource_t(10));
        resource_cache.add<test_resource_t>(key1, std::move(resource1));

        resource_cache_t::key_t key2 = (resource_cache_t::key_t)2;
        std::unique_ptr<resource_t> resource2(new test_resource_t(20));
        resource_cache.add<test_resource_t>(key2, std::move(resource2));

        ASSERT_TRUE(resource_cache.has_resource(key1));
        ASSERT_TRUE(resource_cache.has_resource(key2));

        test_resource_t *resource_ptr1
                = resource_cache.get<test_resource_t>(key1);
        test_resource_t *resource_ptr2
                = resource_cache.get<test_resource_t>(key2);

        ASSERT_EQ(resource_ptr1->data_, 10);
        ASSERT_EQ(resource_ptr2->data_, 20);

        resource_ptr1->data_ = 30;
        resource_ptr1 = resource_cache.get<test_resource_t>(key1);
        ASSERT_EQ(resource_ptr1->data_, 30);

        resource_cache_t::key_t key3 = (resource_cache_t::key_t)3;
        test_resource_t *resource_ptr3
                = resource_cache.get<test_resource_t>(key3, []() {
                      return std::unique_ptr<resource_t>(
                              new test_resource_t(100));
                  });

        ASSERT_EQ(resource_cache.size(), 3);
    };

    std::thread t1(func);
    std::thread t2(func);
    t1.join();
    t2.join();
}

TEST(resource, subgraph_resource) {
    ///////////////////////////
    // val1    val2
    //   \     /
    //    \   /
    //     op1
    //      |
    //     val3   val4
    //       \    /
    //        \  /
    //         op2
    //          |
    //         val5
    ///////////////////////////
    using value_t = impl::value_t;
    using dtype = dnnl::memory::data_type;
    using ftag = dnnl::memory::format_tag;
    using engine = dnnl::engine;
    using execution_args_mgr = impl::dnnl_impl::execution_args_mgr;
    using subgraph_resource_t = impl::dnnl_impl::subgraph_resource_t;

    value_t *val1 = (value_t *)1;
    value_t *val2 = (value_t *)2;
    value_t *val3 = (value_t *)3;
    value_t *val4 = (value_t *)4;
    value_t *val5 = (value_t *)5;

    engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::memory mem1({{1, 2, 3, 4}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem2({{2, 3, 4, 5}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem3({{3, 4, 5, 6}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem4({{4, 5, 6, 7}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem5({{5, 6, 7, 8}, dtype::f32, ftag::abcd}, eng, nullptr);

    // construct the execution_args_mgr
    execution_args_mgr exec_args_mgr;
    exec_args_mgr.add_value_mem_map({val1, mem1});
    exec_args_mgr.add_value_mem_map({val2, mem2});
    exec_args_mgr.add_value_mem_map({val3, mem3});
    exec_args_mgr.add_value_mem_map({val4, mem4});
    exec_args_mgr.add_value_mem_map({val5, mem5});

    exec_args_mgr.add_external_input_mem(mem1);
    exec_args_mgr.add_external_input_mem(mem2);
    exec_args_mgr.add_external_input_mem(mem4);

    exec_args_mgr.add_external_output_mem(mem5);

    exec_args_mgr.add_internal_mem(mem3);

    int64_t op1_key = exec_args_mgr.init_args();
    auto &op1_args = exec_args_mgr.get_args(op1_key);
    op1_args.insert({DNNL_ARG_SRC_0, mem1});
    op1_args.insert({DNNL_ARG_SRC_1, mem2});
    op1_args.insert({DNNL_ARG_DST, mem3});

    int64_t op2_key = exec_args_mgr.init_args();
    auto &op2_args = exec_args_mgr.get_args(op2_key);
    op2_args.insert({DNNL_ARG_SRC_0, mem3});
    op2_args.insert({DNNL_ARG_SRC_1, mem4});
    op2_args.insert({DNNL_ARG_DST, mem5});

    exec_args_mgr.add_topo_ordered_key(op1_key);
    exec_args_mgr.add_topo_ordered_key(op2_key);

    // create the subgraph (will deep copy the exec_args_mgr implicitly)
    subgraph_resource_t subgraph_resource(exec_args_mgr);
    const auto &cloned_exec_args_mgr = subgraph_resource.get_exec_args_mgr();

    dnnl::memory cloned_mem1, cloned_mem2, cloned_mem3, cloned_mem4,
            cloned_mem5;
    ASSERT_TRUE(cloned_exec_args_mgr.find_value_mem_map(val1, cloned_mem1));
    ASSERT_TRUE(cloned_exec_args_mgr.find_value_mem_map(val2, cloned_mem2));
    ASSERT_TRUE(cloned_exec_args_mgr.find_value_mem_map(val3, cloned_mem3));
    ASSERT_TRUE(cloned_exec_args_mgr.find_value_mem_map(val4, cloned_mem4));
    ASSERT_TRUE(cloned_exec_args_mgr.find_value_mem_map(val5, cloned_mem5));

    // because of deep copy, the desc should be same but the address should be
    // different
    ASSERT_TRUE(cloned_mem1.get_desc() == mem1.get_desc()
            && cloned_mem1.get() != mem1.get());
    ASSERT_TRUE(cloned_mem2.get_desc() == mem2.get_desc()
            && cloned_mem2.get() != mem2.get());
    ASSERT_TRUE(cloned_mem3.get_desc() == mem3.get_desc()
            && cloned_mem3.get() != mem3.get());
    ASSERT_TRUE(cloned_mem4.get_desc() == mem4.get_desc()
            && cloned_mem4.get() != mem4.get());
    ASSERT_TRUE(cloned_mem5.get_desc() == mem5.get_desc()
            && cloned_mem5.get() != mem5.get());

    // the external mems and internal mems are just alias to the mem object in
    // val-mem map, so both of their desc and address should be same
    auto external_input_mems = cloned_exec_args_mgr.get_external_input_mems();
    ASSERT_TRUE(cloned_mem1.get_desc() == external_input_mems[0].get_desc()
            && cloned_mem1.get() == external_input_mems[0].get());
    ASSERT_TRUE(cloned_mem2.get_desc() == external_input_mems[1].get_desc()
            && cloned_mem2.get() == external_input_mems[1].get());
    ASSERT_TRUE(cloned_mem4.get_desc() == external_input_mems[2].get_desc()
            && cloned_mem4.get() == external_input_mems[2].get());

    auto external_output_mems = cloned_exec_args_mgr.get_external_output_mems();
    ASSERT_TRUE(cloned_mem5.get_desc() == external_output_mems[0].get_desc()
            && cloned_mem5.get() == external_output_mems[0].get());

    auto internal_mems = cloned_exec_args_mgr.get_internal_mems();
    ASSERT_TRUE(cloned_mem3.get_desc() == internal_mems[0].get_desc()
            && cloned_mem3.get() == internal_mems[0].get());

    // the order should be same
    auto topo_ordered_keys = cloned_exec_args_mgr.get_topo_ordered_keys();
    ASSERT_EQ(topo_ordered_keys[0], op1_key);
    ASSERT_EQ(topo_ordered_keys[1], op2_key);

    // the mems in args should also be alias
    auto cloned_op1_args = cloned_exec_args_mgr.get_args(topo_ordered_keys[0]);
    ASSERT_TRUE(
            cloned_mem1.get_desc() == cloned_op1_args[DNNL_ARG_SRC_0].get_desc()
            && cloned_mem1.get() == cloned_op1_args[DNNL_ARG_SRC_0].get());
    ASSERT_TRUE(
            cloned_mem2.get_desc() == cloned_op1_args[DNNL_ARG_SRC_1].get_desc()
            && cloned_mem2.get() == cloned_op1_args[DNNL_ARG_SRC_1].get());
    ASSERT_TRUE(
            cloned_mem3.get_desc() == cloned_op1_args[DNNL_ARG_DST].get_desc()
            && cloned_mem3.get() == cloned_op1_args[DNNL_ARG_DST].get());

    auto cloned_op2_args = cloned_exec_args_mgr.get_args(topo_ordered_keys[1]);
    ASSERT_TRUE(
            cloned_mem3.get_desc() == cloned_op2_args[DNNL_ARG_SRC_0].get_desc()
            && cloned_mem3.get() == cloned_op2_args[DNNL_ARG_SRC_0].get());
    ASSERT_TRUE(
            cloned_mem4.get_desc() == cloned_op2_args[DNNL_ARG_SRC_1].get_desc()
            && cloned_mem4.get() == cloned_op2_args[DNNL_ARG_SRC_1].get());
    ASSERT_TRUE(
            cloned_mem5.get_desc() == cloned_op2_args[DNNL_ARG_DST].get_desc()
            && cloned_mem5.get() == cloned_op2_args[DNNL_ARG_DST].get());
}

TEST(resource, md_hashing) {
    using dtype = dnnl::memory::data_type;
    using ftag = dnnl::memory::format_tag;
    using engine = dnnl::engine;

    dnnl::memory::desc md1({4, 5, 6, 7}, dtype::f32, ftag::abcd);

    // different from md1
    dnnl::memory::desc md2({5, 6, 7, 8}, dtype::f32, ftag::abcd);
    dnnl::memory::desc md4({4, 5, 6, 7}, dtype::u8, ftag::abcd);
    dnnl::memory::desc md5({4, 5, 6, 7}, dtype::f32, ftag::acdb);

    // same as md2
    dnnl::memory::desc md3({5, 6, 7, 8}, dtype::f32, ftag::abcd);

    size_t key1 = dnnl::graph::impl::dnnl_impl::get_md_hash(md1);
    size_t key2 = dnnl::graph::impl::dnnl_impl::get_md_hash(md2);
    size_t key3 = dnnl::graph::impl::dnnl_impl::get_md_hash(md3);
    size_t key4 = dnnl::graph::impl::dnnl_impl::get_md_hash(md4);
    size_t key5 = dnnl::graph::impl::dnnl_impl::get_md_hash(md5);

    ASSERT_NE(key1, key2);
    ASSERT_NE(key1, key4);
    ASSERT_NE(key1, key5);

    ASSERT_EQ(key2, key3);
}

TEST(resource, execution_args_mgr_hashing) {
    using value_t = impl::value_t;
    using dtype = dnnl::memory::data_type;
    using ftag = dnnl::memory::format_tag;
    using engine = dnnl::engine;
    using execution_args_mgr = impl::dnnl_impl::execution_args_mgr;

    value_t *val1 = (value_t *)1;
    value_t *val2 = (value_t *)2;
    value_t *val3 = (value_t *)3;
    value_t *val4 = (value_t *)4;
    value_t *val5 = (value_t *)5;

    engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::memory mem1({{1, 2, 3, 4}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem2({{2, 3, 4, 5}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem3({{3, 4, 5, 6}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem4({{4, 5, 6, 7}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem5({{5, 6, 7, 8}, dtype::f32, ftag::abcd}, eng, nullptr);

    // construct the execution_args_mgr
    execution_args_mgr exec_args_mgr;
    exec_args_mgr.add_value_mem_map({val1, mem1});
    exec_args_mgr.add_value_mem_map({val2, mem2});
    exec_args_mgr.add_value_mem_map({val3, mem3});
    exec_args_mgr.add_value_mem_map({val4, mem4});
    exec_args_mgr.add_value_mem_map({val5, mem5});

    exec_args_mgr.add_external_input_mem(mem1);
    exec_args_mgr.add_external_input_mem(mem2);
    exec_args_mgr.add_external_input_mem(mem4);

    exec_args_mgr.add_external_output_mem(mem5);

    exec_args_mgr.add_internal_mem(mem3);

    execution_args_mgr exec_args_mgr2(exec_args_mgr);

    // exec_args_mgr represent the following structure
    // val1    val2
    //   \     /
    //    \   /
    //     op1
    //      |
    //     val3   val4
    //       \    /
    //        \  /
    //         op2
    //          |
    //         val5
    int64_t op1_key = exec_args_mgr.init_args();
    auto &op1_args = exec_args_mgr.get_args(op1_key);
    op1_args.insert({DNNL_ARG_SRC_0, mem1});
    op1_args.insert({DNNL_ARG_SRC_1, mem2});
    op1_args.insert({DNNL_ARG_DST, mem3});

    int64_t op2_key = exec_args_mgr.init_args();
    auto &op2_args = exec_args_mgr.get_args(op2_key);
    op2_args.insert({DNNL_ARG_SRC_0, mem3});
    op2_args.insert({DNNL_ARG_SRC_1, mem4});
    op2_args.insert({DNNL_ARG_DST, mem5});

    exec_args_mgr.add_topo_ordered_key(op1_key);
    exec_args_mgr.add_topo_ordered_key(op2_key);

    size_t hash_key1 = std::hash<execution_args_mgr>()(exec_args_mgr);
    size_t tmp = std::hash<execution_args_mgr>()(exec_args_mgr);
    ASSERT_EQ(hash_key1, tmp); // check for reproducible

    // exec_args_mgr2 represent the following structure
    // val1
    //  |
    // op1
    //  |
    // val3  val2 val4
    //   \   /   /
    //    \ /  /
    //     op2
    //      |
    //     val5
    op1_key = exec_args_mgr2.init_args();
    auto &op1_args_2 = exec_args_mgr2.get_args(op1_key);
    op1_args_2.insert({DNNL_ARG_SRC_0, mem1});
    op1_args_2.insert({DNNL_ARG_DST, mem3});

    op2_key = exec_args_mgr2.init_args();
    auto &op2_args_2 = exec_args_mgr2.get_args(op2_key);
    op2_args_2.insert({DNNL_ARG_SRC_0, mem3});
    op2_args_2.insert({DNNL_ARG_SRC_1, mem2});
    op2_args_2.insert({DNNL_ARG_SRC_2, mem4});
    op2_args_2.insert({DNNL_ARG_DST, mem5});

    exec_args_mgr2.add_topo_ordered_key(op1_key);
    exec_args_mgr2.add_topo_ordered_key(op2_key);

    size_t hash_key2 = std::hash<execution_args_mgr>()(exec_args_mgr2);

    ASSERT_NE(hash_key1, hash_key2);
}
