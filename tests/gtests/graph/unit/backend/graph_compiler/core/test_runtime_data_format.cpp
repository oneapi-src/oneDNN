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

#include <chrono>
#include "gtest/gtest.h"
#include <compiler/ir/sc_data_format.hpp>
#include <runtime/dispatch_key.hpp>
#include <runtime/dynamic_dispatch/dyn_dispatch_table.hpp>
#include <runtime/dynamic_dispatch/hash_dispatch_table.hpp>
#include <runtime/dynamic_dispatch/static_dispatch_table.hpp>

using namespace dnnl::impl::graph::gc;

static uint64_t fast_rand(uint64_t &seed) {
    seed = seed * 0x213f23eba219 + 17;
    return seed;
}

//#define DO_BENCH_IN_UT
//#define EXP_DISTRIBUTION

static std::array<uint64_t, 3> generate_format(
        const std::vector<runtime::dyn_dispatch_table_t::format_arg_t>
                &lookuptable,
        int fmt_id, int block_id) {
    auto fmt0 = lookuptable[0].info_[fmt_id % 2].key_;
    auto fmt1 = lookuptable[1].info_[fmt_id / 2 % 2].key_;
    auto fmt2 = lookuptable[2].info_[fmt_id / 4 % 2].key_;

    int impl_type = block_id % 2;
    int M = (block_id / 2 % 4 + 1) * 16;
    int N = (block_id / 8 % 4 + 1) * 16;
    int K = (block_id / 32 % 4 + 1) * 16;
    uint64_t f0 = runtime::dispatch_key(fmt0, M, K, impl_type);
    uint64_t f1 = runtime::dispatch_key(fmt1, K, N, impl_type);
    uint64_t f2 = runtime::dispatch_key(fmt2, M, N, impl_type);
    return {f0, f1, f2};
}

static constexpr uint64_t MK = format_kinds::MK;
static constexpr uint64_t MKmk = format_kinds::MKmk;
static constexpr uint64_t NKkn = format_kinds::NKkn;

static constexpr uint64_t NCHWc = format_kinds::NCHWc;
static constexpr uint64_t ACBD = format_kinds::ACBD;
static constexpr uint64_t ABCD = format_kinds::ABCD;
static constexpr uint64_t ACBDcd = format_kinds::ACBDcd;
static constexpr uint64_t ACBDdc = format_kinds::ACBDdc;

TEST(GCCore_CPU_runtime_data_format, Benchmark) {
    using namespace runtime;
    struct block_compute {
        static uint64_t call(uint64_t *args, uint64_t v) {
            // args[0] for MK blocking and args[1].block_idx2_ for N blocking
            return dispatch_key(args[0]).get_linear_index() * 4
                    + dispatch_key(args[1]).block_idx2_;
        }
    };

    using format_key1 = static_dispatch_keys<MK, MKmk>;
    using format_key2 = static_dispatch_keys<MK, NKkn>;
    using format_key3 = static_dispatch_keys<MK, MKmk>;
    using static_table = static_dispatch_table_t<block_compute, 128,
            format_key1, format_key2, format_key3>;
    static_table sta_table;
    std::vector<dyn_dispatch_table_t::format_arg_t> loopup_table = {
            {{{MKmk}, {MK}}},
            {{{NKkn}, {MK}}},
            {{{MKmk}, {MK}}},
    };

    dyn_dispatch_table_t dyn_table
            = {std::vector<dyn_dispatch_table_t::format_arg_t>(loopup_table),
                    block_compute::call, 128};
    hash_dispatch_table_t hash_table {3, 128 * 8 * 4};

    for (int fmt_id = 0; fmt_id < 8; fmt_id++) {
        for (int block_id = 0; block_id < 128; block_id++) {
            auto fmt = generate_format(loopup_table, fmt_id, block_id);
            void *the_value = (void *)(uint64_t)((block_id << 16) | fmt_id);
            sta_table.set(fmt.data(), 3, the_value);
            dyn_table.set(fmt.data(), 3, the_value);
            hash_table.set(fmt.data(), 3, the_value);
        }
    }

    struct noop_dispatch_table : public dispatch_table_t {
        dispatch_func_t get_dispatch_func() override {
            return [](dispatch_table_t *, uint64_t *keys,
                           uint64_t num_keys) -> void * { return nullptr; };
        }
        void *get(uint64_t *keys, uint64_t num_keys) override {
            return nullptr;
        }
        void set(uint64_t *keys, uint64_t num_keys, void *value) override {}
    };

    noop_dispatch_table noop_table;

    auto run = [](dispatch_table_t *table,
                       const std::vector<dyn_dispatch_table_t::format_arg_t>
                               &lookuptable,
                       bool check) {
        uint64_t rnd = 12232123;
        dispatch_table_t::dispatch_func_t f = table->get_dispatch_func();
        for (int i = 0; i < 1000; i++) {
            auto next_rnd = fast_rand(rnd);
            int block_id = next_rnd % 128;
            int fmt_id = next_rnd / 128 % 8;
            auto fmts = generate_format(lookuptable, fmt_id, block_id);
            f(table, fmts.data(), 3);
        }

        int64_t count = 0;
        uint64_t sum = 0;
        SC_UNUSED(sum);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000000; i++) {
            auto next_rnd = fast_rand(rnd);
            int block_id = next_rnd % 128;
#ifdef EXP_DISTRIBUTION
            int fmt_id = next_rnd / 128 % 128;
            if (fmt_id > 64) {
                fmt_id = 0;
            } else if (fmt_id > 32) {
                fmt_id = 1;
            } else if (fmt_id > 16) {
                fmt_id = 2;
            } else if (fmt_id > 8) {
                fmt_id = 3;
            } else if (fmt_id > 4) {
                fmt_id = 4;
            } else if (fmt_id > 2) {
                fmt_id = 5;
            } else if (fmt_id > 1) {
                fmt_id = 6;
            } else {
                fmt_id = 7;
            }
#else
            int fmt_id = next_rnd / 128 % 8;
#endif
            auto fmts = generate_format(lookuptable, fmt_id, block_id);
            auto ret = (uint64_t)f(table, fmts.data(), 3);

            if (check) {
                if ((int64_t)ret != (int64_t)((block_id << 16) | fmt_id)) {
                    throw std::runtime_error("Check failed");
                }
            }
            sum += ret;
        }
        auto end = std::chrono::high_resolution_clock::now();
        count += std::chrono::duration_cast<std::chrono::microseconds>(
                end - start)
                         .count();
        return count;
    };

    auto base = run(&noop_table, loopup_table, false);
    auto sta = run(&sta_table, loopup_table, true) - base;
    auto hash = run(&hash_table, loopup_table, true) - base;
    auto dyn = run(&dyn_table, loopup_table, true) - base;
    SC_UNUSED(base + sta + hash + dyn);
#ifdef DO_BENCH_IN_UT
    std::cout << "static:" << sta << "\nhash:" << hash << "\ndyn:" << dyn
              << "\n";
#endif
}

TEST(GCCore_CPU_runtime_data_format, TestDataFormat) {
    runtime::dispatch_key a(0);
    a.impl_alg_ = 1;
    a.block_idx1_ = 2;
    a.block_idx2_ = 3;
    a.block1_ = 0;
    a.block2_ = 0;
    a.format_kind_ = 1234;

    EXPECT_FALSE(a.is_blocks_uncompressed());
    EXPECT_EQ(a.get_linear_index(), 1UL * 16 + 3 * 4 + 2);
    EXPECT_EQ(a.format_kind_, 1234UL);
}

TEST(GCCore_CPU_runtime_data_format, TestDataFormatConvert) {
    sc_data_format_t fmt = sc_data_format_t::MKmk(16, 32);
    runtime::dispatch_key rfmt = fmt.to_runtime();

    EXPECT_FALSE(rfmt.is_blocks_uncompressed());
    EXPECT_EQ(rfmt.get_linear_index(), 0UL * 16 + 1UL * 4 + 0);
    EXPECT_EQ(rfmt.format_kind_, (uint32_t)format_kinds::MKmk.storage_);
    EXPECT_EQ(rfmt.get_block1(), 16UL);
    EXPECT_EQ(rfmt.get_block2(), 32UL);

    fmt = sc_data_format_t::NCHWc(16);
    rfmt = fmt.to_runtime();

    EXPECT_FALSE(rfmt.is_blocks_uncompressed());
    EXPECT_EQ(rfmt.get_linear_index(), 0UL * 16 + 0 * 4 + 0);
    EXPECT_EQ(rfmt.format_kind_, (uint32_t)format_kinds::NCHWc.storage_);
    EXPECT_EQ(rfmt.get_block1(), 16UL);
    EXPECT_EQ(rfmt.block2_, 0UL);

    fmt = sc_data_format_t::MKmk(16, 3);
    rfmt = fmt.to_runtime();
    EXPECT_TRUE(rfmt.is_blocks_uncompressed());
    EXPECT_EQ(rfmt.format_kind_, (uint32_t)format_kinds::MKmk.storage_);
    EXPECT_EQ(rfmt.get_block1(), 16UL);
    EXPECT_EQ(rfmt.get_block2(), 3UL);
}

TEST(GCCore_CPU_runtime_data_format, TestDataFormatLinear) {
    sc_data_format_t fmt = sc_data_format_t::MKmk(16, 32);
    runtime::dispatch_key rfmt = fmt.to_runtime();
    auto to_idx = [](runtime::dispatch_key v) {
        return runtime::dispatch_key::linear_converter<format_kinds::MKmk,
                format_kinds::NKkn, format_kinds::ABCD>::call(v);
    };

    EXPECT_EQ(to_idx(rfmt), 2UL);
    EXPECT_EQ(to_idx(sc_data_format_t::NKkn(32, 32).to_runtime()), 1UL);
    EXPECT_EQ(to_idx(runtime::dispatch_key(
                      uint32_t(format_kinds::ABCD), 16, 32, true)),
            0UL);
}

TEST(GCCore_CPU_runtime_data_format, TestDataFormatStaticDispatch) {
    using namespace runtime;
    using format_key1
            = static_dispatch_keys<format_kinds::MKmk, format_kinds::NKkn>;
    using format_key2 = static_dispatch_keys<format_kinds::MKmk,
            format_kinds::NKkn, format_kinds::ABCD>;
    struct block_func {
        static uint64_t call(uint64_t *v, uint64_t num) {
            return dispatch_key(v[0]).get_linear_index() * 32ULL
                    + dispatch_key(v[1]).get_linear_index();
        }
    };
    using the_table = static_dispatch_table_t<block_func, 32 * 32, format_key1,
            format_key2>;
    the_table table;
    uint64_t formats[]
            = {dispatch_key(uint32_t(format_kinds::MKmk), 16, 16, false),
                    dispatch_key(uint32_t(format_kinds::NKkn), 16, 32, true)};
    EXPECT_EQ(the_table::compute_linear_index(formats, 2),
            4UL * 1024 + 1 * 16 + 1 * 4 + 0);
    formats[0] = dispatch_key(uint32_t(format_kinds::MKmk), 48, 16, false);
    EXPECT_EQ(the_table::compute_linear_index(formats, 2),
            4UL * 1024 + 2 * 32ULL + 1 * 16 + 1 * 4 + 0);
}

TEST(GCCore_CPU_runtime_data_format, TestDataFormatDynDispatch) {
    using namespace runtime;
    dyn_dispatch_table_t table(
            {
                    {{{MKmk}, {NKkn}, {NCHWc}}},
                    {{{ACBD}, {ABCD}, {ACBDcd}, {ACBDdc}}},
                    {{{MKmk}}},
            },
            [](uint64_t *v, uint64_t num) -> uint64_t {
                return dispatch_key(v[0]).get_linear_index() * 32ULL * 32
                        + dispatch_key(v[1]).get_linear_index() * 32ULL
                        + dispatch_key(v[2]).get_linear_index();
            },
            32 * 32 * 32);
    uint64_t formats[]
            = {dispatch_key(uint32_t(format_kinds::NKkn), 32, 16, false),
                    dispatch_key(uint32_t(format_kinds::ABCD), 32, 0, true),
                    dispatch_key(uint32_t(format_kinds::MKmk), 16, 16, true)};
    EXPECT_EQ(table.compute_linear_index(formats, 3),
            (1UL + 1 * 3 + 0) * 32ULL * 32 * 32 + 1UL * 32 * 32
                    + (1 * 16 + 1) * 32ULL + 1 * 16);
}

TEST(GCCore_CPU_runtime_data_format, TestDataFormatHashDispatch) {
    using namespace runtime;
    using namespace format_kinds;
    hash_dispatch_table_t table {3, 256};
    std::array<uint64_t, 3> keys;
    keys = {dispatch_key(uint32_t(format_kinds::NKkn), 32, 16, false),
            dispatch_key(uint32_t(format_kinds::ABCD), 32, 0, true),
            dispatch_key(uint32_t(format_kinds::MKmk), 16, 16, true)};
    table.set(keys.data(), 3, (void *)123);

    uint64_t seed = 11234;
    for (uint64_t i = 0; i < 126; i++) {
        keys = {fast_rand(seed), fast_rand(seed), fast_rand(seed)};
        table.set(keys.data(), 3, (void *)i);
    }

    seed = 11234;
    for (uint64_t i = 0; i < 126; i++) {
        keys = {fast_rand(seed), fast_rand(seed), fast_rand(seed)};
        EXPECT_EQ(table.get(keys.data(), 3), (void *)i);
    }
    keys = {dispatch_key(uint32_t(format_kinds::NKkn), 32, 16, false),
            dispatch_key(uint32_t(format_kinds::ABCD), 32, 0, true),
            dispatch_key(uint32_t(format_kinds::MKmk), 16, 16, true)};
    EXPECT_EQ(table.get(keys.data(), 3), (void *)123);
}
