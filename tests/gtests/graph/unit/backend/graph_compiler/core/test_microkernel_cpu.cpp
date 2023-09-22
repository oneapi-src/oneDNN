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

#include <numeric>
#include "reference/act_ref.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/sc_data_type.hpp>
#include <ops/templates/utils.hpp>
#include <runtime/context.hpp>
#include <runtime/microkernel/cpu/brgemm_common.hpp>
#include <runtime/microkernel/cpu/brgemm_range_handle.hpp>
#include <runtime/microkernel/cpu/microkernel.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::test_utils;
using namespace dnnl::impl::graph::gc::brgemm;

TEST(GCCore_CPU_microkernel_cpu_cpp, TestBrgemmOnednnF32) {
    REQUIRE_AVX512();
    const int M = 32;
    const int N = 64;
    const int K = 16;
    const int blocks = 10;
    std::vector<float> A(blocks * M * K, 1.f);
    std::vector<float> B(blocks * N * K, 1.f);
    std::vector<float> C(M * N);
    dnnl_brgemm_init_update(A.data(), B.data(), C.data(), blocks, M, N, K, K, N,
            N, M * K, K * N, datatypes::f32.as_etype_int(),
            datatypes::f32.as_etype_int(), /*attrs*/ nullptr,
            /*bd_mask*/ nullptr, /*postop set*/ nullptr,
            /*postop data*/ nullptr, /*c_buf*/ nullptr,
            /*ctx*/ runtime::get_default_stream());
    for (auto i : C) {
        EXPECT_EQ(i, 160.f);
    }
}

TEST(GCCore_CPU_microkernel_cpu_cpp, TestBrgemmOnednnBF16) {
    REQUIRE_BF16();
    const int M = 32;
    const int N = 64;
    const int K = 16;
    const int blocks = 10;
    std::vector<bf16_t> A_bf16(blocks * M * K);
    std::vector<bf16_t> tmpB_bf16(blocks * K * N);
    fill_data<bf16_t>(A_bf16.data(), blocks * M * K);
    fill_data<bf16_t>(tmpB_bf16.data(), blocks * K * N);
    std::vector<float> A, B;
    A.reserve(blocks * M * K);
    B.reserve(blocks * K * N);
    for (auto &it : A_bf16) {
        A.emplace_back(float(it));
    }
    for (auto &it : tmpB_bf16) {
        B.emplace_back(float(it));
    }
    std::vector<bf16_t> B_bf16
            = reorder_low_accuracy_format<bf16_t>(tmpB_bf16, blocks, K, N);
    std::vector<float> C_bf16(blocks * M * N), C_ref(blocks * M * N);
    dnnl_brgemm_init_update(A_bf16.data(), B_bf16.data(), C_bf16.data(), blocks,
            M, N, K, K, N, N, M * K, K * N, datatypes::bf16.as_etype_int(),
            datatypes::bf16.as_etype_int(), /*attrs*/ nullptr,
            /*bd_mask*/ nullptr, /*postop set*/ nullptr,
            /*postop data*/ nullptr, /*c_buf*/ nullptr,
            /*ctx*/ runtime::get_default_stream());
    dnnl_brgemm_init_update(A.data(), B.data(), C_ref.data(), blocks, M, N, K,
            K, N, N, M * K, K * N, datatypes::f32.as_etype_int(),
            datatypes::f32.as_etype_int(), /*attrs*/ nullptr,
            /*bd_mask*/ nullptr, /*postop set*/ nullptr,
            /*postop data*/ nullptr, /*c_buf*/ nullptr,
            /*ctx*/ runtime::get_default_stream());
    for (unsigned i = 0; i < C_bf16.size(); i++) {
        EXPECT_TRUE(std::abs(C_bf16[i] - C_ref[i]) < 1e-4f);
    }
}

TEST(GCCore_CPU_microkernel_cpu_cpp, TestBrgemmOnednnS8S8) {
    REQUIRE_VNNI();
    const int M = 32;
    const int N = 64;
    const int K = 16;
    const int blocks = 10;
    std::vector<int8_t> qA(blocks * M * K);
    std::vector<int8_t> tmpB(blocks * N * K);
    fill_data(qA.data(), blocks * M * K);
    fill_data(tmpB.data(), blocks * N * K);
    std::vector<float> refA(qA.begin(), qA.end());
    std::vector<float> refB(tmpB.begin(), tmpB.end());
    auto ctx = get_default_context();
    if (!ctx->use_amx()) {
        for (auto &it : refA) {
            it += 128;
        }
    }
    std::vector<int8_t> qB
            = reorder_low_accuracy_format<int8_t>(tmpB, blocks, K, N);
    std::vector<float> refC(M * N);
    std::vector<int32_t> qC(M * N);

    dnnl_brgemm_init_update(qA.data(), qB.data(), qC.data(), blocks, M, N, K, K,
            N, N, M * K, K * N, datatypes::s8.as_etype_int(),
            datatypes::s8.as_etype_int(), /*attrs*/ nullptr,
            /*bd_mask*/ nullptr, /*postop set*/ nullptr,
            /*postop data*/ nullptr, /*c_buf*/ nullptr,
            /*ctx*/ runtime::get_default_stream());
    dnnl_brgemm_init_update(refA.data(), refB.data(), refC.data(), blocks, M, N,
            K, K, N, N, M * K, K * N, datatypes::f32.as_etype_int(),
            datatypes::f32.as_etype_int(), /*attrs*/ nullptr,
            /*bd_mask*/ nullptr, /*postop set*/ nullptr,
            /*postop data*/ nullptr, /*c_buf*/ nullptr,
            /*ctx*/ runtime::get_default_stream());
    for (unsigned i = 0; i < qC.size(); i++) {
        EXPECT_TRUE(std::abs(qC[i] - refC[i]) < 1e-4f);
    }
}

TEST(GCCore_CPU_microkernel_cpu_cpp, TestBrgemmOnednnU8S8) {
    REQUIRE_VNNI();
    const int M = 32;
    const int N = 64;
    const int K = 14;
    const int blocks = 10;
    std::vector<uint8_t> qA(blocks * M * K);
    std::vector<int8_t> tmpB(blocks * N * K);
    fill_data(qA.data(), blocks * M * K);
    fill_data(tmpB.data(), blocks * N * K);
    std::vector<float> refA(qA.begin(), qA.end());
    std::vector<float> refB(tmpB.begin(), tmpB.end());
    std::vector<int8_t> qB
            = reorder_low_accuracy_format<int8_t>(tmpB, blocks, K, N);
    std::vector<float> refC(M * N);
    std::vector<int32_t> qC(M * N);
    dnnl_brgemm_init_update(qA.data(), qB.data(), qC.data(), blocks, M, N, K, K,
            N, N, M * K, 16 * N, datatypes::u8.as_etype_int(),
            datatypes::s8.as_etype_int(), /*attrs*/ nullptr,
            /*bd_mask*/ nullptr, /*postop set*/ nullptr,
            /*postop data*/ nullptr, /*c_buf*/ nullptr,
            /*ctx*/ runtime::get_default_stream());
    dnnl_brgemm_init_update(refA.data(), refB.data(), refC.data(), blocks, M, N,
            K, K, N, N, M * K, K * N, datatypes::f32.as_etype_int(),
            datatypes::f32.as_etype_int(), /*attrs*/ nullptr,
            /*bd_mask*/ nullptr, /*postop set*/ nullptr,
            /*postop data*/ nullptr, /*c_buf*/ nullptr,
            /*ctx*/ runtime::get_default_stream());
    for (unsigned i = 0; i < qC.size(); i++) {
        EXPECT_TRUE(std::abs(qC[i] - refC[i]) < 1e-4f);
    }
}

// fix-me(brgemm-fuse): recover the following tests when postop is fixed
#if 0
TEST(GCCore_CPU_microkernel_cpu_cpp, TestBrgemmOnednnPostOpCombined) {
    REQUIRE_VNNI();
    const int M = 32;
    const int N = 64;
    const int K = 14;
    const int padK = utils::divide_and_ceil(K, 4) * 4;
    const int blocks = 10;
    float single_scale = 0.001f;
    std::vector<uint8_t> qA(blocks * M * K);
    std::vector<int8_t> tmpB(blocks * N * K);

    std::vector<float> bias(N);
    std::vector<float> scales(N, single_scale * single_scale);
    std::vector<float> qbias(N);
    std::vector<float> bin_in(1 * N);
    // currently not support zp because of brgemm interface.
    // But it is effective.
    int a_zp = 0;
    int b_zp = 0;
    int c_zp = 0;

    fill_data(qA.data(), blocks * M * K);
    fill_data(tmpB.data(), blocks * N * K);
    fill_data(bias.data(), N);
    fill_data(bin_in.data(), 1 * N);
    test_utils::parallel_nd(N, [&](int n) { qbias[n] = bias[n] / scales[n]; });
    std::vector<int8_t> qB
            = reorder_low_accuracy_format<int8_t>(tmpB, blocks, K, N);
    std::vector<float> qC(M * N);
    std::vector<int32_t> refC_s32(M * N);
    std::vector<float> refA(qA.begin(), qA.end());
    std::vector<float> refB(tmpB.begin(), tmpB.end());
    std::vector<float> refC_f32(M * N);
    std::vector<int32_t> c_buf(M * N);
    std::vector<int32_t> a_compen(1 * N, 0);
    std::vector<int32_t> b_compen(M * 1, 0);

    test_utils::parallel_nd(blocks * M * K, [&](int bs_m_k) {
        refA[bs_m_k] = (refA[bs_m_k] - a_zp) * single_scale;
    });
    test_utils::parallel_nd(blocks * K * N, [&](int bs_k_n) {
        refB[bs_k_n] = (refB[bs_k_n] - b_zp) * single_scale;
    });
    test_utils::parallel_nd(N, [&](int n) {
        for (int i = 0; i < blocks; i++) {
            for (int j = 0; j < K; j++) {
                a_compen[n] += 0 - a_zp * tmpB[i * K * N + j * N + n];
            }
        }
    });
    test_utils::parallel_nd(M, [&](int m) {
        for (int i = 0; i < blocks; i++) {
            for (int j = 0; j < K; j++) {
                b_compen[m] += b_zp * (a_zp - qA[i * K * M + m * K + j]);
            }
        }
    });

    int postops_num = 5; // 8 for zp
    std::shared_ptr<char> dset(
            new char[sizeof(postop_setting_t) * postops_num + sizeof(int64_t)]);
    std::shared_ptr<char> ddata(new char[postops_data_size]);
    postops_setting_t *pset = reinterpret_cast<postops_setting_t *>(dset.get());
    void *pdata = reinterpret_cast<void *>(ddata.get());
    pset->num_ = postops_num;
    pset->ops_[0].scale_op_ = scale_op_t();
    pset->ops_[1].bias_op_ = bias_op_t(sc_data_etype::F32);
    int bin_shape[2] = {1, N};
    pset->ops_[2].bin_op_
            = bin_op_t(alg_kind_t::binary_add, bin_shape, sc_data_etype::F32);
    pset->ops_[3].elt_op_ = elt_op_t(alg_kind_t::eltwise_relu, 1.f, 0.f);
    pset->ops_[4].out_op_ = out_op_t(sc_data_etype::F32);
    // pset->ops_[5].zp_op_ = zp_op_t(alg_kind_t::a_zp);
    // pset->ops_[6].zp_op_ = zp_op_t(alg_kind_t::b_zp);
    // pset->ops_[7].zp_op_ = zp_op_t(alg_kind_t::c_zp);
    void *bin_ptr = bin_in.data();
    void *bin_ptr2 = &bin_ptr;
    dnnl_brgemm_postops_data_init(pdata, qbias.data(), scales.data(), bin_ptr2,
            0, 0, qC.data(), 0, a_compen.data(), b_compen.data(), &c_zp);
    dnnl_brgemm_init_update(qA.data(), qB.data(), qC.data(), blocks, M, N, K, K,
            N, N, M * K, padK * N, datatypes::u8.as_etype_int(),
            datatypes::s8.as_etype_int(), nullptr, nullptr, pset, pdata,
            c_buf.data(), nullptr);
    dnnl_brgemm_init_update(refA.data(), refB.data(), refC_f32.data(), blocks,
            M, N, K, K, N, N, M * K, K * N, datatypes::f32.as_etype_int(),
            datatypes::f32.as_etype_int(), nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr);
    test_utils::parallel_nd(M, N, [&](int m, int n) {
        refC_f32[m * N + n] = refC_f32[m * N + n] + bias[n];
    });
    test_utils::parallel_nd(
            M, N, [&](int m, int n) { refC_f32[m * N + n] += bin_in[n]; });
    ref_relu(refC_f32.data(), refC_f32.data(), M * N);
    test_utils::parallel_nd(
            M, N, [&](int m, int n) { refC_f32[m * N + n] += c_zp; });
    EXPECT_TRUE(!std::all_of(qC.begin(), qC.end(),
            [&](const float &x) { return std::abs(x) < 1e-6f; }));
    for (unsigned i = 0; i < qC.size(); i++) {
        EXPECT_TRUE(std::abs(qC[i] - refC_f32[i]) < 1e-4f);
    }
}
#endif

TEST(GCCore_CPU_microkernel_cpu_cpp, TestBrgemmOnednnAttrs) {
    // bd mask is only supported when
    // use_uker=true
    // bd_mask_level>0
    // use list_addr brgemm
    REQUIRE_AMX();
    const int M = 63;
    const int N = 64;
    const int K = 64;
    const int blocks = 9;
    const int ow = 9;
    const int attr_num = 4;

    std::vector<char> bd_mask(M, 1);
    for (int i = 0; i < M; i++) {
        bd_mask[i] = (i % ow == (ow - 1) || i % ow == (ow - 2)) ? 0 : 1;
    }

    std::vector<uint8_t> A(blocks * M * K, 1);
    std::vector<int8_t> B(blocks * N * K, 1);
    std::vector<int32_t> C(M * N, 0);
    std::vector<char> attr_data(
            sizeof(int64_t) + sizeof(attrs_setting_t::attrs_map_t) * attr_num);
    attrs_setting_t *attrs
            = reinterpret_cast<attrs_setting_t *>(attr_data.data());
    attrs->num_ = attr_num;
    attrs->map_[0] = std::make_pair(attr_key::bd_mask_level, 2);
    attrs->map_[1] = std::make_pair(attr_key::use_uker, true);
    attrs->map_[2] = std::make_pair(attr_key::max_bs, blocks);
    attrs->map_[3] = std::make_pair(attr_key::use_interleave_stores, true);

    const void *A_ptr = A.data();
    const void *B_ptr = B.data();
    dnnl_brgemm_list_update(&A_ptr, &B_ptr, C.data(), blocks, M, N, K, K, N, N,
            M * K, K * N, 1, datatypes::u8.as_etype_int(),
            datatypes::s8.as_etype_int(), /*attrs*/ attrs,
            /*bd_mask*/ bd_mask.data(), /*postop set*/ nullptr,
            /*postop data*/ nullptr, /*c_buf*/ nullptr,
            /*ctx*/ runtime::get_default_stream());

    int mask_start = std::accumulate(bd_mask.begin(), bd_mask.end(), 0);
    const int expected = K * blocks;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i < mask_start) {
                EXPECT_EQ(C[i * N + j], expected);
            } else {
                EXPECT_EQ(C[i * N + j], 0);
            }
        }
    }
}

TEST(GCCore_CPU_microkernel_cpu_cpp, TestBrgemmOnednnRange) {
    REQUIRE_VNNI();
    const int M = 32;
    const int N = 64;
    const int K = 14;
    const int blocks = 10;
    std::vector<uint8_t> qA(blocks * M * K);
    std::vector<int8_t> tmpB(blocks * N * K);
    fill_data(qA.data(), blocks * M * K);
    fill_data(tmpB.data(), blocks * N * K);
    std::vector<float> refA(qA.begin(), qA.end());
    std::vector<float> refB(tmpB.begin(), tmpB.end());
    std::vector<int8_t> qB
            = reorder_low_accuracy_format<int8_t>(tmpB, blocks, K, N);
    std::vector<float> refC(M * N);
    std::vector<int32_t> qC_strd(M * N), qC_list(M * N);

    brg_range_handle_t stride_handle(M + 1, N, K, K, N, N, M * K, 16 * N, 0.f,
            datatypes::u8.as_etype_int(), datatypes::s8.as_etype_int(),
            /*attrs*/ nullptr, /*M_tail_value*/ brg_range_tail_value::dyn_tail,
            /*N_tail_value*/ brg_range_tail_value::no_tail,
            /*K_tail_value*/ brg_range_tail_value::no_tail);
    brg_range_handle_t list_handle(M + 1, N, K, K, N, N, 0.f,
            datatypes::u8.as_etype_int(), datatypes::s8.as_etype_int(),
            /*attrs*/ nullptr, /*M_tail_value*/ M,
            /*N_tail_value*/ brg_range_tail_value::no_tail,
            /*K_tail_value*/ brg_range_tail_value::no_tail);
    dnnl_brgemm_call_range(&stride_handle, M, N, K, qA.data(), qB.data(),
            qC_strd.data(), blocks, runtime::get_default_stream());
    const void *qA_ptr = qA.data(), *qB_ptr = qB.data();
    dnnl_brgemm_list_call_range(&list_handle, M, N, K, &qA_ptr, &qB_ptr,
            qC_list.data(), blocks, M * K, 16 * N, 1,
            datatypes::u8.as_etype_int(), datatypes::s8.as_etype_int(),
            runtime::get_default_stream());
    dnnl_brgemm_init_update(refA.data(), refB.data(), refC.data(), blocks, M, N,
            K, K, N, N, M * K, K * N, datatypes::f32.as_etype_int(),
            datatypes::f32.as_etype_int(), /*attrs*/ nullptr,
            /*bd_mask*/ nullptr, /*postop set*/ nullptr,
            /*postop data*/ nullptr, /*c_buf*/ nullptr,
            /*ctx*/ runtime::get_default_stream());
    for (unsigned i = 0; i < qC_strd.size(); i++) {
        EXPECT_TRUE(std::abs(qC_strd[i] - refC[i]) < 1e-4f);
        EXPECT_TRUE(std::abs(qC_list[i] - refC[i]) < 1e-4f);
    }
}

template <typename dtype>
static bool check_brgemm_init(const int &M, const int &LDC, const int M1) {
    const int buf_size = M * LDC;
    const int N = LDC / 2;
    auto buf = alloc_array<dtype>(buf_size);
    auto ref = buf.copy();
    for (int i = 0; i < M1; ++i) {
        for (int j = 0; j < N; ++j) {
            auto idx = i * LDC + j;
            ref[idx] = (dtype)0;
        }
    }

    auto sc_dtype = sc_data_traits_t<dtype>::type();
    dnnl_brgemm_init(&buf[0], M1, N, LDC, sc_dtype, 0);

    return (memcmp(&buf[0], &ref[0], M * LDC * sizeof(dtype)) == 0);
}

TEST(GCCore_CPU_microkernel_cpu_cpp, TestBRGEMMInit) {
    const int M = 16;
    const int LDC = 64;
    EXPECT_TRUE(check_brgemm_init<int8_t>(M, LDC, 3));
    EXPECT_TRUE(check_brgemm_init<bf16_t>(M, LDC, 5));
    EXPECT_TRUE(check_brgemm_init<float>(M, LDC, 6));
}
