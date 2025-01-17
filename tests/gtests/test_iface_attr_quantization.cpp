/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include "src/cpu/platform.hpp"

namespace dnnl {

// short names for brevity
using data_type = memory::data_type;
using tag = memory::format_tag;

class attr_quantization_test_t : public ::testing::Test {
protected:
    engine eng = get_test_engine();
    void SetUp() override {}

    static primitive_attr gen_attr_with_scales(bool with_wei = true) {
        primitive_attr attr;
        attr.set_scales_mask(DNNL_ARG_SRC, 0);
        if (with_wei) attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
        attr.set_scales_mask(DNNL_ARG_DST, 0);
        return attr;
    }

    static primitive_attr gen_attr_with_scales(int arg, int mask = 0,
            data_type dt = data_type::f32, const memory::dims &groups = {}) {
        primitive_attr attr;
        attr.set_scales(arg, mask, groups, dt);
        return attr;
    }

    static primitive_attr gen_attr_with_zp(int arg, int mask = 0,
            data_type dt = data_type::s32, const memory::dims &groups = {}) {
        primitive_attr attr;
        attr.set_zero_points(arg, mask, groups, dt);
        return attr;
    }

    template <typename F>
    static void check_status(const F &f, dnnl_status_t status,
            const char *filename, int64_t line_num) {
        catch_expected_failures(
                f, status != dnnl_success, status, false, filename, line_num);
    }
};
#define CHECK_STATUs(status, ...) \
    check_status([&]() { __VA_ARGS__; }, status, __FILE__, __LINE__)
#define CHECK_STATUS(status, ...) CHECK_STATUs(status, __VA_ARGS__)

#define CHECK_OK(...) CHECK_STATUS(dnnl_success, __VA_ARGS__)
#define CHECK_INVALID(...) CHECK_STATUS(dnnl_invalid_arguments, __VA_ARGS__)
#define CHECK_UNIMPL(...) CHECK_STATUS(dnnl_unimplemented, __VA_ARGS__)

// TODO: replace primitive descriptor creation with iterator fetching
//       to test all possible implementations

TEST_F(attr_quantization_test_t, TestBNorm) {
    for (auto dt : {data_type::f32, data_type::s8}) {
        // no s8 -> s8 batch norm on GPU yet
        if (get_test_engine_kind() == engine::kind::gpu && dt == data_type::s8)
            continue;

        memory::desc md {{1, 16, 3, 3}, dt, tag::abcd};
        normalization_flags flags = normalization_flags::use_global_stats;
        CHECK_OK(batch_normalization_forward::primitive_desc(
                eng, prop_kind::forward_inference, md, md, 0.1f, flags));
        CHECK_UNIMPL(batch_normalization_forward::primitive_desc(eng,
                prop_kind::forward_inference, md, md, 0.1f, flags,
                gen_attr_with_scales(
                        /* with_wei = false, qunatization is not supported */)));

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            CHECK_UNIMPL(batch_normalization_forward::primitive_desc(eng,
                    prop_kind::forward_inference, md, md, 0.1f, flags,
                    gen_attr_with_zp(arg)));
        }
        for (auto arg : {DNNL_ARG_BIAS, DNNL_ARG_MEAN, DNNL_ARG_VARIANCE}) {
            CHECK_INVALID(batch_normalization_forward::primitive_desc(eng,
                    prop_kind::forward_inference, md, md, 0.1f, flags,
                    gen_attr_with_zp(arg)));
        }
    }
}

TEST_F(attr_quantization_test_t, TestBinary) {
    memory::desc md {{1, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_OK(binary::primitive_desc(eng, algorithm::binary_add, md, md, md));

    for (auto arg : {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1, DNNL_ARG_DST}) {
        if (arg == DNNL_ARG_DST)
            CHECK_UNIMPL(binary::primitive_desc(eng, algorithm::binary_add, md,
                    md, md, gen_attr_with_scales(arg)));
        else
            CHECK_OK(binary::primitive_desc(eng, algorithm::binary_add, md, md,
                    md, gen_attr_with_scales(arg)));
    }

    for (auto arg : {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1, DNNL_ARG_DST}) {
        if (arg == DNNL_ARG_SRC_1)
            CHECK_INVALID(binary::primitive_desc(eng, algorithm::binary_add, md,
                    md, md, gen_attr_with_zp(arg)));
        else
            CHECK_UNIMPL(binary::primitive_desc(eng, algorithm::binary_add, md,
                    md, md, gen_attr_with_zp(arg)));
    }
}

TEST_F(attr_quantization_test_t, TestConcat) {
    // Operator is not supported in the AMD backend
    SKIP_IF_HIP(true, "Concat operator is not supported in HIP");
    memory::desc md {{1, 16, 3, 3}, data_type::s8, tag::abcd};
    CHECK_OK(concat::primitive_desc(eng, 1, {md, md}));

    for (auto arg : {DNNL_ARG_MULTIPLE_SRC, DNNL_ARG_MULTIPLE_SRC + 1}) {
        CHECK_OK(concat::primitive_desc(
                eng, 1, {md, md}, gen_attr_with_scales(arg)));
        CHECK_INVALID(concat::primitive_desc(
                eng, 1, {md, md}, gen_attr_with_zp(arg)));
    }
}

TEST_F(attr_quantization_test_t, TestConv) {
    // Datatype u8 is not supported in the Nvidia backend
    SKIP_IF_CUDA(true, "Unsupported datatype for CUDA");
    // src, wei and dst needs to have same datatype. Just s8 it is supported.
    SKIP_IF_HIP(true, "Unsupported datatype for HIP");
    memory::desc src_md {{1, 16, 7, 7}, data_type::u8, tag::any};
    memory::desc wei_md {{32, 16, 3, 3}, data_type::s8, tag::any};
    memory::desc dst_md {{1, 32, 7, 7}, data_type::s32, tag::any};

    CHECK_OK(convolution_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}));
    CHECK_OK(convolution_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}, gen_attr_with_scales()));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        if (src_md.get_data_type() == data_type::s8
                || src_md.get_data_type() == data_type::u8) {
            if (arg == DNNL_ARG_SRC || arg == DNNL_ARG_DST) {
                CHECK_OK(convolution_forward::primitive_desc(eng,
                        prop_kind::forward, algorithm::convolution_direct,
                        src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                        gen_attr_with_zp(arg)));
                CHECK_OK(convolution_forward::primitive_desc(eng,
                        prop_kind::forward, algorithm::convolution_direct,
                        src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                        gen_attr_with_scales(arg, 0)));
            } else {
                if (eng.get_kind() == dnnl::engine::kind::cpu)
                    CHECK_UNIMPL(convolution_forward::primitive_desc(eng,
                            prop_kind::forward, algorithm::convolution_direct,
                            src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                            gen_attr_with_zp(arg)));
                else
                    CHECK_OK(convolution_forward::primitive_desc(eng,
                            prop_kind::forward, algorithm::convolution_direct,
                            src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                            gen_attr_with_zp(arg)));
                CHECK_OK(convolution_forward::primitive_desc(eng,
                        prop_kind::forward, algorithm::convolution_direct,
                        src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                        gen_attr_with_scales(arg, 0)));
                CHECK_OK(convolution_forward::primitive_desc(eng,
                        prop_kind::forward, algorithm::convolution_direct,
                        src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                        gen_attr_with_scales(arg, 1 << 0)));
            }
        } else {
            CHECK_UNIMPL(convolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::convolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg)));
            CHECK_UNIMPL(convolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::convolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_zp(arg)));
        }
    }
}

TEST_F(attr_quantization_test_t, TestConvGroup) {
    // Datatype u8 is not supported in the Nvidia backend
    SKIP_IF_CUDA(true, "Unsupported datatype for CUDA");
    // src, wei and dst needs to have same datatype. Just s8 it is supported.
    SKIP_IF_HIP(true, "Unsupported datatype for HIP");
    const int g = 2;
    memory::desc src_md {{1, 16, 7, 7}, data_type::u8, tag::any};
    memory::desc wei_md {{g, 32 / g, 16 / g, 3, 3}, data_type::s8, tag::any};
    memory::desc dst_md {{1, 32, 7, 7}, data_type::s32, tag::any};

    CHECK_OK(convolution_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}));
    CHECK_OK(convolution_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}, gen_attr_with_scales()));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        if (src_md.get_data_type() == data_type::s8
                || src_md.get_data_type() == data_type::u8) {
            if (arg == DNNL_ARG_SRC || arg == DNNL_ARG_DST) {
                CHECK_OK(convolution_forward::primitive_desc(eng,
                        prop_kind::forward, algorithm::convolution_direct,
                        src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                        gen_attr_with_zp(arg)));
                CHECK_OK(convolution_forward::primitive_desc(eng,
                        prop_kind::forward, algorithm::convolution_direct,
                        src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                        gen_attr_with_scales(arg, 0)));
            } else {
                if (eng.get_kind() == dnnl::engine::kind::cpu)
                    CHECK_UNIMPL(convolution_forward::primitive_desc(eng,
                            prop_kind::forward, algorithm::convolution_direct,
                            src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                            gen_attr_with_zp(arg)));
                else
                    CHECK_OK(convolution_forward::primitive_desc(eng,
                            prop_kind::forward, algorithm::convolution_direct,
                            src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                            gen_attr_with_zp(arg)));
                CHECK_OK(convolution_forward::primitive_desc(eng,
                        prop_kind::forward, algorithm::convolution_direct,
                        src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                        gen_attr_with_scales(arg, 0)));
                CHECK_OK(convolution_forward::primitive_desc(eng,
                        prop_kind::forward, algorithm::convolution_direct,
                        src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                        gen_attr_with_scales(arg, (1 << 1) + (1 << 0))));
                CHECK_UNIMPL(convolution_forward::primitive_desc(eng,
                        prop_kind::forward, algorithm::convolution_direct,
                        src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                        gen_attr_with_scales(arg, 1 << 1)));
            }
        } else {
            CHECK_UNIMPL(convolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::convolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg)));
            CHECK_UNIMPL(convolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::convolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_zp(arg)));
        }
    }
}

TEST_F(attr_quantization_test_t, TestDeconv) {
    // src, wei and dst needs to have same datatype. Just s8 it is supported.
    SKIP_IF_HIP(true, "Unsupported datatype for HIP");
    memory::desc src_md {{1, 16, 7, 7}, data_type::u8, tag::any};
    memory::desc wei_md {{32, 16, 3, 3}, data_type::s8, tag::any};
    memory::desc dst_md {{1, 32, 7, 7}, data_type::s8, tag::any};
    CHECK_OK(deconvolution_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::deconvolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}, gen_attr_with_scales()));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        if (arg == DNNL_ARG_SRC || arg == DNNL_ARG_DST) {
            // scales: common mask
            CHECK_OK(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg)));
            // zpoints: common mask
            CHECK_OK(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_zp(arg)));
        } else {
            // scales: common mask
            CHECK_OK(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg, 0)));
            // scales: per_oc mask
            CHECK_OK(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg, 1 << 0)));
            // scales: unsupported mask
            CHECK_UNIMPL(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg, 1 << 1)));
            // zpoints: common mask
            CHECK_UNIMPL(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_zp(arg)));
        }
    }
}

TEST_F(attr_quantization_test_t, TestDeconvGroup) {
    // src, wei and dst needs to have same datatype. Just s8 it is supported.
    SKIP_IF_HIP(true, "Unsupported datatype for HIP");
    const int g = 2;
    memory::desc src_md {{1, 16, 7, 7}, data_type::u8, tag::any};
    memory::desc wei_md {{g, 32 / g, 16 / g, 3, 3}, data_type::s8, tag::any};
    memory::desc dst_md {{1, 32, 7, 7}, data_type::s8, tag::any};
    CHECK_OK(deconvolution_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::deconvolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}, gen_attr_with_scales()));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        if (arg == DNNL_ARG_SRC || arg == DNNL_ARG_DST) {
            // scales: common mask
            CHECK_OK(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg)));
            // zpoints: common mask
            CHECK_OK(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_zp(arg)));
        } else {
            // scales: common mask
            CHECK_OK(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg, 0)));
            // scales: per_oc mask
            CHECK_OK(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg, (1 << 1) + (1 << 0))));
            // scales: unsupported mask
            CHECK_UNIMPL(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_scales(arg, 1 << 1)));
            // zpoints: common mask
            CHECK_UNIMPL(deconvolution_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::deconvolution_direct, src_md,
                    wei_md, dst_md, {1, 1}, {1, 1}, {1, 1},
                    gen_attr_with_zp(arg)));
        }
    }
}

TEST_F(attr_quantization_test_t, TestEltwise) {
    for (auto dt : {data_type::f32, data_type::s8}) {
        // s8 is not supported in HIP
        if (is_amd_gpu(get_test_engine()) && dt == data_type::s8) continue;

        memory::desc md {{1, 16, 3, 3}, dt, tag::abcd};

        CHECK_OK(eltwise_forward::primitive_desc(
                eng, prop_kind::forward, algorithm::eltwise_relu, md, md, 0.f));

        CHECK_UNIMPL(eltwise_forward::primitive_desc(eng, prop_kind::forward,
                algorithm::eltwise_relu, md, md, 0.f,
                gen_attr_with_scales(
                        /* with_wei = false, quantization is not supported */)));

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
            CHECK_UNIMPL(eltwise_forward::primitive_desc(eng,
                    prop_kind::forward, algorithm::eltwise_relu, md, md, 0.f,
                    gen_attr_with_zp(arg)));
        }
    }
}

TEST_F(attr_quantization_test_t, TestInnerProduct) {
    // Datatype u8 is not supported in the Nvidia backend
    SKIP_IF_CUDA(true, "Unsupported datatype for CUDA");
    // src, wei needs to be s8 and dst be s32.
    SKIP_IF_HIP(true, "Unsupported datatype for HIP");
    SKIP_IF_GENERIC(true, "InnerProduct is not supported for Generic");
    memory::desc src_md {{1, 16, 7, 7}, data_type::u8, tag::any};
    memory::desc wei_md {{32, 16, 7, 7}, data_type::s8, tag::any};
    memory::desc dst_md {{1, 32}, data_type::s32, tag::any};
    CHECK_OK(inner_product_forward::primitive_desc(
            eng, prop_kind::forward, src_md, wei_md, dst_md));
    CHECK_OK(inner_product_forward::primitive_desc(eng, prop_kind::forward,
            src_md, wei_md, dst_md, gen_attr_with_scales()));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        CHECK_UNIMPL(
                inner_product_forward::primitive_desc(eng, prop_kind::forward,
                        src_md, wei_md, dst_md, gen_attr_with_zp(arg)));
    }
}

TEST_F(attr_quantization_test_t, TestLNorm) {
    SKIP_IF_CUDA(true, "Layer normalization primitive not supported for CUDA");
    SKIP_IF_HIP(true, "Layer normalization primitive not supported for HIP");

    memory::desc md {{1, 16, 16}, data_type::s8, tag::abc};
    memory::desc stat_md {{1, 16}, data_type::f32, tag::ab};
    normalization_flags flags = normalization_flags::use_global_stats;

    CHECK_OK(layer_normalization_forward::primitive_desc(
            eng, prop_kind::forward_inference, md, md, stat_md, 0.1f, flags));
    CHECK_UNIMPL(layer_normalization_forward::primitive_desc(eng,
            prop_kind::forward_inference, md, md, stat_md, 0.1f, flags,
            gen_attr_with_scales(/* WEIGHTS are not supported */)));
    CHECK_OK(layer_normalization_forward::primitive_desc(eng,
            prop_kind::forward_inference, md, md, stat_md, 0.1f, flags,
            gen_attr_with_scales(/* with_wei = */ false)));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        CHECK_UNIMPL(layer_normalization_forward::primitive_desc(eng,
                prop_kind::forward_inference, md, md, stat_md, 0.1f, flags,
                gen_attr_with_zp(arg)));
    }
    for (auto arg : {DNNL_ARG_MEAN, DNNL_ARG_VARIANCE, DNNL_ARG_BIAS}) {
        CHECK_INVALID(layer_normalization_forward::primitive_desc(eng,
                prop_kind::forward_inference, md, md, stat_md, 0.1f, flags,
                gen_attr_with_zp(arg)));
    }
}

TEST_F(attr_quantization_test_t, TestLRN) {
    for (auto dt : {data_type::f32}) {
        memory::desc md {{1, 16, 3, 3}, dt, tag::abcd};
        CHECK_OK(lrn_forward::primitive_desc(eng, prop_kind::forward_inference,
                algorithm::lrn_across_channels, md, md, 5, 1.f, 0.75f, 1.0f));
        CHECK_UNIMPL(lrn_forward::primitive_desc(eng,
                prop_kind::forward_inference, algorithm::lrn_across_channels,
                md, md, 5, 1.f, 0.75f, 1.0f,
                gen_attr_with_scales(
                        /* with_wei = false, quantization is not supported */)));

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
            CHECK_UNIMPL(lrn_forward::primitive_desc(eng,
                    prop_kind::forward_inference,
                    algorithm::lrn_across_channels, md, md, 5, 1.f, 0.75f, 1.0f,
                    gen_attr_with_zp(arg)));
        }
    }
}

TEST_F(attr_quantization_test_t, TestMatmul) {
    // cuDNN doesn't support zero points
    SKIP_IF_CUDA(true, "Test not supported on cuda");

    for (auto a_dt : {data_type::f32, data_type::u8}) {
        const data_type b_dt
                = a_dt == data_type::f32 ? data_type::f32 : data_type::s8;

        memory::desc a_md {{10, 64}, a_dt, tag::ab};
        memory::desc b_md {{64, 20}, b_dt, tag::ba};
        memory::desc c_md {{10, 20}, data_type::f32, tag::ab};

        CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md));
        CHECK_OK(matmul::primitive_desc(
                eng, a_md, b_md, c_md, gen_attr_with_scales()));

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (a_dt != data_type::u8 && a_dt != data_type::s8) {
                CHECK_UNIMPL(matmul::primitive_desc(
                        eng, a_md, b_md, c_md, gen_attr_with_zp(arg)));
            } else {
                // zpoints: common mask
                CHECK_OK(matmul::primitive_desc(
                        eng, a_md, b_md, c_md, gen_attr_with_zp(arg)));
                // zpoints: per_oc mask
                CHECK_OK(matmul::primitive_desc(
                        eng, a_md, b_md, c_md, gen_attr_with_zp(arg, 1 << 1)));
                // zpoints: per_ocic mask
                if (arg == DNNL_ARG_WEIGHTS) {
                    CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_zp(arg, (1 << 1) + (1 << 0))));
                    CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_zp(
                                    arg, (1 << 1) + (1 << 0), b_dt, {32, 1})));
                } else if (arg == DNNL_ARG_SRC) {
                    CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_zp(arg, (1 << 1) + (1 << 0))));
                    CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_zp(
                                    arg, (1 << 1) + (1 << 0), b_dt, {1, 32})));
                } else {
                    CHECK_UNIMPL(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_zp(arg, (1 << 1) + (1 << 0))));
                }
            }
            // scales: common mask
            CHECK_OK(matmul::primitive_desc(
                    eng, a_md, b_md, c_md, gen_attr_with_scales(arg)));
            // scales: per_oc mask
            if (arg == DNNL_ARG_WEIGHTS) {
                CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                        gen_attr_with_scales(arg, 1 << 1)));
                if (b_dt == data_type::s8) {
                    CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(arg, (1 << 1) + (1 << 0))));
                    // Groups non divisible by 32 are not supported.
                    CHECK_UNIMPL(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(arg, (1 << 1) + (1 << 0),
                                    data_type::f32, {3, 1})));
                    CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(arg, (1 << 1) + (1 << 0),
                                    data_type::f32, {32, 1})));
                } else {
                    CHECK_UNIMPL(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(arg, (1 << 1) + (1 << 0),
                                    data_type::f32, {32, 1})));
                }
            } else if (arg == DNNL_ARG_SRC) {
                // Somehow GPU doeshave this support.
                const bool is_cpu = get_test_engine_kind() == engine::kind::cpu;
                if (is_cpu) {
                    CHECK_UNIMPL(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(arg, 1 << 1)));
                }
                if (a_dt == data_type::u8) {
                    CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(
                                    arg, 1 << 1, data_type::f32, {1, 32})));
                    // Groups non divisible by 32 are not supported.
                    CHECK_UNIMPL(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(arg, (1 << 1) + (1 << 0),
                                    data_type::f32, {1, 3})));
                    CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(arg, (1 << 1) + (1 << 0),
                                    data_type::f32, {1, 32})));
                } else {
                    CHECK_UNIMPL(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(
                                    arg, 1 << 1, data_type::f32, {1, 32})));
                    CHECK_UNIMPL(matmul::primitive_desc(eng, a_md, b_md, c_md,
                            gen_attr_with_scales(arg, (1 << 1) + (1 << 0),
                                    data_type::f32, {1, 32})));
                }
            }
        }
    }
}

CPU_TEST_F(attr_quantization_test_t, TestMatmulBatch) {
    for (auto a_dt : {data_type::f32, data_type::u8}) {
        const data_type b_dt
                = a_dt == data_type::f32 ? data_type::f32 : data_type::s8;

        memory::desc a_md {{2, 5, 10, 64}, a_dt, tag::abcd};
        memory::desc b_md {{2, 5, 64, 20}, b_dt, tag::abdc};
        memory::desc c_md {{2, 5, 10, 20}, data_type::f32, tag::abcd};
        const auto ndims = a_md.get_ndims();

        CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md));
        CHECK_OK(matmul::primitive_desc(
                eng, a_md, b_md, c_md, gen_attr_with_scales()));

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (a_dt != data_type::u8 && a_dt != data_type::s8) {
                CHECK_UNIMPL(matmul::primitive_desc(
                        eng, a_md, b_md, c_md, gen_attr_with_zp(arg)));
            } else {
                // zpoints: common mask
                CHECK_OK(matmul::primitive_desc(
                        eng, a_md, b_md, c_md, gen_attr_with_zp(arg)));
            }
            // scales: common mask
            CHECK_OK(matmul::primitive_desc(
                    eng, a_md, b_md, c_md, gen_attr_with_scales(arg)));
            // scales: per_oc mask
            const auto per_oc_mask = 1 << (ndims - 1);
            if (arg == DNNL_ARG_WEIGHTS) {
                CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                        gen_attr_with_scales(arg, per_oc_mask)));
            }

            if (a_dt != data_type::u8 && a_dt != data_type::s8) continue;
            // scales: per_tensor mask for int8 type only.
            const auto per_tensor_mask = (1 << ndims) - 1;
            const auto per_ocic_mask = (1 << (ndims - 1)) + (1 << (ndims - 2));
            if (arg == DNNL_ARG_WEIGHTS) {
                CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                        gen_attr_with_scales(arg, per_tensor_mask,
                                data_type::f32, {32, 1})));
            } else if (arg == DNNL_ARG_SRC) {
                CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md,
                        gen_attr_with_scales(
                                arg, per_ocic_mask, data_type::f32, {1, 32})));
            } else {
                CHECK_UNIMPL(matmul::primitive_desc(eng, a_md, b_md, c_md,
                        gen_attr_with_scales(arg, per_tensor_mask)));
            }
        }
    }
}

TEST_F(attr_quantization_test_t, TestPool) {
    // Datatype s8 is not supported in the Nvidia backend
    SKIP_IF_HIP(true, "Unsupported datatype for AMD");
    memory::desc src_md {{1, 16, 8, 8}, data_type::s8, tag::abcd};
    memory::desc dst_md {{1, 16, 4, 4}, data_type::s8, tag::abcd};

    CHECK_OK(pooling_forward::primitive_desc(eng, prop_kind::forward_inference,
            algorithm::pooling_max, src_md, dst_md, {2, 2}, {2, 2}, {0, 0},
            {0, 0}, {0, 0}));
    CHECK_UNIMPL(pooling_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::pooling_max, src_md,
            dst_md, {2, 2}, {2, 2}, {0, 0}, {0, 0}, {0, 0},
            gen_attr_with_scales(
                    /* with_wei = false, quantization is not supported */)));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_UNIMPL(pooling_forward::primitive_desc(eng,
                prop_kind::forward_inference, algorithm::pooling_max, src_md,
                dst_md, {2, 2}, {2, 2}, {0, 0}, {0, 0}, {0, 0},
                gen_attr_with_zp(arg)));
    }
}

TEST_F(attr_quantization_test_t, TestPReLU) {
    SKIP_IF_CUDA(true, "Unsupported primitive not supported for CUDA");
    SKIP_IF_HIP(true, "Unsupported primitive not supported for HIP");
    memory::desc data_md {{1, 16, 3, 3}, data_type::f32, tag::abcd};
    memory::desc weights_md {{1, 16, 3, 3}, data_type::f32, tag::abcd};

    CHECK_OK(prelu_forward::primitive_desc(
            eng, prop_kind::forward, data_md, weights_md, data_md));

    CHECK_UNIMPL(prelu_forward::primitive_desc(eng, prop_kind::forward, data_md,
            weights_md, data_md,
            gen_attr_with_scales(
                    /* with_wei = false, quantization is not supported */)));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_UNIMPL(prelu_forward::primitive_desc(eng, prop_kind::forward,
                data_md, weights_md, data_md, gen_attr_with_zp(arg)));
    }
}

CPU_TEST_F(attr_quantization_test_t, TestReorder) {
    memory::desc src_md {{1, 16, 8, 8}, data_type::s8, tag::abcd};
    memory::desc dst_md {{1, 16, 8, 8}, data_type::s8, tag::acdb};
    CHECK_OK(reorder::primitive_desc(eng, src_md, eng, dst_md));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_UNIMPL(reorder::primitive_desc(eng, src_md, eng, dst_md,
                gen_attr_with_scales(/* WEIGHTS are not supported */)));
        CHECK_OK(reorder::primitive_desc(eng, src_md, eng, dst_md,
                gen_attr_with_scales(/* with_wei = */ false)));
        CHECK_OK(reorder::primitive_desc(
                eng, src_md, eng, dst_md, gen_attr_with_zp(arg)));
    }
}

TEST_F(attr_quantization_test_t, TestRNN) {
    SKIP_IF_CUDA(true, "RNN primitive not supported for CUDA");
    SKIP_IF_HIP(true, "RNN primitive not supported for HIP");
    SKIP_IF_GENERIC(true, "RNN primitive not supported for Generic");
    // Int8 RNN relies on packed API solely which is available only for X64.
#if !DNNL_X64
    return;
#endif
    // XXX: Threadpool doesn't work correctly with packed API which is the only
    // working mechanism for int8 computations. Disable it for now.
    SKIP_IF(DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL,
            "Threadpool does not have working packed API");

    memory::dim n = 1, t = 1, l = 10, c = 8, g = 4, d = 1;
    memory::desc src_layer_md {{t, n, c}, data_type::u8, tag::tnc};
    memory::desc src_iter_md {{l, d, n, c}, data_type::u8, tag::ldnc};
    memory::desc src_iter_c_md {{l, d, n, c}, data_type::f32, tag::ldnc};
    memory::desc wei_layer_md {{l, d, c, g, c}, data_type::s8, tag::any};
    memory::desc wei_iter_md {{l, d, c, g, c}, data_type::s8, tag::any};
    memory::desc bia_md {{l, d, g, c}, data_type::f32, tag::ldgo};
    memory::desc dst_layer_md {{t, n, c}, data_type::u8, tag::tnc};
    memory::desc dst_iter_md {{l, d, n, c}, data_type::u8, tag::ldnc};
    memory::desc dst_iter_c_md {{l, d, n, c}, data_type::f32, tag::ldnc};

    for_(auto is_runtime_data_scale : {true, false})
    for_(auto is_runtime_data_shift : {true, false})
    for_(auto is_runtime_weights_scale : {true, false})
    {
        primitive_attr attr;
        attr.set_rnn_data_qparams(
                is_runtime_data_scale ? DNNL_RUNTIME_F32_VAL : 2.f,
                is_runtime_data_shift ? DNNL_RUNTIME_F32_VAL : 2.f);
        attr.set_rnn_weights_qparams(
                0, {is_runtime_weights_scale ? DNNL_RUNTIME_F32_VAL : 2.f});
        bool rt = is_runtime_data_scale || is_runtime_data_shift
                || is_runtime_weights_scale;
        CHECK_STATUS(rt ? dnnl_unimplemented : dnnl_success,
                lstm_forward::primitive_desc(eng, prop_kind::forward_inference,
                        rnn_direction::unidirectional_left2right, src_layer_md,
                        src_iter_md, src_iter_c_md, wei_layer_md, wei_iter_md,
                        bia_md, dst_layer_md, dst_iter_md, dst_iter_c_md,
                        attr));
    }

    for (auto arg :
            {DNNL_ARG_SRC_LAYER, DNNL_ARG_WEIGHTS_LAYER, DNNL_ARG_DST_LAYER}) {
        CHECK_UNIMPL(
                lstm_forward::primitive_desc(eng, prop_kind::forward_inference,
                        rnn_direction::unidirectional_left2right, src_layer_md,
                        src_iter_md, src_iter_c_md, wei_layer_md, wei_iter_md,
                        bia_md, dst_layer_md, dst_iter_md, dst_iter_c_md,
                        gen_attr_with_zp(arg)));
    }
}

TEST_F(attr_quantization_test_t, TestShuffle) {
    SKIP_IF_CUDA(true, "Shuffle primitive not supported for CUDA");
    SKIP_IF_HIP(true, "Shuffle primitive not supported for HIP");
    memory::desc md {{1, 16, 3, 3}, data_type::f32, tag::abcd};

    CHECK_OK(shuffle_forward::primitive_desc pd(
            eng, prop_kind::forward, md, md, 1, 4));
    CHECK_UNIMPL(shuffle_forward::primitive_desc pd(eng, prop_kind::forward, md,
            md, 1, 4,
            gen_attr_with_scales(
                    /* with_wei = false, quantization is not supported */)));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_UNIMPL(shuffle_forward::primitive_desc pd(
                eng, prop_kind::forward, md, md, 1, 4, gen_attr_with_zp(arg)));
    }
}

TEST_F(attr_quantization_test_t, TestSoftmax) {
    SKIP_IF_CUDA(true, "Unsupported datatype for CUDA");
    SKIP_IF_HIP(true, "Unsupported datatype for HIP");

    memory::desc md {{2, 16}, data_type::u8, tag::ab};

    CHECK_OK(softmax_forward::primitive_desc(
            eng, prop_kind::forward, algorithm::softmax_accurate, md, md, 1));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_OK(softmax_forward::primitive_desc(eng, prop_kind::forward,
                algorithm::softmax_accurate, md, md, 1,
                gen_attr_with_scales(arg)));
        CHECK_UNIMPL(softmax_forward::primitive_desc(eng, prop_kind::forward,
                algorithm::softmax_accurate, md, md, 1, gen_attr_with_zp(arg)));
    }
}

TEST_F(attr_quantization_test_t, TestSum) {
    SKIP_IF_HIP(true, "Unsupported operator for HIP");
    memory::desc md {{1, 16, 3, 3}, data_type::s8, tag::abcd};
    CHECK_OK(sum::primitive_desc(eng, {1.f, 1.f}, {md, md}));
    CHECK_UNIMPL(sum::primitive_desc(eng, {1.f, 1.f}, {md, md},
            gen_attr_with_scales(
                    /* with_wei = false, quantization is not supported */)));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_UNIMPL(sum::primitive_desc(
                eng, {1.f, 1.f}, {md, md}, gen_attr_with_zp(arg)));
    }
}

} // namespace dnnl
