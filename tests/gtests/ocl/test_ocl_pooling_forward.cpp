/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "mkldnn.hpp"
#include "gtest/gtest.h"

#include "mkldnn_test_common.hpp"
#include <string>

#include "CL/cl.h"
#include "mkldnn.hpp"

namespace mkldnn {

struct test_pool_desc_t {
    memory::dim mb, c;
    memory::dim id, ih, iw;
    memory::dim od, oh, ow;
    memory::dim kd, kh, kw;
    memory::dim padf, padt, padl;
    memory::dim strd, strh, strw;
};

struct pool_test_params {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    memory::format_tag src_format;
    memory::format_tag dst_format;
    int ndims;
    test_pool_desc_t test_pd;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

void fill_data_f32(memory &m, bool init = false) {
    float *data = (float *)m.get_data_handle();
    const size_t n_elems = m.get_desc().get_size() / sizeof(float);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t n = 0; n < (ptrdiff_t)n_elems; n++) {
        data[n] = init ? 0 : n % 13;
    }
}
template <typename data_t>
int ref_pool_fwd(const pool_test_params &p, const memory &src,
        const memory &dst, const memory &ws) {
    cl_mem cl_src = src.get_ocl_mem_object();
    cl_mem cl_dst = dst.get_ocl_mem_object();

    bool with_workspace = true && p.aprop_kind == prop_kind::forward_training
            && p.aalgorithm == algorithm::pooling_max;

    cl_mem cl_ws = with_workspace ? ws.get_ocl_mem_object() : NULL;

    engine eng = src.get_engine();
    cl_device_id device = eng.get_ocl_device();
    cl_context context = eng.get_ocl_context();
    const char *source_code
            = " \
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable \n \
        void __kernel pool_fwd(__global DT_DATA *src, __global DT_DATA *dst,\
        __global DT_WS *ws) { \
            int i = get_global_id(0); \
            int ow = i % OW; \
            int oh = i / OW % OH; \
            int od = i / (OW * OH) % OD; \
            int c = i / (OW * OH * OD) % C; \
            int mb = i / (OW * OH * OD *C); \
            AC_DT_DATA temp = 0; \
            int count = 0; \
            for (int kd = 0; kd < KD; ++kd)  { \
                for (int kh = 0; kh < KH; ++kh) { \
                    for (int kw = 0; kw < KW; ++kw) { \
                        const int id = od * SD - PF + kd; \
                        const int ih = oh * SH - PT + kh; \
                        const int iw = ow * SW - PL + kw; \
                        if (ih < 0 || ih >= IH) continue; \
                        if (iw < 0 || iw >= IW) continue; \
                        size_t src_offset = mb*C*ID*IH*IW + c*ID*IH*IW \
                            + id*IH*IW + ih*IW + iw; \
\n #if POOL_MAX == 1 \n\
                        if (kw == 0 && kh == 0) \
                            temp = src[src_offset]; \
                        if (src[src_offset] > temp) \
                            temp = src[src_offset]; \n\
\n #else \n \
                        temp += src[src_offset]; \n\
\n #if POOL_AVG == 1 \n\
                        count++; \
\n #endif \n \
\n #endif \n \
                    } \
                } \
            } \
\n #if POOL_AVG == 1 \n \
            temp /= count; \
\n #endif \n \
\n #if POOL_AVG_PAD == 1 \n \
            temp /= KW*KH*KD; \
\n #endif \n \
            size_t dst_offset = mb*C*OD*OH*OW + c*OD*OH*OW + od*OH*OW + oh*OW + ow; \
            dst[dst_offset] = ROUND(temp); \
        }";

    const size_t source_code_length = strlen(source_code);

    cl_int errcode = 0;
#ifdef CL_VERSION_2_0
    cl_command_queue queue = clCreateCommandQueueWithProperties(
            context, device, NULL, &errcode);
#else
    cl_command_queue queue = clCreateCommandQueue(
            context, device, 0, &errcode);
#endif
    if (errcode != 0)
        return (int)errcode;
    cl_program program = clCreateProgramWithSource(
            context, 1, &source_code, &source_code_length, &errcode);
    if (errcode != 0)
        return (int)errcode;

    std::string options;
    auto define_int = [](std::string &str, const std::string &name, int value) {
        str.append(" -D");
        str.append(name);
        str.append("=");
        str.append(std::to_string(value));
    };

    switch (data_traits<data_t>::data_type) {
    case memory::data_type::f32:
        options.append(" -DDT_DATA=float -DAC_DT_DATA=float -DROUND= ");
        break;
    case memory::data_type::f16:
        options.append(" -DDT_DATA=half -DAC_DT_DATA=half -DROUND= ");
        break;
    case memory::data_type::s8:
        options.append(" -DDT_DATA=char -DAC_DT_DATA=float -DROUND=rint ");
        break;
    default: assert(!"unknown data type");
    }

    switch (p.aalgorithm) {
    case algorithm::pooling_max: define_int(options, "POOL_MAX", 1); break;
    case algorithm::pooling_avg_include_padding:
        define_int(options, "POOL_AVG_PAD", 1);
        break;
    case algorithm::pooling_avg_exclude_padding:
        define_int(options, "POOL_AVG", 1);
        break;
    default: assert(!"unknown algorithm");
    }
    options.append(" -DDT_WS=int");
    const auto pd = p.test_pd;
    define_int(options, "IW", pd.iw);
    define_int(options, "IH", pd.ih);
    define_int(options, "ID", pd.id);
    define_int(options, "OW", pd.ow);
    define_int(options, "OH", pd.oh);
    define_int(options, "OD", pd.od);
    define_int(options, "C", pd.c);
    define_int(options, "MB", pd.mb);
    define_int(options, "KW", pd.kw);
    define_int(options, "KH", pd.kh);
    define_int(options, "KD", pd.kd);
    define_int(options, "SW", pd.strw);
    define_int(options, "SH", pd.strh);
    define_int(options, "SD", pd.strd);
    define_int(options, "PF", pd.padf);
    define_int(options, "PT", pd.padt);
    define_int(options, "PL", pd.padl);

    errcode = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL);

    if (errcode != 0) {
        size_t log_length = 0;
        int info_errcode = clGetProgramBuildInfo(
                program, device, CL_PROGRAM_BUILD_LOG, 0, 0, &log_length);
        if (info_errcode != 0)
            return (int)errcode;

        char *log = (char *)malloc(log_length);
        info_errcode = clGetProgramBuildInfo(
                program, device, CL_PROGRAM_BUILD_LOG, log_length, &log[0], 0);
        if (info_errcode != 0)
            return (int)errcode;
        printf("Error during the build of OpenCL program.\nBuild log:\n%s\n",
                log);
        free(log);
        return (int)errcode;
    }
    cl_kernel kernel = clCreateKernel(program, "pool_fwd", &errcode);
    if (errcode != 0)
        return (int)errcode;
    errcode = clSetKernelArg(kernel, 0, sizeof(cl_src), &cl_src);
    if (errcode != 0)
        return (int)errcode;
    errcode = clSetKernelArg(kernel, 1, sizeof(cl_dst), &cl_dst);
    if (errcode != 0)
        return (int)errcode;
    errcode = clSetKernelArg(kernel, 2, sizeof(cl_ws), &cl_ws);
    if (errcode != 0)
        return (int)errcode;

    size_t gwo[1] = { 0 };
    size_t gws[1] = { (size_t)pd.mb * pd.c * pd.od * pd.oh * pd.ow };
    size_t lws[1] = { 1 };
    errcode = clEnqueueNDRangeKernel(
            queue, kernel, 1, gwo, gws, lws, 0, NULL, NULL);
    if (errcode != 0)
        return (int)errcode;
    errcode = clFinish(queue);
    return (int)errcode;
}

template <typename data_t>
class pooling_test : public ::testing::TestWithParam<pool_test_params>
{
    pool_test_params p;

protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng_host = engine(engine::kind::cpu, 0);
        auto eng = engine(engine::kind::gpu, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;

        test_pool_desc_t pd = p.test_pd;
        auto p_src_desc_test = (p.ndims == 5)
                ? create_md({ pd.mb, pd.c, pd.id, pd.ih, pd.iw },
                          memory::data_type::f32, memory::format_tag::ncdhw)
                : create_md({ pd.mb, pd.c, pd.ih, pd.iw },
                          memory::data_type::f32, memory::format_tag::nchw);
        auto p_dst_desc_test = (p.ndims == 5)
                ? create_md({ pd.mb, pd.c, pd.od, pd.oh, pd.ow },
                          memory::data_type::f32, memory::format_tag::ncdhw)
                : create_md({ pd.mb, pd.c, pd.oh, pd.ow },
                          memory::data_type::f32, memory::format_tag::nchw);

        auto p_src_host = memory({ p_src_desc_test, eng_host });
        auto p_dst_host = memory({ p_dst_desc_test, eng_host });

        auto p_src_desc = (p.ndims == 5)
                ? create_md({ pd.mb, pd.c, pd.id, pd.ih, pd.iw }, data_type,
                          p.src_format)
                : create_md({ pd.mb, pd.c, pd.ih, pd.iw }, data_type,
                          p.src_format);
        auto p_dst_desc = (p.ndims == 5)
                ? create_md({ pd.mb, pd.c, pd.od, pd.oh, pd.ow }, data_type,
                          p.dst_format)
                : create_md({ pd.mb, pd.c, pd.oh, pd.ow }, data_type,
                          p.dst_format);

        auto p_src_desc_dev_test = (p.ndims == 5)
                ? create_md({ pd.mb, pd.c, pd.id, pd.ih, pd.iw }, data_type,
                          memory::format_tag::ncdhw)
                : create_md({ pd.mb, pd.c, pd.ih, pd.iw }, data_type,
                          memory::format_tag::nchw);
        auto p_dst_desc_dev_test = (p.ndims == 5)
                ? create_md({ pd.mb, pd.c, pd.od, pd.oh, pd.ow }, data_type,
                          memory::format_tag::ncdhw)
                : create_md({ pd.mb, pd.c, pd.oh, pd.ow }, data_type,
                          memory::format_tag::nchw);
        auto p_src_dev = memory({ p_src_desc, eng });
        auto p_dst_dev = memory({ p_dst_desc, eng });

        auto p_src_dev_test = memory({ p_src_desc_dev_test, eng });
        auto p_dst_dev_test = memory({ p_dst_desc_dev_test, eng });
        auto p_dst_host_test = memory({ p_dst_desc_test, eng_host });

        fill_data_f32(p_src_host);
        fill_data_f32(p_dst_host, 0.0);
        check_zero_tail<float>(1, p_src_host);
        check_zero_tail<float>(1, p_dst_host);

        // calculate right padding exactly
        std::vector<memory::dim> padR_2d
                = { right_padding(pd.ih, pd.oh, pd.kh, pd.padt, pd.strh),
                      right_padding(pd.iw, pd.ow, pd.kw, pd.padl, pd.strw) };
        std::vector<memory::dim> padR_3d
                = { right_padding(pd.id, pd.od, pd.kd, pd.padf, pd.strd),
                      right_padding(pd.ih, pd.oh, pd.kh, pd.padt, pd.strh),
                      right_padding(pd.iw, pd.ow, pd.kw, pd.padl, pd.strw) };

        memory p_workspace;

        auto pool_desc = (p.ndims == 5)
                ? pooling_forward::desc(p.aprop_kind, p.aalgorithm, p_src_desc,
                          p_dst_desc, { pd.strd, pd.strh, pd.strw },
                          { pd.kd, pd.kh, pd.kw },
                          { pd.padf, pd.padt, pd.padl }, padR_3d)
                : pooling_forward::desc(p.aprop_kind, p.aalgorithm, p_src_desc,
                          p_dst_desc, { pd.strh, pd.strw }, { pd.kh, pd.kw },
                          { pd.padt, pd.padl }, padR_2d);

        auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, eng);

        bool with_workspace = true
                && p.aprop_kind == prop_kind::forward_training
                && p.aalgorithm == algorithm::pooling_max;
        auto p_workspace_desc = with_workspace
                ? pool_prim_desc.workspace_desc()
                : memory::desc({}, data_type, p.dst_format);
        if (with_workspace)
            p_workspace = memory(p_workspace_desc, eng);

        auto p_workspace_host_desc = with_workspace
                ? memory::desc({ pd.mb, pd.c, pd.oh, pd.ow },
                          mkldnn::memory::data_type::s32, p.dst_format)
                : memory::desc({}, data_type, p.dst_format);
        memory p_workspace_host(p_workspace_host_desc, eng_host);

        using primitive_exec_t
                = std::pair<primitive, std::unordered_map<int, memory>>;
        primitive_exec_t pool_exec = { pooling_forward(pool_prim_desc),
            { { MKLDNN_ARG_SRC, p_src_dev }, { MKLDNN_ARG_DST, p_dst_dev } } };
        if (with_workspace)
            pool_exec.second.insert({ MKLDNN_ARG_WORKSPACE, p_workspace });

        auto reo1 = reorder(p_src_host, p_src_dev);
        auto reo2 = reorder(p_dst_host, p_dst_dev);
        auto reo3 = reorder(p_dst_dev, p_dst_host);

        auto reo1_test = reorder(p_src_host, p_src_dev_test);
        auto reo2_test = reorder(p_dst_host, p_dst_dev_test);
        auto reo3_test = reorder(p_dst_dev_test, p_dst_host_test);

        std::vector<primitive_exec_t> pipeline, pipeline_test_inputs,
                pipeline_test_outputs;

        pipeline.push_back({ reo1,
                { { MKLDNN_ARG_SRC, p_src_host },
                        { MKLDNN_ARG_DST, p_src_dev } } });
        pipeline_test_inputs.push_back({ reo1_test,
                { { MKLDNN_ARG_SRC, p_src_host },
                        { MKLDNN_ARG_DST, p_src_dev_test } } });
        pipeline.push_back({ reo2,
                { { MKLDNN_ARG_SRC, p_dst_host },
                        { MKLDNN_ARG_DST, p_dst_dev } } });
        pipeline_test_inputs.push_back({ reo2_test,
                { { MKLDNN_ARG_SRC, p_dst_host },
                        { MKLDNN_ARG_DST, p_dst_dev_test } } });
        pipeline.push_back(pool_exec);
        pipeline.push_back({ reo3,
                { { MKLDNN_ARG_SRC, p_dst_dev },
                        { MKLDNN_ARG_DST, p_dst_host } } });
        pipeline_test_outputs.push_back({ reo3_test,
                { { MKLDNN_ARG_SRC, p_dst_dev_test },
                        { MKLDNN_ARG_DST, p_dst_host_test } } });
        if (with_workspace) {
            reorder r(p_workspace, p_workspace_host);
            pipeline.push_back({ r,
                    { { MKLDNN_ARG_SRC, p_workspace },
                            { MKLDNN_ARG_DST, p_workspace_host } } });
        }

        stream strm(eng);
        for (auto &p : pipeline_test_inputs)
            p.first.execute(strm, p.second);
        for (auto &p : pipeline)
            p.first.execute(strm, p.second);

        strm.wait();

        ASSERT_EQ(ref_pool_fwd<data_t>(
                          p, p_src_dev_test, p_dst_dev_test, p_workspace),
                0);

        for (auto &p : pipeline_test_outputs)
            p.first.execute(strm, p.second);

        strm.wait();

        compare_data<float>(p_dst_host_test, p_dst_host);
    }
};

using pooling_test_float = pooling_test<float>;
using pooling_test_half = pooling_test<float16_t>;
using pooling_test_s8 = pooling_test<int8_t>;

#define EXPAND_SIZES_3D(...) \
    5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D(                                        \
        mb, ic, ih, iw, oh, ow, kh, kw, padt, padl, strh, strw) \
    4, { mb, ic, 1, ih, iw, 1, oh, ow, 1, kh, kw, 0, padt, padl, 1, strh, strw }

TEST_P(pooling_test_float, TestsPooling) {}

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardDebug, pooling_test_float,
        ::testing::Values(
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(1, 7, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(8, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1) }));

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardDebugBlocked, pooling_test_float,
        ::testing::Values(
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(1, 16, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                16, 16, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) }));

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardSlipsToPadding,
        pooling_test_float,
        ::testing::Values(
                pool_test_params{ prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                4, 16, 10, 10, 10, 10, 2, 2, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                4, 16, 10, 10, 10, 10, 2, 2, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                4, 16, 10, 10, 10, 10, 2, 2, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                64, 64, 56, 56, 56, 56, 3, 3, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                64, 64, 56, 56, 56, 56, 3, 3, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                64, 64, 56, 56, 56, 56, 3, 3, 1, 1, 1, 1) }));

TEST_P(pooling_test_half, TestsPooling) {}

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardDebugFP16, pooling_test_half,
        ::testing::Values(
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(1, 7, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(8, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1) }));

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardDebugBlockedFP16,
        pooling_test_half,
        ::testing::Values(
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(1, 16, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                16, 16, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) }));

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardSlipsToPaddingF16,
        pooling_test_half,
        ::testing::Values(
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                4, 16, 10, 10, 10, 10, 2, 2, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                4, 16, 10, 10, 10, 10, 2, 2, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                4, 16, 10, 10, 10, 10, 2, 2, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                64, 64, 56, 56, 56, 56, 3, 3, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                64, 64, 56, 56, 56, 56, 3, 3, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                64, 64, 56, 56, 56, 56, 3, 3, 1, 1, 1, 1) }));

TEST_P(pooling_test_s8, TestsPooling) {}

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardDebugS8, pooling_test_s8,
        ::testing::Values(
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(1, 7, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(8, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1) }));

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardDebugBlockedS8, pooling_test_s8,
        ::testing::Values(
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(1, 16, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                16, 16, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1) }));

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardSlipsToPaddingS8,
        pooling_test_s8,
        ::testing::Values(
                pool_test_params{ prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                4, 16, 10, 10, 10, 10, 2, 2, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                4, 16, 10, 10, 10, 10, 2, 2, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                4, 16, 10, 10, 10, 10, 2, 2, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                64, 64, 56, 56, 56, 56, 3, 3, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                64, 64, 56, 56, 56, 56, 3, 3, 1, 1, 1, 1) },
                pool_test_params{ prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(
                                64, 64, 56, 56, 56, 56, 3, 3, 1, 1, 1, 1) }));
} // namespace mkldnn
