/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_AMD_MIOPEN_CONVOLUTION_IMPL_HPP
#define GPU_AMD_MIOPEN_CONVOLUTION_IMPL_HPP

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/utils.hpp"
#include "gpu/amd/miopen_conv_filter_adjustment_base.hpp"
#include "gpu/amd/miopen_convolution_pd.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

#include <vector>
#include <miopen/miopen.h>

#define MIOPEN_MAX_WK_SIZE 6341787648

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_convolution_impl_base_t
    : public miopen_conv_filter_adjustment_base_t {
protected:
    enum io { x = 0, bias, weights, y, NUM_IO };
    memory_desc_t dnnl_descs[NUM_IO];
    miopenConvolutionDescriptor_t conv_desc;
    int padding[MIOPEN_DIM_MAX];
    int dilation[MIOPEN_DIM_MAX];
    miopenTensorDescriptor_t descs[NUM_IO];
    miopenDataType_t data_types[NUM_IO];
    int ndims[NUM_IO];
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    int strides[NUM_IO + 1][DNNL_MAX_NDIMS];
    int filter_strides[DNNL_MAX_NDIMS];
    miopenTensorLayout_t formats[NUM_IO];
    bool filter_needs_transform = false;
    miopenTensorDescriptor_t weights_desc;
    float alpha = 0.f;
    float beta = 0.f;
    int group_count = 1;
    bool with_groups = false;
    size_t workspace_size = 0;
    bool with_bias = false;
    int selected_sol = -1;
    bool do_scaling = false;
    float output_scaling = 1.0f;
    bool runtime_scaling = false;
    bool use_temp_dst_ = false;
    miopenDataType_t computation_data_type = miopenFloat;
    miopenDataType_t reorder_type = miopenInt8;

public:
    virtual ~miopen_convolution_impl_base_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, weights_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyConvolutionDescriptor, conv_desc);
        for (size_t i = 0; i < io::NUM_IO; i++) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, descs[i]);
        }
    }
    virtual status_t configure_alg_kind(engine_t *, convolution_pd_t *pd) = 0;

    virtual bool supported_filter_format(
            const memory_desc_t *md) const override {
        const memory_desc_wrapper mem_wrapper(md);

        return (mem_wrapper.matches_one_of_tag(format_tag::ab, format_tag::abc,
                        format_tag::abcd, format_tag::abcde, format_tag::abcdef)
                || mem_wrapper.matches_one_of_tag(format_tag::acb,
                        format_tag::acdb, format_tag::acdeb, format_tag::cba,
                        format_tag::cdba, format_tag::cdeba));
    }

    bool using_transformed_filter() const { return filter_needs_transform; }
    bool with_scratchpad() const { return workspace_size > 0; }

    virtual status_t init(engine_t *engine, convolution_pd_t *pd,
            bool use_scratch_dst = false) {
        CHECK(configure_parameters(pd));
        CHECK(create_miopen_descs(pd));
        CHECK(check_output_dims());
        CHECK(configure_alg_kind(engine, pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    virtual status_t init_zero_dims(convolution_pd_t *pd) {
        return status::success;
    }
    void get_dims_and_strides(int io) {
        convert_dims(
                dnnl_descs[io].dims, dims[io], dnnl_descs[io].ndims, ndims[io]);
        if (ndims[io] > dnnl_descs[io].ndims) {
            std::swap(dims[io][ndims[io] - 1], dims[io][ndims[io] - 2]);
            if (ndims[io] == 4) {
                if (formats[io] == miopenTensorNHWC) {
                    propagate_strides(strides[io], dims[io], {1, 3, 2, 0});
                } else {
                    propagate_strides(strides[io], dims[io], {3, 2, 1, 0});
                }
            } else { //changes when ndims is 5, dnnl_desc[io].ndims is 4
                // for nhwc format(axb)
                if (formats[x] == miopenTensorLayout_t::miopenTensorNHWC
                        && formats[y] == miopenTensorLayout_t::miopenTensorNHWC
                        && formats[weights]
                                == miopenTensorLayout_t::miopenTensorNHWC) {
                    if (io == 2) {
                        propagate_strides(strides[io], dims[io] + with_groups,
                                {1, 3, 2, 0});
                    } else {
                        propagate_strides(strides[io], dims[io], {1, 3, 2, 0});
                    }
                } else //for nchw format(abx)
                {
                    strides[io][ndims[io]] = 1;
                    for (int k = ndims[io] - 1; k >= 0; k--) {
                        strides[io][k] = (k != ndims[io] - 1)
                                ? strides[io][k + 1] * dims[io][k + 1]
                                : 1;
                    }
                }
            }
        } else { //when ndims[io] == dnnl_desc[io].ndims
            convert_dims(dnnl_descs[io].format_desc.blocking.strides,
                    strides[io], dnnl_descs[io].ndims, ndims[io]);
        }
    }

    status_t configure_parameters(const convolution_pd_t *pd) {
        if (pd->ndims() > MIOPEN_DIM_MAX) { return status::invalid_arguments; }
        CHECK(set_padding_and_dilation(pd));
        with_groups = pd->with_groups();
        with_bias = pd->with_bias();
        alpha = 1.0f;
        beta = 0.0f;
        do_scaling = !pd->attr()->output_scales_.has_default_values();
        output_scaling = !pd->attr()->output_scales_.defined();

        dnnl_descs[x] = *pd->invariant_src_md();
        dnnl_descs[weights] = *pd->invariant_wei_md();
        dnnl_descs[y] = *pd->invariant_dst_md();
        if (with_bias) dnnl_descs[bias] = *pd->invariant_bia_md();

        ndims[x] = std::max(dnnl_descs[x].ndims, 4);
        ndims[weights] = std::max(dnnl_descs[weights].ndims, 4 + with_groups);
        ndims[y] = std::max(dnnl_descs[y].ndims, 4);

        CHECK(convert_data_type(&dnnl_descs[x], &data_types[x]));
        CHECK(convert_data_type(&dnnl_descs[weights], &data_types[weights]));
        CHECK(convert_data_type(&dnnl_descs[y], &data_types[y]));

        CHECK(get_formats());
        set_compute_format();
        get_dims_and_strides(x);
        get_dims_and_strides(weights);
        get_dims_and_strides(y);

        if (!supported_filter_format(&dnnl_descs[weights])) {
            set_filter_format(
                    ndims[weights], dims[weights], strides[NUM_IO], formats[x]);
            CHECK(init_filter_transformation(data_types[weights],
                    ndims[weights], dims[weights], strides[weights],
                    strides[NUM_IO]));
            filter_needs_transform = true;
            // we transform the filter based on src format
            formats[weights] = formats[x];
        } else {
            CHECK(get_filter_format());
            get_dims_and_strides(weights);
        }
        if (with_groups) {
            dims[weights][1] *= pd->G();
            ndims[weights] = std::max(4, ndims[weights] - with_groups);
        }

        if (with_bias) {
            ndims[bias] = dnnl_descs[bias].ndims;
            CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
            convert_dims(
                    dnnl_descs[bias].dims, dims[bias], ndims[bias], ndims[y]);
            std::swap(dims[bias][0], dims[bias][1]);
            convert_dims(dnnl_descs[bias].format_desc.blocking.strides,
                    strides[bias], ndims[bias], ndims[y]);
            ndims[bias] = ndims[y];
        }

        return status::success;
    }

    status_t create_miopen_descs(const convolution_pd_t *pd) {
        CHECK(create_and_set_convolution_desc(pd));
        CHECK(create_and_set_tensor_descriptor(
                &descs[x], data_types[x], ndims[x], dims[x], strides[x]));
        //For abx case
        if (formats[x] != miopenTensorLayout_t::miopenTensorNHWC
                && formats[weights] != miopenTensorLayout_t::miopenTensorNHWC
                && formats[y] != miopenTensorLayout_t::miopenTensorNHWC) {
            CHECK(create_and_set_tensor_descriptor(&weights_desc,
                    data_types[weights], ndims[weights],
                    dims[weights] + with_groups,
                    strides[weights] + with_groups));
        } else { //For axb case
            CHECK(create_and_set_tensor_descriptor(&weights_desc,
                    data_types[weights], ndims[weights],
                    dims[weights] + with_groups, strides[weights]));
        }
        CHECK(create_and_set_tensor_descriptor(
                &descs[y], data_types[y], ndims[y], dims[y], strides[y]));

        if (with_bias) {
            CHECK(create_and_set_tensor_descriptor(&descs[bias],
                    data_types[bias], ndims[bias], dims[bias], strides[bias]));
        }

        return status::success;
    }
    virtual status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) {
        if (filter_needs_transform) {
            auto sz = memory_desc_wrapper(&dnnl_descs[weights]).size();
            auto data_size
                    = types::data_type_size(pd->invariant_wei_md(0)->data_type);
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_miopen_filter, sz,
                    data_size);
        }
        return status::success;
    };

    status_t create_and_set_convolution_desc(const convolution_pd_t *pd) {

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateConvolutionDescriptor, &conv_desc));

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenInitConvolutionNdDescriptor,
                conv_desc, ndims[x] - 2, padding, filter_strides, dilation,
                miopenConvolution));

        // Check for groups and set group count if necessary
        if (with_groups) {
            group_count = pd->G();
            if (group_count > 1)
                CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetConvolutionGroupCount,
                        conv_desc, group_count));
        }
        return status::success;
    }

    status_t set_padding_and_dilation(const convolution_pd_t *pd) {
        int actual_ndims = pd->ndims();
        if (actual_ndims == 3) {
            padding[0] = 0;
            padding[1] = static_cast<int>(pd->padL());
            dilation[0] = 1;
            dilation[1] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = 1;
            filter_strides[1] = static_cast<int>(pd->KSW());
        } else if (actual_ndims == 4) {
            padding[0] = static_cast<int>(pd->padT());
            padding[1] = static_cast<int>(pd->padL());

            dilation[0] = static_cast<int>(pd->KDH() + 1);
            dilation[1] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = static_cast<int>(pd->KSH());
            filter_strides[1] = static_cast<int>(pd->KSW());
        } else {
            padding[0] = static_cast<int>(pd->padFront());
            padding[1] = static_cast<int>(pd->padT());
            padding[2] = static_cast<int>(pd->padL());

            dilation[0] = static_cast<int>(pd->KDD() + 1);
            dilation[1] = static_cast<int>(pd->KDH() + 1);
            dilation[2] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = static_cast<int>(pd->KSD());
            filter_strides[1] = static_cast<int>(pd->KSH());
            filter_strides[2] = static_cast<int>(pd->KSW());
        }
        return status::success;
    }

    virtual void execute(
            miopenHandle_t handle, const std::vector<void *> &args) const = 0;

    void execute_sum(miopenHandle_t handle, void *x, void *y, float alpha_,
            float beta_) const {
        float alpha = alpha_;
        float beta = beta_;
        int alpha2 = 0;
        miopenTensorOp_t tensorOp = miopenTensorOpAdd;
        MIOPEN_EXECUTE_FUNC_V(miopenOpTensor, handle, tensorOp, &alpha,
                descs[io::y], x, &alpha2, descs[io::y], y, &beta, descs[io::y],
                y);
    }

    void execute_scale(miopenHandle_t handle, void *y, void *rt_oscale) const {
        if (do_scaling) {
            const void *s = runtime_scaling ? rt_oscale : &output_scaling;
            MIOPEN_EXECUTE_FUNC_V(
                    miopenScaleTensor, handle, descs[io::y], y, s);
        }
    }

    void execute_set_weights_bias(
            miopenHandle_t handle, void *weights, void *bias, float value) {
        MIOPEN_EXECUTE_FUNC_V(
                miopenSetTensor, handle, descs[io::weights], weights, &value);
        if (bias) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenSetTensor, handle, descs[io::bias], bias, &value);
        }
    }

    bool with_eltwise(const convolution_pd_t *pd, int position) const {
        return pd->attr()->post_ops_.contain(primitive_kind::eltwise, position);
    }

    status_t check_output_dims() {
        int expected_dims[MIOPEN_DIM_MAX] = {};
        MIOPEN_EXECUTE_FUNC_V(miopenGetConvolutionNdForwardOutputDim, conv_desc,
                descs[x], weights_desc, &ndims[y], &expected_dims[0]);
        for (size_t i = 0; i < ndims[y]; i++) {
            if (dims[y][i] != expected_dims[i]) return status::unimplemented;
        }
        return status::success;
    }

    void set_compute_format() {
        if (data_types[x] == miopenInt8) {
            computation_data_type = miopenInt32;
        } else {
            computation_data_type = miopenFloat;
        }
    }

    status_t get_filter_format() {
        memory_desc_wrapper wrapper(&dnnl_descs[weights]);
        if (wrapper.matches_one_of_tag(format_tag::ab, format_tag::abc,
                    format_tag::abcd, format_tag::abcde, format_tag::abcdef)) {
            formats[weights] = miopenTensorLayout_t::miopenTensorNCHW;
        } else if (wrapper.matches_one_of_tag(format_tag::acb, format_tag::acdb,
                           format_tag::acdeb, format_tag::cba, format_tag::cdba,
                           format_tag::cdeba)) {
            formats[weights] = miopenTensorLayout_t::miopenTensorNHWC;
        } else {
            return status::unimplemented;
        }

        return status::success;
    }

    status_t get_formats() {
        CHECK(get_format(&dnnl_descs[x], formats[x]));
        CHECK(get_format(&dnnl_descs[y], formats[y]));
        CHECK(get_format(&dnnl_descs[weights], formats[weights]));
        return status::success;
    }

    void set_filter_nhwc(int filter_ndims, int *transform_filter_strides,
            int *filter_dims) override {
        if (with_groups) {
            switch (filter_ndims) {
                case 4: // Convert to krsc
                    return propagate_strides(transform_filter_strides,
                            filter_dims, {2, 3, 1, 0});
                case 5:
                    return propagate_strides(transform_filter_strides,
                            filter_dims, {2, 4, 3, 1, 0});
                case 6:
                    return propagate_strides(transform_filter_strides,
                            filter_dims, {2, 5, 4, 3, 1, 0});
            }
        } else {
            miopen_conv_filter_adjustment_base_t::set_filter_nhwc(
                    filter_ndims, transform_filter_strides, filter_dims);
        }
    }

    bool use_temp_dst() const { return use_temp_dst_; }
};

struct miopen_convolution_impl_fwd_t : public miopen_convolution_impl_base_t {
protected:
    miopenActivationDescriptor_t activation_desc = nullptr;
    miopenActivationDescriptor_t eltwise_desc = nullptr;
    miopenActivationDescriptor_t act_desc = nullptr;
    miopenTensorDescriptor_t reorder_dst_desc = nullptr;
    miopenConvFwdAlgorithm_t fwd_alg_kind;
    std::vector<miopenConvAlgoPerf_t> perf;

    int requested_algo_count = 5;
    int returned_algo_count = 0;
    int num_post_ops = 0;
    primitive_kind_t post_ops[2];
    bool need_reorder = false;
    float sum_scale = 1.0f;
    size_t maxSolutionCount = 0;
    std::vector<miopenConvSolution_t> solutions;
    size_t actualCount = 0;

    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenOperatorArgs_t fusionArgs;
    miopenFusionOpDescriptor_t convoOp;
    miopenFusionOpDescriptor_t biasOp;
    miopenFusionOpDescriptor_t activOp;
    float activeAlphaFusionAct;
    float activeBetaFusionAct;
    float activeGammaFusionAct;

public:
    virtual ~miopen_convolution_impl_fwd_t() {
        if (activation_desc)
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyActivationDescriptor, activation_desc);
        if (eltwise_desc)
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyActivationDescriptor, eltwise_desc);
        if (reorder_dst_desc)
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyTensorDescriptor, reorder_dst_desc);
    }

    status_t configure_post_ops(convolution_pd_t *pd) {
        auto &p = pd->attr()->post_ops_;
        num_post_ops = p.len();
        for (size_t i = 0; i < p.len(); i++) {
            post_ops[i] = p.entry_[i].kind;
            if (post_ops[i] == dnnl_eltwise) {
                CHECK(create_and_set_eltwise_descriptor(pd));
            }
            if (post_ops[i] == dnnl_sum) { sum_scale = p.entry_[i].sum.scale; }
        }

        if (data_types[y] == miopenInt8 && use_temp_dst_) {
            data_types[y] = miopenFloat;
            need_reorder = true;
            int strides_[DNNL_MAX_NDIMS];
            convert_dims(pd->src_md()->format_desc.blocking.strides, strides_,
                    pd->ndims());
            CHECK(create_and_set_tensor_descriptor(&reorder_dst_desc,
                    reorder_type, ndims[y], dims[y], strides_));
        }
        return status::success;
    }

    status_t init(engine_t *engine, convolution_pd_t *pd,
            bool use_scratch_dst) override {
        use_temp_dst_ = use_scratch_dst;
        CHECK(configure_parameters(pd));
        CHECK(create_miopen_descs(pd));
        CHECK(configure_alg_kind(engine, pd));
        CHECK(configure_post_ops(pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    void execute_reorder(miopenHandle_t handle, void *src, void *dst,
            bool flip_formats) const {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if (flip_formats) {
            MIOPEN_EXECUTE_FUNC_V(miopenTransformTensor, handle, &alpha,
                    reorder_dst_desc, src, &beta, descs[io::y], dst);
        } else {
            MIOPEN_EXECUTE_FUNC_V(miopenTransformTensor, handle, &alpha,
                    descs[io::y], src, &beta, reorder_dst_desc, dst);
        }
    }

    void execute_eltwise(miopenHandle_t handle, void *src, void *dst) const {
        float alpha = 1.0f;
        float beta = 0.0f;
        MIOPEN_EXECUTE_FUNC_V(miopenActivationForward, handle, eltwise_desc,
                &alpha, descs[io::y], src, &beta, descs[io::y], dst);
    }

    void execute(miopenHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4], post_op_scratch = args[6],
             post_op_reorder = args[7], runtime_oscale = args[8];
        void *output = use_temp_dst_ ? post_op_scratch : y;

        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            weights = w_scratch;
        }

        MIOPEN_EXECUTE_FUNC_V(miopenConvolutionForwardImmediate, handle,
                weights_desc, weights, descs[io::x], x, conv_desc, descs[io::y],
                output, scratchpad, solutions[selected_sol].workspace_size,
                solutions[selected_sol].solution_id);

        if (with_bias) {
            float bias_alpha = 0;
            float alpha2 = 1.0f;
            float bias_beta = 1.0f;

            MIOPEN_EXECUTE_FUNC_V(miopenOpTensor, handle, miopenTensorOpAdd,
                    &bias_alpha, descs[io::y], output, &alpha2, descs[io::bias],
                    bias, &bias_beta, descs[io::y], output);
        }

        execute_scale(handle, output, runtime_oscale);
        // skip first eltwise in case it is fused into convolution
        const int post_ops_start_pos = 0;
        for (int i = post_ops_start_pos; i < num_post_ops; i++) {
            bool last_op = i == num_post_ops - 1 && !need_reorder;
            switch (post_ops[i]) {
                case dnnl_sum:
                    if (need_reorder) {
                        execute_reorder(handle, y, post_op_reorder, true);
                        execute_sum(handle, post_op_reorder, post_op_scratch,
                                sum_scale, 1.0f);
                    } else if (last_op) {
                        execute_sum(
                                handle, post_op_scratch, y, 1.0f, sum_scale);
                    } else {
                        execute_sum(
                                handle, y, post_op_scratch, sum_scale, 1.0f);
                    }

                    break;

                case dnnl_eltwise:
                    if (last_op) {
                        execute_eltwise(handle, output, y);
                    } else {
                        execute_eltwise(handle, output, post_op_scratch);
                    }
                    break;
                default: assert(!"unsupported post op");
            }
        }

        if (need_reorder) {
            execute_reorder(handle, post_op_scratch, y, false);
        }
    }
    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));
        auto hip_stream = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionForwardGetSolutionCount,
                handle, weights_desc, descs[io::x], conv_desc, descs[io::y],
                &maxSolutionCount));

        solutions.resize(maxSolutionCount);

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionForwardGetSolution, handle,
                weights_desc, descs[io::x], conv_desc, descs[io::y],
                maxSolutionCount, &actualCount, solutions.data()));

        for (int i = 0; i < actualCount; i++) {
            if (solutions[i].workspace_size > 0) continue;
            selected_sol = i;
            break;
        }

        if (selected_sol == -1) return status::unimplemented;

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionForwardGetSolutionWorkspaceSize, handle,
                weights_desc, descs[io::x], conv_desc, descs[io::y],
                solutions[selected_sol].solution_id, &workspace_size));

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionForwardCompileSolution,
                handle, weights_desc, descs[io::x], conv_desc, descs[io::y],
                solutions[selected_sol].solution_id));

        if (workspace_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_miopen_algo,
                    workspace_size, size_t(1));
        return miopen_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        hip_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        double activAlpha, activBeta, activGamma;
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateActivationDescriptor, &activation_desc));
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetActivationDescriptor,
                activation_desc,
                miopenActivationMode_t::miopenActivationPASTHRU, activAlpha,
                activBeta, activGamma));
        return status::success;
    }

    status_t create_and_set_eltwise_descriptor(const convolution_pd_t *pd) {
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateActivationDescriptor, &eltwise_desc));

        miopenActivationMode_t act_mode;
        switch (eltwise_algorithm_kind(pd)) {
            case alg_kind::eltwise_tanh: act_mode = miopenActivationTANH; break;
            case alg_kind::eltwise_elu: act_mode = miopenActivationELU; break;
            case alg_kind::eltwise_relu:
                act_mode = miopenActivationLEAKYRELU;
                break;
            case alg_kind::eltwise_logistic:
                act_mode = miopenActivationLOGISTIC;
                break;
            default: return status::unimplemented;
        }

        float activAlpha = 0;
        float activBeta = 0;
        float activGamma = 0;

        // For ReLU, a ceiling of 0 means no limit.
        double ceiling = eltwise_alpha(pd);

        if (act_mode == miopenActivationMode_t::miopenActivationTANH)
            activAlpha = activBeta = 1;
        else if (act_mode == miopenActivationMode_t::miopenActivationELU)
            activAlpha = ceiling;
        else if (act_mode
                == miopenActivationMode_t::miopenActivationCLIPPEDRELU)
            activAlpha = ceiling;
        else if (act_mode == miopenActivationMode_t::miopenActivationLEAKYRELU)
            activAlpha = ceiling;
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetActivationDescriptor, eltwise_desc,
                act_mode, activAlpha, activBeta, activGamma));
        return status::success;
    }

    dnnl::impl::alg_kind_t eltwise_algorithm_kind(
            const convolution_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alg;
    }

    float eltwise_alpha(const convolution_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha;
    }
};
struct miopen_convolution_impl_bwd_data_t
    : public miopen_convolution_impl_base_t {
protected:
    status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        hip_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardDataGetSolutionCount, handle,
                descs[io::y], weights_desc, conv_desc, descs[io::x],
                &solutionCountm));

        solutions.resize(solutionCountm);

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionBackwardDataGetSolution,
                handle, descs[io::y], weights_desc, conv_desc, descs[io::x],
                solutionCountm, &solutionCount, solutions.data()));

        for (int i = 0; i < solutionCount; i++) {
            if (selected_sol == -1) {
                ws_size = solutions[i].workspace_size;
                selected_sol = i;
            }
            if (solutions[i].workspace_size < ws_size) {
                ws_size = solutions[i].workspace_size;
                selected_sol = i;
            }
        }

        ws_size = solutions[selected_sol].workspace_size;
        auto solution_id = solutions[selected_sol].solution_id;

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardDataGetSolutionWorkspaceSize, handle,
                descs[io::y], weights_desc, conv_desc, descs[io::x],
                solution_id, &ws_size));

        if (ws_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_miopen_algo, ws_size,
                    size_t(1));

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardDataCompileSolution, handle,
                descs[io::y], weights_desc, conv_desc, descs[io::x],
                solution_id));

        return miopen_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(miopenHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4];
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            weights = w_scratch;
        }
        const float bias_alpha = 1.0f;
        const float bias_beta = 1.0f;

        MIOPEN_EXECUTE_FUNC_V(miopenConvolutionBackwardDataImmediate, handle,
                descs[io::y], y, weights_desc, weights, conv_desc, descs[io::x],
                x, scratchpad, ws_size, solutions[selected_sol].solution_id);

        if (with_bias) {
            float alpha2 = 0;
            miopenTensorOp_t tensorOp = miopenTensorOpAdd;
            MIOPEN_EXECUTE_FUNC_V(miopenOpTensor, handle, tensorOp, &alpha2,
                    descs[io::x], x, &bias_alpha, descs[io::bias], bias,
                    &bias_beta, descs[io::x], x);
        }
    }

public:
    size_t solutionCount = 0;
    size_t solutionCountm = 0;
    std::vector<miopenConvSolution_t> solutions;
    uint8_t *ws = nullptr;
    size_t ws_size;
};

struct miopen_convolution_impl_bwd_weights_t
    : public miopen_convolution_impl_base_t {
protected:
    miopenConvBwdWeightsAlgorithm_t bwd_filter_algo
            = miopenConvolutionBwdWeightsAlgoDirect;
    std::vector<miopenConvAlgoPerf_t> perf;
    int requested_algo_count = 4;
    int returned_algo_count = 0;

public:
    status_t init_zero_dims(convolution_pd_t *pd) override {
        if (pd->ndims() > MIOPEN_DIM_MAX) { return status::invalid_arguments; }
        dnnl_descs[weights] = *pd->invariant_wei_md();
        CHECK(get_format(&dnnl_descs[weights], formats[weights], true));
        ndims[y] = pd->invariant_dst_md()->ndims;
        ndims[weights] = dnnl_descs[weights].ndims - pd->with_groups();

        CHECK(convert_data_type(&dnnl_descs[weights], &data_types[weights]));
        convert_dims(dnnl_descs[weights].dims + pd->with_groups(),
                dims[weights], ndims[weights]);
        ndims[weights] = std::max(4, ndims[weights]);
        convert_dims(dnnl_descs[weights].format_desc.blocking.strides,
                strides[weights], ndims[weights]);
        CHECK(create_and_set_tensor_descriptor(&descs[weights],
                data_types[weights], ndims[weights], dims[weights],
                strides[weights]));

        if (pd->with_bias()) {
            dnnl_descs[bias] = *pd->invariant_bia_md();
            ndims[bias] = dnnl_descs[bias].ndims;
            CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
            convert_dims(dnnl_descs[bias].padded_dims, dims[bias], ndims[bias],
                    ndims[y]);
            std::swap(dims[bias][0], dims[bias][1]);
            convert_dims(dnnl_descs[bias].format_desc.blocking.strides,
                    strides[bias], ndims[bias], ndims[weights]);
            ndims[bias] = ndims[y];
            CHECK(create_and_set_tensor_descriptor(&descs[bias],
                    data_types[bias], ndims[bias], dims[bias], strides[bias]));
        }
        return status::success;
    }
    virtual status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        hip_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardWeightsGetSolutionCount, handle,
                descs[io::y], descs[io::x], conv_desc, weights_desc,
                &solutionCountm));

        solutions.resize(solutionCountm);

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionBackwardWeightsGetSolution,
                handle, descs[io::y], descs[io::x], conv_desc, weights_desc,
                solutionCountm, &solutionCount, solutions.data()));

        for (int i = 0; i < solutionCount; i++) {
            if (selected_sol == -1) {
                ws_size = solutions[i].workspace_size;
                selected_sol = i;
            }
            if (solutions[i].workspace_size <= ws_size) {
                ws_size = solutions[i].workspace_size;
                selected_sol = i;
            }
        }

        ws_size = solutions[selected_sol].workspace_size;
        auto solution_id = solutions[selected_sol].solution_id;

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardWeightsGetWorkSpaceSize, handle,
                descs[io::y], descs[io::x], conv_desc, weights_desc, &ws_size));

        if (ws_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_miopen_algo, ws_size,
                    size_t(1));

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardWeightsCompileSolution, handle,
                descs[io::y], descs[io::x], conv_desc, weights_desc,
                solution_id));

        return miopen_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(miopenHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4];
        auto filter = weights;
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            filter = w_scratch;
        }
        const float bias_alpha = 1.0f;
        const float bias_beta = 0.0f;

        MIOPEN_EXECUTE_FUNC_V(miopenConvolutionBackwardWeightsImmediate, handle,
                descs[io::y], y, descs[io::x], x, conv_desc, weights_desc,
                weights, scratchpad, solutions[selected_sol].workspace_size,
                solutions[selected_sol].solution_id);

        if (with_bias) {
            MIOPEN_EXECUTE_FUNC_V(miopenConvolutionBackwardBias, handle,
                    &bias_alpha, descs[io::y], y, &bias_beta, descs[io::bias],
                    bias);
        }
        if (using_transformed_filter()) {
            undo_transform_filter(handle, filter, weights);
        }
    }

public:
    size_t solutionCount = 0;
    size_t solutionCountm = 0;
    std::vector<miopenConvSolution_t> solutions;
    uint8_t *ws = nullptr;
    size_t ws_size = -1;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
