/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include <stdio.h>
#include "mkldnn.h"

#define BATCH 256

#define CHECK(f) do { \
    mkldnn_status_t s = f; \
    if (s != mkldnn_success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

static void init_data_memory(uint32_t dim, uint32_t *dims,
        mkldnn_memory_format_t user_fmt, mkldnn_precision_t mkldnn_f32,
        const_mkldnn_engine_t engine, float *data, mkldnn_primitive_t *memory)
{
    mkldnn_tensor_desc_t tensor;
    mkldnn_memory_desc_t prim_md;
    mkldnn_memory_primitive_desc_t user_pd;
    CHECK(mkldnn_tensor_desc_init(&tensor, dim, dims));
    CHECK(mkldnn_memory_desc_init(&prim_md, &tensor, mkldnn_f32, user_fmt));
    CHECK(mkldnn_memory_primitive_desc_init(&user_pd, &prim_md, engine));
    CHECK(mkldnn_memory_create(memory, &user_pd, data));
}

mkldnn_status_t prepare_reorder(
        mkldnn_primitive_t *user_memory, /** in */
        mkldnn_memory_primitive_desc_t *prim_memory_pd, /** in */
        int dir_is_user_to_prim, /** in: user -> prim or prim -> user */
        mkldnn_primitive_t *prim_memory, /** out: memory primitive created */
        mkldnn_primitive_t *reorder /** out: reorder primitive created */
        )
{
    mkldnn_memory_primitive_desc_t user_memory_pd;
    mkldnn_memory_get_primitive_desc(*user_memory, &user_memory_pd);

    if (!mkldnn_memory_primitive_desc_equal(&user_memory_pd, prim_memory_pd)) {
        /* memory_create(&p, m, NULL) means allocate memory */
        CHECK(mkldnn_memory_create(prim_memory, prim_memory_pd, NULL));
        mkldnn_reorder_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            /* reorder primitive descriptor doesn't need engine, because it is
             * already appeared in in- and out- memory primitive descriptors */
            CHECK(mkldnn_reorder_primitive_desc_init(&reorder_pd,
                        &user_memory_pd, prim_memory_pd));
            mkldnn_primitive_at_t user_memory_at = { *user_memory };
            CHECK(mkldnn_reorder_create(reorder, &reorder_pd, user_memory_at,
                        *prim_memory));
        } else {
            CHECK(mkldnn_reorder_primitive_desc_init(&reorder_pd,
                        prim_memory_pd, &user_memory_pd));
            mkldnn_primitive_at_t prim_memory_at = { *prim_memory };
            CHECK(mkldnn_reorder_create(reorder, &reorder_pd, prim_memory_at,
                        *user_memory));
        }
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return mkldnn_success;
}

mkldnn_status_t simple_net(){

    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));

    float *net_src = (float*)calloc(BATCH*3*227*227, sizeof(float));
    float *net_dst = (float*)calloc(BATCH*96*27*27, sizeof(float));

    /* AlexNet: conv
     * {BATCH, 3, 227, 227} (x) {96, 3, 11, 11} -> {BATCH, 96, 55, 55}
     * strides: {4, 4}
     */
    float *conv_src = net_src;
    float *conv_weights = (float*)calloc(96*3*11*11, sizeof(float));
    float *conv_bias = (float*)calloc(96, sizeof(float));

    uint32_t conv_src_sizes[4] = {BATCH, 3, 227, 227};
    uint32_t conv_weights_sizes[4] = {96, 3, 11, 11};
    uint32_t conv_bias_sizes[4] = {96};
    uint32_t conv_dst_sizes[4] = {BATCH, 96, 55, 55};
    uint32_t conv_strides[2] = {4, 4};
    int32_t  conv_padding[2] = {0, 0};

    /* create memory for user data */
    mkldnn_primitive_t conv_user_src_memory, conv_user_weights_memory,
        conv_user_bias_memory;
    init_data_memory(4, conv_src_sizes, mkldnn_nchw, mkldnn_f32, engine,
        conv_src, &conv_user_src_memory);
    init_data_memory(4, conv_weights_sizes, mkldnn_oihw, mkldnn_f32, engine,
        conv_weights, &conv_user_weights_memory);
    init_data_memory(1, conv_bias_sizes, mkldnn_x, mkldnn_f32, engine,
        conv_bias, &conv_user_bias_memory);

    /* create data descriptors for convolution w/ no specified format */
    mkldnn_tensor_desc_t conv_src_tz, conv_weights_tz, conv_bias_tz,
        conv_dst_tz;
    CHECK(mkldnn_tensor_desc_init(&conv_src_tz, 4, conv_src_sizes));
    CHECK(mkldnn_tensor_desc_init(&conv_weights_tz, 4, conv_weights_sizes));
    CHECK(mkldnn_tensor_desc_init(&conv_bias_tz, 1, conv_bias_sizes));
    CHECK(mkldnn_tensor_desc_init(&conv_dst_tz, 4, conv_dst_sizes));

    mkldnn_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md,
        conv_dst_md;
    CHECK(mkldnn_memory_desc_init(&conv_src_md, &conv_src_tz,
        mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_weights_md, &conv_weights_tz,
        mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_bias_md, &conv_bias_tz,
        mkldnn_f32, mkldnn_x));
    CHECK(mkldnn_memory_desc_init(&conv_dst_md, &conv_dst_tz,
        mkldnn_f32, mkldnn_any));

    /* create a convolution */
    mkldnn_convolution_desc_t conv_any_desc;
    CHECK(mkldnn_convolution_desc_init(&conv_any_desc, mkldnn_forward,
            mkldnn_convolution_direct, &conv_src_md, &conv_weights_md,
            &conv_bias_md, &conv_dst_md, conv_strides, conv_padding,
            mkldnn_padding_zero));

    mkldnn_convolution_primitive_desc_t conv_pd;
    CHECK(mkldnn_convolution_primitive_desc_init(&conv_pd, &conv_any_desc,
            engine));

    mkldnn_primitive_t conv_internal_src_memory, conv_internal_weights_memory,
        conv_internal_dst_memory;

    /* create memory for dst data, we don't need reorder it to user data
     * memory_create(&p, m, NULL) means allocate memory */
    CHECK(mkldnn_memory_create(&conv_internal_dst_memory,
            &conv_pd.dst_primitive_desc, NULL));

    /* create reorder primitives between user data and convolution srcs
     * if required */
    mkldnn_primitive_t conv_reorder_src, conv_reorder_weights;

    CHECK(prepare_reorder(&conv_user_src_memory,
            &conv_pd.src_primitive_desc, 1, &conv_internal_src_memory,
            &conv_reorder_src));
    CHECK(prepare_reorder(&conv_user_weights_memory,
            &conv_pd.weights_primitive_desc, 1, &conv_internal_weights_memory,
            &conv_reorder_weights));

    mkldnn_primitive_t conv_src_memory = conv_internal_src_memory ?
        conv_internal_src_memory : conv_user_src_memory;
    mkldnn_primitive_t conv_weights_memory = conv_internal_weights_memory ?
        conv_internal_weights_memory : conv_user_weights_memory;

    mkldnn_primitive_at_t conv_srcs[] = {
        mkldnn_primitive_at(conv_src_memory, 0),
        mkldnn_primitive_at(conv_weights_memory, 0),
        mkldnn_primitive_at(conv_user_bias_memory, 0)
    };

    const_mkldnn_primitive_t conv_dsts[] = { conv_internal_dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t conv;
    CHECK(mkldnn_primitive_create(&conv, &conv_pd, conv_srcs, conv_dsts));

    /* AlexNet: relu
     * {BATCH, 96, 55, 55} -> {BATCH, 96, 55, 55}
     */
    double negative_slope = 1.0;

    /* create relu memory descriptor on dst memory descriptor
     * from previos primitive */
    mkldnn_memory_desc_t relu_src_md = conv_pd.dst_primitive_desc.memory_desc;

    /* create a relu */
    mkldnn_relu_desc_t relu_desc;
    CHECK(mkldnn_relu_desc_init(&relu_desc, mkldnn_forward, negative_slope,
            &relu_src_md, &relu_src_md));

    mkldnn_relu_primitive_desc_t relu_pd;
    CHECK(mkldnn_relu_primitive_desc_init(&relu_pd, &relu_desc, engine));

    mkldnn_primitive_t relu_dst_memory;
    CHECK(mkldnn_memory_create(&relu_dst_memory,
                &relu_pd.dst_primitive_desc, NULL));

    /* finally create a relu primitive */
    mkldnn_primitive_t relu;
    mkldnn_primitive_at_t relu_srcs[] = { conv_internal_dst_memory };
    const_mkldnn_primitive_t relu_dsts[] = { relu_dst_memory };

    CHECK(mkldnn_primitive_create(&relu, &relu_pd, relu_srcs, relu_dsts));

    /* AlexNet: lrn
     * {BATCH, 96, 55, 55} -> {BATCH, 96, 55, 55}
     * local size: 5
     * alpha: 0.0001
     * beta: 0.75
     */
    uint32_t local_size = 5;
    double alpha = 0.0001;
    double beta = 0.75;

    /* create lrn memory descriptor on dst memory descriptor
     *  from previos primitive */
    mkldnn_memory_desc_t lrn_src_md = relu_pd.dst_primitive_desc.memory_desc;

    /* create a lrn */
    mkldnn_lrn_desc_t lrn_desc;
    CHECK(mkldnn_lrn_desc_init(&lrn_desc, mkldnn_forward,
            mkldnn_lrn_across_channels, &lrn_src_md, &lrn_src_md,
            alpha, beta, local_size));

    mkldnn_lrn_primitive_desc_t lrn_pd;
    CHECK(mkldnn_lrn_primitive_desc_init(&lrn_pd, &lrn_desc, engine));

    mkldnn_primitive_t lrn_dst_memory, lrn_scratch_memory;
    CHECK(mkldnn_memory_create(&lrn_dst_memory,
        &lrn_pd.dst_primitive_desc, NULL));
    CHECK(mkldnn_memory_create(&lrn_scratch_memory,
        &lrn_pd.scratch_primitive_desc, NULL));

    mkldnn_primitive_at_t lrn_srcs[] = {
        mkldnn_primitive_at(relu_dst_memory, 0),
        mkldnn_primitive_at(lrn_scratch_memory, 0)
    };

    const_mkldnn_primitive_t lrn_dsts[] = { lrn_dst_memory };

    /* finally create a lrn primitive */
    mkldnn_primitive_t lrn;
    CHECK(mkldnn_primitive_create(&lrn, &lrn_pd, lrn_srcs, lrn_dsts));

    /* AlexNet: pool
     * {BATCH, 96, 55, 55} -> {BATCH, 96, 27, 27}
     * kernel: {3, 3}
     * strides: {2, 2}
     */
    uint32_t pool_dst_sizes[4] = {BATCH, 96, 27, 27};
    uint32_t pool_kernel[2] = {3, 3};
    uint32_t pool_strides[2] = {2, 2};
    int32_t pool_padding[2] = {0, 0};

    /* create pooling memory descriptor on dst descriptor
     *  from previos primitive */
    mkldnn_memory_desc_t pool_src_md = lrn_pd.dst_primitive_desc.memory_desc;

    /* create descriptors for dst pooling data */
    mkldnn_tensor_desc_t pool_dst_tz;
    mkldnn_memory_desc_t pool_dst_md;
    CHECK(mkldnn_tensor_desc_init(&pool_dst_tz, 4, pool_dst_sizes));
    CHECK(mkldnn_memory_desc_init(&pool_dst_md, &pool_dst_tz, mkldnn_f32,
            mkldnn_any));

    /* create memory for user data */
    mkldnn_primitive_t pool_user_dst_memory;
    init_data_memory(4, pool_dst_sizes, mkldnn_nchw, mkldnn_f32, engine,
        net_dst, &pool_user_dst_memory);

    /* create a pooling */
    mkldnn_pooling_desc_t pool_desc;
    CHECK(mkldnn_pooling_desc_init(&pool_desc, mkldnn_forward,
            mkldnn_pooling_max, &pool_src_md, &pool_dst_md, pool_strides,
            pool_kernel, pool_padding, mkldnn_padding_zero));

    mkldnn_pooling_primitive_desc_t pool_pd;
    CHECK(mkldnn_pooling_primitive_desc_init(&pool_pd, &pool_desc, engine));

    /* create memory for workspace */
    mkldnn_primitive_t pool_indices_memory;
    CHECK(mkldnn_memory_create(&pool_indices_memory,
            &pool_pd.indices_primitive_desc, NULL));

    mkldnn_primitive_t pool_dst_memory;

    /* create reorder primitives between user data and pooling dsts
     * if required */
    mkldnn_primitive_t pool_reorder_dst, pool_internal_dst_memory;

    CHECK(prepare_reorder(&pool_user_dst_memory,
            &pool_pd.dst_primitive_desc, 0, &pool_internal_dst_memory,
            &pool_reorder_dst));

    mkldnn_primitive_at_t pool_srcs[] = {
        mkldnn_primitive_at(lrn_dst_memory, 0),
        mkldnn_primitive_at(pool_indices_memory, 0)
    };

    pool_dst_memory = pool_internal_dst_memory ? pool_internal_dst_memory
        : pool_user_dst_memory;

    const_mkldnn_primitive_t pool_dsts[] = { pool_dst_memory };

    /* finally create a pooling primitive */
    mkldnn_primitive_t pool;
    CHECK(mkldnn_primitive_create(&pool, &pool_pd, pool_srcs, pool_dsts));

    /* build a simple net */
    uint32_t n = 0;
    mkldnn_primitive_t net[10];

    if (conv_reorder_src) net[n++] = conv_reorder_src;
    if (conv_reorder_weights) net[n++] = conv_reorder_weights;
    net[n++] = conv;
    net[n++] = relu;
    net[n++] = lrn;
    net[n++] = pool;
    if (pool_reorder_dst) net[n++] = pool_reorder_dst;

    mkldnn_stream_t stream;
    CHECK(mkldnn_stream_create(&stream));
    CHECK(mkldnn_stream_submit(stream, n, net, NULL));
    CHECK(mkldnn_stream_wait(stream, n, NULL));

    /* clean-up */
    mkldnn_stream_destroy(stream);

    mkldnn_primitive_destroy(conv_user_src_memory);
    mkldnn_primitive_destroy(conv_user_weights_memory);
    mkldnn_primitive_destroy(conv_user_bias_memory);
    mkldnn_primitive_destroy(conv_internal_src_memory);
    mkldnn_primitive_destroy(conv_internal_weights_memory);
    mkldnn_primitive_destroy(conv_internal_dst_memory);
    mkldnn_primitive_destroy(conv_reorder_src);
    mkldnn_primitive_destroy(conv_reorder_weights);
    mkldnn_primitive_destroy(conv);

    mkldnn_primitive_destroy(relu_dst_memory);
    mkldnn_primitive_destroy(relu);

    mkldnn_primitive_destroy(lrn_scratch_memory);
    mkldnn_primitive_destroy(lrn_dst_memory);
    mkldnn_primitive_destroy(lrn);

    mkldnn_primitive_destroy(pool_user_dst_memory);
    mkldnn_primitive_destroy(pool_internal_dst_memory);
    mkldnn_primitive_destroy(pool_indices_memory);
    mkldnn_primitive_destroy(pool_reorder_dst);
    mkldnn_primitive_destroy(pool);

    return mkldnn_success;
}

int main(int argc, char **argv) {
    mkldnn_status_t result = simple_net();
    printf("%s\n", (result == mkldnn_success) ? "passed" : "failed");
    return result;
}
