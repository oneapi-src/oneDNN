#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#include "mkl_dnn.h"

typedef float real_t;

#define CHECK(f) do { \
    mkl_dnn_status_t s = f; \
    if (s != mkl_dnn_success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

static size_t tensor_size(const mkl_dnn_tensor_desc_t *t)
{
    size_t size = 1;
    for (size_t i = 0; i < t->ndims; ++i)
        size *= t->dims[i];
    return size;
}

static void init_src(uint32_t dim[4], real_t *x)
{
    uint32_t N = dim[0], C = dim[1], H = dim[2], W = dim[3];
    for (uint32_t n = 0; n < N; n += 1)
    for (uint32_t c = 0; c < C; c += 1)
    for (uint32_t h = 2; h+2 <= H; h += 2)
    for (uint32_t w = 2; w+2 <= W; w += 2)
        x[w + W*h + c*W*H + n*W*H*C] = c*n;
}

static int check_dst(uint32_t dim[4], const real_t *x)
{
    int n_errors = 0;
    uint32_t N = dim[0], C = dim[1], H = dim[2], W = dim[3];
    for (uint32_t n = 0; n < N; ++n)
    for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
    {
        if (x[w + W*h + c*W*H + n*W*H*C] != c*n) n_errors += 1;
    }
    return n_errors;
}

static int doit() {
    /* AlexNet: p1
     * {16, 96, 55, 55} -> {16, 96, 27, 27}
     * pad: {0, 0}
     * strides: {2, 2}
     * kernel: {3, 3}
     */

    mkl_dnn_engine_t engine;
    CHECK(mkl_dnn_engine_create(&engine, mkl_dnn_cpu, 0 /* idx */));

    /* first describe user data and create data descriptors for future
    * pooling w/ the specified format -- we do not want to do a reorder */
    uint32_t p1_src_sizes[4] = { 16, 96, 55, 55 };
    mkl_dnn_tensor_desc_t p1_src_tz;
    mkl_dnn_memory_desc_t p1_src_md;
    mkl_dnn_memory_primitive_desc_t p1_src_pd;
    mkl_dnn_primitive_t p1_src;
    CHECK(mkl_dnn_tensor_desc_init(&p1_src_tz, 4, p1_src_sizes));
    CHECK(mkl_dnn_memory_desc_init(&p1_src_md, &p1_src_tz, mkl_dnn_f32, mkl_dnn_nchw));
    CHECK(mkl_dnn_memory_primitive_desc_init(&p1_src_pd, &p1_src_md, engine));
    real_t *src = (real_t*)calloc(tensor_size(&p1_src_md.tensor_desc), sizeof(real_t));
    CHECK(mkl_dnn_memory_create(&p1_src, &p1_src_pd, src));

    uint32_t p1_dst_sizes[4] = { 16, 96, 27, 27 };
    mkl_dnn_tensor_desc_t p1_dst_tz;
    mkl_dnn_memory_desc_t p1_dst_md;
    mkl_dnn_memory_primitive_desc_t p1_dst_pd;
    mkl_dnn_primitive_t p1_dst;
    CHECK(mkl_dnn_tensor_desc_init(&p1_dst_tz, 4, p1_dst_sizes));
    CHECK(mkl_dnn_memory_desc_init(&p1_dst_md, &p1_dst_tz, mkl_dnn_f32, mkl_dnn_nchw));
    CHECK(mkl_dnn_memory_primitive_desc_init(&p1_dst_pd, &p1_dst_md, engine));
    real_t *dst = (real_t*)calloc(tensor_size(&p1_dst_md.tensor_desc), sizeof(real_t));
    CHECK(mkl_dnn_memory_create(&p1_dst, &p1_dst_pd, dst));

    uint32_t strides[] = { 2, 2 };
    uint32_t kernel [] = { 3, 3 };
    int32_t  padding[] = { 0, 0 };
    mkl_dnn_pooling_desc_t p1_desc;
    mkl_dnn_pooling_primitive_desc_t p1_pd;
    CHECK(mkl_dnn_pooling_desc_init(&p1_desc, mkl_dnn_forward, mkl_dnn_pooling_max,
        &p1_src_md, &p1_dst_md, strides, kernel, padding, mkl_dnn_padding_zero));
    CHECK(mkl_dnn_pooling_primitive_desc_init(&p1_pd, &p1_desc, engine));

    mkl_dnn_primitive_t p1_indices;
    CHECK(mkl_dnn_memory_create(&p1_indices, &p1_pd.indices_primitive_desc, NULL));

    /* create a pooling */
    mkl_dnn_primitive_t p1;
    mkl_dnn_primitive_at_t p1_srcs[] = {
        mkl_dnn_primitive_at(p1_src, 0),
        mkl_dnn_primitive_at(p1_indices, 0)
    };
    const_mkl_dnn_primitive_t p1_dsts[] = { p1_dst };

    CHECK(mkl_dnn_primitive_create(&p1, &p1_pd, p1_srcs, p1_dsts));

    assert(mkl_dnn_memory_primitive_desc_equal(&p1_pd.src_primitive_desc, &p1_src_pd));
    assert(mkl_dnn_memory_primitive_desc_equal(&p1_pd.dst_primitive_desc, &p1_dst_pd));

    init_src(p1_src_sizes, src);

    /* let us build a net */
    mkl_dnn_stream_t stream;
    CHECK(mkl_dnn_stream_create(&stream));
    CHECK(mkl_dnn_stream_submit(stream, 1, &p1, NULL));
    CHECK(mkl_dnn_stream_wait(stream, 1, NULL));

    /* clean-up */
    CHECK(mkl_dnn_stream_destroy(stream));
    mkl_dnn_primitive_destroy(p1);
    mkl_dnn_primitive_destroy(p1_src);
    mkl_dnn_primitive_destroy(p1_indices);
    mkl_dnn_primitive_destroy(p1_dst);
    mkl_dnn_engine_destroy(engine);

    int n_errors = check_dst(p1_dst_sizes, dst);

    free(src);
    free(dst);

    return n_errors;
}

int main(int argc, char **argv) {
    int rc = doit();
    return rc;
}
