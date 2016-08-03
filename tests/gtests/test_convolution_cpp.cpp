#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#include "gtest/gtest.h"
#include "mkl_dnn.hpp"

typedef float real_t;

struct test_convolution_descr_t {
   uint32_t mb;
   uint32_t ng;
   uint32_t ic, ih, iw;
   uint32_t oc, oh, ow;
   uint32_t kh, kw;
   int32_t padh, padw;
   uint32_t strh, strw;
};

static uint32_t doFill(size_t index, double sparsity) {
    const size_t group_size = (size_t)(1. / sparsity);
    const size_t group = index / group_size;
    const size_t in_group = index % group_size;
    return in_group == ((group % 1637) % group_size);
}

static void fillData(const uint32_t size, real_t* data, double sparsity = 1.) {
#   pragma omp parallel for
    for (uint32_t n = 0; n < size; n++){
        data[n] = doFill(n, sparsity) ? 1. + 2e-1*sin(n%37) : 0;
    }
}

void compute_ref_conv_fwd(test_convolution_descr_t c, real_t *in, real_t *filt, real_t *bias, real_t *out)
{
#pragma omp parallel for collapse(5)
  for (uint32_t n = 0; n < c.mb; n++) {
    for (uint32_t g = 0; g < c.ng; g++) {
      for (uint32_t oc = 0; oc < c.oc/c.ng; oc++) {
        for (uint32_t oh = 0; oh < c.oh; oh++) {
          for (uint32_t ow = 0; ow < c.ow; ow++) {
            uint32_t oidx = n*c.oc*c.oh*c.ow + g*c.oc/c.ng*c.oh*c.ow +
                              oc*c.oh*c.ow + oh*c.ow + ow;
            out[oidx] = bias ? bias[g*c.oc/c.ng + oc] : 0.0;
            for (uint32_t ic = 0; ic < c.ic/c.ng; ic++) {
              for (uint32_t kh = 0;  kh < c.kh; kh++) {
                for (uint32_t kw = 0; kw < c.kw; kw++) {
                  int32_t iw = ow * c.strw - c.padw + kw;
                  int32_t ih = oh * c.strh - c.padh + kh;
                  if (iw < 0 || iw >= (int32_t)c.iw || 
                      ih < 0 || ih >= (int32_t)c.ih)
                    continue;
                  uint32_t iidx = n*c.ic*c.ih*c.iw + g*c.ic/c.ng*c.ih*c.iw +
                              ic*c.ih*c.ow + ih*c.iw + iw;
                  uint32_t fidx = g*c.oc/c.ng*c.ic/c.ng*c.kh*c.kw + oc*c.ic/c.ng*c.kh*c.kw + 
                         ic*c.kh*c.kw + kh*c.kw + kw;

                  out[oidx] += in[iidx] * filt[fidx];
                }
              }
            }
          }
        }
      }
    }
  }
}

int doit_nchw(test_convolution_descr_t cd, bool lazy) {
    using namespace mkl_dnn;

    printf("There are %zu CPU engines\n", engine::get_count(engine::cpu));
    auto cpu_engine = engine(lazy ? engine::cpu_lazy : engine::cpu, 0);

    real_t *input_data = (real_t*)calloc(cd.mb*cd.ic*cd.ih*cd.iw, sizeof(real_t));
    fillData(cd.mb*cd.ic*cd.ih*cd.iw, input_data);
    real_t *weights_data = (real_t*)calloc(cd.ng*(cd.oc/cd.ng)*(cd.ic/cd.ng)*cd.kh*cd.kw, sizeof(real_t));
    fillData(cd.ng*(cd.oc/cd.ng)*(cd.ic/cd.ng)*cd.kh*cd.kw, weights_data);
    real_t *bias_data = (real_t*)calloc(cd.oc, sizeof(real_t));
    //fillData(cd.ng*(cd.oc, bias_data);
    real_t *output_data = (real_t*)calloc(cd.mb*cd.oc*cd.oh*cd.ow, sizeof(real_t));
    //fillData(cd.mb*cd.oc*cd.oh*cd.ow, output_data);
    real_t *output_ref_data = (real_t*)calloc(cd.mb*cd.oc*cd.oh*cd.ow, sizeof(real_t));
    //fillData(cd.mb*cd.oc*cd.oh*cd.ow, output_ref_data);

    auto c_input_desc = memory::desc({1, 1, 2, {cd.mb, cd.ic, cd.ih, cd.iw}}, memory::precision::f32, memory::format::nchw);
    auto c_weights_desc = cd.ng > 1 ?
         memory::desc({1, 2, 2, {cd.ng, cd.oc/cd.ng, cd.ic/cd.ng, cd.kh, cd.kw}}, memory::precision::f32, memory::format::goihw) :
         memory::desc({0, 2, 2, {cd.oc, cd.ic, cd.kh, cd.kw}}, memory::precision::f32, memory::format::oihw);
    auto c_bias_desc = memory::desc({0, 0, 1, {cd.oc}}, memory::precision::f32, memory::format::n);
    auto c_output_desc = memory::desc({1, 1, 2, {cd.mb, cd.oc, cd.oh, cd.ow}}, memory::precision::f32, memory::format::nchw);

    auto c_input = memory({c_input_desc, cpu_engine}, input_data);
    auto c_weights = memory({c_weights_desc, cpu_engine}, weights_data);
    auto c_bias = memory({c_bias_desc, cpu_engine}, bias_data);
    auto c_output = memory({c_output_desc, cpu_engine}, output_data);

    auto c = convolution(prop_kind::forward, convolution::direct,
            c_input, c_weights, c_bias, c_output,
            {cd.strh, cd.strw}, {cd.padh, cd.padw}, padding_kind::zero);

    stream().submit({c}).wait();

    compute_ref_conv_fwd(cd, input_data, weights_data, bias_data, output_ref_data);

#pragma omp parallel for    
    for (uint32_t i = 0; i < cd.mb*cd.oc*cd.oh*cd.ow; ++i) {
        EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-4);
    }

    free(input_data);
    free(output_data);
    free(weights_data);
    free(bias_data);
    free(output_ref_data);
    return 0;
}

TEST(convolution_test, simple_test) {
    doit_nchw({2, 1, 4, 4, 4, 6, 4, 4, 3, 3, 1, 1, 1, 1}, 1);
     
}

TEST(convolution_test, AlexNet_test) {
    uint32_t mb = 2;
//    doit_nchw({mb, 1, 3, 227, 227, 96, 55, 55, 11, 11, 4, 4, 1, 1}, 1);
    //         mb  g    ic ih  iw   oc  oh  ow  kh  kw  padh  padw  strh  strw
    doit_nchw({mb, 2, 96 , 27, 27, 256, 27, 27,  5,  5,    2,   2,     1,    1}, 1);
    doit_nchw({mb, 1, 256, 13, 13, 384, 13, 13,  3,  3,    1,   1,     1,    1}, 1);
    doit_nchw({mb, 2, 384, 13, 13, 384, 13, 13,  3,  3,    1,   1,     1,    1}, 1);
    doit_nchw({mb, 2, 384, 13, 13, 256, 13, 13,  3,  3,    1,   1,     1,    1}, 1);

     
}
