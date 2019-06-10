/*******************************************************************************
 * Copyright 2018 Intel Corporation
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

#include "rnn/rnn.hpp"
#include <assert.h>
#include <stdlib.h>

namespace rnn {

typedef enum {
    rnn_forward = 0,
    rnn_backward,
} rnn_propagation_t;

typedef enum {
    left2right = 0,
    right2left,
} rnn_iter_direction_t;

typedef enum {
    bottom2top = 0,
    top2bottom,
} rnn_layer_direction_t;

typedef enum { action_copy = 0, action_sum, action_concat } rnn_action_t;

mkldnn_status_t init_rnn_fwd_desc( mkldnn_rnn_desc_t *rd, const prb_t *p,
       mkldnn_prop_kind_t prop_kind, mkldnn_memory_desc_t *src_layer_d,
       mkldnn_memory_desc_t *src_iter_d, mkldnn_memory_desc_t *src_iter_c_d,
       mkldnn_memory_desc_t *weights_layer_d,
       mkldnn_memory_desc_t *weights_iter_d, mkldnn_memory_desc_t *bias_d,
       mkldnn_memory_desc_t *dst_layer_d, mkldnn_memory_desc_t *dst_iter_d,
       mkldnn_memory_desc_t *dst_iter_c_d);

mkldnn_status_t init_rnn_bwd_desc( mkldnn_rnn_desc_t *rd, const prb_t *p,
       mkldnn_prop_kind_t prop_kind, mkldnn_memory_desc_t *src_layer_d,
       mkldnn_memory_desc_t *src_iter_d, mkldnn_memory_desc_t *src_iter_c_d,
       mkldnn_memory_desc_t *weights_layer_d,
       mkldnn_memory_desc_t *weights_iter_d, mkldnn_memory_desc_t *bias_d,
       mkldnn_memory_desc_t *dst_layer_d, mkldnn_memory_desc_t *dst_iter_d,
       mkldnn_memory_desc_t *dst_iter_c_d,
       mkldnn_memory_desc_t *diff_src_layer_d,
       mkldnn_memory_desc_t *diff_src_iter_d,
       mkldnn_memory_desc_t *diff_src_iter_c_d,
       mkldnn_memory_desc_t *diff_weights_layer_d,
       mkldnn_memory_desc_t *diff_weights_iter_d,
       mkldnn_memory_desc_t *diff_bias_d,
       mkldnn_memory_desc_t *diff_dst_layer_d,
       mkldnn_memory_desc_t *diff_dst_iter_d,
       mkldnn_memory_desc_t *diff_dst_iter_c_d
    );

void init_buffer(float *buf, int64_t size, float value);

float logistic(float x);
float dlogistic(float x);
float relu(float x);
float drelu(float x);
float dtanhf(float x);
float one_m_square(float x);
float x_m_square(float x);

int compare_dat(const prb_t *p, rnn_data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare);

int compare_input(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare);
int compare_states(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare);
int compare_weights_input(const prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare);
int compare_weights_states(const prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare);
int compare_bias(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare);
int compare_dst_last_layer(const prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare);
int compare_dst_last_iteration(const prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare);
int compare_dst_c_last_iteration(const prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare);
};
