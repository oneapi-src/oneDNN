/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gpu/ocl/ocl_types.h"

#define INIT_N_INPUTS(i) \
    if (i < N_INPUTS) inputs[i] = input##i;
#define INIT_MAX_N_INPUTS(i) \
    if (i < MAX_N_INPUTS) inputs[i] = input##i;

float get_value(__global SRC_DATA_T *src, ptrdiff_t offset) {
    if (offset >= N_ELEMS) return 0;
    return CONVERT_FLOAT_T(src[offset]);
}
#define many_inputs_sum_impl(inputs, output, scales, num_inputs, local_val) \
    const uint group_id = get_group_id(0); \
    const uint group_size = get_local_size(0); \
    const uint gid = get_global_id(0); \
    ptrdiff_t offset = gid / num_inputs; \
    const int tensor_idx = gid % num_inputs; \
    const int local_id = get_local_id(0); \
    local_val[local_id] \
            = get_value(inputs[tensor_idx], offset) * scales[tensor_idx]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    if (tensor_idx == 0 && offset < N_ELEMS) { \
        float final_val = 0; \
        for (int i = 0; i < num_inputs; i++) { \
            final_val += local_val[local_id + i]; \
        } \
        output[offset] = TO_DST(final_val + CONVERT_FLOAT_T(output[offset])); \
    }

__kernel void many_inputs_sum(__global SRC_DATA_T *input0,
        __global SRC_DATA_T *input1, __global SRC_DATA_T *input2,
        __global SRC_DATA_T *input3, __global SRC_DATA_T *input4,
        __global SRC_DATA_T *input5, __global SRC_DATA_T *input6,
        __global SRC_DATA_T *input7, __global SRC_DATA_T *input8,
        __global SRC_DATA_T *input9, __global SRC_DATA_T *input10,
        __global SRC_DATA_T *input11, __global SRC_DATA_T *input12,
        __global SRC_DATA_T *input13, __global SRC_DATA_T *input14,
        __global SRC_DATA_T *input15, __global SRC_DATA_T *input16,
        __global SRC_DATA_T *input17, __global SRC_DATA_T *input18,
        __global SRC_DATA_T *input19, __global SRC_DATA_T *input20,
        __global SRC_DATA_T *input21, __global SRC_DATA_T *input22,
        __global SRC_DATA_T *input23, __global SRC_DATA_T *input24,
        __global SRC_DATA_T *input25, __global SRC_DATA_T *input26,
        __global SRC_DATA_T *input27, __global SRC_DATA_T *input28,
        __global SRC_DATA_T *input29, __global SRC_DATA_T *input30,
        __global SRC_DATA_T *input31, __global SRC_DATA_T *input32,
        __global SRC_DATA_T *input33, __global SRC_DATA_T *input34,
        __global SRC_DATA_T *input35, __global SRC_DATA_T *input36,
        __global SRC_DATA_T *input37, __global SRC_DATA_T *input38,
        __global SRC_DATA_T *input39, __global SRC_DATA_T *input40,
        __global SRC_DATA_T *input41, __global SRC_DATA_T *input42,
        __global SRC_DATA_T *input43, __global SRC_DATA_T *input44,
        __global SRC_DATA_T *input45, __global SRC_DATA_T *input46,
        __global SRC_DATA_T *input47, __global SRC_DATA_T *input48,
        __global SRC_DATA_T *input49, __global SRC_DATA_T *input50,
        __global SRC_DATA_T *input51, __global SRC_DATA_T *input52,
        __global SRC_DATA_T *input53, __global SRC_DATA_T *input54,
        __global SRC_DATA_T *input55, __global SRC_DATA_T *input56,
        __global SRC_DATA_T *input57, __global SRC_DATA_T *input58,
        __global SRC_DATA_T *input59, __global SRC_DATA_T *input60,
        __global SRC_DATA_T *input61, __global SRC_DATA_T *input62,
        __global SRC_DATA_T *input63, __global SRC_DATA_T *input64,
        __global SRC_DATA_T *input65, __global SRC_DATA_T *input66,
        __global SRC_DATA_T *input67, __global SRC_DATA_T *input68,
        __global SRC_DATA_T *input69, __global SRC_DATA_T *input70,
        __global SRC_DATA_T *input71, __global SRC_DATA_T *input72,
        __global SRC_DATA_T *input73, __global SRC_DATA_T *input74,
        __global SRC_DATA_T *input75, __global SRC_DATA_T *input76,
        __global SRC_DATA_T *input77, __global SRC_DATA_T *input78,
        __global SRC_DATA_T *input79, __global SRC_DATA_T *input80,
        __global SRC_DATA_T *input81, __global SRC_DATA_T *input82,
        __global SRC_DATA_T *input83, __global SRC_DATA_T *input84,
        __global SRC_DATA_T *input85, __global SRC_DATA_T *input86,
        __global SRC_DATA_T *input87, __global SRC_DATA_T *input88,
        __global SRC_DATA_T *input89, __global SRC_DATA_T *input90,
        __global SRC_DATA_T *input91, __global SRC_DATA_T *input92,
        __global SRC_DATA_T *input93, __global DST_DATA_T *output,
        __global float *scales) {

    __local float local_val[256];
    __global SRC_DATA_T *inputs[94];

    INIT_N_INPUTS(0);
    INIT_N_INPUTS(1);
    INIT_N_INPUTS(2);
    INIT_N_INPUTS(3);
    INIT_N_INPUTS(4);
    INIT_N_INPUTS(5);
    INIT_N_INPUTS(6);
    INIT_N_INPUTS(7);
    INIT_N_INPUTS(8);
    INIT_N_INPUTS(9);
    INIT_N_INPUTS(10);
    INIT_N_INPUTS(11);
    INIT_N_INPUTS(12);
    INIT_N_INPUTS(13);
    INIT_N_INPUTS(14);
    INIT_N_INPUTS(15);
    INIT_N_INPUTS(16);
    INIT_N_INPUTS(17);
    INIT_N_INPUTS(18);
    INIT_N_INPUTS(19);
    INIT_N_INPUTS(20);
    INIT_N_INPUTS(21);
    INIT_N_INPUTS(22);
    INIT_N_INPUTS(23);
    INIT_N_INPUTS(24);
    INIT_N_INPUTS(25);
    INIT_N_INPUTS(26);
    INIT_N_INPUTS(27);
    INIT_N_INPUTS(28);
    INIT_N_INPUTS(29);
    INIT_N_INPUTS(30);
    INIT_N_INPUTS(31);
    INIT_N_INPUTS(32);
    INIT_N_INPUTS(33);
    INIT_N_INPUTS(34);
    INIT_N_INPUTS(35);
    INIT_N_INPUTS(36);
    INIT_N_INPUTS(37);
    INIT_N_INPUTS(38);
    INIT_N_INPUTS(39);
    INIT_N_INPUTS(40);
    INIT_N_INPUTS(41);
    INIT_N_INPUTS(42);
    INIT_N_INPUTS(43);
    INIT_N_INPUTS(44);
    INIT_N_INPUTS(45);
    INIT_N_INPUTS(46);
    INIT_N_INPUTS(47);
    INIT_N_INPUTS(48);
    INIT_N_INPUTS(49);
    INIT_N_INPUTS(50);
    INIT_N_INPUTS(51);
    INIT_N_INPUTS(52);
    INIT_N_INPUTS(53);
    INIT_N_INPUTS(54);
    INIT_N_INPUTS(55);
    INIT_N_INPUTS(56);
    INIT_N_INPUTS(57);
    INIT_N_INPUTS(58);
    INIT_N_INPUTS(59);
    INIT_N_INPUTS(60);
    INIT_N_INPUTS(61);
    INIT_N_INPUTS(62);
    INIT_N_INPUTS(63);
    INIT_N_INPUTS(64);
    INIT_N_INPUTS(65);
    INIT_N_INPUTS(66);
    INIT_N_INPUTS(67);
    INIT_N_INPUTS(68);
    INIT_N_INPUTS(69);
    INIT_N_INPUTS(70);
    INIT_N_INPUTS(71);
    INIT_N_INPUTS(72);
    INIT_N_INPUTS(73);
    INIT_N_INPUTS(74);
    INIT_N_INPUTS(75);
    INIT_N_INPUTS(76);
    INIT_N_INPUTS(77);
    INIT_N_INPUTS(78);
    INIT_N_INPUTS(79);
    INIT_N_INPUTS(80);
    INIT_N_INPUTS(81);
    INIT_N_INPUTS(82);
    INIT_N_INPUTS(83);
    INIT_N_INPUTS(84);
    INIT_N_INPUTS(85);
    INIT_N_INPUTS(86);
    INIT_N_INPUTS(87);
    INIT_N_INPUTS(88);
    INIT_N_INPUTS(89);
    INIT_N_INPUTS(90);
    INIT_N_INPUTS(91);
    INIT_N_INPUTS(92);
    INIT_N_INPUTS(93);
    many_inputs_sum_impl(inputs, output, scales, N_INPUTS, local_val);
}

__kernel void many_inputs_sum_batched(__global SRC_DATA_T *input0,
        __global SRC_DATA_T *input1, __global SRC_DATA_T *input2,
        __global SRC_DATA_T *input3, __global SRC_DATA_T *input4,
        __global SRC_DATA_T *input5, __global SRC_DATA_T *input6,
        __global SRC_DATA_T *input7, __global SRC_DATA_T *input8,
        __global SRC_DATA_T *input9, __global SRC_DATA_T *input10,
        __global SRC_DATA_T *input11, __global SRC_DATA_T *input12,
        __global SRC_DATA_T *input13, __global SRC_DATA_T *input14,
        __global SRC_DATA_T *input15, __global SRC_DATA_T *input16,
        __global SRC_DATA_T *input17, __global SRC_DATA_T *input18,
        __global SRC_DATA_T *input19, __global SRC_DATA_T *input20,
        __global SRC_DATA_T *input21, __global SRC_DATA_T *input22,
        __global SRC_DATA_T *input23, __global SRC_DATA_T *input24,
        __global SRC_DATA_T *input25, __global SRC_DATA_T *input26,
        __global SRC_DATA_T *input27, __global SRC_DATA_T *input28,
        __global SRC_DATA_T *input29, __global SRC_DATA_T *input30,
        __global SRC_DATA_T *input31, __global SRC_DATA_T *input32,
        __global SRC_DATA_T *input33, __global SRC_DATA_T *input34,
        __global SRC_DATA_T *input35, __global SRC_DATA_T *input36,
        __global SRC_DATA_T *input37, __global SRC_DATA_T *input38,
        __global SRC_DATA_T *input39, __global SRC_DATA_T *input40,
        __global SRC_DATA_T *input41, __global SRC_DATA_T *input42,
        __global SRC_DATA_T *input43, __global SRC_DATA_T *input44,
        __global SRC_DATA_T *input45, __global SRC_DATA_T *input46,
        __global SRC_DATA_T *input47, __global SRC_DATA_T *input48,
        __global SRC_DATA_T *input49, __global SRC_DATA_T *input50,
        __global SRC_DATA_T *input51, __global SRC_DATA_T *input52,
        __global SRC_DATA_T *input53, __global SRC_DATA_T *input54,
        __global SRC_DATA_T *input55, __global SRC_DATA_T *input56,
        __global SRC_DATA_T *input57, __global SRC_DATA_T *input58,
        __global SRC_DATA_T *input59, __global SRC_DATA_T *input60,
        __global SRC_DATA_T *input61, __global SRC_DATA_T *input62,
        __global SRC_DATA_T *input63, __global SRC_DATA_T *input64,
        __global SRC_DATA_T *input65, __global SRC_DATA_T *input66,
        __global SRC_DATA_T *input67, __global SRC_DATA_T *input68,
        __global SRC_DATA_T *input69, __global SRC_DATA_T *input70,
        __global SRC_DATA_T *input71, __global SRC_DATA_T *input72,
        __global SRC_DATA_T *input73, __global SRC_DATA_T *input74,
        __global SRC_DATA_T *input75, __global SRC_DATA_T *input76,
        __global SRC_DATA_T *input77, __global SRC_DATA_T *input78,
        __global SRC_DATA_T *input79, __global SRC_DATA_T *input80,
        __global SRC_DATA_T *input81, __global SRC_DATA_T *input82,
        __global SRC_DATA_T *input83, __global SRC_DATA_T *input84,
        __global SRC_DATA_T *input85, __global SRC_DATA_T *input86,
        __global SRC_DATA_T *input87, __global SRC_DATA_T *input88,
        __global SRC_DATA_T *input89, __global SRC_DATA_T *input90,
        __global SRC_DATA_T *input91, __global SRC_DATA_T *input92,
        __global SRC_DATA_T *input93, __global DST_DATA_T *output,
        __global float *scales) {

    __local float local_val[256];
    __global SRC_DATA_T *inputs[94];

    INIT_MAX_N_INPUTS(0);
    INIT_MAX_N_INPUTS(1);
    INIT_MAX_N_INPUTS(2);
    INIT_MAX_N_INPUTS(3);
    INIT_MAX_N_INPUTS(4);
    INIT_MAX_N_INPUTS(5);
    INIT_MAX_N_INPUTS(6);
    INIT_MAX_N_INPUTS(7);
    INIT_MAX_N_INPUTS(8);
    INIT_MAX_N_INPUTS(9);
    INIT_MAX_N_INPUTS(10);
    INIT_MAX_N_INPUTS(11);
    INIT_MAX_N_INPUTS(12);
    INIT_MAX_N_INPUTS(13);
    INIT_MAX_N_INPUTS(14);
    INIT_MAX_N_INPUTS(15);
    INIT_MAX_N_INPUTS(16);
    INIT_MAX_N_INPUTS(17);
    INIT_MAX_N_INPUTS(18);
    INIT_MAX_N_INPUTS(19);
    INIT_MAX_N_INPUTS(20);
    INIT_MAX_N_INPUTS(21);
    INIT_MAX_N_INPUTS(22);
    INIT_MAX_N_INPUTS(23);
    INIT_MAX_N_INPUTS(24);
    INIT_MAX_N_INPUTS(25);
    INIT_MAX_N_INPUTS(26);
    INIT_MAX_N_INPUTS(27);
    INIT_MAX_N_INPUTS(28);
    INIT_MAX_N_INPUTS(29);
    INIT_MAX_N_INPUTS(30);
    INIT_MAX_N_INPUTS(31);
    INIT_MAX_N_INPUTS(32);
    INIT_MAX_N_INPUTS(33);
    INIT_MAX_N_INPUTS(34);
    INIT_MAX_N_INPUTS(35);
    INIT_MAX_N_INPUTS(36);
    INIT_MAX_N_INPUTS(37);
    INIT_MAX_N_INPUTS(38);
    INIT_MAX_N_INPUTS(39);
    INIT_MAX_N_INPUTS(40);
    INIT_MAX_N_INPUTS(41);
    INIT_MAX_N_INPUTS(42);
    INIT_MAX_N_INPUTS(43);
    INIT_MAX_N_INPUTS(44);
    INIT_MAX_N_INPUTS(45);
    INIT_MAX_N_INPUTS(46);
    INIT_MAX_N_INPUTS(47);
    INIT_MAX_N_INPUTS(48);
    INIT_MAX_N_INPUTS(49);
    INIT_MAX_N_INPUTS(50);
    INIT_MAX_N_INPUTS(51);
    INIT_MAX_N_INPUTS(52);
    INIT_MAX_N_INPUTS(53);
    INIT_MAX_N_INPUTS(54);
    INIT_MAX_N_INPUTS(55);
    INIT_MAX_N_INPUTS(56);
    INIT_MAX_N_INPUTS(57);
    INIT_MAX_N_INPUTS(58);
    INIT_MAX_N_INPUTS(59);
    INIT_MAX_N_INPUTS(60);
    INIT_MAX_N_INPUTS(61);
    INIT_MAX_N_INPUTS(62);
    INIT_MAX_N_INPUTS(63);
    INIT_MAX_N_INPUTS(64);
    INIT_MAX_N_INPUTS(65);
    INIT_MAX_N_INPUTS(66);
    INIT_MAX_N_INPUTS(67);
    INIT_MAX_N_INPUTS(68);
    INIT_MAX_N_INPUTS(69);
    INIT_MAX_N_INPUTS(70);
    INIT_MAX_N_INPUTS(71);
    INIT_MAX_N_INPUTS(72);
    INIT_MAX_N_INPUTS(73);
    INIT_MAX_N_INPUTS(74);
    INIT_MAX_N_INPUTS(75);
    INIT_MAX_N_INPUTS(76);
    INIT_MAX_N_INPUTS(77);
    INIT_MAX_N_INPUTS(78);
    INIT_MAX_N_INPUTS(79);
    INIT_MAX_N_INPUTS(80);
    INIT_MAX_N_INPUTS(81);
    INIT_MAX_N_INPUTS(82);
    INIT_MAX_N_INPUTS(83);
    INIT_MAX_N_INPUTS(84);
    INIT_MAX_N_INPUTS(85);
    INIT_MAX_N_INPUTS(86);
    INIT_MAX_N_INPUTS(87);
    INIT_MAX_N_INPUTS(88);
    INIT_MAX_N_INPUTS(89);
    INIT_MAX_N_INPUTS(90);
    INIT_MAX_N_INPUTS(91);
    INIT_MAX_N_INPUTS(92);
    INIT_MAX_N_INPUTS(93);
    many_inputs_sum_impl(inputs, output, scales, MAX_N_INPUTS, local_val);
}
