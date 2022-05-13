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

#define INIT(i) \
    if (i < N_INPUTS) inputs[i] = input##i;

float get_value(__global SRC_DATA_T *src, ptrdiff_t offset) {
    if (offset >= N_ELEMS) return 0;
    return CONVERT_FLOAT_T(src[offset]);
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

    const uint group_id = get_group_id(0);
    const uint group_size = get_local_size(0);
    const uint gid = get_global_id(0);
    __local float local_val[256];
    ptrdiff_t offset = gid / N_INPUTS;
    const int tensor_idx = gid % N_INPUTS;
    const int local_id = get_local_id(0);
    float final_val = 0;

    __global SRC_DATA_T *inputs[94];

    INIT(0);
    INIT(1);
    INIT(2);
    INIT(3);
    INIT(4);
    INIT(5);
    INIT(6);
    INIT(7);
    INIT(8);
    INIT(9);
    INIT(10);
    INIT(11);
    INIT(12);
    INIT(13);
    INIT(14);
    INIT(15);
    INIT(16);
    INIT(17);
    INIT(18);
    INIT(19);
    INIT(20);
    INIT(21);
    INIT(22);
    INIT(23);
    INIT(24);
    INIT(25);
    INIT(26);
    INIT(27);
    INIT(28);
    INIT(29);
    INIT(30);
    INIT(31);
    INIT(32);
    INIT(33);
    INIT(34);
    INIT(35);
    INIT(36);
    INIT(37);
    INIT(38);
    INIT(39);
    INIT(40);
    INIT(41);
    INIT(42);
    INIT(43);
    INIT(44);
    INIT(45);
    INIT(46);
    INIT(47);
    INIT(48);
    INIT(49);
    INIT(50);
    INIT(51);
    INIT(52);
    INIT(53);
    INIT(54);
    INIT(55);
    INIT(56);
    INIT(57);
    INIT(58);
    INIT(59);
    INIT(60);
    INIT(61);
    INIT(62);
    INIT(63);
    INIT(64);
    INIT(65);
    INIT(66);
    INIT(67);
    INIT(68);
    INIT(69);
    INIT(70);
    INIT(71);
    INIT(72);
    INIT(73);
    INIT(74);
    INIT(75);
    INIT(76);
    INIT(77);
    INIT(78);
    INIT(79);
    INIT(80);
    INIT(81);
    INIT(82);
    INIT(83);
    INIT(84);
    INIT(85);
    INIT(86);
    INIT(87);
    INIT(88);
    INIT(89);
    INIT(90);
    INIT(91);
    INIT(92);
    INIT(93);

    local_val[local_id]
            = get_value(inputs[tensor_idx], offset) * scales[tensor_idx];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (tensor_idx == 0 && offset < N_ELEMS) {
        float final_val = 0;
        for (int i = 0; i < N_INPUTS; i++) {
            final_val += local_val[local_id + i];
        }
        output[offset] = TO_DST(final_val);
    }
}
