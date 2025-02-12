/*******************************************************************************
* Copyright 2025 Intel Corporation
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

__kernel void gen_index(__global int *dst, int axis) {
    long id = get_global_id(0);
    long result, offset = 0;
    long idx;

    idx = id % D0;
    id = id / D0;
    offset += idx * S0;
    if (axis == 0) result = idx;

    idx = id % D1;
    id = id / D1;
    offset += idx * S1;
    if (axis == 1) result = idx;

    idx = id % D2;
    id = id / D2;
    offset += idx * S2;
    if (axis == 2) result = idx;

    idx = id % D3;
    id = id / D3;
    offset += idx * S3;
    if (axis == 3) result = idx;

    idx = id % D4;
    id = id / D4;
    offset += idx * S4;
    if (axis == 4) result = idx;

    idx = id % D5;
    id = id / D5;
    offset += idx * S5;
    if (axis == 5) result = idx;

    dst[offset] = result;
}
