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

__kernel void gen_index(__global float *src,  __global int *dst) {
        int id = get_global_id(0);
        int result, offset = 0;
        int idx;

        idx = id % D0;
        id = id / D0;
        offset += idx * S0;
#if AXIS == 0
        result = idx;
#endif

        idx = id % D1;
        id = id / D1;
        offset += idx * S1;
#if AXIS == 1
        result = idx;
#endif       

        idx = id % D2;
        id = id / D2;
        offset += idx * S2;
#if AXIS == 2
        result = idx;
#endif

        idx = id % D3;
        id = id / D3;
        offset += idx * S3;
#if AXIS == 3
        result = idx;
#endif

        idx = id % D4;
        id = id / D4;
        offset += idx * S4;
#if AXIS == 4
        result = idx;
#endif

        idx = id % D5;
        id = id / D5;
        offset += idx * S5;
#if AXIS == 5
        result = idx;
#endif

        dst[offset] = result;
}
