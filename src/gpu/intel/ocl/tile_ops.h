/*******************************************************************************
 * Copyright 2024 Intel Corporation
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

#ifndef GPU_OCL_TILE_OPS_H
#define GPU_OCL_TILE_OPS_H

#define tile_elementwise(t, f) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t.x[i] \
                = f(t.x[i]); \
    } while (0)

#define tile_copy(t, t_new) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t_new.x[i] \
                = __builtin_convertvector(t.x[i], __typeof__(t_new.x[i])); \
    } while (0)

#define DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_store(tile_type t, \
            global element_type *ptr, int ld, int m, int n, int offset_r, \
            int offset_c) { \
        ptr += m * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = i0 + get_sub_group_local_id(); \
                    if (offset_r + i < m) \
                        ptr[i] = t.x[i0 / br + nbr * (j / bc)] \
                                    [(i0 % br) / sg + (j % bc) * (br / sg)]; \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store(tile_type t, \
            global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_store(t, ptr, m, m, n, offset_r, offset_c); \
    } \
    __attribute__((overloadable)) void tile_store_full(tile_type t, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                ptr[i] = t.x[i0 / br + nbr * (j / bc)] \
                            [(i0 % br) / sg + (j % bc) * (br / sg)]; \
            } \
        } \
    }

#define DECLARE_2D_TILE(tile_type, element_type, sg, br, bc, nbr, nbc) \
    typedef element_type __attribute__((ext_vector_type(br * bc / sg))) \
            _e_##tile_type; \
    typedef struct { \
        _e_##tile_type x[nbr * nbc]; \
    } tile_type; \
    DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc)

#endif