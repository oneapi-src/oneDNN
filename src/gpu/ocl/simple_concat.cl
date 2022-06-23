/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
#if DATA_TYPE_SIZE == 4
#define DATA_T uint
#define DATA2_T uint2
#define DATA4_T uint4
#define DATA8_T uint8
#define BLOCK_READ intel_sub_group_block_read
#define BLOCK_WRITE intel_sub_group_block_write
#define BLOCK_READ2 intel_sub_group_block_read2
#define BLOCK_WRITE2 intel_sub_group_block_write2
#define BLOCK_READ4 intel_sub_group_block_read4
#define BLOCK_WRITE4 intel_sub_group_block_write4
#define BLOCK_READ8 intel_sub_group_block_read8
#define BLOCK_WRITE8 intel_sub_group_block_write8
#elif DATA_TYPE_SIZE == 2
#define DATA_T ushort
#define DATA2_T ushort2
#define DATA4_T ushort4
#define DATA8_T ushort8
#define BLOCK_READ intel_sub_group_block_read_us
#define BLOCK_WRITE intel_sub_group_block_write_us
#define BLOCK_READ2 intel_sub_group_block_read_us2
#define BLOCK_WRITE2 intel_sub_group_block_write_us2
#define BLOCK_READ4 intel_sub_group_block_read_us4
#define BLOCK_WRITE4 intel_sub_group_block_write_us4
#define BLOCK_READ8 intel_sub_group_block_read_us8
#define BLOCK_WRITE8 intel_sub_group_block_write_us8
#elif DATA_TYPE_SIZE == 1
#define DATA_T uchar
#define DATA2_T uchar2
#define DATA4_T uchar4
#define DATA8_T uchar8
#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ2 intel_sub_group_block_read_uc2
#define BLOCK_WRITE2 intel_sub_group_block_write_uc2
#define BLOCK_READ4 intel_sub_group_block_read_uc4
#define BLOCK_WRITE4 intel_sub_group_block_write_uc4
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
#endif

#define REDUCE_STAGE_1(cat, f) f(0)
#define REDUCE_STAGE_2(cat, f) cat(REDUCE_STAGE_1(cat, f), f(1))
#define REDUCE_STAGE_3(cat, f) cat(REDUCE_STAGE_2(cat, f), f(2))
#define REDUCE_STAGE_4(cat, f) cat(REDUCE_STAGE_3(cat, f), f(3))
#define REDUCE_STAGE_5(cat, f) cat(REDUCE_STAGE_4(cat, f), f(4))
#define REDUCE_STAGE_6(cat, f) cat(REDUCE_STAGE_5(cat, f), f(5))
#define REDUCE_STAGE_7(cat, f) cat(REDUCE_STAGE_6(cat, f), f(6))
#define REDUCE_STAGE_8(cat, f) cat(REDUCE_STAGE_7(cat, f), f(7))
#define REDUCE_STAGE_9(cat, f) cat(REDUCE_STAGE_8(cat, f), f(8))
#define REDUCE_STAGE_10(cat, f) cat(REDUCE_STAGE_9(cat, f), f(9))
#define REDUCE_STAGE_11(cat, f) cat(REDUCE_STAGE_10(cat, f), f(10))
#define REDUCE_STAGE_12(cat, f) cat(REDUCE_STAGE_11(cat, f), f(11))
#define REDUCE_STAGE_13(cat, f) cat(REDUCE_STAGE_12(cat, f), f(12))
#define REDUCE_STAGE_14(cat, f) cat(REDUCE_STAGE_13(cat, f), f(13))
#define REDUCE_STAGE_15(cat, f) cat(REDUCE_STAGE_14(cat, f), f(14))
#define REDUCE_STAGE_16(cat, f) cat(REDUCE_STAGE_15(cat, f), f(15))
#define REDUCE_STAGE_17(cat, f) cat(REDUCE_STAGE_16(cat, f), f(16))
#define REDUCE_STAGE_18(cat, f) cat(REDUCE_STAGE_17(cat, f), f(17))
#define REDUCE_STAGE_19(cat, f) cat(REDUCE_STAGE_18(cat, f), f(18))
#define REDUCE_STAGE_20(cat, f) cat(REDUCE_STAGE_19(cat, f), f(19))
#define REDUCE_STAGE_21(cat, f) cat(REDUCE_STAGE_20(cat, f), f(20))
#define REDUCE_STAGE_22(cat, f) cat(REDUCE_STAGE_21(cat, f), f(21))
#define REDUCE_STAGE_23(cat, f) cat(REDUCE_STAGE_22(cat, f), f(22))
#define REDUCE_STAGE_24(cat, f) cat(REDUCE_STAGE_23(cat, f), f(23))
#define REDUCE_STAGE_25(cat, f) cat(REDUCE_STAGE_24(cat, f), f(24))
#define REDUCE_STAGE_26(cat, f) cat(REDUCE_STAGE_25(cat, f), f(25))
#define REDUCE_STAGE_27(cat, f) cat(REDUCE_STAGE_26(cat, f), f(26))
#define REDUCE_STAGE_28(cat, f) cat(REDUCE_STAGE_27(cat, f), f(27))
#define REDUCE_STAGE_29(cat, f) cat(REDUCE_STAGE_28(cat, f), f(28))
#define REDUCE_STAGE_30(cat, f) cat(REDUCE_STAGE_29(cat, f), f(29))
#define REDUCE_STAGE_31(cat, f) cat(REDUCE_STAGE_30(cat, f), f(30))
#define REDUCE_STAGE_32(cat, f) cat(REDUCE_STAGE_31(cat, f), f(31))
#define REDUCE_STAGE_33(cat, f) cat(REDUCE_STAGE_32(cat, f), f(32))
#define REDUCE_STAGE_34(cat, f) cat(REDUCE_STAGE_33(cat, f), f(33))
#define REDUCE_STAGE_35(cat, f) cat(REDUCE_STAGE_34(cat, f), f(34))
#define REDUCE_STAGE_36(cat, f) cat(REDUCE_STAGE_35(cat, f), f(35))
#define REDUCE_STAGE_37(cat, f) cat(REDUCE_STAGE_36(cat, f), f(36))
#define REDUCE_STAGE_38(cat, f) cat(REDUCE_STAGE_37(cat, f), f(37))
#define REDUCE_STAGE_39(cat, f) cat(REDUCE_STAGE_38(cat, f), f(38))
#define REDUCE_STAGE_40(cat, f) cat(REDUCE_STAGE_39(cat, f), f(39))
#define REDUCE_STAGE_41(cat, f) cat(REDUCE_STAGE_40(cat, f), f(40))
#define REDUCE_STAGE_42(cat, f) cat(REDUCE_STAGE_41(cat, f), f(41))
#define REDUCE_STAGE_43(cat, f) cat(REDUCE_STAGE_42(cat, f), f(42))
#define REDUCE_STAGE_44(cat, f) cat(REDUCE_STAGE_43(cat, f), f(43))
#define REDUCE_STAGE_45(cat, f) cat(REDUCE_STAGE_44(cat, f), f(44))
#define REDUCE_STAGE_46(cat, f) cat(REDUCE_STAGE_45(cat, f), f(45))
#define REDUCE_STAGE_47(cat, f) cat(REDUCE_STAGE_46(cat, f), f(46))
#define REDUCE_STAGE_48(cat, f) cat(REDUCE_STAGE_47(cat, f), f(47))
#define REDUCE_STAGE_49(cat, f) cat(REDUCE_STAGE_48(cat, f), f(48))
#define REDUCE_STAGE_50(cat, f) cat(REDUCE_STAGE_49(cat, f), f(49))
#define REDUCE_STAGE_51(cat, f) cat(REDUCE_STAGE_50(cat, f), f(50))
#define REDUCE_STAGE_52(cat, f) cat(REDUCE_STAGE_51(cat, f), f(51))
#define REDUCE_STAGE_53(cat, f) cat(REDUCE_STAGE_52(cat, f), f(52))
#define REDUCE_STAGE_54(cat, f) cat(REDUCE_STAGE_53(cat, f), f(53))
#define REDUCE_STAGE_55(cat, f) cat(REDUCE_STAGE_54(cat, f), f(54))
#define REDUCE_STAGE_56(cat, f) cat(REDUCE_STAGE_55(cat, f), f(55))
#define REDUCE_STAGE_57(cat, f) cat(REDUCE_STAGE_56(cat, f), f(56))
#define REDUCE_STAGE_58(cat, f) cat(REDUCE_STAGE_57(cat, f), f(57))
#define REDUCE_STAGE_59(cat, f) cat(REDUCE_STAGE_58(cat, f), f(58))
#define REDUCE_STAGE_60(cat, f) cat(REDUCE_STAGE_59(cat, f), f(59))
#define REDUCE_STAGE_61(cat, f) cat(REDUCE_STAGE_60(cat, f), f(60))
#define REDUCE_STAGE_62(cat, f) cat(REDUCE_STAGE_61(cat, f), f(61))
#define REDUCE_STAGE_63(cat, f) cat(REDUCE_STAGE_62(cat, f), f(62))
#define REDUCE_STAGE_64(cat, f) cat(REDUCE_STAGE_63(cat, f), f(63))
#define REDUCE2(n, cat, f) REDUCE_STAGE_##n(cat, f)
#define REDUCE(n, cat, f) REDUCE2(n, cat, f)

#define JOIN_COMMA(x, y) x, y
#define SRC_PTR(n) __global const DATA_T *src##n
#define SRC_PTRS REDUCE(N_INPUTS, JOIN_COMMA, SRC_PTR)
#define JOIN_ELSE(x, y) y else x
#define CHECK_AND_GET(n) \
    if (get_global_id(2) >= OFFSET##n) \
        src = src##n + get_global_id(1) * SRC##n##_EXT_OFFSET + x \
                - OFFSET##n * INNER_OFFSET;
#define SET_SRC REDUCE(N_INPUTS, JOIN_ELSE, CHECK_AND_GET)

#if BLOCK != 1
__attribute__((intel_reqd_sub_group_size(SIMD)))
#endif
__kernel void
simple_concat(__global DATA_T *dst, SRC_PTRS) {
    DATA8_T A0, A1, A2, A3;
    DATA_T B;
    DATA2_T C;
    DATA4_T D;
    const size_t x = (get_global_id(0) / SIMD) * BLOCK
            + get_global_id(2) * INNER_OFFSET;
    __global const DATA_T *src;

    SET_SRC;

#if BLOCK == 1
    B = src[0];
#elif BLOCK == SIMD
    B = BLOCK_READ(src);
#elif BLOCK == 2 * SIMD
    C = BLOCK_READ2(src);
#elif BLOCK == 3 * SIMD
    C = BLOCK_READ2(src);
    B = BLOCK_READ(&src[2 * SIMD]);
#elif BLOCK == 4 * SIMD
    D = BLOCK_READ4(src);
#elif BLOCK == 5 * SIMD
    D = BLOCK_READ4(src);
    B = BLOCK_READ(&src[4 * SIMD]);
#elif BLOCK == 6 * SIMD
    D = BLOCK_READ4(src);
    C = BLOCK_READ2(&src[4 * SIMD]);
#elif BLOCK == 7 * SIMD
    B = BLOCK_READ(src);
    C = BLOCK_READ2(&src[SIMD]);
    D = BLOCK_READ4(&src[3 * SIMD]);
#elif BLOCK >= 8 * SIMD
    A0 = BLOCK_READ8(src);
#elif BLOCK >= 16 * SIMD
    A1 = BLOCK_READ8(&src[8 * SIMD]);
#elif BLOCK >= 24 * SIMD
    A2 = BLOCK_READ8(&src[16 * SIMD]);
#elif BLOCK >= 32 * SIMD
    A3 = BLOCK_READ8(&src[24 * SIMD]);
#endif
    dst += get_global_id(1) * DST_EXT_OFFSET + x;
#if BLOCK == 1
    dst[0] = B;
#elif BLOCK == SIMD
    BLOCK_WRITE(dst, B);
#elif BLOCK == 2 * SIMD
    BLOCK_WRITE2(dst, C);
#elif BLOCK == 3 * SIMD
    BLOCK_WRITE2(dst, C);
    BLOCK_WRITE(&dst[2 * SIMD], B);
#elif BLOCK == 4 * SIMD
    BLOCK_WRITE4(dst, D);
#elif BLOCK == 5 * SIMD
    BLOCK_WRITE4(dst, D);
    BLOCK_WRITE(&dst[4 * SIMD], B);
#elif BLOCK == 6 * SIMD
    BLOCK_WRITE4(dst, D);
    BLOCK_WRITE2(&dst[4 * SIMD], C);
#elif BLOCK == 7 * SIMD
    BLOCK_WRITE(dst, B);
    BLOCK_WRITE2(&dst[SIMD], C);
    BLOCK_WRITE4(&dst[3 * SIMD], D);
#elif BLOCK >= 8 * SIMD
    BLOCK_WRITE8(dst, A0);
#elif BLOCK >= 16 * SIMD
    BLOCK_WRITE8(&dst[8 * SIMD], A1);
#elif BLOCK >= 24 * SIMD
    BLOCK_WRITE8(&dst[16 * SIMD], A2);
#elif BLOCK >= 32 * SIMD
    BLOCK_WRITE8(&dst[24 * SIMD], A3);
#endif
}
