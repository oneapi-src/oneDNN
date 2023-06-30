/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_OCL_CONCAT_COMMON_H
#define GPU_OCL_CONCAT_COMMON_H

#define REDUCE_STAGE_0(cat, f)
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

#endif
