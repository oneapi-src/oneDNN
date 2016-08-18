/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef XBYAK_UTILS_FOR_MKLDNN
#define XBYAK_UTILS_FOR_MKLDNN

#define XBYAK64
#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"

#define XBYAK_VERSION 0x5000

#if XBYAK_VERSION >= 0x5000
    #define ZWORD   zword
    #define ZWORD_b zword_b
    #define YWORD   yword
    #define YWORD_b yzword_b
#else
    #define ZWORD zmmword
    #define YWORD ymmword
#endif

#ifdef XBYAK64
namespace Xbyak { namespace util {
static const Operand::Code reg_to_preserve[] = {
    Operand::RBX, Operand::RSP, Operand::RBP,
    Operand::R12, Operand::R13, Operand::R14, Operand::R15,
#ifdef _WIN
    Operand::RDI, Operand::RSI,
#endif
};
#ifdef _WIN
static const Reg64 cdecl_param1(Operand::RCX), cdecl_param2(Operand::RDX),
                                cdecl_param3(Operand::R8), cdecl_param4(Operand::R9);
#else
static const Reg64 cdecl_param1(Operand::RDI), cdecl_param2(Operand::RSI),
                                cdecl_param3(Operand::RDX), cdecl_param4(Operand::RCX);
#endif
}}
#endif

#endif
