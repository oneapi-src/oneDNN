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
/*******************************************************************************
* Copyright (c) 2007 MITSUNARI Shigeo
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
* Neither the name of the copyright owner nor the names of its contributors may
* be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

const char *getVersionString() const { return "5.00"; }
void packssdw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x6B); }
void packsswb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x63); }
void packuswb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x67); }
void pand(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDB); }
void pandn(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDF); }
void pmaddwd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF5); }
void pmulhuw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE4); }
void pmulhw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE5); }
void pmullw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD5); }
void por(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEB); }
void punpckhbw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x68); }
void punpckhwd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x69); }
void punpckhdq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x6A); }
void punpcklbw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x60); }
void punpcklwd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x61); }
void punpckldq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x62); }
void pxor(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEF); }
void pavgb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE0); }
void pavgw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE3); }
void pmaxsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEE); }
void pmaxub(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDE); }
void pminsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEA); }
void pminub(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDA); }
void psadbw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF6); }
void paddq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD4); }
void pmuludq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF4); }
void psubq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFB); }
void paddb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFC); }
void paddw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFD); }
void paddd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFE); }
void paddsb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEC); }
void paddsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xED); }
void paddusb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDC); }
void paddusw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDD); }
void pcmpeqb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x74); }
void pcmpeqw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x75); }
void pcmpeqd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x76); }
void pcmpgtb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x64); }
void pcmpgtw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x65); }
void pcmpgtd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x66); }
void psllw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF1); }
void pslld(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF2); }
void psllq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF3); }
void psraw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE1); }
void psrad(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE2); }
void psrlw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD1); }
void psrld(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD2); }
void psrlq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD3); }
void psubb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF8); }
void psubw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF9); }
void psubd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFA); }
void psubsb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE8); }
void psubsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE9); }
void psubusb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD8); }
void psubusw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD9); }
void psllw(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x71, 6); }
void pslld(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x72, 6); }
void psllq(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x73, 6); }
void psraw(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x71, 4); }
void psrad(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x72, 4); }
void psrlw(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x71, 2); }
void psrld(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x72, 2); }
void psrlq(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x73, 2); }
void pslldq(const Xmm& xmm, int imm8) { opMMX_IMM(xmm, imm8, 0x73, 7); }
void psrldq(const Xmm& xmm, int imm8) { opMMX_IMM(xmm, imm8, 0x73, 3); }
void pshufw(const Mmx& mmx, const Operand& op, uint8 imm8) { opMMX(mmx, op, 0x70, 0x00, imm8); }
void pshuflw(const Mmx& mmx, const Operand& op, uint8 imm8) { opMMX(mmx, op, 0x70, 0xF2, imm8); }
void pshufhw(const Mmx& mmx, const Operand& op, uint8 imm8) { opMMX(mmx, op, 0x70, 0xF3, imm8); }
void pshufd(const Mmx& mmx, const Operand& op, uint8 imm8) { opMMX(mmx, op, 0x70, 0x66, imm8); }
void movdqa(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x6F, 0x66); }
void movdqa(const Address& addr, const Xmm& xmm) { db(0x66); opModM(addr, xmm, 0x0F, 0x7F); }
void movdqu(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x6F, 0xF3); }
void movdqu(const Address& addr, const Xmm& xmm) { db(0xF3); opModM(addr, xmm, 0x0F, 0x7F); }
void movaps(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x28, 0x100); }
void movaps(const Address& addr, const Xmm& xmm) { opModM(addr, xmm, 0x0F, 0x29); }
void movss(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x10, 0xF3); }
void movss(const Address& addr, const Xmm& xmm) { db(0xF3); opModM(addr, xmm, 0x0F, 0x11); }
void movups(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x10, 0x100); }
void movups(const Address& addr, const Xmm& xmm) { opModM(addr, xmm, 0x0F, 0x11); }
void movapd(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x28, 0x66); }
void movapd(const Address& addr, const Xmm& xmm) { db(0x66); opModM(addr, xmm, 0x0F, 0x29); }
void movsd(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x10, 0xF2); }
void movsd(const Address& addr, const Xmm& xmm) { db(0xF2); opModM(addr, xmm, 0x0F, 0x11); }
void movupd(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x10, 0x66); }
void movupd(const Address& addr, const Xmm& xmm) { db(0x66); opModM(addr, xmm, 0x0F, 0x11); }
void addps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x58, 0x100, isXMM_XMMorMEM); }
void addss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x58, 0xF3, isXMM_XMMorMEM); }
void addpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x58, 0x66, isXMM_XMMorMEM); }
void addsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x58, 0xF2, isXMM_XMMorMEM); }
void andnps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x55, 0x100, isXMM_XMMorMEM); }
void andnpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x55, 0x66, isXMM_XMMorMEM); }
void andps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x54, 0x100, isXMM_XMMorMEM); }
void andpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x54, 0x66, isXMM_XMMorMEM); }
void cmpps(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC2, 0x100, isXMM_XMMorMEM, imm8); }
void cmpss(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC2, 0xF3, isXMM_XMMorMEM, imm8); }
void cmppd(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC2, 0x66, isXMM_XMMorMEM, imm8); }
void cmpsd(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC2, 0xF2, isXMM_XMMorMEM, imm8); }
void divps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5E, 0x100, isXMM_XMMorMEM); }
void divss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5E, 0xF3, isXMM_XMMorMEM); }
void divpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5E, 0x66, isXMM_XMMorMEM); }
void divsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5E, 0xF2, isXMM_XMMorMEM); }
void maxps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5F, 0x100, isXMM_XMMorMEM); }
void maxss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5F, 0xF3, isXMM_XMMorMEM); }
void maxpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5F, 0x66, isXMM_XMMorMEM); }
void maxsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5F, 0xF2, isXMM_XMMorMEM); }
void minps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5D, 0x100, isXMM_XMMorMEM); }
void minss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5D, 0xF3, isXMM_XMMorMEM); }
void minpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5D, 0x66, isXMM_XMMorMEM); }
void minsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5D, 0xF2, isXMM_XMMorMEM); }
void mulps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x59, 0x100, isXMM_XMMorMEM); }
void mulss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x59, 0xF3, isXMM_XMMorMEM); }
void mulpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x59, 0x66, isXMM_XMMorMEM); }
void mulsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x59, 0xF2, isXMM_XMMorMEM); }
void orps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x56, 0x100, isXMM_XMMorMEM); }
void orpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x56, 0x66, isXMM_XMMorMEM); }
void rcpps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x53, 0x100, isXMM_XMMorMEM); }
void rcpss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x53, 0xF3, isXMM_XMMorMEM); }
void rsqrtps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x52, 0x100, isXMM_XMMorMEM); }
void rsqrtss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x52, 0xF3, isXMM_XMMorMEM); }
void shufps(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC6, 0x100, isXMM_XMMorMEM, imm8); }
void shufpd(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC6, 0x66, isXMM_XMMorMEM, imm8); }
void sqrtps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x51, 0x100, isXMM_XMMorMEM); }
void sqrtss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x51, 0xF3, isXMM_XMMorMEM); }
void sqrtpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x51, 0x66, isXMM_XMMorMEM); }
void sqrtsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x51, 0xF2, isXMM_XMMorMEM); }
void subps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5C, 0x100, isXMM_XMMorMEM); }
void subss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5C, 0xF3, isXMM_XMMorMEM); }
void subpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5C, 0x66, isXMM_XMMorMEM); }
void subsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5C, 0xF2, isXMM_XMMorMEM); }
void unpckhps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x15, 0x100, isXMM_XMMorMEM); }
void unpckhpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x15, 0x66, isXMM_XMMorMEM); }
void unpcklps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x14, 0x100, isXMM_XMMorMEM); }
void unpcklpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x14, 0x66, isXMM_XMMorMEM); }
void xorps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x57, 0x100, isXMM_XMMorMEM); }
void xorpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x57, 0x66, isXMM_XMMorMEM); }
void maskmovdqu(const Xmm& reg1, const Xmm& reg2) { db(0x66);  opModR(reg1, reg2, 0x0F, 0xF7); }
void movhlps(const Xmm& reg1, const Xmm& reg2) {  opModR(reg1, reg2, 0x0F, 0x12); }
void movlhps(const Xmm& reg1, const Xmm& reg2) {  opModR(reg1, reg2, 0x0F, 0x16); }
void punpckhqdq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x6D, 0x66, isXMM_XMMorMEM); }
void punpcklqdq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x6C, 0x66, isXMM_XMMorMEM); }
void comiss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2F, 0x100, isXMM_XMMorMEM); }
void ucomiss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2E, 0x100, isXMM_XMMorMEM); }
void comisd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2F, 0x66, isXMM_XMMorMEM); }
void ucomisd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2E, 0x66, isXMM_XMMorMEM); }
void cvtpd2ps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5A, 0x66, isXMM_XMMorMEM); }
void cvtps2pd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5A, 0x100, isXMM_XMMorMEM); }
void cvtsd2ss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5A, 0xF2, isXMM_XMMorMEM); }
void cvtss2sd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5A, 0xF3, isXMM_XMMorMEM); }
void cvtpd2dq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xE6, 0xF2, isXMM_XMMorMEM); }
void cvttpd2dq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xE6, 0x66, isXMM_XMMorMEM); }
void cvtdq2pd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xE6, 0xF3, isXMM_XMMorMEM); }
void cvtps2dq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5B, 0x66, isXMM_XMMorMEM); }
void cvttps2dq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5B, 0xF3, isXMM_XMMorMEM); }
void cvtdq2ps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5B, 0x100, isXMM_XMMorMEM); }
void addsubpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xD0, 0x66, isXMM_XMMorMEM); }
void addsubps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xD0, 0xF2, isXMM_XMMorMEM); }
void haddpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x7C, 0x66, isXMM_XMMorMEM); }
void haddps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x7C, 0xF2, isXMM_XMMorMEM); }
void hsubpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x7D, 0x66, isXMM_XMMorMEM); }
void hsubps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x7D, 0xF2, isXMM_XMMorMEM); }
void movddup(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x12, 0xF2, isXMM_XMMorMEM); }
void movshdup(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x16, 0xF3, isXMM_XMMorMEM); }
void movsldup(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x12, 0xF3, isXMM_XMMorMEM); }
void cvtpi2ps(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2A, 0x100, isXMM_MMXorMEM); }
void cvtps2pi(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2D, 0x100, isMMX_XMMorMEM); }
void cvtsi2ss(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2A, 0xF3, isXMM_REG32orMEM); }
void cvtss2si(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2D, 0xF3, isREG32_XMMorMEM); }
void cvttps2pi(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2C, 0x100, isMMX_XMMorMEM); }
void cvttss2si(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2C, 0xF3, isREG32_XMMorMEM); }
void cvtpi2pd(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2A, 0x66, isXMM_MMXorMEM); }
void cvtpd2pi(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2D, 0x66, isMMX_XMMorMEM); }
void cvtsi2sd(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2A, 0xF2, isXMM_REG32orMEM); }
void cvtsd2si(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2D, 0xF2, isREG32_XMMorMEM); }
void cvttpd2pi(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2C, 0x66, isMMX_XMMorMEM); }
void cvttsd2si(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2C, 0xF2, isREG32_XMMorMEM); }
void prefetcht0(const Address& addr) { opModM(addr, Reg32(1), 0x0F, 0x18); }
void prefetcht1(const Address& addr) { opModM(addr, Reg32(2), 0x0F, 0x18); }
void prefetcht2(const Address& addr) { opModM(addr, Reg32(3), 0x0F, 0x18); }
void prefetchnta(const Address& addr) { opModM(addr, Reg32(0), 0x0F, 0x18); }
void movhps(const Operand& op1, const Operand& op2) { opMovXMM(op1, op2, 0x16, 0x100); }
void movlps(const Operand& op1, const Operand& op2) { opMovXMM(op1, op2, 0x12, 0x100); }
void movhpd(const Operand& op1, const Operand& op2) { opMovXMM(op1, op2, 0x16, 0x66); }
void movlpd(const Operand& op1, const Operand& op2) { opMovXMM(op1, op2, 0x12, 0x66); }
void cmovo(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 0); }
void jo(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x70, 0x80, 0x0F); }
void jo(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x70, 0x80, 0x0F); }
void jo(const char *label, LabelType type = T_AUTO) { jo(std::string(label), type); }
void jo(const void *addr) { opJmpAbs(addr, T_NEAR, 0x70, 0x80, 0x0F); }
void seto(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 0); }
void cmovno(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 1); }
void jno(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x71, 0x81, 0x0F); }
void jno(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x71, 0x81, 0x0F); }
void jno(const char *label, LabelType type = T_AUTO) { jno(std::string(label), type); }
void jno(const void *addr) { opJmpAbs(addr, T_NEAR, 0x71, 0x81, 0x0F); }
void setno(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 1); }
void cmovb(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 2); }
void jb(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void jb(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void jb(const char *label, LabelType type = T_AUTO) { jb(std::string(label), type); }
void jb(const void *addr) { opJmpAbs(addr, T_NEAR, 0x72, 0x82, 0x0F); }
void setb(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 2); }
void cmovc(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 2); }
void jc(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void jc(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void jc(const char *label, LabelType type = T_AUTO) { jc(std::string(label), type); }
void jc(const void *addr) { opJmpAbs(addr, T_NEAR, 0x72, 0x82, 0x0F); }
void setc(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 2); }
void cmovnae(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 2); }
void jnae(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void jnae(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void jnae(const char *label, LabelType type = T_AUTO) { jnae(std::string(label), type); }
void jnae(const void *addr) { opJmpAbs(addr, T_NEAR, 0x72, 0x82, 0x0F); }
void setnae(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 2); }
void cmovnb(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 3); }
void jnb(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void jnb(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void jnb(const char *label, LabelType type = T_AUTO) { jnb(std::string(label), type); }
void jnb(const void *addr) { opJmpAbs(addr, T_NEAR, 0x73, 0x83, 0x0F); }
void setnb(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 3); }
void cmovae(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 3); }
void jae(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void jae(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void jae(const char *label, LabelType type = T_AUTO) { jae(std::string(label), type); }
void jae(const void *addr) { opJmpAbs(addr, T_NEAR, 0x73, 0x83, 0x0F); }
void setae(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 3); }
void cmovnc(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 3); }
void jnc(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void jnc(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void jnc(const char *label, LabelType type = T_AUTO) { jnc(std::string(label), type); }
void jnc(const void *addr) { opJmpAbs(addr, T_NEAR, 0x73, 0x83, 0x0F); }
void setnc(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 3); }
void cmove(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 4); }
void je(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x74, 0x84, 0x0F); }
void je(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x74, 0x84, 0x0F); }
void je(const char *label, LabelType type = T_AUTO) { je(std::string(label), type); }
void je(const void *addr) { opJmpAbs(addr, T_NEAR, 0x74, 0x84, 0x0F); }
void sete(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 4); }
void cmovz(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 4); }
void jz(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x74, 0x84, 0x0F); }
void jz(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x74, 0x84, 0x0F); }
void jz(const char *label, LabelType type = T_AUTO) { jz(std::string(label), type); }
void jz(const void *addr) { opJmpAbs(addr, T_NEAR, 0x74, 0x84, 0x0F); }
void setz(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 4); }
void cmovne(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 5); }
void jne(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x75, 0x85, 0x0F); }
void jne(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x75, 0x85, 0x0F); }
void jne(const char *label, LabelType type = T_AUTO) { jne(std::string(label), type); }
void jne(const void *addr) { opJmpAbs(addr, T_NEAR, 0x75, 0x85, 0x0F); }
void setne(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 5); }
void cmovnz(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 5); }
void jnz(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x75, 0x85, 0x0F); }
void jnz(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x75, 0x85, 0x0F); }
void jnz(const char *label, LabelType type = T_AUTO) { jnz(std::string(label), type); }
void jnz(const void *addr) { opJmpAbs(addr, T_NEAR, 0x75, 0x85, 0x0F); }
void setnz(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 5); }
void cmovbe(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 6); }
void jbe(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x76, 0x86, 0x0F); }
void jbe(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x76, 0x86, 0x0F); }
void jbe(const char *label, LabelType type = T_AUTO) { jbe(std::string(label), type); }
void jbe(const void *addr) { opJmpAbs(addr, T_NEAR, 0x76, 0x86, 0x0F); }
void setbe(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 6); }
void cmovna(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 6); }
void jna(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x76, 0x86, 0x0F); }
void jna(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x76, 0x86, 0x0F); }
void jna(const char *label, LabelType type = T_AUTO) { jna(std::string(label), type); }
void jna(const void *addr) { opJmpAbs(addr, T_NEAR, 0x76, 0x86, 0x0F); }
void setna(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 6); }
void cmovnbe(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 7); }
void jnbe(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x77, 0x87, 0x0F); }
void jnbe(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x77, 0x87, 0x0F); }
void jnbe(const char *label, LabelType type = T_AUTO) { jnbe(std::string(label), type); }
void jnbe(const void *addr) { opJmpAbs(addr, T_NEAR, 0x77, 0x87, 0x0F); }
void setnbe(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 7); }
void cmova(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 7); }
void ja(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x77, 0x87, 0x0F); }
void ja(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x77, 0x87, 0x0F); }
void ja(const char *label, LabelType type = T_AUTO) { ja(std::string(label), type); }
void ja(const void *addr) { opJmpAbs(addr, T_NEAR, 0x77, 0x87, 0x0F); }
void seta(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 7); }
void cmovs(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 8); }
void js(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x78, 0x88, 0x0F); }
void js(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x78, 0x88, 0x0F); }
void js(const char *label, LabelType type = T_AUTO) { js(std::string(label), type); }
void js(const void *addr) { opJmpAbs(addr, T_NEAR, 0x78, 0x88, 0x0F); }
void sets(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 8); }
void cmovns(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 9); }
void jns(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x79, 0x89, 0x0F); }
void jns(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x79, 0x89, 0x0F); }
void jns(const char *label, LabelType type = T_AUTO) { jns(std::string(label), type); }
void jns(const void *addr) { opJmpAbs(addr, T_NEAR, 0x79, 0x89, 0x0F); }
void setns(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 9); }
void cmovp(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 10); }
void jp(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7A, 0x8A, 0x0F); }
void jp(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7A, 0x8A, 0x0F); }
void jp(const char *label, LabelType type = T_AUTO) { jp(std::string(label), type); }
void jp(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7A, 0x8A, 0x0F); }
void setp(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 10); }
void cmovpe(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 10); }
void jpe(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7A, 0x8A, 0x0F); }
void jpe(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7A, 0x8A, 0x0F); }
void jpe(const char *label, LabelType type = T_AUTO) { jpe(std::string(label), type); }
void jpe(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7A, 0x8A, 0x0F); }
void setpe(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 10); }
void cmovnp(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 11); }
void jnp(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7B, 0x8B, 0x0F); }
void jnp(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7B, 0x8B, 0x0F); }
void jnp(const char *label, LabelType type = T_AUTO) { jnp(std::string(label), type); }
void jnp(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7B, 0x8B, 0x0F); }
void setnp(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 11); }
void cmovpo(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 11); }
void jpo(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7B, 0x8B, 0x0F); }
void jpo(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7B, 0x8B, 0x0F); }
void jpo(const char *label, LabelType type = T_AUTO) { jpo(std::string(label), type); }
void jpo(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7B, 0x8B, 0x0F); }
void setpo(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 11); }
void cmovl(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 12); }
void jl(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7C, 0x8C, 0x0F); }
void jl(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7C, 0x8C, 0x0F); }
void jl(const char *label, LabelType type = T_AUTO) { jl(std::string(label), type); }
void jl(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7C, 0x8C, 0x0F); }
void setl(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 12); }
void cmovnge(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 12); }
void jnge(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7C, 0x8C, 0x0F); }
void jnge(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7C, 0x8C, 0x0F); }
void jnge(const char *label, LabelType type = T_AUTO) { jnge(std::string(label), type); }
void jnge(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7C, 0x8C, 0x0F); }
void setnge(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 12); }
void cmovnl(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 13); }
void jnl(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7D, 0x8D, 0x0F); }
void jnl(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7D, 0x8D, 0x0F); }
void jnl(const char *label, LabelType type = T_AUTO) { jnl(std::string(label), type); }
void jnl(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7D, 0x8D, 0x0F); }
void setnl(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 13); }
void cmovge(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 13); }
void jge(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7D, 0x8D, 0x0F); }
void jge(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7D, 0x8D, 0x0F); }
void jge(const char *label, LabelType type = T_AUTO) { jge(std::string(label), type); }
void jge(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7D, 0x8D, 0x0F); }
void setge(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 13); }
void cmovle(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 14); }
void jle(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7E, 0x8E, 0x0F); }
void jle(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7E, 0x8E, 0x0F); }
void jle(const char *label, LabelType type = T_AUTO) { jle(std::string(label), type); }
void jle(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7E, 0x8E, 0x0F); }
void setle(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 14); }
void cmovng(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 14); }
void jng(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7E, 0x8E, 0x0F); }
void jng(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7E, 0x8E, 0x0F); }
void jng(const char *label, LabelType type = T_AUTO) { jng(std::string(label), type); }
void jng(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7E, 0x8E, 0x0F); }
void setng(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 14); }
void cmovnle(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 15); }
void jnle(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7F, 0x8F, 0x0F); }
void jnle(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7F, 0x8F, 0x0F); }
void jnle(const char *label, LabelType type = T_AUTO) { jnle(std::string(label), type); }
void jnle(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7F, 0x8F, 0x0F); }
void setnle(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 15); }
void cmovg(const Reg& reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0x40 | 15); }
void jg(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7F, 0x8F, 0x0F); }
void jg(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7F, 0x8F, 0x0F); }
void jg(const char *label, LabelType type = T_AUTO) { jg(std::string(label), type); }
void jg(const void *addr) { opJmpAbs(addr, T_NEAR, 0x7F, 0x8F, 0x0F); }
void setg(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, 0x90 | 15); }
#ifdef XBYAK32
void jcxz(std::string label) { db(0x67); opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jcxz(const Label& label) { db(0x67); opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jecxz(std::string label) { opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jecxz(const Label& label) { opJmp(label, T_SHORT, 0xe3, 0, 0); }
#else
void jecxz(std::string label) { db(0x67); opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jecxz(const Label& label) { db(0x67); opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jrcxz(std::string label) { opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jrcxz(const Label& label) { opJmp(label, T_SHORT, 0xe3, 0, 0); }
#endif
#ifdef XBYAK64
void cdqe() { db(0x48); db(0x98); }
void cqo() { db(0x48); db(0x99); }
#else
void aaa() { db(0x37); }
void aad() { db(0xD5); db(0x0A); }
void aam() { db(0xD4); db(0x0A); }
void aas() { db(0x3F); }
void daa() { db(0x27); }
void das() { db(0x2F); }
void popad() { db(0x61); }
void popfd() { db(0x9D); }
void pusha() { db(0x60); }
void pushad() { db(0x60); }
void pushfd() { db(0x9C); }
void popa() { db(0x61); }
#endif
void cbw() { db(0x66); db(0x98); }
void cdq() { db(0x99); }
void clc() { db(0xF8); }
void cld() { db(0xFC); }
void cli() { db(0xFA); }
void cmc() { db(0xF5); }
void cpuid() { db(0x0F); db(0xA2); }
void cwd() { db(0x66); db(0x99); }
void cwde() { db(0x98); }
void lahf() { db(0x9F); }
void lock() { db(0xF0); }
void nop() { db(0x90); }
void sahf() { db(0x9E); }
void stc() { db(0xF9); }
void std() { db(0xFD); }
void sti() { db(0xFB); }
void emms() { db(0x0F); db(0x77); }
void pause() { db(0xF3); db(0x90); }
void sfence() { db(0x0F); db(0xAE); db(0xF8); }
void lfence() { db(0x0F); db(0xAE); db(0xE8); }
void mfence() { db(0x0F); db(0xAE); db(0xF0); }
void monitor() { db(0x0F); db(0x01); db(0xC8); }
void mwait() { db(0x0F); db(0x01); db(0xC9); }
void rdmsr() { db(0x0F); db(0x32); }
void rdpmc() { db(0x0F); db(0x33); }
void rdtsc() { db(0x0F); db(0x31); }
void rdtscp() { db(0x0F); db(0x01); db(0xF9); }
void ud2() { db(0x0F); db(0x0B); }
void wait() { db(0x9B); }
void fwait() { db(0x9B); }
void wbinvd() { db(0x0F); db(0x09); }
void wrmsr() { db(0x0F); db(0x30); }
void xlatb() { db(0xD7); }
void popf() { db(0x9D); }
void pushf() { db(0x9C); }
void stac() { db(0x0F); db(0x01); db(0xCB); }
void vzeroall() { db(0xC5); db(0xFC); db(0x77); }
void vzeroupper() { db(0xC5); db(0xF8); db(0x77); }
void xgetbv() { db(0x0F); db(0x01); db(0xD0); }
void f2xm1() { db(0xD9); db(0xF0); }
void fabs() { db(0xD9); db(0xE1); }
void faddp() { db(0xDE); db(0xC1); }
void fchs() { db(0xD9); db(0xE0); }
void fcom() { db(0xD8); db(0xD1); }
void fcomp() { db(0xD8); db(0xD9); }
void fcompp() { db(0xDE); db(0xD9); }
void fcos() { db(0xD9); db(0xFF); }
void fdecstp() { db(0xD9); db(0xF6); }
void fdivp() { db(0xDE); db(0xF9); }
void fdivrp() { db(0xDE); db(0xF1); }
void fincstp() { db(0xD9); db(0xF7); }
void finit() { db(0x9B); db(0xDB); db(0xE3); }
void fninit() { db(0xDB); db(0xE3); }
void fld1() { db(0xD9); db(0xE8); }
void fldl2t() { db(0xD9); db(0xE9); }
void fldl2e() { db(0xD9); db(0xEA); }
void fldpi() { db(0xD9); db(0xEB); }
void fldlg2() { db(0xD9); db(0xEC); }
void fldln2() { db(0xD9); db(0xED); }
void fldz() { db(0xD9); db(0xEE); }
void fmulp() { db(0xDE); db(0xC9); }
void fnop() { db(0xD9); db(0xD0); }
void fpatan() { db(0xD9); db(0xF3); }
void fprem() { db(0xD9); db(0xF8); }
void fprem1() { db(0xD9); db(0xF5); }
void fptan() { db(0xD9); db(0xF2); }
void frndint() { db(0xD9); db(0xFC); }
void fscale() { db(0xD9); db(0xFD); }
void fsin() { db(0xD9); db(0xFE); }
void fsincos() { db(0xD9); db(0xFB); }
void fsqrt() { db(0xD9); db(0xFA); }
void fsubp() { db(0xDE); db(0xE9); }
void fsubrp() { db(0xDE); db(0xE1); }
void ftst() { db(0xD9); db(0xE4); }
void fucom() { db(0xDD); db(0xE1); }
void fucomp() { db(0xDD); db(0xE9); }
void fucompp() { db(0xDA); db(0xE9); }
void fxam() { db(0xD9); db(0xE5); }
void fxch() { db(0xD9); db(0xC9); }
void fxtract() { db(0xD9); db(0xF4); }
void fyl2x() { db(0xD9); db(0xF1); }
void fyl2xp1() { db(0xD9); db(0xF9); }
void adc(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x10); }
void adc(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x10, 2); }
void add(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x00); }
void add(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x00, 0); }
void and_(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x20); }
void and_(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x20, 4); }
#ifndef XBYAK_NO_OP_NAMES
void and(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x20); }
void and(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x20, 4); }
#endif
void cmp(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x38); }
void cmp(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x38, 7); }
void or_(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x08); }
void or_(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x08, 1); }
#ifndef XBYAK_NO_OP_NAMES
void or(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x08); }
void or(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x08, 1); }
#endif
void sbb(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x18); }
void sbb(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x18, 3); }
void sub(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x28); }
void sub(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x28, 5); }
void xor_(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x30); }
void xor_(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x30, 6); }
#ifndef XBYAK_NO_OP_NAMES
void xor(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x30); }
void xor(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x30, 6); }
#endif
void dec(const Operand& op) { opIncDec(op, 0x48, 1); }
void inc(const Operand& op) { opIncDec(op, 0x40, 0); }
void bt(const Operand& op, const Reg& reg) { opModRM(reg, op, op.isREG(16|32|64) && op.getBit() == reg.getBit(), op.isMEM(), 0x0f, 0xA3); }
void bt(const Operand& op, uint8 imm) { opR_ModM(op, 16|32|64, 4, 0x0f, 0xba, NONE, false, 1); db(imm); }
void bts(const Operand& op, const Reg& reg) { opModRM(reg, op, op.isREG(16|32|64) && op.getBit() == reg.getBit(), op.isMEM(), 0x0f, 0xAB); }
void bts(const Operand& op, uint8 imm) { opR_ModM(op, 16|32|64, 5, 0x0f, 0xba, NONE, false, 1); db(imm); }
void btr(const Operand& op, const Reg& reg) { opModRM(reg, op, op.isREG(16|32|64) && op.getBit() == reg.getBit(), op.isMEM(), 0x0f, 0xB3); }
void btr(const Operand& op, uint8 imm) { opR_ModM(op, 16|32|64, 6, 0x0f, 0xba, NONE, false, 1); db(imm); }
void btc(const Operand& op, const Reg& reg) { opModRM(reg, op, op.isREG(16|32|64) && op.getBit() == reg.getBit(), op.isMEM(), 0x0f, 0xBB); }
void btc(const Operand& op, uint8 imm) { opR_ModM(op, 16|32|64, 7, 0x0f, 0xba, NONE, false, 1); db(imm); }
void div(const Operand& op) { opR_ModM(op, 0, 6, 0xF6); }
void idiv(const Operand& op) { opR_ModM(op, 0, 7, 0xF6); }
void imul(const Operand& op) { opR_ModM(op, 0, 5, 0xF6); }
void mul(const Operand& op) { opR_ModM(op, 0, 4, 0xF6); }
void neg(const Operand& op) { opR_ModM(op, 0, 3, 0xF6); }
void not_(const Operand& op) { opR_ModM(op, 0, 2, 0xF6); }
#ifndef XBYAK_NO_OP_NAMES
void not(const Operand& op) { opR_ModM(op, 0, 2, 0xF6); }
#endif
void rcl(const Operand& op, int imm) { opShift(op, imm, 2); }
void rcl(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 2); }
void rcr(const Operand& op, int imm) { opShift(op, imm, 3); }
void rcr(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 3); }
void rol(const Operand& op, int imm) { opShift(op, imm, 0); }
void rol(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 0); }
void ror(const Operand& op, int imm) { opShift(op, imm, 1); }
void ror(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 1); }
void sar(const Operand& op, int imm) { opShift(op, imm, 7); }
void sar(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 7); }
void shl(const Operand& op, int imm) { opShift(op, imm, 4); }
void shl(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 4); }
void shr(const Operand& op, int imm) { opShift(op, imm, 5); }
void shr(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 5); }
void sal(const Operand& op, int imm) { opShift(op, imm, 4); }
void sal(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 4); }
void shld(const Operand& op, const Reg& reg, uint8 imm) { opShxd(op, reg, imm, 0xA4); }
void shld(const Operand& op, const Reg& reg, const Reg8& _cl) { opShxd(op, reg, 0, 0xA4, &_cl); }
void shrd(const Operand& op, const Reg& reg, uint8 imm) { opShxd(op, reg, imm, 0xAC); }
void shrd(const Operand& op, const Reg& reg, const Reg8& _cl) { opShxd(op, reg, 0, 0xAC, &_cl); }
void bsf(const Reg&reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0xBC); }
void bsr(const Reg&reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0xBD); }
void popcnt(const Reg&reg, const Operand& op) { opSp1(reg, op, 0xF3, 0x0F, 0xB8); }
void tzcnt(const Reg&reg, const Operand& op) { opSp1(reg, op, 0xF3, 0x0F, 0xBC); }
void lzcnt(const Reg&reg, const Operand& op) { opSp1(reg, op, 0xF3, 0x0F, 0xBD); }
void pshufb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x00, 0x66, NONE, 0x38); }
void phaddw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x01, 0x66, NONE, 0x38); }
void phaddd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x02, 0x66, NONE, 0x38); }
void phaddsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x03, 0x66, NONE, 0x38); }
void pmaddubsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x04, 0x66, NONE, 0x38); }
void phsubw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x05, 0x66, NONE, 0x38); }
void phsubd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x06, 0x66, NONE, 0x38); }
void phsubsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x07, 0x66, NONE, 0x38); }
void psignb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x08, 0x66, NONE, 0x38); }
void psignw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x09, 0x66, NONE, 0x38); }
void psignd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x0A, 0x66, NONE, 0x38); }
void pmulhrsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x0B, 0x66, NONE, 0x38); }
void pabsb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x1C, 0x66, NONE, 0x38); }
void pabsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x1D, 0x66, NONE, 0x38); }
void pabsd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x1E, 0x66, NONE, 0x38); }
void palignr(const Mmx& mmx, const Operand& op, int imm) { opMMX(mmx, op, 0x0f, 0x66, static_cast<uint8>(imm), 0x3a); }
void blendvpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x15, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void blendvps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x14, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void packusdw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2B, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pblendvb(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x10, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pcmpeqq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x29, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void ptest(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x17, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxbw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x20, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxbd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x21, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxbq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x22, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxwd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x23, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxwq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x24, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxdq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x25, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxbw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x30, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxbd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x31, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxbq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x32, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxwd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x33, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxwq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x34, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxdq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x35, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pminsb(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x38, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pminsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x39, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pminuw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3A, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pminud(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3B, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmaxsb(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3C, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmaxsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3D, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmaxuw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3E, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmaxud(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3F, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmuldq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x28, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmulld(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x40, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void phminposuw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x41, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pcmpgtq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x37, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesdec(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDE, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesdeclast(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDF, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesenc(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDC, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesenclast(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDD, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesimc(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDB, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void blendpd(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0D, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void blendps(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0C, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void dppd(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x41, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void dpps(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x40, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void mpsadbw(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x42, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pblendw(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0E, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void roundps(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x08, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void roundpd(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x09, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void roundss(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0A, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void roundsd(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0B, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pcmpestrm(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x60, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pcmpestri(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x61, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pcmpistrm(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x62, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pcmpistri(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x63, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pclmulqdq(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x44, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void aeskeygenassist(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0xDF, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pclmullqlqdq(const Xmm& xmm, const Operand& op) { pclmulqdq(xmm, op, 0x00); }
void pclmulhqlqdq(const Xmm& xmm, const Operand& op) { pclmulqdq(xmm, op, 0x01); }
void pclmullqhdq(const Xmm& xmm, const Operand& op) { pclmulqdq(xmm, op, 0x10); }
void pclmulhqhdq(const Xmm& xmm, const Operand& op) { pclmulqdq(xmm, op, 0x11); }
void ldmxcsr(const Address& addr) { opModM(addr, Reg32(2), 0x0F, 0xAE); }
void stmxcsr(const Address& addr) { opModM(addr, Reg32(3), 0x0F, 0xAE); }
void clflush(const Address& addr) { opModM(addr, Reg32(7), 0x0F, 0xAE); }
void fldcw(const Address& addr) { opModM(addr, Reg32(5), 0xD9, 0x100); }
void fstcw(const Address& addr) { db(0x9B); opModM(addr, Reg32(7), 0xD9, NONE); }
void movntpd(const Address& addr, const Xmm& reg) { opModM(addr, Reg16(reg.getIdx()), 0x0F, 0x2B); }
void movntdq(const Address& addr, const Xmm& reg) { opModM(addr, Reg16(reg.getIdx()), 0x0F, 0xE7); }
void movsx(const Reg& reg, const Operand& op) { opMovxx(reg, op, 0xBE); }
void movzx(const Reg& reg, const Operand& op) { opMovxx(reg, op, 0xB6); }
void fadd(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 0, 0); }
void fiadd(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 0, 0); }
void fcom(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 2, 0); }
void fcomp(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 3, 0); }
void fdiv(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 6, 0); }
void fidiv(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 6, 0); }
void fdivr(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 7, 0); }
void fidivr(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 7, 0); }
void ficom(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 2, 0); }
void ficomp(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 3, 0); }
void fild(const Address& addr) { opFpuMem(addr, 0xDF, 0xDB, 0xDF, 0, 5); }
void fist(const Address& addr) { opFpuMem(addr, 0xDF, 0xDB, 0x00, 2, 0); }
void fistp(const Address& addr) { opFpuMem(addr, 0xDF, 0xDB, 0xDF, 3, 7); }
void fisttp(const Address& addr) { opFpuMem(addr, 0xDF, 0xDB, 0xDD, 1, 0); }
void fld(const Address& addr) { opFpuMem(addr, 0x00, 0xD9, 0xDD, 0, 0); }
void fmul(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 1, 0); }
void fimul(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 1, 0); }
void fst(const Address& addr) { opFpuMem(addr, 0x00, 0xD9, 0xDD, 2, 0); }
void fstp(const Address& addr) { opFpuMem(addr, 0x00, 0xD9, 0xDD, 3, 0); }
void fsub(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 4, 0); }
void fisub(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 4, 0); }
void fsubr(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 5, 0); }
void fisubr(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 5, 0); }
void fadd(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8C0, 0xDCC0); }
void fadd(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8C0, 0xDCC0); }
void faddp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEC0); }
void faddp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEC0); }
void fcmovb(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDAC0, 0x00C0); }
void fcmovb(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDAC0, 0x00C0); }
void fcmove(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDAC8, 0x00C8); }
void fcmove(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDAC8, 0x00C8); }
void fcmovbe(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDAD0, 0x00D0); }
void fcmovbe(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDAD0, 0x00D0); }
void fcmovu(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDAD8, 0x00D8); }
void fcmovu(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDAD8, 0x00D8); }
void fcmovnb(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBC0, 0x00C0); }
void fcmovnb(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBC0, 0x00C0); }
void fcmovne(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBC8, 0x00C8); }
void fcmovne(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBC8, 0x00C8); }
void fcmovnbe(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBD0, 0x00D0); }
void fcmovnbe(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBD0, 0x00D0); }
void fcmovnu(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBD8, 0x00D8); }
void fcmovnu(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBD8, 0x00D8); }
void fcomi(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBF0, 0x00F0); }
void fcomi(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBF0, 0x00F0); }
void fcomip(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDFF0, 0x00F0); }
void fcomip(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDFF0, 0x00F0); }
void fucomi(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBE8, 0x00E8); }
void fucomi(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBE8, 0x00E8); }
void fucomip(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDFE8, 0x00E8); }
void fucomip(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDFE8, 0x00E8); }
void fdiv(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8F0, 0xDCF8); }
void fdiv(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8F0, 0xDCF8); }
void fdivp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEF8); }
void fdivp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEF8); }
void fdivr(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8F8, 0xDCF0); }
void fdivr(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8F8, 0xDCF0); }
void fdivrp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEF0); }
void fdivrp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEF0); }
void fmul(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8C8, 0xDCC8); }
void fmul(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8C8, 0xDCC8); }
void fmulp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEC8); }
void fmulp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEC8); }
void fsub(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8E0, 0xDCE8); }
void fsub(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8E0, 0xDCE8); }
void fsubp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEE8); }
void fsubp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEE8); }
void fsubr(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8E8, 0xDCE0); }
void fsubr(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8E8, 0xDCE0); }
void fsubrp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEE0); }
void fsubrp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEE0); }
void fcom(const Fpu& reg) { opFpu(reg, 0xD8, 0xD0); }
void fcomp(const Fpu& reg) { opFpu(reg, 0xD8, 0xD8); }
void ffree(const Fpu& reg) { opFpu(reg, 0xDD, 0xC0); }
void fld(const Fpu& reg) { opFpu(reg, 0xD9, 0xC0); }
void fst(const Fpu& reg) { opFpu(reg, 0xDD, 0xD0); }
void fstp(const Fpu& reg) { opFpu(reg, 0xDD, 0xD8); }
void fucom(const Fpu& reg) { opFpu(reg, 0xDD, 0xE0); }
void fucomp(const Fpu& reg) { opFpu(reg, 0xDD, 0xE8); }
void fxch(const Fpu& reg) { opFpu(reg, 0xD9, 0xC8); }
void vaddpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x58); }
void vaddps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x58); }
void vaddsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F2 | T_EW1 | T_EVEX | T_ER_Z | T_N8, 0x58); }
void vaddss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F3 | T_EW0 | T_EVEX | T_ER_Z | T_N4, 0x58); }
void vsubpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x5C); }
void vsubps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x5C); }
void vsubsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F2 | T_EW1 | T_EVEX | T_ER_Z | T_N8, 0x5C); }
void vsubss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F3 | T_EW0 | T_EVEX | T_ER_Z | T_N4, 0x5C); }
void vmulpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x59); }
void vmulps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x59); }
void vmulsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F2 | T_EW1 | T_EVEX | T_ER_Z | T_N8, 0x59); }
void vmulss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F3 | T_EW0 | T_EVEX | T_ER_Z | T_N4, 0x59); }
void vdivpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x5E); }
void vdivps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x5E); }
void vdivsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F2 | T_EW1 | T_EVEX | T_ER_Z | T_N8, 0x5E); }
void vdivss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F3 | T_EW0 | T_EVEX | T_ER_Z | T_N4, 0x5E); }
void vmaxpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x5F); }
void vmaxps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x5F); }
void vmaxsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F2 | T_EW1 | T_EVEX | T_ER_Z | T_N8, 0x5F); }
void vmaxss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F3 | T_EW0 | T_EVEX | T_ER_Z | T_N4, 0x5F); }
void vminpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x5D); }
void vminps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x5D); }
void vminsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F2 | T_EW1 | T_EVEX | T_ER_Z | T_N8, 0x5D); }
void vminss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_F3 | T_EW0 | T_EVEX | T_ER_Z | T_N4, 0x5D); }
void vandpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x54); }
void vandps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x54); }
void vandnpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x55); }
void vandnps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x55); }
void vorpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x56); }
void vorps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x56); }
void vxorpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_66 | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x57); }
void vxorps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x57); }
void vblendpd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0 | T_YMM, 0x0D, imm); }
void vblendpd(const Xmm& x, const Operand& op, uint8 imm) { vblendpd(x, x, op, imm); }
void vblendps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0 | T_YMM, 0x0C, imm); }
void vblendps(const Xmm& x, const Operand& op, uint8 imm) { vblendps(x, x, op, imm); }
void vdppd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0, 0x41, imm); }
void vdppd(const Xmm& x, const Operand& op, uint8 imm) { vdppd(x, x, op, imm); }
void vdpps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0 | T_YMM, 0x40, imm); }
void vdpps(const Xmm& x, const Operand& op, uint8 imm) { vdpps(x, x, op, imm); }
void vmpsadbw(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0 | T_YMM, 0x42, imm); }
void vmpsadbw(const Xmm& x, const Operand& op, uint8 imm) { vmpsadbw(x, x, op, imm); }
void vpblendw(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0 | T_YMM, 0x0E, imm); }
void vpblendw(const Xmm& x, const Operand& op, uint8 imm) { vpblendw(x, x, op, imm); }
void vpblendd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0 | T_YMM, 0x02, imm); }
void vpblendd(const Xmm& x, const Operand& op, uint8 imm) { vpblendd(x, x, op, imm); }
void vroundsd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0, 0x0B, imm); }
void vroundsd(const Xmm& x, const Operand& op, uint8 imm) { vroundsd(x, x, op, imm); }
void vroundss(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0, 0x0A, imm); }
void vroundss(const Xmm& x, const Operand& op, uint8 imm) { vroundss(x, x, op, imm); }
void vpclmulqdq(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0, 0x44, imm); }
void vpclmulqdq(const Xmm& x, const Operand& op, uint8 imm) { vpclmulqdq(x, x, op, imm); }
void vpermilps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x0C); }
void vpermilpd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x0D); }
void vpsllvd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x47); }
void vpsllvq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x47); }
void vpsravd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x46); }
void vpsrlvd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x45); }
void vpsrlvq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x45); }
void vcmppd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0xC2, imm); }
void vcmppd(const Xmm& x, const Operand& op, uint8 imm) { vcmppd(x, x, op, imm); }
void vcmpps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_0F | T_YMM, 0xC2, imm); }
void vcmpps(const Xmm& x, const Operand& op, uint8 imm) { vcmpps(x, x, op, imm); }
void vcmpsd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_F2 | T_0F, 0xC2, imm); }
void vcmpsd(const Xmm& x, const Operand& op, uint8 imm) { vcmpsd(x, x, op, imm); }
void vcmpss(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_F3 | T_0F, 0xC2, imm); }
void vcmpss(const Xmm& x, const Operand& op, uint8 imm) { vcmpss(x, x, op, imm); }
void vcvtsd2ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_F2 | T_0F | T_EW1 | T_EVEX | T_ER_X | T_N8, 0x5A); }
void vcvtsd2ss(const Xmm& x, const Operand& op) { vcvtsd2ss(x, x, op); }
void vcvtss2sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_F3 | T_0F | T_EW0 | T_EVEX | T_SAE_X | T_N4, 0x5A); }
void vcvtss2sd(const Xmm& x, const Operand& op) { vcvtss2sd(x, x, op); }
void vinsertps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_W0 | T_EW0 | T_EVEX, 0x21, imm); }
void vinsertps(const Xmm& x, const Operand& op, uint8 imm) { vinsertps(x, x, op, imm); }
void vpacksswb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0x63); }
void vpacksswb(const Xmm& x, const Operand& op) { vpacksswb(x, x, op); }
void vpackssdw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0x6B); }
void vpackssdw(const Xmm& x, const Operand& op) { vpackssdw(x, x, op); }
void vpackuswb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0x67); }
void vpackuswb(const Xmm& x, const Operand& op) { vpackuswb(x, x, op); }
void vpackusdw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x2B); }
void vpackusdw(const Xmm& x, const Operand& op) { vpackusdw(x, x, op); }
void vpaddb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xFC); }
void vpaddb(const Xmm& x, const Operand& op) { vpaddb(x, x, op); }
void vpaddw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xFD); }
void vpaddw(const Xmm& x, const Operand& op) { vpaddw(x, x, op); }
void vpaddd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0xFE); }
void vpaddd(const Xmm& x, const Operand& op) { vpaddd(x, x, op); }
void vpaddq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0xD4); }
void vpaddq(const Xmm& x, const Operand& op) { vpaddq(x, x, op); }
void vpaddsb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xEC); }
void vpaddsb(const Xmm& x, const Operand& op) { vpaddsb(x, x, op); }
void vpaddsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xED); }
void vpaddsw(const Xmm& x, const Operand& op) { vpaddsw(x, x, op); }
void vpaddusb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xDC); }
void vpaddusb(const Xmm& x, const Operand& op) { vpaddusb(x, x, op); }
void vpaddusw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xDD); }
void vpaddusw(const Xmm& x, const Operand& op) { vpaddusw(x, x, op); }
void vpalignr(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_YMM | T_EVEX, 0x0F, imm); }
void vpalignr(const Xmm& x, const Operand& op, uint8 imm) { vpalignr(x, x, op, imm); }
void vpand(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0xDB); }
void vpand(const Xmm& x, const Operand& op) { vpand(x, x, op); }
void vpandn(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0xDF); }
void vpandn(const Xmm& x, const Operand& op) { vpandn(x, x, op); }
void vpavgb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xE0); }
void vpavgb(const Xmm& x, const Operand& op) { vpavgb(x, x, op); }
void vpavgw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xE3); }
void vpavgw(const Xmm& x, const Operand& op) { vpavgw(x, x, op); }
void vpcmpeqb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0x74); }
void vpcmpeqb(const Xmm& x, const Operand& op) { vpcmpeqb(x, x, op); }
void vpcmpeqw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0x75); }
void vpcmpeqw(const Xmm& x, const Operand& op) { vpcmpeqw(x, x, op); }
void vpcmpeqd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0x76); }
void vpcmpeqd(const Xmm& x, const Operand& op) { vpcmpeqd(x, x, op); }
void vpcmpeqq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x29); }
void vpcmpeqq(const Xmm& x, const Operand& op) { vpcmpeqq(x, x, op); }
void vpcmpgtb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0x64); }
void vpcmpgtb(const Xmm& x, const Operand& op) { vpcmpgtb(x, x, op); }
void vpcmpgtw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0x65); }
void vpcmpgtw(const Xmm& x, const Operand& op) { vpcmpgtw(x, x, op); }
void vpcmpgtd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0x66); }
void vpcmpgtd(const Xmm& x, const Operand& op) { vpcmpgtd(x, x, op); }
void vpcmpgtq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x37); }
void vpcmpgtq(const Xmm& x, const Operand& op) { vpcmpgtq(x, x, op); }
void vphaddw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x01); }
void vphaddw(const Xmm& x, const Operand& op) { vphaddw(x, x, op); }
void vphaddd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x02); }
void vphaddd(const Xmm& x, const Operand& op) { vphaddd(x, x, op); }
void vphaddsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x03); }
void vphaddsw(const Xmm& x, const Operand& op) { vphaddsw(x, x, op); }
void vphsubw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x05); }
void vphsubw(const Xmm& x, const Operand& op) { vphsubw(x, x, op); }
void vphsubd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x06); }
void vphsubd(const Xmm& x, const Operand& op) { vphsubd(x, x, op); }
void vphsubsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x07); }
void vphsubsw(const Xmm& x, const Operand& op) { vphsubsw(x, x, op); }
void vpmaddwd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xF5); }
void vpmaddwd(const Xmm& x, const Operand& op) { vpmaddwd(x, x, op); }
void vpmaddubsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM | T_EVEX, 0x04); }
void vpmaddubsw(const Xmm& x, const Operand& op) { vpmaddubsw(x, x, op); }
void vpmaxsb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM | T_EVEX, 0x3C); }
void vpmaxsb(const Xmm& x, const Operand& op) { vpmaxsb(x, x, op); }
void vpmaxsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xEE); }
void vpmaxsw(const Xmm& x, const Operand& op) { vpmaxsw(x, x, op); }
void vpmaxsd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x3D); }
void vpmaxsd(const Xmm& x, const Operand& op) { vpmaxsd(x, x, op); }
void vpmaxub(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xDE); }
void vpmaxub(const Xmm& x, const Operand& op) { vpmaxub(x, x, op); }
void vpmaxuw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM | T_EVEX, 0x3E); }
void vpmaxuw(const Xmm& x, const Operand& op) { vpmaxuw(x, x, op); }
void vpmaxud(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x3F); }
void vpmaxud(const Xmm& x, const Operand& op) { vpmaxud(x, x, op); }
void vpminsb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM | T_EVEX, 0x38); }
void vpminsb(const Xmm& x, const Operand& op) { vpminsb(x, x, op); }
void vpminsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xEA); }
void vpminsw(const Xmm& x, const Operand& op) { vpminsw(x, x, op); }
void vpminsd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x39); }
void vpminsd(const Xmm& x, const Operand& op) { vpminsd(x, x, op); }
void vpminub(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xDA); }
void vpminub(const Xmm& x, const Operand& op) { vpminub(x, x, op); }
void vpminuw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM | T_EVEX, 0x3A); }
void vpminuw(const Xmm& x, const Operand& op) { vpminuw(x, x, op); }
void vpminud(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x3B); }
void vpminud(const Xmm& x, const Operand& op) { vpminud(x, x, op); }
void vpmulhuw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xE4); }
void vpmulhuw(const Xmm& x, const Operand& op) { vpmulhuw(x, x, op); }
void vpmulhrsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM | T_EVEX, 0x0B); }
void vpmulhrsw(const Xmm& x, const Operand& op) { vpmulhrsw(x, x, op); }
void vpmulhw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xE5); }
void vpmulhw(const Xmm& x, const Operand& op) { vpmulhw(x, x, op); }
void vpmullw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xD5); }
void vpmullw(const Xmm& x, const Operand& op) { vpmullw(x, x, op); }
void vpmulld(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x40); }
void vpmulld(const Xmm& x, const Operand& op) { vpmulld(x, x, op); }
void vpmuludq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0xF4); }
void vpmuludq(const Xmm& x, const Operand& op) { vpmuludq(x, x, op); }
void vpmuldq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x28); }
void vpmuldq(const Xmm& x, const Operand& op) { vpmuldq(x, x, op); }
void vpor(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0xEB); }
void vpor(const Xmm& x, const Operand& op) { vpor(x, x, op); }
void vpsadbw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xF6); }
void vpsadbw(const Xmm& x, const Operand& op) { vpsadbw(x, x, op); }
void vpshufb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM | T_EVEX, 0x00); }
void vpsignb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x08); }
void vpsignb(const Xmm& x, const Operand& op) { vpsignb(x, x, op); }
void vpsignw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x09); }
void vpsignw(const Xmm& x, const Operand& op) { vpsignw(x, x, op); }
void vpsignd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_YMM, 0x0A); }
void vpsignd(const Xmm& x, const Operand& op) { vpsignd(x, x, op); }
void vpsllw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX | T_N16, 0xF1); }
void vpsllw(const Xmm& x, const Operand& op) { vpsllw(x, x, op); }
void vpslld(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_N16, 0xF2); }
void vpslld(const Xmm& x, const Operand& op) { vpslld(x, x, op); }
void vpsllq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_N16, 0xF3); }
void vpsllq(const Xmm& x, const Operand& op) { vpsllq(x, x, op); }
void vpsraw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX | T_N16, 0xE1); }
void vpsraw(const Xmm& x, const Operand& op) { vpsraw(x, x, op); }
void vpsrad(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_N16, 0xE2); }
void vpsrad(const Xmm& x, const Operand& op) { vpsrad(x, x, op); }
void vpsrlw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX | T_N16, 0xD1); }
void vpsrlw(const Xmm& x, const Operand& op) { vpsrlw(x, x, op); }
void vpsrld(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_N16, 0xD2); }
void vpsrld(const Xmm& x, const Operand& op) { vpsrld(x, x, op); }
void vpsrlq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_N16, 0xD3); }
void vpsrlq(const Xmm& x, const Operand& op) { vpsrlq(x, x, op); }
void vpsubb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xF8); }
void vpsubb(const Xmm& x, const Operand& op) { vpsubb(x, x, op); }
void vpsubw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xF9); }
void vpsubw(const Xmm& x, const Operand& op) { vpsubw(x, x, op); }
void vpsubd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0xFA); }
void vpsubd(const Xmm& x, const Operand& op) { vpsubd(x, x, op); }
void vpsubq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0xFB); }
void vpsubq(const Xmm& x, const Operand& op) { vpsubq(x, x, op); }
void vpsubsb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xE8); }
void vpsubsb(const Xmm& x, const Operand& op) { vpsubsb(x, x, op); }
void vpsubsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xE9); }
void vpsubsw(const Xmm& x, const Operand& op) { vpsubsw(x, x, op); }
void vpsubusb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xD8); }
void vpsubusb(const Xmm& x, const Operand& op) { vpsubusb(x, x, op); }
void vpsubusw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0xD9); }
void vpsubusw(const Xmm& x, const Operand& op) { vpsubusw(x, x, op); }
void vpunpckhbw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0x68); }
void vpunpckhbw(const Xmm& x, const Operand& op) { vpunpckhbw(x, x, op); }
void vpunpckhwd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0x69); }
void vpunpckhwd(const Xmm& x, const Operand& op) { vpunpckhwd(x, x, op); }
void vpunpckhdq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0x6A); }
void vpunpckhdq(const Xmm& x, const Operand& op) { vpunpckhdq(x, x, op); }
void vpunpckhqdq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0x6D); }
void vpunpckhqdq(const Xmm& x, const Operand& op) { vpunpckhqdq(x, x, op); }
void vpunpcklbw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0x60); }
void vpunpcklbw(const Xmm& x, const Operand& op) { vpunpcklbw(x, x, op); }
void vpunpcklwd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM | T_EVEX, 0x61); }
void vpunpcklwd(const Xmm& x, const Operand& op) { vpunpcklwd(x, x, op); }
void vpunpckldq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0x62); }
void vpunpckldq(const Xmm& x, const Operand& op) { vpunpckldq(x, x, op); }
void vpunpcklqdq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0x6C); }
void vpunpcklqdq(const Xmm& x, const Operand& op) { vpunpcklqdq(x, x, op); }
void vpxor(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_YMM, 0xEF); }
void vpxor(const Xmm& x, const Operand& op) { vpxor(x, x, op); }
void vrcpss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_F3 | T_0F, 0x53); }
void vrcpss(const Xmm& x, const Operand& op) { vrcpss(x, x, op); }
void vrsqrtss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_F3 | T_0F, 0x52); }
void vrsqrtss(const Xmm& x, const Operand& op) { vrsqrtss(x, x, op); }
void vshufpd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0xC6, imm); }
void vshufpd(const Xmm& x, const Operand& op, uint8 imm) { vshufpd(x, x, op, imm); }
void vshufps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0xC6, imm); }
void vshufps(const Xmm& x, const Operand& op, uint8 imm) { vshufps(x, x, op, imm); }
void vsqrtsd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_F2 | T_0F | T_EW1 | T_EVEX | T_ER_X | T_N8, 0x51); }
void vsqrtsd(const Xmm& x, const Operand& op) { vsqrtsd(x, x, op); }
void vsqrtss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_F3 | T_0F | T_EW0 | T_EVEX | T_ER_X | T_N4, 0x51); }
void vsqrtss(const Xmm& x, const Operand& op) { vsqrtss(x, x, op); }
void vunpckhpd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0x15); }
void vunpckhpd(const Xmm& x, const Operand& op) { vunpckhpd(x, x, op); }
void vunpckhps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0x15); }
void vunpckhps(const Xmm& x, const Operand& op) { vunpckhps(x, x, op); }
void vunpcklpd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0x14); }
void vunpcklpd(const Xmm& x, const Operand& op) { vunpcklpd(x, x, op); }
void vunpcklps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0x14); }
void vunpcklps(const Xmm& x, const Operand& op) { vunpcklps(x, x, op); }
void vaeskeygenassist(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F3A, 0xDF, imm); }
void vroundpd(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F3A | T_YMM, 0x09, imm); }
void vroundps(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F3A | T_YMM, 0x08, imm); }
void vpermilpd(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_EVEX | T_B64, 0x05, imm); }
void vpermilps(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_EVEX | T_B32, 0x04, imm); }
void vpcmpestri(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F3A, 0x61, imm); }
void vpcmpestrm(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F3A, 0x60, imm); }
void vpcmpistri(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F3A, 0x63, imm); }
void vpcmpistrm(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F3A, 0x62, imm); }
void vtestps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM, 0x0E); }
void vtestpd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM, 0x0F); }
void vcomisd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F | T_EW1 | T_EVEX | T_SAE_X | T_N8, 0x2F); }
void vcomiss(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_0F | T_EW0 | T_EVEX | T_SAE_X | T_N4, 0x2F); }
void vcvtdq2ps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x5B); }
void vcvtps2dq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x5B); }
void vcvttps2dq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_F3 | T_0F | T_EW0 | T_YMM | T_EVEX | T_SAE_Z | T_B32, 0x5B); }
void vmovapd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX, 0x28); }
void vmovaps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_0F | T_EW0 | T_YMM | T_EVEX, 0x28); }
void vmovddup(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_F2 | T_0F | T_EW1 | T_YMM | T_EVEX | T_ER_X | T_ER_Y | T_ER_Z | T_DUP, 0x12); }
void vmovdqa(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F | T_YMM, 0x6F); }
void vmovdqu(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_F3 | T_0F | T_YMM, 0x6F); }
void vmovshdup(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_F3 | T_0F | T_EW0 | T_YMM | T_EVEX, 0x16); }
void vmovsldup(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_F3 | T_0F | T_EW0 | T_YMM | T_EVEX, 0x12); }
void vmovupd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX, 0x10); }
void vmovups(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_0F | T_EW0 | T_YMM | T_EVEX, 0x10); }
void vpabsb(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX, 0x1C); }
void vpabsw(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX, 0x1D); }
void vpabsd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x1E); }
void vphminposuw(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38, 0x41); }
void vpmovsxbw(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x20); }
void vpmovsxbd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N4 | T_N_VL, 0x21); }
void vpmovsxbq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N2 | T_N_VL, 0x22); }
void vpmovsxwd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x23); }
void vpmovsxwq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N4 | T_N_VL, 0x24); }
void vpmovsxdq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x25); }
void vpmovzxbw(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x30); }
void vpmovzxbd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N4 | T_N_VL, 0x31); }
void vpmovzxbq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N2 | T_N_VL, 0x32); }
void vpmovzxwd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x33); }
void vpmovzxwq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_YMM | T_EVEX | T_N4 | T_N_VL, 0x34); }
void vpmovzxdq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x35); }
void vpshufd(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0x70, imm); }
void vpshufhw(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_F3 | T_0F | T_YMM | T_EVEX, 0x70, imm); }
void vpshuflw(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, T_F2 | T_0F | T_YMM | T_EVEX, 0x70, imm); }
void vptest(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F38, 0x17); }
void vrcpps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_0F | T_YMM, 0x53); }
void vrsqrtps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_0F | T_YMM, 0x52); }
void vsqrtpd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_ER_Z | T_B64, 0x51); }
void vsqrtps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_0F | T_EW0 | T_YMM | T_EVEX | T_ER_Z | T_B32, 0x51); }
void vucomisd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_66 | T_0F | T_EW1 | T_EVEX | T_SAE_X | T_N8, 0x2E); }
void vucomiss(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, T_0F | T_EW0 | T_EVEX | T_SAE_X | T_N4, 0x2E); }
void vmovapd(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX, 0x29); }
void vmovaps(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, T_0F | T_EW0 | T_YMM | T_EVEX, 0x29); }
void vmovdqa(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, T_66 | T_0F | T_YMM, 0x7F); }
void vmovdqu(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, T_F3 | T_0F | T_YMM, 0x7F); }
void vmovupd(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX, 0x11); }
void vmovups(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, T_0F | T_EW0 | T_YMM | T_EVEX, 0x11); }
void vaddsubpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_66 | T_0F | T_YMM, 0xD0); }
void vaddsubps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_F2 | T_0F | T_YMM, 0xD0); }
void vhaddpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_66 | T_0F | T_YMM, 0x7C); }
void vhaddps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_F2 | T_0F | T_YMM, 0x7C); }
void vhsubpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_66 | T_0F | T_YMM, 0x7D); }
void vhsubps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_F2 | T_0F | T_YMM, 0x7D); }
void vaesenc(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_66 | T_0F38 | T_W0, 0xDC); }
void vaesenclast(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_66 | T_0F38 | T_W0, 0xDD); }
void vaesdec(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_66 | T_0F38 | T_W0, 0xDE); }
void vaesdeclast(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, T_66 | T_0F38 | T_W0, 0xDF); }
void vmaskmovps(const Xmm& x1, const Xmm& x2, const Address& addr) { opAVX_X_X_XM(x1, x2, addr, T_0F38 | T_66 | T_W0 | T_YMM, 0x2C); }
void vmaskmovps(const Address& addr, const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x2, x1, addr, T_0F38 | T_66 | T_W0 | T_YMM, 0x2E); }
void vmaskmovpd(const Xmm& x1, const Xmm& x2, const Address& addr) { opAVX_X_X_XM(x1, x2, addr, T_0F38 | T_66 | T_W0 | T_YMM, 0x2D); }
void vmaskmovpd(const Address& addr, const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x2, x1, addr, T_0F38 | T_66 | T_W0 | T_YMM, 0x2F); }
void vpmaskmovd(const Xmm& x1, const Xmm& x2, const Address& addr) { opAVX_X_X_XM(x1, x2, addr, T_0F38 | T_66 | T_W0 | T_YMM, 0x8C); }
void vpmaskmovd(const Address& addr, const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x2, x1, addr, T_0F38 | T_66 | T_W0 | T_YMM, 0x8E); }
void vpmaskmovq(const Xmm& x1, const Xmm& x2, const Address& addr) { opAVX_X_X_XM(x1, x2, addr, T_0F38 | T_66 | T_W1 | T_YMM, 0x8C); }
void vpmaskmovq(const Address& addr, const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x2, x1, addr, T_0F38 | T_66 | T_W1 | T_YMM, 0x8E); }
void vpermd(const Ymm& y1, const Ymm& y2, const Operand& op) { opAVX_X_X_XM(y1, y2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x36); }
void vpermq(const Ymm& y1, const Ymm& y2, const Operand& op) { opAVX_X_X_XM(y1, y2, op, T_66 | T_0F38 | T_W0 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x36); }
void vpermps(const Ymm& y1, const Ymm& y2, const Operand& op) { opAVX_X_X_XM(y1, y2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x16); }
void vpermpd(const Ymm& y1, const Ymm& y2, const Operand& op) { opAVX_X_X_XM(y1, y2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x16); }
void vpermq(const Ymm& y, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(y, op, T_66 | T_0F3A | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x00, imm); }
void vpermpd(const Ymm& y, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(y, op, T_66 | T_0F3A | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x01, imm); }
void cmpeqpd(const Xmm& x, const Operand& op) { cmppd(x, op, 0); }
void vcmpeqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 0); }
void vcmpeqpd(const Xmm& x, const Operand& op) { vcmpeqpd(x, x, op); }
void cmpltpd(const Xmm& x, const Operand& op) { cmppd(x, op, 1); }
void vcmpltpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 1); }
void vcmpltpd(const Xmm& x, const Operand& op) { vcmpltpd(x, x, op); }
void cmplepd(const Xmm& x, const Operand& op) { cmppd(x, op, 2); }
void vcmplepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 2); }
void vcmplepd(const Xmm& x, const Operand& op) { vcmplepd(x, x, op); }
void cmpunordpd(const Xmm& x, const Operand& op) { cmppd(x, op, 3); }
void vcmpunordpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 3); }
void vcmpunordpd(const Xmm& x, const Operand& op) { vcmpunordpd(x, x, op); }
void cmpneqpd(const Xmm& x, const Operand& op) { cmppd(x, op, 4); }
void vcmpneqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 4); }
void vcmpneqpd(const Xmm& x, const Operand& op) { vcmpneqpd(x, x, op); }
void cmpnltpd(const Xmm& x, const Operand& op) { cmppd(x, op, 5); }
void vcmpnltpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 5); }
void vcmpnltpd(const Xmm& x, const Operand& op) { vcmpnltpd(x, x, op); }
void cmpnlepd(const Xmm& x, const Operand& op) { cmppd(x, op, 6); }
void vcmpnlepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 6); }
void vcmpnlepd(const Xmm& x, const Operand& op) { vcmpnlepd(x, x, op); }
void cmpordpd(const Xmm& x, const Operand& op) { cmppd(x, op, 7); }
void vcmpordpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 7); }
void vcmpordpd(const Xmm& x, const Operand& op) { vcmpordpd(x, x, op); }
void vcmpeq_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 8); }
void vcmpeq_uqpd(const Xmm& x, const Operand& op) { vcmpeq_uqpd(x, x, op); }
void vcmpngepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 9); }
void vcmpngepd(const Xmm& x, const Operand& op) { vcmpngepd(x, x, op); }
void vcmpngtpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 10); }
void vcmpngtpd(const Xmm& x, const Operand& op) { vcmpngtpd(x, x, op); }
void vcmpfalsepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 11); }
void vcmpfalsepd(const Xmm& x, const Operand& op) { vcmpfalsepd(x, x, op); }
void vcmpneq_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 12); }
void vcmpneq_oqpd(const Xmm& x, const Operand& op) { vcmpneq_oqpd(x, x, op); }
void vcmpgepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 13); }
void vcmpgepd(const Xmm& x, const Operand& op) { vcmpgepd(x, x, op); }
void vcmpgtpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 14); }
void vcmpgtpd(const Xmm& x, const Operand& op) { vcmpgtpd(x, x, op); }
void vcmptruepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 15); }
void vcmptruepd(const Xmm& x, const Operand& op) { vcmptruepd(x, x, op); }
void vcmpeq_ospd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 16); }
void vcmpeq_ospd(const Xmm& x, const Operand& op) { vcmpeq_ospd(x, x, op); }
void vcmplt_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 17); }
void vcmplt_oqpd(const Xmm& x, const Operand& op) { vcmplt_oqpd(x, x, op); }
void vcmple_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 18); }
void vcmple_oqpd(const Xmm& x, const Operand& op) { vcmple_oqpd(x, x, op); }
void vcmpunord_spd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 19); }
void vcmpunord_spd(const Xmm& x, const Operand& op) { vcmpunord_spd(x, x, op); }
void vcmpneq_uspd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 20); }
void vcmpneq_uspd(const Xmm& x, const Operand& op) { vcmpneq_uspd(x, x, op); }
void vcmpnlt_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 21); }
void vcmpnlt_uqpd(const Xmm& x, const Operand& op) { vcmpnlt_uqpd(x, x, op); }
void vcmpnle_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 22); }
void vcmpnle_uqpd(const Xmm& x, const Operand& op) { vcmpnle_uqpd(x, x, op); }
void vcmpord_spd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 23); }
void vcmpord_spd(const Xmm& x, const Operand& op) { vcmpord_spd(x, x, op); }
void vcmpeq_uspd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 24); }
void vcmpeq_uspd(const Xmm& x, const Operand& op) { vcmpeq_uspd(x, x, op); }
void vcmpnge_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 25); }
void vcmpnge_uqpd(const Xmm& x, const Operand& op) { vcmpnge_uqpd(x, x, op); }
void vcmpngt_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 26); }
void vcmpngt_uqpd(const Xmm& x, const Operand& op) { vcmpngt_uqpd(x, x, op); }
void vcmpfalse_ospd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 27); }
void vcmpfalse_ospd(const Xmm& x, const Operand& op) { vcmpfalse_ospd(x, x, op); }
void vcmpneq_ospd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 28); }
void vcmpneq_ospd(const Xmm& x, const Operand& op) { vcmpneq_ospd(x, x, op); }
void vcmpge_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 29); }
void vcmpge_oqpd(const Xmm& x, const Operand& op) { vcmpge_oqpd(x, x, op); }
void vcmpgt_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 30); }
void vcmpgt_oqpd(const Xmm& x, const Operand& op) { vcmpgt_oqpd(x, x, op); }
void vcmptrue_uspd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 31); }
void vcmptrue_uspd(const Xmm& x, const Operand& op) { vcmptrue_uspd(x, x, op); }
void cmpeqps(const Xmm& x, const Operand& op) { cmpps(x, op, 0); }
void vcmpeqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 0); }
void vcmpeqps(const Xmm& x, const Operand& op) { vcmpeqps(x, x, op); }
void cmpltps(const Xmm& x, const Operand& op) { cmpps(x, op, 1); }
void vcmpltps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 1); }
void vcmpltps(const Xmm& x, const Operand& op) { vcmpltps(x, x, op); }
void cmpleps(const Xmm& x, const Operand& op) { cmpps(x, op, 2); }
void vcmpleps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 2); }
void vcmpleps(const Xmm& x, const Operand& op) { vcmpleps(x, x, op); }
void cmpunordps(const Xmm& x, const Operand& op) { cmpps(x, op, 3); }
void vcmpunordps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 3); }
void vcmpunordps(const Xmm& x, const Operand& op) { vcmpunordps(x, x, op); }
void cmpneqps(const Xmm& x, const Operand& op) { cmpps(x, op, 4); }
void vcmpneqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 4); }
void vcmpneqps(const Xmm& x, const Operand& op) { vcmpneqps(x, x, op); }
void cmpnltps(const Xmm& x, const Operand& op) { cmpps(x, op, 5); }
void vcmpnltps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 5); }
void vcmpnltps(const Xmm& x, const Operand& op) { vcmpnltps(x, x, op); }
void cmpnleps(const Xmm& x, const Operand& op) { cmpps(x, op, 6); }
void vcmpnleps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 6); }
void vcmpnleps(const Xmm& x, const Operand& op) { vcmpnleps(x, x, op); }
void cmpordps(const Xmm& x, const Operand& op) { cmpps(x, op, 7); }
void vcmpordps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 7); }
void vcmpordps(const Xmm& x, const Operand& op) { vcmpordps(x, x, op); }
void vcmpeq_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 8); }
void vcmpeq_uqps(const Xmm& x, const Operand& op) { vcmpeq_uqps(x, x, op); }
void vcmpngeps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 9); }
void vcmpngeps(const Xmm& x, const Operand& op) { vcmpngeps(x, x, op); }
void vcmpngtps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 10); }
void vcmpngtps(const Xmm& x, const Operand& op) { vcmpngtps(x, x, op); }
void vcmpfalseps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 11); }
void vcmpfalseps(const Xmm& x, const Operand& op) { vcmpfalseps(x, x, op); }
void vcmpneq_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 12); }
void vcmpneq_oqps(const Xmm& x, const Operand& op) { vcmpneq_oqps(x, x, op); }
void vcmpgeps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 13); }
void vcmpgeps(const Xmm& x, const Operand& op) { vcmpgeps(x, x, op); }
void vcmpgtps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 14); }
void vcmpgtps(const Xmm& x, const Operand& op) { vcmpgtps(x, x, op); }
void vcmptrueps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 15); }
void vcmptrueps(const Xmm& x, const Operand& op) { vcmptrueps(x, x, op); }
void vcmpeq_osps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 16); }
void vcmpeq_osps(const Xmm& x, const Operand& op) { vcmpeq_osps(x, x, op); }
void vcmplt_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 17); }
void vcmplt_oqps(const Xmm& x, const Operand& op) { vcmplt_oqps(x, x, op); }
void vcmple_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 18); }
void vcmple_oqps(const Xmm& x, const Operand& op) { vcmple_oqps(x, x, op); }
void vcmpunord_sps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 19); }
void vcmpunord_sps(const Xmm& x, const Operand& op) { vcmpunord_sps(x, x, op); }
void vcmpneq_usps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 20); }
void vcmpneq_usps(const Xmm& x, const Operand& op) { vcmpneq_usps(x, x, op); }
void vcmpnlt_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 21); }
void vcmpnlt_uqps(const Xmm& x, const Operand& op) { vcmpnlt_uqps(x, x, op); }
void vcmpnle_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 22); }
void vcmpnle_uqps(const Xmm& x, const Operand& op) { vcmpnle_uqps(x, x, op); }
void vcmpord_sps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 23); }
void vcmpord_sps(const Xmm& x, const Operand& op) { vcmpord_sps(x, x, op); }
void vcmpeq_usps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 24); }
void vcmpeq_usps(const Xmm& x, const Operand& op) { vcmpeq_usps(x, x, op); }
void vcmpnge_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 25); }
void vcmpnge_uqps(const Xmm& x, const Operand& op) { vcmpnge_uqps(x, x, op); }
void vcmpngt_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 26); }
void vcmpngt_uqps(const Xmm& x, const Operand& op) { vcmpngt_uqps(x, x, op); }
void vcmpfalse_osps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 27); }
void vcmpfalse_osps(const Xmm& x, const Operand& op) { vcmpfalse_osps(x, x, op); }
void vcmpneq_osps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 28); }
void vcmpneq_osps(const Xmm& x, const Operand& op) { vcmpneq_osps(x, x, op); }
void vcmpge_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 29); }
void vcmpge_oqps(const Xmm& x, const Operand& op) { vcmpge_oqps(x, x, op); }
void vcmpgt_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 30); }
void vcmpgt_oqps(const Xmm& x, const Operand& op) { vcmpgt_oqps(x, x, op); }
void vcmptrue_usps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 31); }
void vcmptrue_usps(const Xmm& x, const Operand& op) { vcmptrue_usps(x, x, op); }
void cmpeqsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 0); }
void vcmpeqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 0); }
void vcmpeqsd(const Xmm& x, const Operand& op) { vcmpeqsd(x, x, op); }
void cmpltsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 1); }
void vcmpltsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 1); }
void vcmpltsd(const Xmm& x, const Operand& op) { vcmpltsd(x, x, op); }
void cmplesd(const Xmm& x, const Operand& op) { cmpsd(x, op, 2); }
void vcmplesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 2); }
void vcmplesd(const Xmm& x, const Operand& op) { vcmplesd(x, x, op); }
void cmpunordsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 3); }
void vcmpunordsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 3); }
void vcmpunordsd(const Xmm& x, const Operand& op) { vcmpunordsd(x, x, op); }
void cmpneqsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 4); }
void vcmpneqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 4); }
void vcmpneqsd(const Xmm& x, const Operand& op) { vcmpneqsd(x, x, op); }
void cmpnltsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 5); }
void vcmpnltsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 5); }
void vcmpnltsd(const Xmm& x, const Operand& op) { vcmpnltsd(x, x, op); }
void cmpnlesd(const Xmm& x, const Operand& op) { cmpsd(x, op, 6); }
void vcmpnlesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 6); }
void vcmpnlesd(const Xmm& x, const Operand& op) { vcmpnlesd(x, x, op); }
void cmpordsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 7); }
void vcmpordsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 7); }
void vcmpordsd(const Xmm& x, const Operand& op) { vcmpordsd(x, x, op); }
void vcmpeq_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 8); }
void vcmpeq_uqsd(const Xmm& x, const Operand& op) { vcmpeq_uqsd(x, x, op); }
void vcmpngesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 9); }
void vcmpngesd(const Xmm& x, const Operand& op) { vcmpngesd(x, x, op); }
void vcmpngtsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 10); }
void vcmpngtsd(const Xmm& x, const Operand& op) { vcmpngtsd(x, x, op); }
void vcmpfalsesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 11); }
void vcmpfalsesd(const Xmm& x, const Operand& op) { vcmpfalsesd(x, x, op); }
void vcmpneq_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 12); }
void vcmpneq_oqsd(const Xmm& x, const Operand& op) { vcmpneq_oqsd(x, x, op); }
void vcmpgesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 13); }
void vcmpgesd(const Xmm& x, const Operand& op) { vcmpgesd(x, x, op); }
void vcmpgtsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 14); }
void vcmpgtsd(const Xmm& x, const Operand& op) { vcmpgtsd(x, x, op); }
void vcmptruesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 15); }
void vcmptruesd(const Xmm& x, const Operand& op) { vcmptruesd(x, x, op); }
void vcmpeq_ossd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 16); }
void vcmpeq_ossd(const Xmm& x, const Operand& op) { vcmpeq_ossd(x, x, op); }
void vcmplt_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 17); }
void vcmplt_oqsd(const Xmm& x, const Operand& op) { vcmplt_oqsd(x, x, op); }
void vcmple_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 18); }
void vcmple_oqsd(const Xmm& x, const Operand& op) { vcmple_oqsd(x, x, op); }
void vcmpunord_ssd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 19); }
void vcmpunord_ssd(const Xmm& x, const Operand& op) { vcmpunord_ssd(x, x, op); }
void vcmpneq_ussd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 20); }
void vcmpneq_ussd(const Xmm& x, const Operand& op) { vcmpneq_ussd(x, x, op); }
void vcmpnlt_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 21); }
void vcmpnlt_uqsd(const Xmm& x, const Operand& op) { vcmpnlt_uqsd(x, x, op); }
void vcmpnle_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 22); }
void vcmpnle_uqsd(const Xmm& x, const Operand& op) { vcmpnle_uqsd(x, x, op); }
void vcmpord_ssd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 23); }
void vcmpord_ssd(const Xmm& x, const Operand& op) { vcmpord_ssd(x, x, op); }
void vcmpeq_ussd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 24); }
void vcmpeq_ussd(const Xmm& x, const Operand& op) { vcmpeq_ussd(x, x, op); }
void vcmpnge_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 25); }
void vcmpnge_uqsd(const Xmm& x, const Operand& op) { vcmpnge_uqsd(x, x, op); }
void vcmpngt_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 26); }
void vcmpngt_uqsd(const Xmm& x, const Operand& op) { vcmpngt_uqsd(x, x, op); }
void vcmpfalse_ossd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 27); }
void vcmpfalse_ossd(const Xmm& x, const Operand& op) { vcmpfalse_ossd(x, x, op); }
void vcmpneq_ossd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 28); }
void vcmpneq_ossd(const Xmm& x, const Operand& op) { vcmpneq_ossd(x, x, op); }
void vcmpge_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 29); }
void vcmpge_oqsd(const Xmm& x, const Operand& op) { vcmpge_oqsd(x, x, op); }
void vcmpgt_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 30); }
void vcmpgt_oqsd(const Xmm& x, const Operand& op) { vcmpgt_oqsd(x, x, op); }
void vcmptrue_ussd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 31); }
void vcmptrue_ussd(const Xmm& x, const Operand& op) { vcmptrue_ussd(x, x, op); }
void cmpeqss(const Xmm& x, const Operand& op) { cmpss(x, op, 0); }
void vcmpeqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 0); }
void vcmpeqss(const Xmm& x, const Operand& op) { vcmpeqss(x, x, op); }
void cmpltss(const Xmm& x, const Operand& op) { cmpss(x, op, 1); }
void vcmpltss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 1); }
void vcmpltss(const Xmm& x, const Operand& op) { vcmpltss(x, x, op); }
void cmpless(const Xmm& x, const Operand& op) { cmpss(x, op, 2); }
void vcmpless(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 2); }
void vcmpless(const Xmm& x, const Operand& op) { vcmpless(x, x, op); }
void cmpunordss(const Xmm& x, const Operand& op) { cmpss(x, op, 3); }
void vcmpunordss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 3); }
void vcmpunordss(const Xmm& x, const Operand& op) { vcmpunordss(x, x, op); }
void cmpneqss(const Xmm& x, const Operand& op) { cmpss(x, op, 4); }
void vcmpneqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 4); }
void vcmpneqss(const Xmm& x, const Operand& op) { vcmpneqss(x, x, op); }
void cmpnltss(const Xmm& x, const Operand& op) { cmpss(x, op, 5); }
void vcmpnltss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 5); }
void vcmpnltss(const Xmm& x, const Operand& op) { vcmpnltss(x, x, op); }
void cmpnless(const Xmm& x, const Operand& op) { cmpss(x, op, 6); }
void vcmpnless(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 6); }
void vcmpnless(const Xmm& x, const Operand& op) { vcmpnless(x, x, op); }
void cmpordss(const Xmm& x, const Operand& op) { cmpss(x, op, 7); }
void vcmpordss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 7); }
void vcmpordss(const Xmm& x, const Operand& op) { vcmpordss(x, x, op); }
void vcmpeq_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 8); }
void vcmpeq_uqss(const Xmm& x, const Operand& op) { vcmpeq_uqss(x, x, op); }
void vcmpngess(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 9); }
void vcmpngess(const Xmm& x, const Operand& op) { vcmpngess(x, x, op); }
void vcmpngtss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 10); }
void vcmpngtss(const Xmm& x, const Operand& op) { vcmpngtss(x, x, op); }
void vcmpfalsess(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 11); }
void vcmpfalsess(const Xmm& x, const Operand& op) { vcmpfalsess(x, x, op); }
void vcmpneq_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 12); }
void vcmpneq_oqss(const Xmm& x, const Operand& op) { vcmpneq_oqss(x, x, op); }
void vcmpgess(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 13); }
void vcmpgess(const Xmm& x, const Operand& op) { vcmpgess(x, x, op); }
void vcmpgtss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 14); }
void vcmpgtss(const Xmm& x, const Operand& op) { vcmpgtss(x, x, op); }
void vcmptruess(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 15); }
void vcmptruess(const Xmm& x, const Operand& op) { vcmptruess(x, x, op); }
void vcmpeq_osss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 16); }
void vcmpeq_osss(const Xmm& x, const Operand& op) { vcmpeq_osss(x, x, op); }
void vcmplt_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 17); }
void vcmplt_oqss(const Xmm& x, const Operand& op) { vcmplt_oqss(x, x, op); }
void vcmple_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 18); }
void vcmple_oqss(const Xmm& x, const Operand& op) { vcmple_oqss(x, x, op); }
void vcmpunord_sss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 19); }
void vcmpunord_sss(const Xmm& x, const Operand& op) { vcmpunord_sss(x, x, op); }
void vcmpneq_usss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 20); }
void vcmpneq_usss(const Xmm& x, const Operand& op) { vcmpneq_usss(x, x, op); }
void vcmpnlt_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 21); }
void vcmpnlt_uqss(const Xmm& x, const Operand& op) { vcmpnlt_uqss(x, x, op); }
void vcmpnle_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 22); }
void vcmpnle_uqss(const Xmm& x, const Operand& op) { vcmpnle_uqss(x, x, op); }
void vcmpord_sss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 23); }
void vcmpord_sss(const Xmm& x, const Operand& op) { vcmpord_sss(x, x, op); }
void vcmpeq_usss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 24); }
void vcmpeq_usss(const Xmm& x, const Operand& op) { vcmpeq_usss(x, x, op); }
void vcmpnge_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 25); }
void vcmpnge_uqss(const Xmm& x, const Operand& op) { vcmpnge_uqss(x, x, op); }
void vcmpngt_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 26); }
void vcmpngt_uqss(const Xmm& x, const Operand& op) { vcmpngt_uqss(x, x, op); }
void vcmpfalse_osss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 27); }
void vcmpfalse_osss(const Xmm& x, const Operand& op) { vcmpfalse_osss(x, x, op); }
void vcmpneq_osss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 28); }
void vcmpneq_osss(const Xmm& x, const Operand& op) { vcmpneq_osss(x, x, op); }
void vcmpge_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 29); }
void vcmpge_oqss(const Xmm& x, const Operand& op) { vcmpge_oqss(x, x, op); }
void vcmpgt_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 30); }
void vcmpgt_oqss(const Xmm& x, const Operand& op) { vcmpgt_oqss(x, x, op); }
void vcmptrue_usss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 31); }
void vcmptrue_usss(const Xmm& x, const Operand& op) { vcmptrue_usss(x, x, op); }
void vmovhpd(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !op2.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, op1, op2, T_0F | T_66 | T_EVEX | T_EW1 | T_N8, 0x16); }
void vmovhpd(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, T_0F | T_66 | T_EVEX | T_EW1 | T_N8, 0x17); }
void vmovhps(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !op2.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, op1, op2, T_0F | T_EVEX | T_EW0 | T_N8, 0x16); }
void vmovhps(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, T_0F | T_EVEX | T_EW0 | T_N8, 0x17); }
void vmovlpd(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !op2.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, op1, op2, T_0F | T_66 | T_EVEX | T_EW1 | T_N8, 0x12); }
void vmovlpd(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, T_0F | T_66 | T_EVEX | T_EW1 | T_N8, 0x13); }
void vmovlps(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !op2.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, op1, op2, T_0F | T_EVEX | T_EW0 | T_N8, 0x12); }
void vmovlps(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, T_0F | T_EVEX | T_EW0 | T_N8, 0x13); }
void vfmadd132pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x98); }
void vfmadd213pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xA8); }
void vfmadd231pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xB8); }
void vfmadd132ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x98); }
void vfmadd213ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xA8); }
void vfmadd231ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xB8); }
void vfmadd132sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0x99); }
void vfmadd213sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0xA9); }
void vfmadd231sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0xB9); }
void vfmadd132ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0x99); }
void vfmadd213ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0xA9); }
void vfmadd231ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0xB9); }
void vfmaddsub132pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x96); }
void vfmaddsub213pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xA6); }
void vfmaddsub231pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xB6); }
void vfmaddsub132ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x96); }
void vfmaddsub213ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xA6); }
void vfmaddsub231ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xB6); }
void vfmsubadd132pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x97); }
void vfmsubadd213pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xA7); }
void vfmsubadd231pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xB7); }
void vfmsubadd132ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x97); }
void vfmsubadd213ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xA7); }
void vfmsubadd231ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xB7); }
void vfmsub132pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x9A); }
void vfmsub213pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xAA); }
void vfmsub231pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xBA); }
void vfmsub132ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x9A); }
void vfmsub213ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xAA); }
void vfmsub231ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xBA); }
void vfmsub132sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0x9B); }
void vfmsub213sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0xAB); }
void vfmsub231sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0xBB); }
void vfmsub132ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0x9B); }
void vfmsub213ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0xAB); }
void vfmsub231ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0xBB); }
void vfnmadd132pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x9C); }
void vfnmadd213pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xAC); }
void vfnmadd231pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xBC); }
void vfnmadd132ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x9C); }
void vfnmadd213ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xAC); }
void vfnmadd231ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xBC); }
void vfnmadd132sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0x9D); }
void vfnmadd213sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0xAD); }
void vfnmadd231sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0xBD); }
void vfnmadd132ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0x9D); }
void vfnmadd213ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0xAD); }
void vfnmadd231ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0xBD); }
void vfnmsub132pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0x9E); }
void vfnmsub213pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xAE); }
void vfnmsub231pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_YMM | T_EVEX | T_B64, 0xBE); }
void vfnmsub132ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0x9E); }
void vfnmsub213ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xAE); }
void vfnmsub231ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_YMM | T_EVEX | T_B32, 0xBE); }
void vfnmsub132sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0x9F); }
void vfnmsub213sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0xAF); }
void vfnmsub231sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W1 | T_EW1 | T_EVEX | T_ER_X | T_N8, 0xBF); }
void vfnmsub132ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0x9F); }
void vfnmsub213ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0xAF); }
void vfnmsub231ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_W0 | T_EW0 | T_EVEX | T_ER_X | T_N4, 0xBF); }
void vaesimc(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_0F38 | T_66 | T_W0, 0xDB); }
void vbroadcastf128(const Ymm& y, const Address& addr) { opAVX_X_XM_IMM(y, addr, T_0F38 | T_66 | T_W0 | T_YMM, 0x1A); }
void vbroadcasti128(const Ymm& y, const Address& addr) { opAVX_X_XM_IMM(y, addr, T_0F38 | T_66 | T_W0 | T_YMM, 0x5A); }
void vbroadcastsd(const Ymm& y, const Operand& op) { if (!op.isMEM() && !(y.isYMM() && op.isXMM()) && !(y.isZMM() && op.isXMM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(y, op, T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_EW1 | T_N8, 0x19); }
void vbroadcastss(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_N4, 0x18); }
void vpbroadcastb(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_N4, 0x78); }
void vpbroadcastw(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_N4, 0x79); }
void vpbroadcastd(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_N4, 0x58); }
void vpbroadcastq(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_W0 | T_EW1 | T_YMM | T_EVEX | T_N8, 0x59); }
void vextractf128(const Operand& op, const Ymm& y, uint8 imm) { if (!(op.isXMEM() && y.isYMM())) throw Error(ERR_BAD_COMBINATION); opVex(y, 0, op, T_0F3A | T_66 | T_W0 | T_YMM, 0x19, imm); }
void vextracti128(const Operand& op, const Ymm& y, uint8 imm) { if (!(op.isXMEM() && y.isYMM())) throw Error(ERR_BAD_COMBINATION); opVex(y, 0, op, T_0F3A | T_66 | T_W0 | T_YMM, 0x39, imm); }
void vextractps(const Operand& op, const Xmm& x, uint8 imm) { if (!((op.isREG(32) || op.isMEM()) && x.isXMM())) throw Error(ERR_BAD_COMBINATION); opVex(x, 0, op, T_0F3A | T_66 | T_W0 | T_EVEX | T_N4, 0x17, imm); }
void vinsertf128(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { if (!(y1.isYMM() && y2.isYMM() && op.isXMEM())) throw Error(ERR_BAD_COMBINATION); opVex(y1, &y2, op, T_0F3A | T_66 | T_W0 | T_YMM, 0x18, imm); }
void vinserti128(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { if (!(y1.isYMM() && y2.isYMM() && op.isXMEM())) throw Error(ERR_BAD_COMBINATION); opVex(y1, &y2, op, T_0F3A | T_66 | T_W0 | T_YMM, 0x38, imm); }
void vperm2f128(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { if (!(y1.isYMM() && y2.isYMM() && op.isYMEM())) throw Error(ERR_BAD_COMBINATION); opVex(y1, &y2, op, T_0F3A | T_66 | T_W0 | T_YMM, 0x06, imm); }
void vperm2i128(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { if (!(y1.isYMM() && y2.isYMM() && op.isYMEM())) throw Error(ERR_BAD_COMBINATION); opVex(y1, &y2, op, T_0F3A | T_66 | T_W0 | T_YMM, 0x46, imm); }
void vlddqu(const Xmm& x, const Address& addr) { opAVX_X_X_XM(x, cvtIdx0(x), addr, T_0F | T_F2 | T_W0 | T_YMM, 0xF0); }
void vldmxcsr(const Address& addr) { opAVX_X_X_XM(xm2, xm0, addr, T_0F, 0xAE); }
void vstmxcsr(const Address& addr) { opAVX_X_X_XM(xm3, xm0, addr, T_0F, 0xAE); }
void vmaskmovdqu(const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x1, xm0, x2, T_0F | T_66, 0xF7); }
void vpextrb(const Operand& op, const Xmm& x, uint8 imm) { if (!((op.isREG(8|16|i32e) || op.isMEM()) && x.isXMM())) throw Error(ERR_BAD_COMBINATION); opVex(x, 0, op, T_0F3A | T_66 | T_EVEX | T_N4, 0x14, imm); }
void vpextrw(const Operand& op, const Xmm& x, uint8 imm) { if (!((op.isREG(16|i32e) || op.isMEM()) && x.isXMM())) throw Error(ERR_BAD_COMBINATION); if (op.isREG() && x.getIdx() < 16) { opAVX_X_X_XM(Xmm(op.getIdx()), xm0, x, T_0F | T_66, 0xC5, imm); } else { opVex(x, 0, op, T_0F3A | T_66 | T_EVEX | T_N4, 0x15, imm); } }
void vpextrd(const Operand& op, const Xmm& x, uint8 imm) { if (!((op.isREG(32) || op.isMEM()) && x.isXMM())) throw Error(ERR_BAD_COMBINATION); opVex(x, 0, op, T_0F3A | T_66 | T_W0 | T_EVEX | T_EW0 | T_N4, 0x16, imm); }
void vpinsrb(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { if (!(x1.isXMM() && x2.isXMM() && (op.isREG(32) || op.isMEM()))) throw Error(ERR_BAD_COMBINATION); opVex(x1, &x2, op, T_0F3A | T_66 | T_EVEX | T_N4, 0x20, imm); }
void vpinsrb(const Xmm& x, const Operand& op, uint8 imm) { vpinsrb(x, x, op, imm); }
void vpinsrw(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { if (!(x1.isXMM() && x2.isXMM() && (op.isREG(32) || op.isMEM()))) throw Error(ERR_BAD_COMBINATION); opVex(x1, &x2, op, T_0F | T_66 | T_EVEX | T_N4, 0xC4, imm); }
void vpinsrw(const Xmm& x, const Operand& op, uint8 imm) { vpinsrw(x, x, op, imm); }
void vpinsrd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { if (!(x1.isXMM() && x2.isXMM() && (op.isREG(32) || op.isMEM()))) throw Error(ERR_BAD_COMBINATION); opVex(x1, &x2, op, T_0F3A | T_66 | T_W0 | T_EVEX | T_EW0 | T_N4, 0x22, imm); }
void vpinsrd(const Xmm& x, const Operand& op, uint8 imm) { vpinsrd(x, x, op, imm); }
void vpmovmskb(const Reg32e& r, const Xmm& x) { if (!x.is(Operand::XMM | Operand::YMM)) throw Error(ERR_BAD_COMBINATION); opVex(x.isYMM() ? Ymm(r.getIdx()) : Xmm(r.getIdx()), 0, x, T_0F | T_66 | T_YMM, 0xD7); }
void vpslldq(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 7), x, op, T_66 | T_0F | T_YMM | T_EVEX, 0x73, imm); }
void vpslldq(const Xmm& x, uint8 imm) { vpslldq(x, x, imm); }
void vpsrldq(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 3), x, op, T_66 | T_0F | T_YMM | T_EVEX, 0x73, imm); }
void vpsrldq(const Xmm& x, uint8 imm) { vpsrldq(x, x, imm); }
void vpsllw(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 6), x, op, T_66 | T_0F | T_YMM | T_EVEX, 0x71, imm); }
void vpsllw(const Xmm& x, uint8 imm) { vpsllw(x, x, imm); }
void vpslld(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 6), x, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0x72, imm); }
void vpslld(const Xmm& x, uint8 imm) { vpslld(x, x, imm); }
void vpsllq(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 6), x, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0x73, imm); }
void vpsllq(const Xmm& x, uint8 imm) { vpsllq(x, x, imm); }
void vpsraw(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 4), x, op, T_66 | T_0F | T_YMM | T_EVEX, 0x71, imm); }
void vpsraw(const Xmm& x, uint8 imm) { vpsraw(x, x, imm); }
void vpsrad(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 4), x, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0x72, imm); }
void vpsrad(const Xmm& x, uint8 imm) { vpsrad(x, x, imm); }
void vpsrlw(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 2), x, op, T_66 | T_0F | T_YMM | T_EVEX, 0x71, imm); }
void vpsrlw(const Xmm& x, uint8 imm) { vpsrlw(x, x, imm); }
void vpsrld(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 2), x, op, T_66 | T_0F | T_EW0 | T_YMM | T_EVEX | T_B32, 0x72, imm); }
void vpsrld(const Xmm& x, uint8 imm) { vpsrld(x, x, imm); }
void vpsrlq(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 2), x, op, T_66 | T_0F | T_EW1 | T_YMM | T_EVEX | T_B64, 0x73, imm); }
void vpsrlq(const Xmm& x, uint8 imm) { vpsrlq(x, x, imm); }
void vblendvpd(const Xmm& x1, const Xmm& x2, const Operand& op, const Xmm& x4) { opAVX_X_X_XM(x1, x2, op, T_0F3A | T_66 | T_YMM, 0x4B, x4.getIdx() << 4); }
void vblendvpd(const Xmm& x1, const Operand& op, const Xmm& x4) { vblendvpd(x1, x1, op, x4); }
void vblendvps(const Xmm& x1, const Xmm& x2, const Operand& op, const Xmm& x4) { opAVX_X_X_XM(x1, x2, op, T_0F3A | T_66 | T_YMM, 0x4A, x4.getIdx() << 4); }
void vblendvps(const Xmm& x1, const Operand& op, const Xmm& x4) { vblendvps(x1, x1, op, x4); }
void vpblendvb(const Xmm& x1, const Xmm& x2, const Operand& op, const Xmm& x4) { opAVX_X_X_XM(x1, x2, op, T_0F3A | T_66 | T_YMM, 0x4C, x4.getIdx() << 4); }
void vpblendvb(const Xmm& x1, const Operand& op, const Xmm& x4) { vpblendvb(x1, x1, op, x4); }
void vmovd(const Xmm& x, const Operand& op) { if (!op.isREG(32) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, xm0, op, T_0F | T_66 | T_W0 | T_EVEX | T_N4, 0x6E); }
void vmovd(const Operand& op, const Xmm& x) { if (!op.isREG(32) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, xm0, op, T_0F | T_66 | T_W0 | T_EVEX | T_N4, 0x7E); }
void vmovq(const Xmm& x, const Address& addr) { int type, code; if (x.getIdx() < 16) { type = T_0F | T_F3; code = 0x7E; } else { type = T_0F | T_66 | T_EVEX | T_EW1 | T_N8; code = 0x6E; } opAVX_X_X_XM(x, xm0, addr, type, code); }
void vmovq(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, T_0F | T_66 | T_EVEX | T_EW1 | T_N8, x.getIdx() < 16 ? 0xD6 : 0x7E); }
void vmovq(const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x1, xm0, x2, T_0F | T_F3 | T_EVEX | T_EW1 | T_N8, 0x7E); }
void vmovhlps(const Xmm& x1, const Xmm& x2, const Operand& op = Operand()) { if (!op.isNone() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x1, x2, op, T_0F | T_EVEX | T_EW0, 0x12); }
void vmovlhps(const Xmm& x1, const Xmm& x2, const Operand& op = Operand()) { if (!op.isNone() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x1, x2, op, T_0F | T_EVEX | T_EW0, 0x16); }
void vmovmskpd(const Reg& r, const Xmm& x) { if (!r.isBit(i32e)) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x.isXMM() ? Xmm(r.getIdx()) : Ymm(r.getIdx()), cvtIdx0(x), x, T_0F | T_66 | T_W0 | T_YMM, 0x50); }
void vmovmskps(const Reg& r, const Xmm& x) { if (!r.isBit(i32e)) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x.isXMM() ? Xmm(r.getIdx()) : Ymm(r.getIdx()), cvtIdx0(x), x, T_0F | T_W0 | T_YMM, 0x50); }
void vmovntdq(const Address& addr, const Xmm& x) { opVex(x, 0, addr, T_0F | T_66 | T_YMM | T_EVEX | T_EW0, 0xE7); }
void vmovntpd(const Address& addr, const Xmm& x) { opVex(x, 0, addr, T_0F | T_66 | T_YMM | T_EVEX | T_EW1, 0x2B); }
void vmovntps(const Address& addr, const Xmm& x) { opVex(x, 0, addr, T_0F | T_YMM | T_EVEX | T_EW0, 0x2B); }
void vmovntdqa(const Xmm& x, const Address& addr) { opVex(x, 0, addr, T_0F38 | T_66 | T_YMM | T_EVEX | T_EW0, 0x2A); }
void vmovsd(const Xmm& x1, const Xmm& x2, const Operand& op = Operand()) { if (!op.isNone() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x1, x2, op, T_F2 | T_0F | T_EW1 | T_EVEX | T_N8, 0x10); }
void vmovsd(const Xmm& x, const Address& addr) { opAVX_X_X_XM(x, xm0, addr, T_F2 | T_0F | T_EW1 | T_EVEX | T_N8, 0x10); }
void vmovsd(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, T_F2 | T_0F | T_EW1 | T_EVEX | T_N8 | T_M_K, 0x11); }
void vmovss(const Xmm& x1, const Xmm& x2, const Operand& op = Operand()) { if (!op.isNone() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x1, x2, op, T_F3 | T_0F | T_EW0 | T_EVEX | T_N4, 0x10); }
void vmovss(const Xmm& x, const Address& addr) { opAVX_X_X_XM(x, xm0, addr, T_F3 | T_0F | T_EW0 | T_EVEX | T_N4, 0x10); }
void vmovss(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, T_F3 | T_0F | T_EW0 | T_EVEX | T_N4 | T_M_K, 0x11); }
void vcvtss2si(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_0F | T_F3 | T_W0 | T_EVEX | T_EW0 | T_ER_X | T_N8, 0x2D); }
void vcvttss2si(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_0F | T_F3 | T_W0 | T_EVEX | T_EW0 | T_SAE_X | T_N8, 0x2C); }
void vcvtsd2si(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_0F | T_F2 | T_W0 | T_EVEX | T_EW0 | T_N4 | T_ER_X, 0x2D); }
void vcvttsd2si(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_0F | T_F2 | T_W0 | T_EVEX | T_EW0 | T_N4 | T_SAE_X, 0x2C); }
void vcvtsi2ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opCvt3(x1, x2, op, T_0F | T_F3 | T_EVEX | T_ER_X, T_W1 | T_EW1 | T_N8, T_W0 | T_EW0 | T_N4, 0x2A); }
void vcvtsi2sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opCvt3(x1, x2, op, T_0F | T_F2 | T_EVEX, T_W1 | T_EW1 | T_ER_X | T_N8, T_W0 | T_EW0 | T_N4, 0x2A); }
void vcvtsi2sd(const Xmm& x, const Operand& op) { vcvtsi2sd(x, x, op); }
void vcvtsi2ss(const Xmm& x, const Operand& op) { vcvtsi2ss(x, x, op); }
void vcvtps2pd(const Xmm& x, const Operand& op) { checkCvt1(x, op); opVex(x, 0, op, T_0F | T_YMM | T_EVEX | T_EW0 | T_B32 | T_N8 | T_N_VL | T_SAE_Y, 0x5A); }

void vcvtdq2pd(const Xmm& x, const Operand& op) { checkCvt1(x, op); opVex(x, 0, op, T_0F | T_F3 | T_YMM | T_EVEX | T_EW0 | T_B32 | T_N8 | T_N_VL, 0xE6); }

void vcvtpd2ps(const Xmm& x, const Operand& op) { opCvt2(x, op, T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64 | T_ER_Z, 0x5A); }
void vcvtpd2dq(const Xmm& x, const Operand& op) { opCvt2(x, op, T_0F | T_F2 | T_YMM | T_EVEX | T_EW1 | T_B64 | T_ER_Z, 0xE6); }
void vcvttpd2dq(const Xmm& x, const Operand& op) { opCvt2(x, op, T_66 | T_0F | T_YMM | T_EVEX |T_EW1 | T_B64 | T_ER_Z, 0xE6); }
void vcvtph2ps(const Xmm& x, const Operand& op) { checkCvt1(x, op); opVex(x, 0, op, T_0F38 | T_66 | T_W0 | T_EVEX | T_EW0 | T_N8 | T_N_VL | T_SAE_Y, 0x13); }
void vcvtps2ph(const Operand& op, const Xmm& x, uint8 imm) { checkCvt1(x, op); opVex(x, 0, op, T_0F3A | T_66 | T_W0 | T_EVEX | T_EW0 | T_N8 | T_N_VL | T_SAE_Y, 0x1D, imm); }
#ifdef XBYAK64
void vmovq(const Xmm& x, const Reg64& r) { opAVX_X_X_XM(x, xm0, Xmm(r.getIdx()), T_66 | T_0F | T_W1 | T_EVEX | T_EW1, 0x6E); }
void vmovq(const Reg64& r, const Xmm& x) { opAVX_X_X_XM(x, xm0, Xmm(r.getIdx()), T_66 | T_0F | T_W1 | T_EVEX | T_EW1, 0x7E); }
void vpextrq(const Operand& op, const Xmm& x, uint8 imm) { if (!((op.isREG(64) || op.isMEM()) && x.isXMM())) throw Error(ERR_BAD_COMBINATION); opVex(x, 0, op, T_0F3A | T_66 | T_W1 | T_EVEX | T_EW1 | T_N8, 0x16, imm); }
void vpinsrq(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { if (!(x1.isXMM() && x2.isXMM() && (op.isREG(64) || op.isMEM()))) throw Error(ERR_BAD_COMBINATION); opVex(x1, &x2, op, T_0F3A | T_66 | T_W1 | T_EVEX | T_EW1 | T_N8, 0x22, imm); }
void vpinsrq(const Xmm& x, const Operand& op, uint8 imm) { vpinsrq(x, x, op, imm); }
void vcvtss2si(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_0F | T_F3 | T_W1 | T_EVEX | T_EW1 | T_ER_X | T_N8, 0x2D); }
void vcvttss2si(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_0F | T_F3 | T_W1 | T_EVEX | T_EW1 | T_SAE_X | T_N8, 0x2C); }
void vcvtsd2si(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_0F | T_F2 | T_W1 | T_EVEX | T_EW1 | T_N4 | T_ER_X, 0x2D); }
void vcvttsd2si(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_0F | T_F2 | T_W1 | T_EVEX | T_EW1 | T_N4 | T_SAE_X, 0x2C); }
#endif
void andn(const Reg32e& r1, const Reg32e& r2, const Operand& op) { opGpr(r1, r2, op, T_0F38, 0xf2, true); }
void mulx(const Reg32e& r1, const Reg32e& r2, const Operand& op) { opGpr(r1, r2, op, T_F2 | T_0F38, 0xf6, true); }
void pdep(const Reg32e& r1, const Reg32e& r2, const Operand& op) { opGpr(r1, r2, op, T_F2 | T_0F38, 0xf5, true); }
void pext(const Reg32e& r1, const Reg32e& r2, const Operand& op) { opGpr(r1, r2, op, T_F3 | T_0F38, 0xf5, true); }
void bextr(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, T_0F38, 0xf7, false); }
void bzhi(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, T_0F38, 0xf5, false); }
void sarx(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, T_F3 | T_0F38, 0xf7, false); }
void shlx(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, T_66 | T_0F38, 0xf7, false); }
void shrx(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, T_F2 | T_0F38, 0xf7, false); }
void blsi(const Reg32e& r, const Operand& op) { opGpr(Reg32e(3, r.getBit()), op, r, T_0F38, 0xf3, false); }
void blsmsk(const Reg32e& r, const Operand& op) { opGpr(Reg32e(2, r.getBit()), op, r, T_0F38, 0xf3, false); }
void blsr(const Reg32e& r, const Operand& op) { opGpr(Reg32e(1, r.getBit()), op, r, T_0F38, 0xf3, false); }
void vgatherdpd(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, T_0F38 | T_66 | T_W1, 0x92, 0); }
void vgatherqpd(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, T_0F38 | T_66 | T_W1, 0x93, 1); }
void vgatherdps(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, T_0F38 | T_66 | T_W0, 0x92, 1); }
void vgatherqps(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, T_0F38 | T_66 | T_W0, 0x93, 2); }
void vpgatherdd(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, T_0F38 | T_66 | T_W0, 0x90, 1); }
void vpgatherqd(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, T_0F38 | T_66 | T_W0, 0x91, 2); }
void vpgatherdq(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, T_0F38 | T_66 | T_W1, 0x90, 0); }
void vpgatherqq(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, T_0F38 | T_66 | T_W1, 0x91, 1); }
#ifndef XBYAK_DISABLE_AVX512
void kaddw(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W0, 0x4A); }
void kaddq(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W1, 0x4A); }
void kaddb(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W0, 0x4A); }
void kaddd(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W1, 0x4A); }
void kandw(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W0, 0x41); }
void kandq(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W1, 0x41); }
void kandb(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W0, 0x41); }
void kandd(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W1, 0x41); }
void kandnw(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W0, 0x42); }
void kandnq(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W1, 0x42); }
void kandnb(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W0, 0x42); }
void kandnd(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W1, 0x42); }
void korw(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W0, 0x45); }
void korq(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W1, 0x45); }
void korb(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W0, 0x45); }
void kord(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W1, 0x45); }
void kxnorw(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W0, 0x46); }
void kxnorq(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W1, 0x46); }
void kxnorb(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W0, 0x46); }
void kxnord(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W1, 0x46); }
void kxorw(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W0, 0x47); }
void kxorq(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W1, 0x47); }
void kxorb(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W0, 0x47); }
void kxord(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W1, 0x47); }
void kunpckbw(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_66 | T_W0, 0x4B); }
void kunpckwd(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W0, 0x4B); }
void kunpckdq(const Opmask& r1, const Opmask& r2, const Opmask& r3) { opVex(r1, &r2, r3, T_L1 | T_0F | T_W1, 0x4B); }
void knotw(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_W0, 0x44); }
void knotq(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_W1, 0x44); }
void knotb(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_66 | T_W0, 0x44); }
void knotd(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_66 | T_W1, 0x44); }
void kortestw(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_W0, 0x98); }
void kortestq(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_W1, 0x98); }
void kortestb(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_66 | T_W0, 0x98); }
void kortestd(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_66 | T_W1, 0x98); }
void ktestw(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_W0, 0x99); }
void ktestq(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_W1, 0x99); }
void ktestb(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_66 | T_W0, 0x99); }
void ktestd(const Opmask& r1, const Opmask& r2) { opVex(r1, 0, r2, T_0F | T_66 | T_W1, 0x99); }
void kshiftlw(const Opmask& r1, const Opmask& r2, uint8 imm) { opVex(r1, 0, r2, T_66 | T_0F3A | T_W1, 0x32, imm); }
void kshiftlq(const Opmask& r1, const Opmask& r2, uint8 imm) { opVex(r1, 0, r2, T_66 | T_0F3A | T_W1, 0x33, imm); }
void kshiftlb(const Opmask& r1, const Opmask& r2, uint8 imm) { opVex(r1, 0, r2, T_66 | T_0F3A | T_W0, 0x32, imm); }
void kshiftld(const Opmask& r1, const Opmask& r2, uint8 imm) { opVex(r1, 0, r2, T_66 | T_0F3A | T_W0, 0x33, imm); }
void kshiftrw(const Opmask& r1, const Opmask& r2, uint8 imm) { opVex(r1, 0, r2, T_66 | T_0F3A | T_W1, 0x30, imm); }
void kshiftrq(const Opmask& r1, const Opmask& r2, uint8 imm) { opVex(r1, 0, r2, T_66 | T_0F3A | T_W1, 0x31, imm); }
void kshiftrb(const Opmask& r1, const Opmask& r2, uint8 imm) { opVex(r1, 0, r2, T_66 | T_0F3A | T_W0, 0x30, imm); }
void kshiftrd(const Opmask& r1, const Opmask& r2, uint8 imm) { opVex(r1, 0, r2, T_66 | T_0F3A | T_W0, 0x31, imm); }
void kmovw(const Opmask& k, const Operand& op) { opVex(k, 0, op, T_L0 | T_0F | T_W0, 0x90); }
void kmovq(const Opmask& k, const Operand& op) { opVex(k, 0, op, T_L0 | T_0F | T_W1, 0x90); }
void kmovb(const Opmask& k, const Operand& op) { opVex(k, 0, op, T_L0 | T_0F | T_66 | T_W0, 0x90); }
void kmovd(const Opmask& k, const Operand& op) { opVex(k, 0, op, T_L0 | T_0F | T_66 | T_W1, 0x90); }
void kmovw(const Address& addr, const Opmask& k) { opVex(k, 0, addr, T_L0 | T_0F | T_W0, 0x91); }
void kmovq(const Address& addr, const Opmask& k) { opVex(k, 0, addr, T_L0 | T_0F | T_W1, 0x91); }
void kmovb(const Address& addr, const Opmask& k) { opVex(k, 0, addr, T_L0 | T_0F | T_66 | T_W0, 0x91); }
void kmovd(const Address& addr, const Opmask& k) { opVex(k, 0, addr, T_L0 | T_0F | T_66 | T_W1, 0x91); }
void kmovw(const Opmask& k, const Reg32& r) { opVex(k, 0, r, T_L0 | T_0F | T_W0, 0x92); }
void kmovw(const Reg32& r, const Opmask& k) { opVex(r, 0, k, T_L0 | T_0F | T_W0, 0x93); }
#ifdef XBYAK64
void kmovq(const Opmask& k, const Reg64& r) { opVex(k, 0, r, T_L0 | T_0F | T_F2 | T_W1, 0x92); }
void kmovq(const Reg64& r, const Opmask& k) { opVex(r, 0, k, T_L0 | T_0F | T_F2 | T_W1, 0x93); }
#endif
void kmovb(const Opmask& k, const Reg32& r) { opVex(k, 0, r, T_L0 | T_0F | T_66 | T_W0, 0x92); }
void kmovb(const Reg32& r, const Opmask& k) { opVex(r, 0, k, T_L0 | T_0F | T_66 | T_W0, 0x93); }
void kmovd(const Opmask& k, const Reg32& r) { opVex(k, 0, r, T_L0 | T_0F | T_F2 | T_W0, 0x92); }
void kmovd(const Reg32& r, const Opmask& k) { opVex(r, 0, k, T_L0 | T_0F | T_F2 | T_W0, 0x93); }
void vcmppd(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_66 | T_0F | T_EW1 | T_YMM | T_SAE_Z | T_MUST_EVEX, 0xC2, imm); }
void vcmpps(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_0F | T_EW0 | T_YMM | T_SAE_Z | T_MUST_EVEX, 0xC2, imm); }
void vcmpsd(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_F2 | T_0F | T_EW1 | T_SAE_Z | T_MUST_EVEX | T_N8, 0xC2, imm); }
void vcmpss(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_F3 | T_0F | T_EW0 | T_SAE_Z | T_MUST_EVEX | T_N4, 0xC2, imm); }
void vpcmpeqb(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F | T_YMM | T_MUST_EVEX, 0x74); }
void vpcmpeqw(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F | T_YMM | T_MUST_EVEX, 0x75); }
void vpcmpeqd(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F | T_YMM | T_MUST_EVEX | T_B32, 0x76); }
void vpcmpeqq(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x29); }
void vpcmpgtb(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F | T_YMM | T_MUST_EVEX, 0x64); }
void vpcmpgtw(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F | T_YMM | T_MUST_EVEX, 0x65); }
void vpcmpgtd(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x66); }
void vpcmpgtq(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x37); }
void vpcmpb(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX, 0x3F, imm); }
void vpcmpub(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX, 0x3E, imm); }
void vpcmpw(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX, 0x3F, imm); }
void vpcmpuw(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX, 0x3E, imm); }
void vpcmpd(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x1F, imm); }
void vpcmpud(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x1E, imm); }
void vpcmpq(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x1F, imm); }
void vpcmpuq(const Opmask& k, const Xmm& x, const Operand& op, uint8 imm) { opAVX_K_X_XM(k, x, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x1E, imm); }
void vptestmb(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX, 0x26); }
void vptestmw(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX, 0x26); }
void vptestmd(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x27); }
void vptestmq(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x27); }
void vptestnmb(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX, 0x26); }
void vptestnmw(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_F3 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX, 0x26); }
void vptestnmd(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x27); }
void vptestnmq(const Opmask& k, const Xmm& x, const Operand& op) { opAVX_K_X_XM(k, x, op, T_F3 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x27); }
void vmovdqa32(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F | T_EW0 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x6F); }
void vmovdqa64(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F | T_EW1 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x6F); }
void vmovdqu8(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_F2 | T_0F | T_EW0 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x6F); }
void vmovdqu16(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_F2 | T_0F | T_EW1 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x6F); }
void vmovdqu32(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_F3 | T_0F | T_EW0 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x6F); }
void vmovdqu64(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_F3 | T_0F | T_EW1 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x6F); }
void vcvtpd2qq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F | T_EW1 | T_YMM | T_ER_Z | T_MUST_EVEX | T_B64, 0x7B); }
void vcvtpd2uqq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F | T_EW1 | T_YMM | T_ER_Z | T_MUST_EVEX | T_B64, 0x79); }
void vcvtps2udq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_0F | T_EW0 | T_YMM | T_ER_Z | T_MUST_EVEX | T_B32, 0x79); }
void vcvtqq2pd(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_F3 | T_0F | T_EW1 | T_YMM | T_ER_Z | T_MUST_EVEX | T_B64, 0xE6); }
void vcvttpd2qq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F | T_EW1 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B64, 0x7A); }
void vcvttpd2uqq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F | T_EW1 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B64, 0x78); }
void vcvttps2udq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_0F | T_EW0 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B32, 0x78); }
void vcvtudq2ps(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_F2 | T_0F | T_EW0 | T_YMM | T_ER_Z | T_MUST_EVEX | T_B32, 0x7A); }
void vcvtuqq2pd(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_F3 | T_0F | T_EW1 | T_YMM | T_ER_Z | T_MUST_EVEX | T_B64, 0x7A); }
void vexpandpd(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x88); }
void vexpandps(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x88); }
void vpexpandd(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x89); }
void vpexpandq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x89); }
void vgetexppd(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B64, 0x42); }
void vgetexpps(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B32, 0x42); }
void vpabsq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_MUST_EVEX | T_EW1 | T_B64 | T_YMM, 0x1F); }
void vexp2pd(const Zmm& z, const Operand& op) { opAVX_X_XM_IMM(z, op, T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_SAE_Z, 0xC8); }
void vexp2ps(const Zmm& z, const Operand& op) { opAVX_X_XM_IMM(z, op, T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW0 | T_B32 | T_SAE_Z, 0xC8); }
void vrcp28pd(const Zmm& z, const Operand& op) { opAVX_X_XM_IMM(z, op, T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_SAE_Z, 0xCA); }
void vrcp28ps(const Zmm& z, const Operand& op) { opAVX_X_XM_IMM(z, op, T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW0 | T_B32 | T_SAE_Z, 0xCA); }
void vrsqrt28pd(const Zmm& z, const Operand& op) { opAVX_X_XM_IMM(z, op, T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_SAE_Z, 0xCC); }
void vrsqrt28ps(const Zmm& z, const Operand& op) { opAVX_X_XM_IMM(z, op, T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW0 | T_B32 | T_SAE_Z, 0xCC); }
void vmovdqa32(const Address& addr, const Xmm& x) { opAVX_X_XM_IMM(x, addr, T_66 | T_0F | T_EW0 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x7F); }
void vmovdqa64(const Address& addr, const Xmm& x) { opAVX_X_XM_IMM(x, addr, T_66 | T_0F | T_EW1 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x7F); }
void vmovdqu8(const Address& addr, const Xmm& x) { opAVX_X_XM_IMM(x, addr, T_F2 | T_0F | T_EW0 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x7F); }
void vmovdqu16(const Address& addr, const Xmm& x) { opAVX_X_XM_IMM(x, addr, T_F2 | T_0F | T_EW1 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x7F); }
void vmovdqu32(const Address& addr, const Xmm& x) { opAVX_X_XM_IMM(x, addr, T_F3 | T_0F | T_EW0 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x7F); }
void vmovdqu64(const Address& addr, const Xmm& x) { opAVX_X_XM_IMM(x, addr, T_F3 | T_0F | T_EW1 | T_YMM | T_ER_X | T_ER_Y | T_ER_Z | T_MUST_EVEX, 0x7F); }
void vcompresspd(const Operand& op, const Xmm& x) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x8A); }
void vcompressps(const Operand& op, const Xmm& x) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x8A); }
void vpcompressd(const Operand& op, const Xmm& x) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x8B); }
void vpcompressq(const Operand& op, const Xmm& x) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x8B); }
void valignd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX, 0x03, imm); }
void valignq(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX, 0x03, imm); }
void vpandd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0xDB); }
void vpandq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0xDB); }
void vpandnd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0xDF); }
void vpandnq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0xDF); }
void vpmaxsq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x3D); }
void vpmaxuq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x3F); }
void vpminsq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x39); }
void vpminuq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x3B); }
void vpsraq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_MUST_EVEX | T_N16, 0xE2); }
void vpsravq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x46); }
void vpsravw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX, 0x11); }
void vpsllvw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX, 0x12); }
void vpsrlvw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX, 0x10); }
void vpord(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0xEB); }
void vporq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0xEB); }
void vpxord(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0xEF); }
void vpxorq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0xEF); }
void vpmullq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x40); }
void vpermb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX, 0x8D); }
void vpermw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX, 0x8D); }
void vblendmpd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x65); }
void vblendmps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x65); }
void vpblendmb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX, 0x66); }
void vpblendmw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX, 0x66); }
void vpblendmd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x64); }
void vpblendmq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x64); }
void vpermt2b(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX, 0x7D); }
void vpermt2w(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX, 0x7D); }
void vpermt2d(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x7E); }
void vpermt2q(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x7E); }
void vpermt2ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x7F); }
void vpermt2pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x7F); }
void vpermi2b(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX, 0x75); }
void vpermi2w(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX, 0x75); }
void vpermi2d(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x76); }
void vpermi2q(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x76); }
void vpermi2ps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x77); }
void vpermi2pd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x77); }
void vpternlogd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x25, imm); }
void vpternlogq(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x25, imm); }
void vgetexpsd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_SAE_X | T_MUST_EVEX | T_N8, 0x43); }
void vgetexpss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_SAE_X | T_MUST_EVEX | T_N4, 0x43); }
void vgetmantsd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW1 | T_SAE_X | T_MUST_EVEX | T_N8, 0x27, imm); }
void vgetmantss(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_SAE_X | T_MUST_EVEX | T_N4, 0x27, imm); }
void vfixupimmpd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B64, 0x54, imm); }
void vfixupimmps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B32, 0x54, imm); }
void vfixupimmsd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW1 | T_SAE_Z | T_MUST_EVEX | T_N8, 0x55, imm); }
void vfixupimmss(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_SAE_Z | T_MUST_EVEX | T_N4, 0x55, imm); }
void vrcp14sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_MUST_EVEX | T_N8, 0x4D); }
void vrcp14ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_MUST_EVEX | T_N4, 0x4D); }
void vrsqrt14sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x4F); }
void vrsqrt14ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x4F); }
void vrndscalesd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW1 | T_MUST_EVEX | T_N8, 0x0B, imm); }
void vrndscaless(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_MUST_EVEX | T_N4, 0x0A, imm); }
void vscalefpd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_ER_Z | T_MUST_EVEX | T_B64, 0x2C); }
void vscalefps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_ER_Z | T_MUST_EVEX | T_B32, 0x2C); }
void vscalefsd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_ER_X | T_MUST_EVEX | T_N8, 0x2D); }
void vscalefss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_ER_X | T_MUST_EVEX | T_N4, 0x2D); }
void vdbpsadbw(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX, 0x42, imm); }
void vpmultishiftqb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x83); }
void vprolvd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x15); }
void vprolvq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x15); }
void vprorvd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x14); }
void vprorvq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x14); }
void vrcp28sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_SAE_X | T_MUST_EVEX | T_N8, 0xCB); }
void vrcp28ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_SAE_X | T_MUST_EVEX | T_N4, 0xCB); }
void vrsqrt28sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_SAE_X | T_MUST_EVEX | T_N8, 0xCD); }
void vrsqrt28ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW0 | T_SAE_X | T_MUST_EVEX | T_N4, 0xCD); }
void vrangepd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B64, 0x50, imm); }
void vrangeps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B32, 0x50, imm); }
void vrangesd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW1 | T_SAE_X | T_MUST_EVEX | T_N8, 0x51, imm); }
void vrangess(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_SAE_X | T_MUST_EVEX | T_N4, 0x51, imm); }
void vreducesd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW1 | T_SAE_X | T_MUST_EVEX | T_N8, 0x57, imm); }
void vreducess(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F3A | T_EW0 | T_SAE_X | T_MUST_EVEX | T_N4, 0x57, imm); }
void vpmadd52luq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0xB4); }
void vpmadd52huq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0xB5); }
void vpsraq(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 4), x, op, T_66 | T_0F | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x72, imm); }
void vprold(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 1), x, op, T_66 | T_0F | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x72, imm); }
void vprolq(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 1), x, op, T_66 | T_0F | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x72, imm); }
void vprord(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 0), x, op, T_66 | T_0F | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x72, imm); }
void vprorq(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_X_XM(Xmm(x.getKind(), 0), x, op, T_66 | T_0F | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x72, imm); }
void vextractf32x4(const Operand& op, const Ymm& r, uint8 imm) { if (!op.is(Operand::MEM | Operand::XMM)) throw Error(ERR_BAD_COMBINATION); opVex(r, 0, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_N16, 0x19, imm); }
void vextractf64x2(const Operand& op, const Ymm& r, uint8 imm) { if (!op.is(Operand::MEM | Operand::XMM)) throw Error(ERR_BAD_COMBINATION); opVex(r, 0, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_N16, 0x19, imm); }
void vextractf32x8(const Operand& op, const Zmm& r, uint8 imm) { if (!op.is(Operand::MEM | Operand::YMM)) throw Error(ERR_BAD_COMBINATION); opVex(r, 0, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_N32, 0x1B, imm); }
void vextractf64x4(const Operand& op, const Zmm& r, uint8 imm) { if (!op.is(Operand::MEM | Operand::YMM)) throw Error(ERR_BAD_COMBINATION); opVex(r, 0, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_N32, 0x1B, imm); }
void vextracti32x4(const Operand& op, const Ymm& r, uint8 imm) { if (!op.is(Operand::MEM | Operand::XMM)) throw Error(ERR_BAD_COMBINATION); opVex(r, 0, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_N16, 0x39, imm); }
void vextracti64x2(const Operand& op, const Ymm& r, uint8 imm) { if (!op.is(Operand::MEM | Operand::XMM)) throw Error(ERR_BAD_COMBINATION); opVex(r, 0, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_N16, 0x39, imm); }
void vextracti32x8(const Operand& op, const Zmm& r, uint8 imm) { if (!op.is(Operand::MEM | Operand::YMM)) throw Error(ERR_BAD_COMBINATION); opVex(r, 0, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_N32, 0x3B, imm); }
void vextracti64x4(const Operand& op, const Zmm& r, uint8 imm) { if (!op.is(Operand::MEM | Operand::YMM)) throw Error(ERR_BAD_COMBINATION); opVex(r, 0, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_N32, 0x3B, imm); }
void vinsertf32x4(const Ymm& r1, const Ymm& r2, const Operand& op, uint8 imm) {if (!(r1.getKind() == r2.getKind() && op.is(Operand::MEM | Operand::XMM))) throw Error(ERR_BAD_COMBINATION); opVex(r1, &r2, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_N16, 0x18, imm); }
void vinsertf64x2(const Ymm& r1, const Ymm& r2, const Operand& op, uint8 imm) {if (!(r1.getKind() == r2.getKind() && op.is(Operand::MEM | Operand::XMM))) throw Error(ERR_BAD_COMBINATION); opVex(r1, &r2, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_N16, 0x18, imm); }
void vinsertf32x8(const Zmm& r1, const Zmm& r2, const Operand& op, uint8 imm) {if (!op.is(Operand::MEM | Operand::YMM)) throw Error(ERR_BAD_COMBINATION); opVex(r1, &r2, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_N32, 0x1A, imm); }
void vinsertf64x4(const Zmm& r1, const Zmm& r2, const Operand& op, uint8 imm) {if (!op.is(Operand::MEM | Operand::YMM)) throw Error(ERR_BAD_COMBINATION); opVex(r1, &r2, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_N32, 0x1A, imm); }
void vinserti32x4(const Ymm& r1, const Ymm& r2, const Operand& op, uint8 imm) {if (!(r1.getKind() == r2.getKind() && op.is(Operand::MEM | Operand::XMM))) throw Error(ERR_BAD_COMBINATION); opVex(r1, &r2, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_N16, 0x38, imm); }
void vinserti64x2(const Ymm& r1, const Ymm& r2, const Operand& op, uint8 imm) {if (!(r1.getKind() == r2.getKind() && op.is(Operand::MEM | Operand::XMM))) throw Error(ERR_BAD_COMBINATION); opVex(r1, &r2, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_N16, 0x38, imm); }
void vinserti32x8(const Zmm& r1, const Zmm& r2, const Operand& op, uint8 imm) {if (!op.is(Operand::MEM | Operand::YMM)) throw Error(ERR_BAD_COMBINATION); opVex(r1, &r2, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_N32, 0x3A, imm); }
void vinserti64x4(const Zmm& r1, const Zmm& r2, const Operand& op, uint8 imm) {if (!op.is(Operand::MEM | Operand::YMM)) throw Error(ERR_BAD_COMBINATION); opVex(r1, &r2, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_N32, 0x3A, imm); }
void vbroadcastf32x2(const Ymm& y, const Operand& op) { opAVX_X_XM_IMM(y, op, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW0 | T_N8, 0x19); }
void vbroadcastf32x4(const Ymm& y, const Address& addr) { opAVX_X_XM_IMM(y, addr, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW0 | T_N16, 0x1A); }
void vbroadcastf64x2(const Ymm& y, const Address& addr) { opAVX_X_XM_IMM(y, addr, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_N16, 0x1A); }
void vbroadcastf64x4(const Zmm& y, const Address& addr) { opAVX_X_XM_IMM(y, addr, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_N32, 0x1B); }
void vbroadcastf32x8(const Zmm& y, const Address& addr) { opAVX_X_XM_IMM(y, addr, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW0 | T_N32, 0x1B); }
void vpbroadcastb(const Xmm& x, const Reg8& r) { opVex(x, 0, r, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x7A); }
void vpbroadcastw(const Xmm& x, const Reg16& r) { opVex(x, 0, r, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x7B); }
void vpbroadcastd(const Xmm& x, const Reg32& r) { opVex(x, 0, r, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x7C); }
#ifdef XBYAK64
void vpbroadcastq(const Xmm& x, const Reg64& r) { opVex(x, 0, r, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x7C); }
#endif
void vbroadcasti32x2(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW0 | T_N8, 0x59); }
void vbroadcasti32x4(const Ymm& y, const Operand& op) { opAVX_X_XM_IMM(y, op, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW0 | T_N16, 0x5A); }
void vbroadcasti64x2(const Ymm& y, const Operand& op) { opAVX_X_XM_IMM(y, op, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_N16, 0x5A); }
void vbroadcasti32x8(const Zmm& z, const Operand& op) { opAVX_X_XM_IMM(z, op, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW0 | T_N32, 0x5B); }
void vbroadcasti64x4(const Zmm& z, const Operand& op) { opAVX_X_XM_IMM(z, op, T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_N32, 0x5B); }
void vcvtpd2udq(const Xmm& x, const Operand& op) { opCvt2(x, op, T_0F | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_ER_Z, 0x79); }
void vcvtps2qq(const Xmm& x, const Operand& op) { checkCvt1(x, op); opVex(x, 0, op, T_66 | T_0F | T_YMM | T_MUST_EVEX | T_EW0 | T_B32 | T_N8 | T_N_VL | T_ER_Y, 0x7B); }
void vcvtps2uqq(const Xmm& x, const Operand& op) { checkCvt1(x, op); opVex(x, 0, op, T_66 | T_0F | T_YMM | T_MUST_EVEX | T_EW0 | T_B32 | T_N8 | T_N_VL | T_ER_Y, 0x79); }
void vcvtqq2ps(const Xmm& x, const Operand& op) { opCvt2(x, op, T_0F | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_ER_Z, 0x5B); }
void vcvttpd2udq(const Xmm& x, const Operand& op) { opCvt2(x, op, T_0F | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_SAE_Z, 0x78); }
void vcvttps2qq(const Xmm& x, const Operand& op) { checkCvt1(x, op); opVex(x, 0, op, T_66 | T_0F | T_YMM | T_MUST_EVEX | T_EW0 | T_B32 | T_N8 | T_N_VL | T_SAE_Y, 0x7A); }
void vcvttps2uqq(const Xmm& x, const Operand& op) { checkCvt1(x, op); opVex(x, 0, op, T_66 | T_0F | T_YMM | T_MUST_EVEX | T_EW0 | T_B32 | T_N8 | T_N_VL | T_SAE_Y, 0x78); }
void vcvtudq2pd(const Xmm& x, const Operand& op) { checkCvt1(x, op); opVex(x, 0, op, T_F3 | T_0F | T_YMM | T_MUST_EVEX | T_EW0 | T_B32 | T_N8 | T_N_VL, 0x7A); }
void vcvtsd2usi(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_F2 | T_0F | T_MUST_EVEX | T_EW0 | T_N8 | T_ER_X, 0x79); }
void vcvtss2usi(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_F3 | T_0F | T_MUST_EVEX | T_EW0 | T_N4 | T_ER_X, 0x79); }
void vcvttsd2usi(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_F2 | T_0F | T_MUST_EVEX | T_EW0 | T_N8 | T_SAE_X, 0x78); }
void vcvttss2usi(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_F3 | T_0F | T_MUST_EVEX | T_EW0 | T_N4 | T_SAE_X, 0x78); }
void vcvtuqq2ps(const Xmm& x, const Operand& op) { opCvt2(x, op, T_F2 | T_0F | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_ER_Z, 0x7A); }
void vcvtusi2sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opCvt3(x1, x2, op, T_F2 | T_0F | T_MUST_EVEX, T_W1 | T_EW1 | T_ER_X | T_N8, T_W0 | T_EW0 | T_N4, 0x7B); }
void vcvtusi2ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opCvt3(x1, x2, op, T_F3 | T_0F | T_MUST_EVEX | T_ER_X, T_W1 | T_EW1 | T_N8, T_W0 | T_EW0 | T_N4, 0x7B); }
#ifdef XBYAK64
void vcvtsd2usi(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_F2 | T_0F | T_MUST_EVEX | T_EW1 | T_N8 | T_ER_X, 0x79); }
void vcvtss2usi(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_F3 | T_0F | T_MUST_EVEX | T_EW1 | T_N4 | T_ER_X, 0x79); }
void vcvttsd2usi(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_F2 | T_0F | T_MUST_EVEX | T_EW1 | T_N8 | T_SAE_X, 0x78); }
void vcvttss2usi(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, T_F3 | T_0F | T_MUST_EVEX | T_EW1 | T_N4 | T_SAE_X, 0x78); }
#endif
void vpgatherdd(const Xmm& x, const Address& addr) { opGather2(x, addr, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x90, 0); }
void vpgatherdq(const Xmm& x, const Address& addr) { opGather2(x, addr, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x90, 1); }
void vpgatherqd(const Xmm& x, const Address& addr) { opGather2(x, addr, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x91, 2); }
void vpgatherqq(const Xmm& x, const Address& addr) { opGather2(x, addr, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x91, 0); }
void vgatherdps(const Xmm& x, const Address& addr) { opGather2(x, addr, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x92, 0); }
void vgatherdpd(const Xmm& x, const Address& addr) { opGather2(x, addr, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x92, 1); }
void vgatherqps(const Xmm& x, const Address& addr) { opGather2(x, addr, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4, 0x93, 2); }
void vgatherqpd(const Xmm& x, const Address& addr) { opGather2(x, addr, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_N8, 0x93, 0); }
void vshuff32x4(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { opAVX_X_X_XM(y1, y2, op, T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW0 | T_B32, 0x23, imm); }
void vshuff64x2(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { opAVX_X_X_XM(y1, y2, op, T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x23, imm); }
void vshufi32x4(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { opAVX_X_X_XM(y1, y2, op, T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW0 | T_B32, 0x43, imm); }
void vshufi64x2(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { opAVX_X_X_XM(y1, y2, op, T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x43, imm); }
void vpmovm2b(const Xmm& x, const Opmask& k) { opVex(x, 0, k, T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW0, 0x28); }
void vpmovm2w(const Xmm& x, const Opmask& k) { opVex(x, 0, k, T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x28); }
void vpmovm2d(const Xmm& x, const Opmask& k) { opVex(x, 0, k, T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW0, 0x38); }
void vpmovm2q(const Xmm& x, const Opmask& k) { opVex(x, 0, k, T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x38); }
void vpmovb2m(const Opmask& k, const Xmm& x) { opVex(k, 0, x, T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW0, 0x29); }
void vpmovw2m(const Opmask& k, const Xmm& x) { opVex(k, 0, x, T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x29); }
void vpmovd2m(const Opmask& k, const Xmm& x) { opVex(k, 0, x, T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW0, 0x39); }
void vpmovq2m(const Opmask& k, const Xmm& x) { opVex(k, 0, x, T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x39); }
void vpmovqb(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N2 | T_N_VL, 0x32, false); }
void vpmovsqb(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N2 | T_N_VL, 0x22, false); }
void vpmovusqb(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N2 | T_N_VL, 0x12, false); }
void vpmovqw(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4 | T_N_VL, 0x34, false); }
void vpmovsqw(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4 | T_N_VL, 0x24, false); }
void vpmovusqw(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4 | T_N_VL, 0x14, false); }
void vpmovqd(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N8 | T_N_VL, 0x35, true); }
void vpmovsqd(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N8 | T_N_VL, 0x25, true); }
void vpmovusqd(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N8 | T_N_VL, 0x15, true); }
void vpmovdb(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4 | T_N_VL, 0x31, false); }
void vpmovsdb(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4 | T_N_VL, 0x21, false); }
void vpmovusdb(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N4 | T_N_VL, 0x11, false); }
void vpmovdw(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N8 | T_N_VL, 0x33, true); }
void vpmovsdw(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N8 | T_N_VL, 0x23, true); }
void vpmovusdw(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N8 | T_N_VL, 0x13, true); }
void vpmovwb(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N8 | T_N_VL, 0x30, true); }
void vpmovswb(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N8 | T_N_VL, 0x20, true); }
void vpmovuswb(const Operand& op, const Xmm& x) { opVmov(op, x, T_F3 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_N8 | T_N_VL, 0x10, true); }
void vgetmantpd(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(x, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B64, 0x26, imm); }
void vgetmantps(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(x, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B32, 0x26, imm); }
void vrcp14pd(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x4C); }
void vrcp14ps(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x4C); }
void vrsqrt14pd(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x4E); }
void vrsqrt14ps(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x4E); }
void vrndscalepd(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(x, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x09, imm); }
void vrndscaleps(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(x, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x08, imm); }
void vpconflictd(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0xC4); }
void vpconflictq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0xC4); }
void vplzcntd(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_B32, 0x44); }
void vplzcntq(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_B64, 0x44); }
void vreducepd(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(x, op, T_66 | T_0F3A | T_EW1 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B64, 0x56, imm); }
void vreduceps(const Xmm& x, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(x, op, T_66 | T_0F3A | T_EW0 | T_YMM | T_SAE_Z | T_MUST_EVEX | T_B32, 0x56, imm); }
void vpbroadcastmb2q(const Xmm& x, const Opmask& k) { opVex(x, 0, k, T_F3 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1, 0x2A); }
void vpbroadcastmw2d(const Xmm& x, const Opmask& k) { opVex(x, 0, k, T_F3 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW0, 0x3A); }
void vgatherpf0dps(const Address& addr) { opGatherFetch(addr, zm1, T_66 | T_0F38 | T_EW0 | T_MUST_EVEX | T_N4, 0xC6, Operand::ZMM); }
void vgatherpf0qps(const Address& addr) { opGatherFetch(addr, zm1, T_66 | T_0F38 | T_EW0 | T_MUST_EVEX | T_N4, 0xC7, Operand::ZMM); }
void vgatherpf0dpd(const Address& addr) { opGatherFetch(addr, zm1, T_66 | T_0F38 | T_EW1 | T_MUST_EVEX | T_N8, 0xC6, Operand::YMM); }
void vgatherpf0qpd(const Address& addr) { opGatherFetch(addr, zm1, T_66 | T_0F38 | T_EW1 | T_MUST_EVEX | T_N8, 0xC7, Operand::ZMM); }
void vgatherpf1dps(const Address& addr) { opGatherFetch(addr, zm2, T_66 | T_0F38 | T_EW0 | T_MUST_EVEX | T_N4, 0xC6, Operand::ZMM); }
void vgatherpf1qps(const Address& addr) { opGatherFetch(addr, zm2, T_66 | T_0F38 | T_EW0 | T_MUST_EVEX | T_N4, 0xC7, Operand::ZMM); }
void vgatherpf1dpd(const Address& addr) { opGatherFetch(addr, zm2, T_66 | T_0F38 | T_EW1 | T_MUST_EVEX | T_N8, 0xC6, Operand::YMM); }
void vgatherpf1qpd(const Address& addr) { opGatherFetch(addr, zm2, T_66 | T_0F38 | T_EW1 | T_MUST_EVEX | T_N8, 0xC7, Operand::ZMM); }
void vscatterpf0dps(const Address& addr) { opGatherFetch(addr, zm5, T_66 | T_0F38 | T_EW0 | T_MUST_EVEX | T_N4, 0xC6, Operand::ZMM); }
void vscatterpf0qps(const Address& addr) { opGatherFetch(addr, zm5, T_66 | T_0F38 | T_EW0 | T_MUST_EVEX | T_N4, 0xC7, Operand::ZMM); }
void vscatterpf0dpd(const Address& addr) { opGatherFetch(addr, zm5, T_66 | T_0F38 | T_EW1 | T_MUST_EVEX | T_N8, 0xC6, Operand::YMM); }
void vscatterpf0qpd(const Address& addr) { opGatherFetch(addr, zm5, T_66 | T_0F38 | T_EW1 | T_MUST_EVEX | T_N8, 0xC7, Operand::ZMM); }
void vscatterpf1dps(const Address& addr) { opGatherFetch(addr, zm6, T_66 | T_0F38 | T_EW0 | T_MUST_EVEX | T_N4, 0xC6, Operand::ZMM); }
void vscatterpf1qps(const Address& addr) { opGatherFetch(addr, zm6, T_66 | T_0F38 | T_EW0 | T_MUST_EVEX | T_N4, 0xC7, Operand::ZMM); }
void vscatterpf1dpd(const Address& addr) { opGatherFetch(addr, zm6, T_66 | T_0F38 | T_EW1 | T_MUST_EVEX | T_N8, 0xC6, Operand::YMM); }
void vscatterpf1qpd(const Address& addr) { opGatherFetch(addr, zm6, T_66 | T_0F38 | T_EW1 | T_MUST_EVEX | T_N8, 0xC7, Operand::ZMM); }
void vfpclasspd(const Opmask& k, const Operand& op, uint8 imm) { if (!op.isBit(128|256|512)) throw Error(ERR_BAD_MEM_SIZE); Reg x = k; x.setBit(op.getBit()); opVex(x, 0, op, T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x66, imm); }
void vfpclassps(const Opmask& k, const Operand& op, uint8 imm) { if (!op.isBit(128|256|512)) throw Error(ERR_BAD_MEM_SIZE); Reg x = k; x.setBit(op.getBit()); opVex(x, 0, op, T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_EW0 | T_B32, 0x66, imm); }
void vfpclasssd(const Opmask& k, const Operand& op, uint8 imm) { if (!op.isXMEM()) throw Error(ERR_BAD_MEM_SIZE); opVex(k, 0, op, T_66 | T_0F3A | T_MUST_EVEX | T_EW1 | T_N8, 0x67, imm); }
void vfpclassss(const Opmask& k, const Operand& op, uint8 imm) { if (!op.isXMEM()) throw Error(ERR_BAD_MEM_SIZE); opVex(k, 0, op, T_66 | T_0F3A | T_MUST_EVEX | T_EW0 | T_N4, 0x67, imm); }
void vpscatterdd(const Address& addr, const Xmm& x) { opGather2(x, addr, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_M_K | T_N4, 0xA0, 0); }
void vpscatterdq(const Address& addr, const Xmm& x) { opGather2(x, addr, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_M_K | T_N8, 0xA0, 1); }
void vpscatterqd(const Address& addr, const Xmm& x) { opGather2(x, addr, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_M_K | T_N4, 0xA1, 2); }
void vpscatterqq(const Address& addr, const Xmm& x) { opGather2(x, addr, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_M_K | T_N8, 0xA1, 0); }
void vscatterdps(const Address& addr, const Xmm& x) { opGather2(x, addr, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_M_K | T_N4, 0xA2, 0); }
void vscatterdpd(const Address& addr, const Xmm& x) { opGather2(x, addr, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_M_K | T_N8, 0xA2, 1); }
void vscatterqps(const Address& addr, const Xmm& x) { opGather2(x, addr, T_66 | T_0F38 | T_EW0 | T_YMM | T_MUST_EVEX | T_M_K | T_N4, 0xA3, 2); }
void vscatterqpd(const Address& addr, const Xmm& x) { opGather2(x, addr, T_66 | T_0F38 | T_EW1 | T_YMM | T_MUST_EVEX | T_M_K | T_N8, 0xA3, 0); }
#endif
