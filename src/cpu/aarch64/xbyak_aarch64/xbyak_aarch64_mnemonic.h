/*******************************************************************************
 * Copyright 2019-2020 FUJITSU LIMITED
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
#ifdef XBYAK_USE_FILE_LINE
#define XBYAK_SET_CODE_INFO() setCodeInfo(__FILE__, __LINE__, __func__)
#else
#define XBYAK_SET_CODE_INFO()
#endif
void adr(const XReg &xd, const Label &label) {
  XBYAK_SET_CODE_INFO();
  PCrelAddr(0, xd, label);
}
void adr(const XReg &xd, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  PCrelAddr(0, xd, label);
}
void adrp(const XReg &xd, const Label &label) {
  XBYAK_SET_CODE_INFO();
  PCrelAddr(1, xd, label);
}
void adrp(const XReg &xd, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  PCrelAddr(1, xd, label);
}
void add(const WReg &rd, const WReg &rn, const uint32_t imm,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(0, 0, rd, rn, imm, sh);
}
void add(const XReg &rd, const XReg &rn, const uint32_t imm,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(0, 0, rd, rn, imm, sh);
}
void adds(const WReg &rd, const WReg &rn, const uint32_t imm,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(0, 1, rd, rn, imm, sh);
}
void adds(const XReg &rd, const XReg &rn, const uint32_t imm,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(0, 1, rd, rn, imm, sh);
}
void cmn(const WReg &rn, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(0, 1, WReg(31), rn, imm, sh);
}
void cmn(const XReg &rn, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(0, 1, XReg(31), rn, imm, sh);
}
void sub(const WReg &rd, const WReg &rn, const uint32_t imm,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(1, 0, rd, rn, imm, sh);
}
void sub(const XReg &rd, const XReg &rn, const uint32_t imm,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(1, 0, rd, rn, imm, sh);
}
void subs(const WReg &rd, const WReg &rn, const uint32_t imm,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(1, 1, rd, rn, imm, sh);
}
void subs(const XReg &rd, const XReg &rn, const uint32_t imm,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(1, 1, rd, rn, imm, sh);
}
void cmp(const WReg &rn, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(1, 1, WReg(31), rn, imm, sh);
}
void cmp(const XReg &rn, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubImm(1, 1, XReg(31), rn, imm, sh);
}
void and_(const WReg &rd, const WReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(0, rd, rn, imm);
}
void and_(const XReg &rd, const XReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(0, rd, rn, imm);
}
void orr(const WReg &rd, const WReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(1, rd, rn, imm);
}
void orr(const XReg &rd, const XReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(1, rd, rn, imm);
}
void eor(const WReg &rd, const WReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(2, rd, rn, imm);
}
void eor(const XReg &rd, const XReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(2, rd, rn, imm);
}
void ands(const WReg &rd, const WReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(3, rd, rn, imm);
}
void ands(const XReg &rd, const XReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(3, rd, rn, imm);
}
void tst(const WReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(3, WReg(31), rn, imm, true);
}
void tst(const XReg &rn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  LogicalImm(3, XReg(31), rn, imm, true);
}
void movn(const WReg &rd, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  MvWideImm(0, rd, imm, sh);
}
void movn(const XReg &rd, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  MvWideImm(0, rd, imm, sh);
}
void movz(const WReg &rd, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  MvWideImm(2, rd, imm, sh);
}
void movz(const XReg &rd, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  MvWideImm(2, rd, imm, sh);
}
void movk(const WReg &rd, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  MvWideImm(3, rd, imm, sh);
}
void movk(const XReg &rd, const uint32_t imm, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  MvWideImm(3, rd, imm, sh);
}
void mov(const WReg &rd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  MvImm(rd, imm);
}
void mov(const XReg &rd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  MvImm(rd, imm);
}
void sbfm(const WReg &rd, const WReg &rn, const uint32_t immr,
          const uint32_t imms) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, immr, imms);
}
void sbfm(const XReg &rd, const XReg &rn, const uint32_t immr,
          const uint32_t imms) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, immr, imms);
}
void sbfiz(const WReg &rd, const WReg &rn, const uint32_t lsb,
           const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, (((-1) * lsb) % 32) & ones(6), width - 1);
}
void sbfiz(const XReg &rd, const XReg &rn, const uint32_t lsb,
           const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, (((-1) * lsb) % 64) & ones(6), width - 1);
}
void sbfx(const WReg &rd, const WReg &rn, const uint32_t lsb,
          const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, lsb, lsb + width - 1);
}
void sbfx(const XReg &rd, const XReg &rn, const uint32_t lsb,
          const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, lsb, lsb + width - 1);
}
void sxtb(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, 0, 7);
}
void sxtb(const XReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, 0, 7);
}
void sxth(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, 0, 15);
}
void sxth(const XReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, 0, 15);
}
void sxtw(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, 0, 31);
}
void sxtw(const XReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, 0, 31);
}
void asr(const WReg &rd, const WReg &rn, const uint32_t immr) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, immr, 31);
}
void asr(const XReg &rd, const XReg &rn, const uint32_t immr) {
  XBYAK_SET_CODE_INFO();
  Bitfield(0, rd, rn, immr, 63);
}
void bfm(const WReg &rd, const WReg &rn, const uint32_t immr,
         const uint32_t imms) {
  XBYAK_SET_CODE_INFO();
  Bitfield(1, rd, rn, immr, imms);
}
void bfm(const XReg &rd, const XReg &rn, const uint32_t immr,
         const uint32_t imms) {
  XBYAK_SET_CODE_INFO();
  Bitfield(1, rd, rn, immr, imms);
}
void bfc(const WReg &rd, const uint32_t lsb, const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(1, rd, WReg(31), (((-1) * lsb) % 32) & ones(6), width - 1, false);
}
void bfc(const XReg &rd, const uint32_t lsb, const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(1, rd, XReg(31), (((-1) * lsb) % 64) & ones(6), width - 1, false);
}
void bfi(const WReg &rd, const WReg &rn, const uint32_t lsb,
         const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(1, rd, rn, (((-1) * lsb) % 32) & ones(6), width - 1);
}
void bfi(const XReg &rd, const XReg &rn, const uint32_t lsb,
         const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(1, rd, rn, (((-1) * lsb) % 64) & ones(6), width - 1);
}
void bfxil(const WReg &rd, const WReg &rn, const uint32_t lsb,
           const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(1, rd, rn, lsb, lsb + width - 1);
}
void bfxil(const XReg &rd, const XReg &rn, const uint32_t lsb,
           const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(1, rd, rn, lsb, lsb + width - 1);
}
void ubfm(const WReg &rd, const WReg &rn, const uint32_t immr,
          const uint32_t imms) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, immr, imms);
}
void ubfm(const XReg &rd, const XReg &rn, const uint32_t immr,
          const uint32_t imms) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, immr, imms);
}
void ubfiz(const WReg &rd, const WReg &rn, const uint32_t lsb,
           const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, (((-1) * lsb) % 32) & ones(6), width - 1);
}
void ubfiz(const XReg &rd, const XReg &rn, const uint32_t lsb,
           const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, (((-1) * lsb) % 64) & ones(6), width - 1);
}
void ubfx(const WReg &rd, const WReg &rn, const uint32_t lsb,
          const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, lsb, lsb + width - 1);
}
void ubfx(const XReg &rd, const XReg &rn, const uint32_t lsb,
          const uint32_t width) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, lsb, lsb + width - 1);
}
void lsl(const WReg &rd, const WReg &rn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, (((-1) * sh) % 32) & ones(6), 31 - sh);
}
void lsl(const XReg &rd, const XReg &rn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, (((-1) * sh) % 64) & ones(6), 63 - sh);
}
void lsr(const WReg &rd, const WReg &rn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, sh, 31);
}
void lsr(const XReg &rd, const XReg &rn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, sh, 63);
}
void uxtb(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, 0, 7);
}
void uxtb(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, 0, 7);
}
void uxth(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, 0, 15);
}
void uxth(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  Bitfield(2, rd, rn, 0, 15);
}
void extr(const WReg &rd, const WReg &rn, const WReg &rm, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  Extract(0, 0, rd, rn, rm, imm);
}
void extr(const XReg &rd, const XReg &rn, const XReg &rm, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  Extract(0, 0, rd, rn, rm, imm);
}
void ror(const WReg &rd, const WReg &rn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  Extract(0, 0, rd, rn, rn, imm);
}
void ror(const XReg &rd, const XReg &rn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  Extract(0, 0, rd, rn, rn, imm);
}
void b(const Cond cond, const Label &label) {
  XBYAK_SET_CODE_INFO();
  CondBrImm(cond, label);
}
void b(const Cond cond, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  CondBrImm(cond, label);
}
void svc(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  ExceptionGen(0, 0, 1, imm);
}
void hvc(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  ExceptionGen(0, 0, 2, imm);
}
void smc(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  ExceptionGen(0, 0, 3, imm);
}
void brk(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  ExceptionGen(1, 0, 0, imm);
}
void hlt(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  ExceptionGen(2, 0, 0, imm);
}
void dcps1(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  ExceptionGen(5, 0, 1, imm);
}
void dcps2(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  ExceptionGen(5, 0, 2, imm);
}
void dcps3(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  ExceptionGen(5, 0, 3, imm);
}
void hint(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  Hints(imm);
}
void nop() {
  XBYAK_SET_CODE_INFO();
  Hints(0, 0);
}
void yield() {
  XBYAK_SET_CODE_INFO();
  Hints(0, 1);
}
void wfe() {
  XBYAK_SET_CODE_INFO();
  Hints(0, 2);
}
void wfi() {
  XBYAK_SET_CODE_INFO();
  Hints(0, 3);
}
void sev() {
  XBYAK_SET_CODE_INFO();
  Hints(0, 4);
}
void sevl() {
  XBYAK_SET_CODE_INFO();
  Hints(0, 5);
}
void xpaclri() {
  XBYAK_SET_CODE_INFO();
  Hints(0, 7);
}
void pacia1716() {
  XBYAK_SET_CODE_INFO();
  Hints(1, 0);
}
void pacib1716() {
  XBYAK_SET_CODE_INFO();
  Hints(1, 2);
}
void autia1716() {
  XBYAK_SET_CODE_INFO();
  Hints(1, 4);
}
void autib1716() {
  XBYAK_SET_CODE_INFO();
  Hints(1, 6);
}
void esb() {
  XBYAK_SET_CODE_INFO();
  Hints(2, 0);
}
void psb_csync() {
  XBYAK_SET_CODE_INFO();
  Hints(2, 1);
}
void tsb_csync() {
  XBYAK_SET_CODE_INFO();
  Hints(2, 2);
}
void csdb() {
  XBYAK_SET_CODE_INFO();
  Hints(2, 4);
}
void paciaz() {
  XBYAK_SET_CODE_INFO();
  Hints(3, 0);
}
void paciasp() {
  XBYAK_SET_CODE_INFO();
  Hints(3, 1);
}
void pacibz() {
  XBYAK_SET_CODE_INFO();
  Hints(3, 2);
}
void pacibsp() {
  XBYAK_SET_CODE_INFO();
  Hints(3, 3);
}
void autiaz() {
  XBYAK_SET_CODE_INFO();
  Hints(3, 4);
}
void autiasp() {
  XBYAK_SET_CODE_INFO();
  Hints(3, 5);
}
void autibz() {
  XBYAK_SET_CODE_INFO();
  Hints(3, 6);
}
void autibsp() {
  XBYAK_SET_CODE_INFO();
  Hints(3, 7);
}
void dsb(const BarOpt opt) {
  XBYAK_SET_CODE_INFO();
  BarriersOpt(4, opt, 31);
}
void dmb(const BarOpt opt) {
  XBYAK_SET_CODE_INFO();
  BarriersOpt(5, opt, 31);
}
void isb(const BarOpt opt) {
  XBYAK_SET_CODE_INFO();
  BarriersOpt(6, opt, 31);
}
void clrex(const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  BarriersNoOpt(imm, 2, 31);
}
void ssbb() {
  XBYAK_SET_CODE_INFO();
  BarriersNoOpt(0, 4, 31);
}
void pssbb() {
  XBYAK_SET_CODE_INFO();
  BarriersNoOpt(4, 4, 31);
}
void msr(const PStateField psfield, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  PState(psfield, imm);
}
void cfinv() {
  XBYAK_SET_CODE_INFO();
  PState(0, 0, 0);
}
void sys(const uint32_t op1, const uint32_t CRn, const uint32_t CRm,
         const uint32_t op2, const XReg &rt = XReg(31)) {
  XBYAK_SET_CODE_INFO();
  SysInst(0, op1, CRn, CRm, op2, rt);
}
void sysl(const XReg &rt, const uint32_t op1, const uint32_t CRn,
          const uint32_t CRm, const uint32_t op2) {
  XBYAK_SET_CODE_INFO();
  SysInst(1, op1, CRn, CRm, op2, rt);
}
void msr(const uint32_t op0, const uint32_t op1, const uint32_t CRn,
         const uint32_t CRm, const uint32_t op2, const XReg &rt) {
  XBYAK_SET_CODE_INFO();
  SysRegMove(0, op0, op1, CRn, CRm, op2, rt);
}
void mrs(const XReg &rt, const uint32_t op0, const uint32_t op1,
         const uint32_t CRn, const uint32_t CRm, const uint32_t op2) {
  XBYAK_SET_CODE_INFO();
  SysRegMove(1, op0, op1, CRn, CRm, op2, rt);
}
void ret() {
  XBYAK_SET_CODE_INFO();
  UncondBrNoReg(2, 31, 0, 30, 0);
}
void retaa() {
  XBYAK_SET_CODE_INFO();
  UncondBrNoReg(2, 31, 2, 31, 31);
}
void retab() {
  XBYAK_SET_CODE_INFO();
  UncondBrNoReg(2, 31, 3, 31, 31);
}
void eret() {
  XBYAK_SET_CODE_INFO();
  UncondBrNoReg(4, 31, 0, 31, 0);
}
void eretaa() {
  XBYAK_SET_CODE_INFO();
  UncondBrNoReg(4, 31, 2, 31, 31);
}
void eretab() {
  XBYAK_SET_CODE_INFO();
  UncondBrNoReg(4, 31, 3, 31, 31);
}
void drps() {
  XBYAK_SET_CODE_INFO();
  UncondBrNoReg(5, 31, 0, 31, 0);
}
void br(const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  UncondBr1Reg(0, 31, 0, rn, 0);
}
void braaz(const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  UncondBr1Reg(0, 31, 2, rn, 31);
}
void brabz(const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  UncondBr1Reg(0, 31, 3, rn, 31);
}
void blr(const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  UncondBr1Reg(1, 31, 0, rn, 0);
}
void blraaz(const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  UncondBr1Reg(1, 31, 2, rn, 31);
}
void blrabz(const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  UncondBr1Reg(1, 31, 3, rn, 31);
}
void ret(const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  UncondBr1Reg(2, 31, 0, rn, 0);
}
void braa(const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  UncondBr2Reg(8, 31, 2, rn, rm);
}
void brab(const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  UncondBr2Reg(8, 31, 3, rn, rm);
}
void blraa(const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  UncondBr2Reg(9, 31, 2, rn, rm);
}
void blrab(const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  UncondBr2Reg(9, 31, 3, rn, rm);
}
void b(const Label &label) {
  XBYAK_SET_CODE_INFO();
  UncondBrImm(0, label);
}
void b(const int64_t label) {
  XBYAK_SET_CODE_INFO();
  UncondBrImm(0, label);
}
void bl(const Label &label) {
  XBYAK_SET_CODE_INFO();
  UncondBrImm(1, label);
}
void bl(const int64_t label) {
  XBYAK_SET_CODE_INFO();
  UncondBrImm(1, label);
}
void cbz(const WReg &rt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  CompareBr(0, rt, label);
}
void cbz(const XReg &rt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  CompareBr(0, rt, label);
}
void cbz(const WReg &rt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  CompareBr(0, rt, label);
}
void cbz(const XReg &rt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  CompareBr(0, rt, label);
}
void cbnz(const WReg &rt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  CompareBr(1, rt, label);
}
void cbnz(const XReg &rt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  CompareBr(1, rt, label);
}
void cbnz(const WReg &rt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  CompareBr(1, rt, label);
}
void cbnz(const XReg &rt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  CompareBr(1, rt, label);
}
void tbz(const WReg &rt, const uint32_t imm, const Label &label) {
  XBYAK_SET_CODE_INFO();
  TestBr(0, rt, imm, label);
}
void tbz(const XReg &rt, const uint32_t imm, const Label &label) {
  XBYAK_SET_CODE_INFO();
  TestBr(0, rt, imm, label);
}
void tbz(const WReg &rt, const uint32_t imm, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  TestBr(0, rt, imm, label);
}
void tbz(const XReg &rt, const uint32_t imm, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  TestBr(0, rt, imm, label);
}
void tbnz(const WReg &rt, const uint32_t imm, const Label &label) {
  XBYAK_SET_CODE_INFO();
  TestBr(1, rt, imm, label);
}
void tbnz(const XReg &rt, const uint32_t imm, const Label &label) {
  XBYAK_SET_CODE_INFO();
  TestBr(1, rt, imm, label);
}
void tbnz(const WReg &rt, const uint32_t imm, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  TestBr(1, rt, imm, label);
}
void tbnz(const XReg &rt, const uint32_t imm, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  TestBr(1, rt, imm, label);
}
void st1(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr);
}
void st1(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr);
}
void st1(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr);
}
void st1(const VReg1DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr);
}
void st1(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr);
}
void st1(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr);
}
void st1(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr);
}
void st1(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr);
}
void ld1(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg1DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr);
}
void st4(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr);
}
void st3(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr);
}
void st2(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr);
}
void ld4(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr);
}
void ld3(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr);
}
void ld2(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr);
}
void st1(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr);
}
void st1(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr);
}
void st1(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr);
}
void st1(const VReg1DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr);
}
void st1(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr);
}
void st1(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr);
}
void st1(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr);
}
void st1(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr);
}
void ld1(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg1DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr);
}
void st4(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr);
}
void st3(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr);
}
void st2(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr);
}
void ld4(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr);
}
void ld3(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr);
}
void ld2(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr);
}
void st1(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr);
}
void st1(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr);
}
void st1(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr);
}
void st1(const VReg1DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr);
}
void st1(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr);
}
void st1(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr);
}
void st1(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr);
}
void st1(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr);
}
void ld1(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg1DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr);
}
void ld1(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr);
}
void st4(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr);
}
void st4(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr);
}
void st3(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr);
}
void st3(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr);
}
void st2(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr);
}
void st2(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr);
}
void ld4(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr);
}
void ld4(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr);
}
void ld3(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr);
}
void ld3(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr);
}
void ld2(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr);
}
void ld2(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr);
}
void st4(const VRegBElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 1, 4, vt, adr);
}
void st4(const VRegHElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 1, 4, vt, adr);
}
void st4(const VRegSElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 1, 4, vt, adr);
}
void st4(const VRegDElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 1, 4, vt, adr);
}
void st3(const VRegBElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 0, 3, vt, adr);
}
void st3(const VRegHElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 0, 3, vt, adr);
}
void st3(const VRegSElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 0, 3, vt, adr);
}
void st3(const VRegDElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 0, 3, vt, adr);
}
void st2(const VRegBElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 1, 2, vt, adr);
}
void st2(const VRegHElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 1, 2, vt, adr);
}
void st2(const VRegSElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 1, 2, vt, adr);
}
void st2(const VRegDElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 1, 2, vt, adr);
}
void st1(const VRegBElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 0, 1, vt, adr);
}
void st1(const VRegHElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 0, 1, vt, adr);
}
void st1(const VRegSElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 0, 1, vt, adr);
}
void st1(const VRegDElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(0, 0, 1, vt, adr);
}
void ld4(const VRegBElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 1, 4, vt, adr);
}
void ld4(const VRegHElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 1, 4, vt, adr);
}
void ld4(const VRegSElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 1, 4, vt, adr);
}
void ld4(const VRegDElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 1, 4, vt, adr);
}
void ld3(const VRegBElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 0, 3, vt, adr);
}
void ld3(const VRegHElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 0, 3, vt, adr);
}
void ld3(const VRegSElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 0, 3, vt, adr);
}
void ld3(const VRegDElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 0, 3, vt, adr);
}
void ld2(const VRegBElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 1, 2, vt, adr);
}
void ld2(const VRegHElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 1, 2, vt, adr);
}
void ld2(const VRegSElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 1, 2, vt, adr);
}
void ld2(const VRegDElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 1, 2, vt, adr);
}
void ld1(const VRegBElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 0, 1, vt, adr);
}
void ld1(const VRegHElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 0, 1, vt, adr);
}
void ld1(const VRegSElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 0, 1, vt, adr);
}
void ld1(const VRegDElem &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStruct(1, 0, 1, vt, adr);
}
void ld4r(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg1DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr);
}
void ld3r(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg1DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr);
}
void ld2r(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg1DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr);
}
void ld1r(const VReg8BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg4HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg2SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg1DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg16BList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg8HList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg4SList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg2DList &vt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr);
}
void st4(const VRegBElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 1, 4, vt, adr);
}
void st4(const VRegHElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 1, 4, vt, adr);
}
void st4(const VRegSElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 1, 4, vt, adr);
}
void st4(const VRegDElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 1, 4, vt, adr);
}
void st3(const VRegBElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 0, 3, vt, adr);
}
void st3(const VRegHElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 0, 3, vt, adr);
}
void st3(const VRegSElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 0, 3, vt, adr);
}
void st3(const VRegDElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 0, 3, vt, adr);
}
void st2(const VRegBElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 1, 2, vt, adr);
}
void st2(const VRegHElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 1, 2, vt, adr);
}
void st2(const VRegSElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 1, 2, vt, adr);
}
void st2(const VRegDElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 1, 2, vt, adr);
}
void st1(const VRegBElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 0, 1, vt, adr);
}
void st1(const VRegHElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 0, 1, vt, adr);
}
void st1(const VRegSElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 0, 1, vt, adr);
}
void st1(const VRegDElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(0, 0, 1, vt, adr);
}
void ld4(const VRegBElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 1, 4, vt, adr);
}
void ld4(const VRegHElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 1, 4, vt, adr);
}
void ld4(const VRegSElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 1, 4, vt, adr);
}
void ld4(const VRegDElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 1, 4, vt, adr);
}
void ld3(const VRegBElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 0, 3, vt, adr);
}
void ld3(const VRegHElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 0, 3, vt, adr);
}
void ld3(const VRegSElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 0, 3, vt, adr);
}
void ld3(const VRegDElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 0, 3, vt, adr);
}
void ld2(const VRegBElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 1, 2, vt, adr);
}
void ld2(const VRegHElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 1, 2, vt, adr);
}
void ld2(const VRegSElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 1, 2, vt, adr);
}
void ld2(const VRegDElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 1, 2, vt, adr);
}
void ld1(const VRegBElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 0, 1, vt, adr);
}
void ld1(const VRegHElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 0, 1, vt, adr);
}
void ld1(const VRegSElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 0, 1, vt, adr);
}
void ld1(const VRegDElem &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostReg(1, 0, 1, vt, adr);
}
void ld4r(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg1DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr);
}
void ld3r(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg1DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr);
}
void ld2r(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg1DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr);
}
void ld1r(const VReg8BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg4HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg2SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg1DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg16BList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg8HList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg4SList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg2DList &vt, const AdrPostReg &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr);
}
void st4(const VRegBElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 1, 4, vt, adr);
}
void st4(const VRegHElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 1, 4, vt, adr);
}
void st4(const VRegSElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 1, 4, vt, adr);
}
void st4(const VRegDElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 1, 4, vt, adr);
}
void st3(const VRegBElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 0, 3, vt, adr);
}
void st3(const VRegHElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 0, 3, vt, adr);
}
void st3(const VRegSElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 0, 3, vt, adr);
}
void st3(const VRegDElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 0, 3, vt, adr);
}
void st2(const VRegBElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 1, 2, vt, adr);
}
void st2(const VRegHElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 1, 2, vt, adr);
}
void st2(const VRegSElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 1, 2, vt, adr);
}
void st2(const VRegDElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 1, 2, vt, adr);
}
void st1(const VRegBElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 0, 1, vt, adr);
}
void st1(const VRegHElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 0, 1, vt, adr);
}
void st1(const VRegSElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 0, 1, vt, adr);
}
void st1(const VRegDElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(0, 0, 1, vt, adr);
}
void ld4(const VRegBElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 1, 4, vt, adr);
}
void ld4(const VRegHElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 1, 4, vt, adr);
}
void ld4(const VRegSElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 1, 4, vt, adr);
}
void ld4(const VRegDElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 1, 4, vt, adr);
}
void ld3(const VRegBElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 0, 3, vt, adr);
}
void ld3(const VRegHElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 0, 3, vt, adr);
}
void ld3(const VRegSElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 0, 3, vt, adr);
}
void ld3(const VRegDElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 0, 3, vt, adr);
}
void ld2(const VRegBElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 1, 2, vt, adr);
}
void ld2(const VRegHElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 1, 2, vt, adr);
}
void ld2(const VRegSElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 1, 2, vt, adr);
}
void ld2(const VRegDElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 1, 2, vt, adr);
}
void ld1(const VRegBElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 0, 1, vt, adr);
}
void ld1(const VRegHElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 0, 1, vt, adr);
}
void ld1(const VRegSElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 0, 1, vt, adr);
}
void ld1(const VRegDElem &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdStSingleStructPostImm(1, 0, 1, vt, adr);
}
void ld4r(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg1DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr);
}
void ld4r(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr);
}
void ld3r(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg1DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr);
}
void ld3r(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr);
}
void ld2r(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg1DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr);
}
void ld2r(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr);
}
void ld1r(const VReg8BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg4HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg2SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg1DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg16BList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg8HList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg4SList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr);
}
void ld1r(const VReg2DList &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr);
}
void stxrb(const WReg &ws, const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusive(0, 0, ws, rt, adr);
}
void stlxrb(const WReg &ws, const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusive(0, 1, ws, rt, adr);
}
void stxrh(const WReg &ws, const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusive(1, 0, ws, rt, adr);
}
void stlxrh(const WReg &ws, const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusive(1, 1, ws, rt, adr);
}
void stxr(const WReg &ws, const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusive(2, 0, ws, rt, adr);
}
void stlxr(const WReg &ws, const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusive(2, 1, ws, rt, adr);
}
void stxr(const WReg &ws, const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusive(3, 0, ws, rt, adr);
}
void stlxr(const WReg &ws, const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusive(3, 1, ws, rt, adr);
}
void ldxrb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusive(0, 0, rt, adr);
}
void ldaxrb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusive(0, 1, rt, adr);
}
void ldxrh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusive(1, 0, rt, adr);
}
void ldaxrh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusive(1, 1, rt, adr);
}
void ldxr(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusive(2, 0, rt, adr);
}
void ldaxr(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusive(2, 1, rt, adr);
}
void ldxr(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusive(3, 0, rt, adr);
}
void ldaxr(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusive(3, 1, rt, adr);
}
void stllrb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StLORelase(0, 0, rt, adr);
}
void stlrb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StLORelase(0, 1, rt, adr);
}
void stllrh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StLORelase(1, 0, rt, adr);
}
void stlrh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StLORelase(1, 1, rt, adr);
}
void stllr(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StLORelase(2, 0, rt, adr);
}
void stlr(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StLORelase(2, 1, rt, adr);
}
void stllr(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StLORelase(3, 0, rt, adr);
}
void stlr(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StLORelase(3, 1, rt, adr);
}
void ldlarb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdLOAcquire(0, 0, rt, adr);
}
void ldarb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdLOAcquire(0, 1, rt, adr);
}
void ldlarh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdLOAcquire(1, 0, rt, adr);
}
void ldarh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdLOAcquire(1, 1, rt, adr);
}
void ldlar(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdLOAcquire(2, 0, rt, adr);
}
void ldar(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdLOAcquire(2, 1, rt, adr);
}
void ldlar(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdLOAcquire(3, 0, rt, adr);
}
void ldar(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdLOAcquire(3, 1, rt, adr);
}
void casb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(0, 1, 0, 1, 0, rs, rt, adr);
}
void caslb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(0, 1, 0, 1, 1, rs, rt, adr);
}
void casab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(0, 1, 1, 1, 0, rs, rt, adr);
}
void casalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(0, 1, 1, 1, 1, rs, rt, adr);
}
void cash(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(1, 1, 0, 1, 0, rs, rt, adr);
}
void caslh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(1, 1, 0, 1, 1, rs, rt, adr);
}
void casah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(1, 1, 1, 1, 0, rs, rt, adr);
}
void casalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(1, 1, 1, 1, 1, rs, rt, adr);
}
void cas(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(2, 1, 0, 1, 0, rs, rt, adr);
}
void casl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(2, 1, 0, 1, 1, rs, rt, adr);
}
void casa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(2, 1, 1, 1, 0, rs, rt, adr);
}
void casal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(2, 1, 1, 1, 1, rs, rt, adr);
}
void cas(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(3, 1, 0, 1, 0, rs, rt, adr);
}
void casl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(3, 1, 0, 1, 1, rs, rt, adr);
}
void casa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(3, 1, 1, 1, 0, rs, rt, adr);
}
void casal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Cas(3, 1, 1, 1, 1, rs, rt, adr);
}
void stxp(const WReg &ws, const WReg &rt1, const WReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusivePair(0, 1, 0, ws, rt1, rt2, adr);
}
void stxp(const WReg &ws, const XReg &rt1, const XReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusivePair(0, 1, 0, ws, rt1, rt2, adr);
}
void stlxp(const WReg &ws, const WReg &rt1, const WReg &rt2,
           const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusivePair(0, 1, 1, ws, rt1, rt2, adr);
}
void stlxp(const WReg &ws, const XReg &rt1, const XReg &rt2,
           const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  StExclusivePair(0, 1, 1, ws, rt1, rt2, adr);
}
void ldxp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusivePair(1, 1, 0, rt1, rt2, adr);
}
void ldxp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusivePair(1, 1, 0, rt1, rt2, adr);
}
void ldaxp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusivePair(1, 1, 1, rt1, rt2, adr);
}
void ldaxp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdExclusivePair(1, 1, 1, rt1, rt2, adr);
}
void casp(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  CasPair(0, 1, 0, rs, rt, adr);
}
void casp(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  CasPair(0, 1, 0, rs, rt, adr);
}
void caspl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  CasPair(0, 1, 1, rs, rt, adr);
}
void caspl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  CasPair(0, 1, 1, rs, rt, adr);
}
void caspa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  CasPair(1, 1, 0, rs, rt, adr);
}
void caspa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  CasPair(1, 1, 0, rs, rt, adr);
}
void caspal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  CasPair(1, 1, 1, rs, rt, adr);
}
void caspal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  CasPair(1, 1, 1, rs, rt, adr);
}
void stlurb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(0, 0, rt, adr);
}
void ldapurb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(0, 1, rt, adr);
}
void ldapursb(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(0, 2, rt, adr);
}
void ldapursb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(0, 3, rt, adr);
}
void stlurh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(1, 0, rt, adr);
}
void ldapurh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(1, 1, rt, adr);
}
void ldapursh(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(1, 2, rt, adr);
}
void ldapursh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(1, 3, rt, adr);
}
void stlur(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(2, 0, rt, adr);
}
void ldapur(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(2, 1, rt, adr);
}
void ldapursw(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(2, 2, rt, adr);
}
void stlur(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(3, 0, rt, adr);
}
void ldapur(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdaprStlr(3, 1, rt, adr);
}
void ldr(const WReg &rt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  LdRegLiteral((rt.getBit() == 64) ? 1 : 0, 0, rt, label);
}
void ldr(const XReg &rt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  LdRegLiteral((rt.getBit() == 64) ? 1 : 0, 0, rt, label);
}
void ldr(const WReg &rt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  LdRegLiteral((rt.getBit() == 64) ? 1 : 0, 0, rt, label);
}
void ldr(const XReg &rt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  LdRegLiteral((rt.getBit() == 64) ? 1 : 0, 0, rt, label);
}
void ldrsw(const WReg &rt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  LdRegLiteral(2, 0, rt, label);
}
void ldrsw(const XReg &rt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  LdRegLiteral(2, 0, rt, label);
}
void ldrsw(const WReg &rt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  LdRegLiteral(2, 0, rt, label);
}
void ldrsw(const XReg &rt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  LdRegLiteral(2, 0, rt, label);
}
void ldr(const SReg &vt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  LdRegSimdFpLiteral(vt, label);
}
void ldr(const DReg &vt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  LdRegSimdFpLiteral(vt, label);
}
void ldr(const QReg &vt, const Label &label) {
  XBYAK_SET_CODE_INFO();
  LdRegSimdFpLiteral(vt, label);
}
void ldr(const SReg &vt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  LdRegSimdFpLiteral(vt, label);
}
void ldr(const DReg &vt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  LdRegSimdFpLiteral(vt, label);
}
void ldr(const QReg &vt, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  LdRegSimdFpLiteral(vt, label);
}
void prfm(const Prfop prfop, const Label &label) {
  XBYAK_SET_CODE_INFO();
  PfLiteral(prfop, label);
}
void prfm(const Prfop prfop, const int64_t label) {
  XBYAK_SET_CODE_INFO();
  PfLiteral(prfop, label);
}
void stnp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStNoAllocPair(0, rt1, rt2, adr);
}
void stnp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStNoAllocPair(0, rt1, rt2, adr);
}
void ldnp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStNoAllocPair(1, rt1, rt2, adr);
}
void ldnp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStNoAllocPair(1, rt1, rt2, adr);
}
void stnp(const SReg &vt1, const SReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpNoAllocPair(0, vt1, vt2, adr);
}
void stnp(const DReg &vt1, const DReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpNoAllocPair(0, vt1, vt2, adr);
}
void stnp(const QReg &vt1, const QReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpNoAllocPair(0, vt1, vt2, adr);
}
void ldnp(const SReg &vt1, const SReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpNoAllocPair(1, vt1, vt2, adr);
}
void ldnp(const DReg &vt1, const DReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpNoAllocPair(1, vt1, vt2, adr);
}
void ldnp(const QReg &vt1, const QReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpNoAllocPair(1, vt1, vt2, adr);
}
void stp(const WReg &rt1, const WReg &rt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPostImm((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr);
}
void stp(const XReg &rt1, const XReg &rt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPostImm((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr);
}
void ldp(const WReg &rt1, const WReg &rt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPostImm((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr);
}
void ldp(const XReg &rt1, const XReg &rt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPostImm((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr);
}
void ldpsw(const XReg &rt1, const XReg &rt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPostImm(1, 1, rt1, rt2, adr);
}
void stp(const SReg &vt1, const SReg &vt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPostImm(0, vt1, vt2, adr);
}
void stp(const DReg &vt1, const DReg &vt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPostImm(0, vt1, vt2, adr);
}
void stp(const QReg &vt1, const QReg &vt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPostImm(0, vt1, vt2, adr);
}
void ldp(const SReg &vt1, const SReg &vt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPostImm(1, vt1, vt2, adr);
}
void ldp(const DReg &vt1, const DReg &vt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPostImm(1, vt1, vt2, adr);
}
void ldp(const QReg &vt1, const QReg &vt2, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPostImm(1, vt1, vt2, adr);
}
void stp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPair((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr);
}
void stp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPair((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr);
}
void ldp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPair((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr);
}
void ldp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPair((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr);
}
void ldpsw(const XReg &rt1, const XReg &rt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPair(1, 1, rt1, rt2, adr);
}
void stp(const SReg &vt1, const SReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPair(0, vt1, vt2, adr);
}
void stp(const DReg &vt1, const DReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPair(0, vt1, vt2, adr);
}
void stp(const QReg &vt1, const QReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPair(0, vt1, vt2, adr);
}
void ldp(const SReg &vt1, const SReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPair(1, vt1, vt2, adr);
}
void ldp(const DReg &vt1, const DReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPair(1, vt1, vt2, adr);
}
void ldp(const QReg &vt1, const QReg &vt2, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPair(1, vt1, vt2, adr);
}
void stp(const WReg &rt1, const WReg &rt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPre((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr);
}
void stp(const XReg &rt1, const XReg &rt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPre((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr);
}
void ldp(const WReg &rt1, const WReg &rt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPre((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr);
}
void ldp(const XReg &rt1, const XReg &rt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPre((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr);
}
void ldpsw(const XReg &rt1, const XReg &rt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPairPre(1, 1, rt1, rt2, adr);
}
void stp(const SReg &vt1, const SReg &vt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPre(0, vt1, vt2, adr);
}
void stp(const DReg &vt1, const DReg &vt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPre(0, vt1, vt2, adr);
}
void stp(const QReg &vt1, const QReg &vt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPre(0, vt1, vt2, adr);
}
void ldp(const SReg &vt1, const SReg &vt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPre(1, vt1, vt2, adr);
}
void ldp(const DReg &vt1, const DReg &vt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPre(1, vt1, vt2, adr);
}
void ldp(const QReg &vt1, const QReg &vt2, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpPairPre(1, vt1, vt2, adr);
}
void sturb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(0, 0, rt, adr);
}
void ldurb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(0, 1, rt, adr);
}
void ldursb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(0, 3, rt, adr);
}
void sturh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(1, 0, rt, adr);
}
void ldurh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(1, 1, rt, adr);
}
void ldursh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(1, 3, rt, adr);
}
void stur(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(2, 0, rt, adr);
}
void ldur(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(2, 1, rt, adr);
}
void ldursb(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(0, 2, rt, adr);
}
void ldursh(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(1, 2, rt, adr);
}
void ldursw(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(2, 2, rt, adr);
}
void stur(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(3, 0, rt, adr);
}
void ldur(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnsImm(3, 1, rt, adr);
}
void stur(const BReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void stur(const HReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void stur(const SReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void stur(const DReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void stur(const QReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void ldur(const BReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldur(const HReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldur(const SReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldur(const DReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldur(const QReg &vt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void prfum(const Prfop prfop, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  PfRegUnsImm(prfop, adr);
}
void strb(const WReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(0, 0, rt, adr);
}
void ldrb(const WReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(0, 1, rt, adr);
}
void ldrsb(const WReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(0, 3, rt, adr);
}
void strh(const WReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(1, 0, rt, adr);
}
void ldrh(const WReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(1, 1, rt, adr);
}
void ldrsh(const WReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(1, 3, rt, adr);
}
void str(const WReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(2, 0, rt, adr);
}
void ldr(const WReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(2, 1, rt, adr);
}
void ldrsb(const XReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(0, 2, rt, adr);
}
void ldrsh(const XReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(1, 2, rt, adr);
}
void ldrsw(const XReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(2, 2, rt, adr);
}
void str(const XReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(3, 0, rt, adr);
}
void ldr(const XReg &rt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPostImm(3, 1, rt, adr);
}
void str(const BReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const HReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const SReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const DReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const QReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void ldr(const BReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const HReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const SReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const DReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const QReg &vt, const AdrPostImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void sttrb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(0, 0, rt, adr);
}
void ldtrb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(0, 1, rt, adr);
}
void ldtrsb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(0, 3, rt, adr);
}
void sttrh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(1, 0, rt, adr);
}
void ldtrh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(1, 1, rt, adr);
}
void ldtrsh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(1, 3, rt, adr);
}
void sttr(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(2, 0, rt, adr);
}
void ldtr(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(2, 1, rt, adr);
}
void ldtrsb(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(0, 2, rt, adr);
}
void ldtrsh(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(1, 2, rt, adr);
}
void ldtrsw(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(2, 2, rt, adr);
}
void sttr(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(3, 0, rt, adr);
}
void ldtr(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnpriv(3, 1, rt, adr);
}
void strb(const WReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(0, 0, rt, adr);
}
void ldrb(const WReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(0, 1, rt, adr);
}
void ldrsb(const WReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(0, 3, rt, adr);
}
void strh(const WReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(1, 0, rt, adr);
}
void ldrh(const WReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(1, 1, rt, adr);
}
void ldrsh(const WReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(1, 3, rt, adr);
}
void str(const WReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(2, 0, rt, adr);
}
void ldr(const WReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(2, 1, rt, adr);
}
void ldrsb(const XReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(0, 2, rt, adr);
}
void ldrsh(const XReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(1, 2, rt, adr);
}
void ldrsw(const XReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(2, 2, rt, adr);
}
void str(const XReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(3, 0, rt, adr);
}
void ldr(const XReg &rt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPre(3, 1, rt, adr);
}
void str(const BReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const HReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const SReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const DReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const QReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void ldr(const BReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const HReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const SReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const DReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const QReg &vt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldaddb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 0, rs, rt, adr);
}
void ldclrb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 1, rs, rt, adr);
}
void ldeorb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 2, rs, rt, adr);
}
void ldsetb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 3, rs, rt, adr);
}
void ldsmaxb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 4, rs, rt, adr);
}
void ldsminb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 5, rs, rt, adr);
}
void ldumaxb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 6, rs, rt, adr);
}
void stumaxb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 6, rs, rt, adr);
}
void lduminb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 7, rs, rt, adr);
}
void swapb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 1, 0, rs, rt, adr);
}
void ldaddlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 0, rs, rt, adr);
}
void ldclrlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 1, rs, rt, adr);
}
void ldeorlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 2, rs, rt, adr);
}
void ldsetlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 3, rs, rt, adr);
}
void ldsmaxlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 4, rs, rt, adr);
}
void ldsminlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 5, rs, rt, adr);
}
void ldumaxlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 6, rs, rt, adr);
}
void lduminlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 7, rs, rt, adr);
}
void swaplb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 1, 0, rs, rt, adr);
}
void ldaddab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 0, rs, rt, adr);
}
void ldclrab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 1, rs, rt, adr);
}
void ldeorab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 2, rs, rt, adr);
}
void ldsetab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 3, rs, rt, adr);
}
void ldsmaxab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 4, rs, rt, adr);
}
void ldsminab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 5, rs, rt, adr);
}
void ldumaxab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 6, rs, rt, adr);
}
void lduminab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 7, rs, rt, adr);
}
void swapab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 1, 0, rs, rt, adr);
}
void ldaprb(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 1, 4, WReg(31), rt, adr);
}
void ldaddalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 0, rs, rt, adr);
}
void ldclralb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 1, rs, rt, adr);
}
void ldeoralb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 2, rs, rt, adr);
}
void ldsetalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 3, rs, rt, adr);
}
void ldsmaxalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 4, rs, rt, adr);
}
void ldsminalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 5, rs, rt, adr);
}
void ldumaxalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 6, rs, rt, adr);
}
void lduminalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 7, rs, rt, adr);
}
void swapalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 1, 0, rs, rt, adr);
}
void ldaddh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 0, rs, rt, adr);
}
void ldclrh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 1, rs, rt, adr);
}
void ldeorh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 2, rs, rt, adr);
}
void ldseth(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 3, rs, rt, adr);
}
void ldsmaxh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 4, rs, rt, adr);
}
void ldsminh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 5, rs, rt, adr);
}
void ldumaxh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 6, rs, rt, adr);
}
void lduminh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 7, rs, rt, adr);
}
void swaph(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 1, 0, rs, rt, adr);
}
void ldaddlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 0, rs, rt, adr);
}
void ldclrlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 1, rs, rt, adr);
}
void ldeorlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 2, rs, rt, adr);
}
void ldsetlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 3, rs, rt, adr);
}
void ldsmaxlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 4, rs, rt, adr);
}
void ldsminlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 5, rs, rt, adr);
}
void ldumaxlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 6, rs, rt, adr);
}
void lduminlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 7, rs, rt, adr);
}
void swaplh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 1, 0, rs, rt, adr);
}
void ldaddah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 0, rs, rt, adr);
}
void ldclrah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 1, rs, rt, adr);
}
void ldeorah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 2, rs, rt, adr);
}
void ldsetah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 3, rs, rt, adr);
}
void ldsmaxah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 4, rs, rt, adr);
}
void ldsminah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 5, rs, rt, adr);
}
void ldumaxah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 6, rs, rt, adr);
}
void lduminah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 7, rs, rt, adr);
}
void swapah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 1, 0, rs, rt, adr);
}
void ldaprh(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 1, 4, WReg(31), rt, adr);
}
void ldaddalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 0, rs, rt, adr);
}
void ldclralh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 1, rs, rt, adr);
}
void ldeoralh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 2, rs, rt, adr);
}
void ldsetalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 3, rs, rt, adr);
}
void ldsmaxalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 4, rs, rt, adr);
}
void ldsminalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 5, rs, rt, adr);
}
void ldumaxalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 6, rs, rt, adr);
}
void lduminalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 7, rs, rt, adr);
}
void swapalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 1, 0, rs, rt, adr);
}
void ldadd(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 0, rs, rt, adr);
}
void ldclr(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 1, rs, rt, adr);
}
void ldeor(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 2, rs, rt, adr);
}
void ldset(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 3, rs, rt, adr);
}
void ldsmax(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 4, rs, rt, adr);
}
void ldsmin(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 5, rs, rt, adr);
}
void ldumax(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 6, rs, rt, adr);
}
void ldumin(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 7, rs, rt, adr);
}
void swap(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 1, 0, rs, rt, adr);
}
void ldaddl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 0, rs, rt, adr);
}
void ldclrl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 1, rs, rt, adr);
}
void ldeorl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 2, rs, rt, adr);
}
void ldsetl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 3, rs, rt, adr);
}
void ldsmaxl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 4, rs, rt, adr);
}
void ldsminl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 5, rs, rt, adr);
}
void ldumaxl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 6, rs, rt, adr);
}
void lduminl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 7, rs, rt, adr);
}
void swapl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 1, 0, rs, rt, adr);
}
void ldadda(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 0, rs, rt, adr);
}
void ldclra(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 1, rs, rt, adr);
}
void ldeora(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 2, rs, rt, adr);
}
void ldseta(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 3, rs, rt, adr);
}
void ldsmaxa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 4, rs, rt, adr);
}
void ldsmina(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 5, rs, rt, adr);
}
void ldumaxa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 6, rs, rt, adr);
}
void ldumina(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 7, rs, rt, adr);
}
void swapa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 1, 0, rs, rt, adr);
}
void ldapr(const WReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 1, 4, WReg(31), rt, adr);
}
void ldaddal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 0, rs, rt, adr);
}
void ldclral(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 1, rs, rt, adr);
}
void ldeoral(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 2, rs, rt, adr);
}
void ldsetal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 3, rs, rt, adr);
}
void ldsmaxal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 4, rs, rt, adr);
}
void ldsminal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 5, rs, rt, adr);
}
void ldumaxal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 6, rs, rt, adr);
}
void lduminal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 7, rs, rt, adr);
}
void swapal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 1, 0, rs, rt, adr);
}
void ldadd(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 0, rs, rt, adr);
}
void ldclr(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 1, rs, rt, adr);
}
void ldeor(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 2, rs, rt, adr);
}
void ldset(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 3, rs, rt, adr);
}
void ldsmax(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 4, rs, rt, adr);
}
void ldsmin(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 5, rs, rt, adr);
}
void ldumax(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 6, rs, rt, adr);
}
void ldumin(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 7, rs, rt, adr);
}
void swap(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 1, 0, rs, rt, adr);
}
void ldaddl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 0, rs, rt, adr);
}
void ldclrl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 1, rs, rt, adr);
}
void ldeorl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 2, rs, rt, adr);
}
void ldsetl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 3, rs, rt, adr);
}
void ldsmaxl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 4, rs, rt, adr);
}
void ldsminl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 5, rs, rt, adr);
}
void ldumaxl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 6, rs, rt, adr);
}
void lduminl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 7, rs, rt, adr);
}
void swapl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 1, 0, rs, rt, adr);
}
void ldadda(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 0, rs, rt, adr);
}
void ldclra(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 1, rs, rt, adr);
}
void ldeora(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 2, rs, rt, adr);
}
void ldseta(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 3, rs, rt, adr);
}
void ldsmaxa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 4, rs, rt, adr);
}
void ldsmina(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 5, rs, rt, adr);
}
void ldumaxa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 6, rs, rt, adr);
}
void ldumina(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 7, rs, rt, adr);
}
void swapa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 1, 0, rs, rt, adr);
}
void ldapr(const XReg &rt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 1, 4, XReg(31), rt, adr);
}
void ldaddal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 0, rs, rt, adr);
}
void ldclral(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 1, rs, rt, adr);
}
void ldeoral(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 2, rs, rt, adr);
}
void ldsetal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 3, rs, rt, adr);
}
void ldsmaxal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 4, rs, rt, adr);
}
void ldsminal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 5, rs, rt, adr);
}
void ldumaxal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 6, rs, rt, adr);
}
void lduminal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 7, rs, rt, adr);
}
void swapal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 1, 0, rs, rt, adr);
}
void staddb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 0, rs, WReg(31), adr);
}
void stclrb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 1, rs, WReg(31), adr);
}
void steorb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 2, rs, WReg(31), adr);
}
void stsetb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 3, rs, WReg(31), adr);
}
void stsmaxb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 4, rs, WReg(31), adr);
}
void stsminb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 5, rs, WReg(31), adr);
}
void stumaxb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 6, rs, WReg(31), adr);
}
void stuminb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 0, 0, 7, rs, WReg(31), adr);
}
void staddlb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 0, rs, WReg(31), adr);
}
void stclrlb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 1, rs, WReg(31), adr);
}
void steorlb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 2, rs, WReg(31), adr);
}
void stsetlb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 3, rs, WReg(31), adr);
}
void stsmaxlb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 4, rs, WReg(31), adr);
}
void stsminlb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 5, rs, WReg(31), adr);
}
void stumaxlb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 6, rs, WReg(31), adr);
}
void stuminlb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 0, 1, 0, 7, rs, WReg(31), adr);
}
void staddab(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 0, rs, WReg(31), adr);
}
void stclrab(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 1, rs, WReg(31), adr);
}
void steorab(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 2, rs, WReg(31), adr);
}
void stsetab(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 3, rs, WReg(31), adr);
}
void stsmaxab(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 4, rs, WReg(31), adr);
}
void stsminab(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 5, rs, WReg(31), adr);
}
void stumaxab(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 6, rs, WReg(31), adr);
}
void stuminab(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 0, 0, 7, rs, WReg(31), adr);
}
void staddalb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 0, rs, WReg(31), adr);
}
void stclralb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 1, rs, WReg(31), adr);
}
void steoralb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 2, rs, WReg(31), adr);
}
void stsetalb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 3, rs, WReg(31), adr);
}
void stsmaxalb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 4, rs, WReg(31), adr);
}
void stsminalb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 5, rs, WReg(31), adr);
}
void stumaxalb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 6, rs, WReg(31), adr);
}
void stuminalb(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(0, 0, 1, 1, 0, 7, rs, WReg(31), adr);
}
void staddh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 0, rs, WReg(31), adr);
}
void stclrh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 1, rs, WReg(31), adr);
}
void steorh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 2, rs, WReg(31), adr);
}
void stseth(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 3, rs, WReg(31), adr);
}
void stsmaxh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 4, rs, WReg(31), adr);
}
void stsminh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 5, rs, WReg(31), adr);
}
void stumaxh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 6, rs, WReg(31), adr);
}
void stuminh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 0, 0, 7, rs, WReg(31), adr);
}
void staddlh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 0, rs, WReg(31), adr);
}
void stclrlh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 1, rs, WReg(31), adr);
}
void steorlh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 2, rs, WReg(31), adr);
}
void stsetlh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 3, rs, WReg(31), adr);
}
void stsmaxlh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 4, rs, WReg(31), adr);
}
void stsminlh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 5, rs, WReg(31), adr);
}
void stumaxlh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 6, rs, WReg(31), adr);
}
void stuminlh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 0, 1, 0, 7, rs, WReg(31), adr);
}
void staddah(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 0, rs, WReg(31), adr);
}
void stclrah(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 1, rs, WReg(31), adr);
}
void steorah(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 2, rs, WReg(31), adr);
}
void stsetah(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 3, rs, WReg(31), adr);
}
void stsmaxah(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 4, rs, WReg(31), adr);
}
void stsminah(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 5, rs, WReg(31), adr);
}
void stumaxah(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 6, rs, WReg(31), adr);
}
void stuminah(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 0, 0, 7, rs, WReg(31), adr);
}
void staddalh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 0, rs, WReg(31), adr);
}
void stclralh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 1, rs, WReg(31), adr);
}
void steoralh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 2, rs, WReg(31), adr);
}
void stsetalh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 3, rs, WReg(31), adr);
}
void stsmaxalh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 4, rs, WReg(31), adr);
}
void stsminalh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 5, rs, WReg(31), adr);
}
void stumaxalh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 6, rs, WReg(31), adr);
}
void stuminalh(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(1, 0, 1, 1, 0, 7, rs, WReg(31), adr);
}
void stadd(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 0, rs, WReg(31), adr);
}
void stclr(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 1, rs, WReg(31), adr);
}
void steor(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 2, rs, WReg(31), adr);
}
void stset(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 3, rs, WReg(31), adr);
}
void stsmax(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 4, rs, WReg(31), adr);
}
void stsmin(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 5, rs, WReg(31), adr);
}
void stumax(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 6, rs, WReg(31), adr);
}
void stumin(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 0, 0, 7, rs, WReg(31), adr);
}
void staddl(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 0, rs, WReg(31), adr);
}
void stclrl(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 1, rs, WReg(31), adr);
}
void steorl(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 2, rs, WReg(31), adr);
}
void stsetl(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 3, rs, WReg(31), adr);
}
void stsmaxl(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 4, rs, WReg(31), adr);
}
void stsminl(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 5, rs, WReg(31), adr);
}
void stumaxl(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 6, rs, WReg(31), adr);
}
void stuminl(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 0, 1, 0, 7, rs, WReg(31), adr);
}
void stadda(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 0, rs, WReg(31), adr);
}
void stclra(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 1, rs, WReg(31), adr);
}
void steora(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 2, rs, WReg(31), adr);
}
void stseta(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 3, rs, WReg(31), adr);
}
void stsmaxa(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 4, rs, WReg(31), adr);
}
void stsmina(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 5, rs, WReg(31), adr);
}
void stumaxa(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 6, rs, WReg(31), adr);
}
void stumina(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 0, 0, 7, rs, WReg(31), adr);
}
void staddal(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 0, rs, WReg(31), adr);
}
void stclral(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 1, rs, WReg(31), adr);
}
void steoral(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 2, rs, WReg(31), adr);
}
void stsetal(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 3, rs, WReg(31), adr);
}
void stsmaxal(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 4, rs, WReg(31), adr);
}
void stsminal(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 5, rs, WReg(31), adr);
}
void stumaxal(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 6, rs, WReg(31), adr);
}
void stuminal(const WReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(2, 0, 1, 1, 0, 7, rs, WReg(31), adr);
}
void stadd(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 0, rs, XReg(31), adr);
}
void stclr(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 1, rs, XReg(31), adr);
}
void steor(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 2, rs, XReg(31), adr);
}
void stset(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 3, rs, XReg(31), adr);
}
void stsmax(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 4, rs, XReg(31), adr);
}
void stsmin(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 5, rs, XReg(31), adr);
}
void stumax(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 6, rs, XReg(31), adr);
}
void stumin(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 0, 0, 7, rs, XReg(31), adr);
}
void staddl(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 0, rs, XReg(31), adr);
}
void stclrl(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 1, rs, XReg(31), adr);
}
void steorl(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 2, rs, XReg(31), adr);
}
void stsetl(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 3, rs, XReg(31), adr);
}
void stsmaxl(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 4, rs, XReg(31), adr);
}
void stsminl(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 5, rs, XReg(31), adr);
}
void stumaxl(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 6, rs, XReg(31), adr);
}
void stuminl(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 0, 1, 0, 7, rs, XReg(31), adr);
}
void stadda(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 0, rs, XReg(31), adr);
}
void stclra(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 1, rs, XReg(31), adr);
}
void steora(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 2, rs, XReg(31), adr);
}
void stseta(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 3, rs, XReg(31), adr);
}
void stsmaxa(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 4, rs, XReg(31), adr);
}
void stsmina(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 5, rs, XReg(31), adr);
}
void stumaxa(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 6, rs, XReg(31), adr);
}
void stumina(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 0, 0, 7, rs, XReg(31), adr);
}
void staddal(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 0, rs, XReg(31), adr);
}
void stclral(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 1, rs, XReg(31), adr);
}
void steoral(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 2, rs, XReg(31), adr);
}
void stsetal(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 3, rs, XReg(31), adr);
}
void stsmaxal(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 4, rs, XReg(31), adr);
}
void stsminal(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 5, rs, XReg(31), adr);
}
void stumaxal(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 6, rs, XReg(31), adr);
}
void stuminal(const XReg &rs, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  AtomicMemOp(3, 0, 1, 1, 0, 7, rs, XReg(31), adr);
}
void strb(const WReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(0, 0, rt, adr);
}
void strb(const WReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(0, 0, rt, adr);
}
void ldrb(const WReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(0, 1, rt, adr);
}
void ldrb(const WReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(0, 1, rt, adr);
}
void ldrsb(const XReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(0, 2, rt, adr);
}
void ldrsb(const XReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(0, 2, rt, adr);
}
void ldrsb(const WReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(0, 3, rt, adr);
}
void ldrsb(const WReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(0, 3, rt, adr);
}
void strh(const WReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(1, 0, rt, adr);
}
void strh(const WReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(1, 0, rt, adr);
}
void ldrh(const WReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(1, 1, rt, adr);
}
void ldrh(const WReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(1, 1, rt, adr);
}
void ldrsh(const XReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(1, 2, rt, adr);
}
void ldrsh(const XReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(1, 2, rt, adr);
}
void ldrsh(const WReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(1, 3, rt, adr);
}
void ldrsh(const WReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(1, 3, rt, adr);
}
void str(const WReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(2, 0, rt, adr);
}
void str(const WReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(2, 0, rt, adr);
}
void ldr(const WReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(2, 1, rt, adr);
}
void ldr(const WReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(2, 1, rt, adr);
}
void ldrsw(const XReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(2, 2, rt, adr);
}
void ldrsw(const XReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(2, 2, rt, adr);
}
void str(const XReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(3, 0, rt, adr);
}
void str(const XReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(3, 0, rt, adr);
}
void ldr(const XReg &rt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(3, 1, rt, adr);
}
void ldr(const XReg &rt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStReg(3, 1, rt, adr);
}
void str(const BReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const HReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const SReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const DReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const QReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const BReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const HReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const SReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const DReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const QReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void ldr(const BReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const HReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const SReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const DReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const QReg &vt, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const BReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const HReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const SReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const DReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const QReg &vt, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void prfm(const Prfop prfop, const AdrReg &adr) {
  XBYAK_SET_CODE_INFO();
  PfExt(prfop, adr);
}
void prfm(const Prfop prfop, const AdrExt &adr) {
  XBYAK_SET_CODE_INFO();
  PfExt(prfop, adr);
}
void ldraa(const XReg &xt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPac(0, 0, xt, adr);
}
void ldrab(const XReg &xt, const AdrImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPac(1, 0, xt, adr);
}
void ldraa(const XReg &xt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPac(0, 1, xt, adr);
}
void ldrab(const XReg &xt, const AdrPreImm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegPac(1, 1, xt, adr);
}
void strb(const WReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(0, 0, rt, adr);
}
void ldrb(const WReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(0, 1, rt, adr);
}
void ldrsb(const XReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(0, 2, rt, adr);
}
void ldrsb(const WReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(0, 3, rt, adr);
}
void strh(const WReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(1, 0, rt, adr);
}
void ldrh(const WReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(1, 1, rt, adr);
}
void ldrsh(const XReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(1, 2, rt, adr);
}
void ldrsh(const WReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(1, 3, rt, adr);
}
void str(const WReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(2, 0, rt, adr);
}
void ldr(const WReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(2, 1, rt, adr);
}
void ldrsw(const XReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(2, 2, rt, adr);
}
void str(const XReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(3, 0, rt, adr);
}
void ldr(const XReg &rt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStRegUnImm(3, 1, rt, adr);
}
void str(const BReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const HReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const SReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const DReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void str(const QReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr);
}
void ldr(const BReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const HReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const SReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const DReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void ldr(const QReg &vt, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr);
}
void prfm(const Prfop prfop, const AdrUimm &adr) {
  XBYAK_SET_CODE_INFO();
  PfRegImm(prfop, adr);
}
void udiv(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(2, rd, rn, rm);
}
void sdiv(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(3, rd, rn, rm);
}
void lslv(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(8, rd, rn, rm);
}
void lsl(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(8, rd, rn, rm);
}
void lsrv(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(9, rd, rn, rm);
}
void lsr(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(9, rd, rn, rm);
}
void asrv(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(10, rd, rn, rm);
}
void asr(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(10, rd, rn, rm);
}
void rorv(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(11, rd, rn, rm);
}
void ror(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(11, rd, rn, rm);
}
void crc32b(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(16, rd, rn, rm);
}
void crc32h(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(17, rd, rn, rm);
}
void crc32w(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(18, rd, rn, rm);
}
void crc32cb(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(20, rd, rn, rm);
}
void crc32ch(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(21, rd, rn, rm);
}
void crc32cw(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(22, rd, rn, rm);
}
void udiv(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(2, rd, rn, rm);
}
void sdiv(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(3, rd, rn, rm);
}
void lslv(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(8, rd, rn, rm);
}
void lsl(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(8, rd, rn, rm);
}
void lsrv(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(9, rd, rn, rm);
}
void lsr(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(9, rd, rn, rm);
}
void asrv(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(10, rd, rn, rm);
}
void asr(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(10, rd, rn, rm);
}
void rorv(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(11, rd, rn, rm);
}
void ror(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(11, rd, rn, rm);
}
void pacga(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(12, rd, rn, rm);
}
void crc32x(const WReg &rd, const WReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(19, rd, rn, rm);
}
void crc32cx(const WReg &rd, const WReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc2Src(23, rd, rn, rm);
}
void rbit(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 0, rd, rn);
}
void rev16(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 1, rd, rn);
}
void rev(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 2, rd, rn);
}
void clz(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 4, rd, rn);
}
void cls(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 5, rd, rn);
}
void rbit(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 0, rd, rn);
}
void rev16(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 1, rd, rn);
}
void rev32(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 2, rd, rn);
}
void rev(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 3, rd, rn);
}
void rev64(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 3, rd, rn);
}
void clz(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 4, rd, rn);
}
void cls(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(0, 5, rd, rn);
}
void pacia(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 0, rd, rn);
}
void pacib(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 1, rd, rn);
}
void pacda(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 2, rd, rn);
}
void pacdb(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 3, rd, rn);
}
void autia(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 4, rd, rn);
}
void autib(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 5, rd, rn);
}
void autda(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 6, rd, rn);
}
void autdb(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 7, rd, rn);
}
void paciza(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 8, rd);
}
void pacizb(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 9, rd);
}
void pacdza(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 10, rd);
}
void pacdzb(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 11, rd);
}
void autiza(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 12, rd);
}
void autizb(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 13, rd);
}
void autdza(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 14, rd);
}
void autdzb(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 15, rd);
}
void xpaci(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 16, rd);
}
void xpacd(const XReg &rd) {
  XBYAK_SET_CODE_INFO();
  DataProc1Src(1, 17, rd);
}
void and_(const WReg &rd, const WReg &rn, const WReg &rm,
          const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(0, 0, rd, rn, rm, shmod, sh);
}
void and_(const XReg &rd, const XReg &rn, const XReg &rm,
          const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(0, 0, rd, rn, rm, shmod, sh);
}
void bic(const WReg &rd, const WReg &rn, const WReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(0, 1, rd, rn, rm, shmod, sh);
}
void bic(const XReg &rd, const XReg &rn, const XReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(0, 1, rd, rn, rm, shmod, sh);
}
void orr(const WReg &rd, const WReg &rn, const WReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(1, 0, rd, rn, rm, shmod, sh);
}
void orr(const XReg &rd, const XReg &rn, const XReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(1, 0, rd, rn, rm, shmod, sh);
}
void orn(const WReg &rd, const WReg &rn, const WReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(1, 1, rd, rn, rm, shmod, sh);
}
void orn(const XReg &rd, const XReg &rn, const XReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(1, 1, rd, rn, rm, shmod, sh);
}
void mvn(const WReg &rd, const WReg &rm, const ShMod shmod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(1, 1, rd, WReg(31), rm, shmod, sh);
}
void mvn(const XReg &rd, const XReg &rm, const ShMod shmod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(1, 1, rd, XReg(31), rm, shmod, sh);
}
void eor(const WReg &rd, const WReg &rn, const WReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(2, 0, rd, rn, rm, shmod, sh);
}
void eor(const XReg &rd, const XReg &rn, const XReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(2, 0, rd, rn, rm, shmod, sh);
}
void eon(const WReg &rd, const WReg &rn, const WReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(2, 1, rd, rn, rm, shmod, sh);
}
void eon(const XReg &rd, const XReg &rn, const XReg &rm,
         const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(2, 1, rd, rn, rm, shmod, sh);
}
void ands(const WReg &rd, const WReg &rn, const WReg &rm,
          const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(3, 0, rd, rn, rm, shmod, sh);
}
void ands(const XReg &rd, const XReg &rn, const XReg &rm,
          const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(3, 0, rd, rn, rm, shmod, sh);
}
void tst(const WReg &rn, const WReg &rm, const ShMod shmod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(3, 0, WReg(31), rn, rm, shmod, sh);
}
void tst(const XReg &rn, const XReg &rm, const ShMod shmod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(3, 0, XReg(31), rn, rm, shmod, sh);
}
void bics(const WReg &rd, const WReg &rn, const WReg &rm,
          const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(3, 1, rd, rn, rm, shmod, sh);
}
void bics(const XReg &rd, const XReg &rn, const XReg &rm,
          const ShMod shmod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  LogicalShiftReg(3, 1, rd, rn, rm, shmod, sh);
}
void mov(const WReg &rd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  MvReg(rd, rn);
}
void mov(const XReg &rd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  MvReg(rd, rn);
}
void add(const WReg &rd, const WReg &rn, const WReg &rm,
         const ShMod shmod = NONE, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(0, 0, rd, rn, rm, shmod, sh);
}
void add(const XReg &rd, const XReg &rn, const XReg &rm,
         const ShMod shmod = NONE, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(0, 0, rd, rn, rm, shmod, sh);
}
void adds(const WReg &rd, const WReg &rn, const WReg &rm,
          const ShMod shmod = NONE, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(0, 1, rd, rn, rm, shmod, sh);
}
void adds(const XReg &rd, const XReg &rn, const XReg &rm,
          const ShMod shmod = NONE, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(0, 1, rd, rn, rm, shmod, sh);
}
void cmn(const WReg &rn, const WReg &rm, const ShMod shmod = NONE,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(0, 1, WReg(31), rn, rm, shmod, sh, true);
}
void cmn(const XReg &rn, const XReg &rm, const ShMod shmod = NONE,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(0, 1, XReg(31), rn, rm, shmod, sh, true);
}
void sub(const WReg &rd, const WReg &rn, const WReg &rm,
         const ShMod shmod = NONE, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 0, rd, rn, rm, shmod, sh);
}
void sub(const XReg &rd, const XReg &rn, const XReg &rm,
         const ShMod shmod = NONE, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 0, rd, rn, rm, shmod, sh);
}
void neg(const WReg &rd, const WReg &rm, const ShMod shmod = NONE,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 0, rd, WReg(31), rm, shmod, sh, true);
}
void neg(const XReg &rd, const XReg &rm, const ShMod shmod = NONE,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 0, rd, XReg(31), rm, shmod, sh, true);
}
void subs(const WReg &rd, const WReg &rn, const WReg &rm,
          const ShMod shmod = NONE, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 1, rd, rn, rm, shmod, sh);
}
void subs(const XReg &rd, const XReg &rn, const XReg &rm,
          const ShMod shmod = NONE, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 1, rd, rn, rm, shmod, sh);
}
void negs(const WReg &rd, const WReg &rm, const ShMod shmod = NONE,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 1, rd, WReg(31), rm, shmod, sh, true);
}
void negs(const XReg &rd, const XReg &rm, const ShMod shmod = NONE,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 1, rd, XReg(31), rm, shmod, sh, true);
}
void cmp(const WReg &rn, const WReg &rm, const ShMod shmod = NONE,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 1, WReg(31), rn, rm, shmod, sh, true);
}
void cmp(const XReg &rn, const XReg &rm, const ShMod shmod = NONE,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubShiftReg(1, 1, XReg(31), rn, rm, shmod, sh, true);
}
void add(const WReg &rd, const WReg &rn, const WReg &rm, const ExtMod extmod,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(0, 0, rd, rn, rm, extmod, sh);
}
void add(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(0, 0, rd, rn, rm, extmod, sh);
}
void adds(const WReg &rd, const WReg &rn, const WReg &rm, const ExtMod extmod,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(0, 1, rd, rn, rm, extmod, sh);
}
void adds(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(0, 1, rd, rn, rm, extmod, sh);
}
void cmn(const WReg &rn, const WReg &rm, const ExtMod extmod,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(0, 1, WReg(31), rn, rm, extmod, sh);
}
void cmn(const XReg &rn, const XReg &rm, const ExtMod extmod,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(0, 1, XReg(31), rn, rm, extmod, sh);
}
void sub(const WReg &rd, const WReg &rn, const WReg &rm, const ExtMod extmod,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(1, 0, rd, rn, rm, extmod, sh);
}
void sub(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(1, 0, rd, rn, rm, extmod, sh);
}
void subs(const WReg &rd, const WReg &rn, const WReg &rm, const ExtMod extmod,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(1, 1, rd, rn, rm, extmod, sh);
}
void subs(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(1, 1, rd, rn, rm, extmod, sh);
}
void cmp(const WReg &rn, const WReg &rm, const ExtMod extmod,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(1, 1, WReg(31), rn, rm, extmod, sh);
}
void cmp(const XReg &rn, const XReg &rm, const ExtMod extmod,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AddSubExtReg(1, 1, XReg(31), rn, rm, extmod, sh);
}
void adc(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(0, 0, rd, rn, rm);
}
void adc(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(0, 0, rd, rn, rm);
}
void adcs(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(0, 1, rd, rn, rm);
}
void adcs(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(0, 1, rd, rn, rm);
}
void sbc(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(1, 0, rd, rn, rm);
}
void sbc(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(1, 0, rd, rn, rm);
}
void ngc(const WReg &rd, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(1, 0, rd, WReg(31), rm);
}
void ngc(const XReg &rd, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(1, 0, rd, XReg(31), rm);
}
void sbcs(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(1, 1, rd, rn, rm);
}
void sbcs(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(1, 1, rd, rn, rm);
}
void ngcs(const WReg &rd, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(1, 1, rd, WReg(31), rm);
}
void ngcs(const XReg &rd, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  AddSubCarry(1, 1, rd, XReg(31), rm);
}
void rmif(const XReg &xn, const uint32_t sh, const uint32_t mask) {
  XBYAK_SET_CODE_INFO();
  RotateR(0, 1, 0, xn, sh, mask);
}
void setf8(const WReg &wn) {
  XBYAK_SET_CODE_INFO();
  Evaluate(0, 1, 0, 0, 0, 13, wn);
}
void setf16(const WReg &wn) {
  XBYAK_SET_CODE_INFO();
  Evaluate(0, 1, 0, 1, 0, 13, wn);
}
void ccmn(const WReg &rn, const WReg &rm, const uint32_t nczv,
          const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondCompReg(0, 1, 0, 0, rn, rm, nczv, cond);
}
void ccmn(const XReg &rn, const XReg &rm, const uint32_t nczv,
          const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondCompReg(0, 1, 0, 0, rn, rm, nczv, cond);
}
void ccmp(const WReg &rn, const WReg &rm, const uint32_t nczv,
          const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondCompReg(1, 1, 0, 0, rn, rm, nczv, cond);
}
void ccmp(const XReg &rn, const XReg &rm, const uint32_t nczv,
          const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondCompReg(1, 1, 0, 0, rn, rm, nczv, cond);
}
void ccmn(const WReg &rn, const uint32_t imm, const uint32_t nczv,
          const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondCompImm(0, 1, 0, 0, rn, imm, nczv, cond);
}
void ccmn(const XReg &rn, const uint32_t imm, const uint32_t nczv,
          const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondCompImm(0, 1, 0, 0, rn, imm, nczv, cond);
}
void ccmp(const WReg &rn, const uint32_t imm, const uint32_t nczv,
          const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondCompImm(1, 1, 0, 0, rn, imm, nczv, cond);
}
void ccmp(const XReg &rn, const uint32_t imm, const uint32_t nczv,
          const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondCompImm(1, 1, 0, 0, rn, imm, nczv, cond);
}
void csel(const WReg &rd, const WReg &rn, const WReg &rm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(0, 0, 0, rd, rn, rm, cond);
}
void csel(const XReg &rd, const XReg &rn, const XReg &rm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(0, 0, 0, rd, rn, rm, cond);
}
void csinc(const WReg &rd, const WReg &rn, const WReg &rm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(0, 0, 1, rd, rn, rm, cond);
}
void csinc(const XReg &rd, const XReg &rn, const XReg &rm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(0, 0, 1, rd, rn, rm, cond);
}
void cinc(const WReg &rd, const WReg &rn, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(0, 0, 1, rd, rn, rn, invert(cond));
}
void cinc(const XReg &rd, const XReg &rn, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(0, 0, 1, rd, rn, rn, invert(cond));
}
void cset(const WReg &rd, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(0, 0, 1, rd, WReg(31), WReg(31), invert(cond));
}
void cset(const XReg &rd, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(0, 0, 1, rd, XReg(31), XReg(31), invert(cond));
}
void csinv(const WReg &rd, const WReg &rn, const WReg &rm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 0, rd, rn, rm, cond);
}
void csinv(const XReg &rd, const XReg &rn, const XReg &rm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 0, rd, rn, rm, cond);
}
void cinv(const WReg &rd, const WReg &rn, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 0, rd, rn, rn, invert(cond));
}
void cinv(const XReg &rd, const XReg &rn, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 0, rd, rn, rn, invert(cond));
}
void csetm(const WReg &rd, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 0, rd, WReg(31), WReg(31), invert(cond));
}
void csetm(const XReg &rd, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 0, rd, XReg(31), XReg(31), invert(cond));
}
void csneg(const WReg &rd, const WReg &rn, const WReg &rm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 1, rd, rn, rm, cond);
}
void csneg(const XReg &rd, const XReg &rn, const XReg &rm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 1, rd, rn, rm, cond);
}
void cneg(const WReg &rd, const WReg &rn, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 1, rd, rn, rn, invert(cond));
}
void cneg(const XReg &rd, const XReg &rn, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  CondSel(1, 0, 1, rd, rn, rn, invert(cond));
}
void madd(const WReg &rd, const WReg &rn, const WReg &rm, const WReg &ra) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 0, 0, rd, rn, rm, ra);
}
void madd(const XReg &rd, const XReg &rn, const XReg &rm, const XReg &ra) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 0, 0, rd, rn, rm, ra);
}
void mul(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 0, 0, rd, rn, rm, WReg(31));
}
void mul(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 0, 0, rd, rn, rm, XReg(31));
}
void msub(const WReg &rd, const WReg &rn, const WReg &rm, const WReg &ra) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 0, 1, rd, rn, rm, ra);
}
void msub(const XReg &rd, const XReg &rn, const XReg &rm, const XReg &ra) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 0, 1, rd, rn, rm, ra);
}
void mneg(const WReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 0, 1, rd, rn, rm, WReg(31));
}
void mneg(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 0, 1, rd, rn, rm, XReg(31));
}
void smaddl(const XReg &rd, const WReg &rn, const WReg &rm, const XReg &ra) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 1, 0, rd, rn, rm, ra);
}
void smull(const XReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 1, 0, rd, rn, rm, XReg(31));
}
void smsubl(const XReg &rd, const WReg &rn, const WReg &rm, const XReg &ra) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 1, 1, rd, rn, rm, ra);
}
void smnegl(const XReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 1, 1, rd, rn, rm, XReg(31));
}
void smulh(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 2, 0, rd, rn, rm);
}
void umaddl(const XReg &rd, const WReg &rn, const WReg &rm, const XReg &ra) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 5, 0, rd, rn, rm, ra);
}
void umull(const XReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 5, 0, rd, rn, rm, XReg(31));
}
void umsubl(const XReg &rd, const WReg &rn, const WReg &rm, const XReg &ra) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 5, 1, rd, rn, rm, ra);
}
void umnegl(const XReg &rd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 5, 1, rd, rn, rm, XReg(31));
}
void umulh(const XReg &rd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  DataProc3Reg(0, 6, 0, rd, rn, rm);
}
void aese(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  CryptAES(4, vd, vn);
}
void aesd(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  CryptAES(5, vd, vn);
}
void aesmc(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  CryptAES(6, vd, vn);
}
void aesimc(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  CryptAES(7, vd, vn);
}
void sha1c(const QReg &qd, const SReg &sn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypt3RegSHA(0, qd, sn, vm);
}
void sha1p(const QReg &qd, const SReg &sn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypt3RegSHA(1, qd, sn, vm);
}
void sha1m(const QReg &qd, const SReg &sn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypt3RegSHA(2, qd, sn, vm);
}
void sha1su0(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypt3RegSHA(3, vd, vn, vm);
}
void sha256h(const QReg &qd, const QReg &qn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypt3RegSHA(4, qd, qn, vm);
}
void sha256h2(const QReg &qd, const QReg &qn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypt3RegSHA(5, qd, qn, vm);
}
void sha256su1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypt3RegSHA(6, vd, vn, vm);
}
void sha1h(const SReg &sd, const SReg &sn) {
  XBYAK_SET_CODE_INFO();
  Crypt2RegSHA(0, sd, sn);
}
void sha1su1(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  Crypt2RegSHA(1, vd, vn);
}
void sha256su0(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  Crypt2RegSHA(2, vd, vn);
}
void dup(const BReg &vd, const VRegBElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScCopy(0, 0, vd, vn);
}
void dup(const HReg &vd, const VRegHElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScCopy(0, 0, vd, vn);
}
void dup(const SReg &vd, const VRegSElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScCopy(0, 0, vd, vn);
}
void dup(const DReg &vd, const VRegDElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScCopy(0, 0, vd, vn);
}
void mov(const BReg &vd, const VRegBElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScCopy(0, 0, vd, vn);
}
void mov(const HReg &vd, const VRegHElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScCopy(0, 0, vd, vn);
}
void mov(const SReg &vd, const VRegSElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScCopy(0, 0, vd, vn);
}
void mov(const DReg &vd, const VRegDElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScCopy(0, 0, vd, vn);
}
void fmulx(const HReg &hd, const HReg &hn, const HReg &hm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameFp16(0, 0, 3, hd, hn, hm);
}
void fcmeq(const HReg &hd, const HReg &hn, const HReg &hm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameFp16(0, 0, 4, hd, hn, hm);
}
void frecps(const HReg &hd, const HReg &hn, const HReg &hm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameFp16(0, 0, 7, hd, hn, hm);
}
void frsqrts(const HReg &hd, const HReg &hn, const HReg &hm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameFp16(0, 1, 7, hd, hn, hm);
}
void fcmge(const HReg &hd, const HReg &hn, const HReg &hm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameFp16(1, 0, 4, hd, hn, hm);
}
void facge(const HReg &hd, const HReg &hn, const HReg &hm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameFp16(1, 0, 5, hd, hn, hm);
}
void fabd(const HReg &hd, const HReg &hn, const HReg &hm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameFp16(1, 1, 2, hd, hn, hm);
}
void fcmgt(const HReg &hd, const HReg &hn, const HReg &hm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameFp16(1, 1, 4, hd, hn, hm);
}
void facgt(const HReg &hd, const HReg &hn, const HReg &hm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameFp16(1, 1, 5, hd, hn, hm);
}
void fcvtns(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 0, 26, hd, hn);
}
void fcvtms(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 0, 27, hd, hn);
}
void fcvtas(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 0, 28, hd, hn);
}
void scvtf(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 0, 29, hd, hn);
}
void fcmgt(const HReg &hd, const HReg &hn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 1, 12, hd, hn, zero);
}
void fcmeq(const HReg &hd, const HReg &hn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 1, 13, hd, hn, zero);
}
void fcmlt(const HReg &hd, const HReg &hn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 1, 14, hd, hn, zero);
}
void fcvtps(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 1, 26, hd, hn);
}
void fcvtzs(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 1, 27, hd, hn);
}
void frecpe(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 1, 29, hd, hn);
}
void frecpx(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(0, 1, 31, hd, hn);
}
void fcvtnu(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(1, 0, 26, hd, hn);
}
void fcvtmu(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(1, 0, 27, hd, hn);
}
void fcvtau(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(1, 0, 28, hd, hn);
}
void ucvtf(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(1, 0, 29, hd, hn);
}
void fcmge(const HReg &hd, const HReg &hn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(1, 1, 12, hd, hn, zero);
}
void fcmle(const HReg &hd, const HReg &hn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(1, 1, 13, hd, hn, zero);
}
void fcvtpu(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(1, 1, 26, hd, hn);
}
void fcvtzu(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(1, 1, 27, hd, hn);
}
void frsqrte(const HReg &hd, const HReg &hn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscFp16(1, 1, 29, hd, hn);
}
void sqrdmlah(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameExtra(1, 0, vd, vn, vm);
}
void sqrdmlah(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameExtra(1, 0, vd, vn, vm);
}
void sqrdmlsh(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameExtra(1, 1, vd, vn, vm);
}
void sqrdmlsh(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameExtra(1, 1, vd, vn, vm);
}
void suqadd(const BReg &vd, const BReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 3, vd, vn);
}
void suqadd(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 3, vd, vn);
}
void suqadd(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 3, vd, vn);
}
void suqadd(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 3, vd, vn);
}
void sqabs(const BReg &vd, const BReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 7, vd, vn);
}
void sqabs(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 7, vd, vn);
}
void sqabs(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 7, vd, vn);
}
void sqabs(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 7, vd, vn);
}
void cmgt(const DReg &vd, const DReg &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 8, vd, vn, zero);
}
void cmeq(const DReg &vd, const DReg &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 9, vd, vn, zero);
}
void cmlt(const DReg &vd, const DReg &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 10, vd, vn, zero);
}
void abs(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 11, vd, vn);
}
void sqxtn(const BReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 20, vd, vn);
}
void sqxtn(const HReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 20, vd, vn);
}
void sqxtn(const SReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(0, 20, vd, vn);
}
void usqadd(const BReg &vd, const BReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 3, vd, vn);
}
void usqadd(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 3, vd, vn);
}
void usqadd(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 3, vd, vn);
}
void usqadd(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 3, vd, vn);
}
void sqneg(const BReg &vd, const BReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 7, vd, vn);
}
void sqneg(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 7, vd, vn);
}
void sqneg(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 7, vd, vn);
}
void sqneg(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 7, vd, vn);
}
void cmge(const DReg &vd, const DReg &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 8, vd, vn, zero);
}
void cmle(const DReg &vd, const DReg &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 9, vd, vn, zero);
}
void neg(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 11, vd, vn);
}
void sqxtun(const BReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 18, vd, vn);
}
void sqxtun(const HReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 18, vd, vn);
}
void sqxtun(const SReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 18, vd, vn);
}
void uqxtn(const BReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 20, vd, vn);
}
void uqxtn(const HReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 20, vd, vn);
}
void uqxtn(const SReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMisc(1, 20, vd, vn);
}
void fcvtns(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(0, 26, vd, vn);
}
void fcvtns(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(0, 26, vd, vn);
}
void fcvtms(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(0, 27, vd, vn);
}
void fcvtms(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(0, 27, vd, vn);
}
void fcvtas(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(0, 28, vd, vn);
}
void fcvtas(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(0, 28, vd, vn);
}
void scvtf(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(0, 29, vd, vn);
}
void scvtf(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(0, 29, vd, vn);
}
void fcvtxn(const SReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(1, 22, vd, vn);
}
void fcvtnu(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(1, 26, vd, vn);
}
void fcvtnu(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(1, 26, vd, vn);
}
void fcvtmu(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(1, 27, vd, vn);
}
void fcvtmu(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(1, 27, vd, vn);
}
void fcvtau(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(1, 28, vd, vn);
}
void fcvtau(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(1, 28, vd, vn);
}
void ucvtf(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(1, 29, vd, vn);
}
void ucvtf(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz0x(1, 29, vd, vn);
}
void fcmgt(const SReg &vd, const SReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 12, vd, vn, zero);
}
void fcmgt(const DReg &vd, const DReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 12, vd, vn, zero);
}
void fcmeq(const SReg &vd, const SReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 13, vd, vn, zero);
}
void fcmeq(const DReg &vd, const DReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 13, vd, vn, zero);
}
void fcmlt(const SReg &vd, const SReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 14, vd, vn, zero);
}
void fcmlt(const DReg &vd, const DReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 14, vd, vn, zero);
}
void fcvtps(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 26, vd, vn);
}
void fcvtps(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 26, vd, vn);
}
void fcvtzs(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 27, vd, vn);
}
void fcvtzs(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 27, vd, vn);
}
void frecpe(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 29, vd, vn);
}
void frecpe(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 29, vd, vn);
}
void frecpx(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 31, vd, vn);
}
void frecpx(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(0, 31, vd, vn);
}
void fcmge(const SReg &vd, const SReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 12, vd, vn, zero);
}
void fcmge(const DReg &vd, const DReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 12, vd, vn, zero);
}
void fcmle(const SReg &vd, const SReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 13, vd, vn, zero);
}
void fcmle(const DReg &vd, const DReg &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 13, vd, vn, zero);
}
void fcvtpu(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 26, vd, vn);
}
void fcvtpu(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 26, vd, vn);
}
void fcvtzu(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 27, vd, vn);
}
void fcvtzu(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 27, vd, vn);
}
void frsqrte(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 29, vd, vn);
}
void frsqrte(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc2RegMiscSz1x(1, 29, vd, vn);
}
void addp(const DReg &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(0, 3, 27, vd, vn);
}
void fmaxnmp(const HReg &vd, const VReg2H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(0, 0, 12, vd, vn);
}
void faddp(const HReg &vd, const VReg2H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(0, 0, 13, vd, vn);
}
void fmaxp(const HReg &vd, const VReg2H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(0, 0, 15, vd, vn);
}
void fminnmp(const HReg &vd, const VReg2H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(0, 2, 12, vd, vn);
}
void fminp(const HReg &vd, const VReg2H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(0, 2, 15, vd, vn);
}
void fmaxnmp(const SReg &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 12, vd, vn);
}
void fmaxnmp(const DReg &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 12, vd, vn);
}
void faddp(const SReg &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 13, vd, vn);
}
void faddp(const DReg &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 13, vd, vn);
}
void fmaxp(const SReg &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 15, vd, vn);
}
void fmaxp(const DReg &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 15, vd, vn);
}
void fminnmp(const SReg &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 2 : 3, 12, vd, vn);
}
void fminnmp(const DReg &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 2 : 3, 12, vd, vn);
}
void fminp(const SReg &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 2 : 3, 15, vd, vn);
}
void fminp(const DReg &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScPairwise(1, (vd.getBit() == 32) ? 2 : 3, 15, vd, vn);
}
void sqdmlal(const SReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Diff(0, 9, vd, vn, vm);
}
void sqdmlal(const DReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Diff(0, 9, vd, vn, vm);
}
void sqdmlsl(const SReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Diff(0, 11, vd, vn, vm);
}
void sqdmlsl(const DReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Diff(0, 11, vd, vn, vm);
}
void sqdmull(const SReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Diff(0, 13, vd, vn, vm);
}
void sqdmull(const DReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Diff(0, 13, vd, vn, vm);
}
void sqadd(const BReg &vd, const BReg &vn, const BReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 1, vd, vn, vm);
}
void sqadd(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 1, vd, vn, vm);
}
void sqadd(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 1, vd, vn, vm);
}
void sqadd(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 1, vd, vn, vm);
}
void sqsub(const BReg &vd, const BReg &vn, const BReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 5, vd, vn, vm);
}
void sqsub(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 5, vd, vn, vm);
}
void sqsub(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 5, vd, vn, vm);
}
void sqsub(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 5, vd, vn, vm);
}
void cmgt(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 6, vd, vn, vm);
}
void cmge(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 7, vd, vn, vm);
}
void sshl(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 8, vd, vn, vm);
}
void sqshl(const BReg &vd, const BReg &vn, const BReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 9, vd, vn, vm);
}
void sqshl(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 9, vd, vn, vm);
}
void sqshl(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 9, vd, vn, vm);
}
void sqshl(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 9, vd, vn, vm);
}
void srshl(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 10, vd, vn, vm);
}
void sqrshl(const BReg &vd, const BReg &vn, const BReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 11, vd, vn, vm);
}
void sqrshl(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 11, vd, vn, vm);
}
void sqrshl(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 11, vd, vn, vm);
}
void sqrshl(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 11, vd, vn, vm);
}
void add(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 16, vd, vn, vm);
}
void cmtst(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 17, vd, vn, vm);
}
void sqdmulh(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 22, vd, vn, vm);
}
void sqdmulh(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(0, 22, vd, vn, vm);
}
void uqadd(const BReg &vd, const BReg &vn, const BReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 1, vd, vn, vm);
}
void uqadd(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 1, vd, vn, vm);
}
void uqadd(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 1, vd, vn, vm);
}
void uqadd(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 1, vd, vn, vm);
}
void uqsub(const BReg &vd, const BReg &vn, const BReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 5, vd, vn, vm);
}
void uqsub(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 5, vd, vn, vm);
}
void uqsub(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 5, vd, vn, vm);
}
void uqsub(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 5, vd, vn, vm);
}
void cmhi(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 6, vd, vn, vm);
}
void cmhs(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 7, vd, vn, vm);
}
void ushl(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 8, vd, vn, vm);
}
void uqshl(const BReg &vd, const BReg &vn, const BReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 9, vd, vn, vm);
}
void uqshl(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 9, vd, vn, vm);
}
void uqshl(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 9, vd, vn, vm);
}
void uqshl(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 9, vd, vn, vm);
}
void urshl(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 10, vd, vn, vm);
}
void uqrshl(const BReg &vd, const BReg &vn, const BReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 11, vd, vn, vm);
}
void uqrshl(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 11, vd, vn, vm);
}
void uqrshl(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 11, vd, vn, vm);
}
void uqrshl(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 11, vd, vn, vm);
}
void sub(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 16, vd, vn, vm);
}
void cmeq(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 17, vd, vn, vm);
}
void sqrdmulh(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 22, vd, vn, vm);
}
void sqrdmulh(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3Same(1, 22, vd, vn, vm);
}
void fmulx(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(0, 27, vd, vn, vm);
}
void fmulx(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(0, 27, vd, vn, vm);
}
void fcmeq(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(0, 28, vd, vn, vm);
}
void fcmeq(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(0, 28, vd, vn, vm);
}
void frecps(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(0, 31, vd, vn, vm);
}
void frecps(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(0, 31, vd, vn, vm);
}
void fcmge(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(1, 28, vd, vn, vm);
}
void fcmge(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(1, 28, vd, vn, vm);
}
void facge(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(1, 29, vd, vn, vm);
}
void facge(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz0x(1, 29, vd, vn, vm);
}
void frsqrts(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz1x(0, 31, vd, vn, vm);
}
void frsqrts(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz1x(0, 31, vd, vn, vm);
}
void fabd(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz1x(1, 26, vd, vn, vm);
}
void fabd(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz1x(1, 26, vd, vn, vm);
}
void fcmgt(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz1x(1, 28, vd, vn, vm);
}
void fcmgt(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz1x(1, 28, vd, vn, vm);
}
void facgt(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz1x(1, 29, vd, vn, vm);
}
void facgt(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdSc3SameSz1x(1, 29, vd, vn, vm);
}
void sshr(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 0, vd, vn, imm);
}
void ssra(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 2, vd, vn, imm);
}
void srshr(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 4, vd, vn, imm);
}
void srsra(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 6, vd, vn, imm);
}
void shl(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 10, vd, vn, imm);
}
void sqshl(const BReg &vd, const BReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 14, vd, vn, imm);
}
void sqshl(const HReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 14, vd, vn, imm);
}
void sqshl(const SReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 14, vd, vn, imm);
}
void sqshl(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 14, vd, vn, imm);
}
void sqshl(const BReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 14, vd, vn, imm);
}
void sqshl(const HReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 14, vd, vn, imm);
}
void sqshl(const SReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 14, vd, vn, imm);
}
void sqshrn(const BReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 18, vd, vn, imm);
}
void sqshrn(const HReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 18, vd, vn, imm);
}
void sqshrn(const SReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 18, vd, vn, imm);
}
void sqrshrn(const BReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 19, vd, vn, imm);
}
void sqrshrn(const HReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 19, vd, vn, imm);
}
void sqrshrn(const SReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 19, vd, vn, imm);
}
void scvtf(const HReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 28, vd, vn, imm);
}
void scvtf(const SReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 28, vd, vn, imm);
}
void scvtf(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 28, vd, vn, imm);
}
void fcvtzs(const HReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 31, vd, vn, imm);
}
void fcvtzs(const SReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 31, vd, vn, imm);
}
void fcvtzs(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(0, 31, vd, vn, imm);
}
void ushr(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 0, vd, vn, imm);
}
void usra(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 2, vd, vn, imm);
}
void urshr(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 4, vd, vn, imm);
}
void ursra(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 6, vd, vn, imm);
}
void sri(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 8, vd, vn, imm);
}
void sli(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 10, vd, vn, imm);
}
void sqshlu(const BReg &vd, const BReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 12, vd, vn, imm);
}
void sqshlu(const HReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 12, vd, vn, imm);
}
void sqshlu(const SReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 12, vd, vn, imm);
}
void sqshlu(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 12, vd, vn, imm);
}
void sqshlu(const BReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 12, vd, vn, imm);
}
void sqshlu(const HReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 12, vd, vn, imm);
}
void sqshlu(const SReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 12, vd, vn, imm);
}
void uqshl(const BReg &vd, const BReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 14, vd, vn, imm);
}
void uqshl(const HReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 14, vd, vn, imm);
}
void uqshl(const SReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 14, vd, vn, imm);
}
void uqshl(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 14, vd, vn, imm);
}
void uqshl(const BReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 14, vd, vn, imm);
}
void uqshl(const HReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 14, vd, vn, imm);
}
void uqshl(const SReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 14, vd, vn, imm);
}
void sqshrun(const BReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 16, vd, vn, imm);
}
void sqshrun(const HReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 16, vd, vn, imm);
}
void sqshrun(const SReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 16, vd, vn, imm);
}
void sqrshrun(const BReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 17, vd, vn, imm);
}
void sqrshrun(const HReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 17, vd, vn, imm);
}
void sqrshrun(const SReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 17, vd, vn, imm);
}
void uqshrn(const BReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 18, vd, vn, imm);
}
void uqshrn(const HReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 18, vd, vn, imm);
}
void uqshrn(const SReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 18, vd, vn, imm);
}
void uqrshrn(const BReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 19, vd, vn, imm);
}
void uqrshrn(const HReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 19, vd, vn, imm);
}
void uqrshrn(const SReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 19, vd, vn, imm);
}
void ucvtf(const HReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 28, vd, vn, imm);
}
void ucvtf(const SReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 28, vd, vn, imm);
}
void ucvtf(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 28, vd, vn, imm);
}
void fcvtzu(const HReg &vd, const HReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 31, vd, vn, imm);
}
void fcvtzu(const SReg &vd, const SReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 31, vd, vn, imm);
}
void fcvtzu(const DReg &vd, const DReg &vn, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScShImm(1, 31, vd, vn, imm);
}
void sqdmlal(const SReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 3, vd, vn, vm);
}
void sqdmlal(const DReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 3, vd, vn, vm);
}
void sqdmlsl(const SReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 7, vd, vn, vm);
}
void sqdmlsl(const DReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 7, vd, vn, vm);
}
void sqdmull(const SReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 11, vd, vn, vm);
}
void sqdmull(const DReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 11, vd, vn, vm);
}
void sqdmulh(const HReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 12, vd, vn, vm);
}
void sqdmulh(const SReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 12, vd, vn, vm);
}
void sqrdmulh(const HReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 13, vd, vn, vm);
}
void sqrdmulh(const SReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(0, 13, vd, vn, vm);
}
void sqrdmlah(const HReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(1, 13, vd, vn, vm);
}
void sqrdmlah(const SReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(1, 13, vd, vn, vm);
}
void sqrdmlsh(const HReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(1, 15, vd, vn, vm);
}
void sqrdmlsh(const SReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElem(1, 15, vd, vn, vm);
}
void fmla(const HReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(0, 0, 1, vd, vn, vm);
}
void fmls(const HReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(0, 0, 5, vd, vn, vm);
}
void fmul(const HReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(0, 0, 9, vd, vn, vm);
}
void fmla(const SReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(0, 2, 1, vd, vn, vm);
}
void fmls(const SReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(0, 2, 5, vd, vn, vm);
}
void fmul(const SReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(0, 2, 9, vd, vn, vm);
}
void fmla(const DReg &vd, const DReg &vn, const VRegDElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(0, 3, 1, vd, vn, vm);
}
void fmls(const DReg &vd, const DReg &vn, const VRegDElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(0, 3, 5, vd, vn, vm);
}
void fmul(const DReg &vd, const DReg &vn, const VRegDElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(0, 3, 9, vd, vn, vm);
}
void fmulx(const HReg &vd, const HReg &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(1, 0, 9, vd, vn, vm);
}
void fmulx(const SReg &vd, const SReg &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(1, 2, 9, vd, vn, vm);
}
void fmulx(const DReg &vd, const DReg &vn, const VRegDElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdScXIndElemSz(1, 3, 9, vd, vn, vm);
}
void tbl(const VReg8B &vd, const VReg16B &vn, const uint32_t len,
         const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdTblLkup(0, len, 0, vd, vn, vm);
}
void tbl(const VReg16B &vd, const VReg16B &vn, const uint32_t len,
         const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdTblLkup(0, len, 0, vd, vn, vm);
}
void tbx(const VReg8B &vd, const VReg16B &vn, const uint32_t len,
         const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdTblLkup(0, len, 1, vd, vn, vm);
}
void tbx(const VReg16B &vd, const VReg16B &vn, const uint32_t len,
         const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdTblLkup(0, len, 1, vd, vn, vm);
}
void tbl(const VReg8B &vd, const VReg16BList &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdTblLkup(0, 0, vd, vn, vm);
}
void tbl(const VReg16B &vd, const VReg16BList &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdTblLkup(0, 0, vd, vn, vm);
}
void tbx(const VReg8B &vd, const VReg16BList &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdTblLkup(0, 1, vd, vn, vm);
}
void tbx(const VReg16B &vd, const VReg16BList &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdTblLkup(0, 1, vd, vn, vm);
}
void uzp1(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(1, vd, vn, vm);
}
void uzp1(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(1, vd, vn, vm);
}
void uzp1(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(1, vd, vn, vm);
}
void uzp1(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(1, vd, vn, vm);
}
void uzp1(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(1, vd, vn, vm);
}
void uzp1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(1, vd, vn, vm);
}
void uzp1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(1, vd, vn, vm);
}
void trn1(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(2, vd, vn, vm);
}
void trn1(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(2, vd, vn, vm);
}
void trn1(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(2, vd, vn, vm);
}
void trn1(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(2, vd, vn, vm);
}
void trn1(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(2, vd, vn, vm);
}
void trn1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(2, vd, vn, vm);
}
void trn1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(2, vd, vn, vm);
}
void zip1(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(3, vd, vn, vm);
}
void zip1(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(3, vd, vn, vm);
}
void zip1(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(3, vd, vn, vm);
}
void zip1(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(3, vd, vn, vm);
}
void zip1(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(3, vd, vn, vm);
}
void zip1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(3, vd, vn, vm);
}
void zip1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(3, vd, vn, vm);
}
void uzp2(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(5, vd, vn, vm);
}
void uzp2(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(5, vd, vn, vm);
}
void uzp2(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(5, vd, vn, vm);
}
void uzp2(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(5, vd, vn, vm);
}
void uzp2(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(5, vd, vn, vm);
}
void uzp2(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(5, vd, vn, vm);
}
void uzp2(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(5, vd, vn, vm);
}
void trn2(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(6, vd, vn, vm);
}
void trn2(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(6, vd, vn, vm);
}
void trn2(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(6, vd, vn, vm);
}
void trn2(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(6, vd, vn, vm);
}
void trn2(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(6, vd, vn, vm);
}
void trn2(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(6, vd, vn, vm);
}
void trn2(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(6, vd, vn, vm);
}
void zip2(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(7, vd, vn, vm);
}
void zip2(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(7, vd, vn, vm);
}
void zip2(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(7, vd, vn, vm);
}
void zip2(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(7, vd, vn, vm);
}
void zip2(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(7, vd, vn, vm);
}
void zip2(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(7, vd, vn, vm);
}
void zip2(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdPermute(7, vd, vn, vm);
}
void ext(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm,
         const uint32_t index) {
  XBYAK_SET_CODE_INFO();
  AdvSimdExtract(0, vd, vn, vm, index);
}
void ext(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm,
         const uint32_t index) {
  XBYAK_SET_CODE_INFO();
  AdvSimdExtract(0, vd, vn, vm, index);
}
void dup(const VReg8B &vd, const VRegBElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupElem(0, 0, vd, vn);
}
void dup(const VReg16B &vd, const VRegBElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupElem(0, 0, vd, vn);
}
void dup(const VReg4H &vd, const VRegHElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupElem(0, 0, vd, vn);
}
void dup(const VReg8H &vd, const VRegHElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupElem(0, 0, vd, vn);
}
void dup(const VReg2S &vd, const VRegSElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupElem(0, 0, vd, vn);
}
void dup(const VReg4S &vd, const VRegSElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupElem(0, 0, vd, vn);
}
void dup(const VReg2D &vd, const VRegDElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupElem(0, 0, vd, vn);
}
void dup(const VReg8B &vd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupGen(0, 0, vd, rn);
}
void dup(const VReg16B &vd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupGen(0, 0, vd, rn);
}
void dup(const VReg4H &vd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupGen(0, 0, vd, rn);
}
void dup(const VReg8H &vd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupGen(0, 0, vd, rn);
}
void dup(const VReg2S &vd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupGen(0, 0, vd, rn);
}
void dup(const VReg4S &vd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupGen(0, 0, vd, rn);
}
void dup(const VReg2D &vd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyDupGen(0, 0, vd, rn);
}
void smov(const WReg &rd, const VRegBElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 5, rd, vn);
}
void smov(const WReg &rd, const VRegHElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 5, rd, vn);
}
void smov(const XReg &rd, const VRegBElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 5, rd, vn);
}
void smov(const XReg &rd, const VRegHElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 5, rd, vn);
}
void smov(const XReg &rd, const VRegSElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 5, rd, vn);
}
void umov(const WReg &rd, const VRegBElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 7, rd, vn);
}
void umov(const WReg &rd, const VRegHElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 7, rd, vn);
}
void umov(const WReg &rd, const VRegSElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 7, rd, vn);
}
void umov(const XReg &rd, const VRegDElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 7, rd, vn);
}
void mov(const WReg &rd, const VRegSElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 7, rd, vn);
}
void mov(const XReg &rd, const VRegDElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyMov(0, 7, rd, vn);
}
#ifdef DNNL_AARCH64
void ins_(const VRegBElem &vd, const WReg &rn) {
#else
void ins(const VRegBElem &vd, const WReg &rn) {
#endif
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
#ifdef DNNL_AARCH64
void ins_(const VRegHElem &vd, const WReg &rn) {
#else
void ins(const VRegHElem &vd, const WReg &rn) {
#endif
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
#ifdef DNNL_AARCH64
void ins_(const VRegSElem &vd, const WReg &rn) {
#else
void ins(const VRegSElem &vd, const WReg &rn) {
#endif
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
#ifdef DNNL_AARCH64
void ins_(const VRegDElem &vd, const XReg &rn) {
#else
void ins(const VRegDElem &vd, const XReg &rn) {
#endif
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
void mov(const VRegBElem &vd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
void mov(const VRegHElem &vd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
void mov(const VRegSElem &vd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
void mov(const VRegDElem &vd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
#ifdef DNNL_AARCH64
void ins_(const VRegBElem &vd, const VRegBElem &vn) {
#else
void ins(const VRegBElem &vd, const VRegBElem &vn) {
#endif
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyElemIns(1, vd, vn);
}
#ifdef DNNL_AARCH64
void ins_(const VRegHElem &vd, const VRegHElem &vn) {
#else
void ins(const VRegHElem &vd, const VRegHElem &vn) {
#endif
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyElemIns(1, vd, vn);
}
#ifdef DNNL_AARCH64
void ins_(const VRegSElem &vd, const VRegSElem &vn) {
#else
void ins(const VRegSElem &vd, const VRegSElem &vn) {
#endif
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyElemIns(1, vd, vn);
}
#ifdef DNNL_AARCH64
void ins_(const VRegDElem &vd, const VRegDElem &vn) {
#else
void ins(const VRegDElem &vd, const VRegDElem &vn) {
#endif
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyElemIns(1, vd, vn);
}
void mov(const VRegBElem &vd, const VRegBElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyElemIns(1, vd, vn);
}
void mov(const VRegHElem &vd, const VRegHElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyElemIns(1, vd, vn);
}
void mov(const VRegSElem &vd, const VRegSElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyElemIns(1, vd, vn);
}
void mov(const VRegDElem &vd, const VRegDElem &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdCopyElemIns(1, vd, vn);
}
void fmaxnm(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 0, vd, vn, vm);
}
void fmaxnm(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 0, vd, vn, vm);
}
void fmla(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 1, vd, vn, vm);
}
void fmla(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 1, vd, vn, vm);
}
void fadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 2, vd, vn, vm);
}
void fadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 2, vd, vn, vm);
}
void fmulx(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 3, vd, vn, vm);
}
void fmulx(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 3, vd, vn, vm);
}
void fcmeq(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 4, vd, vn, vm);
}
void fcmeq(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 4, vd, vn, vm);
}
void fmax(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 6, vd, vn, vm);
}
void fmax(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 6, vd, vn, vm);
}
void frecps(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 7, vd, vn, vm);
}
void frecps(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 0, 7, vd, vn, vm);
}
void fminnm(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 0, vd, vn, vm);
}
void fminnm(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 0, vd, vn, vm);
}
void fmls(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 1, vd, vn, vm);
}
void fmls(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 1, vd, vn, vm);
}
void fsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 2, vd, vn, vm);
}
void fsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 2, vd, vn, vm);
}
void fmin(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 6, vd, vn, vm);
}
void fmin(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 6, vd, vn, vm);
}
void frsqrts(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 7, vd, vn, vm);
}
void frsqrts(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(0, 1, 7, vd, vn, vm);
}
void fmaxnmp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 0, vd, vn, vm);
}
void fmaxnmp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 0, vd, vn, vm);
}
void faddp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 2, vd, vn, vm);
}
void faddp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 2, vd, vn, vm);
}
void fmul(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 3, vd, vn, vm);
}
void fmul(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 3, vd, vn, vm);
}
void fcmge(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 4, vd, vn, vm);
}
void fcmge(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 4, vd, vn, vm);
}
void facge(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 5, vd, vn, vm);
}
void facge(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 5, vd, vn, vm);
}
void fmaxp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 6, vd, vn, vm);
}
void fmaxp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 6, vd, vn, vm);
}
void fdiv(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 7, vd, vn, vm);
}
void fdiv(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 0, 7, vd, vn, vm);
}
void fminnmp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 0, vd, vn, vm);
}
void fminnmp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 0, vd, vn, vm);
}
void fabd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 2, vd, vn, vm);
}
void fabd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 2, vd, vn, vm);
}
void fcmgt(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 4, vd, vn, vm);
}
void fcmgt(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 4, vd, vn, vm);
}
void facgt(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 5, vd, vn, vm);
}
void facgt(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 5, vd, vn, vm);
}
void fminp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 6, vd, vn, vm);
}
void fminp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameFp16(1, 1, 6, vd, vn, vm);
}
void frintn(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 24, vd, vn);
}
void frintn(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 24, vd, vn);
}
void frintm(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 25, vd, vn);
}
void frintm(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 25, vd, vn);
}
void fcvtns(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 26, vd, vn);
}
void fcvtns(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 26, vd, vn);
}
void fcvtms(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 27, vd, vn);
}
void fcvtms(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 27, vd, vn);
}
void fcvtas(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 28, vd, vn);
}
void fcvtas(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 28, vd, vn);
}
void scvtf(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 29, vd, vn);
}
void scvtf(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 0, 29, vd, vn);
}
void fcmgt(const VReg4H &vd, const VReg4H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 12, vd, vn, zero);
}
void fcmgt(const VReg8H &vd, const VReg8H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 12, vd, vn, zero);
}
void fcmeq(const VReg4H &vd, const VReg4H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 13, vd, vn, zero);
}
void fcmeq(const VReg8H &vd, const VReg8H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 13, vd, vn, zero);
}
void fcmlt(const VReg4H &vd, const VReg4H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 14, vd, vn, zero);
}
void fcmlt(const VReg8H &vd, const VReg8H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 14, vd, vn, zero);
}
void fabs(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 15, vd, vn);
}
void fabs(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 15, vd, vn);
}
void frintp(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 24, vd, vn);
}
void frintp(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 24, vd, vn);
}
void frintz(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 25, vd, vn);
}
void frintz(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 25, vd, vn);
}
void fcvtps(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 26, vd, vn);
}
void fcvtps(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 26, vd, vn);
}
void fcvtzs(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 27, vd, vn);
}
void fcvtzs(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 27, vd, vn);
}
void frecpe(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 29, vd, vn);
}
void frecpe(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(0, 1, 29, vd, vn);
}
void frinta(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 24, vd, vn);
}
void frinta(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 24, vd, vn);
}
void frintx(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 25, vd, vn);
}
void frintx(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 25, vd, vn);
}
void fcvtnu(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 26, vd, vn);
}
void fcvtnu(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 26, vd, vn);
}
void fcvtmu(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 27, vd, vn);
}
void fcvtmu(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 27, vd, vn);
}
void fcvtau(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 28, vd, vn);
}
void fcvtau(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 28, vd, vn);
}
void ucvtf(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 29, vd, vn);
}
void ucvtf(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 0, 29, vd, vn);
}
void fcmge(const VReg4H &vd, const VReg4H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 12, vd, vn, zero);
}
void fcmge(const VReg8H &vd, const VReg8H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 12, vd, vn, zero);
}
void fcmle(const VReg4H &vd, const VReg4H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 13, vd, vn, zero);
}
void fcmle(const VReg8H &vd, const VReg8H &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 13, vd, vn, zero);
}
void fneg(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 15, vd, vn);
}
void fneg(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 15, vd, vn);
}
void frinti(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 25, vd, vn);
}
void frinti(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 25, vd, vn);
}
void fcvtpu(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 26, vd, vn);
}
void fcvtpu(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 26, vd, vn);
}
void fcvtzu(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 27, vd, vn);
}
void fcvtzu(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 27, vd, vn);
}
void frsqrte(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 29, vd, vn);
}
void frsqrte(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 29, vd, vn);
}
void fsqrt(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 31, vd, vn);
}
void fsqrt(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscFp16(1, 1, 31, vd, vn);
}
void sdot(const VReg2S &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(0, 2, vd, vn, vm);
}
void sdot(const VReg4S &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(0, 2, vd, vn, vm);
}
void sqrdmlah(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 0, vd, vn, vm);
}
void sqrdmlah(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 0, vd, vn, vm);
}
void sqrdmlah(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 0, vd, vn, vm);
}
void sqrdmlah(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 0, vd, vn, vm);
}
void sqrdmlsh(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 1, vd, vn, vm);
}
void sqrdmlsh(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 1, vd, vn, vm);
}
void sqrdmlsh(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 1, vd, vn, vm);
}
void sqrdmlsh(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 1, vd, vn, vm);
}
void udot(const VReg2S &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 2, vd, vn, vm);
}
void udot(const VReg4S &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtra(1, 2, vd, vn, vm);
}
void fcmla(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate);
}
void fcmla(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate);
}
void fcmla(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate);
}
void fcmla(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate);
}
void fcmla(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate);
}
void fcadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate);
}
void fcadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate);
}
void fcadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate);
}
void fcadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate);
}
void fcadd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate);
}
void rev64(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 0, vd, vn);
}
void rev64(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 0, vd, vn);
}
void rev64(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 0, vd, vn);
}
void rev64(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 0, vd, vn);
}
void rev64(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 0, vd, vn);
}
void rev64(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 0, vd, vn);
}
void rev16(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 1, vd, vn);
}
void rev16(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 1, vd, vn);
}
void saddlp(const VReg4H &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 2, vd, vn);
}
void saddlp(const VReg8H &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 2, vd, vn);
}
void saddlp(const VReg2S &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 2, vd, vn);
}
void saddlp(const VReg4S &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 2, vd, vn);
}
void saddlp(const VReg1D &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 2, vd, vn);
}
void saddlp(const VReg2D &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 2, vd, vn);
}
void suqadd(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 3, vd, vn);
}
void suqadd(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 3, vd, vn);
}
void suqadd(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 3, vd, vn);
}
void suqadd(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 3, vd, vn);
}
void suqadd(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 3, vd, vn);
}
void suqadd(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 3, vd, vn);
}
void suqadd(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 3, vd, vn);
}
void cls(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 4, vd, vn);
}
void cls(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 4, vd, vn);
}
void cls(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 4, vd, vn);
}
void cls(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 4, vd, vn);
}
void cls(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 4, vd, vn);
}
void cls(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 4, vd, vn);
}
void cnt(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 5, vd, vn);
}
void cnt(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 5, vd, vn);
}
void sadalp(const VReg4H &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 6, vd, vn);
}
void sadalp(const VReg8H &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 6, vd, vn);
}
void sadalp(const VReg2S &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 6, vd, vn);
}
void sadalp(const VReg4S &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 6, vd, vn);
}
void sadalp(const VReg1D &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 6, vd, vn);
}
void sadalp(const VReg2D &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 6, vd, vn);
}
void sqabs(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 7, vd, vn);
}
void sqabs(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 7, vd, vn);
}
void sqabs(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 7, vd, vn);
}
void sqabs(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 7, vd, vn);
}
void sqabs(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 7, vd, vn);
}
void sqabs(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 7, vd, vn);
}
void sqabs(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 7, vd, vn);
}
void abs(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 11, vd, vn);
}
void abs(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 11, vd, vn);
}
void abs(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 11, vd, vn);
}
void abs(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 11, vd, vn);
}
void abs(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 11, vd, vn);
}
void abs(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 11, vd, vn);
}
void abs(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 11, vd, vn);
}
void xtn(const VReg8B &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 18, vd, vn);
}
void xtn(const VReg4H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 18, vd, vn);
}
void xtn(const VReg2S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 18, vd, vn);
}
void xtn2(const VReg16B &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 18, vd, vn);
}
void xtn2(const VReg8H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 18, vd, vn);
}
void xtn2(const VReg4S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 18, vd, vn);
}
void sqxtn(const VReg8B &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 20, vd, vn);
}
void sqxtn(const VReg4H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 20, vd, vn);
}
void sqxtn(const VReg2S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 20, vd, vn);
}
void sqxtn2(const VReg16B &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 20, vd, vn);
}
void sqxtn2(const VReg8H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 20, vd, vn);
}
void sqxtn2(const VReg4S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(0, 20, vd, vn);
}
void rev32(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 0, vd, vn);
}
void rev32(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 0, vd, vn);
}
void rev32(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 0, vd, vn);
}
void rev32(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 0, vd, vn);
}
void uaddlp(const VReg4H &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 2, vd, vn);
}
void uaddlp(const VReg8H &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 2, vd, vn);
}
void uaddlp(const VReg2S &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 2, vd, vn);
}
void uaddlp(const VReg4S &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 2, vd, vn);
}
void uaddlp(const VReg1D &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 2, vd, vn);
}
void uaddlp(const VReg2D &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 2, vd, vn);
}
void usqadd(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 3, vd, vn);
}
void usqadd(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 3, vd, vn);
}
void usqadd(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 3, vd, vn);
}
void usqadd(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 3, vd, vn);
}
void usqadd(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 3, vd, vn);
}
void usqadd(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 3, vd, vn);
}
void usqadd(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 3, vd, vn);
}
void clz(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 4, vd, vn);
}
void clz(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 4, vd, vn);
}
void clz(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 4, vd, vn);
}
void clz(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 4, vd, vn);
}
void clz(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 4, vd, vn);
}
void clz(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 4, vd, vn);
}
void uadalp(const VReg4H &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 6, vd, vn);
}
void uadalp(const VReg8H &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 6, vd, vn);
}
void uadalp(const VReg2S &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 6, vd, vn);
}
void uadalp(const VReg4S &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 6, vd, vn);
}
void uadalp(const VReg1D &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 6, vd, vn);
}
void uadalp(const VReg2D &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 6, vd, vn);
}
void sqneg(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 7, vd, vn);
}
void sqneg(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 7, vd, vn);
}
void sqneg(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 7, vd, vn);
}
void sqneg(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 7, vd, vn);
}
void sqneg(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 7, vd, vn);
}
void sqneg(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 7, vd, vn);
}
void sqneg(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 7, vd, vn);
}
void neg(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 11, vd, vn);
}
void neg(const VReg4H &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 11, vd, vn);
}
void neg(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 11, vd, vn);
}
void neg(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 11, vd, vn);
}
void neg(const VReg8H &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 11, vd, vn);
}
void neg(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 11, vd, vn);
}
void neg(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 11, vd, vn);
}
void sqxtun(const VReg8B &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 18, vd, vn);
}
void sqxtun(const VReg4H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 18, vd, vn);
}
void sqxtun(const VReg2S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 18, vd, vn);
}
void sqxtun2(const VReg16B &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 18, vd, vn);
}
void sqxtun2(const VReg8H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 18, vd, vn);
}
void sqxtun2(const VReg4S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 18, vd, vn);
}
void shll(const VReg8H &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 19, vd, vn, sh);
}
void shll(const VReg4S &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 19, vd, vn, sh);
}
void shll(const VReg2D &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 19, vd, vn, sh);
}
void shll2(const VReg8H &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 19, vd, vn, sh);
}
void shll2(const VReg4S &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 19, vd, vn, sh);
}
void shll2(const VReg2D &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 19, vd, vn, sh);
}
void uqxtn(const VReg8B &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 20, vd, vn);
}
void uqxtn(const VReg4H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 20, vd, vn);
}
void uqxtn(const VReg2S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 20, vd, vn);
}
void uqxtn2(const VReg16B &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 20, vd, vn);
}
void uqxtn2(const VReg8H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 20, vd, vn);
}
void uqxtn2(const VReg4S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMisc(1, 20, vd, vn);
}
void cmgt(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 8, vd, vn, zero);
}
void cmgt(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 8, vd, vn, zero);
}
void cmgt(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 8, vd, vn, zero);
}
void cmgt(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 8, vd, vn, zero);
}
void cmgt(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 8, vd, vn, zero);
}
void cmgt(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 8, vd, vn, zero);
}
void cmgt(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 8, vd, vn, zero);
}
void cmeq(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 9, vd, vn, zero);
}
void cmeq(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 9, vd, vn, zero);
}
void cmeq(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 9, vd, vn, zero);
}
void cmeq(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 9, vd, vn, zero);
}
void cmeq(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 9, vd, vn, zero);
}
void cmeq(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 9, vd, vn, zero);
}
void cmeq(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 9, vd, vn, zero);
}
void cmlt(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 10, vd, vn, zero);
}
void cmlt(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 10, vd, vn, zero);
}
void cmlt(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 10, vd, vn, zero);
}
void cmlt(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 10, vd, vn, zero);
}
void cmlt(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 10, vd, vn, zero);
}
void cmlt(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 10, vd, vn, zero);
}
void cmlt(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(0, 10, vd, vn, zero);
}
void cmge(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 8, vd, vn, zero);
}
void cmge(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 8, vd, vn, zero);
}
void cmge(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 8, vd, vn, zero);
}
void cmge(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 8, vd, vn, zero);
}
void cmge(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 8, vd, vn, zero);
}
void cmge(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 8, vd, vn, zero);
}
void cmge(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 8, vd, vn, zero);
}
void cmle(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 9, vd, vn, zero);
}
void cmle(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 9, vd, vn, zero);
}
void cmle(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 9, vd, vn, zero);
}
void cmle(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 9, vd, vn, zero);
}
void cmle(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 9, vd, vn, zero);
}
void cmle(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 9, vd, vn, zero);
}
void cmle(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscZero(1, 9, vd, vn, zero);
}
void not_(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz(1, 0, 5, vd, vn);
}
void not_(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz(1, 0, 5, vd, vn);
}
void mvn(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz(1, 0, 5, vd, vn);
}
void mvn(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz(1, 0, 5, vd, vn);
}
void rbit(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz(1, 1, 5, vd, vn);
}
void rbit(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz(1, 1, 5, vd, vn);
}
void fcvtn(const VReg4H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 22, vd, vn);
}
void fcvtn(const VReg2S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 22, vd, vn);
}
void fcvtn2(const VReg8H &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 22, vd, vn);
}
void fcvtn2(const VReg4S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 22, vd, vn);
}
void fcvtl(const VReg4S &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 23, vd, vn);
}
void fcvtl(const VReg2D &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 23, vd, vn);
}
void fcvtl2(const VReg4S &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 23, vd, vn);
}
void fcvtl2(const VReg2D &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 23, vd, vn);
}
void frintn(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 24, vd, vn);
}
void frintn(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 24, vd, vn);
}
void frintn(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 24, vd, vn);
}
void frintm(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 25, vd, vn);
}
void frintm(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 25, vd, vn);
}
void frintm(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 25, vd, vn);
}
void fcvtns(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 26, vd, vn);
}
void fcvtns(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 26, vd, vn);
}
void fcvtns(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 26, vd, vn);
}
void fcvtms(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 27, vd, vn);
}
void fcvtms(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 27, vd, vn);
}
void fcvtms(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 27, vd, vn);
}
void fcvtas(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 28, vd, vn);
}
void fcvtas(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 28, vd, vn);
}
void fcvtas(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 28, vd, vn);
}
void scvtf(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 29, vd, vn);
}
void scvtf(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 29, vd, vn);
}
void scvtf(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(0, 29, vd, vn);
}
void fcvtxn(const VReg2S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 22, vd, vn);
}
void fcvtxn(const VReg4S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 22, vd, vn);
}
void fcvtxn2(const VReg2S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 22, vd, vn);
}
void fcvtxn2(const VReg4S &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 22, vd, vn);
}
void frinta(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 24, vd, vn);
}
void frinta(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 24, vd, vn);
}
void frinta(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 24, vd, vn);
}
void frintx(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 25, vd, vn);
}
void frintx(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 25, vd, vn);
}
void frintx(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 25, vd, vn);
}
void fcvtnu(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 26, vd, vn);
}
void fcvtnu(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 26, vd, vn);
}
void fcvtnu(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 26, vd, vn);
}
void fcvtmu(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 27, vd, vn);
}
void fcvtmu(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 27, vd, vn);
}
void fcvtmu(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 27, vd, vn);
}
void fcvtau(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 28, vd, vn);
}
void fcvtau(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 28, vd, vn);
}
void fcvtau(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 28, vd, vn);
}
void ucvtf(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 29, vd, vn);
}
void ucvtf(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 29, vd, vn);
}
void ucvtf(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz0x(1, 29, vd, vn);
}
void fcmgt(const VReg2S &vd, const VReg2S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 12, vd, vn, zero);
}
void fcmgt(const VReg4S &vd, const VReg4S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 12, vd, vn, zero);
}
void fcmgt(const VReg2D &vd, const VReg2D &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 12, vd, vn, zero);
}
void fcmeq(const VReg2S &vd, const VReg2S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 13, vd, vn, zero);
}
void fcmeq(const VReg4S &vd, const VReg4S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 13, vd, vn, zero);
}
void fcmeq(const VReg2D &vd, const VReg2D &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 13, vd, vn, zero);
}
void fcmlt(const VReg2S &vd, const VReg2S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 14, vd, vn, zero);
}
void fcmlt(const VReg4S &vd, const VReg4S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 14, vd, vn, zero);
}
void fcmlt(const VReg2D &vd, const VReg2D &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 14, vd, vn, zero);
}
void fabs(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 15, vd, vn);
}
void fabs(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 15, vd, vn);
}
void fabs(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 15, vd, vn);
}
void frintp(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 24, vd, vn);
}
void frintp(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 24, vd, vn);
}
void frintp(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 24, vd, vn);
}
void frintz(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 25, vd, vn);
}
void frintz(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 25, vd, vn);
}
void frintz(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 25, vd, vn);
}
void fcvtps(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 26, vd, vn);
}
void fcvtps(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 26, vd, vn);
}
void fcvtps(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 26, vd, vn);
}
void fcvtzs(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 27, vd, vn);
}
void fcvtzs(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 27, vd, vn);
}
void fcvtzs(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 27, vd, vn);
}
void urecpe(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 28, vd, vn);
}
void urecpe(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 28, vd, vn);
}
void frecpe(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 29, vd, vn);
}
void frecpe(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 29, vd, vn);
}
void frecpe(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(0, 29, vd, vn);
}
void fcmge(const VReg2S &vd, const VReg2S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 12, vd, vn, zero);
}
void fcmge(const VReg4S &vd, const VReg4S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 12, vd, vn, zero);
}
void fcmge(const VReg2D &vd, const VReg2D &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 12, vd, vn, zero);
}
void fcmle(const VReg2S &vd, const VReg2S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 13, vd, vn, zero);
}
void fcmle(const VReg4S &vd, const VReg4S &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 13, vd, vn, zero);
}
void fcmle(const VReg2D &vd, const VReg2D &vn, const double zero) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 13, vd, vn, zero);
}
void fneg(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 15, vd, vn);
}
void fneg(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 15, vd, vn);
}
void fneg(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 15, vd, vn);
}
void frinti(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 25, vd, vn);
}
void frinti(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 25, vd, vn);
}
void frinti(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 25, vd, vn);
}
void fcvtpu(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 26, vd, vn);
}
void fcvtpu(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 26, vd, vn);
}
void fcvtpu(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 26, vd, vn);
}
void fcvtzu(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 27, vd, vn);
}
void fcvtzu(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 27, vd, vn);
}
void fcvtzu(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 27, vd, vn);
}
void ursqrte(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 28, vd, vn);
}
void ursqrte(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 28, vd, vn);
}
void frsqrte(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 29, vd, vn);
}
void frsqrte(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 29, vd, vn);
}
void frsqrte(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 29, vd, vn);
}
void fsqrt(const VReg2S &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 31, vd, vn);
}
void fsqrt(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 31, vd, vn);
}
void fsqrt(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd2RegMiscSz1x(1, 31, vd, vn);
}
void saddlv(const HReg &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 3, vd, vn);
}
void saddlv(const HReg &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 3, vd, vn);
}
void saddlv(const SReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 3, vd, vn);
}
void saddlv(const SReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 3, vd, vn);
}
void saddlv(const DReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 3, vd, vn);
}
void smaxv(const BReg &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 10, vd, vn);
}
void smaxv(const BReg &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 10, vd, vn);
}
void smaxv(const HReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 10, vd, vn);
}
void smaxv(const HReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 10, vd, vn);
}
void smaxv(const SReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 10, vd, vn);
}
void sminv(const BReg &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 26, vd, vn);
}
void sminv(const BReg &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 26, vd, vn);
}
void sminv(const HReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 26, vd, vn);
}
void sminv(const HReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 26, vd, vn);
}
void sminv(const SReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 26, vd, vn);
}
void addv(const BReg &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 27, vd, vn);
}
void addv(const BReg &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 27, vd, vn);
}
void addv(const HReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 27, vd, vn);
}
void addv(const HReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 27, vd, vn);
}
void addv(const SReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(0, 27, vd, vn);
}
void uaddlv(const HReg &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 3, vd, vn);
}
void uaddlv(const HReg &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 3, vd, vn);
}
void uaddlv(const SReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 3, vd, vn);
}
void uaddlv(const SReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 3, vd, vn);
}
void uaddlv(const DReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 3, vd, vn);
}
void umaxv(const BReg &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 10, vd, vn);
}
void umaxv(const BReg &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 10, vd, vn);
}
void umaxv(const HReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 10, vd, vn);
}
void umaxv(const HReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 10, vd, vn);
}
void umaxv(const SReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 10, vd, vn);
}
void uminv(const BReg &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 26, vd, vn);
}
void uminv(const BReg &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 26, vd, vn);
}
void uminv(const HReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 26, vd, vn);
}
void uminv(const HReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 26, vd, vn);
}
void uminv(const SReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanes(1, 26, vd, vn);
}
void fmaxnmv(const HReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz0x(0, 12, vd, vn);
}
void fmaxnmv(const HReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz0x(0, 12, vd, vn);
}
void fmaxv(const HReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz0x(0, 15, vd, vn);
}
void fmaxv(const HReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz0x(0, 15, vd, vn);
}
void fmaxnmv(const SReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz0x(1, 12, vd, vn);
}
void fmaxv(const SReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz0x(1, 15, vd, vn);
}
void fminnmv(const HReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz1x(0, 12, vd, vn);
}
void fminnmv(const HReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz1x(0, 12, vd, vn);
}
void fminv(const HReg &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz1x(0, 15, vd, vn);
}
void fminv(const HReg &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz1x(0, 15, vd, vn);
}
void fminnmv(const SReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz1x(1, 12, vd, vn);
}
void fminv(const SReg &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdAcrossLanesSz1x(1, 15, vd, vn);
}
void saddl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 0, vd, vn, vm);
}
void saddl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 0, vd, vn, vm);
}
void saddl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 0, vd, vn, vm);
}
void saddl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 0, vd, vn, vm);
}
void saddl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 0, vd, vn, vm);
}
void saddl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 0, vd, vn, vm);
}
void saddw(const VReg8H &vd, const VReg8H &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 1, vd, vn, vm);
}
void saddw(const VReg4S &vd, const VReg4S &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 1, vd, vn, vm);
}
void saddw(const VReg2D &vd, const VReg2D &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 1, vd, vn, vm);
}
void saddw2(const VReg8H &vd, const VReg8H &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 1, vd, vn, vm);
}
void saddw2(const VReg4S &vd, const VReg4S &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 1, vd, vn, vm);
}
void saddw2(const VReg2D &vd, const VReg2D &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 1, vd, vn, vm);
}
void ssubl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 2, vd, vn, vm);
}
void ssubl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 2, vd, vn, vm);
}
void ssubl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 2, vd, vn, vm);
}
void ssubl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 2, vd, vn, vm);
}
void ssubl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 2, vd, vn, vm);
}
void ssubl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 2, vd, vn, vm);
}
void ssubw(const VReg8H &vd, const VReg8H &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 3, vd, vn, vm);
}
void ssubw(const VReg4S &vd, const VReg4S &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 3, vd, vn, vm);
}
void ssubw(const VReg2D &vd, const VReg2D &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 3, vd, vn, vm);
}
void ssubw2(const VReg8H &vd, const VReg8H &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 3, vd, vn, vm);
}
void ssubw2(const VReg4S &vd, const VReg4S &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 3, vd, vn, vm);
}
void ssubw2(const VReg2D &vd, const VReg2D &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 3, vd, vn, vm);
}
void addhn(const VReg8B &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 4, vd, vn, vm);
}
void addhn(const VReg4H &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 4, vd, vn, vm);
}
void addhn(const VReg2S &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 4, vd, vn, vm);
}
void addhn2(const VReg16B &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 4, vd, vn, vm);
}
void addhn2(const VReg8H &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 4, vd, vn, vm);
}
void addhn2(const VReg4S &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 4, vd, vn, vm);
}
void sabal(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 5, vd, vn, vm);
}
void sabal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 5, vd, vn, vm);
}
void sabal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 5, vd, vn, vm);
}
void sabal2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 5, vd, vn, vm);
}
void sabal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 5, vd, vn, vm);
}
void sabal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 5, vd, vn, vm);
}
void subhn(const VReg8B &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 6, vd, vn, vm);
}
void subhn(const VReg4H &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 6, vd, vn, vm);
}
void subhn(const VReg2S &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 6, vd, vn, vm);
}
void subhn2(const VReg16B &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 6, vd, vn, vm);
}
void subhn2(const VReg8H &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 6, vd, vn, vm);
}
void subhn2(const VReg4S &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 6, vd, vn, vm);
}
void sabdl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 7, vd, vn, vm);
}
void sabdl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 7, vd, vn, vm);
}
void sabdl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 7, vd, vn, vm);
}
void sabdl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 7, vd, vn, vm);
}
void sabdl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 7, vd, vn, vm);
}
void sabdl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 7, vd, vn, vm);
}
void smlal(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 8, vd, vn, vm);
}
void smlal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 8, vd, vn, vm);
}
void smlal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 8, vd, vn, vm);
}
void smlal2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 8, vd, vn, vm);
}
void smlal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 8, vd, vn, vm);
}
void smlal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 8, vd, vn, vm);
}
void sqdmlal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 9, vd, vn, vm);
}
void sqdmlal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 9, vd, vn, vm);
}
void sqdmlal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 9, vd, vn, vm);
}
void sqdmlal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 9, vd, vn, vm);
}
void smlsl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 10, vd, vn, vm);
}
void smlsl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 10, vd, vn, vm);
}
void smlsl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 10, vd, vn, vm);
}
void smlsl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 10, vd, vn, vm);
}
void smlsl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 10, vd, vn, vm);
}
void smlsl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 10, vd, vn, vm);
}
void sqdmlsl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 11, vd, vn, vm);
}
void sqdmlsl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 11, vd, vn, vm);
}
void sqdmlsl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 11, vd, vn, vm);
}
void sqdmlsl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 11, vd, vn, vm);
}
void smull(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 12, vd, vn, vm);
}
void smull(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 12, vd, vn, vm);
}
void smull(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 12, vd, vn, vm);
}
void smull2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 12, vd, vn, vm);
}
void smull2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 12, vd, vn, vm);
}
void smull2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 12, vd, vn, vm);
}
void sqdmull(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 13, vd, vn, vm);
}
void sqdmull(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 13, vd, vn, vm);
}
void sqdmull2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 13, vd, vn, vm);
}
void sqdmull2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 13, vd, vn, vm);
}
void pmull(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 14, vd, vn, vm);
}
void pmull(const VReg1Q &vd, const VReg1D &vn, const VReg1D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 14, vd, vn, vm);
}
void pmull2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 14, vd, vn, vm);
}
void pmull2(const VReg1Q &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(0, 14, vd, vn, vm);
}
void uaddl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 0, vd, vn, vm);
}
void uaddl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 0, vd, vn, vm);
}
void uaddl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 0, vd, vn, vm);
}
void uaddl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 0, vd, vn, vm);
}
void uaddl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 0, vd, vn, vm);
}
void uaddl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 0, vd, vn, vm);
}
void uaddw(const VReg8H &vd, const VReg8H &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 1, vd, vn, vm);
}
void uaddw(const VReg4S &vd, const VReg4S &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 1, vd, vn, vm);
}
void uaddw(const VReg2D &vd, const VReg2D &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 1, vd, vn, vm);
}
void uaddw2(const VReg8H &vd, const VReg8H &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 1, vd, vn, vm);
}
void uaddw2(const VReg4S &vd, const VReg4S &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 1, vd, vn, vm);
}
void uaddw2(const VReg2D &vd, const VReg2D &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 1, vd, vn, vm);
}
void usubl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 2, vd, vn, vm);
}
void usubl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 2, vd, vn, vm);
}
void usubl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 2, vd, vn, vm);
}
void usubl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 2, vd, vn, vm);
}
void usubl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 2, vd, vn, vm);
}
void usubl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 2, vd, vn, vm);
}
void usubw(const VReg8H &vd, const VReg8H &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 3, vd, vn, vm);
}
void usubw(const VReg4S &vd, const VReg4S &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 3, vd, vn, vm);
}
void usubw(const VReg2D &vd, const VReg2D &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 3, vd, vn, vm);
}
void usubw2(const VReg8H &vd, const VReg8H &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 3, vd, vn, vm);
}
void usubw2(const VReg4S &vd, const VReg4S &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 3, vd, vn, vm);
}
void usubw2(const VReg2D &vd, const VReg2D &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 3, vd, vn, vm);
}
void raddhn(const VReg8B &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 4, vd, vn, vm);
}
void raddhn(const VReg4H &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 4, vd, vn, vm);
}
void raddhn(const VReg2S &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 4, vd, vn, vm);
}
void raddhn2(const VReg16B &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 4, vd, vn, vm);
}
void raddhn2(const VReg8H &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 4, vd, vn, vm);
}
void raddhn2(const VReg4S &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 4, vd, vn, vm);
}
void uabal(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 5, vd, vn, vm);
}
void uabal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 5, vd, vn, vm);
}
void uabal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 5, vd, vn, vm);
}
void uabal2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 5, vd, vn, vm);
}
void uabal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 5, vd, vn, vm);
}
void uabal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 5, vd, vn, vm);
}
void rsubhn(const VReg8B &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 6, vd, vn, vm);
}
void rsubhn(const VReg4H &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 6, vd, vn, vm);
}
void rsubhn(const VReg2S &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 6, vd, vn, vm);
}
void rsubhn2(const VReg16B &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 6, vd, vn, vm);
}
void rsubhn2(const VReg8H &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 6, vd, vn, vm);
}
void rsubhn2(const VReg4S &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 6, vd, vn, vm);
}
void uabdl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 7, vd, vn, vm);
}
void uabdl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 7, vd, vn, vm);
}
void uabdl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 7, vd, vn, vm);
}
void uabdl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 7, vd, vn, vm);
}
void uabdl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 7, vd, vn, vm);
}
void uabdl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 7, vd, vn, vm);
}
void umlal(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 8, vd, vn, vm);
}
void umlal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 8, vd, vn, vm);
}
void umlal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 8, vd, vn, vm);
}
void umlal2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 8, vd, vn, vm);
}
void umlal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 8, vd, vn, vm);
}
void umlal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 8, vd, vn, vm);
}
void umlsl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 10, vd, vn, vm);
}
void umlsl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 10, vd, vn, vm);
}
void umlsl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 10, vd, vn, vm);
}
void umlsl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 10, vd, vn, vm);
}
void umlsl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 10, vd, vn, vm);
}
void umlsl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 10, vd, vn, vm);
}
void umull(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 12, vd, vn, vm);
}
void umull(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 12, vd, vn, vm);
}
void umull(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 12, vd, vn, vm);
}
void umull2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 12, vd, vn, vm);
}
void umull2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 12, vd, vn, vm);
}
void umull2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Diff(1, 12, vd, vn, vm);
}
void shadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 0, vd, vn, vm);
}
void shadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 0, vd, vn, vm);
}
void shadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 0, vd, vn, vm);
}
void shadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 0, vd, vn, vm);
}
void shadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 0, vd, vn, vm);
}
void shadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 0, vd, vn, vm);
}
void sqadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 1, vd, vn, vm);
}
void sqadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 1, vd, vn, vm);
}
void sqadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 1, vd, vn, vm);
}
void sqadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 1, vd, vn, vm);
}
void sqadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 1, vd, vn, vm);
}
void sqadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 1, vd, vn, vm);
}
void sqadd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 1, vd, vn, vm);
}
void srhadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 2, vd, vn, vm);
}
void srhadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 2, vd, vn, vm);
}
void srhadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 2, vd, vn, vm);
}
void srhadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 2, vd, vn, vm);
}
void srhadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 2, vd, vn, vm);
}
void srhadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 2, vd, vn, vm);
}
void shsub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 4, vd, vn, vm);
}
void shsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 4, vd, vn, vm);
}
void shsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 4, vd, vn, vm);
}
void shsub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 4, vd, vn, vm);
}
void shsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 4, vd, vn, vm);
}
void shsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 4, vd, vn, vm);
}
void sqsub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 5, vd, vn, vm);
}
void sqsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 5, vd, vn, vm);
}
void sqsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 5, vd, vn, vm);
}
void sqsub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 5, vd, vn, vm);
}
void sqsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 5, vd, vn, vm);
}
void sqsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 5, vd, vn, vm);
}
void sqsub(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 5, vd, vn, vm);
}
void cmgt(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 6, vd, vn, vm);
}
void cmgt(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 6, vd, vn, vm);
}
void cmgt(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 6, vd, vn, vm);
}
void cmgt(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 6, vd, vn, vm);
}
void cmgt(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 6, vd, vn, vm);
}
void cmgt(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 6, vd, vn, vm);
}
void cmgt(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 6, vd, vn, vm);
}
void cmge(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 7, vd, vn, vm);
}
void cmge(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 7, vd, vn, vm);
}
void cmge(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 7, vd, vn, vm);
}
void cmge(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 7, vd, vn, vm);
}
void cmge(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 7, vd, vn, vm);
}
void cmge(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 7, vd, vn, vm);
}
void cmge(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 7, vd, vn, vm);
}
void sshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 8, vd, vn, vm);
}
void sshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 8, vd, vn, vm);
}
void sshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 8, vd, vn, vm);
}
void sshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 8, vd, vn, vm);
}
void sshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 8, vd, vn, vm);
}
void sshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 8, vd, vn, vm);
}
void sshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 8, vd, vn, vm);
}
void sqshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 9, vd, vn, vm);
}
void sqshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 9, vd, vn, vm);
}
void sqshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 9, vd, vn, vm);
}
void sqshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 9, vd, vn, vm);
}
void sqshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 9, vd, vn, vm);
}
void sqshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 9, vd, vn, vm);
}
void sqshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 9, vd, vn, vm);
}
void srshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 10, vd, vn, vm);
}
void srshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 10, vd, vn, vm);
}
void srshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 10, vd, vn, vm);
}
void srshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 10, vd, vn, vm);
}
void srshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 10, vd, vn, vm);
}
void srshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 10, vd, vn, vm);
}
void srshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 10, vd, vn, vm);
}
void sqrshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 11, vd, vn, vm);
}
void sqrshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 11, vd, vn, vm);
}
void sqrshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 11, vd, vn, vm);
}
void sqrshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 11, vd, vn, vm);
}
void sqrshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 11, vd, vn, vm);
}
void sqrshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 11, vd, vn, vm);
}
void sqrshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 11, vd, vn, vm);
}
void smax(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 12, vd, vn, vm);
}
void smax(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 12, vd, vn, vm);
}
void smax(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 12, vd, vn, vm);
}
void smax(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 12, vd, vn, vm);
}
void smax(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 12, vd, vn, vm);
}
void smax(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 12, vd, vn, vm);
}
void smin(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 13, vd, vn, vm);
}
void smin(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 13, vd, vn, vm);
}
void smin(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 13, vd, vn, vm);
}
void smin(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 13, vd, vn, vm);
}
void smin(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 13, vd, vn, vm);
}
void smin(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 13, vd, vn, vm);
}
void sabd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 14, vd, vn, vm);
}
void sabd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 14, vd, vn, vm);
}
void sabd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 14, vd, vn, vm);
}
void sabd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 14, vd, vn, vm);
}
void sabd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 14, vd, vn, vm);
}
void sabd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 14, vd, vn, vm);
}
void saba(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 15, vd, vn, vm);
}
void saba(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 15, vd, vn, vm);
}
void saba(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 15, vd, vn, vm);
}
void saba(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 15, vd, vn, vm);
}
void saba(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 15, vd, vn, vm);
}
void saba(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 15, vd, vn, vm);
}
void add(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 16, vd, vn, vm);
}
void add(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 16, vd, vn, vm);
}
void add(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 16, vd, vn, vm);
}
void add(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 16, vd, vn, vm);
}
void add(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 16, vd, vn, vm);
}
void add(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 16, vd, vn, vm);
}
void add(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 16, vd, vn, vm);
}
void cmtst(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 17, vd, vn, vm);
}
void cmtst(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 17, vd, vn, vm);
}
void cmtst(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 17, vd, vn, vm);
}
void cmtst(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 17, vd, vn, vm);
}
void cmtst(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 17, vd, vn, vm);
}
void cmtst(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 17, vd, vn, vm);
}
void cmtst(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 17, vd, vn, vm);
}
void mla(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 18, vd, vn, vm);
}
void mla(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 18, vd, vn, vm);
}
void mla(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 18, vd, vn, vm);
}
void mla(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 18, vd, vn, vm);
}
void mla(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 18, vd, vn, vm);
}
void mla(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 18, vd, vn, vm);
}
void mul(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 19, vd, vn, vm);
}
void mul(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 19, vd, vn, vm);
}
void mul(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 19, vd, vn, vm);
}
void mul(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 19, vd, vn, vm);
}
void mul(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 19, vd, vn, vm);
}
void mul(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 19, vd, vn, vm);
}
void smaxp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 20, vd, vn, vm);
}
void smaxp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 20, vd, vn, vm);
}
void smaxp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 20, vd, vn, vm);
}
void smaxp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 20, vd, vn, vm);
}
void smaxp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 20, vd, vn, vm);
}
void smaxp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 20, vd, vn, vm);
}
void sminp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 21, vd, vn, vm);
}
void sminp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 21, vd, vn, vm);
}
void sminp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 21, vd, vn, vm);
}
void sminp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 21, vd, vn, vm);
}
void sminp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 21, vd, vn, vm);
}
void sminp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 21, vd, vn, vm);
}
void sqdmulh(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 22, vd, vn, vm);
}
void sqdmulh(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 22, vd, vn, vm);
}
void sqdmulh(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 22, vd, vn, vm);
}
void sqdmulh(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 22, vd, vn, vm);
}
void addp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 23, vd, vn, vm);
}
void addp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 23, vd, vn, vm);
}
void addp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 23, vd, vn, vm);
}
void addp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 23, vd, vn, vm);
}
void addp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 23, vd, vn, vm);
}
void addp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 23, vd, vn, vm);
}
void addp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(0, 23, vd, vn, vm);
}
void uhadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 0, vd, vn, vm);
}
void uhadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 0, vd, vn, vm);
}
void uhadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 0, vd, vn, vm);
}
void uhadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 0, vd, vn, vm);
}
void uhadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 0, vd, vn, vm);
}
void uhadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 0, vd, vn, vm);
}
void uqadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 1, vd, vn, vm);
}
void uqadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 1, vd, vn, vm);
}
void uqadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 1, vd, vn, vm);
}
void uqadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 1, vd, vn, vm);
}
void uqadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 1, vd, vn, vm);
}
void uqadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 1, vd, vn, vm);
}
void uqadd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 1, vd, vn, vm);
}
void urhadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 2, vd, vn, vm);
}
void urhadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 2, vd, vn, vm);
}
void urhadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 2, vd, vn, vm);
}
void urhadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 2, vd, vn, vm);
}
void urhadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 2, vd, vn, vm);
}
void urhadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 2, vd, vn, vm);
}
void uhsub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 4, vd, vn, vm);
}
void uhsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 4, vd, vn, vm);
}
void uhsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 4, vd, vn, vm);
}
void uhsub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 4, vd, vn, vm);
}
void uhsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 4, vd, vn, vm);
}
void uhsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 4, vd, vn, vm);
}
void uqsub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 5, vd, vn, vm);
}
void uqsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 5, vd, vn, vm);
}
void uqsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 5, vd, vn, vm);
}
void uqsub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 5, vd, vn, vm);
}
void uqsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 5, vd, vn, vm);
}
void uqsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 5, vd, vn, vm);
}
void uqsub(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 5, vd, vn, vm);
}
void cmhi(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 6, vd, vn, vm);
}
void cmhi(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 6, vd, vn, vm);
}
void cmhi(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 6, vd, vn, vm);
}
void cmhi(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 6, vd, vn, vm);
}
void cmhi(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 6, vd, vn, vm);
}
void cmhi(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 6, vd, vn, vm);
}
void cmhi(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 6, vd, vn, vm);
}
void cmhs(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 7, vd, vn, vm);
}
void cmhs(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 7, vd, vn, vm);
}
void cmhs(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 7, vd, vn, vm);
}
void cmhs(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 7, vd, vn, vm);
}
void cmhs(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 7, vd, vn, vm);
}
void cmhs(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 7, vd, vn, vm);
}
void cmhs(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 7, vd, vn, vm);
}
void ushl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 8, vd, vn, vm);
}
void ushl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 8, vd, vn, vm);
}
void ushl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 8, vd, vn, vm);
}
void ushl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 8, vd, vn, vm);
}
void ushl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 8, vd, vn, vm);
}
void ushl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 8, vd, vn, vm);
}
void ushl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 8, vd, vn, vm);
}
void uqshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 9, vd, vn, vm);
}
void uqshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 9, vd, vn, vm);
}
void uqshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 9, vd, vn, vm);
}
void uqshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 9, vd, vn, vm);
}
void uqshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 9, vd, vn, vm);
}
void uqshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 9, vd, vn, vm);
}
void uqshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 9, vd, vn, vm);
}
void urshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 10, vd, vn, vm);
}
void urshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 10, vd, vn, vm);
}
void urshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 10, vd, vn, vm);
}
void urshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 10, vd, vn, vm);
}
void urshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 10, vd, vn, vm);
}
void urshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 10, vd, vn, vm);
}
void urshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 10, vd, vn, vm);
}
void uqrshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 11, vd, vn, vm);
}
void uqrshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 11, vd, vn, vm);
}
void uqrshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 11, vd, vn, vm);
}
void uqrshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 11, vd, vn, vm);
}
void uqrshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 11, vd, vn, vm);
}
void uqrshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 11, vd, vn, vm);
}
void uqrshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 11, vd, vn, vm);
}
void umax(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 12, vd, vn, vm);
}
void umax(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 12, vd, vn, vm);
}
void umax(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 12, vd, vn, vm);
}
void umax(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 12, vd, vn, vm);
}
void umax(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 12, vd, vn, vm);
}
void umax(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 12, vd, vn, vm);
}
void umin(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 13, vd, vn, vm);
}
void umin(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 13, vd, vn, vm);
}
void umin(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 13, vd, vn, vm);
}
void umin(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 13, vd, vn, vm);
}
void umin(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 13, vd, vn, vm);
}
void umin(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 13, vd, vn, vm);
}
void uabd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 14, vd, vn, vm);
}
void uabd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 14, vd, vn, vm);
}
void uabd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 14, vd, vn, vm);
}
void uabd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 14, vd, vn, vm);
}
void uabd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 14, vd, vn, vm);
}
void uabd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 14, vd, vn, vm);
}
void uaba(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 15, vd, vn, vm);
}
void uaba(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 15, vd, vn, vm);
}
void uaba(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 15, vd, vn, vm);
}
void uaba(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 15, vd, vn, vm);
}
void uaba(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 15, vd, vn, vm);
}
void uaba(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 15, vd, vn, vm);
}
void sub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 16, vd, vn, vm);
}
void sub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 16, vd, vn, vm);
}
void sub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 16, vd, vn, vm);
}
void sub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 16, vd, vn, vm);
}
void sub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 16, vd, vn, vm);
}
void sub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 16, vd, vn, vm);
}
void sub(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 16, vd, vn, vm);
}
void cmeq(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 17, vd, vn, vm);
}
void cmeq(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 17, vd, vn, vm);
}
void cmeq(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 17, vd, vn, vm);
}
void cmeq(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 17, vd, vn, vm);
}
void cmeq(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 17, vd, vn, vm);
}
void cmeq(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 17, vd, vn, vm);
}
void cmeq(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 17, vd, vn, vm);
}
void mls(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 18, vd, vn, vm);
}
void mls(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 18, vd, vn, vm);
}
void mls(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 18, vd, vn, vm);
}
void mls(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 18, vd, vn, vm);
}
void mls(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 18, vd, vn, vm);
}
void mls(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 18, vd, vn, vm);
}
void pmul(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 19, vd, vn, vm);
}
void pmul(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 19, vd, vn, vm);
}
void umaxp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 20, vd, vn, vm);
}
void umaxp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 20, vd, vn, vm);
}
void umaxp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 20, vd, vn, vm);
}
void umaxp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 20, vd, vn, vm);
}
void umaxp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 20, vd, vn, vm);
}
void umaxp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 20, vd, vn, vm);
}
void uminp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 21, vd, vn, vm);
}
void uminp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 21, vd, vn, vm);
}
void uminp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 21, vd, vn, vm);
}
void uminp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 21, vd, vn, vm);
}
void uminp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 21, vd, vn, vm);
}
void uminp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 21, vd, vn, vm);
}
void sqrdmulh(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 22, vd, vn, vm);
}
void sqrdmulh(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 22, vd, vn, vm);
}
void sqrdmulh(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 22, vd, vn, vm);
}
void sqrdmulh(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3Same(1, 22, vd, vn, vm);
}
void fmaxnm(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 24, vd, vn, vm);
}
void fmaxnm(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 24, vd, vn, vm);
}
void fmaxnm(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 24, vd, vn, vm);
}
void fmla(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 25, vd, vn, vm);
}
void fmla(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 25, vd, vn, vm);
}
void fmla(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 25, vd, vn, vm);
}
void fadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 26, vd, vn, vm);
}
void fadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 26, vd, vn, vm);
}
void fadd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 26, vd, vn, vm);
}
void fmulx(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 27, vd, vn, vm);
}
void fmulx(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 27, vd, vn, vm);
}
void fmulx(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 27, vd, vn, vm);
}
void fcmeq(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 28, vd, vn, vm);
}
void fcmeq(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 28, vd, vn, vm);
}
void fcmeq(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 28, vd, vn, vm);
}
void fmax(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 30, vd, vn, vm);
}
void fmax(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 30, vd, vn, vm);
}
void fmax(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 30, vd, vn, vm);
}
void frecps(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 31, vd, vn, vm);
}
void frecps(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 31, vd, vn, vm);
}
void frecps(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(0, 31, vd, vn, vm);
}
void fmaxnmp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 24, vd, vn, vm);
}
void fmaxnmp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 24, vd, vn, vm);
}
void fmaxnmp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 24, vd, vn, vm);
}
void faddp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 26, vd, vn, vm);
}
void faddp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 26, vd, vn, vm);
}
void faddp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 26, vd, vn, vm);
}
void fmul(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 27, vd, vn, vm);
}
void fmul(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 27, vd, vn, vm);
}
void fmul(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 27, vd, vn, vm);
}
void fcmge(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 28, vd, vn, vm);
}
void fcmge(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 28, vd, vn, vm);
}
void fcmge(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 28, vd, vn, vm);
}
void facge(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 29, vd, vn, vm);
}
void facge(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 29, vd, vn, vm);
}
void facge(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 29, vd, vn, vm);
}
void fmaxp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 30, vd, vn, vm);
}
void fmaxp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 30, vd, vn, vm);
}
void fmaxp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 30, vd, vn, vm);
}
void fdiv(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 31, vd, vn, vm);
}
void fdiv(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 31, vd, vn, vm);
}
void fdiv(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz0x(1, 31, vd, vn, vm);
}
void fminnm(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 24, vd, vn, vm);
}
void fminnm(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 24, vd, vn, vm);
}
void fminnm(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 24, vd, vn, vm);
}
void fmls(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 25, vd, vn, vm);
}
void fmls(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 25, vd, vn, vm);
}
void fmls(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 25, vd, vn, vm);
}
void fsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 26, vd, vn, vm);
}
void fsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 26, vd, vn, vm);
}
void fsub(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 26, vd, vn, vm);
}
void fmin(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 30, vd, vn, vm);
}
void fmin(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 30, vd, vn, vm);
}
void fmin(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 30, vd, vn, vm);
}
void frsqrts(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 31, vd, vn, vm);
}
void frsqrts(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 31, vd, vn, vm);
}
void frsqrts(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(0, 31, vd, vn, vm);
}
void fminnmp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 24, vd, vn, vm);
}
void fminnmp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 24, vd, vn, vm);
}
void fminnmp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 24, vd, vn, vm);
}
void fabd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 26, vd, vn, vm);
}
void fabd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 26, vd, vn, vm);
}
void fabd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 26, vd, vn, vm);
}
void fcmgt(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 28, vd, vn, vm);
}
void fcmgt(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 28, vd, vn, vm);
}
void fcmgt(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 28, vd, vn, vm);
}
void facgt(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 29, vd, vn, vm);
}
void facgt(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 29, vd, vn, vm);
}
void facgt(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 29, vd, vn, vm);
}
void fminp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 30, vd, vn, vm);
}
void fminp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 30, vd, vn, vm);
}
void fminp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz1x(1, 30, vd, vn, vm);
}
void and_(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 0, 3, vd, vn, vm);
}
void and_(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 0, 3, vd, vn, vm);
}
void fmlal(const VReg2S &vd, const VReg2H &vn, const VReg2H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 0, 29, vd, vn, vm);
}
void fmlal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 0, 29, vd, vn, vm);
}
void bic(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 1, 3, vd, vn, vm);
}
void bic(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 1, 3, vd, vn, vm);
}
void orr(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 2, 3, vd, vn, vm);
}
void orr(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 2, 3, vd, vn, vm);
}
void mov(const VReg8B &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 2, 3, vd, vn, vn);
}
void mov(const VReg16B &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 2, 3, vd, vn, vn);
}
void fmlsl(const VReg2S &vd, const VReg2H &vn, const VReg2H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 2, 29, vd, vn, vm);
}
void fmlsl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 2, 29, vd, vn, vm);
}
void orn(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 3, 3, vd, vn, vm);
}
void orn(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(0, 3, 3, vd, vn, vm);
}
void eor(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 0, 3, vd, vn, vm);
}
void eor(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 0, 3, vd, vn, vm);
}
void fmlal2(const VReg2S &vd, const VReg2H &vn, const VReg2H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 0, 25, vd, vn, vm);
}
void fmlal2(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 0, 25, vd, vn, vm);
}
void bsl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 1, 3, vd, vn, vm);
}
void bsl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 1, 3, vd, vn, vm);
}
void bit(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 2, 3, vd, vn, vm);
}
void bit(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 2, 3, vd, vn, vm);
}
void fmlsl2(const VReg2S &vd, const VReg2H &vn, const VReg2H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 2, 25, vd, vn, vm);
}
void fmlsl2(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 2, 25, vd, vn, vm);
}
void bif(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 3, 3, vd, vn, vm);
}
void bif(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimd3SameSz(1, 3, 3, vd, vn, vm);
}
void movi(const VReg2S &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh);
}
void movi(const VReg4S &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh);
}
void movi(const VReg8B &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh);
}
void movi(const VReg16B &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh);
}
void movi(const VReg4H &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh);
}
void movi(const VReg8H &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh);
}
void movi(const DReg &vd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(1, 0, vd, imm);
}
void movi(const VReg2D &vd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(1, 0, vd, imm);
}
void mvni(const VReg2S &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(1, 0, vd, imm8, mod, sh);
}
void mvni(const VReg4S &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(1, 0, vd, imm8, mod, sh);
}
void mvni(const VReg4H &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(1, 0, vd, imm8, mod, sh);
}
void mvni(const VReg8H &vd, const uint32_t imm8, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmMoviMvni(1, 0, vd, imm8, mod, sh);
}
void orr(const VReg4H &vd, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmOrrBic(0, 0, vd, imm, mod, sh);
}
void orr(const VReg8H &vd, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmOrrBic(0, 0, vd, imm, mod, sh);
}
void orr(const VReg2S &vd, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmOrrBic(0, 0, vd, imm, mod, sh);
}
void orr(const VReg4S &vd, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmOrrBic(0, 0, vd, imm, mod, sh);
}
void bic(const VReg4H &vd, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmOrrBic(1, 0, vd, imm, mod, sh);
}
void bic(const VReg8H &vd, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmOrrBic(1, 0, vd, imm, mod, sh);
}
void bic(const VReg2S &vd, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmOrrBic(1, 0, vd, imm, mod, sh);
}
void bic(const VReg4S &vd, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmOrrBic(1, 0, vd, imm, mod, sh);
}
void fmov(const VReg2S &vd, const double imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmFmov(0, 0, vd, imm);
}
void fmov(const VReg4S &vd, const double imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmFmov(0, 0, vd, imm);
}
void fmov(const VReg4H &vd, const double imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmFmov(0, 1, vd, imm);
}
void fmov(const VReg8H &vd, const double imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmFmov(0, 1, vd, imm);
}
void fmov(const VReg2D &vd, const double imm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdModiImmFmov(1, 0, vd, imm);
}
void sshr(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 0, vd, vn, sh);
}
void sshr(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 0, vd, vn, sh);
}
void sshr(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 0, vd, vn, sh);
}
void sshr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 0, vd, vn, sh);
}
void sshr(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 0, vd, vn, sh);
}
void sshr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 0, vd, vn, sh);
}
void sshr(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 0, vd, vn, sh);
}
void ssra(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 2, vd, vn, sh);
}
void ssra(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 2, vd, vn, sh);
}
void ssra(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 2, vd, vn, sh);
}
void ssra(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 2, vd, vn, sh);
}
void ssra(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 2, vd, vn, sh);
}
void ssra(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 2, vd, vn, sh);
}
void ssra(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 2, vd, vn, sh);
}
void srshr(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 4, vd, vn, sh);
}
void srshr(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 4, vd, vn, sh);
}
void srshr(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 4, vd, vn, sh);
}
void srshr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 4, vd, vn, sh);
}
void srshr(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 4, vd, vn, sh);
}
void srshr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 4, vd, vn, sh);
}
void srshr(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 4, vd, vn, sh);
}
void srsra(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 6, vd, vn, sh);
}
void srsra(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 6, vd, vn, sh);
}
void srsra(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 6, vd, vn, sh);
}
void srsra(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 6, vd, vn, sh);
}
void srsra(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 6, vd, vn, sh);
}
void srsra(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 6, vd, vn, sh);
}
void srsra(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 6, vd, vn, sh);
}
void shl(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 10, vd, vn, sh);
}
void shl(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 10, vd, vn, sh);
}
void shl(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 10, vd, vn, sh);
}
void shl(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 10, vd, vn, sh);
}
void shl(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 10, vd, vn, sh);
}
void shl(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 10, vd, vn, sh);
}
void shl(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 10, vd, vn, sh);
}
void sqshl(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 14, vd, vn, sh);
}
void sqshl(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 14, vd, vn, sh);
}
void sqshl(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 14, vd, vn, sh);
}
void sqshl(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 14, vd, vn, sh);
}
void sqshl(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 14, vd, vn, sh);
}
void sqshl(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 14, vd, vn, sh);
}
void sqshl(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 14, vd, vn, sh);
}
void shrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 16, vd, vn, sh);
}
void shrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 16, vd, vn, sh);
}
void shrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 16, vd, vn, sh);
}
void shrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 16, vd, vn, sh);
}
void shrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 16, vd, vn, sh);
}
void shrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 16, vd, vn, sh);
}
void rshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 17, vd, vn, sh);
}
void rshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 17, vd, vn, sh);
}
void rshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 17, vd, vn, sh);
}
void rshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 17, vd, vn, sh);
}
void rshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 17, vd, vn, sh);
}
void rshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 17, vd, vn, sh);
}
void sqshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 18, vd, vn, sh);
}
void sqshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 18, vd, vn, sh);
}
void sqshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 18, vd, vn, sh);
}
void sqshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 18, vd, vn, sh);
}
void sqshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 18, vd, vn, sh);
}
void sqshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 18, vd, vn, sh);
}
void sqrshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 19, vd, vn, sh);
}
void sqrshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 19, vd, vn, sh);
}
void sqrshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 19, vd, vn, sh);
}
void sqrshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 19, vd, vn, sh);
}
void sqrshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 19, vd, vn, sh);
}
void sqrshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 19, vd, vn, sh);
}
void sshll(const VReg8H &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, sh);
}
void sshll(const VReg4S &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, sh);
}
void sshll(const VReg2D &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, sh);
}
void sshll2(const VReg8H &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, sh);
}
void sshll2(const VReg4S &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, sh);
}
void sshll2(const VReg2D &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, sh);
}
void sxtl(const VReg8H &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, 0);
}
void sxtl(const VReg4S &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, 0);
}
void sxtl(const VReg2D &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, 0);
}
void sxtl2(const VReg8H &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, 0);
}
void sxtl2(const VReg4S &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, 0);
}
void sxtl2(const VReg2D &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 20, vd, vn, 0);
}
void scvtf(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 28, vd, vn, sh);
}
void scvtf(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 28, vd, vn, sh);
}
void scvtf(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 28, vd, vn, sh);
}
void scvtf(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 28, vd, vn, sh);
}
void scvtf(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 28, vd, vn, sh);
}
void fcvtzs(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 31, vd, vn, sh);
}
void fcvtzs(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 31, vd, vn, sh);
}
void fcvtzs(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 31, vd, vn, sh);
}
void fcvtzs(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 31, vd, vn, sh);
}
void fcvtzs(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(0, 31, vd, vn, sh);
}
void ushr(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 0, vd, vn, sh);
}
void ushr(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 0, vd, vn, sh);
}
void ushr(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 0, vd, vn, sh);
}
void ushr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 0, vd, vn, sh);
}
void ushr(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 0, vd, vn, sh);
}
void ushr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 0, vd, vn, sh);
}
void ushr(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 0, vd, vn, sh);
}
void usra(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 2, vd, vn, sh);
}
void usra(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 2, vd, vn, sh);
}
void usra(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 2, vd, vn, sh);
}
void usra(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 2, vd, vn, sh);
}
void usra(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 2, vd, vn, sh);
}
void usra(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 2, vd, vn, sh);
}
void usra(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 2, vd, vn, sh);
}
void urshr(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 4, vd, vn, sh);
}
void urshr(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 4, vd, vn, sh);
}
void urshr(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 4, vd, vn, sh);
}
void urshr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 4, vd, vn, sh);
}
void urshr(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 4, vd, vn, sh);
}
void urshr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 4, vd, vn, sh);
}
void urshr(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 4, vd, vn, sh);
}
void ursra(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 6, vd, vn, sh);
}
void ursra(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 6, vd, vn, sh);
}
void ursra(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 6, vd, vn, sh);
}
void ursra(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 6, vd, vn, sh);
}
void ursra(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 6, vd, vn, sh);
}
void ursra(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 6, vd, vn, sh);
}
void ursra(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 6, vd, vn, sh);
}
void sri(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 8, vd, vn, sh);
}
void sri(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 8, vd, vn, sh);
}
void sri(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 8, vd, vn, sh);
}
void sri(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 8, vd, vn, sh);
}
void sri(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 8, vd, vn, sh);
}
void sri(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 8, vd, vn, sh);
}
void sri(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 8, vd, vn, sh);
}
void sli(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 10, vd, vn, sh);
}
void sli(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 10, vd, vn, sh);
}
void sli(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 10, vd, vn, sh);
}
void sli(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 10, vd, vn, sh);
}
void sli(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 10, vd, vn, sh);
}
void sli(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 10, vd, vn, sh);
}
void sli(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 10, vd, vn, sh);
}
void sqshlu(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 12, vd, vn, sh);
}
void sqshlu(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 12, vd, vn, sh);
}
void sqshlu(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 12, vd, vn, sh);
}
void sqshlu(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 12, vd, vn, sh);
}
void sqshlu(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 12, vd, vn, sh);
}
void sqshlu(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 12, vd, vn, sh);
}
void sqshlu(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 12, vd, vn, sh);
}
void uqshl(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 14, vd, vn, sh);
}
void uqshl(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 14, vd, vn, sh);
}
void uqshl(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 14, vd, vn, sh);
}
void uqshl(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 14, vd, vn, sh);
}
void uqshl(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 14, vd, vn, sh);
}
void uqshl(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 14, vd, vn, sh);
}
void uqshl(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 14, vd, vn, sh);
}
void sqshrun(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 16, vd, vn, sh);
}
void sqshrun(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 16, vd, vn, sh);
}
void sqshrun(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 16, vd, vn, sh);
}
void sqshrun2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 16, vd, vn, sh);
}
void sqshrun2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 16, vd, vn, sh);
}
void sqshrun2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 16, vd, vn, sh);
}
void sqrshrun(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 17, vd, vn, sh);
}
void sqrshrun(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 17, vd, vn, sh);
}
void sqrshrun(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 17, vd, vn, sh);
}
void sqrshrun2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 17, vd, vn, sh);
}
void sqrshrun2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 17, vd, vn, sh);
}
void sqrshrun2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 17, vd, vn, sh);
}
void uqshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 18, vd, vn, sh);
}
void uqshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 18, vd, vn, sh);
}
void uqshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 18, vd, vn, sh);
}
void uqshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 18, vd, vn, sh);
}
void uqshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 18, vd, vn, sh);
}
void uqshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 18, vd, vn, sh);
}
void uqrshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 19, vd, vn, sh);
}
void uqrshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 19, vd, vn, sh);
}
void uqrshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 19, vd, vn, sh);
}
void uqrshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 19, vd, vn, sh);
}
void uqrshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 19, vd, vn, sh);
}
void uqrshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 19, vd, vn, sh);
}
void ushll(const VReg8H &vd, const VReg8B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, sh);
}
void ushll(const VReg4S &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, sh);
}
void ushll(const VReg2D &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, sh);
}
void ushll2(const VReg8H &vd, const VReg16B &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, sh);
}
void ushll2(const VReg4S &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, sh);
}
void ushll2(const VReg2D &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, sh);
}
void uxtl(const VReg8H &vd, const VReg8B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, 0);
}
void uxtl(const VReg4S &vd, const VReg4H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, 0);
}
void uxtl(const VReg2D &vd, const VReg2S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, 0);
}
void uxtl2(const VReg8H &vd, const VReg16B &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, 0);
}
void uxtl2(const VReg4S &vd, const VReg8H &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, 0);
}
void uxtl2(const VReg2D &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 20, vd, vn, 0);
}
void ucvtf(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 28, vd, vn, sh);
}
void ucvtf(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 28, vd, vn, sh);
}
void ucvtf(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 28, vd, vn, sh);
}
void ucvtf(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 28, vd, vn, sh);
}
void ucvtf(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 28, vd, vn, sh);
}
void fcvtzu(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 31, vd, vn, sh);
}
void fcvtzu(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 31, vd, vn, sh);
}
void fcvtzu(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 31, vd, vn, sh);
}
void fcvtzu(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 31, vd, vn, sh);
}
void fcvtzu(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) {
  XBYAK_SET_CODE_INFO();
  AdvSimdShImm(1, 31, vd, vn, sh);
}
void smlal(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 2, vd, vn, vm);
}
void smlal(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 2, vd, vn, vm);
}
void smlal2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 2, vd, vn, vm);
}
void smlal2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 2, vd, vn, vm);
}
void sqdmlal(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 3, vd, vn, vm);
}
void sqdmlal(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 3, vd, vn, vm);
}
void sqdmlal2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 3, vd, vn, vm);
}
void sqdmlal2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 3, vd, vn, vm);
}
void smlsl(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 6, vd, vn, vm);
}
void smlsl(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 6, vd, vn, vm);
}
void smlsl2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 6, vd, vn, vm);
}
void smlsl2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 6, vd, vn, vm);
}
void sqdmlsl(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 7, vd, vn, vm);
}
void sqdmlsl(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 7, vd, vn, vm);
}
void sqdmlsl2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 7, vd, vn, vm);
}
void sqdmlsl2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 7, vd, vn, vm);
}
void mul(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 8, vd, vn, vm);
}
void mul(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 8, vd, vn, vm);
}
void mul(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 8, vd, vn, vm);
}
void mul(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 8, vd, vn, vm);
}
void smull(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 10, vd, vn, vm);
}
void smull(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 10, vd, vn, vm);
}
void smull2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 10, vd, vn, vm);
}
void smull2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 10, vd, vn, vm);
}
void sqdmull(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 11, vd, vn, vm);
}
void sqdmull(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 11, vd, vn, vm);
}
void sqdmull2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 11, vd, vn, vm);
}
void sqdmull2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 11, vd, vn, vm);
}
void sqdmulh(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 12, vd, vn, vm);
}
void sqdmulh(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 12, vd, vn, vm);
}
void sqdmulh(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 12, vd, vn, vm);
}
void sqdmulh(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 12, vd, vn, vm);
}
void sqrdmulh(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 13, vd, vn, vm);
}
void sqrdmulh(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 13, vd, vn, vm);
}
void sqrdmulh(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 13, vd, vn, vm);
}
void sqrdmulh(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 13, vd, vn, vm);
}
void sdot(const VReg2S &vd, const VReg8B &vn, const VRegBElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 14, vd, vn, vm);
}
void sdot(const VReg4S &vd, const VReg16B &vn, const VRegBElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(0, 14, vd, vn, vm);
}
void mla(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 0, vd, vn, vm);
}
void mla(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 0, vd, vn, vm);
}
void mla(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 0, vd, vn, vm);
}
void mla(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 0, vd, vn, vm);
}
void umlal(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 2, vd, vn, vm);
}
void umlal(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 2, vd, vn, vm);
}
void umlal2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 2, vd, vn, vm);
}
void umlal2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 2, vd, vn, vm);
}
void mls(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 4, vd, vn, vm);
}
void mls(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 4, vd, vn, vm);
}
void mls(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 4, vd, vn, vm);
}
void mls(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 4, vd, vn, vm);
}
void umlsl(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 6, vd, vn, vm);
}
void umlsl(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 6, vd, vn, vm);
}
void umlsl2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 6, vd, vn, vm);
}
void umlsl2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 6, vd, vn, vm);
}
void umull(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 10, vd, vn, vm);
}
void umull(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 10, vd, vn, vm);
}
void umull2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 10, vd, vn, vm);
}
void umull2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 10, vd, vn, vm);
}
void sqrdmlah(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 13, vd, vn, vm);
}
void sqrdmlah(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 13, vd, vn, vm);
}
void sqrdmlah(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 13, vd, vn, vm);
}
void sqrdmlah(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 13, vd, vn, vm);
}
void udot(const VReg2S &vd, const VReg8B &vn, const VRegBElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 14, vd, vn, vm);
}
void udot(const VReg4S &vd, const VReg16B &vn, const VRegBElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 14, vd, vn, vm);
}
void sqrdmlsh(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 15, vd, vn, vm);
}
void sqrdmlsh(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 15, vd, vn, vm);
}
void sqrdmlsh(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 15, vd, vn, vm);
}
void sqrdmlsh(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 15, vd, vn, vm);
}
void fcmla(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 1, vd, vn, vm, rotate);
}
void fcmla(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 1, vd, vn, vm, rotate);
}
void fcmla(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm,
           const uint32_t rotate) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElem(1, 1, vd, vn, vm, rotate);
}
void fmla(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 0, 1, vd, vn, vm);
}
void fmla(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 0, 1, vd, vn, vm);
}
void fmls(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 0, 5, vd, vn, vm);
}
void fmls(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 0, 5, vd, vn, vm);
}
void fmul(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 0, 9, vd, vn, vm);
}
void fmul(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 0, 9, vd, vn, vm);
}
void fmla(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 1, vd, vn, vm);
}
void fmla(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 1, vd, vn, vm);
}
void fmls(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 5, vd, vn, vm);
}
void fmls(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 5, vd, vn, vm);
}
void fmul(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 9, vd, vn, vm);
}
void fmul(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 9, vd, vn, vm);
}
void fmla(const VReg2D &vd, const VReg2D &vn, const VRegDElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 3, 1, vd, vn, vm);
}
void fmls(const VReg2D &vd, const VReg2D &vn, const VRegDElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 3, 5, vd, vn, vm);
}
void fmul(const VReg2D &vd, const VReg2D &vn, const VRegDElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 3, 9, vd, vn, vm);
}
void fmlal(const VReg2S &vd, const VReg2H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 0, vd, vn, vm);
}
void fmlal(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 0, vd, vn, vm);
}
void fmlsl(const VReg2S &vd, const VReg2H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 4, vd, vn, vm);
}
void fmlsl(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(0, 2, 4, vd, vn, vm);
}
void fmulx(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(1, 0, 9, vd, vn, vm);
}
void fmulx(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(1, 0, 9, vd, vn, vm);
}
void fmulx(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(1, 2, 9, vd, vn, vm);
}
void fmulx(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(1, 2, 9, vd, vn, vm);
}
void fmulx(const VReg2D &vd, const VReg2D &vn, const VRegDElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(1, 3, 9, vd, vn, vm);
}
void fmlal2(const VReg2S &vd, const VReg2H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(1, 2, 8, vd, vn, vm);
}
void fmlal2(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(1, 2, 8, vd, vn, vm);
}
void fmlsl2(const VReg2S &vd, const VReg2H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(1, 2, 12, vd, vn, vm);
}
void fmlsl2(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) {
  XBYAK_SET_CODE_INFO();
  AdvSimdVecXindElemSz(1, 2, 12, vd, vn, vm);
}
void sm3tt1a(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegImm2(0, vd, vn, vm);
}
void sm3tt1b(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegImm2(1, vd, vn, vm);
}
void sm3tt2a(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegImm2(2, vd, vn, vm);
}
void sm3tt2b(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegImm2(3, vd, vn, vm);
}
void sha512h(const QReg &vd, const QReg &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegSHA512(0, 0, vd, vn, vm);
}
void sha512h2(const QReg &vd, const QReg &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegSHA512(0, 1, vd, vn, vm);
}
void sha512su1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegSHA512(0, 2, vd, vn, vm);
}
void rax1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegSHA512(0, 3, vd, vn, vm);
}
void sm3partw1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegSHA512(1, 0, vd, vn, vm);
}
void sm3partw2(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegSHA512(1, 1, vd, vn, vm);
}
void sm4ekey(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) {
  XBYAK_SET_CODE_INFO();
  Crypto3RegSHA512(1, 2, vd, vn, vm);
}
void xar(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm,
         const uint32_t imm6) {
  XBYAK_SET_CODE_INFO();
  CryptoSHA(vd, vn, vm, imm6);
}
void eor3(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm,
          const VReg16B &va) {
  XBYAK_SET_CODE_INFO();
  Crypto4Reg(0, vd, vn, vm, va);
}
void bcax(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm,
          const VReg16B &va) {
  XBYAK_SET_CODE_INFO();
  Crypto4Reg(1, vd, vn, vm, va);
}
void sm3ss1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm,
            const VReg4S &va) {
  XBYAK_SET_CODE_INFO();
  Crypto4Reg(2, vd, vn, vm, va);
}
void sha512su0(const VReg2D &vd, const VReg2D &vn) {
  XBYAK_SET_CODE_INFO();
  Crypto2RegSHA512(0, vd, vn);
}
void sm4e(const VReg4S &vd, const VReg4S &vn) {
  XBYAK_SET_CODE_INFO();
  Crypto2RegSHA512(1, vd, vn);
}
void scvtf(const SReg &d, const WReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 0, 0, 2, d, n, fbits);
}
void scvtf(const SReg &d, const XReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 0, 0, 2, d, n, fbits);
}
void ucvtf(const SReg &d, const WReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 0, 0, 3, d, n, fbits);
}
void ucvtf(const SReg &d, const XReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 0, 0, 3, d, n, fbits);
}
void fcvtzs(const WReg &d, const SReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 0, 3, 0, d, n, fbits);
}
void fcvtzs(const XReg &d, const SReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 0, 3, 0, d, n, fbits);
}
void fcvtzu(const WReg &d, const SReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 0, 3, 1, d, n, fbits);
}
void fcvtzu(const XReg &d, const SReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 0, 3, 1, d, n, fbits);
}
void scvtf(const DReg &d, const WReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 1, 0, 2, d, n, fbits);
}
void scvtf(const DReg &d, const XReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 1, 0, 2, d, n, fbits);
}
void ucvtf(const DReg &d, const WReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 1, 0, 3, d, n, fbits);
}
void ucvtf(const DReg &d, const XReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 1, 0, 3, d, n, fbits);
}
void fcvtzs(const WReg &d, const DReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 1, 3, 0, d, n, fbits);
}
void fcvtzs(const XReg &d, const DReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 1, 3, 0, d, n, fbits);
}
void fcvtzu(const WReg &d, const DReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 1, 3, 1, d, n, fbits);
}
void fcvtzu(const XReg &d, const DReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 1, 3, 1, d, n, fbits);
}
void scvtf(const HReg &d, const WReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 3, 0, 2, d, n, fbits);
}
void scvtf(const HReg &d, const XReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 3, 0, 2, d, n, fbits);
}
void ucvtf(const HReg &d, const WReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 3, 0, 3, d, n, fbits);
}
void ucvtf(const HReg &d, const XReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 3, 0, 3, d, n, fbits);
}
void fcvtzs(const WReg &d, const HReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 3, 3, 0, d, n, fbits);
}
void fcvtzs(const XReg &d, const HReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 3, 3, 0, d, n, fbits);
}
void fcvtzu(const WReg &d, const HReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 3, 3, 1, d, n, fbits);
}
void fcvtzu(const XReg &d, const HReg &n, const uint32_t fbits) {
  XBYAK_SET_CODE_INFO();
  ConversionFpFix(0, 3, 3, 1, d, n, fbits);
}
void fcvtns(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 0, 0, d, n);
}
void fcvtnu(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 0, 1, d, n);
}
void fcvtas(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 0, 4, d, n);
}
void fcvtau(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 0, 5, d, n);
}
void fmov(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 0, 6, d, n);
}
void fcvtps(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 1, 0, d, n);
}
void fcvtpu(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 1, 1, d, n);
}
void fcvtms(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 2, 0, d, n);
}
void fcvtmu(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 2, 1, d, n);
}
void fcvtzs(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 3, 0, d, n);
}
void fcvtzu(const WReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 3, 1, d, n);
}
void scvtf(const SReg &d, const WReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 0, 2, d, n);
}
void ucvtf(const SReg &d, const WReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 0, 3, d, n);
}
void fmov(const SReg &d, const WReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 0, 0, 7, d, n);
}
void fcvtns(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 0, 0, d, n);
}
void fcvtnu(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 0, 1, d, n);
}
void fcvtas(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 0, 4, d, n);
}
void fcvtau(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 0, 5, d, n);
}
void fcvtps(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 1, 0, d, n);
}
void fcvtpu(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 1, 1, d, n);
}
void fcvtms(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 2, 0, d, n);
}
void fcvtmu(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 2, 1, d, n);
}
void fcvtzs(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 3, 0, d, n);
}
void fcvtzu(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 3, 1, d, n);
}
void fjcvtzs(const WReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 3, 6, d, n);
}
void scvtf(const DReg &d, const WReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 0, 2, d, n);
}
void ucvtf(const DReg &d, const WReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 1, 0, 3, d, n);
}
void fcvtns(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 0, 0, d, n);
}
void fcvtnu(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 0, 1, d, n);
}
void fcvtas(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 0, 4, d, n);
}
void fcvtau(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 0, 5, d, n);
}
void fmov(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 0, 6, d, n);
}
void fcvtps(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 1, 0, d, n);
}
void fcvtpu(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 1, 1, d, n);
}
void fcvtms(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 2, 0, d, n);
}
void fcvtmu(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 2, 1, d, n);
}
void fcvtzs(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 3, 0, d, n);
}
void fcvtzu(const WReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 3, 1, d, n);
}
void scvtf(const HReg &d, const WReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 0, 2, d, n);
}
void ucvtf(const HReg &d, const WReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 0, 3, d, n);
}
void fmov(const HReg &d, const WReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(0, 0, 3, 0, 7, d, n);
}
void fcvtns(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 0, 0, d, n);
}
void fcvtnu(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 0, 1, d, n);
}
void fcvtas(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 0, 4, d, n);
}
void fcvtau(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 0, 5, d, n);
}
void fcvtps(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 1, 0, d, n);
}
void fcvtpu(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 1, 1, d, n);
}
void fcvtms(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 2, 0, d, n);
}
void fcvtmu(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 2, 1, d, n);
}
void fcvtzs(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 3, 0, d, n);
}
void fcvtzu(const XReg &d, const SReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 3, 1, d, n);
}
void scvtf(const SReg &d, const XReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 0, 2, d, n);
}
void ucvtf(const SReg &d, const XReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 0, 0, 3, d, n);
}
void fcvtns(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 0, 0, d, n);
}
void fcvtnu(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 0, 1, d, n);
}
void fcvtas(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 0, 4, d, n);
}
void fcvtau(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 0, 5, d, n);
}
void fmov(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 0, 6, d, n);
}
void fcvtps(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 1, 0, d, n);
}
void fcvtpu(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 1, 1, d, n);
}
void fcvtms(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 2, 0, d, n);
}
void fcvtmu(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 2, 1, d, n);
}
void fcvtzs(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 3, 0, d, n);
}
void fcvtzu(const XReg &d, const DReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 3, 1, d, n);
}
void scvtf(const DReg &d, const XReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 0, 2, d, n);
}
void ucvtf(const DReg &d, const XReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 0, 3, d, n);
}
void fmov(const DReg &d, const XReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 1, 0, 7, d, n);
}
void fmov(const XReg &d, const VRegDElem &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 2, 1, 6, d, n);
}
void fmov(const VRegDElem &d, const XReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 2, 1, 7, d, n);
}
void fcvtns(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 0, 0, d, n);
}
void fcvtnu(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 0, 1, d, n);
}
void fcvtas(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 0, 4, d, n);
}
void fcvtau(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 0, 5, d, n);
}
void fmov(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 0, 6, d, n);
}
void fcvtps(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 1, 0, d, n);
}
void fcvtpu(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 1, 1, d, n);
}
void fcvtms(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 2, 0, d, n);
}
void fcvtmu(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 2, 1, d, n);
}
void fcvtzs(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 3, 0, d, n);
}
void fcvtzu(const XReg &d, const HReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 3, 1, d, n);
}
void scvtf(const HReg &d, const XReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 0, 2, d, n);
}
void ucvtf(const HReg &d, const XReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 0, 3, d, n);
}
void fmov(const HReg &d, const XReg &n) {
  XBYAK_SET_CODE_INFO();
  ConversionFpInt(1, 0, 3, 0, 7, d, n);
}
void fmov(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 0, vd, vn);
}
void fabs(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 1, vd, vn);
}
void fneg(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 2, vd, vn);
}
void fsqrt(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 3, vd, vn);
}
void frintn(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 8, vd, vn);
}
void frintp(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 9, vd, vn);
}
void frintm(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 10, vd, vn);
}
void frintz(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 11, vd, vn);
}
void frinta(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 12, vd, vn);
}
void frintx(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 14, vd, vn);
}
void frinti(const SReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 15, vd, vn);
}
void fcvt(const DReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 5, vd, vn);
}
void fcvt(const HReg &vd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 0, 7, vd, vn);
}
void fmov(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 0, vd, vn);
}
void fabs(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 1, vd, vn);
}
void fneg(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 2, vd, vn);
}
void fsqrt(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 3, vd, vn);
}
void frintn(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 8, vd, vn);
}
void frintp(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 9, vd, vn);
}
void frintm(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 10, vd, vn);
}
void frintz(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 11, vd, vn);
}
void frinta(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 12, vd, vn);
}
void frintx(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 14, vd, vn);
}
void frinti(const DReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 15, vd, vn);
}
void fcvt(const SReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 4, vd, vn);
}
void fcvt(const HReg &vd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 1, 7, vd, vn);
}
void fmov(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 0, vd, vn);
}
void fabs(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 1, vd, vn);
}
void fneg(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 2, vd, vn);
}
void fsqrt(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 3, vd, vn);
}
void frintn(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 8, vd, vn);
}
void frintp(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 9, vd, vn);
}
void frintm(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 10, vd, vn);
}
void frintz(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 11, vd, vn);
}
void frinta(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 12, vd, vn);
}
void frintx(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 14, vd, vn);
}
void frinti(const HReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 15, vd, vn);
}
void fcvt(const SReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 4, vd, vn);
}
void fcvt(const DReg &vd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  FpDataProc1Reg(0, 0, 3, 5, vd, vn);
}
void fcmp(const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 0, 0, 0, vn, vm);
}
void fcmpe(const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 0, 0, 16, vn, vm);
}
void fcmp(const SReg &vn, const double imm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 0, 0, 8, vn, imm);
}
void fcmpe(const SReg &vn, const double imm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 0, 0, 24, vn, imm);
}
void fcmp(const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 1, 0, 0, vn, vm);
}
void fcmpe(const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 1, 0, 16, vn, vm);
}
void fcmp(const DReg &vn, const double imm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 1, 0, 8, vn, imm);
}
void fcmpe(const DReg &vn, const double imm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 1, 0, 24, vn, imm);
}
void fcmp(const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 3, 0, 0, vn, vm);
}
void fcmpe(const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 3, 0, 16, vn, vm);
}
void fcmp(const HReg &vn, const double imm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 3, 0, 8, vn, imm);
}
void fcmpe(const HReg &vn, const double imm) {
  XBYAK_SET_CODE_INFO();
  FpComp(0, 0, 3, 0, 24, vn, imm);
}
void fmov(const SReg &vd, const double imm) {
  XBYAK_SET_CODE_INFO();
  FpImm(0, 0, 0, vd, imm);
}
void fmov(const DReg &vd, const double imm) {
  XBYAK_SET_CODE_INFO();
  FpImm(0, 0, 1, vd, imm);
}
void fmov(const HReg &vd, const double imm) {
  XBYAK_SET_CODE_INFO();
  FpImm(0, 0, 3, vd, imm);
}
void fccmp(const SReg &vn, const SReg &vm, const uint32_t nzcv,
           const Cond cond) {
  XBYAK_SET_CODE_INFO();
  FpCondComp(0, 0, 0, 0, vn, vm, nzcv, cond);
}
void fccmpe(const SReg &vn, const SReg &vm, const uint32_t nzcv,
            const Cond cond) {
  XBYAK_SET_CODE_INFO();
  FpCondComp(0, 0, 0, 1, vn, vm, nzcv, cond);
}
void fccmp(const DReg &vn, const DReg &vm, const uint32_t nzcv,
           const Cond cond) {
  XBYAK_SET_CODE_INFO();
  FpCondComp(0, 0, 1, 0, vn, vm, nzcv, cond);
}
void fccmpe(const DReg &vn, const DReg &vm, const uint32_t nzcv,
            const Cond cond) {
  XBYAK_SET_CODE_INFO();
  FpCondComp(0, 0, 1, 1, vn, vm, nzcv, cond);
}
void fccmp(const HReg &vn, const HReg &vm, const uint32_t nzcv,
           const Cond cond) {
  XBYAK_SET_CODE_INFO();
  FpCondComp(0, 0, 3, 0, vn, vm, nzcv, cond);
}
void fccmpe(const HReg &vn, const HReg &vm, const uint32_t nzcv,
            const Cond cond) {
  XBYAK_SET_CODE_INFO();
  FpCondComp(0, 0, 3, 1, vn, vm, nzcv, cond);
}
void fmul(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 0, 0, vd, vn, vm);
}
void fdiv(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 0, 1, vd, vn, vm);
}
void fadd(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 0, 2, vd, vn, vm);
}
void fsub(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 0, 3, vd, vn, vm);
}
void fmax(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 0, 4, vd, vn, vm);
}
void fmin(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 0, 5, vd, vn, vm);
}
void fmaxnm(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 0, 6, vd, vn, vm);
}
void fminnm(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 0, 7, vd, vn, vm);
}
void fnmul(const SReg &vd, const SReg &vn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 0, 8, vd, vn, vm);
}
void fmul(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 1, 0, vd, vn, vm);
}
void fdiv(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 1, 1, vd, vn, vm);
}
void fadd(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 1, 2, vd, vn, vm);
}
void fsub(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 1, 3, vd, vn, vm);
}
void fmax(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 1, 4, vd, vn, vm);
}
void fmin(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 1, 5, vd, vn, vm);
}
void fmaxnm(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 1, 6, vd, vn, vm);
}
void fminnm(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 1, 7, vd, vn, vm);
}
void fnmul(const DReg &vd, const DReg &vn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 1, 8, vd, vn, vm);
}
void fmul(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 3, 0, vd, vn, vm);
}
void fdiv(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 3, 1, vd, vn, vm);
}
void fadd(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 3, 2, vd, vn, vm);
}
void fsub(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 3, 3, vd, vn, vm);
}
void fmax(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 3, 4, vd, vn, vm);
}
void fmin(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 3, 5, vd, vn, vm);
}
void fmaxnm(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 3, 6, vd, vn, vm);
}
void fminnm(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 3, 7, vd, vn, vm);
}
void fnmul(const HReg &vd, const HReg &vn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  FpDataProc2Reg(0, 0, 3, 8, vd, vn, vm);
}
void fcsel(const SReg &vd, const SReg &vn, const SReg &vm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  FpCondSel(0, 0, 0, vd, vn, vm, cond);
}
void fcsel(const DReg &vd, const DReg &vn, const DReg &vm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  FpCondSel(0, 0, 1, vd, vn, vm, cond);
}
void fcsel(const HReg &vd, const HReg &vn, const HReg &vm, const Cond cond) {
  XBYAK_SET_CODE_INFO();
  FpCondSel(0, 0, 3, vd, vn, vm, cond);
}
void fmadd(const SReg &vd, const SReg &vn, const SReg &vm, const SReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 0, 0, 0, vd, vn, vm, va);
}
void fmsub(const SReg &vd, const SReg &vn, const SReg &vm, const SReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 0, 0, 1, vd, vn, vm, va);
}
void fnmadd(const SReg &vd, const SReg &vn, const SReg &vm, const SReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 0, 1, 0, vd, vn, vm, va);
}
void fnmsub(const SReg &vd, const SReg &vn, const SReg &vm, const SReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 0, 1, 1, vd, vn, vm, va);
}
void fmadd(const DReg &vd, const DReg &vn, const DReg &vm, const DReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 1, 0, 0, vd, vn, vm, va);
}
void fmsub(const DReg &vd, const DReg &vn, const DReg &vm, const DReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 1, 0, 1, vd, vn, vm, va);
}
void fnmadd(const DReg &vd, const DReg &vn, const DReg &vm, const DReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 1, 1, 0, vd, vn, vm, va);
}
void fnmsub(const DReg &vd, const DReg &vn, const DReg &vm, const DReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 1, 1, 1, vd, vn, vm, va);
}
void fmadd(const HReg &vd, const HReg &vn, const HReg &vm, const HReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 3, 0, 0, vd, vn, vm, va);
}
void fmsub(const HReg &vd, const HReg &vn, const HReg &vm, const HReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 3, 0, 1, vd, vn, vm, va);
}
void fnmadd(const HReg &vd, const HReg &vn, const HReg &vm, const HReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 3, 1, 0, vd, vn, vm, va);
}
void fnmsub(const HReg &vd, const HReg &vn, const HReg &vm, const HReg &va) {
  XBYAK_SET_CODE_INFO();
  FpDataProc3Reg(0, 0, 3, 1, 1, vd, vn, vm, va);
}
void orr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(0, zdn, pg, zm);
}
void orr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(0, zdn, pg, zm);
}
void orr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(0, zdn, pg, zm);
}
void orr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(0, zdn, pg, zm);
}
void eor(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(1, zdn, pg, zm);
}
void eor(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(1, zdn, pg, zm);
}
void eor(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(1, zdn, pg, zm);
}
void eor(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(1, zdn, pg, zm);
}
void and_(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(2, zdn, pg, zm);
}
void and_(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(2, zdn, pg, zm);
}
void and_(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(2, zdn, pg, zm);
}
void and_(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(2, zdn, pg, zm);
}
void bic(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(3, zdn, pg, zm);
}
void bic(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(3, zdn, pg, zm);
}
void bic(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(3, zdn, pg, zm);
}
void bic(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpPred(3, zdn, pg, zm);
}
void add(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(0, zdn, pg, zm);
}
void add(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(0, zdn, pg, zm);
}
void add(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(0, zdn, pg, zm);
}
void add(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(0, zdn, pg, zm);
}
void sub(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(1, zdn, pg, zm);
}
void sub(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(1, zdn, pg, zm);
}
void sub(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(1, zdn, pg, zm);
}
void sub(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(1, zdn, pg, zm);
}
void subr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(3, zdn, pg, zm);
}
void subr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(3, zdn, pg, zm);
}
void subr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(3, zdn, pg, zm);
}
void subr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubVecPred(3, zdn, pg, zm);
}
void smax(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(0, 0, zdn, pg, zm);
}
void smax(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(0, 0, zdn, pg, zm);
}
void smax(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(0, 0, zdn, pg, zm);
}
void smax(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(0, 0, zdn, pg, zm);
}
void umax(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(0, 1, zdn, pg, zm);
}
void umax(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(0, 1, zdn, pg, zm);
}
void umax(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(0, 1, zdn, pg, zm);
}
void umax(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(0, 1, zdn, pg, zm);
}
void smin(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(1, 0, zdn, pg, zm);
}
void smin(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(1, 0, zdn, pg, zm);
}
void smin(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(1, 0, zdn, pg, zm);
}
void smin(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(1, 0, zdn, pg, zm);
}
void umin(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(1, 1, zdn, pg, zm);
}
void umin(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(1, 1, zdn, pg, zm);
}
void umin(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(1, 1, zdn, pg, zm);
}
void umin(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(1, 1, zdn, pg, zm);
}
void sabd(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(2, 0, zdn, pg, zm);
}
void sabd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(2, 0, zdn, pg, zm);
}
void sabd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(2, 0, zdn, pg, zm);
}
void sabd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(2, 0, zdn, pg, zm);
}
void uabd(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(2, 1, zdn, pg, zm);
}
void uabd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(2, 1, zdn, pg, zm);
}
void uabd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(2, 1, zdn, pg, zm);
}
void uabd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxDiffPred(2, 1, zdn, pg, zm);
}
void mul(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(0, 0, zdn, pg, zm);
}
void mul(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(0, 0, zdn, pg, zm);
}
void mul(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(0, 0, zdn, pg, zm);
}
void mul(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(0, 0, zdn, pg, zm);
}
void smulh(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(1, 0, zdn, pg, zm);
}
void smulh(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(1, 0, zdn, pg, zm);
}
void smulh(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(1, 0, zdn, pg, zm);
}
void smulh(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(1, 0, zdn, pg, zm);
}
void umulh(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(1, 1, zdn, pg, zm);
}
void umulh(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(1, 1, zdn, pg, zm);
}
void umulh(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(1, 1, zdn, pg, zm);
}
void umulh(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(1, 1, zdn, pg, zm);
}
void sdiv(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(2, 0, zdn, pg, zm);
}
void sdiv(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(2, 0, zdn, pg, zm);
}
void udiv(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(2, 1, zdn, pg, zm);
}
void udiv(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(2, 1, zdn, pg, zm);
}
void sdivr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(3, 0, zdn, pg, zm);
}
void sdivr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(3, 0, zdn, pg, zm);
}
void udivr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(3, 1, zdn, pg, zm);
}
void udivr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultDivVecPred(3, 1, zdn, pg, zm);
}
void orv(const BReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(0, vd, pg, zn);
}
void orv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(0, vd, pg, zn);
}
void orv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(0, vd, pg, zn);
}
void orv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(0, vd, pg, zn);
}
void eorv(const BReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(1, vd, pg, zn);
}
void eorv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(1, vd, pg, zn);
}
void eorv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(1, vd, pg, zn);
}
void eorv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(1, vd, pg, zn);
}
void andv(const BReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(2, vd, pg, zn);
}
void andv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(2, vd, pg, zn);
}
void andv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(2, vd, pg, zn);
}
void andv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLReductPred(2, vd, pg, zn);
}
void movprfx(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveConstPrefPred(0, zd, pg, zn);
}
void movprfx(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveConstPrefPred(0, zd, pg, zn);
}
void movprfx(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveConstPrefPred(0, zd, pg, zn);
}
void movprfx(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveConstPrefPred(0, zd, pg, zn);
}
void saddv(const DReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntAddReductPred(0, 0, vd, pg, zn);
}
void saddv(const DReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntAddReductPred(0, 0, vd, pg, zn);
}
void saddv(const DReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntAddReductPred(0, 0, vd, pg, zn);
}
void uaddv(const DReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntAddReductPred(0, 1, vd, pg, zn);
}
void uaddv(const DReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntAddReductPred(0, 1, vd, pg, zn);
}
void uaddv(const DReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntAddReductPred(0, 1, vd, pg, zn);
}
void uaddv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntAddReductPred(0, 1, vd, pg, zn);
}
void smaxv(const BReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(0, 0, vd, pg, zn);
}
void smaxv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(0, 0, vd, pg, zn);
}
void smaxv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(0, 0, vd, pg, zn);
}
void smaxv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(0, 0, vd, pg, zn);
}
void umaxv(const BReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(0, 1, vd, pg, zn);
}
void umaxv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(0, 1, vd, pg, zn);
}
void umaxv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(0, 1, vd, pg, zn);
}
void umaxv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(0, 1, vd, pg, zn);
}
void sminv(const BReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(1, 0, vd, pg, zn);
}
void sminv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(1, 0, vd, pg, zn);
}
void sminv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(1, 0, vd, pg, zn);
}
void sminv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(1, 0, vd, pg, zn);
}
void uminv(const BReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(1, 1, vd, pg, zn);
}
void uminv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(1, 1, vd, pg, zn);
}
void uminv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(1, 1, vd, pg, zn);
}
void uminv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxReductPred(1, 1, vd, pg, zn);
}
void asr(const ZRegB &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(0, zdn, pg, amount);
}
void asr(const ZRegH &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(0, zdn, pg, amount);
}
void asr(const ZRegS &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(0, zdn, pg, amount);
}
void asr(const ZRegD &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(0, zdn, pg, amount);
}
void lsr(const ZRegB &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(1, zdn, pg, amount);
}
void lsr(const ZRegH &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(1, zdn, pg, amount);
}
void lsr(const ZRegS &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(1, zdn, pg, amount);
}
void lsr(const ZRegD &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(1, zdn, pg, amount);
}
void lsl(const ZRegB &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(3, zdn, pg, amount);
}
void lsl(const ZRegH &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(3, zdn, pg, amount);
}
void lsl(const ZRegS &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(3, zdn, pg, amount);
}
void lsl(const ZRegD &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(3, zdn, pg, amount);
}
void asrd(const ZRegB &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(4, zdn, pg, amount);
}
void asrd(const ZRegH &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(4, zdn, pg, amount);
}
void asrd(const ZRegS &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(4, zdn, pg, amount);
}
void asrd(const ZRegD &zdn, const _PReg &pg, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmPred(4, zdn, pg, amount);
}
void asr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(0, zdn, pg, zm);
}
void asr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(0, zdn, pg, zm);
}
void asr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(0, zdn, pg, zm);
}
void asr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(0, zdn, pg, zm);
}
void lsr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(1, zdn, pg, zm);
}
void lsr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(1, zdn, pg, zm);
}
void lsr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(1, zdn, pg, zm);
}
void lsr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(1, zdn, pg, zm);
}
void lsl(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(3, zdn, pg, zm);
}
void lsl(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(3, zdn, pg, zm);
}
void lsl(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(3, zdn, pg, zm);
}
void lsl(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(3, zdn, pg, zm);
}
void asrr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(4, zdn, pg, zm);
}
void asrr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(4, zdn, pg, zm);
}
void asrr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(4, zdn, pg, zm);
}
void asrr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(4, zdn, pg, zm);
}
void lsrr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(5, zdn, pg, zm);
}
void lsrr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(5, zdn, pg, zm);
}
void lsrr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(5, zdn, pg, zm);
}
void lsrr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(5, zdn, pg, zm);
}
void lslr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(7, zdn, pg, zm);
}
void lslr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(7, zdn, pg, zm);
}
void lslr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(7, zdn, pg, zm);
}
void lslr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShVecPred(7, zdn, pg, zm);
}
void asr(const ZRegB &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShWElemPred(0, zdn, pg, zm);
}
void asr(const ZRegH &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShWElemPred(0, zdn, pg, zm);
}
void asr(const ZRegS &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShWElemPred(0, zdn, pg, zm);
}
void lsr(const ZRegB &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShWElemPred(1, zdn, pg, zm);
}
void lsr(const ZRegH &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShWElemPred(1, zdn, pg, zm);
}
void lsr(const ZRegS &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShWElemPred(1, zdn, pg, zm);
}
void lsl(const ZRegB &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShWElemPred(3, zdn, pg, zm);
}
void lsl(const ZRegH &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShWElemPred(3, zdn, pg, zm);
}
void lsl(const ZRegS &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShWElemPred(3, zdn, pg, zm);
}
void cls(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(0, zd, pg, zn);
}
void cls(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(0, zd, pg, zn);
}
void cls(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(0, zd, pg, zn);
}
void cls(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(0, zd, pg, zn);
}
void clz(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(1, zd, pg, zn);
}
void clz(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(1, zd, pg, zn);
}
void clz(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(1, zd, pg, zn);
}
void clz(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(1, zd, pg, zn);
}
void cnt(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(2, zd, pg, zn);
}
void cnt(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(2, zd, pg, zn);
}
void cnt(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(2, zd, pg, zn);
}
void cnt(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(2, zd, pg, zn);
}
void cnot(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(3, zd, pg, zn);
}
void cnot(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(3, zd, pg, zn);
}
void cnot(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(3, zd, pg, zn);
}
void cnot(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(3, zd, pg, zn);
}
void fabs(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(4, zd, pg, zn);
}
void fabs(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(4, zd, pg, zn);
}
void fabs(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(4, zd, pg, zn);
}
void fneg(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(5, zd, pg, zn);
}
void fneg(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(5, zd, pg, zn);
}
void fneg(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(5, zd, pg, zn);
}
void not_(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(6, zd, pg, zn);
}
void not_(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(6, zd, pg, zn);
}
void not_(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(6, zd, pg, zn);
}
void not_(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseUnaryOpPred(6, zd, pg, zn);
}
void sxtb(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(0, zd, pg, zn);
}
void sxtb(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(0, zd, pg, zn);
}
void sxtb(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(0, zd, pg, zn);
}
void uxtb(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(1, zd, pg, zn);
}
void uxtb(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(1, zd, pg, zn);
}
void uxtb(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(1, zd, pg, zn);
}
void sxth(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(2, zd, pg, zn);
}
void sxth(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(2, zd, pg, zn);
}
void uxth(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(3, zd, pg, zn);
}
void uxth(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(3, zd, pg, zn);
}
void sxtw(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(4, zd, pg, zn);
}
void uxtw(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(5, zd, pg, zn);
}
void abs(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(6, zd, pg, zn);
}
void abs(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(6, zd, pg, zn);
}
void abs(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(6, zd, pg, zn);
}
void abs(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(6, zd, pg, zn);
}
void neg(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(7, zd, pg, zn);
}
void neg(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(7, zd, pg, zn);
}
void neg(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(7, zd, pg, zn);
}
void neg(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntUnaryOpPred(7, zd, pg, zn);
}
void mla(const ZRegB &zda, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAccumPred(0, zda, pg, zn, zm);
}
void mla(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAccumPred(0, zda, pg, zn, zm);
}
void mla(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAccumPred(0, zda, pg, zn, zm);
}
void mla(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAccumPred(0, zda, pg, zn, zm);
}
void mls(const ZRegB &zda, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAccumPred(1, zda, pg, zn, zm);
}
void mls(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAccumPred(1, zda, pg, zn, zm);
}
void mls(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAccumPred(1, zda, pg, zn, zm);
}
void mls(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAccumPred(1, zda, pg, zn, zm);
}
void mad(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm, const ZRegB &za) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAddPred(0, zdn, pg, zm, za);
}
void mad(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAddPred(0, zdn, pg, zm, za);
}
void mad(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAddPred(0, zdn, pg, zm, za);
}
void mad(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAddPred(0, zdn, pg, zm, za);
}
void msb(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm, const ZRegB &za) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAddPred(1, zdn, pg, zm, za);
}
void msb(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAddPred(1, zdn, pg, zm, za);
}
void msb(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAddPred(1, zdn, pg, zm, za);
}
void msb(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) {
  XBYAK_SET_CODE_INFO();
  SveIntMultAddPred(1, zdn, pg, zm, za);
}
void add(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(0, zd, zn, zm);
}
void add(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(0, zd, zn, zm);
}
void add(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(0, zd, zn, zm);
}
void add(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(0, zd, zn, zm);
}
void sub(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(1, zd, zn, zm);
}
void sub(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(1, zd, zn, zm);
}
void sub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(1, zd, zn, zm);
}
void sub(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(1, zd, zn, zm);
}
void sqadd(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(4, zd, zn, zm);
}
void sqadd(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(4, zd, zn, zm);
}
void sqadd(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(4, zd, zn, zm);
}
void sqadd(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(4, zd, zn, zm);
}
void uqadd(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(5, zd, zn, zm);
}
void uqadd(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(5, zd, zn, zm);
}
void uqadd(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(5, zd, zn, zm);
}
void uqadd(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(5, zd, zn, zm);
}
void sqsub(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(6, zd, zn, zm);
}
void sqsub(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(6, zd, zn, zm);
}
void sqsub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(6, zd, zn, zm);
}
void sqsub(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(6, zd, zn, zm);
}
void uqsub(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(7, zd, zn, zm);
}
void uqsub(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(7, zd, zn, zm);
}
void uqsub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(7, zd, zn, zm);
}
void uqsub(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubUnpred(7, zd, zn, zm);
}
void and_(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpUnpred(0, zd, zn, zm);
}
void orr(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpUnpred(1, zd, zn, zm);
}
void mov(const ZRegD &zd, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpUnpred(1, zd, zn, zn);
}
void eor(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpUnpred(2, zd, zn, zm);
}
void bic(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLOpUnpred(3, zd, zn, zm);
}
void index(const ZRegB &zd, const int32_t imm1, const int32_t imm2) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenImmImmInc(zd, imm1, imm2);
}
void index(const ZRegH &zd, const int32_t imm1, const int32_t imm2) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenImmImmInc(zd, imm1, imm2);
}
void index(const ZRegS &zd, const int32_t imm1, const int32_t imm2) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenImmImmInc(zd, imm1, imm2);
}
void index(const ZRegD &zd, const int32_t imm1, const int32_t imm2) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenImmImmInc(zd, imm1, imm2);
}
void index(const ZRegB &zd, const int32_t imm, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenImmRegInc(zd, imm, rm);
}
void index(const ZRegH &zd, const int32_t imm, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenImmRegInc(zd, imm, rm);
}
void index(const ZRegS &zd, const int32_t imm, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenImmRegInc(zd, imm, rm);
}
void index(const ZRegD &zd, const int32_t imm, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenImmRegInc(zd, imm, rm);
}
void index(const ZRegB &zd, const WReg &rn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenRegImmInc(zd, rn, imm);
}
void index(const ZRegH &zd, const WReg &rn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenRegImmInc(zd, rn, imm);
}
void index(const ZRegS &zd, const WReg &rn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenRegImmInc(zd, rn, imm);
}
void index(const ZRegD &zd, const XReg &rn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenRegImmInc(zd, rn, imm);
}
void index(const ZRegB &zd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenRegRegInc(zd, rn, rm);
}
void index(const ZRegH &zd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenRegRegInc(zd, rn, rm);
}
void index(const ZRegS &zd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenRegRegInc(zd, rn, rm);
}
void index(const ZRegD &zd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIndexGenRegRegInc(zd, rn, rm);
}
void addvl(const XReg &xd, const XReg &xn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveStackFrameAdjust(0, xd, xn, imm);
}
void addpl(const XReg &xd, const XReg &xn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveStackFrameAdjust(1, xd, xn, imm);
}
void rdvl(const XReg &xd, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveStackFrameSize(0, 31, xd, imm);
}
void asr(const ZRegB &zd, const ZRegB &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(0, zd, zn, amount);
}
void asr(const ZRegH &zd, const ZRegH &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(0, zd, zn, amount);
}
void asr(const ZRegS &zd, const ZRegS &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(0, zd, zn, amount);
}
void asr(const ZRegD &zd, const ZRegD &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(0, zd, zn, amount);
}
void lsr(const ZRegB &zd, const ZRegB &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(1, zd, zn, amount);
}
void lsr(const ZRegH &zd, const ZRegH &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(1, zd, zn, amount);
}
void lsr(const ZRegS &zd, const ZRegS &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(1, zd, zn, amount);
}
void lsr(const ZRegD &zd, const ZRegD &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(1, zd, zn, amount);
}
void lsl(const ZRegB &zd, const ZRegB &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(3, zd, zn, amount);
}
void lsl(const ZRegH &zd, const ZRegH &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(3, zd, zn, amount);
}
void lsl(const ZRegS &zd, const ZRegS &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(3, zd, zn, amount);
}
void lsl(const ZRegD &zd, const ZRegD &zn, const uint32_t amount) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByImmUnpred(3, zd, zn, amount);
}
void asr(const ZRegB &zd, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByWideElemUnPred(0, zd, zn, zm);
}
void asr(const ZRegH &zd, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByWideElemUnPred(0, zd, zn, zm);
}
void asr(const ZRegS &zd, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByWideElemUnPred(0, zd, zn, zm);
}
void lsr(const ZRegB &zd, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByWideElemUnPred(1, zd, zn, zm);
}
void lsr(const ZRegH &zd, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByWideElemUnPred(1, zd, zn, zm);
}
void lsr(const ZRegS &zd, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByWideElemUnPred(1, zd, zn, zm);
}
void lsl(const ZRegB &zd, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByWideElemUnPred(3, zd, zn, zm);
}
void lsl(const ZRegH &zd, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByWideElemUnPred(3, zd, zn, zm);
}
void lsl(const ZRegS &zd, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseShByWideElemUnPred(3, zd, zn, zm);
}
void adr(const ZRegS &zd, const AdrVec &adr) {
  XBYAK_SET_CODE_INFO();
  SveAddressGen(zd, adr);
}
void adr(const ZRegD &zd, const AdrVec &adr) {
  XBYAK_SET_CODE_INFO();
  SveAddressGen(zd, adr);
}
void adr(const ZRegD &zd, const AdrVecU &adr) {
  XBYAK_SET_CODE_INFO();
  SveAddressGen(zd, adr);
}
void movprfx(const ZReg &zd, const ZReg &zn) {
  XBYAK_SET_CODE_INFO();
  SveConstPrefUnpred(0, 0, zd, zn);
}
void fexpa(const ZRegH &zd, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpExpAccel(0, zd, zn);
}
void fexpa(const ZRegS &zd, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpExpAccel(0, zd, zn);
}
void fexpa(const ZRegD &zd, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpExpAccel(0, zd, zn);
}
void ftssel(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpTrigSelCoef(0, zd, zn, zm);
}
void ftssel(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpTrigSelCoef(0, zd, zn, zm);
}
void ftssel(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpTrigSelCoef(0, zd, zn, zm);
}
void cntb(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveElemCount(0, 0, xd, pat, mod, imm);
}
void cnth(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveElemCount(1, 0, xd, pat, mod, imm);
}
void cntw(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveElemCount(2, 0, xd, pat, mod, imm);
}
void cntd(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveElemCount(3, 0, xd, pat, mod, imm);
}
void incb(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByElemCount(0, 0, xd, pat, mod, imm);
}
void decb(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByElemCount(0, 1, xd, pat, mod, imm);
}
void inch(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByElemCount(1, 0, xd, pat, mod, imm);
}
void dech(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByElemCount(1, 1, xd, pat, mod, imm);
}
void incw(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByElemCount(2, 0, xd, pat, mod, imm);
}
void decw(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByElemCount(2, 1, xd, pat, mod, imm);
}
void incd(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByElemCount(3, 0, xd, pat, mod, imm);
}
void decd(const XReg &xd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByElemCount(3, 1, xd, pat, mod, imm);
}
void inch(const ZRegH &zd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByElemCount(1, 0, zd, pat, mod, imm);
}
void dech(const ZRegH &zd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByElemCount(1, 1, zd, pat, mod, imm);
}
void incw(const ZRegS &zd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByElemCount(2, 0, zd, pat, mod, imm);
}
void decw(const ZRegS &zd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByElemCount(2, 1, zd, pat, mod, imm);
}
void incd(const ZRegD &zd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByElemCount(3, 0, zd, pat, mod, imm);
}
void decd(const ZRegD &zd, const Pattern pat = ALL, const ExtMod mod = MUL,
          const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByElemCount(3, 1, zd, pat, mod, imm);
}
void sqincb(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(0, 0, 0, rdn, pat, mod, imm);
}
void sqincb(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(0, 0, 0, rdn, pat, mod, imm);
}
void uqincb(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(0, 0, 1, rdn, pat, mod, imm);
}
void uqincb(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(0, 0, 1, rdn, pat, mod, imm);
}
void sqdecb(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(0, 1, 0, rdn, pat, mod, imm);
}
void sqdecb(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(0, 1, 0, rdn, pat, mod, imm);
}
void uqdecb(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(0, 1, 1, rdn, pat, mod, imm);
}
void uqdecb(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(0, 1, 1, rdn, pat, mod, imm);
}
void sqinch(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(1, 0, 0, rdn, pat, mod, imm);
}
void sqinch(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(1, 0, 0, rdn, pat, mod, imm);
}
void uqinch(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(1, 0, 1, rdn, pat, mod, imm);
}
void uqinch(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(1, 0, 1, rdn, pat, mod, imm);
}
void sqdech(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(1, 1, 0, rdn, pat, mod, imm);
}
void sqdech(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(1, 1, 0, rdn, pat, mod, imm);
}
void uqdech(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(1, 1, 1, rdn, pat, mod, imm);
}
void uqdech(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(1, 1, 1, rdn, pat, mod, imm);
}
void sqincw(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(2, 0, 0, rdn, pat, mod, imm);
}
void sqincw(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(2, 0, 0, rdn, pat, mod, imm);
}
void uqincw(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(2, 0, 1, rdn, pat, mod, imm);
}
void uqincw(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(2, 0, 1, rdn, pat, mod, imm);
}
void sqdecw(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(2, 1, 0, rdn, pat, mod, imm);
}
void sqdecw(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(2, 1, 0, rdn, pat, mod, imm);
}
void uqdecw(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(2, 1, 1, rdn, pat, mod, imm);
}
void uqdecw(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(2, 1, 1, rdn, pat, mod, imm);
}
void sqincd(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(3, 0, 0, rdn, pat, mod, imm);
}
void sqincd(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(3, 0, 0, rdn, pat, mod, imm);
}
void uqincd(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(3, 0, 1, rdn, pat, mod, imm);
}
void uqincd(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(3, 0, 1, rdn, pat, mod, imm);
}
void sqdecd(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(3, 1, 0, rdn, pat, mod, imm);
}
void sqdecd(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(3, 1, 0, rdn, pat, mod, imm);
}
void uqdecd(const WReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(3, 1, 1, rdn, pat, mod, imm);
}
void uqdecd(const XReg &rdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByElemCount(3, 1, 1, rdn, pat, mod, imm);
}
void sqinch(const ZRegH &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(1, 0, 0, zdn, pat, mod, imm);
}
void uqinch(const ZRegH &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(1, 0, 1, zdn, pat, mod, imm);
}
void sqdech(const ZRegH &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(1, 1, 0, zdn, pat, mod, imm);
}
void uqdech(const ZRegH &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(1, 1, 1, zdn, pat, mod, imm);
}
void sqincw(const ZRegS &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(2, 0, 0, zdn, pat, mod, imm);
}
void uqincw(const ZRegS &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(2, 0, 1, zdn, pat, mod, imm);
}
void sqdecw(const ZRegS &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(2, 1, 0, zdn, pat, mod, imm);
}
void uqdecw(const ZRegS &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(2, 1, 1, zdn, pat, mod, imm);
}
void sqincd(const ZRegD &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(3, 0, 0, zdn, pat, mod, imm);
}
void uqincd(const ZRegD &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(3, 0, 1, zdn, pat, mod, imm);
}
void sqdecd(const ZRegD &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(3, 1, 0, zdn, pat, mod, imm);
}
void uqdecd(const ZRegD &zdn, const Pattern pat = ALL, const ExtMod mod = MUL,
            const uint32_t imm = 1) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByElemCount(3, 1, 1, zdn, pat, mod, imm);
}
void orr(const ZRegB &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(0, zdn, imm);
}
void orr(const ZRegH &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(0, zdn, imm);
}
void orr(const ZRegS &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(0, zdn, imm);
}
void orr(const ZRegD &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(0, zdn, imm);
}
void orn(const ZRegB &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(0, zdn, ((-1) * imm - 1));
}
void orn(const ZRegH &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(0, zdn, ((-1) * imm - 1));
}
void orn(const ZRegS &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(0, zdn, ((-1) * imm - 1));
}
void orn(const ZRegD &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(0, zdn, ((-1) * imm - 1));
}
void eor(const ZRegB &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(1, zdn, imm);
}
void eor(const ZRegH &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(1, zdn, imm);
}
void eor(const ZRegS &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(1, zdn, imm);
}
void eor(const ZRegD &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(1, zdn, imm);
}
void eon(const ZRegB &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(1, zdn, ((-1) * imm - 1));
}
void eon(const ZRegH &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(1, zdn, ((-1) * imm - 1));
}
void eon(const ZRegS &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(1, zdn, ((-1) * imm - 1));
}
void eon(const ZRegD &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(1, zdn, ((-1) * imm - 1));
}
void and_(const ZRegB &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(2, zdn, imm);
}
void and_(const ZRegH &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(2, zdn, imm);
}
void and_(const ZRegS &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(2, zdn, imm);
}
void and_(const ZRegD &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(2, zdn, imm);
}
void bic(const ZRegB &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(2, zdn, ((-1) * imm - 1));
}
void bic(const ZRegH &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(2, zdn, ((-1) * imm - 1));
}
void bic(const ZRegS &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(2, zdn, ((-1) * imm - 1));
}
void bic(const ZRegD &zdn, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBitwiseLogicalImmUnpred(2, zdn, ((-1) * imm - 1));
}
void dupm(const ZRegB &zd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBcBitmaskImm(zd, imm);
}
void dupm(const ZRegH &zd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBcBitmaskImm(zd, imm);
}
void dupm(const ZRegS &zd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBcBitmaskImm(zd, imm);
}
void dupm(const ZRegD &zd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBcBitmaskImm(zd, imm);
}
void mov(const ZRegB &zd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBcBitmaskImm(zd, genMoveMaskPrefferd(imm));
}
void mov(const ZRegH &zd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBcBitmaskImm(zd, genMoveMaskPrefferd(imm));
}
void mov(const ZRegS &zd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBcBitmaskImm(zd, genMoveMaskPrefferd(imm));
}
void mov(const ZRegD &zd, const uint64_t imm) {
  XBYAK_SET_CODE_INFO();
  SveBcBitmaskImm(zd, genMoveMaskPrefferd(imm));
}
void fcpy(const ZRegH &zd, const _PReg &pg, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveCopyFpImmPred(zd, pg, imm);
}
void fcpy(const ZRegS &zd, const _PReg &pg, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveCopyFpImmPred(zd, pg, imm);
}
void fcpy(const ZRegD &zd, const _PReg &pg, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveCopyFpImmPred(zd, pg, imm);
}
void fmov(const ZRegH &zd, const _PReg &pg, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveCopyFpImmPred(zd, pg, imm);
}
void fmov(const ZRegS &zd, const _PReg &pg, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveCopyFpImmPred(zd, pg, imm);
}
void fmov(const ZRegD &zd, const _PReg &pg, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveCopyFpImmPred(zd, pg, imm);
}
void cpy(const ZRegB &zd, const _PReg &pg, const uint32_t imm,
         const ShMod mod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, mod, sh);
}
void cpy(const ZRegH &zd, const _PReg &pg, const uint32_t imm,
         const ShMod mod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, mod, sh);
}
void cpy(const ZRegS &zd, const _PReg &pg, const uint32_t imm,
         const ShMod mod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, mod, sh);
}
void cpy(const ZRegD &zd, const _PReg &pg, const uint32_t imm,
         const ShMod mod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, mod, sh);
}
void mov(const ZRegB &zd, const _PReg &pg, const uint32_t imm,
         const ShMod mod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, mod, sh);
}
void mov(const ZRegH &zd, const _PReg &pg, const uint32_t imm,
         const ShMod mod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, mod, sh);
}
void mov(const ZRegS &zd, const _PReg &pg, const uint32_t imm,
         const ShMod mod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, mod, sh);
}
void mov(const ZRegD &zd, const _PReg &pg, const uint32_t imm,
         const ShMod mod = LSL, const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, mod, sh);
}
void fmov(const ZRegH &zd, const _PReg &pg, const uint32_t imm = 0.0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, LSL, 0);
}
void fmov(const ZRegS &zd, const _PReg &pg, const uint32_t imm = 0.0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, LSL, 0);
}
void fmov(const ZRegD &zd, const _PReg &pg, const uint32_t imm = 0.0) {
  XBYAK_SET_CODE_INFO();
  SveCopyIntImmPred(zd, pg, imm, LSL, 0);
}
void ext(const ZRegB &zdn, const ZRegB &zm, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveExtVec(zdn, zm, imm);
}
void dup(const ZRegB &zd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveBcGeneralReg(zd, rn);
}
void dup(const ZRegH &zd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveBcGeneralReg(zd, rn);
}
void dup(const ZRegS &zd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveBcGeneralReg(zd, rn);
}
void dup(const ZRegD &zd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveBcGeneralReg(zd, rn);
}
void mov(const ZRegB &zd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveBcGeneralReg(zd, rn);
}
void mov(const ZRegH &zd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveBcGeneralReg(zd, rn);
}
void mov(const ZRegS &zd, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveBcGeneralReg(zd, rn);
}
void mov(const ZRegD &zd, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveBcGeneralReg(zd, rn);
}
void dup(const ZRegB &zd, const ZRegBElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void dup(const ZRegH &zd, const ZRegHElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void dup(const ZRegS &zd, const ZRegSElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void dup(const ZRegD &zd, const ZRegDElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void dup(const ZRegQ &zd, const ZRegQElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void mov(const ZRegB &zd, const ZRegBElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void mov(const ZRegH &zd, const ZRegHElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void mov(const ZRegS &zd, const ZRegSElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void mov(const ZRegD &zd, const ZRegDElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void mov(const ZRegQ &zd, const ZRegQElem &zn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, zn);
}
void mov(const ZRegB &zd, const BReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit()));
}
void mov(const ZRegH &zd, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit()));
}
void mov(const ZRegS &zd, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit()));
}
void mov(const ZRegD &zd, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit()));
}
void mov(const ZRegQ &zd, const QReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit()));
}
void insr(const ZRegB &zdn, const BReg &vm) {
  XBYAK_SET_CODE_INFO();
  SveInsSimdFpSclarReg(zdn, vm);
}
void insr(const ZRegH &zdn, const HReg &vm) {
  XBYAK_SET_CODE_INFO();
  SveInsSimdFpSclarReg(zdn, vm);
}
void insr(const ZRegS &zdn, const SReg &vm) {
  XBYAK_SET_CODE_INFO();
  SveInsSimdFpSclarReg(zdn, vm);
}
void insr(const ZRegD &zdn, const DReg &vm) {
  XBYAK_SET_CODE_INFO();
  SveInsSimdFpSclarReg(zdn, vm);
}
void insr(const ZRegB &zdn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveInsGeneralReg(zdn, rm);
}
void insr(const ZRegH &zdn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveInsGeneralReg(zdn, rm);
}
void insr(const ZRegS &zdn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveInsGeneralReg(zdn, rm);
}
void insr(const ZRegD &zdn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveInsGeneralReg(zdn, rm);
}
void rev(const ZRegB &zd, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevVecElem(zd, zn);
}
void rev(const ZRegH &zd, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevVecElem(zd, zn);
}
void rev(const ZRegS &zd, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevVecElem(zd, zn);
}
void rev(const ZRegD &zd, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevVecElem(zd, zn);
}
void tbl(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveTableLookup(zd, zn, zm);
}
void tbl(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveTableLookup(zd, zn, zm);
}
void tbl(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveTableLookup(zd, zn, zm);
}
void tbl(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveTableLookup(zd, zn, zm);
}
void sunpklo(const ZRegH &zd, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(0, 0, zd, zn);
}
void sunpklo(const ZRegS &zd, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(0, 0, zd, zn);
}
void sunpklo(const ZRegD &zd, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(0, 0, zd, zn);
}
void sunpkhi(const ZRegH &zd, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(0, 1, zd, zn);
}
void sunpkhi(const ZRegS &zd, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(0, 1, zd, zn);
}
void sunpkhi(const ZRegD &zd, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(0, 1, zd, zn);
}
void uunpklo(const ZRegH &zd, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(1, 0, zd, zn);
}
void uunpklo(const ZRegS &zd, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(1, 0, zd, zn);
}
void uunpklo(const ZRegD &zd, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(1, 0, zd, zn);
}
void uunpkhi(const ZRegH &zd, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(1, 1, zd, zn);
}
void uunpkhi(const ZRegS &zd, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(1, 1, zd, zn);
}
void uunpkhi(const ZRegD &zd, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackVecElem(1, 1, zd, zn);
}
void zip1(const PRegB &pd, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(0, 0, pd, pn, pm);
}
void zip1(const PRegH &pd, const PRegH &pn, const PRegH &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(0, 0, pd, pn, pm);
}
void zip1(const PRegS &pd, const PRegS &pn, const PRegS &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(0, 0, pd, pn, pm);
}
void zip1(const PRegD &pd, const PRegD &pn, const PRegD &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(0, 0, pd, pn, pm);
}
void zip2(const PRegB &pd, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(0, 1, pd, pn, pm);
}
void zip2(const PRegH &pd, const PRegH &pn, const PRegH &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(0, 1, pd, pn, pm);
}
void zip2(const PRegS &pd, const PRegS &pn, const PRegS &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(0, 1, pd, pn, pm);
}
void zip2(const PRegD &pd, const PRegD &pn, const PRegD &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(0, 1, pd, pn, pm);
}
void uzp1(const PRegB &pd, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(1, 0, pd, pn, pm);
}
void uzp1(const PRegH &pd, const PRegH &pn, const PRegH &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(1, 0, pd, pn, pm);
}
void uzp1(const PRegS &pd, const PRegS &pn, const PRegS &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(1, 0, pd, pn, pm);
}
void uzp1(const PRegD &pd, const PRegD &pn, const PRegD &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(1, 0, pd, pn, pm);
}
void uzp2(const PRegB &pd, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(1, 1, pd, pn, pm);
}
void uzp2(const PRegH &pd, const PRegH &pn, const PRegH &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(1, 1, pd, pn, pm);
}
void uzp2(const PRegS &pd, const PRegS &pn, const PRegS &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(1, 1, pd, pn, pm);
}
void uzp2(const PRegD &pd, const PRegD &pn, const PRegD &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(1, 1, pd, pn, pm);
}
void trn1(const PRegB &pd, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(2, 0, pd, pn, pm);
}
void trn1(const PRegH &pd, const PRegH &pn, const PRegH &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(2, 0, pd, pn, pm);
}
void trn1(const PRegS &pd, const PRegS &pn, const PRegS &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(2, 0, pd, pn, pm);
}
void trn1(const PRegD &pd, const PRegD &pn, const PRegD &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(2, 0, pd, pn, pm);
}
void trn2(const PRegB &pd, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(2, 1, pd, pn, pm);
}
void trn2(const PRegH &pd, const PRegH &pn, const PRegH &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(2, 1, pd, pn, pm);
}
void trn2(const PRegS &pd, const PRegS &pn, const PRegS &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(2, 1, pd, pn, pm);
}
void trn2(const PRegD &pd, const PRegD &pn, const PRegD &pm) {
  XBYAK_SET_CODE_INFO();
  SvePermutePredElem(2, 1, pd, pn, pm);
}
void rev(const PRegB &pd, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SveRevPredElem(pd, pn);
}
void rev(const PRegH &pd, const PRegH &pn) {
  XBYAK_SET_CODE_INFO();
  SveRevPredElem(pd, pn);
}
void rev(const PRegS &pd, const PRegS &pn) {
  XBYAK_SET_CODE_INFO();
  SveRevPredElem(pd, pn);
}
void rev(const PRegD &pd, const PRegD &pn) {
  XBYAK_SET_CODE_INFO();
  SveRevPredElem(pd, pn);
}
void punpklo(const PRegH &pd, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackPredElem(0, pd, pn);
}
void punpkhi(const PRegH &pd, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SveUnpackPredElem(1, pd, pn);
}
void zip1(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(0, zd, zn, zm);
}
void zip1(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(0, zd, zn, zm);
}
void zip1(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(0, zd, zn, zm);
}
void zip1(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(0, zd, zn, zm);
}
void zip2(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(1, zd, zn, zm);
}
void zip2(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(1, zd, zn, zm);
}
void zip2(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(1, zd, zn, zm);
}
void zip2(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(1, zd, zn, zm);
}
void uzp1(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(2, zd, zn, zm);
}
void uzp1(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(2, zd, zn, zm);
}
void uzp1(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(2, zd, zn, zm);
}
void uzp1(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(2, zd, zn, zm);
}
void uzp2(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(3, zd, zn, zm);
}
void uzp2(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(3, zd, zn, zm);
}
void uzp2(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(3, zd, zn, zm);
}
void uzp2(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(3, zd, zn, zm);
}
void trn1(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(4, zd, zn, zm);
}
void trn1(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(4, zd, zn, zm);
}
void trn1(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(4, zd, zn, zm);
}
void trn1(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(4, zd, zn, zm);
}
void trn2(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(5, zd, zn, zm);
}
void trn2(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(5, zd, zn, zm);
}
void trn2(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(5, zd, zn, zm);
}
void trn2(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SvePermuteVecElem(5, zd, zn, zm);
}
void compact(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveCompressActElem(zd, pg, zn);
}
void compact(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveCompressActElem(zd, pg, zn);
}
void clasta(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondBcElemToVec(0, zdn, pg, zm);
}
void clasta(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondBcElemToVec(0, zdn, pg, zm);
}
void clasta(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondBcElemToVec(0, zdn, pg, zm);
}
void clasta(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondBcElemToVec(0, zdn, pg, zm);
}
void clastb(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondBcElemToVec(1, zdn, pg, zm);
}
void clastb(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondBcElemToVec(1, zdn, pg, zm);
}
void clastb(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondBcElemToVec(1, zdn, pg, zm);
}
void clastb(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondBcElemToVec(1, zdn, pg, zm);
}
void clasta(const BReg &vdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToSimdFpScalar(0, vdn, pg, zm);
}
void clasta(const HReg &vdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToSimdFpScalar(0, vdn, pg, zm);
}
void clasta(const SReg &vdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToSimdFpScalar(0, vdn, pg, zm);
}
void clasta(const DReg &vdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToSimdFpScalar(0, vdn, pg, zm);
}
void clastb(const BReg &vdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToSimdFpScalar(1, vdn, pg, zm);
}
void clastb(const HReg &vdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToSimdFpScalar(1, vdn, pg, zm);
}
void clastb(const SReg &vdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToSimdFpScalar(1, vdn, pg, zm);
}
void clastb(const DReg &vdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToSimdFpScalar(1, vdn, pg, zm);
}
void clasta(const WReg &rdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToGeneralReg(0, rdn, pg, zm);
}
void clasta(const WReg &rdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToGeneralReg(0, rdn, pg, zm);
}
void clasta(const WReg &rdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToGeneralReg(0, rdn, pg, zm);
}
void clasta(const XReg &rdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToGeneralReg(0, rdn, pg, zm);
}
void clastb(const WReg &rdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToGeneralReg(1, rdn, pg, zm);
}
void clastb(const WReg &rdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToGeneralReg(1, rdn, pg, zm);
}
void clastb(const WReg &rdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToGeneralReg(1, rdn, pg, zm);
}
void clastb(const XReg &rdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveCondExtElemToGeneralReg(1, rdn, pg, zm);
}
void cpy(const ZRegB &zd, const _PReg &pg, const BReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveCopySimdFpScalarToVecPred(zd, pg, vn);
}
void cpy(const ZRegH &zd, const _PReg &pg, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveCopySimdFpScalarToVecPred(zd, pg, vn);
}
void cpy(const ZRegS &zd, const _PReg &pg, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveCopySimdFpScalarToVecPred(zd, pg, vn);
}
void cpy(const ZRegD &zd, const _PReg &pg, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveCopySimdFpScalarToVecPred(zd, pg, vn);
}
void mov(const ZRegB &zd, const _PReg &pg, const BReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveCopySimdFpScalarToVecPred(zd, pg, vn);
}
void mov(const ZRegH &zd, const _PReg &pg, const HReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveCopySimdFpScalarToVecPred(zd, pg, vn);
}
void mov(const ZRegS &zd, const _PReg &pg, const SReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveCopySimdFpScalarToVecPred(zd, pg, vn);
}
void mov(const ZRegD &zd, const _PReg &pg, const DReg &vn) {
  XBYAK_SET_CODE_INFO();
  SveCopySimdFpScalarToVecPred(zd, pg, vn);
}
void cpy(const ZRegB &zd, const _PReg &pg, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveCopyGeneralRegToVecPred(zd, pg, rn);
}
void cpy(const ZRegH &zd, const _PReg &pg, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveCopyGeneralRegToVecPred(zd, pg, rn);
}
void cpy(const ZRegS &zd, const _PReg &pg, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveCopyGeneralRegToVecPred(zd, pg, rn);
}
void cpy(const ZRegD &zd, const _PReg &pg, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveCopyGeneralRegToVecPred(zd, pg, rn);
}
void mov(const ZRegB &zd, const _PReg &pg, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveCopyGeneralRegToVecPred(zd, pg, rn);
}
void mov(const ZRegH &zd, const _PReg &pg, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveCopyGeneralRegToVecPred(zd, pg, rn);
}
void mov(const ZRegS &zd, const _PReg &pg, const WReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveCopyGeneralRegToVecPred(zd, pg, rn);
}
void mov(const ZRegD &zd, const _PReg &pg, const XReg &rn) {
  XBYAK_SET_CODE_INFO();
  SveCopyGeneralRegToVecPred(zd, pg, rn);
}
void lasta(const BReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToSimdFpScalar(0, vd, pg, zn);
}
void lasta(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToSimdFpScalar(0, vd, pg, zn);
}
void lasta(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToSimdFpScalar(0, vd, pg, zn);
}
void lasta(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToSimdFpScalar(0, vd, pg, zn);
}
void lastb(const BReg &vd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToSimdFpScalar(1, vd, pg, zn);
}
void lastb(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToSimdFpScalar(1, vd, pg, zn);
}
void lastb(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToSimdFpScalar(1, vd, pg, zn);
}
void lastb(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToSimdFpScalar(1, vd, pg, zn);
}
void lasta(const WReg &rd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToGeneralReg(0, rd, pg, zn);
}
void lasta(const WReg &rd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToGeneralReg(0, rd, pg, zn);
}
void lasta(const WReg &rd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToGeneralReg(0, rd, pg, zn);
}
void lasta(const XReg &rd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToGeneralReg(0, rd, pg, zn);
}
void lastb(const WReg &rd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToGeneralReg(1, rd, pg, zn);
}
void lastb(const WReg &rd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToGeneralReg(1, rd, pg, zn);
}
void lastb(const WReg &rd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToGeneralReg(1, rd, pg, zn);
}
void lastb(const XReg &rd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveExtElemToGeneralReg(1, rd, pg, zn);
}
void revb(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(0, zd, pg, zn);
}
void revb(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(0, zd, pg, zn);
}
void revb(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(0, zd, pg, zn);
}
void revh(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(1, zd, pg, zn);
}
void revh(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(1, zd, pg, zn);
}
void revw(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(2, zd, pg, zn);
}
void rbit(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(3, zd, pg, zn);
}
void rbit(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(3, zd, pg, zn);
}
void rbit(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(3, zd, pg, zn);
}
void rbit(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveRevWithinElem(3, zd, pg, zn);
}
void splice(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveSelVecSplice(zdn, pg, zm);
}
void splice(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveSelVecSplice(zdn, pg, zm);
}
void splice(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveSelVecSplice(zdn, pg, zm);
}
void splice(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveSelVecSplice(zdn, pg, zm);
}
void sel(const ZRegB &zd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveSelVecElemPred(zd, pg, zn, zm);
}
void sel(const ZRegH &zd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveSelVecElemPred(zd, pg, zn, zm);
}
void sel(const ZRegS &zd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveSelVecElemPred(zd, pg, zn, zm);
}
void sel(const ZRegD &zd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveSelVecElemPred(zd, pg, zn, zm);
}
void mov(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) {
  XBYAK_SET_CODE_INFO();
  SveSelVecElemPred(zd, pg, zn, zd);
}
void mov(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveSelVecElemPred(zd, pg, zn, zd);
}
void mov(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveSelVecElemPred(zd, pg, zn, zd);
}
void mov(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveSelVecElemPred(zd, pg, zn, zd);
}
void cmphs(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 0, pd, pg, zn, zm);
}
void cmphs(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 0, pd, pg, zn, zm);
}
void cmphs(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 0, pd, pg, zn, zm);
}
void cmphs(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 0, pd, pg, zn, zm);
}
void cmphi(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 1, pd, pg, zn, zm);
}
void cmphi(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 1, pd, pg, zn, zm);
}
void cmphi(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 1, pd, pg, zn, zm);
}
void cmphi(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 1, pd, pg, zn, zm);
}
void cmpeq(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 1, 0, pd, pg, zn, zm);
}
void cmpeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 1, 0, pd, pg, zn, zm);
}
void cmpeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 1, 0, pd, pg, zn, zm);
}
void cmpne(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 1, 1, pd, pg, zn, zm);
}
void cmpne(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 1, 1, pd, pg, zn, zm);
}
void cmpne(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 1, 1, pd, pg, zn, zm);
}
void cmpge(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 0, pd, pg, zn, zm);
}
void cmpge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 0, pd, pg, zn, zm);
}
void cmpge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 0, pd, pg, zn, zm);
}
void cmpge(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 0, pd, pg, zn, zm);
}
void cmpgt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 1, pd, pg, zn, zm);
}
void cmpgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 1, pd, pg, zn, zm);
}
void cmpgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 1, pd, pg, zn, zm);
}
void cmpgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 1, pd, pg, zn, zm);
}
void cmpeq(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 1, 0, pd, pg, zn, zm);
}
void cmpeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 1, 0, pd, pg, zn, zm);
}
void cmpeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 1, 0, pd, pg, zn, zm);
}
void cmpeq(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 1, 0, pd, pg, zn, zm);
}
void cmpne(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 1, 1, pd, pg, zn, zm);
}
void cmpne(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 1, 1, pd, pg, zn, zm);
}
void cmpne(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 1, 1, pd, pg, zn, zm);
}
void cmpne(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 1, 1, pd, pg, zn, zm);
}
void cmple(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 0, pd, pg, zm, zn);
}
void cmple(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 0, pd, pg, zm, zn);
}
void cmple(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 0, pd, pg, zm, zn);
}
void cmple(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 0, pd, pg, zm, zn);
}
void cmplo(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 1, pd, pg, zm, zn);
}
void cmplo(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 1, pd, pg, zm, zn);
}
void cmplo(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 1, pd, pg, zm, zn);
}
void cmplo(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 1, pd, pg, zm, zn);
}
void cmpls(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 0, pd, pg, zm, zn);
}
void cmpls(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 0, pd, pg, zm, zn);
}
void cmpls(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 0, pd, pg, zm, zn);
}
void cmpls(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(0, 0, 0, pd, pg, zm, zn);
}
void cmplt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 1, pd, pg, zm, zn);
}
void cmplt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 1, pd, pg, zm, zn);
}
void cmplt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 1, pd, pg, zm, zn);
}
void cmplt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompVec(1, 0, 1, pd, pg, zm, zn);
}
void cmpge(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 0, 0, pd, pg, zn, zm);
}
void cmpge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 0, 0, pd, pg, zn, zm);
}
void cmpge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 0, 0, pd, pg, zn, zm);
}
void cmpgt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 0, 1, pd, pg, zn, zm);
}
void cmpgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 0, 1, pd, pg, zn, zm);
}
void cmpgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 0, 1, pd, pg, zn, zm);
}
void cmplt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 1, 0, pd, pg, zn, zm);
}
void cmplt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 1, 0, pd, pg, zn, zm);
}
void cmplt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 1, 0, pd, pg, zn, zm);
}
void cmple(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 1, 1, pd, pg, zn, zm);
}
void cmple(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 1, 1, pd, pg, zn, zm);
}
void cmple(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(0, 1, 1, pd, pg, zn, zm);
}
void cmphs(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 0, 0, pd, pg, zn, zm);
}
void cmphs(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 0, 0, pd, pg, zn, zm);
}
void cmphs(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 0, 0, pd, pg, zn, zm);
}
void cmphi(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 0, 1, pd, pg, zn, zm);
}
void cmphi(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 0, 1, pd, pg, zn, zm);
}
void cmphi(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 0, 1, pd, pg, zn, zm);
}
void cmplo(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 1, 0, pd, pg, zn, zm);
}
void cmplo(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 1, 0, pd, pg, zn, zm);
}
void cmplo(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 1, 0, pd, pg, zn, zm);
}
void cmpls(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 1, 1, pd, pg, zn, zm);
}
void cmpls(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 1, 1, pd, pg, zn, zm);
}
void cmpls(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompWideElem(1, 1, 1, pd, pg, zn, zm);
}
void cmphs(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(0, 0, pd, pg, zn, imm);
}
void cmphs(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(0, 0, pd, pg, zn, imm);
}
void cmphs(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(0, 0, pd, pg, zn, imm);
}
void cmphs(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(0, 0, pd, pg, zn, imm);
}
void cmphi(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(0, 1, pd, pg, zn, imm);
}
void cmphi(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(0, 1, pd, pg, zn, imm);
}
void cmphi(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(0, 1, pd, pg, zn, imm);
}
void cmphi(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(0, 1, pd, pg, zn, imm);
}
void cmplo(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(1, 0, pd, pg, zn, imm);
}
void cmplo(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(1, 0, pd, pg, zn, imm);
}
void cmplo(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(1, 0, pd, pg, zn, imm);
}
void cmplo(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(1, 0, pd, pg, zn, imm);
}
void cmpls(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(1, 1, pd, pg, zn, imm);
}
void cmpls(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(1, 1, pd, pg, zn, imm);
}
void cmpls(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(1, 1, pd, pg, zn, imm);
}
void cmpls(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompUImm(1, 1, pd, pg, zn, imm);
}
void and_(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 0, 0, 0, pd, pg, pn, pm);
}
void mov(const PRegB &pd, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 0, pg.isM(), pg.isM(), pd, pg, pn, (pg.isZ()) ? pn : pd);
}
void bic(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 0, 0, 1, pd, pg, pn, pm);
}
void eor(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 0, 1, 0, pd, pg, pn, pm);
}
void not_(const PRegB &pd, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 0, 1, 0, pd, pg, pn, pg);
}
void sel(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 0, 1, 1, pd, pg, pn, pm);
}
void ands(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 1, 0, 0, pd, pg, pn, pm);
}
void movs(const PRegB &pd, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 1, 0, 0, pd, pg, pn, pn);
}
void bics(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 1, 0, 1, pd, pg, pn, pm);
}
void eors(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 1, 1, 0, pd, pg, pn, pm);
}
void nots(const PRegB &pd, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(0, 1, 1, 0, pd, pg, pn, pg);
}
void orr(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 0, 0, 0, pd, pg, pn, pm);
}
void mov(const PRegB &pd, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 0, 0, 0, pd, pn, pn, pn);
}
void orn(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 0, 0, 1, pd, pg, pn, pm);
}
void nor(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 0, 1, 0, pd, pg, pn, pm);
}
void nand(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 0, 1, 1, pd, pg, pn, pm);
}
void orrs(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 1, 0, 0, pd, pg, pn, pm);
}
void movs(const PRegB &pd, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 1, 0, 0, pd, pn, pn, pn);
}
void orns(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 1, 0, 1, pd, pg, pn, pm);
}
void nors(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 1, 1, 0, pd, pg, pn, pm);
}
void nands(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePredLOp(1, 1, 1, 1, pd, pg, pn, pm);
}
void brkpa(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePropagateBreakPrevPtn(0, 0, 0, pd, pg, pn, pm);
}
void brkpb(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePropagateBreakPrevPtn(0, 0, 1, pd, pg, pn, pm);
}
void brkpas(const PRegB &pd, const _PReg &pg, const PRegB &pn,
            const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePropagateBreakPrevPtn(0, 1, 0, pd, pg, pn, pm);
}
void brkpbs(const PRegB &pd, const _PReg &pg, const PRegB &pn,
            const PRegB &pm) {
  XBYAK_SET_CODE_INFO();
  SvePropagateBreakPrevPtn(0, 1, 1, pd, pg, pn, pm);
}
void brka(const PRegB &pd, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePartitionBreakCond(0, 0, pd, pg, pn);
}
void brkas(const PRegB &pd, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePartitionBreakCond(0, 1, pd, pg, pn);
}
void brkb(const PRegB &pd, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePartitionBreakCond(1, 0, pd, pg, pn);
}
void brkbs(const PRegB &pd, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePartitionBreakCond(1, 1, pd, pg, pn);
}
void brkn(const PRegB &pdm, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePropagateBreakNextPart(0, pdm, pg, pn);
}
void brkns(const PRegB &pdm, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePropagateBreakNextPart(1, pdm, pg, pn);
}
void pfirst(const PRegB &pdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SvePredFirstAct(0, 1, pdn, pg);
}
void ptrue(const PRegB &pd, const Pattern pat = ALL) {
  XBYAK_SET_CODE_INFO();
  SvePredInit(0, pd, pat);
}
void ptrue(const PRegH &pd, const Pattern pat = ALL) {
  XBYAK_SET_CODE_INFO();
  SvePredInit(0, pd, pat);
}
void ptrue(const PRegS &pd, const Pattern pat = ALL) {
  XBYAK_SET_CODE_INFO();
  SvePredInit(0, pd, pat);
}
void ptrue(const PRegD &pd, const Pattern pat = ALL) {
  XBYAK_SET_CODE_INFO();
  SvePredInit(0, pd, pat);
}
void ptrues(const PRegB &pd, const Pattern pat = ALL) {
  XBYAK_SET_CODE_INFO();
  SvePredInit(1, pd, pat);
}
void ptrues(const PRegH &pd, const Pattern pat = ALL) {
  XBYAK_SET_CODE_INFO();
  SvePredInit(1, pd, pat);
}
void ptrues(const PRegS &pd, const Pattern pat = ALL) {
  XBYAK_SET_CODE_INFO();
  SvePredInit(1, pd, pat);
}
void ptrues(const PRegD &pd, const Pattern pat = ALL) {
  XBYAK_SET_CODE_INFO();
  SvePredInit(1, pd, pat);
}
void pnext(const PRegB &pdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SvePredNextAct(pdn, pg);
}
void pnext(const PRegH &pdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SvePredNextAct(pdn, pg);
}
void pnext(const PRegS &pdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SvePredNextAct(pdn, pg);
}
void pnext(const PRegD &pdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SvePredNextAct(pdn, pg);
}
void rdffr(const PRegB &pd, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SvePredReadFFRPred(0, 0, pd, pg);
}
void rdffrs(const PRegB &pd, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SvePredReadFFRPred(0, 1, pd, pg);
}
void rdffr(const PRegB &pd) {
  XBYAK_SET_CODE_INFO();
  SvePredReadFFRUnpred(0, 0, pd);
}
void ptest(const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredTest(0, 1, 0, pg, pn);
}
void pfalse(const PRegB &pd) {
  XBYAK_SET_CODE_INFO();
  SvePredZero(0, 0, pd);
}
void cmpge(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 0, 0, pd, pg, zn, imm);
}
void cmpge(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 0, 0, pd, pg, zn, imm);
}
void cmpge(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 0, 0, pd, pg, zn, imm);
}
void cmpge(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 0, 0, pd, pg, zn, imm);
}
void cmpgt(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 0, 1, pd, pg, zn, imm);
}
void cmpgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 0, 1, pd, pg, zn, imm);
}
void cmpgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 0, 1, pd, pg, zn, imm);
}
void cmpgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 0, 1, pd, pg, zn, imm);
}
void cmplt(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 1, 0, pd, pg, zn, imm);
}
void cmplt(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 1, 0, pd, pg, zn, imm);
}
void cmplt(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 1, 0, pd, pg, zn, imm);
}
void cmplt(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 1, 0, pd, pg, zn, imm);
}
void cmple(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 1, 1, pd, pg, zn, imm);
}
void cmple(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 1, 1, pd, pg, zn, imm);
}
void cmple(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 1, 1, pd, pg, zn, imm);
}
void cmple(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(0, 1, 1, pd, pg, zn, imm);
}
void cmpeq(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(1, 0, 0, pd, pg, zn, imm);
}
void cmpeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(1, 0, 0, pd, pg, zn, imm);
}
void cmpeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(1, 0, 0, pd, pg, zn, imm);
}
void cmpeq(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(1, 0, 0, pd, pg, zn, imm);
}
void cmpne(const PRegB &pd, const _PReg &pg, const ZRegB &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(1, 0, 1, pd, pg, zn, imm);
}
void cmpne(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(1, 0, 1, pd, pg, zn, imm);
}
void cmpne(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(1, 0, 1, pd, pg, zn, imm);
}
void cmpne(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompSImm(1, 0, 1, pd, pg, zn, imm);
}
void cntp(const XReg &rd, const _PReg &pg, const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredCount(0, 0, rd, pg, pn);
}
void cntp(const XReg &rd, const _PReg &pg, const PRegH &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredCount(0, 0, rd, pg, pn);
}
void cntp(const XReg &rd, const _PReg &pg, const PRegS &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredCount(0, 0, rd, pg, pn);
}
void cntp(const XReg &rd, const _PReg &pg, const PRegD &pn) {
  XBYAK_SET_CODE_INFO();
  SvePredCount(0, 0, rd, pg, pn);
}
void incp(const XReg &xdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByPredCount(0, 0, 0, xdn, pg);
}
void incp(const XReg &xdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByPredCount(0, 0, 0, xdn, pg);
}
void incp(const XReg &xdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByPredCount(0, 0, 0, xdn, pg);
}
void incp(const XReg &xdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByPredCount(0, 0, 0, xdn, pg);
}
void decp(const XReg &xdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByPredCount(0, 1, 0, xdn, pg);
}
void decp(const XReg &xdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByPredCount(0, 1, 0, xdn, pg);
}
void decp(const XReg &xdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByPredCount(0, 1, 0, xdn, pg);
}
void decp(const XReg &xdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecRegByPredCount(0, 1, 0, xdn, pg);
}
void incp(const ZRegH &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByPredCount(0, 0, 0, zdn, pg);
}
void incp(const ZRegS &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByPredCount(0, 0, 0, zdn, pg);
}
void incp(const ZRegD &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByPredCount(0, 0, 0, zdn, pg);
}
void decp(const ZRegH &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByPredCount(0, 1, 0, zdn, pg);
}
void decp(const ZRegS &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByPredCount(0, 1, 0, zdn, pg);
}
void decp(const ZRegD &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveIncDecVecByPredCount(0, 1, 0, zdn, pg);
}
void sqincp(const WReg &rdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg);
}
void sqincp(const WReg &rdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg);
}
void sqincp(const WReg &rdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg);
}
void sqincp(const WReg &rdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg);
}
void sqincp(const XReg &rdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg);
}
void sqincp(const XReg &rdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg);
}
void sqincp(const XReg &rdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg);
}
void sqincp(const XReg &rdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg);
}
void uqincp(const WReg &rdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg);
}
void uqincp(const WReg &rdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg);
}
void uqincp(const WReg &rdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg);
}
void uqincp(const WReg &rdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg);
}
void uqincp(const XReg &rdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg);
}
void uqincp(const XReg &rdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg);
}
void uqincp(const XReg &rdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg);
}
void uqincp(const XReg &rdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg);
}
void sqdecp(const WReg &rdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg);
}
void sqdecp(const WReg &rdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg);
}
void sqdecp(const WReg &rdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg);
}
void sqdecp(const WReg &rdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg);
}
void sqdecp(const XReg &rdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg);
}
void sqdecp(const XReg &rdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg);
}
void sqdecp(const XReg &rdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg);
}
void sqdecp(const XReg &rdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg);
}
void uqdecp(const WReg &rdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg);
}
void uqdecp(const WReg &rdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg);
}
void uqdecp(const WReg &rdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg);
}
void uqdecp(const WReg &rdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg);
}
void uqdecp(const XReg &rdn, const PRegB &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg);
}
void uqdecp(const XReg &rdn, const PRegH &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg);
}
void uqdecp(const XReg &rdn, const PRegS &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg);
}
void uqdecp(const XReg &rdn, const PRegD &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg);
}
void sqincp(const ZRegH &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(0, 0, 0, zdn, pg);
}
void sqincp(const ZRegS &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(0, 0, 0, zdn, pg);
}
void sqincp(const ZRegD &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(0, 0, 0, zdn, pg);
}
void uqincp(const ZRegH &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(0, 1, 0, zdn, pg);
}
void uqincp(const ZRegS &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(0, 1, 0, zdn, pg);
}
void uqincp(const ZRegD &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(0, 1, 0, zdn, pg);
}
void sqdecp(const ZRegH &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(1, 0, 0, zdn, pg);
}
void sqdecp(const ZRegS &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(1, 0, 0, zdn, pg);
}
void sqdecp(const ZRegD &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(1, 0, 0, zdn, pg);
}
void uqdecp(const ZRegH &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(1, 1, 0, zdn, pg);
}
void uqdecp(const ZRegS &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(1, 1, 0, zdn, pg);
}
void uqdecp(const ZRegD &zdn, const _PReg &pg) {
  XBYAK_SET_CODE_INFO();
  SveSatuIncDecVecByPredCount(1, 1, 0, zdn, pg);
}
void setffr() {
  XBYAK_SET_CODE_INFO();
  SveFFRInit(0);
}
void wrffr(const PRegB &pn) {
  XBYAK_SET_CODE_INFO();
  SveFFRWritePred(0, pn);
}
void ctermeq(const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveCondTermScalars(1, 0, rn, rm);
}
void ctermeq(const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveCondTermScalars(1, 0, rn, rm);
}
void ctermne(const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveCondTermScalars(1, 1, rn, rm);
}
void ctermne(const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveCondTermScalars(1, 1, rn, rm);
}
void whilelt(const PRegB &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm);
}
void whilelt(const PRegH &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm);
}
void whilelt(const PRegS &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm);
}
void whilelt(const PRegD &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm);
}
void whilelt(const PRegB &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm);
}
void whilelt(const PRegH &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm);
}
void whilelt(const PRegS &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm);
}
void whilelt(const PRegD &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm);
}
void whilele(const PRegB &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm);
}
void whilele(const PRegH &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm);
}
void whilele(const PRegS &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm);
}
void whilele(const PRegD &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm);
}
void whilele(const PRegB &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm);
}
void whilele(const PRegH &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm);
}
void whilele(const PRegS &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm);
}
void whilele(const PRegD &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm);
}
void whilelo(const PRegB &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm);
}
void whilelo(const PRegH &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm);
}
void whilelo(const PRegS &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm);
}
void whilelo(const PRegD &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm);
}
void whilelo(const PRegB &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm);
}
void whilelo(const PRegH &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm);
}
void whilelo(const PRegS &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm);
}
void whilelo(const PRegD &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm);
}
void whilels(const PRegB &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm);
}
void whilels(const PRegH &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm);
}
void whilels(const PRegS &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm);
}
void whilels(const PRegD &pd, const WReg &rn, const WReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm);
}
void whilels(const PRegB &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm);
}
void whilels(const PRegH &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm);
}
void whilels(const PRegS &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm);
}
void whilels(const PRegD &pd, const XReg &rn, const XReg &rm) {
  XBYAK_SET_CODE_INFO();
  SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm);
}
void fdup(const ZRegH &zd, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveBcFpImmUnpred(0, 0, zd, imm);
}
void fdup(const ZRegS &zd, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveBcFpImmUnpred(0, 0, zd, imm);
}
void fdup(const ZRegD &zd, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveBcFpImmUnpred(0, 0, zd, imm);
}
void fmov(const ZRegH &zd, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveBcFpImmUnpred(0, 0, zd, imm);
}
void fmov(const ZRegS &zd, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveBcFpImmUnpred(0, 0, zd, imm);
}
void fmov(const ZRegD &zd, const double imm) {
  XBYAK_SET_CODE_INFO();
  SveBcFpImmUnpred(0, 0, zd, imm);
}
void dup(const ZRegB &zd, const int32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, imm, mod, sh);
}
void dup(const ZRegH &zd, const int32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, imm, mod, sh);
}
void dup(const ZRegS &zd, const int32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, imm, mod, sh);
}
void dup(const ZRegD &zd, const int32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, imm, mod, sh);
}
void mov(const ZRegB &zd, const int32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, imm, mod, sh);
}
void mov(const ZRegH &zd, const int32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, imm, mod, sh);
}
void mov(const ZRegS &zd, const int32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, imm, mod, sh);
}
void mov(const ZRegD &zd, const int32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, imm, mod, sh);
}
void fmov(const ZRegB &zd, const float imm = 0.0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), LSL, 0);
}
void fmov(const ZRegH &zd, const float imm = 0.0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), LSL, 0);
}
void fmov(const ZRegS &zd, const float imm = 0.0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), LSL, 0);
}
void fmov(const ZRegD &zd, const float imm = 0.0) {
  XBYAK_SET_CODE_INFO();
  SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), LSL, 0);
}
void add(const ZRegB &zdn, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(0, zdn, imm, mod, sh);
}
void add(const ZRegH &zdn, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(0, zdn, imm, mod, sh);
}
void add(const ZRegS &zdn, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(0, zdn, imm, mod, sh);
}
void add(const ZRegD &zdn, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(0, zdn, imm, mod, sh);
}
void sub(const ZRegB &zdn, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(1, zdn, imm, mod, sh);
}
void sub(const ZRegH &zdn, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(1, zdn, imm, mod, sh);
}
void sub(const ZRegS &zdn, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(1, zdn, imm, mod, sh);
}
void sub(const ZRegD &zdn, const uint32_t imm, const ShMod mod = LSL,
         const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(1, zdn, imm, mod, sh);
}
void subr(const ZRegB &zdn, const uint32_t imm, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(3, zdn, imm, mod, sh);
}
void subr(const ZRegH &zdn, const uint32_t imm, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(3, zdn, imm, mod, sh);
}
void subr(const ZRegS &zdn, const uint32_t imm, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(3, zdn, imm, mod, sh);
}
void subr(const ZRegD &zdn, const uint32_t imm, const ShMod mod = LSL,
          const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(3, zdn, imm, mod, sh);
}
void sqadd(const ZRegB &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(4, zdn, imm, mod, sh);
}
void sqadd(const ZRegH &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(4, zdn, imm, mod, sh);
}
void sqadd(const ZRegS &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(4, zdn, imm, mod, sh);
}
void sqadd(const ZRegD &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(4, zdn, imm, mod, sh);
}
void uqadd(const ZRegB &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(5, zdn, imm, mod, sh);
}
void uqadd(const ZRegH &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(5, zdn, imm, mod, sh);
}
void uqadd(const ZRegS &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(5, zdn, imm, mod, sh);
}
void uqadd(const ZRegD &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(5, zdn, imm, mod, sh);
}
void sqsub(const ZRegB &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(6, zdn, imm, mod, sh);
}
void sqsub(const ZRegH &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(6, zdn, imm, mod, sh);
}
void sqsub(const ZRegS &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(6, zdn, imm, mod, sh);
}
void sqsub(const ZRegD &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(6, zdn, imm, mod, sh);
}
void uqsub(const ZRegB &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(7, zdn, imm, mod, sh);
}
void uqsub(const ZRegH &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(7, zdn, imm, mod, sh);
}
void uqsub(const ZRegS &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(7, zdn, imm, mod, sh);
}
void uqsub(const ZRegD &zdn, const uint32_t imm, const ShMod mod = LSL,
           const uint32_t sh = 0) {
  XBYAK_SET_CODE_INFO();
  SveIntAddSubImmUnpred(7, zdn, imm, mod, sh);
}
void smax(const ZRegB &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(0, 0, zdn, imm);
}
void smax(const ZRegH &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(0, 0, zdn, imm);
}
void smax(const ZRegS &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(0, 0, zdn, imm);
}
void smax(const ZRegD &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(0, 0, zdn, imm);
}
void umax(const ZRegB &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(1, 0, zdn, imm);
}
void umax(const ZRegH &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(1, 0, zdn, imm);
}
void umax(const ZRegS &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(1, 0, zdn, imm);
}
void umax(const ZRegD &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(1, 0, zdn, imm);
}
void smin(const ZRegB &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(2, 0, zdn, imm);
}
void smin(const ZRegH &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(2, 0, zdn, imm);
}
void smin(const ZRegS &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(2, 0, zdn, imm);
}
void smin(const ZRegD &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(2, 0, zdn, imm);
}
void umin(const ZRegB &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(3, 0, zdn, imm);
}
void umin(const ZRegH &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(3, 0, zdn, imm);
}
void umin(const ZRegS &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(3, 0, zdn, imm);
}
void umin(const ZRegD &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMinMaxImmUnpred(3, 0, zdn, imm);
}
void mul(const ZRegB &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultImmUnpred(0, 0, zdn, imm);
}
void mul(const ZRegH &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultImmUnpred(0, 0, zdn, imm);
}
void mul(const ZRegS &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultImmUnpred(0, 0, zdn, imm);
}
void mul(const ZRegD &zdn, const int32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveIntMultImmUnpred(0, 0, zdn, imm);
}
void sdot(const ZRegS &zda, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntDotProdcutUnpred(0, zda, zn, zm);
}
void sdot(const ZRegD &zda, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntDotProdcutUnpred(0, zda, zn, zm);
}
void udot(const ZRegS &zda, const ZRegB &zn, const ZRegB &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntDotProdcutUnpred(1, zda, zn, zm);
}
void udot(const ZRegD &zda, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntDotProdcutUnpred(1, zda, zn, zm);
}
void sdot(const ZRegS &zda, const ZRegB &zn, const ZRegBElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntDotProdcutIndexed(2, 0, zda, zn, zm);
}
void udot(const ZRegS &zda, const ZRegB &zn, const ZRegBElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntDotProdcutIndexed(2, 1, zda, zn, zm);
}
void sdot(const ZRegD &zda, const ZRegH &zn, const ZRegHElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntDotProdcutIndexed(3, 0, zda, zn, zm);
}
void udot(const ZRegD &zda, const ZRegH &zn, const ZRegHElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveIntDotProdcutIndexed(3, 1, zda, zn, zm);
}
void fcadd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm,
           const uint32_t ct) {
  XBYAK_SET_CODE_INFO();
  SveFpComplexAddPred(zdn, pg, zm, ct);
}
void fcadd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm,
           const uint32_t ct) {
  XBYAK_SET_CODE_INFO();
  SveFpComplexAddPred(zdn, pg, zm, ct);
}
void fcadd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm,
           const uint32_t ct) {
  XBYAK_SET_CODE_INFO();
  SveFpComplexAddPred(zdn, pg, zm, ct);
}
void fcmla(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm,
           const uint32_t ct) {
  XBYAK_SET_CODE_INFO();
  SveFpComplexMultAddPred(zda, pg, zn, zm, ct);
}
void fcmla(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm,
           const uint32_t ct) {
  XBYAK_SET_CODE_INFO();
  SveFpComplexMultAddPred(zda, pg, zn, zm, ct);
}
void fcmla(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm,
           const uint32_t ct) {
  XBYAK_SET_CODE_INFO();
  SveFpComplexMultAddPred(zda, pg, zn, zm, ct);
}
void fmla(const ZRegH &zda, const ZRegH &zn, const ZRegHElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAddIndexed(0, zda, zn, zm);
}
void fmla(const ZRegS &zda, const ZRegS &zn, const ZRegSElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAddIndexed(0, zda, zn, zm);
}
void fmla(const ZRegD &zda, const ZRegD &zn, const ZRegDElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAddIndexed(0, zda, zn, zm);
}
void fmls(const ZRegH &zda, const ZRegH &zn, const ZRegHElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAddIndexed(1, zda, zn, zm);
}
void fmls(const ZRegS &zda, const ZRegS &zn, const ZRegSElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAddIndexed(1, zda, zn, zm);
}
void fmls(const ZRegD &zda, const ZRegD &zn, const ZRegDElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAddIndexed(1, zda, zn, zm);
}
void fcmla(const ZRegH &zda, const ZRegH &zn, const ZRegHElem &zm,
           const uint32_t ct) {
  XBYAK_SET_CODE_INFO();
  SveFpComplexMultAddIndexed(zda, zn, zm, ct);
}
void fcmla(const ZRegS &zda, const ZRegS &zn, const ZRegSElem &zm,
           const uint32_t ct) {
  XBYAK_SET_CODE_INFO();
  SveFpComplexMultAddIndexed(zda, zn, zm, ct);
}
void fmul(const ZRegH &zd, const ZRegH &zn, const ZRegHElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultIndexed(zd, zn, zm);
}
void fmul(const ZRegS &zd, const ZRegS &zn, const ZRegSElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultIndexed(zd, zn, zm);
}
void fmul(const ZRegD &zd, const ZRegD &zn, const ZRegDElem &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultIndexed(zd, zn, zm);
}
void faddv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(0, vd, pg, zn);
}
void faddv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(0, vd, pg, zn);
}
void faddv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(0, vd, pg, zn);
}
void fmaxnmv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(4, vd, pg, zn);
}
void fmaxnmv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(4, vd, pg, zn);
}
void fmaxnmv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(4, vd, pg, zn);
}
void fminnmv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(5, vd, pg, zn);
}
void fminnmv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(5, vd, pg, zn);
}
void fminnmv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(5, vd, pg, zn);
}
void fmaxv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(6, vd, pg, zn);
}
void fmaxv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(6, vd, pg, zn);
}
void fmaxv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(6, vd, pg, zn);
}
void fminv(const HReg &vd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(7, vd, pg, zn);
}
void fminv(const SReg &vd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(7, vd, pg, zn);
}
void fminv(const DReg &vd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRecurReduct(7, vd, pg, zn);
}
void frecpe(const ZRegH &zd, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpReciproEstUnPred(6, zd, zn);
}
void frecpe(const ZRegS &zd, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpReciproEstUnPred(6, zd, zn);
}
void frecpe(const ZRegD &zd, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpReciproEstUnPred(6, zd, zn);
}
void frsqrte(const ZRegH &zd, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpReciproEstUnPred(7, zd, zn);
}
void frsqrte(const ZRegS &zd, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpReciproEstUnPred(7, zd, zn);
}
void frsqrte(const ZRegD &zd, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpReciproEstUnPred(7, zd, zn);
}
void fcmge(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 0, 0, pd, pg, zn, zero);
}
void fcmge(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 0, 0, pd, pg, zn, zero);
}
void fcmge(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 0, 0, pd, pg, zn, zero);
}
void fcmgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 0, 1, pd, pg, zn, zero);
}
void fcmgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 0, 1, pd, pg, zn, zero);
}
void fcmgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 0, 1, pd, pg, zn, zero);
}
void fcmlt(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 1, 0, pd, pg, zn, zero);
}
void fcmlt(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 1, 0, pd, pg, zn, zero);
}
void fcmlt(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 1, 0, pd, pg, zn, zero);
}
void fcmle(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 1, 1, pd, pg, zn, zero);
}
void fcmle(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 1, 1, pd, pg, zn, zero);
}
void fcmle(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(0, 1, 1, pd, pg, zn, zero);
}
void fcmeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(1, 0, 0, pd, pg, zn, zero);
}
void fcmeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(1, 0, 0, pd, pg, zn, zero);
}
void fcmeq(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(1, 0, 0, pd, pg, zn, zero);
}
void fcmne(const PRegH &pd, const _PReg &pg, const ZRegH &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(1, 1, 0, pd, pg, zn, zero);
}
void fcmne(const PRegS &pd, const _PReg &pg, const ZRegS &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(1, 1, 0, pd, pg, zn, zero);
}
void fcmne(const PRegD &pd, const _PReg &pg, const ZRegD &zn,
           const double zero) {
  XBYAK_SET_CODE_INFO();
  SveFpCompWithZero(1, 1, 0, pd, pg, zn, zero);
}
void fadda(const HReg &vdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpSerialReductPred(0, vdn, pg, zm);
}
void fadda(const SReg &vdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpSerialReductPred(0, vdn, pg, zm);
}
void fadda(const DReg &vdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpSerialReductPred(0, vdn, pg, zm);
}
void fadd(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(0, zd, zn, zm);
}
void fadd(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(0, zd, zn, zm);
}
void fadd(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(0, zd, zn, zm);
}
void fsub(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(1, zd, zn, zm);
}
void fsub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(1, zd, zn, zm);
}
void fsub(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(1, zd, zn, zm);
}
void fmul(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(2, zd, zn, zm);
}
void fmul(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(2, zd, zn, zm);
}
void fmul(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(2, zd, zn, zm);
}
void ftsmul(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(3, zd, zn, zm);
}
void ftsmul(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(3, zd, zn, zm);
}
void ftsmul(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(3, zd, zn, zm);
}
void frecps(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(6, zd, zn, zm);
}
void frecps(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(6, zd, zn, zm);
}
void frecps(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(6, zd, zn, zm);
}
void frsqrts(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(7, zd, zn, zm);
}
void frsqrts(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(7, zd, zn, zm);
}
void frsqrts(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticUnpred(7, zd, zn, zm);
}
void fadd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(0, zdn, pg, zm);
}
void fadd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(0, zdn, pg, zm);
}
void fadd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(0, zdn, pg, zm);
}
void fsub(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(1, zdn, pg, zm);
}
void fsub(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(1, zdn, pg, zm);
}
void fsub(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(1, zdn, pg, zm);
}
void fmul(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(2, zdn, pg, zm);
}
void fmul(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(2, zdn, pg, zm);
}
void fmul(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(2, zdn, pg, zm);
}
void fsubr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(3, zdn, pg, zm);
}
void fsubr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(3, zdn, pg, zm);
}
void fsubr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(3, zdn, pg, zm);
}
void fmaxnm(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(4, zdn, pg, zm);
}
void fmaxnm(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(4, zdn, pg, zm);
}
void fmaxnm(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(4, zdn, pg, zm);
}
void fminnm(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(5, zdn, pg, zm);
}
void fminnm(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(5, zdn, pg, zm);
}
void fminnm(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(5, zdn, pg, zm);
}
void fmax(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(6, zdn, pg, zm);
}
void fmax(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(6, zdn, pg, zm);
}
void fmax(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(6, zdn, pg, zm);
}
void fmin(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(7, zdn, pg, zm);
}
void fmin(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(7, zdn, pg, zm);
}
void fmin(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(7, zdn, pg, zm);
}
void fabd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(8, zdn, pg, zm);
}
void fabd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(8, zdn, pg, zm);
}
void fabd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(8, zdn, pg, zm);
}
void fscale(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(9, zdn, pg, zm);
}
void fscale(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(9, zdn, pg, zm);
}
void fscale(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(9, zdn, pg, zm);
}
void fmulx(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(10, zdn, pg, zm);
}
void fmulx(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(10, zdn, pg, zm);
}
void fmulx(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(10, zdn, pg, zm);
}
void fdivr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(12, zdn, pg, zm);
}
void fdivr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(12, zdn, pg, zm);
}
void fdivr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(12, zdn, pg, zm);
}
void fdiv(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(13, zdn, pg, zm);
}
void fdiv(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(13, zdn, pg, zm);
}
void fdiv(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticPred(13, zdn, pg, zm);
}
void fadd(const ZRegH &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(0, zdn, pg, ct);
}
void fadd(const ZRegS &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(0, zdn, pg, ct);
}
void fadd(const ZRegD &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(0, zdn, pg, ct);
}
void fsub(const ZRegH &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(1, zdn, pg, ct);
}
void fsub(const ZRegS &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(1, zdn, pg, ct);
}
void fsub(const ZRegD &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(1, zdn, pg, ct);
}
void fmul(const ZRegH &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(2, zdn, pg, ct);
}
void fmul(const ZRegS &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(2, zdn, pg, ct);
}
void fmul(const ZRegD &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(2, zdn, pg, ct);
}
void fsubr(const ZRegH &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(3, zdn, pg, ct);
}
void fsubr(const ZRegS &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(3, zdn, pg, ct);
}
void fsubr(const ZRegD &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(3, zdn, pg, ct);
}
void fmaxnm(const ZRegH &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(4, zdn, pg, ct);
}
void fmaxnm(const ZRegS &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(4, zdn, pg, ct);
}
void fmaxnm(const ZRegD &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(4, zdn, pg, ct);
}
void fminnm(const ZRegH &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(5, zdn, pg, ct);
}
void fminnm(const ZRegS &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(5, zdn, pg, ct);
}
void fminnm(const ZRegD &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(5, zdn, pg, ct);
}
void fmax(const ZRegH &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(6, zdn, pg, ct);
}
void fmax(const ZRegS &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(6, zdn, pg, ct);
}
void fmax(const ZRegD &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(6, zdn, pg, ct);
}
void fmin(const ZRegH &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(7, zdn, pg, ct);
}
void fmin(const ZRegS &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(7, zdn, pg, ct);
}
void fmin(const ZRegD &zdn, const _PReg &pg, const float ct) {
  XBYAK_SET_CODE_INFO();
  SveFpArithmeticImmPred(7, zdn, pg, ct);
}
void ftmad(const ZRegH &zdn, const ZRegH &zm, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveFpTrigMultAddCoef(zdn, zm, imm);
}
void ftmad(const ZRegS &zdn, const ZRegS &zm, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveFpTrigMultAddCoef(zdn, zm, imm);
}
void ftmad(const ZRegD &zdn, const ZRegD &zm, const uint32_t imm) {
  XBYAK_SET_CODE_INFO();
  SveFpTrigMultAddCoef(zdn, zm, imm);
}
void fcvt(const ZRegH &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtPrecision(2, 0, zd, pg, zn);
}
void fcvt(const ZRegS &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtPrecision(2, 1, zd, pg, zn);
}
void fcvt(const ZRegH &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtPrecision(3, 0, zd, pg, zn);
}
void fcvt(const ZRegD &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtPrecision(3, 1, zd, pg, zn);
}
void fcvt(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtPrecision(3, 2, zd, pg, zn);
}
void fcvt(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtPrecision(3, 3, zd, pg, zn);
}
void fcvtzs(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(1, 1, 0, zd, pg, zn);
}
void fcvtzu(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(1, 1, 1, zd, pg, zn);
}
void fcvtzs(const ZRegS &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(1, 2, 0, zd, pg, zn);
}
void fcvtzu(const ZRegS &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(1, 2, 1, zd, pg, zn);
}
void fcvtzs(const ZRegD &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(1, 3, 0, zd, pg, zn);
}
void fcvtzu(const ZRegD &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(1, 3, 1, zd, pg, zn);
}
void fcvtzs(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(2, 2, 0, zd, pg, zn);
}
void fcvtzu(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(2, 2, 1, zd, pg, zn);
}
void fcvtzs(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(3, 0, 0, zd, pg, zn);
}
void fcvtzu(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(3, 0, 1, zd, pg, zn);
}
void fcvtzs(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(3, 2, 0, zd, pg, zn);
}
void fcvtzu(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(3, 2, 1, zd, pg, zn);
}
void fcvtzs(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(3, 3, 0, zd, pg, zn);
}
void fcvtzu(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpCvtToInt(3, 3, 1, zd, pg, zn);
}
void frintn(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(0, zd, pg, zn);
}
void frintn(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(0, zd, pg, zn);
}
void frintn(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(0, zd, pg, zn);
}
void frintp(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(1, zd, pg, zn);
}
void frintp(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(1, zd, pg, zn);
}
void frintp(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(1, zd, pg, zn);
}
void frintm(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(2, zd, pg, zn);
}
void frintm(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(2, zd, pg, zn);
}
void frintm(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(2, zd, pg, zn);
}
void frintz(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(3, zd, pg, zn);
}
void frintz(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(3, zd, pg, zn);
}
void frintz(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(3, zd, pg, zn);
}
void frinta(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(4, zd, pg, zn);
}
void frinta(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(4, zd, pg, zn);
}
void frinta(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(4, zd, pg, zn);
}
void frintx(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(6, zd, pg, zn);
}
void frintx(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(6, zd, pg, zn);
}
void frintx(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(6, zd, pg, zn);
}
void frinti(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(7, zd, pg, zn);
}
void frinti(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(7, zd, pg, zn);
}
void frinti(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpRoundToIntegral(7, zd, pg, zn);
}
void frecpx(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpUnaryOp(0, zd, pg, zn);
}
void frecpx(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpUnaryOp(0, zd, pg, zn);
}
void frecpx(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpUnaryOp(0, zd, pg, zn);
}
void fsqrt(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpUnaryOp(1, zd, pg, zn);
}
void fsqrt(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpUnaryOp(1, zd, pg, zn);
}
void fsqrt(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveFpUnaryOp(1, zd, pg, zn);
}
void scvtf(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(1, 1, 0, zd, pg, zn);
}
void ucvtf(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(1, 1, 1, zd, pg, zn);
}
void scvtf(const ZRegH &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(1, 2, 0, zd, pg, zn);
}
void ucvtf(const ZRegH &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(1, 2, 1, zd, pg, zn);
}
void scvtf(const ZRegH &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(1, 3, 0, zd, pg, zn);
}
void ucvtf(const ZRegH &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(1, 3, 1, zd, pg, zn);
}
void scvtf(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(2, 2, 0, zd, pg, zn);
}
void ucvtf(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(2, 2, 1, zd, pg, zn);
}
void scvtf(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(3, 0, 0, zd, pg, zn);
}
void ucvtf(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(3, 0, 1, zd, pg, zn);
}
void scvtf(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(3, 2, 0, zd, pg, zn);
}
void ucvtf(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(3, 2, 1, zd, pg, zn);
}
void scvtf(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(3, 3, 0, zd, pg, zn);
}
void ucvtf(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) {
  XBYAK_SET_CODE_INFO();
  SveIntCvtToFp(3, 3, 1, zd, pg, zn);
}
void fcmge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 0, pd, pg, zn, zm);
}
void fcmge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 0, pd, pg, zn, zm);
}
void fcmge(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 0, pd, pg, zn, zm);
}
void fcmgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 1, pd, pg, zn, zm);
}
void fcmgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 1, pd, pg, zn, zm);
}
void fcmgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 1, pd, pg, zn, zm);
}
void fcmle(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 0, pd, pg, zm, zn);
}
void fcmle(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 0, pd, pg, zm, zn);
}
void fcmle(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 0, pd, pg, zm, zn);
}
void fcmlt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 1, pd, pg, zm, zn);
}
void fcmlt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 1, pd, pg, zm, zn);
}
void fcmlt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 0, 1, pd, pg, zm, zn);
}
void fcmeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 1, 0, pd, pg, zn, zm);
}
void fcmeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 1, 0, pd, pg, zn, zm);
}
void fcmeq(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 1, 0, pd, pg, zn, zm);
}
void fcmne(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 1, 1, pd, pg, zn, zm);
}
void fcmne(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 1, 1, pd, pg, zn, zm);
}
void fcmne(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(0, 1, 1, pd, pg, zn, zm);
}
void fcmuo(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 0, 0, pd, pg, zn, zm);
}
void fcmuo(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 0, 0, pd, pg, zn, zm);
}
void fcmuo(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 0, 0, pd, pg, zn, zm);
}
void facge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 0, 1, pd, pg, zn, zm);
}
void facge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 0, 1, pd, pg, zn, zm);
}
void facge(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 0, 1, pd, pg, zn, zm);
}
void facgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 1, 1, pd, pg, zn, zm);
}
void facgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 1, 1, pd, pg, zn, zm);
}
void facgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 1, 1, pd, pg, zn, zm);
}
void facle(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 0, 1, pd, pg, zm, zn);
}
void facle(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 0, 1, pd, pg, zm, zn);
}
void facle(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 0, 1, pd, pg, zm, zn);
}
void faclt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 1, 1, pd, pg, zm, zn);
}
void faclt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 1, 1, pd, pg, zm, zn);
}
void faclt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpCompVec(1, 1, 1, pd, pg, zm, zn);
}
void fmla(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(0, zda, pg, zn, zm);
}
void fmla(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(0, zda, pg, zn, zm);
}
void fmla(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(0, zda, pg, zn, zm);
}
void fmls(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(1, zda, pg, zn, zm);
}
void fmls(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(1, zda, pg, zn, zm);
}
void fmls(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(1, zda, pg, zn, zm);
}
void fnmla(const ZRegH &zda, const _PReg &pg, const ZRegH &zn,
           const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(2, zda, pg, zn, zm);
}
void fnmla(const ZRegS &zda, const _PReg &pg, const ZRegS &zn,
           const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(2, zda, pg, zn, zm);
}
void fnmla(const ZRegD &zda, const _PReg &pg, const ZRegD &zn,
           const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(2, zda, pg, zn, zm);
}
void fnmls(const ZRegH &zda, const _PReg &pg, const ZRegH &zn,
           const ZRegH &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(3, zda, pg, zn, zm);
}
void fnmls(const ZRegS &zda, const _PReg &pg, const ZRegS &zn,
           const ZRegS &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(3, zda, pg, zn, zm);
}
void fnmls(const ZRegD &zda, const _PReg &pg, const ZRegD &zn,
           const ZRegD &zm) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumAddend(3, zda, pg, zn, zm);
}
void fmad(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(0, zdn, pg, zm, za);
}
void fmad(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(0, zdn, pg, zm, za);
}
void fmad(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(0, zdn, pg, zm, za);
}
void fmsb(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(1, zdn, pg, zm, za);
}
void fmsb(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(1, zdn, pg, zm, za);
}
void fmsb(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(1, zdn, pg, zm, za);
}
void fnmad(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm,
           const ZRegH &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(2, zdn, pg, zm, za);
}
void fnmad(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm,
           const ZRegS &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(2, zdn, pg, zm, za);
}
void fnmad(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm,
           const ZRegD &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(2, zdn, pg, zm, za);
}
void fnmsb(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm,
           const ZRegH &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(3, zdn, pg, zm, za);
}
void fnmsb(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm,
           const ZRegS &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(3, zdn, pg, zm, za);
}
void fnmsb(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm,
           const ZRegD &za) {
  XBYAK_SET_CODE_INFO();
  SveFpMultAccumMulti(3, zdn, pg, zm, za);
}
void ld1sb(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(0, 0, 0, zt, pg, adr);
}
void ldff1sb(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(0, 0, 1, zt, pg, adr);
}
void ld1b(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(0, 1, 0, zt, pg, adr);
}
void ldff1b(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(0, 1, 1, zt, pg, adr);
}
void ld1sh(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(1, 0, 0, zt, pg, adr);
}
void ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(1, 0, 1, zt, pg, adr);
}
void ld1h(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(1, 1, 0, zt, pg, adr);
}
void ldff1h(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(1, 1, 1, zt, pg, adr);
}
void ld1w(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(2, 1, 0, zt, pg, adr);
}
void ldff1w(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdSc32U(2, 1, 1, zt, pg, adr);
}
void ld1sb(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(0, 0, 0, zt, pg, adr);
}
void ldff1sb(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(0, 0, 1, zt, pg, adr);
}
void ld1b(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(0, 1, 0, zt, pg, adr);
}
void ldff1b(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(0, 1, 1, zt, pg, adr);
}
void ld1sh(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(1, 0, 0, zt, pg, adr);
}
void ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(1, 0, 1, zt, pg, adr);
}
void ld1h(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(1, 1, 0, zt, pg, adr);
}
void ldff1h(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(1, 1, 1, zt, pg, adr);
}
void ld1w(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(2, 1, 0, zt, pg, adr);
}
void ldff1w(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdVecImm(2, 1, 1, zt, pg, adr);
}
void ld1sh(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdHSc32S(0, 0, zt, pg, adr);
}
void ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdHSc32S(0, 1, zt, pg, adr);
}
void ld1h(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdHSc32S(1, 0, zt, pg, adr);
}
void ldff1h(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdHSc32S(1, 1, zt, pg, adr);
}
void ld1w(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdWSc32S(1, 0, zt, pg, adr);
}
void ldff1w(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherLdWSc32S(1, 1, zt, pg, adr);
}
void prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherPfSc32S(prfop_sve, 0, pg, adr);
}
void prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherPfSc32S(prfop_sve, 1, pg, adr);
}
void prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherPfSc32S(prfop_sve, 2, pg, adr);
}
void prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherPfSc32S(prfop_sve, 3, pg, adr);
}
void prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherPfVecImm(prfop_sve, 0, pg, adr);
}
void prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherPfVecImm(prfop_sve, 1, pg, adr);
}
void prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherPfVecImm(prfop_sve, 2, pg, adr);
}
void prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32GatherPfVecImm(prfop_sve, 3, pg, adr);
}
void prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScImm(prfop_sve, 0, pg, adr);
}
void prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScImm(prfop_sve, 0, pg, adr);
}
void prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScImm(prfop_sve, 1, pg, adr);
}
void prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScImm(prfop_sve, 1, pg, adr);
}
void prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScImm(prfop_sve, 2, pg, adr);
}
void prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScImm(prfop_sve, 2, pg, adr);
}
void prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScImm(prfop_sve, 3, pg, adr);
}
void prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScImm(prfop_sve, 3, pg, adr);
}
void prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScSc(prfop_sve, 0, pg, adr);
}
void prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScSc(prfop_sve, 1, pg, adr);
}
void prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScSc(prfop_sve, 2, pg, adr);
}
void prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ContiPfScSc(prfop_sve, 3, pg, adr);
}
void ld1rb(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(0, 0, zt, pg, adr);
}
void ld1rb(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(0, 0, zt, pg, adr);
}
void ld1rb(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(0, 1, zt, pg, adr);
}
void ld1rb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(0, 1, zt, pg, adr);
}
void ld1rb(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(0, 2, zt, pg, adr);
}
void ld1rb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(0, 2, zt, pg, adr);
}
void ld1rb(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(0, 3, zt, pg, adr);
}
void ld1rb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(0, 3, zt, pg, adr);
}
void ld1rsw(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(1, 0, zt, pg, adr);
}
void ld1rsw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(1, 0, zt, pg, adr);
}
void ld1rh(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(1, 1, zt, pg, adr);
}
void ld1rh(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(1, 1, zt, pg, adr);
}
void ld1rh(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(1, 2, zt, pg, adr);
}
void ld1rh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(1, 2, zt, pg, adr);
}
void ld1rh(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(1, 3, zt, pg, adr);
}
void ld1rh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(1, 3, zt, pg, adr);
}
void ld1rsh(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(2, 0, zt, pg, adr);
}
void ld1rsh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(2, 0, zt, pg, adr);
}
void ld1rsh(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(2, 1, zt, pg, adr);
}
void ld1rsh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(2, 1, zt, pg, adr);
}
void ld1rw(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(2, 2, zt, pg, adr);
}
void ld1rw(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(2, 2, zt, pg, adr);
}
void ld1rw(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(2, 3, zt, pg, adr);
}
void ld1rw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(2, 3, zt, pg, adr);
}
void ld1rsb(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(3, 0, zt, pg, adr);
}
void ld1rsb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(3, 0, zt, pg, adr);
}
void ld1rsb(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(3, 1, zt, pg, adr);
}
void ld1rsb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(3, 1, zt, pg, adr);
}
void ld1rsb(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(3, 2, zt, pg, adr);
}
void ld1rsb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(3, 2, zt, pg, adr);
}
void ld1rd(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(3, 3, zt, pg, adr);
}
void ld1rd(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadAndBcElem(3, 3, zt, pg, adr);
}
void ldr(const _PReg &pt, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadPredReg(pt, adr);
}
void ldr(const _PReg &pt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadPredReg(pt, adr);
}
void ldr(const ZReg &zt, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadPredVec(zt, adr);
}
void ldr(const ZReg &zt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLoadPredVec(zt, adr);
}
void ldff1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(0, zt, pg, adr);
}
void ldff1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(0, zt, pg, adr);
}
void ldff1b(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(1, zt, pg, adr);
}
void ldff1b(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(1, zt, pg, adr);
}
void ldff1b(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(2, zt, pg, adr);
}
void ldff1b(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(2, zt, pg, adr);
}
void ldff1b(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(3, zt, pg, adr);
}
void ldff1b(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(3, zt, pg, adr);
}
void ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(4, zt, pg, adr);
}
void ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(4, zt, pg, adr);
}
void ldff1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(5, zt, pg, adr);
}
void ldff1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(5, zt, pg, adr);
}
void ldff1h(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(6, zt, pg, adr);
}
void ldff1h(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(6, zt, pg, adr);
}
void ldff1h(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(7, zt, pg, adr);
}
void ldff1h(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(7, zt, pg, adr);
}
void ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(8, zt, pg, adr);
}
void ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(8, zt, pg, adr);
}
void ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(9, zt, pg, adr);
}
void ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(9, zt, pg, adr);
}
void ldff1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(10, zt, pg, adr);
}
void ldff1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(10, zt, pg, adr);
}
void ldff1w(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(11, zt, pg, adr);
}
void ldff1w(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(11, zt, pg, adr);
}
void ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(12, zt, pg, adr);
}
void ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(12, zt, pg, adr);
}
void ldff1sb(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(13, zt, pg, adr);
}
void ldff1sb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(13, zt, pg, adr);
}
void ldff1sb(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(14, zt, pg, adr);
}
void ldff1sb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(14, zt, pg, adr);
}
void ldff1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(15, zt, pg, adr);
}
void ldff1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiFFLdScSc(15, zt, pg, adr);
}
void ld1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(0, zt, pg, adr);
}
void ld1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(0, zt, pg, adr);
}
void ld1b(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(1, zt, pg, adr);
}
void ld1b(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(1, zt, pg, adr);
}
void ld1b(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(2, zt, pg, adr);
}
void ld1b(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(2, zt, pg, adr);
}
void ld1b(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(3, zt, pg, adr);
}
void ld1b(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(3, zt, pg, adr);
}
void ld1sw(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(4, zt, pg, adr);
}
void ld1sw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(4, zt, pg, adr);
}
void ld1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(5, zt, pg, adr);
}
void ld1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(5, zt, pg, adr);
}
void ld1h(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(6, zt, pg, adr);
}
void ld1h(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(6, zt, pg, adr);
}
void ld1h(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(7, zt, pg, adr);
}
void ld1h(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(7, zt, pg, adr);
}
void ld1sh(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(8, zt, pg, adr);
}
void ld1sh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(8, zt, pg, adr);
}
void ld1sh(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(9, zt, pg, adr);
}
void ld1sh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(9, zt, pg, adr);
}
void ld1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(10, zt, pg, adr);
}
void ld1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(10, zt, pg, adr);
}
void ld1w(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(11, zt, pg, adr);
}
void ld1w(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(11, zt, pg, adr);
}
void ld1sb(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(12, zt, pg, adr);
}
void ld1sb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(12, zt, pg, adr);
}
void ld1sb(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(13, zt, pg, adr);
}
void ld1sb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(13, zt, pg, adr);
}
void ld1sb(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(14, zt, pg, adr);
}
void ld1sb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(14, zt, pg, adr);
}
void ld1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(15, zt, pg, adr);
}
void ld1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScImm(15, zt, pg, adr);
}
void ld1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(0, zt, pg, adr);
}
void ld1b(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(1, zt, pg, adr);
}
void ld1b(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(2, zt, pg, adr);
}
void ld1b(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(3, zt, pg, adr);
}
void ld1sw(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(4, zt, pg, adr);
}
void ld1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(5, zt, pg, adr);
}
void ld1h(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(6, zt, pg, adr);
}
void ld1h(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(7, zt, pg, adr);
}
void ld1sh(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(8, zt, pg, adr);
}
void ld1sh(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(9, zt, pg, adr);
}
void ld1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(10, zt, pg, adr);
}
void ld1w(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(11, zt, pg, adr);
}
void ld1sb(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(12, zt, pg, adr);
}
void ld1sb(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(13, zt, pg, adr);
}
void ld1sb(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(14, zt, pg, adr);
}
void ld1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiLdScSc(15, zt, pg, adr);
}
void ldnf1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(0, zt, pg, adr);
}
void ldnf1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(0, zt, pg, adr);
}
void ldnf1b(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(1, zt, pg, adr);
}
void ldnf1b(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(1, zt, pg, adr);
}
void ldnf1b(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(2, zt, pg, adr);
}
void ldnf1b(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(2, zt, pg, adr);
}
void ldnf1b(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(3, zt, pg, adr);
}
void ldnf1b(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(3, zt, pg, adr);
}
void ldnf1sw(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(4, zt, pg, adr);
}
void ldnf1sw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(4, zt, pg, adr);
}
void ldnf1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(5, zt, pg, adr);
}
void ldnf1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(5, zt, pg, adr);
}
void ldnf1h(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(6, zt, pg, adr);
}
void ldnf1h(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(6, zt, pg, adr);
}
void ldnf1h(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(7, zt, pg, adr);
}
void ldnf1h(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(7, zt, pg, adr);
}
void ldnf1sh(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(8, zt, pg, adr);
}
void ldnf1sh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(8, zt, pg, adr);
}
void ldnf1sh(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(9, zt, pg, adr);
}
void ldnf1sh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(9, zt, pg, adr);
}
void ldnf1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(10, zt, pg, adr);
}
void ldnf1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(10, zt, pg, adr);
}
void ldnf1w(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(11, zt, pg, adr);
}
void ldnf1w(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(11, zt, pg, adr);
}
void ldnf1sb(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(12, zt, pg, adr);
}
void ldnf1sb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(12, zt, pg, adr);
}
void ldnf1sb(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(13, zt, pg, adr);
}
void ldnf1sb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(13, zt, pg, adr);
}
void ldnf1sb(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(14, zt, pg, adr);
}
void ldnf1sb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(14, zt, pg, adr);
}
void ldnf1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(15, zt, pg, adr);
}
void ldnf1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNFLdScImm(15, zt, pg, adr);
}
void ldnt1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScImm(0, zt, pg, adr);
}
void ldnt1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScImm(0, zt, pg, adr);
}
void ldnt1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScImm(1, zt, pg, adr);
}
void ldnt1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScImm(1, zt, pg, adr);
}
void ldnt1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScImm(2, zt, pg, adr);
}
void ldnt1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScImm(2, zt, pg, adr);
}
void ldnt1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScImm(3, zt, pg, adr);
}
void ldnt1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScImm(3, zt, pg, adr);
}
void ldnt1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScSc(0, zt, pg, adr);
}
void ldnt1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScSc(1, zt, pg, adr);
}
void ldnt1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScSc(2, zt, pg, adr);
}
void ldnt1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTLdScSc(3, zt, pg, adr);
}
void ld1rqb(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScImm(0, 0, zt, pg, adr);
}
void ld1rqb(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScImm(0, 0, zt, pg, adr);
}
void ld1rqh(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScImm(1, 0, zt, pg, adr);
}
void ld1rqh(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScImm(1, 0, zt, pg, adr);
}
void ld1rqw(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScImm(2, 0, zt, pg, adr);
}
void ld1rqw(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScImm(2, 0, zt, pg, adr);
}
void ld1rqd(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScImm(3, 0, zt, pg, adr);
}
void ld1rqd(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScImm(3, 0, zt, pg, adr);
}
void ld1rqb(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScSc(0, 0, zt, pg, adr);
}
void ld1rqh(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScSc(1, 0, zt, pg, adr);
}
void ld1rqw(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScSc(2, 0, zt, pg, adr);
}
void ld1rqd(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdBcQuadScSc(3, 0, zt, pg, adr);
}
void ld2b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(0, 1, zt, pg, adr);
}
void ld2b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(0, 1, zt, pg, adr);
}
void ld3b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(0, 2, zt, pg, adr);
}
void ld3b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(0, 2, zt, pg, adr);
}
void ld4b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(0, 3, zt, pg, adr);
}
void ld4b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(0, 3, zt, pg, adr);
}
void ld2h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(1, 1, zt, pg, adr);
}
void ld2h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(1, 1, zt, pg, adr);
}
void ld3h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(1, 2, zt, pg, adr);
}
void ld3h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(1, 2, zt, pg, adr);
}
void ld4h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(1, 3, zt, pg, adr);
}
void ld4h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(1, 3, zt, pg, adr);
}
void ld2w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(2, 1, zt, pg, adr);
}
void ld2w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(2, 1, zt, pg, adr);
}
void ld3w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(2, 2, zt, pg, adr);
}
void ld3w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(2, 2, zt, pg, adr);
}
void ld4w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(2, 3, zt, pg, adr);
}
void ld4w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(2, 3, zt, pg, adr);
}
void ld2d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(3, 1, zt, pg, adr);
}
void ld2d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(3, 1, zt, pg, adr);
}
void ld3d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(3, 2, zt, pg, adr);
}
void ld3d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(3, 2, zt, pg, adr);
}
void ld4d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(3, 3, zt, pg, adr);
}
void ld4d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScImm(3, 3, zt, pg, adr);
}
void ld2b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(0, 1, zt, pg, adr);
}
void ld3b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(0, 2, zt, pg, adr);
}
void ld4b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(0, 3, zt, pg, adr);
}
void ld2h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(1, 1, zt, pg, adr);
}
void ld3h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(1, 2, zt, pg, adr);
}
void ld4h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(1, 3, zt, pg, adr);
}
void ld2w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(2, 1, zt, pg, adr);
}
void ld3w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(2, 2, zt, pg, adr);
}
void ld4w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(2, 3, zt, pg, adr);
}
void ld2d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(3, 1, zt, pg, adr);
}
void ld3d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(3, 2, zt, pg, adr);
}
void ld4d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveLdMultiStructScSc(3, 3, zt, pg, adr);
}
void ld1sh(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(1, 0, 0, zt, pg, adr);
}
void ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(1, 0, 1, zt, pg, adr);
}
void ld1h(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(1, 1, 0, zt, pg, adr);
}
void ldff1h(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(1, 1, 1, zt, pg, adr);
}
void ld1sw(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(2, 0, 0, zt, pg, adr);
}
void ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(2, 0, 1, zt, pg, adr);
}
void ld1w(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(2, 1, 0, zt, pg, adr);
}
void ldff1w(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(2, 1, 1, zt, pg, adr);
}
void ld1d(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(3, 1, 0, zt, pg, adr);
}
void ldff1d(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32US(3, 1, 1, zt, pg, adr);
}
void ld1sh(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(1, 0, 0, zt, pg, adr);
}
void ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(1, 0, 1, zt, pg, adr);
}
void ld1h(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(1, 1, 0, zt, pg, adr);
}
void ldff1h(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(1, 1, 1, zt, pg, adr);
}
void ld1sw(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(2, 0, 0, zt, pg, adr);
}
void ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(2, 0, 1, zt, pg, adr);
}
void ld1w(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(2, 1, 0, zt, pg, adr);
}
void ldff1w(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(2, 1, 1, zt, pg, adr);
}
void ld1d(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(3, 1, 0, zt, pg, adr);
}
void ldff1d(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64S(3, 1, 1, zt, pg, adr);
}
void ld1sb(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(0, 0, 0, zt, pg, adr);
}
void ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(0, 0, 1, zt, pg, adr);
}
void ld1b(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(0, 1, 0, zt, pg, adr);
}
void ldff1b(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(0, 1, 1, zt, pg, adr);
}
void ld1sh(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(1, 0, 0, zt, pg, adr);
}
void ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(1, 0, 1, zt, pg, adr);
}
void ld1h(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(1, 1, 0, zt, pg, adr);
}
void ldff1h(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(1, 1, 1, zt, pg, adr);
}
void ld1sw(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(2, 0, 0, zt, pg, adr);
}
void ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(2, 0, 1, zt, pg, adr);
}
void ld1w(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(2, 1, 0, zt, pg, adr);
}
void ldff1w(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(2, 1, 1, zt, pg, adr);
}
void ld1d(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(3, 1, 0, zt, pg, adr);
}
void ldff1d(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc64U(3, 1, 1, zt, pg, adr);
}
void ld1sb(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(0, 0, 0, zt, pg, adr);
}
void ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(0, 0, 1, zt, pg, adr);
}
void ld1b(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(0, 1, 0, zt, pg, adr);
}
void ldff1b(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(0, 1, 1, zt, pg, adr);
}
void ld1sh(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(1, 0, 0, zt, pg, adr);
}
void ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(1, 0, 1, zt, pg, adr);
}
void ld1h(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(1, 1, 0, zt, pg, adr);
}
void ldff1h(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(1, 1, 1, zt, pg, adr);
}
void ld1sw(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(2, 0, 0, zt, pg, adr);
}
void ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(2, 0, 1, zt, pg, adr);
}
void ld1w(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(2, 1, 0, zt, pg, adr);
}
void ldff1w(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(2, 1, 1, zt, pg, adr);
}
void ld1d(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(3, 1, 0, zt, pg, adr);
}
void ldff1d(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdSc32UU(3, 1, 1, zt, pg, adr);
}
void ld1sb(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(0, 0, 0, zt, pg, adr);
}
void ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(0, 0, 1, zt, pg, adr);
}
void ld1b(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(0, 1, 0, zt, pg, adr);
}
void ldff1b(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(0, 1, 1, zt, pg, adr);
}
void ld1sh(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(1, 0, 0, zt, pg, adr);
}
void ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(1, 0, 1, zt, pg, adr);
}
void ld1h(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(1, 1, 0, zt, pg, adr);
}
void ldff1h(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(1, 1, 1, zt, pg, adr);
}
void ld1sw(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(2, 0, 0, zt, pg, adr);
}
void ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(2, 0, 1, zt, pg, adr);
}
void ld1w(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(2, 1, 0, zt, pg, adr);
}
void ldff1w(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(2, 1, 1, zt, pg, adr);
}
void ld1d(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(3, 1, 0, zt, pg, adr);
}
void ldff1d(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherLdVecImm(3, 1, 1, zt, pg, adr);
}
void prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfSc64S(prfop_sve, 0, pg, adr);
}
void prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfSc64S(prfop_sve, 1, pg, adr);
}
void prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfSc64S(prfop_sve, 2, pg, adr);
}
void prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfSc64S(prfop_sve, 3, pg, adr);
}
void prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfSc32US(prfop_sve, 0, pg, adr);
}
void prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfSc32US(prfop_sve, 1, pg, adr);
}
void prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfSc32US(prfop_sve, 2, pg, adr);
}
void prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfSc32US(prfop_sve, 3, pg, adr);
}
void prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfVecImm(prfop_sve, 0, pg, adr);
}
void prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfVecImm(prfop_sve, 1, pg, adr);
}
void prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfVecImm(prfop_sve, 2, pg, adr);
}
void prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64GatherPfVecImm(prfop_sve, 3, pg, adr);
}
void st1h(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ScatterStSc32S(1, zt, pg, adr);
}
void st1w(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ScatterStSc32S(2, zt, pg, adr);
}
void st1b(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ScatterStSc32U(0, zt, pg, adr);
}
void st1h(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ScatterStSc32U(1, zt, pg, adr);
}
void st1w(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ScatterStSc32U(2, zt, pg, adr);
}
void st1b(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ScatterStVecImm(0, zt, pg, adr);
}
void st1h(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ScatterStVecImm(1, zt, pg, adr);
}
void st1w(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve32ScatterStVecImm(2, zt, pg, adr);
}
void st1h(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc64S(1, zt, pg, adr);
}
void st1w(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc64S(2, zt, pg, adr);
}
void st1d(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc64S(3, zt, pg, adr);
}
void st1b(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc64U(0, zt, pg, adr);
}
void st1h(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc64U(1, zt, pg, adr);
}
void st1w(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc64U(2, zt, pg, adr);
}
void st1d(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc64U(3, zt, pg, adr);
}
void st1h(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc32US(1, zt, pg, adr);
}
void st1w(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc32US(2, zt, pg, adr);
}
void st1d(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc32US(3, zt, pg, adr);
}
void st1b(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc32UU(0, zt, pg, adr);
}
void st1h(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc32UU(1, zt, pg, adr);
}
void st1w(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc32UU(2, zt, pg, adr);
}
void st1d(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStSc32UU(3, zt, pg, adr);
}
void st1b(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStVecImm(0, zt, pg, adr);
}
void st1h(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStVecImm(1, zt, pg, adr);
}
void st1w(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStVecImm(2, zt, pg, adr);
}
void st1d(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  XBYAK_SET_CODE_INFO();
  Sve64ScatterStVecImm(3, zt, pg, adr);
}
void stnt1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScImm(0, zt, pg, adr);
}
void stnt1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScImm(0, zt, pg, adr);
}
void stnt1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScImm(1, zt, pg, adr);
}
void stnt1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScImm(1, zt, pg, adr);
}
void stnt1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScImm(2, zt, pg, adr);
}
void stnt1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScImm(2, zt, pg, adr);
}
void stnt1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScImm(3, zt, pg, adr);
}
void stnt1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScImm(3, zt, pg, adr);
}
void stnt1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScSc(0, zt, pg, adr);
}
void stnt1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScSc(1, zt, pg, adr);
}
void stnt1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScSc(2, zt, pg, adr);
}
void stnt1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiNTStScSc(3, zt, pg, adr);
}
void st1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(0, zt, pg, adr);
}
void st1b(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(0, zt, pg, adr);
}
void st1b(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(0, zt, pg, adr);
}
void st1b(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(0, zt, pg, adr);
}
void st1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(0, zt, pg, adr);
}
void st1b(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(0, zt, pg, adr);
}
void st1b(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(0, zt, pg, adr);
}
void st1b(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(0, zt, pg, adr);
}
void st1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(1, zt, pg, adr);
}
void st1h(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(1, zt, pg, adr);
}
void st1h(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(1, zt, pg, adr);
}
void st1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(1, zt, pg, adr);
}
void st1h(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(1, zt, pg, adr);
}
void st1h(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(1, zt, pg, adr);
}
void st1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(2, zt, pg, adr);
}
void st1w(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(2, zt, pg, adr);
}
void st1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(2, zt, pg, adr);
}
void st1w(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(2, zt, pg, adr);
}
void st1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(3, zt, pg, adr);
}
void st1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScImm(3, zt, pg, adr);
}
void st1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(0, zt, pg, adr);
}
void st1b(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(0, zt, pg, adr);
}
void st1b(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(0, zt, pg, adr);
}
void st1b(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(0, zt, pg, adr);
}
void st1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(1, zt, pg, adr);
}
void st1h(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(1, zt, pg, adr);
}
void st1h(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(1, zt, pg, adr);
}
void st1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(2, zt, pg, adr);
}
void st1w(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(2, zt, pg, adr);
}
void st1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveContiStScSc(3, zt, pg, adr);
}
void st2b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(0, 1, zt, pg, adr);
}
void st2b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(0, 1, zt, pg, adr);
}
void st3b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(0, 2, zt, pg, adr);
}
void st3b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(0, 2, zt, pg, adr);
}
void st4b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(0, 3, zt, pg, adr);
}
void st4b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(0, 3, zt, pg, adr);
}
void st2h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(1, 1, zt, pg, adr);
}
void st2h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(1, 1, zt, pg, adr);
}
void st3h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(1, 2, zt, pg, adr);
}
void st3h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(1, 2, zt, pg, adr);
}
void st4h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(1, 3, zt, pg, adr);
}
void st4h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(1, 3, zt, pg, adr);
}
void st2w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(2, 1, zt, pg, adr);
}
void st2w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(2, 1, zt, pg, adr);
}
void st3w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(2, 2, zt, pg, adr);
}
void st3w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(2, 2, zt, pg, adr);
}
void st4w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(2, 3, zt, pg, adr);
}
void st4w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(2, 3, zt, pg, adr);
}
void st2d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(3, 1, zt, pg, adr);
}
void st2d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(3, 1, zt, pg, adr);
}
void st3d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(3, 2, zt, pg, adr);
}
void st3d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(3, 2, zt, pg, adr);
}
void st4d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(3, 3, zt, pg, adr);
}
void st4d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScImm(3, 3, zt, pg, adr);
}
void st2b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(0, 1, zt, pg, adr);
}
void st3b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(0, 2, zt, pg, adr);
}
void st4b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(0, 3, zt, pg, adr);
}
void st2h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(1, 1, zt, pg, adr);
}
void st3h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(1, 2, zt, pg, adr);
}
void st4h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(1, 3, zt, pg, adr);
}
void st2w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(2, 1, zt, pg, adr);
}
void st3w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(2, 2, zt, pg, adr);
}
void st4w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(2, 3, zt, pg, adr);
}
void st2d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(3, 1, zt, pg, adr);
}
void st3d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(3, 2, zt, pg, adr);
}
void st4d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) {
  XBYAK_SET_CODE_INFO();
  SveStMultiStructScSc(3, 3, zt, pg, adr);
}
void str(const _PReg &pt, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStorePredReg(pt, adr);
}
void str(const _PReg &pt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStorePredReg(pt, adr);
}
void str(const ZReg &zt, const AdrScImm &adr) {
  XBYAK_SET_CODE_INFO();
  SveStorePredVec(zt, adr);
}
void str(const ZReg &zt, const AdrNoOfs &adr) {
  XBYAK_SET_CODE_INFO();
  SveStorePredVec(zt, adr);
}
#undef XBYAK_SET_CODE_INFO
