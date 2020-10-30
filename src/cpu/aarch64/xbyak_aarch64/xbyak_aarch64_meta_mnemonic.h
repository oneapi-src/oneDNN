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
template <typename T, typename std::enable_if<std::is_unsigned<T>::value,
                                              std::nullptr_t>::type = nullptr>
void add_imm(const XReg &dst, const XReg &src, T imm, const XReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  uint64_t bit_ptn = static_cast<uint64_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  const uint64_t IMM12_MASK = ~uint64_t(0xfff);
  if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
    add(dst, src, static_cast<uint32_t>(imm & 0xfff));
    return;
  }

  mov_imm(tmp, imm);
  add(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value,
                                              std::nullptr_t>::type = nullptr>
void add_imm(const XReg &dst, const XReg &src, T imm, const XReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  /* Sign bit must be extended. */
  int64_t bit_ptn = static_cast<int64_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  if (imm >= 0) {
    const uint64_t IMM12_MASK = ~uint64_t(0xfff);
    if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
      add(dst, src, static_cast<uint32_t>(imm & 0xfff));
      return;
    }
  }

  mov_imm(tmp, imm);
  add(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value,
                                              std::nullptr_t>::type = nullptr>
void add_imm(const WReg &dst, const WReg &src, T imm, const WReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  uint32_t bit_ptn = static_cast<uint32_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  const uint32_t IMM12_MASK = ~uint32_t(0xfff);
  if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
    add(dst, src, static_cast<uint32_t>(imm & 0xfff));
    return;
  }

  mov_imm(tmp, imm);
  add(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value,
                                              std::nullptr_t>::type = nullptr>
void add_imm(const WReg &dst, const WReg &src, T imm, const WReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  /* Sign bit must be extended. */
  int32_t bit_ptn = static_cast<int32_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  if (imm >= 0) {
    const uint32_t IMM12_MASK = ~uint32_t(0xfff);
    if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
      add(dst, src, static_cast<uint32_t>(imm & 0xfff));
      return;
    }
  }

  mov_imm(tmp, imm);
  add(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value,
                                              std::nullptr_t>::type = nullptr>
void sub_imm(const XReg &dst, const XReg &src, T imm, const XReg &tmp) {
  /* This sub_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  uint64_t bit_ptn = static_cast<uint64_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  const uint64_t IMM12_MASK = ~uint64_t(0xfff);
  if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
    sub(dst, src, static_cast<uint32_t>(imm & 0xfff));
    return;
  }

  mov_imm(tmp, imm);
  sub(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value,
                                              std::nullptr_t>::type = nullptr>
void sub_imm(const XReg &dst, const XReg &src, T imm, const XReg &tmp) {
  /* This sub_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  /* Sign bit must be extended. */
  int64_t bit_ptn = static_cast<int64_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  if (imm >= 0) {
    const uint64_t IMM12_MASK = ~uint64_t(0xfff);
    if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
      sub(dst, src, static_cast<uint32_t>(imm & 0xfff));
      return;
    }
  }

  mov_imm(tmp, imm);
  sub(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value,
                                              std::nullptr_t>::type = nullptr>
void sub_imm(const WReg &dst, const WReg &src, T imm, const WReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  uint32_t bit_ptn = static_cast<uint32_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  const uint32_t IMM12_MASK = ~uint32_t(0xfff);
  if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
    sub(dst, src, static_cast<uint32_t>(imm & 0xfff));
    return;
  }

  mov_imm(tmp, imm);
  sub(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value,
                                              std::nullptr_t>::type = nullptr>
void sub_imm(const WReg &dst, const WReg &src, T imm, const WReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  /* Sign bit must be extended. */
  int32_t bit_ptn = static_cast<int32_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  if (imm >= 0) {
    const uint32_t IMM12_MASK = ~uint32_t(0xfff);
    if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
      sub(dst, src, static_cast<uint32_t>(imm & 0xfff));
      return;
    }
  }

  mov_imm(tmp, imm);
  sub(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value,
                                              std::nullptr_t>::type = nullptr>
void adds_imm(const XReg &dst, const XReg &src, T imm, const XReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  uint64_t bit_ptn = static_cast<uint64_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  const uint64_t IMM12_MASK = ~uint64_t(0xfff);
  if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
    adds(dst, src, static_cast<uint32_t>(imm & 0xfff));
    return;
  }

  mov_imm(tmp, imm);
  adds(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value,
                                              std::nullptr_t>::type = nullptr>
void adds_imm(const XReg &dst, const XReg &src, T imm, const XReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  /* Sign bit must be extended. */
  int64_t bit_ptn = static_cast<int64_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  if (imm >= 0) {
    const uint64_t IMM12_MASK = ~uint64_t(0xfff);
    if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
      adds(dst, src, static_cast<uint32_t>(imm & 0xfff));
      return;
    }
  }

  mov_imm(tmp, imm);
  adds(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value,
                                              std::nullptr_t>::type = nullptr>
void adds_imm(const WReg &dst, const WReg &src, T imm, const WReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  uint32_t bit_ptn = static_cast<uint32_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  const uint32_t IMM12_MASK = ~uint32_t(0xfff);
  if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
    adds(dst, src, static_cast<uint32_t>(imm & 0xfff));
    return;
  }

  mov_imm(tmp, imm);
  adds(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value,
                                              std::nullptr_t>::type = nullptr>
void adds_imm(const WReg &dst, const WReg &src, T imm, const WReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  /* Sign bit must be extended. */
  int32_t bit_ptn = static_cast<int32_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  if (imm >= 0) {
    const uint32_t IMM12_MASK = ~uint32_t(0xfff);
    if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
      adds(dst, src, static_cast<uint32_t>(imm & 0xfff));
      return;
    }
  }

  mov_imm(tmp, imm);
  adds(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value,
                                              std::nullptr_t>::type = nullptr>
void subs_imm(const XReg &dst, const XReg &src, T imm, const XReg &tmp) {
  /* This sub_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  uint64_t bit_ptn = static_cast<uint64_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  const uint64_t IMM12_MASK = ~uint64_t(0xfff);
  if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
    subs(dst, src, static_cast<uint32_t>(imm & 0xfff));
    return;
  }

  mov_imm(tmp, imm);
  subs(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value,
                                              std::nullptr_t>::type = nullptr>
void subs_imm(const XReg &dst, const XReg &src, T imm, const XReg &tmp) {
  /* This sub_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  /* Sign bit must be extended. */
  int64_t bit_ptn = static_cast<int64_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  if (imm >= 0) {
    const uint64_t IMM12_MASK = ~uint64_t(0xfff);
    if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
      subs(dst, src, static_cast<uint32_t>(imm & 0xfff));
      return;
    }
  }

  mov_imm(tmp, imm);
  subs(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value,
                                              std::nullptr_t>::type = nullptr>
void subs_imm(const WReg &dst, const WReg &src, T imm, const WReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  uint32_t bit_ptn = static_cast<uint32_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  const uint32_t IMM12_MASK = ~uint32_t(0xfff);
  if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
    subs(dst, src, static_cast<uint32_t>(imm & 0xfff));
    return;
  }

  mov_imm(tmp, imm);
  subs(dst, src, tmp);

  return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value,
                                              std::nullptr_t>::type = nullptr>
void subs_imm(const WReg &dst, const WReg &src, T imm, const WReg &tmp) {
  /* This add_imm function allows dst == src,
   but tmp must be different from src */
  assert(src.getIdx() != tmp.getIdx());

  /* Sign bit must be extended. */
  int32_t bit_ptn = static_cast<int32_t>(imm);

  /* ADD(immediate) supports unsigned imm12 */
  if (imm >= 0) {
    const uint32_t IMM12_MASK = ~uint32_t(0xfff);
    if ((bit_ptn & IMM12_MASK) == 0) { // <= 4095
      subs(dst, src, static_cast<uint32_t>(imm & 0xfff));
      return;
    }
  }

  mov_imm(tmp, imm);
  subs(dst, src, tmp);

  return;
}

#define UNUSED_PARAM(x) ((void)(x))
template <typename T> void mov_imm(const XReg &dst, T imm) {
  bool flag = false;
  uint64_t bit_ptn = static_cast<uint64_t>(imm);

  if (imm == 0) {
    mov(dst, 0);
    return;
  }

  if (bit_ptn == ~uint64_t(0)) {
    movn(dst, 0);
    return;
  }

  if (isBitMask(bit_ptn)) {
    mov(dst, imm);
    return;
  }

  for (int i = 0; i < 4; i++) {
    uint64_t tag_bit = (bit_ptn >> (16 * i)) & 0xFFFF;
    if (tag_bit) {
      if (flag == false) {
        movz(dst, tag_bit, 16 * i);
        flag = true;
      } else {
        movk(dst, tag_bit, 16 * i);
      }
    }
  }

  return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value,
                                              std::nullptr_t>::type = nullptr>
void mov_imm(const WReg &dst, T imm) {
  bool flag = false;
  uint32_t bit_ptn = static_cast<uint32_t>(imm);

  if (imm == 0) {
    mov(dst, 0);
    return;
  }

  if (uint64_t(0xFFFFFFFF) < imm && imm < uint64_t(0xFFFFFFFF80000000)) {
    throw Error(ERR_ILLEGAL_IMM_RANGE, genErrMsg());
  }

  if (isBitMask(imm)) {
    mov(dst, imm);
    return;
  }

  for (int i = 0; i < 2; i++) {
    if (bit_ptn & (0xFFFF << 16 * i)) {
      if (flag == false) {
        movz(dst, (bit_ptn >> (16 * i)) & 0xFFFF, 16 * i);
        flag = true;
      } else {
        movk(dst, (bit_ptn >> (16 * i)) & 0xFFFF, 16 * i);
      }
    }
  }

  return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value,
                                              std::nullptr_t>::type = nullptr>
void mov_imm(const WReg &dst, T imm) {
  bool flag = false;

  /* T may be int32_t or int64_t. */
  uint32_t bit_ptn = static_cast<uint32_t>(imm);

  if (imm == 0) {
    mov(dst, 0);
    return;
  }

  if (imm < std::numeric_limits<int32_t>::min()) {
    throw Error(ERR_ILLEGAL_IMM_RANGE, genErrMsg());
  }

  if (isBitMask(bit_ptn)) {
    mov(dst, bit_ptn);
    return;
  }

  for (int i = 0; i < 2; i++) {
    if (bit_ptn & (0xFFFF << 16 * i)) {
      if (flag == false) {
        movz(dst, (bit_ptn >> (16 * i)) & 0xFFFF, 16 * i);
        flag = true;
      } else {
        movk(dst, (bit_ptn >> (16 * i)) & 0xFFFF, 16 * i);
      }
    }
  }

  return;
}

template <typename T> void cmp_imm(const XReg &a, T imm, const XReg &b) {

  if ((imm >> 12) == 0) {
    cmp(a, imm);
  } else {
    mov_imm(b, imm);
    cmp(a, b);
  }

  return;
}

template <typename T> void cmp_imm(const WReg &a, T imm, const WReg &b) {

  if ((imm >> 12) == 0) {
    cmp(a, imm);
  } else {
    mov_imm(b, imm);
    cmp(a, b);
  }

  return;
}
