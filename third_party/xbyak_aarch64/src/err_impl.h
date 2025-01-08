#pragma once
/*******************************************************************************
 * Copyright 2020-2023 FUJITSU LIMITED
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

Error::Error(int err) : err_(err), msg_("") {
  if (err_ <= 0)
    return;
  fprintf(stderr, "bad err=%d in Xbyak::Error\n", err_);
  static const char *tbl[32] = {
      "none",
      "code is too big",
      "label is redefined",
      "label is too far",
      "label is not found",
      "bad parameter",
      "can't protect",
      "offset is too big",
      "can't alloc",
      "label is not set by L()",
      "label is already set by L()",
      "internal error",
      "illegal register index (can not encoding register index)",
      "illegal register element index (can not encoding element index)",
      "illegal predicate register type",
      "illegal immediate parameter (range error)",
      "illegal immediate parameter (unavailable value error)",
      "illegal immediate parameter (condition error)",
      "illegal shift-mode paramater",
      "illegal extend-mode parameter",
      "illegal condition parameter",
      "illegal barrier option",
      "illegal const parameter (range error)",
      "illegal const parameter (unavailable error)",
      "illegal const parameter (condition error)",
      "illegal type",
      "bad align",
      "bad addressing",
      "bad scale",
  };
  if ((size_t)err_ >= sizeof(tbl) / sizeof(tbl[0])) {
    msg_ = "bad err num";
  } else {
    msg_ = tbl[err_];
  }
}
