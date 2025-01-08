#pragma once
/*******************************************************************************
 * Copyright 2019-2023 FUJITSU LIMITED
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
#include <exception>

namespace Xbyak_aarch64 {

enum {
  ERR_NONE = 0,
  ERR_CODE_IS_TOO_BIG,           // use at CodeArray
  ERR_LABEL_IS_REDEFINED,        // use at LabelMgr
  ERR_LABEL_IS_TOO_FAR,          // use at CodeGenerator
  ERR_LABEL_IS_NOT_FOUND,        // use at LabelMgr
  ERR_BAD_PARAMETER,             // use at CodeArray
  ERR_CANT_PROTECT,              // use at CodeArray
  ERR_OFFSET_IS_TOO_BIG,         // use at CodeArray
  ERR_CANT_ALLOC,                // use at CodeArray
  ERR_LABEL_ISNOT_SET_BY_L,      // use at LabelMgr
  ERR_LABEL_IS_ALREADY_SET_BY_L, // use at LabelMgr
  ERR_INTERNAL,                  // use at Error
  ERR_ILLEGAL_REG_IDX,           // use at CodeGenerator
  ERR_ILLEGAL_REG_ELEM_IDX,      // use at CodeGenerator
  ERR_ILLEGAL_PREDICATE_TYPE,    // use at CodeGenerator
  ERR_ILLEGAL_IMM_RANGE,         // use at CodeGenerator
  ERR_ILLEGAL_IMM_VALUE,         // use at CodeGenerator
  ERR_ILLEGAL_IMM_COND,          // use at CodeGenerator
  ERR_ILLEGAL_SHMOD,             // use at CodeGenerator
  ERR_ILLEGAL_EXTMOD,            // use at CodeGenerator
  ERR_ILLEGAL_COND,              // use at CodeGenerator
  ERR_ILLEGAL_BARRIER_OPT,       // use at CodeGenerator
  ERR_ILLEGAL_CONST_RANGE,       // use at CodeGenerator
  ERR_ILLEGAL_CONST_VALUE,       // use at CodeGenerator
  ERR_ILLEGAL_CONST_COND,        // use at CodeGenerator
  ERR_ILLEGAL_TYPE,
  ERR_BAD_ALIGN,
  ERR_BAD_ADDRESSING,
  ERR_BAD_SCALE,
  ERR_MUNMAP,
};

class Error : public std::exception {
  int err_;
  const char *msg_;

public:
  explicit Error(int err);
  operator int() const { return err_; }
  const char *what() const throw() { return msg_; }
};

inline const char *ConvertErrorToString(const Error &err) { return err.what(); }

} // namespace Xbyak_aarch64
