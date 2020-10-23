#pragma once
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
  std::string msg_;
  std::string opt_msg_;

public:
  explicit Error(int err, const std::string &opt_msg = "")
      : err_(err), msg_(""), opt_msg_(opt_msg) {
    if (err_ > 0) {
      fprintf(stderr, "bad err=%d in Xbyak::Error\n", err_);
      const std::initializer_list<const char *> &errTbl = getErrTbl();
      assert((size_t)err_ < errTbl.size());
      msg_ = "";
      msg_ += *(errTbl.begin() + err_);
      msg_ += "\n";
      msg_ += opt_msg;
      // exit(1);
    }
  }
  operator int() const { return err_; }
  const char *what() const throw() { return msg_.c_str(); }

private:
  static const std::initializer_list<const char *> &getErrTbl() {
    static const std::initializer_list<const char *> tbl = {
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
        "illegal register element index (can not encoding element "
        "index)",
        "illegal predicate register type",
        "illegal immediate parameter (range error)",
        "illegal immediate parameter (unavailable value error)",
        "illegal immediate parameter (condition error)",
        "illegal shift-mode paramater",
        "illegal extend-mode parameter",
        "illegal condition parameter",
        "illegal type",
        "illegal barrier option",
        "illegal const parameter (range error)",
        "illegal const parameter (unavailable error)",
        "illegal const parameter (condition error)",
    };
    return tbl;
  };
};

inline const char *ConvertErrorToString(const Error &err) { return err.what(); }
