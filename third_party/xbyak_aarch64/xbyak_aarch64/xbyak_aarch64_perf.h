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
/*
 * perf utility class
 * @author herumi
 * Linux only
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

namespace Xbyak_aarch64 {

class Profiler {
  int mode_;
  const char *suffix_;
  const void *startAddr_;
  FILE *fp_;

public:
  enum { None = 0, Perf = 1, VTune = 2 };
  Profiler() : mode_(None), suffix_(""), startAddr_(0), fp_(0) {}
  // append suffix to funcName
  void setNameSuffix(const char *suffix) { suffix_ = suffix; }
  void setStartAddr(const void *startAddr) { startAddr_ = startAddr; }
  void init(int mode = Perf) {
    mode_ = None;
    switch (mode) {
    default:
    case None:
      return;
    case Perf:
      close();
      {
        const int pid = getpid();
        char name[128];
        snprintf(name, sizeof(name), "/tmp/perf-%d.map", pid);
        fp_ = fopen(name, "a+");
        if (fp_ == 0) {
          fprintf(stderr, "can't open %s\n", name);
          return;
        }
      }
      mode_ = Perf;
      return;
    }
  }
  ~Profiler() { close(); }
  void close() {
    if (fp_ == 0) {
      return;
    }
    fclose(fp_);
    fp_ = 0;
  }
  void set(const char *funcName, const void *startAddr, size_t funcSize) const {
    if (mode_ == None) {
      return;
    }
    if (mode_ == Perf) {
      if (fp_ == 0) {
        return;
      }
      fprintf(fp_, "%llx %zx %s%s", (long long)startAddr, funcSize, funcName, suffix_);
      /*
          perf does not recognize the function name which is less than 3,
          so append '_' at the end of the name if necessary
      */
      size_t n = strlen(funcName) + strlen(suffix_);
      for (size_t i = n; i < 3; i++) {
        fprintf(fp_, "_");
      }
      fprintf(fp_, "\n");
      fflush(fp_);
    }
  }
  /*
      for continuous set
      funcSize = endAddr - <previous set endAddr>
  */
  void set(const char *funcName, const void *endAddr) {
    set(funcName, startAddr_, (size_t)endAddr - (size_t)startAddr_);
    startAddr_ = endAddr;
  }
};

} // namespace Xbyak_aarch64
