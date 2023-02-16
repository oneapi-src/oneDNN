/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include "fdstream.hpp"
#include <cstdio>
#include <string.h>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

void fdoutbuf_t::close() {
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
}

fdoutbuf_t::int_type fdoutbuf_t::overflow(int_type c) {
    if (c != EOF) {
        if (write(fd_, &c, 1) != 1) { return EOF; }
    }
    return c;
}

std::streamsize fdoutbuf_t::xsputn(const char *s, std::streamsize num) {
    return write(fd_, s, num);
}

void fdinbuf_t::close() {
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
}

int fdinbuf_t::underflow() {
    if (gptr() < egptr()) { return *gptr(); }

    int putback_cnt = gptr() - eback();
    if (putback_cnt > putback_size_) { putback_cnt = putback_size_; }
    memmove(data_.get() + putback_size_ - putback_cnt, gptr() - putback_cnt,
            putback_cnt);

    int ret = read(fd_, data_.get() + putback_size_, buf_size_);
    if (ret <= 0) { return EOF; }

    setg(data_.get() + putback_size_ - putback_cnt, data_.get() + putback_size_,
            data_.get() + putback_size_ + ret);
    return *gptr();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
