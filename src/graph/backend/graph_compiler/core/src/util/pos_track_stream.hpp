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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_POS_TRACK_STREAM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_POS_TRACK_STREAM_HPP
#include <istream>
#include <ostream>
#include <streambuf>
#include <utility>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
class track_pos_buf_t : public std::streambuf {
public:
    std::ostream &os_;
    track_pos_buf_t(std::ostream &os) : os_(os) {}
    int pos_ = 1;
    int line_ = 1;

protected:
    void process_char(char c) {
        if (c == '\n') {
            line_++;
            pos_ = 0;
        } else {
            pos_++;
        }
    }
    std::streamsize xsputn(const char *s, std::streamsize num) override {
        for (std::streamsize i = 0; i < num; i++) {
            process_char(s[i]);
        }
        os_.write(s, num);
        return num;
    }
    int_type overflow(int_type c) override {
        process_char(c);
        os_ << (char)c;
        return c;
    }
};

// an ostream to track the current line position in the output text
class track_pos_stream_t : public std::ostream {
public:
    track_pos_buf_t buf_;
    track_pos_stream_t(std::ostream &os) : std::ostream(nullptr), buf_(os) {
        rdbuf(&buf_);
    }
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
