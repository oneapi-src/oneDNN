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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_FDSTREAM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_FDSTREAM_HPP
#include <istream>
#include <memory>
#include <ostream>
#include <streambuf>
#include <utility>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * fd output buffer
 * */
class fdoutbuf_t : public std::streambuf {
public:
    fdoutbuf_t(int fd = -1) : fd_(fd) {}
    fdoutbuf_t(fdoutbuf_t &&other) {
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    fdoutbuf_t &operator=(fdoutbuf_t &&other) {
        close();
        fd_ = other.fd_;
        other.fd_ = -1;
        return *this;
    }
    void close();
    ~fdoutbuf_t() override { close(); }

protected:
    int fd_;
    std::streamsize xsputn(const char *s, std::streamsize num) override;
    int_type overflow(int_type c) override;
};

/**
 * The output stream that operates on an fd. Will take the ownership of the fd
 * */
class ofdstream_t : public std::ostream {
protected:
    fdoutbuf_t buf_;

public:
    ofdstream_t(int fd = -1) : std::ostream(nullptr), buf_(fd) { rdbuf(&buf_); }
    void reset(int fd) {
        this->~ofdstream_t();
        new (this) ofdstream_t(-1);
        buf_ = fdoutbuf_t(fd);
    }
    // g++ 4.8 don't support these
    // ofdstream_t(ofdstream_t &&other)
    //     : std::ostream(std::move(other)), buf_(std::move(other.buf_)) {}
    // ofdstream_t &operator=(ofdstream_t &&other) {
    //     this->std::ostream::operator=(std::move(other));
    //     buf_ = std::move(other.buf_);
    //     return *this;
    // }
};

class fdinbuf_t : public std::streambuf {
protected:
    int fd_;
    std::unique_ptr<char[]> data_;
    static constexpr int putback_size_ = 4;
    static constexpr int buf_size_ = 1024;

public:
    fdinbuf_t(int fd = -1)
        : fd_(fd)
        , data_(std::unique_ptr<char[]>(new char[putback_size_ + buf_size_])) {}
    fdinbuf_t(fdinbuf_t &&other) : data_(std::move(other.data_)) {
        setg(other.eback(), other.gptr(), other.egptr());
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    fdinbuf_t &operator=(fdinbuf_t &&other) {
        close();
        setg(other.eback(), other.gptr(), other.egptr());
        fd_ = other.fd_;
        other.fd_ = -1;
        data_ = std::move(other.data_);
        return *this;
    }
    void close();
    ~fdinbuf_t() override { close(); }

protected:
    int underflow() override;
};

class ifdstream_t : public std::istream {
protected:
    fdinbuf_t buf_;

public:
    ifdstream_t(int fd = -1) : std::istream(nullptr), buf_(fd) { rdbuf(&buf_); }
    void reset(int fd) {
        this->~ifdstream_t();
        new (this) ifdstream_t(-1);
        buf_ = fdinbuf_t(fd);
    }
    // g++ 4.8 don't support these
    // ifdstream_t(ifdstream_t &&other)
    //     : std::istream(std::move(other)), buf(std::move(other.buf)) {}
    // ifdstream_t &operator=(ifdstream_t &&other) {
    //     this->std::istream::operator=(std::move(other));
    //     buf = std::move(other.buf);
    //     return *this;
    // }
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
