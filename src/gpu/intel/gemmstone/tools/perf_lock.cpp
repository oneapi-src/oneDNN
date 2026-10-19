/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "perf_lock.hpp"

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace {
thread_local int g_depth = 0;

#if defined(_WIN32)
constexpr const char *filename = "C:\\ProgramData\\onednn_perf.lock";
#else
constexpr const char *filename = "/tmp/onednn_perf.lock";
#endif

[[noreturn]] void perf_lock_fatal(const char *action) {
    std::fprintf(stderr,
            "Error: couldn't %s the GPU perf lock. Aborting to avoid "
            "running GPU kernels unsynchronized with other processes.\n",
            action);
    std::exit(2);
}
} // namespace

#if defined(_WIN32)

perf_lock_t::perf_lock_t(bool enabled) {
    if (!enabled) return;

    active_ = true;
    if (g_depth++ > 0) return;

    // `OPEN_ALWAYS` creates the file if it doesn't exist yet, otherwise
    // opens the existing one, matching the POSIX `O_CREAT` behavior below.
    HANDLE h = CreateFileA(filename, GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) perf_lock_fatal("open");

    OVERLAPPED overlapped = {};
    // Blocking, exclusive lock over the whole file.
    if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD,
                &overlapped)) {
        CloseHandle(h);
        perf_lock_fatal("acquire");
    }
    handle_ = h;
}

perf_lock_t::~perf_lock_t() {
    if (!active_) return;
    if (--g_depth > 0) return;

    if (!handle_) return;
    HANDLE h = static_cast<HANDLE>(handle_);
    OVERLAPPED overlapped = {};
    UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &overlapped);
    CloseHandle(h);
}

#else

perf_lock_t::perf_lock_t(bool enabled) {
    if (!enabled) return;

    active_ = true;
    if (g_depth++ > 0) return;

    int fd = open(filename, O_CREAT | O_RDWR, 0666);
    if (fd < 0) perf_lock_fatal("open");

    // Blocking, exclusive lock, released automatically once the last
    // descriptor referring to it (held by this or any other process) is
    // closed, including on abnormal process termination.
    if (flock(fd, LOCK_EX) != 0) {
        close(fd);
        perf_lock_fatal("acquire");
    }
    fd_ = fd;
}

perf_lock_t::~perf_lock_t() {
    if (!active_) return;
    if (--g_depth > 0) return;

    if (fd_ < 0) return;
    flock(fd_, LOCK_UN);
    close(fd_);
}

#endif
