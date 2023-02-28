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
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <string.h>
#include "utils.hpp"
#ifdef _WIN32
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
extern char **environ;
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {

#ifdef _WIN32
bool create_process_and_await(const std::string &program,
        const std::vector<std::string> &args, int &exit_code,
        const std::string *rstdin, std::string *rstdout, std::string *rstderr) {
    // fix-me: (win32)
    throw std::runtime_error("create_process_and_await");
}

bool wait_process(uintptr_t pid, int &exit_code) {
    int status;
    // fix-me: (win32)
    throw std::runtime_error("wait_process");
}

bool create_process(const std::string &program,
        const std::vector<std::string> &args, uintptr_t &outhandle,
        ofdstream_t *rstdin, ifdstream_t *rstdout, ifdstream_t *rstderr) {
    // fix-me: (win32)
    throw std::runtime_error("create_process");
}

#else
void read_all_from_fd(int fd, std::string &str) {
    constexpr int buff_size = 256;
    char buf[buff_size]; // NOLINT
    ssize_t written = read(fd, buf, buff_size);
    while (written > 0) {
        str.insert(str.size(), buf, written);
        written = read(fd, buf, buff_size);
    }
}

struct posix_spawn_file_actions_helper_t {
    posix_spawn_file_actions_t child_fd_actions;
    bool success;
    posix_spawn_file_actions_helper_t() {
        if (auto retv = posix_spawn_file_actions_init(&child_fd_actions)) {
            perror("posix_spawn_file_actions_init");
            success = false;
        } else {
            success = true;
        }
    }
    ~posix_spawn_file_actions_helper_t() {
        if (success) { posix_spawn_file_actions_destroy(&child_fd_actions); }
    }
};

struct posix_spawnattr_helper_t {
    posix_spawnattr_t attr;
    bool success;
    posix_spawnattr_helper_t() {
        if (auto retv = posix_spawnattr_init(&attr)) {
            perror("posix_spawn_file_actions_init");
            success = false;
        } else {
            success = true;
        }
    }
    ~posix_spawnattr_helper_t() {
        if (success) { posix_spawnattr_destroy(&attr); }
    }
};

bool create_process_and_await(const std::string &program,
        const std::vector<std::string> &args, int &exit_code,
        const std::string *rstdin, std::string *rstdout, std::string *rstderr) {
    int stdinpipefd[2];
    if (rstdin) {
        if (pipe(stdinpipefd)) {
            perror("pipe stdin failed:");
            return false;
        }
    }

    int stdoutpipefd[2];
    if (rstdout) {
        if (pipe(stdoutpipefd)) {
            perror("pipe stdout failed:");
            return false;
        }
    }

    int stderrpipefd[2];
    if (rstderr) {
        if (pipe(stderrpipefd)) {
            perror("pipe stderr failed:");
            return false;
        }
    }
    char *opt_char[args.size() + 1];
    opt_char[args.size()] = nullptr;
    for (unsigned i = 0; i < args.size(); i++) {
        opt_char[i] = const_cast<char *>(args.at(i).c_str());
    }

    pid_t pid;
    posix_spawn_file_actions_helper_t child_fd_actions_helper;
    if (!child_fd_actions_helper.success) { return false; }
    posix_spawn_file_actions_t &child_fd_actions
            = child_fd_actions_helper.child_fd_actions;

#define ADD_ACTION(action, ...) \
    if (auto retv = posix_spawn_file_actions_add##action( \
                &child_fd_actions, __VA_ARGS__)) { \
        perror("posix_spawn_file_actions_add" #action); \
        return false; \
    }

    if (rstdin) {
        ADD_ACTION(dup2, stdinpipefd[0], STDIN_FILENO);
        ADD_ACTION(close, stdinpipefd[1]);
    }

    if (rstdout) {
        ADD_ACTION(dup2, stdoutpipefd[1], STDOUT_FILENO);
        ADD_ACTION(close, stdoutpipefd[0]);
    }
    if (rstderr) {
        ADD_ACTION(dup2, stderrpipefd[1], STDERR_FILENO);
        ADD_ACTION(close, stderrpipefd[0]);
    }

    posix_spawnattr_helper_t attr_helper;
    if (!attr_helper.success) { return false; }
    posix_spawnattr_t &attr = attr_helper.attr;
    if (auto vret = posix_spawnattr_setflags(&attr, POSIX_SPAWN_USEVFORK)) {
        perror("posix_spawnattr_setflags");
        return false;
    }

    if (auto retv = posix_spawnp(&pid, program.c_str(), &child_fd_actions,
                &attr, opt_char, environ)) {
        perror("posix_spawnp");
        return false;
    }

    if (rstdin) {
        close(stdinpipefd[0]);
        if (write(stdinpipefd[1], rstdin->c_str(), rstdin->size())
                != (int)rstdin->size()) {
            perror("write stdout failed:");
            return false;
        }
        close(stdinpipefd[1]);
    }
    if (rstdout) { close(stdoutpipefd[1]); }
    if (rstderr) { close(stderrpipefd[1]); }

    if (rstdout) { read_all_from_fd(stdoutpipefd[0], *rstdout); }
    if (rstderr) { read_all_from_fd(stderrpipefd[0], *rstderr); }

    int status;
    waitpid(pid, &status, 0);

    if (rstdout) { close(stdoutpipefd[0]); }
    if (rstderr) { close(stderrpipefd[0]); }
    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
    } else {
        exit_code = -1;
    }
    return true;
}

bool wait_process(uintptr_t pid, int &exit_code) {
    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) { exit_code = WEXITSTATUS(status); }
    return true;
}

bool create_process(const std::string &program,
        const std::vector<std::string> &args, uintptr_t &outhandle,
        ofdstream_t *rstdin, ifdstream_t *rstdout, ifdstream_t *rstderr) {
    int stdinpipefd[2];
    if (rstdin) {
        if (pipe(stdinpipefd)) {
            perror("pipe stdin failed:");
            return false;
        }
        rstdin->reset(stdinpipefd[1]);
    }

    int stdoutpipefd[2];
    if (rstdout) {
        if (pipe(stdoutpipefd)) {
            perror("pipe stdout failed:");
            return false;
        }
        rstdout->reset(stdoutpipefd[0]);
    }

    int stderrpipefd[2];
    if (rstderr) {
        if (pipe(stderrpipefd)) {
            perror("pipe stderr failed:");
            return false;
        }
        rstderr->reset(stderrpipefd[0]);
    }
    auto pid = fork();
    if (pid > 0) {
        if (rstdin) { close(stdinpipefd[0]); }
        if (rstdout) { close(stdoutpipefd[1]); }
        if (rstderr) { close(stderrpipefd[1]); }
        outhandle = pid;
        return true;
    } else if (pid == 0) {
        int r = prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (r == -1) {
            perror(nullptr);
            exit(1);
        }
        if (rstdin) {
            dup2(stdinpipefd[0], STDIN_FILENO);
            close(stdinpipefd[1]);
        }
        if (rstdout) {
            dup2(stdoutpipefd[1], STDOUT_FILENO);
            close(stdoutpipefd[0]);
        }
        if (rstderr) {
            dup2(stderrpipefd[1], STDERR_FILENO);
            close(stderrpipefd[0]);
        }
        char *opt_char[args.size() + 1];
        opt_char[args.size()] = nullptr;
        for (unsigned i = 0; i < args.size(); i++) {
            opt_char[i] = const_cast<char *>(args.at(i).c_str());
        }
        execvp(program.c_str(), opt_char);
        exit(0);
    } else {
        perror("Error when fork");
    }
    return false;
}
#endif

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
