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

#ifndef GPU_JIT_V2_CONV_PLANNER_ARG_PARSER_HPP
#define GPU_JIT_V2_CONV_PLANNER_ARG_PARSER_HPP

#include <any>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

class parser_arg_t {
public:
    parser_arg_t(const std::string &name) : name_(name) {
        action_func_ = std::function<std::string(const std::string &s)>(
                [=](const std::string &s) { return s; });
    }

    const std::string &name() const { return name_; }
    const std::string &help_msg() const { return help_msg_; }
    bool is_implicit() const { return is_implicit_; }

    parser_arg_t &help(const std::string &msg) {
        help_msg_ = msg;
        return *this;
    }

    parser_arg_t &implicit_value(bool value) {
        is_implicit_ = value;
        return *this;
    }

    template <typename Func>
    parser_arg_t &action(Func func) {
        using arg_type_t = typename std::result_of<Func(std::string)>::type;
        std::function<arg_type_t(const std::string &s)> f
                = [=](const std::string &s) { return func(s); };
        action_func_ = f;
        return *this;
    }

    template <typename T>
    parser_arg_t &default_value(const T &value) {
        value_ = value;
        return *this;
    }

    template <typename T>
    void set(const T &t) {
        value_ = t;
    }

    void set_str(const std::string &s) { str_value_ = s; }

    template <typename T>
    T get() const {
        if (is_implicit_ && !std::is_same<T, bool>::value) {
            handle_error("Implicit arguments must be boolean.");
        }
        if (str_value_.empty()) return std::any_cast<T>(value_);
        auto f = std::any_cast<std::function<T(const std::string &s)>>(
                action_func_);
        return f(str_value_);
    }

private:
    static void handle_error(const char *msg) { throw std::runtime_error(msg); }

    std::string name_;
    std::string help_msg_;
    std::any action_func_;
    std::any value_;
    std::string str_value_;
    bool is_implicit_ = false;
};

class arg_parser_t {
public:
    arg_parser_t(const std::string &name) : name_(name) {}

    parser_arg_t &add_argument(const std::string &name) {
        args_.emplace_back(name);
        return args_.back();
    }

    void parse_args(int argc, const char **argv) {
        maybe_print_help(argc, argv);
        for (int i = 1; i < argc; i++) {
            bool found = false;
            for (auto &a : args_) {
                if (a.name() == argv[i]) {
                    if (a.is_implicit()) {
                        a.set(true);
                    } else {
                        a.set_str(argv[i + 1]);
                        i++;
                    }
                    found = true;
                    break;
                }
            }
            if (!found) handle_error("Invalid argument.");
        }
        return;
    }

    template <typename T>
    T get(const std::string &name) const {
        for (auto &a : args_) {
            if (a.name() == name) return a.get<T>();
        }
        handle_error("Invalid name.");
        return T();
    }

private:
    static void handle_error(const char *msg) { throw std::runtime_error(msg); }

    void maybe_print_help(int argc, const char **argv) const {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_help();
                exit(0);
            }
        }
    }

    void print_help() const {
        std::cout << "Usage: " << name_;
        for (auto &a : args_) {
            std::cout << "[" << a.name() << "] ";
        }
        std::cout << std::endl << std::endl;
        std::cout << "Optional arguments:";
        for (auto &a : args_) {
            std::cout << "  " << a.name() << " " << a.help_msg() << std::endl;
        }
    }

    std::string name_;
    std::list<parser_arg_t> args_;
};

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
