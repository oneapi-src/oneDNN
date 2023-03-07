/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <util/string_utils.hpp>
#include <util/utils.hpp>

#include "stack_frame_model.hpp"

using std::cout;
using std::endl;
using std::ostringstream;
using std::string;
using std::vector;

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

using utils::make_unique;
using utils::replace_newlines;

//==============================================================================
// Methods for logging / tracing...
//==============================================================================

std::vector<std::string> stack_frame_model::stack_item::dump_members() const {
    vector<string> v;

    ostringstream os;
    os << "stack_size_before_item_=" << stack_size_before_item_;
    v.push_back(os.str());

    if (!debug_comment_.empty()) {
        os.str("");
        os << "debug_comment_=\"" << replace_newlines(debug_comment_, " ")
           << "\"";
        v.push_back(os.str());
    }

    return v;
}

std::vector<std::string> stack_frame_model::slot::dump_members() const {
    vector<string> v(stack_item::dump_members());

    ostringstream os;
    os << "slot_size_=" << slot_size_;
    v.push_back(os.str());

    os.str("");
    os << "val_type_=" << val_type_;
    v.push_back(os.str());

    const int64_t rbp_offset = get_rbp_offset();
    os.str("");
    os << "%rbp offset:";
    os << " (dec)" << rbp_offset;
    if (rbp_offset >= 0) {
        os << " (hex)0x" << std::hex << rbp_offset << std::dec;
    } else {
        os << " (hex)-0x" << std::hex << (-1 * rbp_offset) << std::dec;
    }
    v.push_back(os.str());

    return v;
}

std::vector<std::string>
stack_frame_model::anonymous_slot::dump_members() const {
    return slot::dump_members();
}

std::string stack_frame_model::anonymous_slot::dump_brief() const {
    ostringstream os;
    os << "(size:" << slot_size_ << ")";
    return os.str();
}

std::string stack_frame_model::anonymous_slot::dump_item_kind() const {
    return "ANONYMOUS SLOT";
};

std::vector<std::string>
stack_frame_model::named_tensor_buffer_slot::dump_members() const {
    vector<string> v(named_slot::dump_members());

    ostringstream os;
    os << "num_elements_=" << num_elements_;
    v.push_back(os.str());

    return v;
}

std::vector<std::string> stack_frame_model::named_slot::dump_members() const {
    vector<string> v(slot::dump_members());

    ostringstream os;
    os << "name_=\"" << name_ << "\"";
    v.push_back(os.str());

    return v;
}

std::vector<std::string>
stack_frame_model::caller_param_slot::dump_members() const {
    vector<string> v(named_slot::dump_members());

    ostringstream os;
    os << "callee_rbp_offset_=" << callee_rbp_offset_;
    v.push_back(os.str());

    return v;
}

std::string stack_frame_model::caller_param_slot::dump_brief() const {
    ostringstream os;
    os << "(";
    os << "param-name:\"" << name_ << "\"";
    os << " size:" << slot_size_;
    os << ")";
    return os.str();
}

std::string stack_frame_model::caller_param_slot::dump_item_kind() const {
    return "CALLER-PARAM SLOT";
}

std::string stack_frame_model::named_slot::dump_brief() const {
    ostringstream os;
    os << "(";
    os << "name:\"" << name_ << "\"";
    os << " size:" << slot_size_;
    os << ")";
    return os.str();
}

std::string stack_frame_model::named_slot::dump_item_kind() const {
    return "NAMED SLOT";
};

std::string stack_frame_model::named_tensor_buffer_slot::dump_brief() const {
    ostringstream os;
    os << "(";
    os << "name:\"" << name_ << "\"";
    os << " #elem:" << num_elements_;
    os << " size:" << slot_size_;
    os << ")";
    return os.str();
}

std::string
stack_frame_model::named_tensor_buffer_slot::dump_item_kind() const {
    return "NAMED TENSOR BUFFER SLOT";
};

std::vector<std::string>
stack_frame_model::lexical_scope::dump_members() const {
    return stack_item::dump_members();
}

std::string stack_frame_model::lexical_scope::dump_brief() const {
    return "(S)";
}

std::string stack_frame_model::lexical_scope::dump_item_kind() const {
    return "LEXICAL SCOPE";
};

void stack_frame_model::dump(std::ostream &os) const {
    os << "stack_frame_model:" << endl;
}

std::string stack_frame_model::one_line_summary() const {
    ostringstream os;
    os << "{";

    for (const auto &kv : name_to_caller_param_slot_) {
        os << " " << kv.second.dump_brief();
    }

    for (size_t i = 0; i < stack_.size(); ++i) {
        const stack_item *const item = stack_[i].get();
        os << " " << stack_[i]->dump_brief();
    }

    os << " }";
    return os.str();
}

#define LOG_LINE(S1, ...) \
    if (logging_enabled_) { \
        cout << "[" << utils::brief_lineloc(__FILE__, __LINE__) << "]" \
             << " " << S1 __VA_ARGS__ << endl; \
    }

#define LOG_FUNC_ENTRY \
    if (logging_enabled_) { cout << __PRETTY_FUNCTION__ << " : ENTER" << endl; }

#define LOG_FUNC_ENTRY_WITH_TEXT(S1, ...) \
    if (logging_enabled_) { \
        cout << __PRETTY_FUNCTION__ << " : ENTER: " << S1 __VA_ARGS__ << endl; \
    }

#define LOG_MODIFIER_FUNC_EXIT \
    if (logging_enabled_) { \
        cout << __PRETTY_FUNCTION__ << " : PRE-EXIT DUMP:" << endl; \
        dump(cout); \
        cout << endl; \
    }

#define LOG_FUNC_EXIT_WITH_TEXT(S1, ...) \
    if (logging_enabled_) { \
        cout << __PRETTY_FUNCTION__ << " : PRE-EXIT: " << S1 __VA_ARGS__ \
             << endl; \
    }

#define LOG_MODIFIER_FUNC_EXIT_WITH_TEXT(S1, ...) \
    if (logging_enabled_) { \
        cout << __PRETTY_FUNCTION__ << " : PRE-EXIT DUMP:" << S1 __VA_ARGS__ \
             << endl; \
        dump(cout); \
        cout << endl; \
    }

//==============================================================================

stack_frame_model::stack_frame_model(bool logging_enabled)
    : logging_enabled_(logging_enabled) {}

void stack_frame_model::pop_top() {
    LOG_FUNC_ENTRY
    COMPILE_ASSERT(!stack_.empty(), "stack is empty already");

    const stack_item *top = stack_.back().get();

    stack_.pop_back();
    LOG_MODIFIER_FUNC_EXIT
}

void stack_frame_model::push_scope(const std::string &debug_comment) {
    LOG_FUNC_ENTRY_WITH_TEXT("debug_comment=\"" << debug_comment)
    const size_t old_stack_size = get_size();
    stack_.push_back(
            utils::make_unique<lexical_scope>(old_stack_size, debug_comment));
    LOG_MODIFIER_FUNC_EXIT
}

size_t stack_frame_model::get_top_scope_size() const {
    LOG_FUNC_ENTRY
    size_t scope_size = 0;

    for (auto iter = stack_.crbegin(); iter != stack_.crend(); ++iter) {
        const stack_item *item = iter->get();
        if (const slot *s = dynamic_cast<const slot *>(item)) {
            scope_size += s->slot_size_;
        } else {
            LOG_FUNC_EXIT_WITH_TEXT("scope_size=" << scope_size)
            return scope_size;
        }
    }

    COMPILE_ASSERT(false, "stack frame model has no scopes");
}

size_t stack_frame_model::pop_scope() {
    LOG_FUNC_ENTRY
    size_t scope_size = 0;

    while (!stack_.empty()) {
        const stack_item *top = stack_.back().get();

        if (const slot *s = dynamic_cast<const slot *>(top)) {
            scope_size += s->slot_size_;
            pop_top();
        } else {
            pop_top();
            LOG_MODIFIER_FUNC_EXIT_WITH_TEXT("scope_size="
                    << scope_size << " stack-size=" << get_size());
            return scope_size;
        }
    }

    COMPILE_ASSERT(false, "stack frame model has no scopes");
}

void stack_frame_model::push_named_object(const std::string &name,
        x86_64::cpu_data_type val_type, size_t num_bytes,
        const std::string &debug_comment) {
    LOG_FUNC_ENTRY_WITH_TEXT("name=\"" << name << "\""
                                       << " num_bytes=" << num_bytes
                                       << " debug_comment=\"" << debug_comment
                                       << "\"")
    assert_unused_name(name);
    COMPILE_ASSERT(!name.empty(), "named objects cannot have blank name");
    COMPILE_ASSERT(
            num_bytes > 0, "stack_frame_model items must have positive sizes");

    const size_t old_stack_size = get_size();
    stack_.push_back(utils::make_unique<named_slot>(
            name, old_stack_size, num_bytes, val_type, debug_comment));

    LOG_MODIFIER_FUNC_EXIT_WITH_TEXT("stack-size=" << get_size());
}

void stack_frame_model::push_named_tensor_buffer_object(const std::string &name,
        x86_64::cpu_data_type val_type, size_t num_elements, size_t num_bytes,
        const std::string &debug_comment) {
    LOG_FUNC_ENTRY_WITH_TEXT("name=\""
            << name << "\""
            << " num_elements=" << num_elements << " num_bytes=" << num_bytes
            << " debug_comment=\"" << debug_comment << "\"")
    assert_unused_name(name);
    COMPILE_ASSERT(!name.empty(), "named objects cannot have blank name");
    COMPILE_ASSERT(
            num_bytes > 0, "stack_frame_model items must have positive sizes");

    const size_t old_stack_size = get_size();
    stack_.push_back(utils::make_unique<named_tensor_buffer_slot>(name,
            old_stack_size, num_bytes, val_type, num_elements, debug_comment));

    LOG_MODIFIER_FUNC_EXIT_WITH_TEXT("stack-size=" << get_size());
}

void stack_frame_model::push_anonymous_object(x86_64::cpu_data_type val_type,
        size_t num_bytes, const std::string &debug_comment) {
    LOG_FUNC_ENTRY_WITH_TEXT("num_bytes=" << num_bytes << " debug_comment=\""
                                          << debug_comment << "\"")

    COMPILE_ASSERT(
            num_bytes > 0, "stack_frame_model items must have positive sizes");

    const size_t old_stack_size = get_size();
    stack_.push_back(utils::make_unique<anonymous_slot>(
            old_stack_size, num_bytes, val_type, debug_comment));
    LOG_MODIFIER_FUNC_EXIT_WITH_TEXT("stack-size=" << get_size());
}

void stack_frame_model::shrink(size_t num_bytes) {
    LOG_FUNC_ENTRY_WITH_TEXT("num_bytes=" << num_bytes)

    const size_t old_stack_size = get_size();
    COMPILE_ASSERT(num_bytes <= get_size(),
            "stack_frame_model size is " << old_stack_size
                                         << " but trying to shrink by "
                                         << num_bytes);

    size_t remaining_bytes_to_remove = num_bytes;
    while (remaining_bytes_to_remove > 0) {
        LOG_LINE("remaining_bytes_to_remove = " << remaining_bytes_to_remove);
        assert(!stack_.empty());
        stack_item *top = stack_.back().get();

        if (slot *s = dynamic_cast<slot *>(top)) {
            COMPILE_ASSERT(remaining_bytes_to_remove >= s->slot_size_,
                    "shrink(...) can't remove partial slots");
            remaining_bytes_to_remove -= s->slot_size_;
            pop_top();
        } else {
            COMPILE_ASSERT(false, "shrink(...) can't remove lexical scopes.");
        }
    }
    LOG_MODIFIER_FUNC_EXIT_WITH_TEXT(
            "old_stack_size=" << old_stack_size << " (new)size=" << get_size());
}

void stack_frame_model::shrink_to_size(size_t new_size) {
    LOG_FUNC_ENTRY_WITH_TEXT("new_size=" << new_size)
    size_t current_size = get_size();
    assert(new_size <= current_size);

    size_t remaining_bytes_to_remove = current_size - new_size;

    while (remaining_bytes_to_remove > 0) {
        assert(!stack_.empty());
        stack_item *top = stack_.back().get();

        if (slot *s = dynamic_cast<slot *>(top)) {
            COMPILE_ASSERT(remaining_bytes_to_remove >= s->slot_size_,
                    "shrink(...) can't remove partial slots");
            remaining_bytes_to_remove -= s->slot_size_;
            pop_top();
        } else {
            // the stack item is a lexical scope.
            pop_top();
        }
    }

    // Eliminate any lexical scopes still at the top...
    while ((remaining_bytes_to_remove == 0)
            && (dynamic_cast<lexical_scope *>(stack_.back().get()))) {
        pop_top();
    }

    LOG_MODIFIER_FUNC_EXIT_WITH_TEXT("(old)current_size="
            << current_size << " (new)size=" << get_size());
}

size_t stack_frame_model::get_size() const {
    // LOG_FUNC_ENTRY
    if (stack_.empty()) {
        // LOG_FUNC_EXIT_WITH_TEXT("(empty) return 0")
        return 0;
    }

    const stack_item *item = stack_.back().get();
    assert(item);

    if (const slot *s = dynamic_cast<const slot *>(item)) {
        const size_t stack_size = s->stack_size_before_item_ + s->slot_size_;
        // LOG_FUNC_EXIT_WITH_TEXT("(slot) return " << stack_size)
        return stack_size;
    } else {
        const size_t stack_size = item->stack_size_before_item_;
        // LOG_FUNC_EXIT_WITH_TEXT("(scope) return " << stack_size)
        return stack_size;
    }
}

void stack_frame_model::clear() {
    LOG_FUNC_ENTRY
    stack_.clear();
    name_to_caller_param_slot_.clear();
    LOG_MODIFIER_FUNC_EXIT
};

void stack_frame_model::assert_unused_name(const std::string &name) {
    COMPILE_ASSERT(name_to_caller_param_slot_.find(name)
                    == name_to_caller_param_slot_.end(),
            "A caller-parameter slot with name \"" << name
                                                   << "\" already exists.");
}

void stack_frame_model::add_caller_param_slot(const caller_param_slot &s) {
    LOG_FUNC_ENTRY_WITH_TEXT("name=\"" << s.name_ << "\""
                                       << " slot_size=" << s.slot_size_
                                       << " debug_comment=\""
                                       << s.debug_comment_ << "\"")
    COMPILE_ASSERT(!s.name_.empty(), "named objects cannot have blank name");
    COMPILE_ASSERT(s.slot_size_ > 0,
            "stack_frame_model items must have positive sizes");
    assert_unused_name(s.name_);
    name_to_caller_param_slot_.insert(std::make_pair(s.name_, s));
}

int64_t stack_frame_model::get_named_object_rbp_offset(
        const std::string &name) const {
    int64_t offset;
    const bool success = try_get_named_object_rbp_offset(name, offset);
    COMPILE_ASSERT(success,
            "stack_frame_model has no stack item named '" << name << "'");
    return offset;
}

const stack_frame_model::named_slot *stack_frame_model::try_get_named_slot(
        const std::string &name) const {
    const auto iter2 = name_to_caller_param_slot_.find(name);
    if (iter2 != name_to_caller_param_slot_.end()) { return &(iter2->second); }

    return nullptr;
}

const stack_frame_model::slot *stack_frame_model::get_top_slot() const {
    const stack_item *top = stack_.back().get();
    const slot *s = dynamic_cast<const slot *>(top);
    assert(s);
    return s;
}

int64_t stack_frame_model::slot::get_rbp_offset() const {
    return -1 * (stack_size_before_item_ + slot_size_);
}

bool stack_frame_model::try_get_named_object_rbp_offset(
        const std::string &name, int64_t &offset) const {
    LOG_FUNC_ENTRY_WITH_TEXT("name=\"" << name << "\"");

    {
        const auto iter = name_to_caller_param_slot_.find(name);
        if (iter != name_to_caller_param_slot_.end()) {
            offset = iter->second.callee_rbp_offset_;

            LOG_FUNC_EXIT_WITH_TEXT("offset=" << offset);
            return true;
        }
    }

    LOG_FUNC_EXIT_WITH_TEXT("name not found");
    return false;
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
