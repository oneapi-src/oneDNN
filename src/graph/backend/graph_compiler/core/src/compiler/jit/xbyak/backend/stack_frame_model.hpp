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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_STACK_FRAME_MODEL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_STACK_FRAME_MODEL_HPP

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>

#ifdef _MSC_VER
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

/**
 * Model the state of the stack frame at a particular moment in the execution of
 * some function.
 *
 * Useful for tracking which Variables/Tensors are currently
 * stored on the stack, and determining the assembly operand needed to access
 * them.
 *
 * # STACK OBJECTS:
 * The model supports four kinds of stack-based objects:
 *
 *   - named slots. By assigning a name to an object on the stack model, we
 *     have an easy way to find that object's (modeled) location during
 *     JIT translation.
 *
 *   - anonymous slots. Used for things such as temporary rvalues, where
 *     there's no obvious name for the object, nor is there a need to look it up
 *     by name.
 *
 *   - lexical scopes. Used to model lexical scopes from the source program. A
 *     scope doesn't occupy any storage space by itself, but it provides a
 *     logical grouping of all objects above it (i.e., closer to the top) on the
 *     stack.  Provides information needed for the generated code to efficiently
 *     shrink the stack frame.
 *
 *   - caller stack parameter. Used to model stack slots that were created by
 *     the caller to communicate parameter values.
 *
 * # STACK SLOTS: TYPES AND SIZES:
 * Each stack slot holds a single value of some CPU-native type.
 *   - For this discussion, we consider a SIMD vector to be a single value.
 *
 *   - This scheme isn't intended to support C/C++ structs or bitfields.
 *
 * Each slot's size (in bytes) will be equal to or greater than the size
 * of the slot's CPU-native type.
 * (See \c x86_64::cpu_data_types::size_in_bytes_.)
 *
 * Sometimes a slot's size will be larger. Some reasons for this:
 *
 *   - The slot is used to store a function-call's parameter or return value,
 *     and the psABI requires slot sizes to be multiples of 8 bytes.
 *
 *   - The CPU-native data value, stored within the slot, must have a
 *     certain address alignment. Required by the psABI and/or the
 *     Xbyak JIT engine's code-generator design.
 *
 * Whenever a slot size exceeds the CPU-native type's size, the CPU-native
 * value will be stored in the *lowest* address range of the slot.
 *
 * # OBJECT ADDRESSES:
 * There are two reasonable ways to compute the runtime address of a
 * stack-allocated object:
 *
 *   - relative to the base of the stack frame (%rbp), or
 *
 *   - relative to the top of the stack frame (%rsp).
 *
 * This class currently supports %rbp-relative addressing
 * (see \c get_named_object_rbp_offset()), but could easily add %rsp-relative
 * addressing if needed.
 *
 * # DEBUG COMMENTS:
 * Each stack object has an optional debug_comment string. This string is used
 * only to annotate the human-readable output of the \c dump() and
 * \c one_line_summary() methods.
 */
class stack_frame_model {
public:
    stack_frame_model(bool logging_enabled);

    ///=========================================================================
    /// Class hierarchy for individual objects on the modeled stack...
    ///=========================================================================
    struct stack_item {
        stack_item(size_t stack_size_before_item, std::string debug_comment)
            : stack_size_before_item_(stack_size_before_item)
            , debug_comment_(std::move(debug_comment)) {}

        virtual ~stack_item() {};

        /// Same value as you'd get from adding together the 'item_size_' field
        /// from all lower stack items. Stored here for convenient lookups.
        /// Value is 0 for caller_param_slot objects.
        size_t stack_size_before_item_;

        // For debugging / tracing / logging...
        std::string debug_comment_;
        virtual std::vector<std::string> dump_members() const;
        virtual std::string dump_brief() const = 0;
        virtual std::string dump_item_kind() const = 0;
    };

    struct slot : public stack_item {
        slot(size_t stack_size_before_item, size_t slot_size,
                x86_64::cpu_data_type val_type, std::string debug_comment)
            : stack_item(stack_size_before_item, debug_comment)
            , slot_size_(slot_size)
            , val_type_(val_type) {
            assert(slot_size >= x86_64::get_cpu_data_types()
                                        .lookup(val_type)
                                        .size_in_bytes_);
        }

        /// Number of bytes this occupies on the stack.
        size_t slot_size_;

        /// The CPU-level datatype stored in this slot.
        x86_64::cpu_data_type val_type_;

        virtual std::vector<std::string> dump_members() const override;

        int64_t get_rbp_offset() const;
    };

    struct anonymous_slot : public slot {
        anonymous_slot(size_t stack_size_before_item, size_t slot_size,
                x86_64::cpu_data_type val_type, std::string debug_comment)
            : slot(stack_size_before_item, slot_size, val_type, debug_comment) {
        }

        virtual std::vector<std::string> dump_members() const override;
        virtual std::string dump_brief() const override;
        virtual std::string dump_item_kind() const override;
    };

    struct named_slot : public slot {
        named_slot(std::string name, size_t stack_size_before_item,
                size_t slot_size, x86_64::cpu_data_type val_type,
                std::string debug_comment)
            : slot(stack_size_before_item, slot_size, val_type, debug_comment)
            , name_(name) {}

        std::string name_;

        virtual std::vector<std::string> dump_members() const override;
        virtual std::string dump_brief() const override;
        virtual std::string dump_item_kind() const override;
    };

    struct named_tensor_buffer_slot : public named_slot {
        named_tensor_buffer_slot(std::string name,
                size_t stack_size_before_item, size_t slot_size,
                x86_64::cpu_data_type element_type, size_t num_elements,
                std::string debug_comment)
            : named_slot(name, stack_size_before_item, slot_size, element_type,
                    debug_comment)
            , num_elements_(num_elements) {}

        size_t num_elements_;

        virtual std::vector<std::string> dump_members() const override;
        virtual std::string dump_brief() const override;
        virtual std::string dump_item_kind() const override;
    };

    struct caller_param_slot : public named_slot {
        caller_param_slot(std::string name, size_t slot_size,
                x86_64::cpu_data_type val_type, size_t callee_rbp_offset,
                std::string debug_comment)
            : named_slot(name, 0, slot_size, val_type, debug_comment)
            , callee_rbp_offset_(callee_rbp_offset) {}

        // The offset relative to the value of $rbp after the callee has created
        // its stack frame.
        size_t callee_rbp_offset_;

        virtual std::vector<std::string> dump_members() const override;
        virtual std::string dump_brief() const override;
        virtual std::string dump_item_kind() const override;
    };

    struct lexical_scope : public stack_item {
        lexical_scope(size_t stack_size_before_item, std::string debug_comment)
            : stack_item(stack_size_before_item, debug_comment) {}

        virtual std::vector<std::string> dump_members() const override;
        virtual std::string dump_brief() const override;
        virtual std::string dump_item_kind() const override;
    };

    //=========================================================================

    void push_named_object(const std::string &name,
            x86_64::cpu_data_type val_type, size_t num_bytes,
            const std::string &debug_comment = "");

    void push_named_tensor_buffer_object(const std::string &name,
            x86_64::cpu_data_type val_type, size_t num_elements,
            size_t num_bytes, const std::string &debug_comment = "");

    void push_anonymous_object(x86_64::cpu_data_type val_type, size_t num_bytes,
            const std::string &debug_comment = "");

    void add_caller_param_slot(const caller_param_slot &s);

    /// Start a new scope on the call stack.
    /// This is for book-keeping only: it lets us keep track of how much the
    /// stack has grown, and which named objects have been added to the stack,
    /// since the current scope began.
    /// Useful for generating stack-cleanup code when the target program's
    /// control flow actually exits the scope.
    void push_scope(const std::string &debug_comment = "");

    /// Returns the number of bytes occupied by the current scope on the
    /// stack. Computed as (current stack size) minus (stack size when the
    /// current top scope was pushed).
    size_t get_top_scope_size() const;

    /// Removes the top scope from the stack, and shrinks the stack size
    /// accordingly. The stack shrinkage is equivalent to calling
    /// \c shrink(get_top_scope_size()).
    /// Returns the number of bytes removed from the stack.
    /// It is an error to call this when the stack contains no scope.
    size_t pop_scope();

    /// Reduce the size of the stack by the specified number of bytes.
    /// Remove all named objects in the freed region of the stack.
    /// Raise an exception if anything seems strange, given the intended usage.
    void shrink(size_t num_bytes);

    /// Eliminate as many stack items as necessary to reach the indicated stack
    /// size. Also remove any lexical scopes that are at the top of the stack
    /// once the stack size is \p new_size.
    ///
    /// It's an error to specify a new stack size that would require partial
    /// deletion of a stack item.
    ///
    /// \param new_size The new size (in bytes) of the modeled stack.
    /// Cannot be greater than the current size (see \c get_size()).
    void shrink_to_size(size_t new_size);

    /// Size of the stack. I.e.: (%rbp - %rsp)
    size_t get_size() const;

    /// If \p name refers to an existing \c named_slot or
    /// \c named_tensor_buffer_slot object, return a pointer to that object.
    /// Otherwise return \c nullptr.
    /// NOTE! ANY modification to this \c stack_frame_model may invalidate
    /// this pointer.
    const named_slot *try_get_named_slot(const std::string &name) const;

    /// Get the solt on top of stack
    const slot *get_top_slot() const;

    /// Return the stack size that resulted from pushing the specified object.
    /// Used for computing the %rbp-relative address of that object at runtime.
    /// This value will always be negative, because no x86-64 the call stack
    /// grows downward.
    int64_t get_named_object_rbp_offset(const std::string &name) const;

    /// If the stack has a slot named \p name, set \p offset accordingly and
    /// return \c true. Otherwise return \c false.
    bool try_get_named_object_rbp_offset(
            const std::string &name, int64_t &rbp_offset) const;

    /// Reset the stack frame model to a fully empty state.
    void clear();

    /// Prints a human-readable, ASCII-art representation of the current state
    /// of the stack model. Useful for debugging.
    void dump(std::ostream &os) const;

    /// Returns a human-readable, single-line summary of the current state of
    /// the stack model. Useful for inclusion in log file and assembly
    /// annontations.
    std::string one_line_summary() const;

private:
    /// If \p name is already a key in
    /// \c name_to_caller_param_slot_, report an error via
    /// \c COMPILE_ASSERT. Otherwise do nothing.
    void assert_unused_name(const std::string &name);

    std::vector<std::unique_ptr<stack_item>> stack_;

    void pop_top();

    /// This map contains an entry for each function argument that was passed
    /// via a slot on the call stack, instead of via a register.
    std::map<std::string, caller_param_slot> name_to_caller_param_slot_;

    bool logging_enabled_;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
