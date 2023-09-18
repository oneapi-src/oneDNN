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
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <utility>

#include <mutex>
#include <compiler/ir/pass/printer.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/jit/xbyak/backend/operations.hpp>
#include <compiler/jit/xbyak/debug/debug_info_mgr.hpp>
#include <compiler/jit/xbyak/ir/transform/call_transform.hpp>
#include <compiler/jit/xbyak/ir/transform/register_allocation.hpp>
#include <compiler/jit/xbyak/ir/util/utils.hpp>
#include <compiler/jit/xbyak/x86_64/type_mapping.hpp>
#include <compiler/jit/xbyak/xbyak_jit.hpp>
#include <util/optional.hpp>
#include <util/utils.hpp>

#include "xbyak_lowering_viewer.hpp"

SC_MODULE(xbyakjit.xbyak_lowering_viewer)

using std::endl;
using std::ostringstream;
using std::string;
using std::vector;

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

using namespace utils;
using namespace xbyak::x86_64;

static const bool log_module_info_enabled
        = bool(runtime::get_info_logging_stream(__sc_module_name));

// An ostream-expression fragment to briefly summarize the current content
// of the stack model.
#define SM_BRIEF " sm:" << sf_model_.one_line_summary()

//==============================================================================
// MACROS / DEFINITIONS FOR (CONDITIONAL) GENERATION OF ASM LISTINGS
//==============================================================================

// Adds an asm-listing comment at the current Xbyak insertion point, using
// the current asm-listing indentation level.
#define ASM_COMMENT(S1, ...) \
    if (utils::compiler_configs_t::get().xbyak_jit_asm_listing_) { \
        ostringstream os; \
        os << asm_listing_ind_ << S1 __VA_ARGS__; \
        add_code_comment(os.str()); \
    }

#define ASM_COMMENT_WITH_IR(NODE, S1, ...) \
    if (utils::compiler_configs_t::get().xbyak_jit_asm_listing_) { \
        ostringstream os; \
        os << asm_listing_ind_ << S1 __VA_ARGS__; \
        add_code_comment(os.str(), NODE.get()); \
    }

// Like ASM_COMMENT, but add simd constant and dtype in the output text.
#define ASM_CONSTANT_COMMENT(MAP) \
    if (utils::compiler_configs_t::get().xbyak_jit_asm_listing_) { \
        for (const auto &kv : (MAP)) { \
            ostringstream os; \
            os << asm_listing_ind_ << c2s(kv.first); \
            code_comments_.emplace_back(kv.second, os.str()); \
        } \
    }

xbyak_lowering_viewer::code_comment::code_comment(
        const Xbyak::Label &label, std::string comment)
    : label_(label), comment_(std::move(comment)) {}

void xbyak_lowering_viewer::add_code_comment(std::string text) {
    code_comments_.emplace_back(Xbyak::Label(), std::move(text));
    gen_->L(code_comments_.back().label_);
}

std::mutex debug_info_lock;

std::vector<std::unique_ptr<debug_info_mgr>>
xbyak_lowering_viewer::dump_code_comments(std::ostream &os) {
    std::vector<debug_line> lines;
    std::vector<func_symbol> symbols;
    // Use a map to sort the comments by memory address, to help humans relate
    // the dump output to gdb's asm display of the JIT'ed code.
    for (auto &iter : func_name_to_entry_label_) {
        const auto &key = iter.first;
        void *entry = (void *)(func_name_to_entry_label_[key].getAddress());
        void *exit = (void *)(func_name_to_exit_label_[key].getAddress());
        if (entry == nullptr || exit == nullptr) { continue; }
        os << utils::as_hex_t(entry) << " - " << utils::as_hex_t(exit) << endl;
        symbols.emplace_back(func_symbol {key, entry, exit});
    }

    for (auto &line : debug_lines_) {
        void *p = (void *)(line.label_.getAddress());
        const source_pos *src
                = some_opt(line.ir_node_)
                          .map([](const node_base *p) {
                              return p->attr_.get();
                          })
                          .map([](const any_map_t *v) {
                              return v->get_or_null<source_pos>("source_pos");
                          })
                          .get_or_else(nullptr);
        if (src) { lines.emplace_back(debug_line {p, src->line_, src->pos_}); }
    }

    std::map<void *, std::vector<std::string>> m;
    for (auto &item : code_comments_) {
        void *p = (void *)(item.label_.getAddress());
        m[p].push_back(item.comment_);
    }

    for (const auto &item : m) {
        bool first_line = true;

        ostringstream os_a;
        os_a << utils::as_hex_t(item.first);
        const string a = os_a.str();

        for (auto &comment : item.second) {
            if (first_line) {
                os << a;
                first_line = false;
            } else {
                os << string(a.size(), ' ');
            }
            os << " : " << comment << endl;
        }
    }
    std::lock_guard<std::mutex> guard {debug_info_lock};
    return create_debug_info_mgr((void *)gen_->getCode(), gen_->getSize(),
            "xbyak_ir.txt", symbols, lines);
}

//==============================================================================
// MACRO DEFINITIONS TO SUPPORT LOGGING TO 'SC_MODULE_INFO'.
//
// NOTE:
// - This is performance critical code, so in some cases we check to see if
//   SC_MODULE_INFO is actually enabled before executing logging-specific code.
//==============================================================================

// Increases the logging indentation until the end of the lexical scope in
// which this macro is invoked.
#define LOGGING_INDENT auto __ind__ = logging_ind_.indent();

#define LOGGING_OUT SC_MODULE_INFO << logging_ind_

#define FUNC_INFO "[" << brief_pretty_function(__PRETTY_FUNCTION__) << "] "

// Emits a single line of text to the logging output.
#define LOG_LINE(S1, ...) \
    LOGGING_OUT << "[" << brief_lineloc(__FILE__, __LINE__) << "]" << SM_BRIEF \
                << " " << S1 __VA_ARGS__;

// Emits a single line of logging output describing the code location
// at which the macro was invoked.
#define LOG_METHOD_BODY \
    LOGGING_INDENT; \
    { \
        ostringstream os; \
        os << brief_pretty_function(__PRETTY_FUNCTION__) << "[" \
           << brief_lineloc(__FILE__, __LINE__) << "]"; \
        os << SM_BRIEF; \
        LOGGING_OUT << replace_newlines(os.str(), " \\ "); \
    }

// Similar to LOG_METHOD_BODY, but also logs the string representation of
// the variable/parameter 'v' at the macro's point of invocation.
// Useful for IR-node-handling methods, which often use the symbol
// 'v' to indicate the IR node being handled.
#define LOG_METHOD_BODY_V \
    LOGGING_INDENT; \
    { \
        ostringstream os; \
        os << brief_pretty_function(__PRETTY_FUNCTION__) << "[" \
           << brief_lineloc(__FILE__, __LINE__) << "]"; \
        os << SM_BRIEF; \
        os << " v=" << v; \
        LOGGING_OUT << replace_newlines(os.str(), " \\ "); \
    }

// Add more info to constant asm comment
std::string c2s(const expr_c &e) {
    std::stringstream ss;
    auto v = e.static_as<constant_c>();
    ss << v << " [";
    for (unsigned i = 0; i < v->value_.size(); i++) {
        ss << std::uppercase << std::hex << "0x" << v->value_[i].u64;
        if (i != v->value_.size() - 1) { ss << ',' << ' '; }
    }
    ss << "] ";
    ss << "[" << v->dtype_ << "]";
    return ss.str();
}

//==============================================================================
// CODEGEN MARCO
//==============================================================================

#define GET_OPERAND(ARG) (location_manager_->get_operand(ARG))
#define GET_OPERAND_SIB(...) (location_manager_->get_operand_sib(__VA_ARGS__))
#define XBYAK_GEN(INS, PATTERN, ...) \
    SC_EXPAND(PATTERN(*gen_, INS, ##__VA_ARGS__))
// SC_EXPAND because of MSVC:
// https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly

//==============================================================================
// SPECIAL MEMBER FUNCTION SECTION
//==============================================================================

xbyak_lowering_viewer::xbyak_lowering_viewer(const xbyak_jit &xje,
        const ir_module_t &ir_mod, const x86_64::target_profile_t &profile)
    : xje_(xje)
    , p_ir_mod_(&ir_mod)
    , profile_(profile)
    , cpu_flags_(profile.target_machine_.cpu_flags_)
    , sf_model_(
              utils::compiler_configs_t::get().xbyak_jit_log_stack_frame_model_)
    , logging_ind_(4)
    , asm_listing_ind_(4) {
    gen_.reset(new xbyak_jit_generator);
    location_manager_.reset(new location_manager(sf_model_, *gen_, profile_));

    // Set SIMD level
    if (cpu_flags_.fAVX512F) {
        simd_level_ = simd_level::avx512;
    } else if (cpu_flags_.fAVX2) {
        simd_level_ = simd_level::avx2;
    } else if (cpu_flags_.fAVX) {
        simd_level_ = simd_level::avx;
    } else {
        assert(cpu_flags_.fSSE42);
        simd_level_ = simd_level::sse;
    }
    // Prepopulate the function-name symbol table, to let us lower a caller
    // before lowering its callee...
    for (const func_t &f : ir_mod.get_contents()) {
        func_name_to_entry_label_[f->name_] = Xbyak::Label();
        func_name_to_exit_label_[f->name_] = Xbyak::Label();
    }
    // default jmp type tp T_NEAR
    gen_->setDefaultJmpNEAR(true);
    // if support avx512, use EvexEncoding
    gen_->setDefaultEncoding(simd_level_ == simd_level::avx512
                    ? Xbyak::EvexEncoding
                    : Xbyak::VexEncoding);

    // ir_mod.get_module_vars() have been resolved in __module_data

    // Manage module funcs generate
    for (const func_t &f : ir_mod.get_contents()) {
        // Do not generate should_inline functions
        if (f->name_.find("_should_inline_") != string::npos) { continue; }
        dispatch(f);
    }

    // We don't use jit_generator::getCode() because we potentially have
    // multiple functions defined in this module.
    gen_->readyRE();

    SC_MODULE_INFO << "simd level: " << (int)simd_level_;
    for (const auto &iter : func_name_to_entry_label_) {
        void *entry_point = (void *)(iter.second.getAddress());
        gen_->func_name_to_address_[iter.first] = entry_point;

        SC_MODULE_INFO << "function '" << iter.first << "' has entry address "
                       << utils::as_hex_t(entry_point);
    };
    auto &config = utils::compiler_configs_t::get();
    if (config.xbyak_jit_pause_after_codegen_) {
        std::cout << "Hit ENTER to continue." << std::endl;
        std::cin.get();
    }

    if (config.xbyak_jit_asm_listing_) {
        string fname = "asm_comments.txt";
        std::ofstream f(fname);
        f << "Code comments:" << endl;
        gen_->debug_info_ = dump_code_comments(f);
    }

    if (config.xbyak_jit_save_obj_) {
        // Save the generated object code to a file.
        // Embed the code's starting virtual address in the
        // filename, to help with disassembly.
        const uint8_t *p_obj = gen_->getCode();
        const size_t obj_size = gen_->getSize();

        std::ostringstream os;
        os << "jit-obj-from-" << as_hex_t(p_obj) << ".bin";
        std::ofstream f(os.str(), std::ios::binary);
        f.write((const char *)(p_obj), obj_size);
    }

    p_ir_mod_ = nullptr;
}

xbyak_lowering_viewer::~xbyak_lowering_viewer() = default;

//==============================================================================
// MEMBER FUNCTION SECTION
//==============================================================================

std::shared_ptr<xbyak_jit_generator>
xbyak_lowering_viewer::get_jit_output() const {
    return gen_;
}

const std::set<virt_reg_index_t> &
xbyak_lowering_viewer::cached_func_register_usage(const func_t &v) {
    assert(v->attr_ && v->attr_->has_key(attr_keys::register_usage));
    const auto &register_usage = v->attr_->get<std::set<virt_reg_index_t>>(
            attr_keys::register_usage);
    return register_usage;
}

const std::vector<expr_c> &xbyak_lowering_viewer::cached_func_global_spilled(
        const func_t &v) {
    assert(v->attr_ && v->attr_->has_key(attr_keys::global_spilled));
    const auto &global_spilled
            = v->attr_->get<std::vector<expr_c>>(attr_keys::global_spilled);
    return global_spilled;
}

void xbyak_lowering_viewer::handle_func_resolve(const std::string &name,
        const execute_func_label &label_f, const execute_func_addr &addr_f) {
    const auto ir_sym_table_iter = func_name_to_entry_label_.find(name);

    if (ir_sym_table_iter != func_name_to_entry_label_.end()) {
        const Xbyak::Label &callee_label = ir_sym_table_iter->second;
        label_f(callee_label);
    } else if (const void *callee_addr = xje_.external_symbol_resolver_(name)) {
        addr_f(reinterpret_cast<uint64_t>(callee_addr));
    } else {
        COMPILE_ASSERT(false,
                "Address/label lookup failed for function name \"" << name
                                                                   << "\"");
    }
}

void xbyak_lowering_viewer::handle_local_definition(
        const expr_c &v, const expr_c &v_init) {
    location_manager_->handle_definition(v);
    if (v_init.defined()) { handle_operations(v, v_init); }
}

//==============================================================================
// DISPATCH MEMBER FUNCTION SECTION
//==============================================================================

stmt_c xbyak_lowering_viewer::dispatch(stmt_c v) {
    if (utils::compiler_configs_t::get().xbyak_jit_asm_listing_) {
        debug_lines_.emplace_back(label_line {gen_->L(), v.get()});
    }
    stmt_c vv;
    auto &stmt_data = GET_STMT_DATA(v);
    if (!stmt_data.optimized_out_) { vv = ir_viewer_t::dispatch(std::move(v)); }
    location_manager_->expire(stmt_data.index_);
    return vv;
}

func_c xbyak_lowering_viewer::dispatch(func_c v) {
    // FIXME: For now we're assuming that the only function in the supplied
    // ir_module_t is the entry function.
    // FIXME: We're also assuming that although the entry function's parameters
    // are fully typed, right now we're going to emit the generic wrapper
    // version of the function.

    sf_model_.clear();
    l_func_epilogue_.clear();

    func_t func = std::const_pointer_cast<func_base>(v);
    const auto &register_usage = cached_func_register_usage(func);
    const auto &global_spilled = cached_func_global_spilled(func);
    func_iface_ = cached_func_abi_interface(func);

    gen_->align(16);
    gen_->L(func_name_to_entry_label_[v->name_]);
    ASM_COMMENT("function entry: <" << v->name_ << ">");

    // Save caller-save registers, and create a stack frame
    // After all argument-carrying and callee-save registers have been copied
    // onto the stack, we can use whichever registers (within reason) we want.
    ASM_COMMENT("prologue");
    location_manager_->clear();
    location_manager_->emit_callee_prologue(register_usage);
    location_manager_->handle_func_params(v->params_, *func_iface_);
    location_manager_->handle_spilled_definition(global_spilled);

    dispatch(v->body_);

    gen_->L(l_func_epilogue_);
    ASM_COMMENT("epilogue");
    location_manager_->emit_callee_epilogue();

    gen_->ret();
    gen_->L(func_name_to_exit_label_[v->name_]);
    ASM_COMMENT("function exit: <" << v->name_ << ">");

    // Encode all simd constants inside code after ret
    auto &constant_map = location_manager_->encode_simd_constant();
    ASM_CONSTANT_COMMENT(constant_map);

    return std::const_pointer_cast<func_base>(v);
}

//==============================================================================
// HANDLE OPERATIONS MEMBER FUNCTION SECTION
//==============================================================================

void xbyak_lowering_viewer::handle_operations(
        const expr_c &lhs, const expr_c &rhs) {
    switch (rhs->node_type_) {
        case sc_expr_type::var:
        case sc_expr_type::tensor:
        case sc_expr_type::constant:
        case sc_expr_type::indexing: {
            handle_assign(lhs, rhs);
        } break;
        case sc_expr_type::cast: {
            handle_cast(lhs, rhs.static_as<cast_c>());
        } break;
        case sc_expr_type::call: {
            handle_call(lhs, rhs.static_as<call_c>());
        } break;
        case sc_expr_type::func_addr: {
            handle_func_addr(lhs, rhs.static_as<func_addr_c>());
        } break;
        case sc_expr_type::tensorptr: {
            handle_tensorptr(lhs, rhs.static_as<tensorptr_c>());
        } break;
        case sc_expr_type::low_level_intrin: {
            handle_xbyak_intrin(lhs, rhs.checked_as<xbyak_intrin_c>());
        } break;
        default: {
            COMPILE_ASSERT(false, "Not supported op: " << lhs << " = " << rhs);
        } break;
    }
}

void xbyak_lowering_viewer::handle_xbyak_intrin(
        const expr_c &lhs, const xbyak_intrin_c &rhs) {
    auto intrin = static_cast<xbyak_intrin_type>(rhs->type_);
    switch (intrin) {
        case xbyak_intrin_type::call_arg: {
            location_manager_->handle_call_arg(lhs, rhs->args_[0]);
        } break;
        case xbyak_intrin_type::reinterpret: {
            handle_reinterpret(lhs, rhs->args_[0]);
        } break;
        case xbyak_intrin_type::saturated_cast: {
            handle_saturated_cast(lhs, rhs->args_[0]);
        } break;
        case xbyak_intrin_type::round_and_cast: {
            handle_round_and_cast(lhs, rhs->args_[0]);
        } break;
        default: {
            ASM_COMMENT(lhs << " = " << rhs);
            switch (rhs->isa_) {
                case xbyak_intrin_isa::x86: {
                    handle_x86_intrisic(
                            lhs, rhs->args_, intrin, rhs->modifier_);
                } break;
                case xbyak_intrin_isa::avx: {
                    handle_avx_intrisic(
                            lhs, rhs->args_, intrin, rhs->modifier_);
                } break;
                default: {
                    COMPILE_ASSERT(false, FUNC_INFO << "Invalid isa.");
                } break;
            }
        } break;
    }
}

void xbyak_lowering_viewer::handle_x86_intrisic(const expr_c &dst,
        array_ref<expr> args, const xbyak_intrin_type &intrin,
        const xbyak_intrin_modifier &modifier) {
    // Get low level data type
    const cpu_data_type cpu_dtype = get_cpu_data_type(dst->dtype_);
    // Generate x86 intrisics
    switch (intrin) {
        case xbyak_intrin_type::add: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            XBYAK_GEN(add, X86_RM_RMI, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::sub: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            XBYAK_GEN(sub, X86_RM_RMI, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::bit_or: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            XBYAK_GEN(or_, X86_RM_RMI, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::bit_and: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            XBYAK_GEN(and_, X86_RM_RMI, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::bit_xor: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            XBYAK_GEN(xor_, X86_RM_RMI, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::shl: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            XBYAK_GEN(shl, X86_RM_R8I, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::shr: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            XBYAK_GEN(shr, X86_RM_R8I, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::sar: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            XBYAK_GEN(sar, X86_RM_R8I, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::mul: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            XBYAK_GEN(imul, X86_R_RM, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::muli: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            XBYAK_GEN(imul, X86_R_RM_I, op_dst, op_lhs, op_rhs);
        } break;
        case xbyak_intrin_type::mulhl: {
            // %rdx:%rax = mulhl([0]%rax, [1]src)
            auto op_dst = GET_OPERAND(dst);
            auto op_rax = GET_OPERAND(args[0]);
            auto op_src = GET_OPERAND(args[1]);
            assert(op_dst == operand(regs::rdx));
            assert(op_rax == operand(regs::rax));
            switch (get_type_category(dst->dtype_)) {
                case type_category::CATE_INT: {
                    XBYAK_GEN(imul, X86_RM, op_src);
                } break;
                case type_category::CATE_UINT: {
                    XBYAK_GEN(mul, X86_RM, op_src);
                } break;
                default: COMPILE_ASSERT(false, "x86 mulhl type error");
            }
        } break;
        case xbyak_intrin_type::sign_ext: {
            // %rdx = x86_sign_ext([0]%rax)
            auto op_rdx = GET_OPERAND(dst);
            handle_x86_sign_ext(op_rdx, cpu_dtype);
        } break;
        case xbyak_intrin_type::div: {
            // dst = x86_div([0]rhs, [1]%rax, [2]%rdx)
            auto op_dst = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            auto op_rax = GET_OPERAND(args[1]);
            handle_x86_div(op_rhs, cpu_dtype);
            handle_x86_mov(op_dst, op_rax);
        } break;
        case xbyak_intrin_type::mod: {
            // dst = x86_mod([0]rhs, [1]%rax, [2]%rdx)
            auto op_dst = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            auto op_rdx = GET_OPERAND(args[2]);
            handle_x86_div(op_rhs, cpu_dtype);
            handle_x86_mov(op_dst, op_rdx);
        } break;
        case xbyak_intrin_type::min: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            // compare order: (rhs, lhs)
            handle_x86_cmp(op_rhs, op_lhs);
            handle_x86_cmov(op_lhs, op_rhs, xbyak_condition::lt, cpu_dtype);
        } break;
        case xbyak_intrin_type::max: {
            auto op_lhs = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            // compare order: (rhs, lhs)
            handle_x86_cmp(op_rhs, op_lhs);
            handle_x86_cmov(op_lhs, op_rhs, xbyak_condition::gt, cpu_dtype);
        } break;
        case xbyak_intrin_type::test: {
            // test single condition
            assert(dst.ptr_same(args[0]));
            assert(cpu_dtype == cpu_data_type::uint_8);
            auto op_cond = GET_OPERAND(dst);
            handle_x86_test(op_cond);
        } break;
        case xbyak_intrin_type::neg: {
            auto op_dst = GET_OPERAND(dst);
            XBYAK_GEN(neg, X86_RM, op_dst);
        } break;
        case xbyak_intrin_type::cmov: {
            const auto &code = modifier.cond_code_;
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_x86_cmov(op_dst, op_src, code, cpu_dtype);
        } break;
        case xbyak_intrin_type::cmp_set: {
            const auto cmp_dtype = get_cpu_data_type(args[0]->dtype_);
            const auto &code = modifier.cond_code_;
            assert(code != xbyak_condition::none);
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_x86_cmp(op_lhs, op_rhs);
            handle_x86_set(op_dst, code, cmp_dtype);
        } break;
        case xbyak_intrin_type::bmi_pext: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            XBYAK_GEN(pext, X86_R64_R64_R64, op_dst, op_lhs, op_rhs);
        } break;
        default: {
            COMPILE_ASSERT(false,
                    FUNC_INFO << "Invalid intrisic: "
                              << "intrin");
        } break;
    }
}

void xbyak_lowering_viewer::handle_avx_intrisic(const expr_c &dst,
        array_ref<expr> args, const xbyak_intrin_type &intrin,
        const xbyak_intrin_modifier &modifier) {
    // Get low level data type
    const cpu_data_type cpu_dtype = get_cpu_data_type(dst->dtype_);
    // Generate avx intrisics
    switch (intrin) {
        case xbyak_intrin_type::add: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_add(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::sub: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_sub(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::mul: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_mul(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::mulhl: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_mulhl(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::div: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_div(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::shl: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_sft = GET_OPERAND(args[1]);
            auto variable = args[1]->dtype_.lanes_ > 1;
            const auto src_dtype = get_cpu_data_type(modifier.type_hint_);
            handle_avx_shl(op_dst, op_lhs, op_sft, src_dtype, variable);
        } break;
        case xbyak_intrin_type::shr: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_sft = GET_OPERAND(args[1]);
            auto variable = args[1]->dtype_.lanes_ > 1;
            const auto src_dtype = get_cpu_data_type(modifier.type_hint_);
            handle_avx_shr(op_dst, op_lhs, op_sft, src_dtype, variable);
        } break;
        case xbyak_intrin_type::sar: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_sft = GET_OPERAND(args[1]);
            auto variable = args[1]->dtype_.lanes_ > 1;
            const auto src_dtype = get_cpu_data_type(modifier.type_hint_);
            handle_avx_sar(op_dst, op_lhs, op_sft, src_dtype, variable);
        } break;
        case xbyak_intrin_type::min: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_min(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::max: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_max(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::abs: {
            auto op_lst = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            handle_avx_abs(op_lst, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::bit_or: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_bit_or(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::bit_and: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_bit_and(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::bit_xor: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_bit_xor(op_dst, op_lhs, op_rhs, cpu_dtype);
        } break;
        case xbyak_intrin_type::ceil: {
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_avx_round(op_dst, op_src, cpu_dtype, INT64_C(0x2));
        } break;
        case xbyak_intrin_type::floor: {
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_avx_round(op_dst, op_src, cpu_dtype, INT64_C(0x1));
        } break;
        case xbyak_intrin_type::round: {
            auto op_lst = GET_OPERAND(dst);
            auto op_rhs = GET_OPERAND(args[0]);
            handle_avx_round(op_lst, op_rhs, cpu_dtype, INT64_C(0x0));
        } break;
        case xbyak_intrin_type::sqrt: {
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_avx_sqrt(op_dst, op_src, cpu_dtype);
        } break;
        case xbyak_intrin_type::rsqrt: {
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_avx_rsqrt(op_dst, op_src, cpu_dtype);
        } break;
        case xbyak_intrin_type::fmadd: {
            auto op_dst = GET_OPERAND(dst);
            auto op_mul = GET_OPERAND(args[0]);
            auto op_add = GET_OPERAND(args[1]);
            handle_avx_fmadd(op_dst, op_mul, op_add, cpu_dtype);
        } break;
        case xbyak_intrin_type::broadcast: {
            const auto src_dtype = get_cpu_data_type(modifier.type_hint_);
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_avx_broadcast(op_dst, op_src, cpu_dtype, src_dtype);
        } break;
        case xbyak_intrin_type::pshuffle: {
            const auto src_dtype = get_cpu_data_type(modifier.type_hint_);
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_pshuffle(op_dst, op_lhs, op_rhs, src_dtype);
        } break;
        case xbyak_intrin_type::shuffle: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            auto op_imm = GET_OPERAND(args[2]);
            auto op_type_bits = GET_OPERAND(args[3]);
            handle_avx_shuffle(op_dst, op_lhs, op_rhs, op_imm, op_type_bits);
        } break;
        case xbyak_intrin_type::permute: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            auto op_imm = GET_OPERAND(args[2]);
            handle_avx_permute(op_dst, op_lhs, op_rhs, op_imm, cpu_dtype);
        } break;
        case xbyak_intrin_type::gather: {
            auto &mask = modifier.cond_mask_;
            assert(mask.defined());
            auto op_msk = GET_OPERAND(mask);
            auto op_dst = GET_OPERAND(dst);
            auto op_ptr = GET_OPERAND(args[0]);
            auto op_idx = GET_OPERAND(args[1]);
            handle_avx_gather(op_dst, op_ptr, op_idx, op_msk, cpu_dtype);
        } break;
        case xbyak_intrin_type::insert: {
            auto op_dst = GET_OPERAND(dst);
            auto op_b = GET_OPERAND(args[0]);
            auto op_imm = GET_OPERAND(args[1]);
            auto op_elem_bits = GET_OPERAND(args[2]);
            handle_avx_insert(op_dst, op_b, op_imm, op_elem_bits);
        } break;
        case xbyak_intrin_type::extract: {
            auto op_dst = GET_OPERAND(dst);
            auto op_b = GET_OPERAND(args[0]);
            auto op_imm = GET_OPERAND(args[1]);
            auto op_elem_bits = GET_OPERAND(args[2]);
            handle_avx_extract(op_dst, op_b, op_imm, op_elem_bits);
        } break;
        case xbyak_intrin_type::permutex2var: {
            auto op_dst = GET_OPERAND(dst);
            auto op_idx = GET_OPERAND(args[0]);
            auto op_src = GET_OPERAND(args[1]);
            handle_avx_permutex2var(op_dst, op_idx, op_src, cpu_dtype);
        } break;
        case xbyak_intrin_type::permutexvar: {
            auto op_dst = GET_OPERAND(dst);
            auto op_idx = GET_OPERAND(args[0]);
            auto op_src = GET_OPERAND(args[1]);
            operand bits = GET_OPERAND(args[2]);
            handle_avx_permutexvar(op_dst, op_idx, op_src, cpu_dtype, bits);
        } break;
        case xbyak_intrin_type::unpack_low: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            auto op_imm = GET_OPERAND(args[2]);
            handle_avx_unpack_low(op_dst, op_lhs, op_rhs, op_imm);
        } break;
        case xbyak_intrin_type::unpack_high: {
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            auto op_imm = GET_OPERAND(args[2]);
            handle_avx_unpack_high(op_dst, op_lhs, op_rhs, op_imm);
        } break;
        case xbyak_intrin_type::extract_low: {
            const auto in_dtype = get_cpu_data_type(modifier.type_hint_);
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_avx_extract_low(op_dst, op_src, in_dtype);
        } break;
        case xbyak_intrin_type::extract_high: {
            const auto in_dtype = get_cpu_data_type(modifier.type_hint_);
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_avx_extract_high(op_dst, op_src, in_dtype);
        } break;
        case xbyak_intrin_type::cmov: {
            const auto &code = modifier.cond_code_;
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_avx_cmov(op_dst, op_src, code, cpu_dtype);
        } break;
        case xbyak_intrin_type::movd: {
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            XBYAK_GEN(vmovd, AVX_XMR32_XMR32, op_dst, op_src)
        } break;
        case xbyak_intrin_type::mask_mov: {
            auto &msk_cond = modifier.cond_mask_;
            assert(msk_cond.defined());
            auto op_cond = GET_OPERAND(msk_cond);
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            handle_avx_mask_mov(
                    op_dst, op_src, op_cond, cpu_dtype, modifier.zero_mask_);
        } break;
        case xbyak_intrin_type::blend: {
            // Reminder: operand order is reversed comparing to select(l,r)
            auto &msk_cond = modifier.cond_mask_;
            assert(msk_cond.defined());
            auto op_cond = GET_OPERAND(msk_cond);
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_blend(op_dst, op_lhs, op_rhs, op_cond, cpu_dtype);
        } break;
        case xbyak_intrin_type::cmp_set: {
            auto &code = modifier.cond_code_;
            assert(code != xbyak_condition::none);
            const auto cmp_dtype = get_cpu_data_type(modifier.type_hint_);
            auto op_dst = GET_OPERAND(dst);
            auto op_lhs = GET_OPERAND(args[0]);
            auto op_rhs = GET_OPERAND(args[1]);
            handle_avx_cmp_set(op_dst, op_lhs, op_rhs, code, cmp_dtype);
        } break;
        case xbyak_intrin_type::mov_mask: {
            auto op_dst = GET_OPERAND(dst);
            auto op_src = GET_OPERAND(args[0]);
            const auto src_dtype = get_cpu_data_type(modifier.type_hint_);
            handle_avx_mov_mask(op_dst, op_src, src_dtype);
        } break;
        default: {
            COMPILE_ASSERT(false,
                    FUNC_INFO << "Invalid intrisic: "
                              << "intrin");
        } break;
    }
}

//==============================================================================
// GENERAL OPERATIONS MEMBER FUNCTION SECTION
//==============================================================================

void xbyak_lowering_viewer::handle_assign(
        const expr_c &lhs, const expr_c &rhs) {
    auto cpu_dtype = get_cpu_data_type(rhs->dtype_);

    auto rhs_op = GET_OPERAND(rhs);
    auto lhs_op = GET_OPERAND(lhs);

    switch (cpu_dtype) {
        case cpu_data_type::uint_8:
        case cpu_data_type::sint_8:
        case cpu_data_type::uint_16:
        case cpu_data_type::uint_32:
        case cpu_data_type::sint_32:
        case cpu_data_type::uint_64: {
            // 8,16,32,64-bit scalar mov
            handle_x86_mov(lhs_op, rhs_op);
        } break;
        case cpu_data_type::float_16: {
            handle_avx_movsh(lhs_op, rhs_op);
        } break;
        case cpu_data_type::float_32: {
            // 32-bit simd mov
            handle_avx_movss(lhs_op, rhs_op);
        } break;
        case cpu_data_type::uint_8_x8:
        case cpu_data_type::sint_8_x8:
        case cpu_data_type::uint_16_x4:
        case cpu_data_type::sint_32_x2:
        case cpu_data_type::float_16_x4:
        case cpu_data_type::float_32_x2: {
            // 64-bit simd mov
            handle_avx_movq(lhs_op, rhs_op);
        } break;
        case cpu_data_type::uint_8_x16:
        case cpu_data_type::uint_8_x32:
        case cpu_data_type::uint_8_x64:
        case cpu_data_type::sint_8_x16:
        case cpu_data_type::sint_8_x32:
        case cpu_data_type::sint_8_x64: {
            handle_avx_movps(lhs_op, rhs_op);
        } break;
        case cpu_data_type::uint_16_x8:
        case cpu_data_type::uint_16_x16:
        case cpu_data_type::uint_16_x32: {
            handle_avx_movps(lhs_op, rhs_op);
        } break;
        case cpu_data_type::uint_32_x4:
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::uint_32_x16:
        case cpu_data_type::sint_32_x4:
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::sint_32_x16:
        case x86_64::cpu_data_type::uint_64_x2:
        case x86_64::cpu_data_type::uint_64_x4:
        case x86_64::cpu_data_type::uint_64_x8: {
            handle_avx_movps(lhs_op, rhs_op);
        } break;
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_32_x4:
        case cpu_data_type::float_32_x8:
        case cpu_data_type::float_32_x16: {
            handle_avx_movps(lhs_op, rhs_op);
        } break;
        case cpu_data_type::mask_x4:
        case cpu_data_type::mask_x8:
        case cpu_data_type::mask_x16:
        case cpu_data_type::mask_x32:
        case cpu_data_type::mask_x64: {
            handle_avx512_kmov(lhs_op, rhs_op, cpu_dtype);
        } break;
        default: {
            COMPILE_ASSERT(false,
                    FUNC_INFO << "Invalid type: " << cpu_dtype << " " << lhs
                              << " = " << rhs);
        }
    }
}

void xbyak_lowering_viewer::handle_tensorptr(
        const expr_c &lhs, const tensorptr_c &rhs) {
    auto cpu_dtype = get_cpu_data_type(lhs->dtype_);

    COMPILE_ASSERT(cpu_dtype == cpu_data_type::uint_64,
            "Invlaid tensorptr dst type" << lhs);
    COMPILE_ASSERT(lhs.isa<tensor>() || lhs.isa<var>(),
            "Invalid assign_from_tensorptr lvalue node type: " << lhs);

    const auto &base = rhs->base_;
    const auto &ptr = base->ptr_;
    const auto &idx = base->idx_.back();

    if (const_exceed_32bit(idx)) {
        const tensor_c tsr = ptr.dyn_as<tensor_c>();
        assert(tsr.defined());

        auto op_lhs = GET_OPERAND(lhs);
        auto op_tsr = GET_OPERAND(tsr);
        auto op_idx = GET_OPERAND(idx);

        auto scale = location_manager_->get_data_type_size(
                get_cpu_data_type(tsr->elem_dtype_));

        handle_x86_mov(op_lhs, operand((int64_t)scale * op_idx.get_imm()));
        XBYAK_GEN(add, X86_RM_RMI, op_lhs, op_tsr);
    } else {
        auto op_lhs = GET_OPERAND(lhs);
        auto op_rhs = GET_OPERAND(base);

        XBYAK_GEN(lea, X86_R64_M, op_lhs, op_rhs);
    }
}

void xbyak_lowering_viewer::handle_func_addr(
        const expr_c &lhs, const func_addr_c &rhs) {
    auto cpu_dtype = get_cpu_data_type(lhs->dtype_);

    COMPILE_ASSERT(cpu_dtype == cpu_data_type::uint_64,
            "Invlaid func_addr dst type" << lhs);
    COMPILE_ASSERT(lhs.isa<var>(),
            "Invalid assign_from_func_addr lvalue node type: " << lhs);

    auto func_name = rhs->func_->name_;
    auto op_lhs = GET_OPERAND(lhs);

    handle_func_resolve(
            func_name,
            [&](const Xbyak::Label &label) {
                // mov label
                gen_->mov(to_reg64(op_lhs.get_reg()), label);
            },
            [&](const uint64_t &addr) {
                // mov addr
                gen_->mov(to_reg64(op_lhs.get_reg()), addr);
            });
}

//==============================================================================
// HANDLE CAST MEMBER FUNCTION SECTION
//==============================================================================
inline bool is_type_macth(sc_data_type_t dst_dtype, sc_data_type_t src_dtype, //
        sc_data_etype dst_etype, sc_data_etype src_etype) {
    return (dst_dtype.type_code_ == dst_etype)
            && (src_dtype.type_code_ == src_etype);
}

template <typename... Args>
inline bool is_lane_macth(sc_data_type_t dst_dtype, sc_data_type_t src_dtype, //
        Args... args) {
    return (dst_dtype.lanes_ == src_dtype.lanes_)
            && utils::is_one_of((int)src_dtype.lanes_, args...);
}

inline uint64_t scalar_bit(sc_data_type_t dtype) {
    return is_x86_simd(dtype) ? UINT64_C(0)
                              : UINT64_C(8) * utils::get_sizeof_type(dtype);
}

void xbyak_lowering_viewer::handle_cast(const expr_c &lhs, const cast_c &v) {
    const sc_data_type_t in_dtype = v->in_->dtype_;
    const sc_data_type_t out_dtype = v->dtype_;

    auto elem_cast_simd = [&](sc_data_etype out_etype, sc_data_etype in_etype) {
        return is_lane_macth(out_dtype, in_dtype, 4, 8, 16, 32, 64)
                && is_type_macth(out_dtype, in_dtype, out_etype, in_etype);
    };

    auto op_in = GET_OPERAND(v->in_);
    auto op_out = GET_OPERAND(lhs);
    if (location_manager_->is_stack_tensor(v->in_)) {
        assert((out_dtype.is_pointer() || out_dtype == datatypes::index
                || out_dtype == datatypes::generic));
        XBYAK_GEN(lea, X86_R64_M, op_out, op_in);
    } else if ((out_dtype.is_pointer() || out_dtype == datatypes::index
                       || out_dtype == datatypes::generic)
            && (in_dtype.is_pointer() || in_dtype == datatypes::index
                    || in_dtype == datatypes::generic)) {
        // At the lowered asm level, u64 and all pointers are basically
        // interchangable (assuming we're using 64-bit pointers).
        handle_x86_mov(op_out, op_in);
    } else if ((out_dtype == datatypes::s32 || out_dtype == datatypes::u32
                       || out_dtype == datatypes::s8
                       || out_dtype == datatypes::u8
                       || out_dtype == datatypes::u16)
            && (in_dtype == datatypes::generic
                    || in_dtype == datatypes::index)) {
        handle_x86_mov(op_out, op_in);
    } else if ((out_dtype == datatypes::u8 || out_dtype == datatypes::s8)
            && (in_dtype == datatypes::u32 || in_dtype == datatypes::s32)) {
        handle_x86_mov(op_out, op_in);
    } else if ((out_dtype == datatypes::u16 || out_dtype == datatypes::bf16)
            && (in_dtype == datatypes::u32 || in_dtype == datatypes::s32)) {
        handle_x86_mov(op_out, op_in);
    } else if (out_dtype == datatypes::generic && in_dtype == datatypes::s32) {
        handle_x86_mov(op_out, op_in);
    } else if ((out_dtype == datatypes::u32 || out_dtype == datatypes::u16)
            && in_dtype == datatypes::index) {
        handle_x86_mov(op_out, op_in);
    } else if ((out_dtype == datatypes::s32 || out_dtype == datatypes::u32)
            && (in_dtype == datatypes::u32 || in_dtype == datatypes::s32)) {
        handle_x86_mov(op_out, op_in);
    } else if (out_dtype == datatypes::index && in_dtype == datatypes::s32) {
        XBYAK_GEN(movsxd, X86_R64_RM, op_out, op_in); // sign extension
    } else if (out_dtype == datatypes::index && in_dtype == datatypes::u32) {
        XBYAK_GEN(mov, X86_R32_RM, op_out, op_in); // zero extension
    } else if (out_dtype == datatypes::u32
            && (in_dtype == datatypes::u16 || in_dtype == datatypes::bf16)) {
        XBYAK_GEN(movzx, X86_R_RM, op_out, op_in); // zero extension
    } else if (out_dtype == datatypes::f32 && in_dtype == datatypes::f16) {
        XBYAK_GEN(vcvtsh2ss, AVX_X_X_XM, op_out, op_out, op_in);
    } else if (out_dtype == datatypes::index && in_dtype == datatypes::f16) {
        XBYAK_GEN(vcvtsh2usi, AVX_R64_XM, op_out, op_in);
    } else if (out_dtype == datatypes::s32 && in_dtype == datatypes::f16) {
        XBYAK_GEN(vcvtsh2si, AVX_R32_XM, op_out, op_in);
    } else if (out_dtype == datatypes::u32 && in_dtype == datatypes::f16) {
        XBYAK_GEN(vcvtsh2usi, AVX_R32_XM, op_out, op_in);
    } else if (out_dtype == datatypes::f16 && in_dtype == datatypes::f32) {
        XBYAK_GEN(vcvtss2sh, AVX_X_X_XM, op_out, op_out, op_in);
    } else if (out_dtype == datatypes::f16
            && (in_dtype == datatypes::index || in_dtype == datatypes::u32)) {
        XBYAK_GEN(vcvtusi2sh, AVX_X_X_RM, op_out, op_out, op_in);
    } else if (out_dtype == datatypes::f16 && in_dtype == datatypes::s32) {
        XBYAK_GEN(vcvtsi2sh, AVX_X_X_RM, op_out, op_out, op_in);
    } else if (out_dtype == datatypes::s32 && in_dtype == datatypes::u16) {
        XBYAK_GEN(movzx, X86_R_RM, op_out, op_in); // zero extension
    } else if (out_dtype == datatypes::s32 && in_dtype == datatypes::u16) {
        XBYAK_GEN(movzx, X86_R_RM, op_out, op_in); // zero extension
    } else if (out_dtype == datatypes::s32 && in_dtype == datatypes::u16) {
        XBYAK_GEN(movzx, X86_R_RM, op_out, op_in); // zero extension
    } else if ((out_dtype == datatypes::s32 || out_dtype == datatypes::u32)
            && in_dtype == datatypes::s8) {
        XBYAK_GEN(movsx, X86_R_RM, op_out, op_in); // sign extension
    } else if ((out_dtype == datatypes::s32 || out_dtype == datatypes::index
                       || out_dtype == datatypes::u32)
            && (in_dtype == datatypes::u8 || in_dtype == datatypes::u16)) {
        XBYAK_GEN(movzx, X86_R_RM, op_out, op_in); // zero extension
    } else if (out_dtype == datatypes::f32 && in_dtype == datatypes::generic) {
        XBYAK_GEN(vmovd, AVX_XMR32_XMR32, op_out, op_in);
    } else if (out_dtype == datatypes::s32 && in_dtype == datatypes::f32) {
        XBYAK_GEN(vcvttss2si, AVX_R32_XM, op_out, op_in);
    } else if (out_dtype == datatypes::f32 && in_dtype == datatypes::s32) {
        XBYAK_GEN(vcvtsi2ss, AVX_X_X_RM, op_out, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::S32, sc_data_etype::F32)) {
        XBYAK_GEN(vcvttps2dq, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::F32, sc_data_etype::S32)) {
        XBYAK_GEN(vcvtdq2ps, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::S32, sc_data_etype::S8)) {
        XBYAK_GEN(vpmovsxbd, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::S32, sc_data_etype::U8)) {
        XBYAK_GEN(vpmovzxbd, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::U32, sc_data_etype::U16)) {
        XBYAK_GEN(vpmovzxwd, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::S32, sc_data_etype::U16)) {
        XBYAK_GEN(vpmovzxwd, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::U16, sc_data_etype::U32)) {
        assert(cpu_flags_.fAVX512F);
        XBYAK_GEN(vpmovdw, AVX_XM_X, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::U8, sc_data_etype::S32)) {
        assert(cpu_flags_.fAVX512F);
        XBYAK_GEN(vpmovdb, AVX_XM_X, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::S8, sc_data_etype::S32)) {
        assert(cpu_flags_.fAVX512F);
        XBYAK_GEN(vpmovsdb, AVX_XM_X, op_out, op_in);
    } else if (out_dtype == sc_data_type_t::u16(in_dtype.lanes_ * 2)
            && in_dtype == sc_data_type_t::s32(in_dtype.lanes_)) {
        assert(cpu_flags_.fAVX2);
        XBYAK_GEN(vpackssdw, AVX_X_X_XM, op_out, op_in, op_in);
    } else if (out_dtype == sc_data_type_t::u16(in_dtype.lanes_ * 2)
            && in_dtype == sc_data_type_t::u32(in_dtype.lanes_)) {
        assert(cpu_flags_.fAVX2);
        XBYAK_GEN(vpackusdw, AVX_X_X_XM, op_out, op_in, op_in);
    } else if (elem_cast_simd(sc_data_etype::U8, sc_data_etype::U16)) {
        assert(cpu_flags_.fAVX512F);
        XBYAK_GEN(vpmovdb, AVX_XM_X, op_out, op_in);
    } else if (out_dtype == sc_data_type_t::u8(in_dtype.lanes_ * 2)
            && in_dtype == sc_data_type_t::u16(in_dtype.lanes_)) {
        assert(cpu_flags_.fAVX2);
        XBYAK_GEN(vpackuswb, AVX_X_X_XM, op_out, op_in, op_in);
    } else if (out_dtype == sc_data_type_t::s8(in_dtype.lanes_ * 2)
            && in_dtype == sc_data_type_t::u16(in_dtype.lanes_)) {
        assert(cpu_flags_.fAVX2);
        XBYAK_GEN(vpacksswb, AVX_X_X_XM, op_out, op_in, op_in);
    } else if (elem_cast_simd(sc_data_etype::S8, sc_data_etype::U16)) {
        assert(cpu_flags_.fAVX512F);
        XBYAK_GEN(vpmovsdb, AVX_XM_X, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::BF16, sc_data_etype::F32)) {
        assert(cpu_flags_.fAVX512BF16);
        XBYAK_GEN(vcvtneps2bf16, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::F16, sc_data_etype::F32)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtps2phx, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::F16, sc_data_etype::INDEX)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtuqq2ph, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::F16, sc_data_etype::U32)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtudq2ph, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::F16, sc_data_etype::S32)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtdq2ph, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::F16, sc_data_etype::U16)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtuw2ph, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::F32, sc_data_etype::F16)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtph2psx, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::INDEX, sc_data_etype::F16)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtph2uqq, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::U32, sc_data_etype::F16)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtph2udq, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::S32, sc_data_etype::F16)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtph2dq, AVX_X_XM, op_out, op_in);
    } else if (elem_cast_simd(sc_data_etype::U16, sc_data_etype::F16)) {
        assert(cpu_flags_.fAVX512FP16);
        XBYAK_GEN(vcvtph2uw, AVX_X_XM, op_out, op_in);
    } else if ((out_dtype == sc_data_type_t::boolean(8)
                       || out_dtype == sc_data_type_t::boolean(4))
            && scalar_bit(in_dtype) >= 8) {
        assert(cpu_flags_.fAVX512DQ);
        XBYAK_GEN(kmovb, AVX_KMR32_KMR32, op_out, op_in);
    } else if (out_dtype == sc_data_type_t::boolean(16)
            && scalar_bit(in_dtype) >= 16) {
        assert(cpu_flags_.fAVX512F);
        XBYAK_GEN(kmovw, AVX_KMR32_KMR32, op_out, op_in);
    } else if (out_dtype == sc_data_type_t::boolean(32)
            && scalar_bit(in_dtype) >= 32) {
        assert(cpu_flags_.fAVX512BW);
        XBYAK_GEN(kmovd, AVX_KMR32_KMR32, op_out, op_in);
    } else if (out_dtype == sc_data_type_t::boolean(64)
            && scalar_bit(in_dtype) >= 64) {
        assert(cpu_flags_.fAVX512BW);
        XBYAK_GEN(kmovq, AVX_KMR64_KMR64, op_out, op_in);
    } else if (out_dtype == in_dtype) {
        handle_assign(lhs, v->in_);
    } else {
        COMPILE_ASSERT(false,
                FUNC_INFO << "Invalid type: " << out_dtype << " <- " << in_dtype
                          << ". v=" << v);
    }
}

void xbyak_lowering_viewer::handle_saturated_cast(
        const expr_c &dst, const expr_c &src) {
    const sc_data_type_t src_dtype = src->dtype_;
    const sc_data_type_t dst_dtype = dst->dtype_;

    auto op_src = GET_OPERAND(src);
    auto op_dst = GET_OPERAND(dst);

    if (dst_dtype == sc_data_type_t::s8(16)
            && src_dtype == sc_data_type_t::s32(16)) {
        assert(cpu_flags_.fAVX512F);
        XBYAK_GEN(vpmovsdb, AVX_XM_X, op_dst, op_src);
    } else if (dst_dtype == sc_data_type_t::u8(16)
            && src_dtype == sc_data_type_t::s32(16)) {
        assert(cpu_flags_.fAVX512F);
        XBYAK_GEN(vpmovusdb, AVX_XM_X, op_dst, op_src);
    } else {
        COMPILE_ASSERT(false,
                FUNC_INFO << "Invalid type: " << dst_dtype << " <- "
                          << src_dtype);
    }
}

void xbyak_lowering_viewer::handle_round_and_cast(
        const expr_c &dst, const expr_c &src) {
    const sc_data_type_t src_dtype = src->dtype_;
    const sc_data_type_t dst_dtype = dst->dtype_;

    auto elem_cast_simd = [&](sc_data_etype out_etype, sc_data_etype in_etype) {
        return is_lane_macth(dst_dtype, src_dtype, 4, 8, 16, 32, 64)
                && is_type_macth(dst_dtype, src_dtype, out_etype, in_etype);
    };

    auto op_src = GET_OPERAND(src);
    auto op_dst = GET_OPERAND(dst);

    if (elem_cast_simd(sc_data_etype::S32, sc_data_etype::F32)) {
        XBYAK_GEN(vcvtps2dq, AVX_X_XM, op_dst, op_src);
    } else if (dst_dtype == sc_data_type_t::s32(1)
            && src_dtype == sc_data_type_t::f32(1)) {
        XBYAK_GEN(vcvtss2si, AVX_R32_XM, op_dst, op_src);
    } else {
        COMPILE_ASSERT(false,
                FUNC_INFO << "Invalid type: " << dst_dtype << " <- "
                          << src_dtype);
    }
}

void xbyak_lowering_viewer::handle_reinterpret(
        const expr_c &lhs, const expr_c &rhs) {
    const sc_data_type_t dtype_dst = lhs->dtype_;
    const sc_data_type_t dtype_src = rhs->dtype_;

    auto op_dst = GET_OPERAND(lhs);
    auto op_src = GET_OPERAND(rhs);

    auto size_of_dst = utils::get_sizeof_type(dtype_dst);
    auto size_of_src = utils::get_sizeof_type(dtype_src);

    COMPILE_ASSERT(size_of_dst == size_of_src,
            "Reinterpret must match data size: " << dtype_dst << ", "
                                                 << dtype_src);

    switch (size_of_dst) {
        case 1: { // 8-bit
            handle_x86_mov(op_dst, op_src);
        } break;
        case 2: { // 16-bit
            handle_x86_mov(op_dst, op_src);
        } break;
        case 4: { // 32-bit
            if (dtype_dst != datatypes::f32 && op_src.is_addr()) {
                // int <- addr
                handle_x86_mov(op_dst, op_src);
            } else if (dtype_dst == datatypes::f32 && op_src.is_addr()) {
                // float <- addr
                handle_avx_movss(op_dst, op_src);
            } else if (dtype_dst == datatypes::f32
                    || dtype_src == datatypes::f32) {
                // float <-> int
                XBYAK_GEN(vmovd, AVX_XMR32_XMR32, op_dst, op_src);
            } else {
                // int <-> int
                handle_x86_mov(op_dst, op_src);
            }
        } break;
        case 8: { // 64-bit
            if (op_dst.is_reg() && op_src.is_addr()) {
                // reg64 <- addr
                handle_x86_mov(op_dst, op_src);
            } else if (!is_x86_simd(dtype_dst) && !is_x86_simd(dtype_src)) {
                // 64-bit int
                handle_x86_mov(op_dst, op_src);
            } else {
                handle_avx_movq(op_dst, op_src);
            }
        } break;
        case 16: { // 128-bit xmm
            handle_avx_movps(op_dst, op_src);
        } break;
        case 32: { // 256-bit ymm
            handle_avx_movps(op_dst, op_src);
        } break;
        case 64: { // 512-bit zmm
            handle_avx_movps(op_dst, op_src);
        } break;
        default:
            COMPILE_ASSERT(false,
                    FUNC_INFO << "Invalid type: " << dtype_dst << " <- "
                              << dtype_src);
    }
}

//==============================================================================
// X86 INTRINSIC HELPER MEMBER FUNCTION SECTION
//==============================================================================

void xbyak_lowering_viewer::handle_x86_mov(
        const operand &op_dst, const operand &op_src) {
    if (op_dst == op_src) { return; }
    XBYAK_GEN(mov, X86_RM_RMI, op_dst, op_src);
}

void xbyak_lowering_viewer::handle_x86_test(const operand &op_cond) {
    // Test bool type, set SF, ZF, PF flags
    // Functionally same as: TEST cond, cond
    gen_->cmp(op_cond.get_operand(), 0);
}

void xbyak_lowering_viewer::handle_x86_sign_ext(
        const operand &op_rdx, const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::sint_8: {
            // Signed div, sign-extend of AX
            gen_->cwd();
        } break;
        case cpu_data_type::sint_32: {
            // Signed div, sign-extend of EAX
            gen_->cdq();
        } break;
        case cpu_data_type::uint_8:
        case cpu_data_type::uint_16:
        case cpu_data_type::uint_32:
        case cpu_data_type::uint_64: {
            // Unsigned div, zero out rdx
            gen_->xor_(op_rdx.get_reg(), op_rdx.get_reg());
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_x86_div(
        const operand &op_div, const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::sint_8:
        case cpu_data_type::sint_32: {
            // Signed div
            XBYAK_GEN(idiv, X86_RM, op_div);
        } break;
        case cpu_data_type::uint_8:
        case cpu_data_type::uint_16:
        case cpu_data_type::uint_32:
        case cpu_data_type::uint_64: {
            // Unsigned div
            XBYAK_GEN(div, X86_RM, op_div);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_x86_cmp(
        const operand &op_lhs, const operand &op_rhs) {
    XBYAK_GEN(cmp, X86_RM_RMI, op_lhs, op_rhs);
}

void xbyak_lowering_viewer::handle_x86_set(const operand &op_dst,
        const xbyak_condition &code, const x86_64::cpu_data_type &cpu_dtype) {
    const auto &op = op_dst.get_operand();
    switch (cpu_dtype) {
        case cpu_data_type::uint_8:
        case cpu_data_type::uint_16:
        case cpu_data_type::uint_32:
        case cpu_data_type::uint_64: {
            // Unsigned condition set
            switch (code) {
                case xbyak_condition::eq: gen_->sete(op); break;
                case xbyak_condition::lt: gen_->setnae(op); break;
                case xbyak_condition::le: gen_->setna(op); break;
                case xbyak_condition::ne: gen_->setne(op); break;
                case xbyak_condition::ge: gen_->setnb(op); break;
                case xbyak_condition::gt: gen_->setnbe(op); break;
                default:
                    COMPILE_ASSERT(
                            false, FUNC_INFO << "Invalid condition: " << code);
            }
        } break;
        case cpu_data_type::sint_8:
        case cpu_data_type::sint_32: {
            // Signed condition set
            switch (code) {
                case xbyak_condition::eq: gen_->sete(op); break;
                case xbyak_condition::lt: gen_->setl(op); break;
                case xbyak_condition::le: gen_->setle(op); break;
                case xbyak_condition::ne: gen_->setne(op); break;
                case xbyak_condition::ge: gen_->setge(op); break;
                case xbyak_condition::gt: gen_->setg(op); break;
                default:
                    COMPILE_ASSERT(
                            false, FUNC_INFO << "Invalid condition: " << code);
            }
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_x86_cmov(const operand &op_dst,
        const operand &op_src, const xbyak_condition &code,
        const x86_64::cpu_data_type &cpu_dtype) {
    const auto get_cmov_op = [](const operand &op) {
        // cmov instructions do not support 8-bit reg as operand
        if (op.is_reg(8)) { return operand(op.get_reg32()); }
        return op;
    };
    const auto op_dst_c = get_cmov_op(op_dst);
    const auto op_src_c = get_cmov_op(op_src);
    switch (cpu_dtype) {
        case cpu_data_type::uint_8:
        case cpu_data_type::uint_16:
        case cpu_data_type::uint_32:
        case cpu_data_type::uint_64: {
            switch (code) {
                case xbyak_condition::eq: {
                    XBYAK_GEN(cmove, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::lt: {
                    XBYAK_GEN(cmovb, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::le: {
                    XBYAK_GEN(cmovbe, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::ne: {
                    XBYAK_GEN(cmovne, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::ge: {
                    XBYAK_GEN(cmovae, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::gt: {
                    XBYAK_GEN(cmova, X86_R_RM, op_dst_c, op_src_c);
                } break;
                default: {
                    COMPILE_ASSERT(false, FUNC_INFO << "Invalid condition.");
                } break;
            }
        } break;
        case cpu_data_type::sint_8:
        case cpu_data_type::sint_32: {
            switch (code) {
                case xbyak_condition::eq: {
                    XBYAK_GEN(cmove, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::lt: {
                    XBYAK_GEN(cmovl, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::le: {
                    XBYAK_GEN(cmovle, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::ne: {
                    XBYAK_GEN(cmovne, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::ge: {
                    XBYAK_GEN(cmovge, X86_R_RM, op_dst_c, op_src_c);
                } break;
                case xbyak_condition::gt: {
                    XBYAK_GEN(cmovg, X86_R_RM, op_dst_c, op_src_c);
                } break;
                default: {
                    COMPILE_ASSERT(false, FUNC_INFO << "Invalid condition.");
                } break;
            }
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

//==============================================================================
// AVX INTRINSIC HELPER MEMBER FUNCTION SECTION
//==============================================================================

void xbyak_lowering_viewer::handle_avx_movq(
        const operand &op_dst, const operand &op_src) {
    if (op_dst == op_src) { return; }
    XBYAK_GEN(vmovq, AVX_XMR64_XMR64, op_dst, op_src);
}

void xbyak_lowering_viewer::handle_avx_movss(
        const operand &op_dst, const operand &op_src) {
    if (op_dst == op_src) { return; }
    XBYAK_GEN(vmovss, AVX_XM_XM, op_dst, op_src);
}

void xbyak_lowering_viewer::handle_avx_movsh(
        const operand &op_dst, const operand &op_src) {
    if (op_dst == op_src) { return; }
    XBYAK_GEN(vmovw, AVX_XM_XM, op_dst, op_src);
}

void xbyak_lowering_viewer::handle_avx_movps(
        const operand &op_dst, const operand &op_src) {
    if (op_dst == op_src) {
        return;
    } else if (op_dst.is_xyz() && op_src.is_xyz()) {
        gen_->vmovaps(op_dst.get_xmm(), op_src.get_xmm());
    } else if (op_dst.is_xyz() && op_src.is_addr()) {
        gen_->vmovups(op_dst.get_xmm(), op_src.get_addr());
    } else if (op_dst.is_addr() && op_src.is_xyz()) {
        gen_->vmovups(op_dst.get_addr(), op_src.get_xmm());
    } else {
        COMPILE_ASSERT(false,
                FUNC_INFO << "Invalid operand: " << op_dst << ", " << op_src);
    }
}

void xbyak_lowering_viewer::handle_avx512_kmov(const operand &op_dst,
        const operand &op_src, const x86_64::cpu_data_type &cpu_dtype) {
    if (op_dst == op_src) { return; }
    switch (cpu_dtype) {
        case cpu_data_type::mask_x4: {
            XBYAK_GEN(kmovb, AVX_KMR32_KMR32, op_dst, op_src);
        } break;
        case cpu_data_type::mask_x8: {
            XBYAK_GEN(kmovb, AVX_KMR32_KMR32, op_dst, op_src);
        } break;
        case cpu_data_type::mask_x16: {
            XBYAK_GEN(kmovw, AVX_KMR32_KMR32, op_dst, op_src);
        } break;
        case cpu_data_type::mask_x32: {
            XBYAK_GEN(kmovd, AVX_KMR32_KMR32, op_dst, op_src);
        } break;
        case cpu_data_type::mask_x64: {
            XBYAK_GEN(kmovq, AVX_KMR64_KMR64, op_dst, op_src);
        } break;
        default: {
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    }
}

void xbyak_lowering_viewer::handle_avx_add(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_16_x4: {
            assert(cpu_flags_.fAVX512BF16);
            XBYAK_GEN(vaddph, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_16: {
            XBYAK_GEN(vaddsh, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::float_32_x8:
        case cpu_data_type::float_32_x4:
        case cpu_data_type::float_32_x2: {
            XBYAK_GEN(vaddps, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32: {
            XBYAK_GEN(vaddss, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::uint_32_x16:
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::uint_32_x4:
        case cpu_data_type::sint_32_x4:
        case cpu_data_type::uint_32_x2:
        case cpu_data_type::sint_32_x2: {
            XBYAK_GEN(vpaddd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::uint_8_x16:
        case cpu_data_type::sint_8_x16: {
            XBYAK_GEN(vpaddb, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_sub(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_16_x4: {
            XBYAK_GEN(vsubph, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_16: {
            XBYAK_GEN(vsubsh, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::float_32_x8:
        case cpu_data_type::float_32_x4:
        case cpu_data_type::float_32_x2: {
            XBYAK_GEN(vsubps, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32: {
            XBYAK_GEN(vsubss, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::uint_32_x8: {
            XBYAK_GEN(vpsubd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_mul(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_16_x4: {
            assert(cpu_flags_.fAVX512FP16);
            XBYAK_GEN(vmulph, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_16: {
            XBYAK_GEN(vmulsh, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::float_32_x8:
        case cpu_data_type::float_32_x4:
        case cpu_data_type::float_32_x2: {
            XBYAK_GEN(vmulps, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32: {
            XBYAK_GEN(vmulss, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::sint_32_x16: {
            XBYAK_GEN(vpmulld, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_mulhl(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::sint_32_x8: {
            XBYAK_GEN(vpmuldq, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_div(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_16: {
            assert(cpu_flags_.fAVX512FP16);
            XBYAK_GEN(vdivsh, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_16_x32: {
            assert(cpu_flags_.fAVX512FP16);
            XBYAK_GEN(vdivph, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::float_32_x8: {
            XBYAK_GEN(vdivps, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32: {
            XBYAK_GEN(vdivss, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_bit_or(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    auto gen_avx_bit_or = [&]() {
        switch (cpu_dtype) {
            case cpu_data_type::uint_16_x16:
            case cpu_data_type::uint_16_x8:
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8:
            case cpu_data_type::uint_8_x8:
            case cpu_data_type::uint_8_x16:
            case cpu_data_type::uint_8_x32:
            case cpu_data_type::sint_8_x32: {
                XBYAK_GEN(vpor, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    auto gen_avx512_bit_or = [&]() {
        assert(cpu_flags_.fAVX512F);
        switch (cpu_dtype) {
            case cpu_data_type::uint_32_x16:
            case cpu_data_type::sint_32_x16:
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8: {
                XBYAK_GEN(vpord, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Gen bit_or
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_bit_or(); break;
        case simd_level::avx2:
        case simd_level::avx: gen_avx_bit_or(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_bit_and(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    auto gen_avx_bit_and = [&]() {
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32_x4:
            case cpu_data_type::float_32: {
                XBYAK_GEN(vandps, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
            } break;
            case cpu_data_type::uint_32_x4:
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8:
            case cpu_data_type::uint_16_x8:
            case cpu_data_type::uint_16_x16: {
                XBYAK_GEN(vpand, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    auto gen_avx512_bit_and = [&]() {
        assert(cpu_flags_.fAVX512F);
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x16:
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32: {
                XBYAK_GEN(vandps, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
            } break;
            case cpu_data_type::uint_32_x16:
            case cpu_data_type::sint_32_x16:
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8: {
                XBYAK_GEN(vpandd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
            } break;
            case cpu_data_type::float_16_x16:
            case cpu_data_type::float_16_x8:
            case cpu_data_type::uint_16_x32:
            case cpu_data_type::uint_16_x16: {
                // Use Bitwise AND of packed quadword integers, do not use
                // writemask
                XBYAK_GEN(vpandq, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Gen bit_and
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_bit_and(); break;
        case simd_level::avx2:
        case simd_level::avx: gen_avx_bit_and(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_bit_xor(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    auto gen_avx_bit_xor = [&]() {
        switch (cpu_dtype) {
            case cpu_data_type::uint_16_x16:
            case cpu_data_type::uint_16_x8:
            case cpu_data_type::uint_8_x8:
            case cpu_data_type::uint_8_x16:
            case cpu_data_type::uint_8_x32:
            case cpu_data_type::sint_8_x8:
            case cpu_data_type::sint_8_x16:
            case cpu_data_type::sint_8_x32:
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8: {
                XBYAK_GEN(vpxor, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    auto gen_avx512_bit_xor = [&]() {
        assert(cpu_flags_.fAVX512F);
        switch (cpu_dtype) {
            case cpu_data_type::float_16_x32:
            case cpu_data_type::float_16_x16:
            case cpu_data_type::uint_32_x16:
            case cpu_data_type::sint_32_x16:
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8: {
                XBYAK_GEN(vpxord, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Gen bit_xor
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_bit_xor(); break;
        case simd_level::avx2:
        case simd_level::avx: gen_avx_bit_xor(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_min(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::sint_32_x8: {
            XBYAK_GEN(vpminsd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::float_32_x8:
        case cpu_data_type::float_32_x4:
        case cpu_data_type::float_32_x2: {
            XBYAK_GEN(vminps, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32: {
            XBYAK_GEN(vminss, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_16_x4: {
            XBYAK_GEN(vminph, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_16: {
            XBYAK_GEN(vminsh, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::uint_16_x16:
        case cpu_data_type::uint_16_x8: {
            XBYAK_GEN(vpminsw, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_max(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_16_x4: {
            assert(cpu_flags_.fAVX512FP16);
            XBYAK_GEN(vmaxph, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_16: {
            XBYAK_GEN(vmaxsh, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::float_32_x8:
        case cpu_data_type::float_32_x4:
        case cpu_data_type::float_32_x2: {
            XBYAK_GEN(vmaxps, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::float_32: {
            XBYAK_GEN(vmaxss, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::uint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::uint_32_x4: {
            XBYAK_GEN(vpmaxud, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::sint_32_x4: {
            XBYAK_GEN(vpmaxsd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::uint_8_x32:
        case cpu_data_type::uint_8_x16:
        case cpu_data_type::uint_8_x8: {
            XBYAK_GEN(vpmaxub, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::sint_8_x32:
        case cpu_data_type::sint_8_x16:
        case cpu_data_type::sint_8_x8: {
            XBYAK_GEN(vpmaxsb, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case cpu_data_type::uint_16_x16:
        case cpu_data_type::uint_16_x8: {
            XBYAK_GEN(vpmaxsw, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_abs(const operand &op_lhs,
        const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::sint_8_x64: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::sint_8_x32:
        case cpu_data_type::sint_8_x16: {
            XBYAK_GEN(vpabsb, AVX_X_XM, op_lhs, op_rhs);
        } break;
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::sint_32_x4:
        case cpu_data_type::sint_32_x2: {
            XBYAK_GEN(vpabsd, AVX_X_XM, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_shr(const operand &op_dst,
        const operand &op_lhs, const operand &op_sft,
        const x86_64::cpu_data_type &cpu_dtype, bool variable) {
    switch (cpu_dtype) {
        case cpu_data_type::uint_64_x8: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::uint_64_x4: {
            if (variable) {
                XBYAK_GEN(vpsrlvq, AVX_X_X_XM, op_dst, op_lhs, op_sft);
            } else {
                XBYAK_GEN(vpsrlq, AVX_X_XM_XMI, op_dst, op_lhs, op_sft);
            }
        } break;
        case cpu_data_type::sint_32_x16:
        case cpu_data_type::uint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::sint_32_x4:
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::uint_32_x4: {
            if (variable) {
                XBYAK_GEN(vpsrlvd, AVX_X_X_XM, op_dst, op_lhs, op_sft);
            } else {
                XBYAK_GEN(vpsrld, AVX_X_XM_XMI, op_dst, op_lhs, op_sft);
            }
        } break;
        case cpu_data_type::uint_16_x16:
        case cpu_data_type::uint_16_x8: {
            if (variable) {
                XBYAK_GEN(vpsrlvw, AVX_X_X_XM, op_dst, op_lhs, op_sft);
            } else {
                XBYAK_GEN(vpsrlw, AVX_X_XM_XMI, op_dst, op_lhs, op_sft);
            }
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_shl(const operand &op_dst,
        const operand &op_lhs, const operand &op_sft,
        const x86_64::cpu_data_type &cpu_dtype, bool variable) {
    switch (cpu_dtype) {
        case cpu_data_type::uint_64_x8: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::uint_64_x4: {
            if (variable) {
                XBYAK_GEN(vpsllvq, AVX_X_X_XM, op_dst, op_lhs, op_sft);
            } else {
                XBYAK_GEN(vpsllq, AVX_X_XM_XMI, op_dst, op_lhs, op_sft);
            }
        } break;
        case cpu_data_type::uint_32_x16:
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::uint_32_x4:
        case cpu_data_type::sint_32_x4: {
            if (variable) {
                XBYAK_GEN(vpsllvd, AVX_X_X_XM, op_dst, op_lhs, op_sft);
            } else {
                XBYAK_GEN(vpslld, AVX_X_XM_XMI, op_dst, op_lhs, op_sft);
            }
        } break;
        case cpu_data_type::uint_16_x16:
        case cpu_data_type::uint_16_x8: {
            if (variable) {
                XBYAK_GEN(vpsllvw, AVX_X_X_XM, op_dst, op_lhs, op_sft);
            } else {
                XBYAK_GEN(vpsllw, AVX_X_XM_XMI, op_dst, op_lhs, op_sft);
            }
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_sar(const operand &op_dst,
        const operand &op_lhs, const operand &op_sft,
        const x86_64::cpu_data_type &cpu_dtype, bool variable) {
    switch (cpu_dtype) {
        case cpu_data_type::uint_32_x16:
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::uint_32_x4:
        case cpu_data_type::sint_32_x4: {
            if (variable) {
                XBYAK_GEN(vpsravd, AVX_X_X_XM, op_dst, op_lhs, op_sft);
            } else {
                XBYAK_GEN(vpsrad, AVX_X_XM_XMI, op_dst, op_lhs, op_sft);
            }
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_round(const operand &op_lhs,
        const operand &op_rhs, const x86_64::cpu_data_type &cpu_dtype,
        const int64_t &imm) {
    // avx round
    auto gen_avx_round = [&]() {
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32_x4: {
                XBYAK_GEN(vroundps, AVX_X_XM_I, //
                        op_lhs, op_rhs, operand(imm));
            } break;
            case cpu_data_type::float_32: {
                XBYAK_GEN(vroundss, AVX_X_X_XM_I, //
                        op_lhs, op_rhs, op_rhs, operand(imm));
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // avx512 round
    auto gen_avx512_round = [&]() {
        assert(cpu_flags_.fAVX512F);
        switch (cpu_dtype) {
            case cpu_data_type::float_16_x32: {
                assert(cpu_flags_.fAVX512FP16);
            }
            case cpu_data_type::float_16_x16:
            case cpu_data_type::float_16_x8:
            case cpu_data_type::float_16_x4: {
                XBYAK_GEN(vrndscaleph, AVX_X_XM_I, //
                        op_lhs, op_rhs, operand(imm));
            } break;
            case cpu_data_type::float_16: {
                XBYAK_GEN(vrndscalesh, AVX_X_X_XM_I, //
                        op_lhs, op_rhs, op_rhs, operand(imm));
            } break;
            case cpu_data_type::float_32_x16:
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32_x4: {
                XBYAK_GEN(vrndscaleps, AVX_X_XM_I, //
                        op_lhs, op_rhs, operand(imm));
            } break;
            case cpu_data_type::float_32: {
                XBYAK_GEN(vrndscaless, AVX_X_X_XM_I, //
                        op_lhs, op_rhs, op_rhs, operand(imm));
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Gen round
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_round(); break;
        case simd_level::avx2:
        case simd_level::avx: gen_avx_round(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_sqrt(const operand &op_dst,
        const operand &op_src, const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::float_32_x8: {
            XBYAK_GEN(vsqrtps, AVX_X_XM, op_dst, op_src);
        } break;
        case cpu_data_type::float_32: {
            XBYAK_GEN(vsqrtss, AVX_X_X_XM, op_dst, op_src, op_src);
        } break;
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_16_x4: {
            assert(cpu_flags_.fAVX512FP16);
            XBYAK_GEN(vsqrtph, AVX_X_XM, op_dst, op_src);
        } break;
        case cpu_data_type::float_16: {
            XBYAK_GEN(vsqrtsh, AVX_X_X_XM, op_dst, op_src, op_src);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_rsqrt(const operand &op_dst,
        const operand &op_src, const x86_64::cpu_data_type &cpu_dtype) {
    // Use avx for low percision rsqrt
    auto gen_avx_rsqrt = [&]() {
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x8: {
                XBYAK_GEN(vrsqrtps, AVX_X_XM, op_dst, op_src);
            } break;
            case cpu_data_type::float_32: {
                XBYAK_GEN(vrsqrtss, AVX_X_X_XM, op_dst, op_src, op_src);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Use avx512 for higher percision rsqrt
    auto gen_avx512_rsqrt = [&]() {
        assert(cpu_flags_.fAVX512F);
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x16:
            case cpu_data_type::float_32_x8: {
                XBYAK_GEN(vrsqrt14ps, AVX_X_XM, op_dst, op_src);
            } break;
            case cpu_data_type::float_32: {
                XBYAK_GEN(vrsqrt14ss, AVX_X_X_XM, op_dst, op_src, op_src);
            } break;
            case cpu_data_type::float_16_x32:
            case cpu_data_type::float_16_x16:
            case cpu_data_type::float_16_x8:
            case cpu_data_type::float_16_x4: {
                XBYAK_GEN(vrsqrtph, AVX_X_XM, op_dst, op_src);
            } break;
            case cpu_data_type::float_16: {
                XBYAK_GEN(vrsqrtsh, AVX_X_X_XM, op_dst, op_src, op_src);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Gen rsqrt
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_rsqrt(); break;
        case simd_level::avx2:
        case simd_level::avx: gen_avx_rsqrt(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_fmadd(const operand &op_dst,
        const operand &op_mul, const operand &op_add,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_32_x16: {
            assert(cpu_flags_.fAVX512F);
        } // fall-through
        case cpu_data_type::float_32_x8:
        case cpu_data_type::float_32_x4: {
            if (op_mul.is_addr()) {
                XBYAK_GEN(vfmadd132ps, AVX_X_X_XM, op_dst, op_add, op_mul);
            } else {
                XBYAK_GEN(vfmadd213ps, AVX_X_X_XM, op_dst, op_mul, op_add);
            }
        } break;
        case cpu_data_type::float_32: {
            if (op_mul.is_addr()) {
                XBYAK_GEN(vfmadd132ss, AVX_X_X_XM, op_dst, op_add, op_mul);
            } else {
                XBYAK_GEN(vfmadd213ss, AVX_X_X_XM, op_dst, op_mul, op_add);
            }
        } break;
        case cpu_data_type::float_16_x32:
        case cpu_data_type::float_16_x16:
        case cpu_data_type::float_16_x8:
        case cpu_data_type::float_16_x4: {
            assert(cpu_flags_.fAVX512FP16);
            if (op_mul.is_addr()) {
                XBYAK_GEN(vfmadd132ph, AVX_X_X_XM, op_dst, op_add, op_mul);
            } else {
                XBYAK_GEN(vfmadd213ph, AVX_X_X_XM, op_dst, op_mul, op_add);
            }
        } break;
        case cpu_data_type::float_16: {
            if (op_mul.is_addr()) {
                XBYAK_GEN(vfmadd132sh, AVX_X_X_XM, op_dst, op_add, op_mul);
            } else {
                XBYAK_GEN(vfmadd213sh, AVX_X_X_XM, op_dst, op_mul, op_add);
            }
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_pshuffle(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::uint_8_x8:
        case cpu_data_type::uint_8_x16:
        case cpu_data_type::uint_8_x32: {
            XBYAK_GEN(vpshufb, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_shuffle(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs, const operand &op_imm,
        const operand &op_bits) {
    // Currently we assume that similar instructions do not use masks, and these
    // instructions only need to pay attention to how many bits are operated. So
    // we only need to use bits to choose instructions.
    auto type_bits = op_bits.get_imm();

    // Generate code for avx shuffle
    auto gen_avx_shuffle = [&]() {
        assert(cpu_flags_.fAVX);
        switch (type_bits) {
            case 32: {
                XBYAK_GEN(
                        vshufps, AVX_X_X_XM_I, op_dst, op_lhs, op_rhs, op_imm);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type_bits: " << type_bits);
        }
    };
    // Generate code for avx512 shuffle
    auto gen_avx512_shuffle = [&]() {
        assert(cpu_flags_.fAVX512F);
        switch (type_bits) {
            case 32: {
                XBYAK_GEN(
                        vshufps, AVX_X_X_XM_I, op_dst, op_lhs, op_rhs, op_imm);
            } break;
            case 128: {
                XBYAK_GEN(vshuff32x4, AVX_Y_Y_XM_I, op_dst, op_lhs, op_rhs,
                        op_imm);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type_bits: " << type_bits);
        }
    };
    // Generate shuffle
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_shuffle(); break;
        case simd_level::avx2:
        case simd_level::avx: gen_avx_shuffle(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_permute(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs, const operand &op_imm,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::uint_16_x16:
        case cpu_data_type::float_32_x8: {
            assert(op_imm.get_imm() == 32 || op_imm.get_imm() == 49);
            XBYAK_GEN(vperm2f128, AVX_Y_Y_XM_I, op_dst, op_lhs, op_rhs, op_imm);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_gather(const operand &op_dst,
        const operand &op_ptr, const operand &op_idx, const operand &mask,
        const x86_64::cpu_data_type &cpu_dtype) {
    auto op_addr = operand(
            gen_->ptr[op_ptr.get_reg() + op_idx.get_xmm() * sizeof(float)]);
    // Generate code for avx2 gather
    auto gen_avx2_gather = [&]() {
        assert(cpu_flags_.fAVX2);
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32_x4: {
                XBYAK_GEN(vgatherdps, AVX_X_M_X, //
                        op_dst, op_addr, mask);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Generate code for avx512 gather
    auto gen_avx512_gather = [&]() {
        assert(cpu_flags_.fAVX512VL);
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x16:
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32_x4: {
                XBYAK_GEN(vgatherdps, AVX_X_M, //
                        op_dst.set_evex(mask), op_addr);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Generate gather
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_gather(); break;
        case simd_level::avx2: gen_avx2_gather(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_insert(const operand &op_dst,
        const operand &op_b, const operand &op_imm,
        const operand &op_elem_bits) {
    // Currently we assume that similar instructions do not use masks, and these
    // instructions only need to pay attention to how many bits are operated. So
    // we only need to use bits to choose instructions.
    auto elem_bits = op_elem_bits.get_imm();
    switch (elem_bits) {
        case 8: {
            XBYAK_GEN(vpinsrb, AVX_X_X_RM_I, op_dst, op_dst, op_b, op_imm);
        } break;
        case 16: {
            XBYAK_GEN(vpinsrw, AVX_X_X_RM_I, op_dst, op_dst, op_b, op_imm);
        } break;
        case 32: {
            XBYAK_GEN(vpinsrd, AVX_X_X_RM_I, op_dst, op_dst, op_b, op_imm);
        } break;
        case 64: {
            XBYAK_GEN(vpinsrq, AVX_X_X_RM_I, op_dst, op_dst, op_b, op_imm);
        } break;
        case 128: {
            if (simd_level_ == simd_level::avx512) {
                assert(cpu_flags_.fAVX512VL);
                XBYAK_GEN(vinserti32x4, AVX_Y_Y_XM_I, op_dst, op_dst, op_b,
                        op_imm);
            } else {
                XBYAK_GEN(vinsertf128, AVX_Y_Y_YM_I, op_dst, op_dst, op_b,
                        op_imm);
            }
        } break;
        case 256: {
            assert(cpu_flags_.fAVX512DQ);
            XBYAK_GEN(vinserti32x8, AVX_Z_Z_XM_I, op_dst, op_dst, op_b, op_imm);
        } break;
        default:
            COMPILE_ASSERT(
                    false, FUNC_INFO << "Invalid elem_bits: " << elem_bits);
    };
}

void xbyak_lowering_viewer::handle_avx_extract(const operand &op_dst,
        const operand &op_b, const operand &op_imm,
        const operand &op_elem_bits) {
    // Currently we assume that similar instructions do not use masks, and these
    // instructions only need to pay attention to how many bits are operated. So
    // we only need to use bits to choose instructions.
    auto elem_bits = op_elem_bits.get_imm();
    switch (elem_bits) {
        case 8: {
            XBYAK_GEN(vpextrb, AVX_RM_X_I, op_dst, op_b, op_imm);
        } break;
        case 16: {
            XBYAK_GEN(vpextrw, AVX_RM_X_I, op_dst, op_b, op_imm);
        } break;
        case 32: {
            XBYAK_GEN(vpextrd, AVX_RM_X_I, op_dst, op_b, op_imm);
        } break;
        case 64: {
            XBYAK_GEN(vpextrq, AVX_RM_X_I, op_dst, op_b, op_imm);
        } break;
        case 128: {
            if (simd_level_ == simd_level::avx512) {
                assert(cpu_flags_.fAVX512VL);
                XBYAK_GEN(vextractf32x4, AVX_XM_Y_I, op_dst, op_b, op_imm);
            } else {
                XBYAK_GEN(vextractf128, AVX_XM_Y_I, op_dst, op_b, op_imm);
            }
        } break;
        case 256: {
            assert(cpu_flags_.fAVX512DQ);
            XBYAK_GEN(vextractf32x8, AVX_YM_Z_I, op_dst, op_b, op_imm);
        } break;
        default:
            COMPILE_ASSERT(
                    false, FUNC_INFO << "Invalid elem_bits: " << elem_bits);
    };
}

void xbyak_lowering_viewer::handle_avx_unpack_low(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs, const operand &op_imm) {
    auto elem_bits = op_imm.get_imm();
    // Generate code for avx unpacklow
    switch (elem_bits) {
        case 8: {
            XBYAK_GEN(vpunpcklbw, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case 16: {
            XBYAK_GEN(vpunpcklwd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case 32: {
            XBYAK_GEN(vpunpckldq, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case 64: {
            XBYAK_GEN(vpunpcklqdq, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(
                    false, FUNC_INFO << "Invalid elem_bits: " << elem_bits);
    }
}

void xbyak_lowering_viewer::handle_avx_unpack_high(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs, const operand &op_imm) {
    auto elem_bits = op_imm.get_imm();
    // Generate code for avx unpackhigh
    switch (elem_bits) {
        case 8: {
            XBYAK_GEN(vpunpckhbw, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case 16: {
            XBYAK_GEN(vpunpckhwd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case 32: {
            XBYAK_GEN(vpunpckhdq, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        case 64: {
            XBYAK_GEN(vpunpckhqdq, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
        } break;
        default:
            COMPILE_ASSERT(
                    false, FUNC_INFO << "Invalid elem_bits: " << elem_bits);
    };
}

void xbyak_lowering_viewer::handle_avx_extract_low(const operand &op_dst,
        const operand &op_src, const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_32_x16:
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512VL);
        } // fall-through
        case cpu_data_type::float_32_x8:
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::float_32_x4:
        case cpu_data_type::sint_32_x4:
        case cpu_data_type::float_32_x2: {
            handle_avx_movps(op_dst, op_src);
        } break;
        case cpu_data_type::uint_32_x2:
        case cpu_data_type::sint_32_x2: {
            XBYAK_GEN(vmovd, AVX_XMR32_XMR32, op_dst, op_src);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    };
}

void xbyak_lowering_viewer::handle_avx_extract_high(const operand &op_dst,
        const operand &op_src, const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_32_x16: {
            assert(cpu_flags_.fAVX512F);
            XBYAK_GEN(vextractf64x4, AVX_XM_Z_I, op_dst, //
                    op_src, operand(INT64_C(0x01)));
        } break;
        case cpu_data_type::sint_32_x16: {
            assert(cpu_flags_.fAVX512F);
            XBYAK_GEN(vextracti64x4, AVX_XM_Z_I, op_dst, //
                    op_src, operand(INT64_C(0x01)));
        } break;
        case cpu_data_type::float_32_x8: {
            switch (simd_level_) {
                case simd_level::avx512: {
                    assert(cpu_flags_.fAVX512VL);
                    XBYAK_GEN(vextractf32x4, AVX_XM_Y_I, op_dst, //
                            op_src, operand(INT64_C(0x01)));
                } break;
                case simd_level::avx2: {
                    XBYAK_GEN(vextractf128, AVX_XM_Y_I, op_dst, //
                            op_src, operand(INT64_C(0x01)));
                } break;
                default: COMPILE_ASSERT(false, FUNC_INFO << "No simd support");
            }
        } break;
        case cpu_data_type::sint_32_x8: {
            switch (simd_level_) {
                case simd_level::avx512: {
                    assert(cpu_flags_.fAVX512VL);
                    XBYAK_GEN(vextracti32x4, AVX_XM_Y_I, op_dst, //
                            op_src, operand(INT64_C(0x01)));
                } break;
                case simd_level::avx2: {
                    XBYAK_GEN(vextracti128, AVX_XM_Y_I, op_dst, //
                            op_src, operand(INT64_C(0x01)));
                } break;
                default: COMPILE_ASSERT(false, FUNC_INFO << "No simd support");
            }
        } break;
        case cpu_data_type::float_32_x4: {
            XBYAK_GEN(vpermilpd, AVX_X_X_XI, op_dst, //
                    op_src, operand(INT64_C(0x01)));
        } break;
        case cpu_data_type::sint_32_x4: {
            XBYAK_GEN(vpshufd, AVX_X_X_I, op_dst, //
                    op_src, operand(INT64_C(0x4e)));
        } break;
        case cpu_data_type::float_32_x2: {
            XBYAK_GEN(vmovshdup, AVX_X_XM, op_dst, //
                    op_src);
        } break;
        case cpu_data_type::sint_32_x2: {
            XBYAK_GEN(vpextrd, AVX_RM_X_I, op_dst, //
                    op_src, operand(INT64_C(0x01)));
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_permutex2var(const operand &op_dst,
        const operand &op_idx, const operand &op_src,
        const x86_64::cpu_data_type &cpu_dtype) {
    assert(cpu_flags_.fAVX512F);
    switch (cpu_dtype) {
        case cpu_data_type::float_32_x16:
        case cpu_data_type::float_32_x4: {
            XBYAK_GEN(vpermt2ps, AVX_X_X_XM, op_dst, op_idx, op_src);
        } break;
        case cpu_data_type::uint_8_x16: {
            XBYAK_GEN(vpermt2b, AVX_X_X_XM, op_dst, op_idx, op_src);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_permutexvar(const operand &op_dst,
        const operand &op_idx, const operand &op_src,
        const x86_64::cpu_data_type &cpu_dtype, const operand &bits) {
    switch (bits.get_imm()) {
        case 8: {
            assert(cpu_flags_.fAVX512VBMI);
            XBYAK_GEN(vpermb, AVX_X_X_XM, op_dst, op_idx, op_src);
        } break;
        case 16: {
            assert(cpu_flags_.fAVX512BW);
            XBYAK_GEN(vpermw, AVX_X_X_XM, op_dst, op_idx, op_src);
        } break;
        case 64: {
            if (op_idx.is_imm()) {
                XBYAK_GEN(vpermq, AVX_Y_YM_I, op_dst, op_src, op_idx);
            } else {
                assert(cpu_flags_.fAVX512VL);
                XBYAK_GEN(vpermq, AVX_Y_Y_YM, op_dst, op_idx, op_src);
            }
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_broadcast(const operand &op_dst,
        const operand &op_src, const x86_64::cpu_data_type &cpu_dtype,
        const x86_64::cpu_data_type &src_dtype) {
    switch (src_dtype) {
        case cpu_data_type::uint_16_x8: {
            assert(cpu_flags_.fAVX512F);
            if (op_src.is_addr()) {
                XBYAK_GEN(vbroadcasti32x4, AVX_Y_M, op_dst, op_src);
            } else {
                assert(op_src.is_xyz());
                switch (cpu_dtype) {
                    case cpu_data_type::uint_16_x32: {
                        auto op_src_z = operand(to_zmm(op_src.get_reg()));
                        XBYAK_GEN(vshufi32x4, AVX_Y_Y_XM_I, op_dst, op_src_z,
                                op_src_z, operand(INT64_C(0x0)));
                    } break;
                    case cpu_data_type::uint_16_x16: {
                        auto op_src_y = operand(to_ymm(op_src.get_reg()));
                        XBYAK_GEN(vshufi32x4, AVX_Y_Y_XM_I, op_dst, op_src_y,
                                op_src_y, operand(INT64_C(0x0)));
                    } break;
                    default:
                        COMPILE_ASSERT(
                                false, "Invalid broadcast: " << cpu_dtype);
                }
            }
        } break;
        case cpu_data_type::float_32: {
            XBYAK_GEN(vbroadcastss, AVX_X_XM, op_dst, op_src);
        } break;
        case cpu_data_type::uint_32:
        case cpu_data_type::sint_32: {
            XBYAK_GEN(vpbroadcastd, AVX_X_XM, op_dst, op_src);
        } break;
        case cpu_data_type::float_16:
        case cpu_data_type::uint_16: {
            XBYAK_GEN(vpbroadcastw, AVX_X_XM, op_dst, op_src);
        } break;
        case cpu_data_type::uint_8:
        case cpu_data_type::sint_8: {
            XBYAK_GEN(vpbroadcastb, AVX_X_XM, op_dst, op_src);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << src_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_blend(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs, const operand &op_cond,
        const x86_64::cpu_data_type &cpu_dtype) {
    // Generate code for avx blend
    auto gen_avx_blend = [&]() {
        // Get avx mask of correct size
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32_x4: {
                XBYAK_GEN(vblendvps, AVX_X_X_XM_X, op_dst, op_lhs, op_rhs,
                        op_cond);
            } break;
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8: {
                XBYAK_GEN(vblendvps, AVX_X_X_XM_X, op_dst, op_lhs, op_rhs,
                        op_cond);
            } break;
            case cpu_data_type::uint_8_x32:
            case cpu_data_type::sint_8_x32:
            case cpu_data_type::uint_8_x16:
            case cpu_data_type::sint_8_x16:
            case cpu_data_type::uint_8_x8:
            case cpu_data_type::sint_8_x8: {
                XBYAK_GEN(vpblendvb, AVX_X_X_XM_X, op_dst, op_lhs, op_rhs,
                        op_cond);
            } break;
            case cpu_data_type::uint_16_x16:
            case cpu_data_type::uint_16_x8: {
                // Because each 16-bit element of our __m256i mask will be
                // either 0xFFFF (all ones) or 0x0 (all zeros), this mask can be
                // used as a mask for blend byte operand to select data.
                XBYAK_GEN(vpblendvb, AVX_X_X_XM_X, op_dst, op_lhs, op_rhs,
                        op_cond);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Generate code for avx512 blend using opmask reg
    auto gen_avx512_blend = [&]() {
        COMPILE_ASSERT(op_cond.is_mask(), "op_cond must be Opmask.");
        assert(cpu_flags_.fAVX512F);
        switch (cpu_dtype) {
            case cpu_data_type::float_16_x32:
            case cpu_data_type::float_16_x16:
            case cpu_data_type::float_16_x8: {
                XBYAK_GEN(vpblendmw, AVX_X_X_XM, op_dst.set_evex(op_cond),
                        op_lhs, op_rhs);
            } break;
            case cpu_data_type::float_32_x16:
            case cpu_data_type::float_32_x8: {
                XBYAK_GEN(vblendmps, AVX_X_X_XM, op_dst.set_evex(op_cond),
                        op_lhs, op_rhs);
            } break;
            case cpu_data_type::uint_32_x16:
            case cpu_data_type::sint_32_x16:
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8: {
                XBYAK_GEN(vpblendmd, AVX_X_X_XM, op_dst.set_evex(op_cond),
                        op_lhs, op_rhs);
            } break;
            case cpu_data_type::uint_16_x32:
            case cpu_data_type::uint_16_x16:
            case cpu_data_type::uint_16_x8: {
                XBYAK_GEN(vpblendmw, AVX_X_X_XM, op_dst.set_evex(op_cond),
                        op_lhs, op_rhs);
            } break;
            case cpu_data_type::uint_8_x64:
            case cpu_data_type::sint_8_x64:
            case cpu_data_type::uint_8_x32:
            case cpu_data_type::sint_8_x32:
            case cpu_data_type::uint_8_x16:
            case cpu_data_type::sint_8_x16:
            case cpu_data_type::uint_8_x8:
            case cpu_data_type::sint_8_x8: {
                XBYAK_GEN(vpblendmb, AVX_X_X_XM, op_dst.set_evex(op_cond),
                        op_lhs, op_rhs);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // gen blend
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_blend(); break;
        case simd_level::avx2:
        case simd_level::avx: gen_avx_blend(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_mask_mov(const operand &op_dst,
        const operand &op_src, const operand &op_cond,
        const x86_64::cpu_data_type &cpu_dtype, bool zero) {
    // Generate code for avx2 mask_mov
    auto gen_avx2_mask_mov = [&]() {
        // Get avx mask of correct size
        COMPILE_ASSERT(!(op_dst.is_addr() && zero), "cannot zero mask store.");
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32_x4: {
                XBYAK_GEN(vmaskmovps, AVX_XM_X_XM, //
                        op_dst, op_cond, op_src);
            } break;
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8:
            case cpu_data_type::uint_32_x4:
            case cpu_data_type::sint_32_x4: {
                XBYAK_GEN(vpmaskmovd, AVX_XM_X_XM, //
                        op_dst, op_cond, op_src);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Generate code for avx512 mask_mov using opmask reg
    auto gen_avx512_mask_mov = [&]() {
        COMPILE_ASSERT(op_cond.is_mask(), "op_cond must be Opmask.");
        COMPILE_ASSERT(!(op_dst.is_addr() && zero), "cannot zero mask store.");
        assert(cpu_flags_.fAVX512F);
        switch (cpu_dtype) {
            // may have other datatypes needs to support
            case cpu_data_type::uint_8_x64:
            case cpu_data_type::sint_8_x64:
            case cpu_data_type::uint_8_x32:
            case cpu_data_type::sint_8_x32:
            case cpu_data_type::uint_8_x16:
            case cpu_data_type::sint_8_x16:
            case cpu_data_type::uint_8_x8:
            case cpu_data_type::sint_8_x8: {
                XBYAK_GEN(vmovdqu8, AVX_XM_XM, //
                        op_dst.set_evex(op_cond, zero), op_src);
            } break;
            case cpu_data_type::float_16_x32:
            case cpu_data_type::float_16_x16:
            case cpu_data_type::float_16_x8:
            case cpu_data_type::float_16_x4:
            case cpu_data_type::float_16:
            case cpu_data_type::uint_16_x32:
            case cpu_data_type::uint_16_x16:
            case cpu_data_type::uint_16_x8: {
                XBYAK_GEN(vmovdqu16, AVX_XM_XM, //
                        op_dst.set_evex(op_cond, zero), op_src);
            } break;
            case cpu_data_type::uint_32_x16:
            case cpu_data_type::sint_32_x16:
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::sint_32_x8:
            case cpu_data_type::uint_32_x4:
            case cpu_data_type::sint_32_x4:
            case cpu_data_type::sint_32: {
                XBYAK_GEN(vmovdqu32, AVX_XM_XM, //
                        op_dst.set_evex(op_cond, zero), op_src);
            } break;
            case cpu_data_type::float_32_x16:
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32_x4:
            case cpu_data_type::float_32: {
                XBYAK_GEN(vmovups, AVX_XM_XM, //
                        op_dst.set_evex(op_cond, zero), op_src);
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // gen code
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_mask_mov(); break;
        case simd_level::avx2: gen_avx2_mask_mov(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_cmov(const operand &op_dst,
        const operand &op_src, const xbyak_condition &code,
        const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_32: {
            Xbyak::Label l_end_cmov;
            switch (code) {
                case xbyak_condition::eq: {
                    // if (ZF=1) jmp over, else do mov
                    gen_->jne(l_end_cmov, Xbyak::CodeGenerator::T_NEAR);
                } break;
                case xbyak_condition::ne: {
                    // if (ZF=0) jmp over, else do mov
                    gen_->je(l_end_cmov, Xbyak::CodeGenerator::T_NEAR);
                } break;
                default: COMPILE_ASSERT(false, "Invalid condition: " << code);
            }
            handle_avx_movss(op_dst, op_src);
            // end
            gen_->L(l_end_cmov);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

void xbyak_lowering_viewer::handle_avx_cmp_set(const operand &op_dst,
        const operand &op_lhs, const operand &op_rhs,
        const xbyak_condition &code, const x86_64::cpu_data_type &cpu_dtype) {
    // cmp type code
    auto op_imm = [](const xbyak_condition &code) {
        switch (code) {
            case xbyak_condition::eq: return operand(INT64_C(0x0));
            case xbyak_condition::lt: return operand(INT64_C(0x1));
            case xbyak_condition::le: return operand(INT64_C(0x2));
            case xbyak_condition::ne: return operand(INT64_C(0x4));
            case xbyak_condition::ge: return operand(INT64_C(0x5));
            case xbyak_condition::gt: return operand(INT64_C(0x6));
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid condition: " << code);
        }
        return operand();
    };
    // Generate code for avx cmp_set
    auto gen_avx_cmp_set = [&]() {
        // Get avx mask of correct size
        switch (cpu_dtype) {
            case cpu_data_type::float_32_x8:
            case cpu_data_type::float_32_x4: {
                XBYAK_GEN(vcmpps, AVX_X_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            case cpu_data_type::float_32: {
                XBYAK_GEN(vcmpss, AVX_X_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            case cpu_data_type::uint_32_x8:
            case cpu_data_type::uint_32_x4: {
                switch (code) {
                    case xbyak_condition::eq: {
                        XBYAK_GEN(vpcmpeqd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
                    } break;
                    default: COMPILE_ASSERT(false, "No avx_cmp for: " << code);
                }
            } break;
            case cpu_data_type::sint_32_x8:
            case cpu_data_type::sint_32_x4: {
                switch (code) {
                    case xbyak_condition::eq: {
                        XBYAK_GEN(vpcmpeqd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
                    } break;
                    case xbyak_condition::gt: {
                        XBYAK_GEN(vpcmpgtd, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
                    } break;
                    default: COMPILE_ASSERT(false, "No avx_cmp for: " << code);
                }
            } break;
            case cpu_data_type::uint_16_x16:
            case cpu_data_type::uint_16_x8: {
                switch (code) {
                    case xbyak_condition::eq: {
                        XBYAK_GEN(vpcmpeqw, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
                    } break;
                    default: COMPILE_ASSERT(false, "No avx_cmp for: " << code);
                }
            } break;
            case cpu_data_type::uint_8_x32:
            case cpu_data_type::uint_8_x16:
            case cpu_data_type::uint_8_x8: {
                switch (code) {
                    case xbyak_condition::eq: {
                        XBYAK_GEN(vpcmpeqb, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
                    } break;
                    default: COMPILE_ASSERT(false, "No avx_cmp for: " << code);
                }
            } break;
            case cpu_data_type::sint_8_x32:
            case cpu_data_type::sint_8_x16:
            case cpu_data_type::sint_8_x8: {
                switch (code) {
                    case xbyak_condition::eq: {
                        XBYAK_GEN(vpcmpeqb, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
                    } break;
                    case xbyak_condition::gt: {
                        XBYAK_GEN(vpcmpgtb, AVX_X_X_XM, op_dst, op_lhs, op_rhs);
                    } break;
                    default: COMPILE_ASSERT(false, "No avx_cmp for: " << code);
                }
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // Generate code for avx512 cmp_set
    auto gen_avx512_cmp_set = [&]() {
        switch (cpu_dtype) {
            case cpu_data_type::float_16_x32:
            case cpu_data_type::float_16_x16:
            case cpu_data_type::float_16_x8:
            case cpu_data_type::float_16_x4: {
                assert(cpu_flags_.fAVX512FP16);
                XBYAK_GEN(vcmpph, AVX_K_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            case cpu_data_type::float_16: {
                XBYAK_GEN(vcomish, AVX_X_XM, op_lhs, op_rhs);
                // boolean cpu_dtype is uint_8, we just use it directly.
                handle_x86_set(op_dst, code, cpu_data_type::uint_8);
            } break;
            case cpu_data_type::float_32_x16:
            case cpu_data_type::float_32_x8: {
                XBYAK_GEN(vcmpps, AVX_K_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            case cpu_data_type::uint_32_x16:
            case cpu_data_type::uint_32_x8: {
                XBYAK_GEN(vpcmpud, AVX_K_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            case cpu_data_type::sint_32_x16:
            case cpu_data_type::sint_32_x8: {
                XBYAK_GEN(vpcmpd, AVX_K_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            case cpu_data_type::uint_16_x32:
            case cpu_data_type::uint_16_x16:
            case cpu_data_type::uint_16_x8: {
                XBYAK_GEN(vpcmpuw, AVX_K_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            case cpu_data_type::uint_8_x64:
            case cpu_data_type::uint_8_x32:
            case cpu_data_type::uint_8_x16:
            case cpu_data_type::uint_8_x8: {
                XBYAK_GEN(vpcmpub, AVX_K_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            case cpu_data_type::sint_8_x64:
            case cpu_data_type::sint_8_x32:
            case cpu_data_type::sint_8_x16:
            case cpu_data_type::sint_8_x8: {
                XBYAK_GEN(vpcmpb, AVX_K_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            case cpu_data_type::float_32: {
                XBYAK_GEN(vcmpss, AVX_X_X_XM_I, op_dst, op_lhs, op_rhs, //
                        op_imm(code));
            } break;
            default:
                COMPILE_ASSERT(
                        false, FUNC_INFO << "Invalid type: " << cpu_dtype);
        }
    };
    // gen code
    switch (simd_level_) {
        case simd_level::avx512: gen_avx512_cmp_set(); break;
        case simd_level::avx2:
        case simd_level::avx: gen_avx_cmp_set(); break;
        default: assert(false && "Unreachable");
    }
}

void xbyak_lowering_viewer::handle_avx_mov_mask(const operand &op_dst,
        const operand &op_src, const x86_64::cpu_data_type &cpu_dtype) {
    switch (cpu_dtype) {
        case cpu_data_type::float_32_x8:
        case cpu_data_type::float_32_x4:
        case cpu_data_type::uint_32_x8:
        case cpu_data_type::uint_32_x4:
        case cpu_data_type::sint_32_x8:
        case cpu_data_type::sint_32_x4: {
            XBYAK_GEN(vmovmskps, AVX_R64_X, op_dst, op_src);
        } break;
        case cpu_data_type::uint_8_x8:
        case cpu_data_type::sint_8_x8:
        case cpu_data_type::uint_8_x32:
        case cpu_data_type::uint_8_x16:
        case cpu_data_type::sint_8_x32:
        case cpu_data_type::sint_8_x16: {
            XBYAK_GEN(vpmovmskb, AVX_R64_X, op_dst, op_src);
        } break;
        case cpu_data_type::uint_16_x8:
        case cpu_data_type::uint_16_x16: {
            XBYAK_GEN(vpmovmskb, AVX_R64_X, op_dst, op_src);
        } break;
        default:
            COMPILE_ASSERT(false, FUNC_INFO << "Invalid type: " << cpu_dtype);
    }
}

//==============================================================================
// HANDLE CALL MEMBER FUNCTION SECTION
//==============================================================================

void xbyak_lowering_viewer::handle_pre_call(const stmts_c &v) {
    // Get attrs
    assert(v->attr_);
    assert(v->attr_->has_key(attr_keys::abi_interface));

    auto callee_iface = v->attr_->get<abi_function_interface::ptr>(
            attr_keys::abi_interface);

    // STEP 1: Save stack info before call prepare
    location_manager_->conserve_stack_size();
    // STEP 2: Align stack before call
    location_manager_->align_call_stack(*callee_iface);
}

void xbyak_lowering_viewer::handle_call(const expr_c &lhs, const call_c &v) {
    COMPILE_ASSERT(v->para_attr_.empty(), "Xbyak JIT not support.");

    func_t callee = v->get_prototype();
    auto expr_func = std::dynamic_pointer_cast<expr_base>(v->func_);

    // STEP 1: Shadow space padding
    if (profile_.call_convention_ == call_convention::microsoft) {
        // Microsoft x64 calling convention: caller allocate 32 bytes of
        // "shadow space" on the stack right before calling the function
        ASM_COMMENT("caller: allocate shadow space");
        location_manager_->stack_padding(profile_.shadow_space_bytes_);
    }

    // STEP 2: Call function address
    if (expr_func) {
        // Dynamic function call, func is a expr
        auto op_func = GET_OPERAND(expr(expr_func));
        if (op_func.is_reg()) {
            gen_->call(op_func.get_reg64());
        } else {
            gen_->mov(regs::rax, op_func.get_operand());
            gen_->call(regs::rax);
        }
    } else {
        // Static function call, func is a func_t
        const auto func_name = callee->name_;
        ASM_COMMENT("call: <" + func_name + ">");
        handle_func_resolve(
                func_name,
                [&](const Xbyak::Label &label) {
                    // call label
                    gen_->call(label);
                },
                [&](const uint64_t &addr) {
                    // call addr
                    gen_->vzeroupper();
                    gen_->mov(regs::rax, addr);
                    gen_->call(regs::rax);
                });
    }

    // STEP 3: Post-call cleanup
    // The System V ABI requires the caller to clean up the callstack after
    // the callee returns.
    ASM_COMMENT("caller: post-call cleanup");
    location_manager_->restore_stack_size();

    // STEP 4: Handle return vaule
    const sc_data_type_t ret_val_dtype = callee->ret_type_;

    if (lhs.defined() && ret_val_dtype != datatypes::void_t) {
        // Get the ABI requirements for the call
        auto callee_iface = cached_func_abi_interface(callee);
        const abi_value_location ret_val_loc = callee_iface->return_val_loc_;

        // We don't yet support function calls that return their value via
        // memory. (If/when we do, the 'STACK' tag won't really be accurate;
        // at least not for psABI-compliant calls.)
        const abi_value_location::tag_type tag = ret_val_loc.get_type();
        assert(tag == abi_value_location::tag_type::REGISTER);

        const Xbyak::Reg reg_ret = ret_val_loc.get_register();
        auto op_ret = GET_OPERAND(lhs);

        if (reg_ret.isXMM()) {
            handle_avx_movss(op_ret, operand(to_xmm(reg_ret)));
        } else {
            handle_x86_mov(op_ret, operand(reg_ret));
        }
    }
}

//==============================================================================
// VIEW STMT SECTION
//==============================================================================

void xbyak_lowering_viewer::view(stmts_c v) {
    if (TRANSFORMED_CALL(v)) {
        auto conserved = location_manager_->get_conserved_stack_size();
        ASM_COMMENT("call-scope");
        handle_pre_call(v);
        ir_viewer_t::view(v);
        COMPILE_ASSERT(
                conserved == location_manager_->get_conserved_stack_size(),
                "Stack frame has been corrupted after call-scope.")
    } else {
        ir_viewer_t::view(v);
    }
}

void xbyak_lowering_viewer::view(evaluate_c v) {
    if (v->value_.isa<call>() || v->value_.isa<low_level_intrin>()) {
        handle_operations(expr_c(), v->value_);
    }
}

void xbyak_lowering_viewer::view(assign_c v) {
    handle_operations(v->var_, v->value_);
}

void xbyak_lowering_viewer::view(define_c v) {
    assert(v->var_.defined());
    COMPILE_ASSERT(v->var_.isa<tensor_c>() || v->var_.isa<var_c>(),
            "Not supported local define: " << v);
    handle_local_definition(v->var_, v->init_);
}

void xbyak_lowering_viewer::view(returns_c v) {
    if (v->value_.defined()) {
        auto cpu_dtype = get_cpu_data_type(v->value_->dtype_);
        auto slot_size = get_local_value_stack_slot_size(cpu_dtype);
        assert(slot_size == 8);

        const Xbyak::Reg reg_ret = func_iface_->return_val_loc_.get_register();
        auto op_val = GET_OPERAND(v->value_);

        switch (cpu_dtype) {
            case cpu_data_type::uint_8:
            case cpu_data_type::sint_8:
            case cpu_data_type::uint_16:
            case cpu_data_type::uint_32:
            case cpu_data_type::sint_32:
            case cpu_data_type::uint_64: {
                handle_x86_mov(operand(reg_ret), op_val);
            } break;
            case cpu_data_type::float_16: {
                handle_avx_movsh(operand(to_xmm(reg_ret)), op_val);
            } break;
            case cpu_data_type::float_32: {
                handle_avx_movss(operand(to_xmm(reg_ret)), op_val);
            } break;
            default: COMPILE_ASSERT(false, "Unsupported type: " << cpu_dtype);
        }
    }

    gen_->jmp(l_func_epilogue_);
}

void xbyak_lowering_viewer::view(if_else_c v) {
    COMPILE_ASSERT(v->condition_->dtype_ == datatypes::boolean,
            "Invalid predicate dtype_: " << v->condition_->dtype_);
    // Prepare Label for end of if
    Xbyak::Label l_after_if_statement;

    auto cond = GET_OPERAND(v->condition_);

    // if (condition==false)
    ASM_COMMENT("if condition: " << v->condition_);
    handle_x86_test(cond);
    // Emit if bodys
    if (v->else_case_.defined()) {
        Xbyak::Label l_else_block;
        // If (condition==false) jump to the 'else' block.
        gen_->jz(l_else_block, Xbyak::CodeGenerator::T_NEAR);
        // Otherwise, fall through to the 'then' block...
        ASM_COMMENT("if then case: " << v->condition_);
        dispatch(v->then_case_);
        gen_->jmp(l_after_if_statement, Xbyak::CodeGenerator::T_NEAR);
        // Emit the 'else' block...
        ASM_COMMENT("if else case: " << v->condition_);
        gen_->L(l_else_block);
        dispatch(v->else_case_);
    } else {
        // If (condition==false) jump to the end.
        gen_->jz(l_after_if_statement, Xbyak::CodeGenerator::T_NEAR);
        // Emit the 'then' block...
        ASM_COMMENT("if then case: " << v->condition_);
        dispatch(v->then_case_);
    }

    // Define Label for end of if
    gen_->L(l_after_if_statement);
    ASM_COMMENT("if end: " << v->condition_);
}

void xbyak_lowering_viewer::view(for_loop_c v) {
    var_c loop_var = v->var_.checked_as<var>();

    COMPILE_ASSERT(loop_var->dtype_ == v->iter_begin_->dtype_,
            "Mismatched loop expression types");
    COMPILE_ASSERT(loop_var->dtype_ == v->iter_end_->dtype_,
            "Mismatched loop expression types");
    COMPILE_ASSERT(loop_var->dtype_ == v->step_->dtype_,
            "Mismatched loop expression types");

    const cpu_data_type loop_var_cpu_dtype
            = get_cpu_data_type(loop_var->dtype_);

    if (v->kind_ == for_type::PARALLEL) {
        COMPILE_ASSERT(false, "parallel for-loops not handled yet");
    } else {
        // ALLOCATE AND INITIALIZE LOOP VAR
        ASM_COMMENT("for begin: " << loop_var << " = " << v->iter_begin_);
        if (v->attr_ && v->attr_->has_key(attr_keys::load_loop_begin)) {
            auto load_begin = v->attr_->get<stmt>(attr_keys::load_loop_begin);
            dispatch(load_begin);
        }
        handle_local_definition(loop_var, v->iter_begin_);

        // LOOP LABELS
        Xbyak::Label l_loop_test;
        Xbyak::Label l_after_loop;

        // LOOP TEST
        // not required, but considered good practice for performance
        gen_->align(16);
        gen_->L(l_loop_test);

        // test the loop repeat condition...
        ASM_COMMENT("for condition: " << loop_var << " < " << v->iter_end_);
        if (v->attr_ && v->attr_->has_key(attr_keys::load_loop_end)) {
            auto load_end = v->attr_->get<stmt>(attr_keys::load_loop_end);
            dispatch(load_end);
        }
        auto var_op = GET_OPERAND(loop_var);
        auto end_op = GET_OPERAND(v->iter_end_);
        handle_x86_cmp(var_op, end_op);
        gen_->jge(l_after_loop, Xbyak::CodeGenerator::T_NEAR);

        // LOOP BODY
        dispatch(v->body_);

        // LOOP INCREMENT
        ASM_COMMENT("for step: " << loop_var << " += " << v->step_);
        if (v->attr_ && v->attr_->has_key(attr_keys::load_loop_step)) {
            auto load_step = v->attr_->get<stmt>(attr_keys::load_loop_step);
            dispatch(load_step);
        }
        handle_x86_intrisic(loop_var, {v->step_}, xbyak_intrin_type::add);

        // JMP TO LOOP TEST
        gen_->jmp(l_loop_test, Xbyak::CodeGenerator::T_NEAR);

        // AFTER LOOP
        gen_->L(l_after_loop);
        ASM_COMMENT("for end: " << loop_var);
    }
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
