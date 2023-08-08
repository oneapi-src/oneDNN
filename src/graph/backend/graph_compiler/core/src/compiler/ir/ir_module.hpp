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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_IR_MODULE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_IR_MODULE_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <compiler/config/context.hpp>
#include <compiler/ir/sc_expr.hpp>
#include <compiler/ir/sc_function.hpp>
#include <unordered_map>
#include <util/any_map.hpp>
struct brg_range_handle_t;
using brg_range_handle_ptr = std::shared_ptr<brg_range_handle_t>;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct op_dispatch_tables_t;
using op_dispatch_tables_ptr = std::shared_ptr<op_dispatch_tables_t>;
using dispatch_table_map_t
        = std::unordered_map<std::string, op_dispatch_tables_ptr>;

class function_pass_t;

class SC_INTERNAL_API ir_module_t {
    // Items to appear in the module.
    // Might be useful at some point to be able to include data,
    // initializations, finalizations, etc.
    std::vector<func_t> contents_;
    // the function name -> func_idx mapping
    std::unordered_map<std::string, int> symbols_;
    // the "main" function of the module, <0 if no entry func_t is set
    int entry_func_idx_;
    // the global variables
    std::vector<define> module_vars_;
    std::unordered_map<std::string, define> var_symbols_;
    // hold all op tables in ir module and pass to jit module for dynamic
    // dispatch. the op function name -> op tables ptr
    dispatch_table_map_t op_table_map_;
    // vector of brgemm range handler.
    std::vector<brg_range_handle_ptr> brg_handles_;

public:
    // the attr keys for ir_module
    struct attr_key_t {
        // the statics_table_t for the global data buffer
        static constexpr const char *MODULE_DATA_BUFFERS
                = "MODULE_DATA_BUFFERS";
        // float, the estimated number of FP operations in the module (GFLOPS)
        static constexpr const char *GFLOP = "gflop";
        // bool, whether to use managed thread pool
        static constexpr const char *MANAGED_THREAD_POOL
                = "MANAGED_THREAD_POOL";
        // string, the name of the module
        static constexpr const char *NAME = "name";
        // bool, default=false. whether the addresses of global tensors and
        // variables will be hardcoded in the JIT'd code
        static constexpr const char *STATIC_GLOBALS = "static_globals";
        // vector<shared_ptr<shared_const_wrapper>>. The list of base tensors
        // for shared consts
        static constexpr const char *SHARED_CONST_BASES = "shared_const_bases";
    };

    context_ptr ctx_;
    any_map_t attr_;
    ir_module_t(context_ptr ctx) : ctx_(std::move(ctx)) {
        entry_func_idx_ = -1;
    }

    // creates an ir_module_t with given functions. If entry_func_idx >=0, will
    // set entry_func_ = contents[entry_func_idx]. Otherwise, will leave
    // entry_func_ as empty
    ir_module_t(context_ptr ctx, const std::vector<func_t> &contents,
            int entry_func_idx = -1)
        : contents_(contents), ctx_(std::move(ctx)) {
        entry_func_idx_ = entry_func_idx;
    }

    int get_entry_func_idx() const { return entry_func_idx_; }
    void set_entry_func_idx(int entry_func_idx) {
        COMPILE_ASSERT(entry_func_idx == -1
                        || (entry_func_idx >= 0
                                && size_t(entry_func_idx) < contents_.size()),
                "Invalid entry_func_idx");
        entry_func_idx_ = entry_func_idx;
    }

    std::vector<func_t> &get_contents() { return contents_; }
    const std::vector<func_t> &get_contents() const { return contents_; }

    const std::vector<define> &get_module_vars() const { return module_vars_; }
    std::vector<define> &get_module_vars() { return module_vars_; }

    const dispatch_table_map_t &get_op_table_map() const {
        return op_table_map_;
    }
    dispatch_table_map_t &get_op_table_map() { return op_table_map_; }
    const std::vector<brg_range_handle_ptr> &get_brg_range_handle_vec() const {
        return brg_handles_;
    }
    std::vector<brg_range_handle_ptr> &get_brg_range_handle_vec() {
        return brg_handles_;
    }

    // run the pass on all functions in this module
    void run_pass(function_pass_t &pass);

    func_t make_init_func() const;

    var make_global_var(sc_data_type_t dtype, const std::string &name,
            linkage linkage = linkage::private_global, expr init = expr());
    tensor make_global_tensor(sc_data_type_t dtype, const std::string &name,
            const std::vector<expr> &dims,
            linkage linkage = linkage::private_global);
    // make a global tensor with strides
    tensor make_global_stensor(sc_data_type_t dtype, const std::string &name,
            const std::vector<expr> &dims, const std::vector<expr> &strides,
            linkage linkage = linkage::private_global,
            stmt *out_def_node = nullptr);
    // adds a global var def, handles renaming
    void add_global_var(define def);
    // adds a pair of name and op table for dynamic dispatch
    void add_op_table(const std::pair<std::string, op_dispatch_tables_ptr> &tb);
    // gets the entry func_t. nullable
    func_t get_entry_func() const {
        return entry_func_idx_ >= 0 ? contents_[entry_func_idx_] : func_t();
    }
    // get var define node from input symbol, if could not find, return nullptr.
    define get_var_def_from_symbol(const std::string &symbol) const {
        auto it = var_symbols_.find(symbol);
        if (it != var_symbols_.end()) { return it->second; }
        return define();
    }

    // adds a list of functions to the module, resolves dependencies and handles
    // name duplications
    void add_func(const std::vector<func_t> &f);

    // adds the functions of another module to this module. Will handle the
    // renaming of the functions returns this
    ir_module_t *merge(const ir_module_t &m);

    // adds the function of another module list to this module.
    ir_module_t *merge(const std::vector<std::shared_ptr<ir_module_t>> &list);

    // gets a function from the module. If the name is not in the symbol table,
    // returns null
    func_t get_func(const std::string &name) const;

    // copies the module
    std::shared_ptr<ir_module_t> copy() const;
    // deep copies the module
    std::shared_ptr<ir_module_t> deep_copy() const;

    // copies the module and remove the specified funcs by mask. A function will
    // be copied to the returned IR module only if it is get_contents()[i] and
    // mask[i] is true.
    std::shared_ptr<ir_module_t> copy_and_remove_funcs(
            const std::vector<bool> &mask) const;

    /**
     * Creates an IR module from a list of functions. Finds the direct and
     * indirect dependent functions of the given functions. Also rename the
     * functions with duplicated names to "XXX_1", "XXX_1_1", etc.
     * @param ctx the context
     * @param f the list of functions
     * @return the `ir_module_t` containing the given list of functions and all
     *  dependent functions. The names of the functions in the module are unique
     * */
    static std::shared_ptr<ir_module_t> from_entry_func(
            context_ptr ctx, const std::vector<func_t> &f);

    /**
     * @see from_entry_func overloaded function
     * */
    static std::shared_ptr<ir_module_t> from_entry_func(
            context_ptr ctx, func_t f);

private:
    // adds a list of functions to the module, dependency already resolved
    void add_resolved_func(const std::vector<func_t> &f);
};

using ir_module_ptr = std::shared_ptr<ir_module_t>;
using const_ir_module_ptr = std::shared_ptr<const ir_module_t>;

extern ostream &operator<<(ostream &os, const const_ir_module_ptr &);
SC_API extern ostream &operator<<(ostream &os, const ir_module_ptr &);
SC_API extern ostream &operator<<(ostream &os, const ir_module_t &);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
