/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_IR_BUILDER_HPP
#define GPU_INTEL_JIT_V2_IR_BUILDER_HPP

#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/reduce.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/jit/v2/ir/bridge.hpp"
#include "gpu/intel/jit/v2/ir/plan.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

struct loop_t {
    size_t idx = 0;
    pvar_t dim;
    expr_t var;
    expr_t init;
    expr_t bound;

    loop_t() = default;
    loop_t(size_t idx, const pvar_t &dim, const expr_t &var, const expr_t &init,
            const expr_t &bound)
        : idx(idx), dim(dim), var(var), init(init), bound(bound) {}
};

class loop_nest_t {
public:
    loop_nest_t() = default;

    void add_loop(const pvar_t &dim, const expr_t &idx, const expr_t &init,
            const expr_t &bound) {
        loops_.emplace_back(loops_.size(), dim, idx, init, bound);
    }

    void set_linear_bound(const expr_t &linear_bound) {
        linear_bound_ = linear_bound;
    }

    size_t nloops() const { return loops_.size(); }
    const expr_t &linear_bound() const { return linear_bound_; }
    const loop_t &operator[](size_t idx) const { return loops_[idx]; }
    std::vector<expr_t> indices() const {
        std::vector<expr_t> ret;
        ret.reserve(nloops());
        for (size_t i = 0; i < nloops(); i++) {
            ret.push_back(loops_[i].var);
        }
        return ret;
    }

    std::vector<expr_t> init_exprs() const {
        std::vector<expr_t> ret;
        ret.reserve(nloops());
        for (size_t i = 0; i < nloops(); i++) {
            ret.push_back(loops_[i].init);
        }
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "nloops: " << nloops();
        for (size_t i = 0; i < nloops(); i++) {
            oss << std::endl;
            oss << "  var: " << loops_[i].var << " init: " << loops_[i].init
                << " bound: " << loops_[i].bound;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    std::vector<loop_t> loops_;
    expr_t linear_bound_;
};

struct offset_params_t {
    // Type of the offset.
    type_t type;
    // Execution size:
    // - esize = 1: used as a scalar
    // - esize > 1: used as a vector
    // Note that esize > 1 may be used with scalar offsets, in this case the
    // offset is broadcasted when used.
    int esize = 0;
    // Offset buffer size alignment (e.g. used for header allocations, aligned
    // at a GRF boundary).
    int buf_align = 0;
    // Whether the offset can be used with broadcasting (e.g. scalar mask with
    // multiple slots).
    bool allow_bcast = false;
    // Whether the offset can be used directly from the base (if the offset is
    // equal to the base).
    bool allow_reuse = false;
    // Whether inline initialization can be used (see offset_t for details).
    bool allow_inline_init = false;
    // Optional pre-allocated buffer for the offset.
    expr_t buf;
    // Prefix for the buffer name.
    std::string buf_prefix;

    offset_params_t(
            const type_t &type, int esize = 1, const char *buf_prefix = nullptr)
        : type(type), esize(esize) {
        if (buf_prefix) this->buf_prefix = buf_prefix;
    }

    expr_t get_buffer(buffer_manager_t &buf_mgr, int size) const {
        if (buf_prefix.empty()) return buf;
        auto buf_name = buf_mgr.ir_ctx().create_tmp_name(buf_prefix);
        return buf_mgr.get(buf_name, size);
    }
};

// Offset is represented as the sum of three terms:
//     base + shift + shift_vec
// where:
// - (base + shift) is a scalar portion
// - shift_vec is a vector portion
//
// base/shift split is relative which is determined during load/store planning
// to group instructions performing access to a shifted tiles of the same
// sub-layout. In general "shift" portion consists of simpler expressions
// comparing with "base".
// shift_vec is a vector of offsets (e.g. for per slot in a message or per lane
// in a mask comparison).
struct offset_t {
    // Offset version. This is relevant for offsets that are used in multiple
    // versions of the same loop, e.g. load and prefetch.
    int version = -1;
    // GRF buffer for the offset. If empty, the base storage is used for the
    // offset.
    expr_t buf;
    // Offset type (scalar or vector).
    type_t type;
    // Scalar base.
    expr_t base;
    // Scalar shift.
    expr_t shift;
    // Vector shift.
    expr_t shift_vec;
    // Loop increments, used to implement strength reduction.
    std::vector<expr_t> loop_incs;
    // Execution size.
    int esize;
    // Inline initialization. When set, the offset is initialized right before
    // use. This implies no loop increments and no pre-initialization. This is
    // used as an optimization when offset A is a shifted version of another
    // offset B: in this case we can do A = B + shift and avoid any other
    // operations.
    stmt_t inline_init;

    bool is_equal(const offset_t &other, bool compare_shift = true) const {
        if (version != other.version) return false;
        if (type != other.type) return false;
        if (!base.is_equal(other.base)) return false;
        if (compare_shift && !shift.is_equal(other.shift)) return false;
        if (!shift_vec.is_equal(other.shift_vec)) return false;
        if (!ir_utils::is_equal(loop_incs, other.loop_incs)) return false;
        if (esize != other.esize) return false;
        if (!ir_utils::is_equal(inline_init, other.inline_init)) return false;
        return true;
    }

    bool operator==(const offset_t &other) const { return is_equal(other); }

    expr_t load() const {
        if (buf.is_empty()) return make_broadcast(base);
        return make_broadcast(load_t::make(type, buf, 0));
    }

    stmt_t store(const expr_t &_value) const {
        auto value = _value;
        if (value.type() != type) value = cast(value, type);
        return store_t::make(buf, 0, value);
    }

    stmt_t init_stmt() const {
        if (buf.is_empty() || !inline_init.is_empty()) return stmt_t();
        auto base_bcast = shuffle_t::make_broadcast(base + shift, type.elems());
        return store(base_bcast + shift_vec);
    }

    stmt_t inc_stmt(int loop_idx) const {
        if (loop_incs.empty()) return stmt_t();
        auto inc = loop_incs[loop_idx];
        if (is_zero(inc)) return stmt_t();
        inc = shuffle_t::make_broadcast(inc, type.elems());
        auto value = load_t::make(type, buf, 0) + inc;
        return store(value);
    }

    expr_t make_broadcast(const expr_t &e) const {
        if (e.type().elems() == esize) return e;
        return shuffle_t::make_broadcast(e, esize);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "buf:       " << buf << std::endl;
        oss << "base:      " << base << std::endl;
        oss << "shift:     " << shift << std::endl;
        oss << "shift_vec: " << shift_vec << std::endl;
        oss << "loop_incs:";
        for (int i = 0; i < (int)loop_incs.size(); i++) {
            oss << std::endl;
            oss << "  " << loop_incs[i];
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static bool can_reuse_base(const type_t &type, const expr_t &base,
            const expr_t &shift, const expr_t &shift_vec,
            const std::vector<expr_t> &loop_incs) {
        if (!type.is_scalar()) return false;
        if (!is_zero(shift)) return false;
        if (!is_var(base) && !is_const(base)) return false;
        if (!all_of(shift_vec, 0)) return false;
        for (auto &e : loop_incs)
            if (!is_zero(e)) return false;
        return true;
    }
};

class send_header_t {
public:
    send_header_t() = default;
    send_header_t(const offset_t &off) : off_(off) {}
    const offset_t &off() const { return off_; }
    const expr_t &to_expr() const { return off_.buf; }

private:
    offset_t off_;
};

class send_mask_t {
public:
    send_mask_t() = default;

    void add_mask(
            const offset_t &off, const expr_t &bound, bool has_underflow) {
        entries_.emplace_back(off, bound, has_underflow);
    }

    expr_t to_expr() const {
        if (entries_.empty()) return expr_t();
        expr_t ret;
        for (auto &e : entries_) {
            auto cmp = (e.off.load() < e.off.make_broadcast(e.bound));
            ret = (ret.is_empty() ? std::move(cmp) : (ret & cmp));
            if (e.has_underflow)
                ret &= (e.off.load() >= e.off.make_broadcast(0));
        }
        return ret;
    }

private:
    struct entry_t {
        entry_t() = default;
        entry_t(const offset_t &off, const expr_t &bound, bool has_underflow)
            : off(off), bound(bound), has_underflow(has_underflow) {}
        offset_t off;
        expr_t bound;
        bool has_underflow = false;
    };

    std::vector<entry_t> entries_;
};

class offset_scope_t {
public:
    offset_scope_t(buffer_manager_t &buf_mgr) : buf_mgr_(buf_mgr) {}
    buffer_manager_t &buf_mgr() { return buf_mgr_; }
    int make_version() { return version_++; }

    send_header_t add_header(int version, const send_1d_desc_t &desc,
            const expr_t &mem_buf, const addr_t &addr, const expr_t &addr_inc,
            const loop_nest_t &loop_nest) {
        auto base0 = cast(mem_buf, type_t::u64());
        auto params = offset_params_t(type_t::u64(), desc.slots, "h");
        params.buf_align = buf_mgr_.ir_ctx().grf_size();
        params.allow_inline_init = true;
        auto off = get_offset(version, base0, addr.base, addr.slot_incs,
                addr_inc, params, loop_nest);
        return send_header_t(off);
    }

    send_header_t add_header(int version, const send_2d_desc_t &desc,
            const expr_t &mem_buf, const expr_t &base, const expr_t &x_base,
            const expr_t &y_base, const expr_t &x_inc, const expr_t &y_inc,
            const loop_nest_t &loop_nest);
    send_mask_t add_mask(int version, const mask_t &mask,
            const std::vector<expr_t> &mask_incs, const loop_nest_t &loop_nest);

    stmt_t init_stmt(int version) const {
        stmt_t ret;
        for (auto &o : offsets_) {
            if (o.version != version) continue;
            ret = ret.append(o.init_stmt());
        }
        return ret;
    }

    stmt_t inc_loop_stmt(int loop_idx, int version) const {
        stmt_t ret;
        for (auto &o : offsets_) {
            if (o.version != version) continue;
            auto inc = o.inc_stmt(loop_idx);
            ret = ret.append(inc);
        }
        return ret;
    }

    stmt_t inject_let_stmts(const stmt_t &stmt) const {
        return jit::inject_let_stmts(stmt, let_stmts_);
    }

private:
    // base0 - memory buffer base address
    // base, shift_vec, shift - offset parts (see offset_t description)
    offset_t get_offset(int version, const expr_t &base0, const expr_t &base,
            const std::vector<expr_t> &_shift_vec, const expr_t &_shift,
            const offset_params_t &_params, const loop_nest_t &loop_nest);

    offset_t get_offset(int version, const expr_t &base0, const expr_t &base,
            const expr_t &shift, const offset_params_t &_params,
            const loop_nest_t &loop_nest) {
        return get_offset(version, base0, base, std::vector<expr_t>(), shift,
                _params, loop_nest);
    }

    offset_t get_offset(int version, const expr_t &base, const expr_t &buf,
            const loop_nest_t &loop_nest) {
        offset_t ret;
        ret.version = version;
        ret.buf = buf;
        ret.type = base.type();
        ret.base = base;
        ret.shift = expr_t(0);
        ret.shift_vec = expr_t(0);
        ret.esize = 1;
        return add_offset(ret);
    }

    offset_t add_offset(const offset_t &off) {
        offsets_.push_back(off);
        return off;
    }

    expr_t to_simple_expr(const expr_t &e) {
        if (is_const(e) || e.is<const_var_t>() || e.is<var_t>()) return e;
        auto it = expr_to_let_var_.find(e);
        if (it != expr_to_let_var_.end()) return it->second;
        auto tmp_var = buf_mgr_.ir_ctx().create_tmp_var(type_t::s32());
        let_stmts_.push_back(let_t::make(tmp_var, e));
        expr_to_let_var_.emplace(e, tmp_var);
        return tmp_var;
    }

    buffer_manager_t &buf_mgr_;
    object_eq_map_t<expr_t, expr_t> expr_to_let_var_;
    std::vector<stmt_t> let_stmts_;
    std::vector<offset_t> offsets_;
    object_map_t<expr_t, int> buf_versions_;
    int version_ = 0;
};

class offset_ctx_t {
public:
    offset_ctx_t() = default;
    offset_ctx_t(offset_scope_t *scope, const loop_nest_t &loop_nest = {})
        : scope_(scope)
        , loop_nest_(loop_nest)
        , version_(scope->make_version()) {}

    send_header_t add_header(const send_1d_desc_t &desc, const expr_t &mem_buf,
            const addr_t &addr, const expr_t &addr_inc) {
        return scope_->add_header(
                version_, desc, mem_buf, addr, addr_inc, loop_nest_);
    }

    send_header_t add_header(const send_2d_desc_t &desc, const expr_t &mem_buf,
            const expr_t &base, const expr_t &x_base, const expr_t &y_base,
            const expr_t &x_inc, const expr_t &y_inc) {
        return scope_->add_header(version_, desc, mem_buf, base, x_base, y_base,
                x_inc, y_inc, loop_nest_);
    }

    send_mask_t add_mask(const mask_t &mask,
            const std::vector<expr_t> &mask_incs = std::vector<expr_t>()) {
        return scope_->add_mask(version_, mask, mask_incs, loop_nest_);
    }

    stmt_t init_stmt() const { return scope_->init_stmt(version_); }
    stmt_t inc_loop_stmt(int loop_idx) const {
        return scope_->inc_loop_stmt(loop_idx, version_);
    }

private:
    offset_scope_t *scope_ = nullptr;
    loop_nest_t loop_nest_;
    int version_ = -1;
};

inline stmt_t create_stmt(const reduce_plan_t &plan, const expr_t &src_buf,
        const expr_t &dst_buf) {
    if (!plan) return stmt_t();
    return create_reduce_stmt(
            to_ir(plan.src), to_ir(plan.dst), src_buf, dst_buf);
}

inline stmt_t create_stmt(const reorder_plan_t &plan, const expr_t &src_buf,
        const expr_t &dst_buf) {
    if (!plan) return stmt_t();
    return create_reorder_stmt(
            to_ir(plan.src), to_ir(plan.dst), src_buf, dst_buf);
}

stmt_t create_stmt(const send_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const pvar_coord_t<dim_t> &coord, const pvar_tile_t &tile);

inline stmt_t create_stmt(const send_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx) {
    return create_stmt(plan, mem_buf, reg_buf, off_ctx, pvar_coord_t<dim_t>(),
            plan.reg_layout().int_dim_sizes());
}

class ir_builder_t;

class var_ref_t {
public:
    var_ref_t(ir_builder_t *parent, const type_t &type, const expr_t &buf)
        : parent_(parent), type_(type), buf_(buf) {
        gpu_assert(buf_.type().is_ptr());
    }

    operator expr_t() const { return load_t::make(type_, buf_, 0); }
    var_ref_t &operator=(const expr_t &value);
    var_ref_t &operator=(const var_ref_t &other) = default;
    std::string str() const { return buf_.str(); }

    IR_DEFINE_DUMP()

private:
    ir_builder_t *parent_;
    type_t type_;
    expr_t buf_;
};

class ir_builder_t {
public:
    ir_builder_t(ir_context_t &ir_ctx)
        : buf_mgr_(std::make_shared<buffer_manager_t>(ir_ctx))
        , off_scope_(std::make_shared<offset_scope_t>(*buf_mgr_))
        , off_ctx_(off_scope_.get()) {
        enter_scope();
    }
    ir_builder_t(ir_builder_t &parent, const loop_nest_t &loop_nest)
        : buf_mgr_(parent.buf_mgr_)
        , off_scope_(parent.off_scope_)
        , off_ctx_(off_scope_.get(), loop_nest) {
        enter_scope();
    }
    ir_builder_t(const ir_builder_t &parent) = delete;
    const hw_t &hw() const { return buf_mgr_->ir_ctx().hw(); }
    ir_context_t &ir_ctx() { return buf_mgr_->ir_ctx(); }
    buffer_manager_t &buf_mgr() { return *buf_mgr_; }
    const offset_scope_t &off_scope() const { return *off_scope_; }
    const offset_ctx_t &off_ctx() const { return off_ctx_; }
    expr_t alloc(const std::string &_name, int size) {
        auto name = (buf_mgr_->has(_name)
                        ? buf_mgr_->ir_ctx().create_tmp_name(_name)
                        : _name);
        return buf_mgr_->get(name, size);
    }
    var_ref_t alloc_var(const type_t &type, const std::string &_name) {
        auto name = (buf_mgr_->has(_name)
                        ? buf_mgr_->ir_ctx().create_tmp_name(_name)
                        : _name);
        auto buf = alloc(name, type.size());
        return var_ref_t(this, type, buf);
    }
    expr_t get_or_alloc(const std::string &name, int size) {
        return buf_mgr_->get(name, size);
    }

    void zero_out(const expr_t &reg_buf, int size = 0) {
        if (size == 0) size = buf_mgr_->size(reg_buf);
        auto stmt = funcs::zero_out(reg_buf, size);
        emit(stmt);
    }

    expr_t let(const std::string &prefix, const expr_t &value) {
        auto name = buf_mgr_->ir_ctx().create_tmp_name(prefix);
        auto var = var_t::make(value.type(), name);
        let(var, value);
        return var;
    }

    expr_t let(const expr_t &var, const expr_t &value) {
        emit(let_t::make(var, value, stmt_t()));
        return var;
    }

    expr_t let(const expr_t &value) { return let("tmp", value); }

    expr_t load(const view_t &mem_view, const expr_t &mem_buf,
            const expr_t &reg_buf = {}, layout_t *reg_layout = nullptr) {
        send_params_t params;
        params.hw = hw();
        params.address = send_address_t::a64;
        params.op = send_op_t::load;
        params.init_max_entry_reg_size();
        auto plan = create_send_plan(params, mem_view);
        return load(plan, mem_buf, reg_buf, reg_layout);
    }

    expr_t load(const send_plan_t &plan, const expr_t &mem_buf,
            expr_t reg_buf = {}, layout_t *reg_layout = nullptr) {
        auto buf_prefix = get_buf_name(mem_buf) + "_buf";
        if (plan.op() != send_op_t::prefetch && reg_buf.is_empty()) {
            reg_buf = alloc(buf_prefix,
                    utils::rnd_up(plan.reg_layout().size(), hw().grf_size()));
        }
        auto load_stmt = create_stmt(plan, mem_buf, reg_buf, off_ctx_);
        emit(load_stmt);
        if (reg_layout) *reg_layout = plan.reg_layout();
        return reg_buf;
    }

    void store(const view_t &mem_view, const expr_t &mem_buf,
            const layout_t &reg_layout, const expr_t &reg_buf) {
        send_params_t params;
        params.hw = hw();
        params.address = send_address_t::a64;
        params.op = send_op_t::store;
        params.init_max_entry_reg_size();
        auto plan = create_send_plan(params, mem_view);
        gpu_assert(plan.reg_layout() == reg_layout);
        store(plan, mem_buf, reg_buf);
    }

    void store(const send_plan_t &plan, const expr_t &mem_buf,
            const expr_t &reg_buf) {
        auto store_stmt = create_stmt(plan, mem_buf, reg_buf, off_ctx_);
        emit(store_stmt);
    }

    void store(const send_plan_t &plan, const expr_t &mem_buf,
            const expr_t &reg_buf, const pvar_coord_t<dim_t> &coord,
            const pvar_tile_t &tile) {
        auto store_stmt
                = create_stmt(plan, mem_buf, reg_buf, off_ctx_, coord, tile);
        emit(store_stmt);
    }

    expr_t reorder(const layout_t &src, const layout_t &dst,
            const expr_t &src_buf, const expr_t &dst_buf = {}) {
        if (src == dst) return src_buf;
        auto plan = reorder_plan_t(hw(), src, dst);
        return reorder(plan, src_buf, dst_buf);
    }

    expr_t reorder(const reorder_plan_t &plan, const expr_t &src_buf,
            expr_t dst_buf = {}) {
        if (dst_buf.is_empty()) { dst_buf = alloc("tmp", plan.dst.size()); }
        auto reorder_stmt = create_stmt(plan, src_buf, dst_buf);
        emit(reorder_stmt);
        return dst_buf;
    }

    void reduce(const reduce_plan_t &plan, const expr_t &src_buf,
            const expr_t &dst_buf) {
        auto reduce_stmt = create_stmt(plan, src_buf, dst_buf);
        emit(reduce_stmt);
    }

    void reduce(const layout_t &src, const layout_t &dst, const expr_t &src_buf,
            const expr_t &dst_buf, uint32_t mask) {
        auto stmt = create_reduce_stmt(
                to_ir(src), to_ir(dst), src_buf, dst_buf, tensor_t(), mask);
        emit(stmt);
    }

    void barrier() { emit(funcs::barrier()); }

    template <typename BodyFuncT>
    void _if(const expr_t &cond, const BodyFuncT &body_func) {
        enter_scope();
        body_func();
        emit(if_t::make(cond, exit_scope()));
    }

    template <typename BodyFuncT>
    void _for(const expr_t &var, const expr_t &init, const expr_t &bound,
            const BodyFuncT &body_func) {
        enter_scope();
        body_func();
        emit(for_t::make(var, init, bound, exit_scope()));
    }

    template <typename BodyFuncT>
    void _while(const expr_t &cond, const BodyFuncT &body_func) {
        enter_scope();
        body_func();
        emit(while_t::make(cond, exit_scope()));
    }

    void emit(const stmt_t &stmt) { top_stmt() = top_stmt().append(stmt); }
    stmt_t get_stmt() const { return top_stmt(); }
    void set_stmt(const stmt_t &stmt) {
        gpu_assert(stmt_stack_.size() == 1);
        top_stmt() = stmt;
    }
    stmt_t get_init_stmt() const { return off_ctx().init_stmt(); }

private:
    static const std::string &get_buf_name(const expr_t &e) {
        auto *var = e.as_ptr<var_t>();
        gpu_assert(var) << e;
        return var->name;
    }

    stmt_t &top_stmt() {
        gpu_assert(!stmt_stack_.empty());
        return stmt_stack_.back();
    }
    const stmt_t &top_stmt() const {
        gpu_assert(!stmt_stack_.empty());
        return stmt_stack_.back();
    }
    void enter_scope() { stmt_stack_.emplace_back(); }
    stmt_t exit_scope() {
        auto ret = stmt_stack_.back();
        stmt_stack_.pop_back();
        return ret;
    }

    std::shared_ptr<buffer_manager_t> buf_mgr_;
    std::shared_ptr<offset_scope_t> off_scope_;
    offset_ctx_t off_ctx_;
    std::vector<stmt_t> stmt_stack_;
};

inline var_ref_t &var_ref_t::operator=(const expr_t &value) {
    gpu_assert(value.type() == type_);
    parent_->emit(store_t::make(buf_, 0, value));
    return *this;
}

class var_manager_t {
public:
    var_manager_t(const kernel_iface_t &kernel_iface)
        : kernel_iface_(kernel_iface) {}

    std::vector<expr_t> ptr_args() const {
        std::vector<expr_t> ret;
        for (int i = 0; i < kernel_iface_.nargs(); i++) {
            auto &var = kernel_iface_.arg_var(i);
            if (var.type().is_ptr()) ret.push_back(var);
        }
        return ret;
    }

    expr_t get_arg(const std::string &name, bool allow_empty = false) const {
        return kernel_iface_.find_arg(name, allow_empty);
    }

    expr_t get_grid_size(const std::string &name) const {
        return get_arg(type_t::u32(), name + "_grid_size");
    }

    expr_t get_idiv_magic(const expr_t &value) const {
        std::string name;
        if (auto *op = value.as_ptr<binary_op_t>()) {
            if (op->op_kind == op_kind_t::_div_up) {
                gpu_assert(is_const(op->b))
                        << "Expected constant denominator: " << value;
                if (is_one(op->b)) return get_idiv_magic(op->a);
                gpu_assert(op->a.is<var_t>() || op->a.is<const_var_t>())
                        << "Expected var/const var: " << op->a;
                name = op->a.str();
                name += "_divup_" + op->b.str();
            }
        } else {
            gpu_assert(value.is<var_t>() || value.is<const_var_t>())
                    << "Expected var/const var: " << value;
            name = value.str();
        }
        return get_arg(type_t::u64(), name + "_magic");
    }

    expr_t get_arg(const type_t &type, const std::string &name) const {
        gpu_assert(kernel_iface_.has(name)) << "Cannot find argument " << name;
        auto var = kernel_iface_.find_arg(name);
        gpu_assert(var.type() == type) << "Type mismatch, found: " << var.type()
                                       << " expected: " << type;
        return var;
    }

private:
    const kernel_iface_t &kernel_iface_;
};

} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
