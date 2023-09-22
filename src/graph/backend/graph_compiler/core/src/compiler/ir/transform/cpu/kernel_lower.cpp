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
#include <unordered_map>

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "../../builder.hpp"
#include "../../content_hash.hpp"
#include "../../visitor.hpp"
#include "../buffer_schedule.hpp"
#include "../index_flatten.hpp"
#include "kernel_lower.hpp"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <runtime/microkernel/cpu/brgemm_range_handle.hpp>
#include <util/hash_utils.hpp>

SC_MODULE(pass.kernel_lowering_cpu)
namespace std {
template <>
struct hash<dnnl::impl::graph::gc::brgemm::postop_setting_t> {
    std::size_t operator()(
            const dnnl::impl::graph::gc::brgemm::postop_setting_t &k) const {
        size_t ret = 0;
        dnnl::impl::graph::gc::hash_combine(ret, k.pack_info_[0]);
        dnnl::impl::graph::gc::hash_combine(ret, k.pack_info_[1]);
        return ret;
    }
};

template <>
struct hash<std::pair<dnnl::impl::graph::gc::sc_brgemm_bd_mask_t, int>> {
    std::size_t operator()(
            const std::pair<dnnl::impl::graph::gc::sc_brgemm_bd_mask_t, int>
                    &bdmask_pair) const {
        size_t ret = 0;
        for (auto &m : bdmask_pair.first) {
            dnnl::impl::graph::gc::hash_combine(ret, m);
        }
        dnnl::impl::graph::gc::hash_combine(ret, bdmask_pair.second);
        return ret;
    }
};
} // namespace std
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(kernel_lowering_cpu,
        SC_PASS_DEPENDS_ON(constant_folder, buffer_scheduler),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

using namespace builtin;
using namespace brgemm;

static bool check_arg_range(const std::vector<expr> &args,
        std::vector<expr_c> &cached_args, int start, int end) {
    cached_args.reserve(cached_args.size() + end - start);
    // check the parameters, if they are all constants,
    // we can cache the kernel
    for (int i = start; i < end; i++) {
        if (!args[i].isa<constant>() && !args[i].isa<tensor>()
                && !args[i].isa<indexing>()) {
            // not const, don't cache
            return false;
        }
        cached_args.emplace_back(args[i]);
    }
    return true;
}

static std::vector<int64_t> get_brgemm_attrs_key(const sc_brgemm_attrs_t &attrs,
        size_t &valid_sz, bool &range_optimize) {
    std::vector<int64_t> key(attr_key::nkeys, 0);
    valid_sz = 0;
    for (auto &attr : attrs) {
        if (attr.first < brgemm::attr_key::nkeys) {
            valid_sz++;
            key[attr.first] = attr.second;
        } else if (utils::is_one_of(attr.first,
                           brgemm::attr_key::M_range_upper_bound,
                           brgemm::attr_key::N_range_upper_bound,
                           brgemm::attr_key::K_range_upper_bound)) {
            range_optimize = true;
        }
    }
    return key;
}

static void validate_brgemm_attrs(const sc_brgemm_attrs_t &brg_attrs,
        const sc_brgemm_bd_mask_t &bd_mask, const int bd_mask_set_num,
        bool &use_bd_mask) {
    size_t dummy;
    bool dummy1;
    auto keys = get_brgemm_attrs_key(brg_attrs, dummy, dummy1);
    if (keys[attr_key::use_uker]) {
        COMPILE_ASSERT(keys[attr_key::max_bs] > 0,
                "max_bs should be >0 and valid number for the bs used by "
                "brgemm when use_uker=true, but got "
                        << keys[attr_key::max_bs] << ".");
    }
    use_bd_mask = false;
    if (keys[attr_key::bd_mask_level] > 0) {
        COMPILE_ASSERT(!bd_mask.empty(),
                "bd_mask should be specified when bd_mask_level>0.");
        COMPILE_ASSERT(bd_mask_set_num > 0,
                "bd_mask_set_num should be >0 when bd_mask_level>0, but got "
                        << bd_mask_set_num << ".");
        use_bd_mask = true;
    }
}

static expr get_brgemm_attrs_arg(const ir_module_ptr &mod,
        const sc_brgemm_attrs_t &attrs,
        std::unordered_map<std::vector<int64_t>, expr> &cache,
        bool &range_optimize) {
    size_t sz;
    std::vector<int64_t> key = get_brgemm_attrs_key(attrs, sz, range_optimize);
    if (sz == 0) { return get_ir_null(); }
    COMPILE_ASSERT(sz <= attr_key::nkeys,
            "Size of user defined attributes should not exceed nkeys!");
    if (cache.find(key) != cache.end()) { return cache[key]; }
    size_t tsr_sz = sz * sizeof(attrs_setting_t::attrs_map_t) + sizeof(int64_t);
    std::vector<char> data(tsr_sz, 0);
    attrs_setting_t *setting = reinterpret_cast<attrs_setting_t *>(data.data());
    setting->num_ = sz;
    int c = 0;
    for (auto &it : attrs) {
        if (it.first < brgemm::attr_key::nkeys) { setting->map_[c++] = it; }
    }
    auto init = std::make_shared<static_data_t>(data.data(), tsr_sz);
    auto tsr = builder::make_tensor("__brgemm_attrs", {tsr_sz}, datatypes::u8,
            address_space::automatic, init);
    mod->add_global_var(builder::make_var_tensor_def_unattached(
            tsr, linkage::private_global)
                                .static_as<define>());
    cache[key] = tsr;
    return tsr;
}

static std::pair<expr, expr> get_brgemm_bd_mask_arg(const ir_module_ptr &mod,
        const sc_brgemm_bd_mask_t &bd_mask, const expr &bd_mask_idx,
        const expr &bd_mask_len, const int bd_mask_set_num,
        std::unordered_map<std::pair<sc_brgemm_bd_mask_t, int>,
                std::pair<expr, expr>> &cur_cache,
        std::unordered_map<sc_brgemm_bd_mask_t, expr> &full_cached,
        std::vector<std::vector<stmt>> &brg_bd_mask_arr_assignment) {
    size_t sz = bd_mask.size();
    auto bd_mask_pair
            = std::pair<sc_brgemm_bd_mask_t, int>(bd_mask, bd_mask_set_num);
    if (sz == 0) { return std::pair<expr, expr>(get_ir_null(), get_ir_null()); }
    if (cur_cache.find(bd_mask_pair) != cur_cache.end()) {
        return cur_cache[bd_mask_pair];
    }

    expr full_bd_mask;
    if (full_cached.find(bd_mask) == full_cached.end()) {
        size_t tsr_sz = sz;
        auto init = std::make_shared<static_data_t>(bd_mask);
        expr full_bd_mask_ = builder::make_tensor("__brgemm_full_bd_mask",
                {tsr_sz}, datatypes::u8, address_space::automatic, init);
        mod->add_global_var(builder::make_var_tensor_def_unattached(
                full_bd_mask_, linkage::private_global)
                                    .static_as<define>());
        full_bd_mask = full_bd_mask_;
        full_cached[bd_mask] = full_bd_mask;
    } else {
        full_bd_mask = full_cached[bd_mask];
    }

    expr bd_mask_arr = mod->make_global_tensor(datatypes::pointer,
            "__brgemm_bd_mask_arr", {bd_mask_set_num}, linkage::private_global);
    std::vector<stmt> bd_mask_arr_assignment(bd_mask_set_num);
    for (int i = 0; i < bd_mask_set_num; ++i) {
        bd_mask_arr_assignment.at(i) = builder::make_assign_unattached(
                builder::make_indexing(bd_mask_arr, {i}),
                builder::tensor_ptr(full_bd_mask, {i * bd_mask_len}));
    }
    brg_bd_mask_arr_assignment.push_back(bd_mask_arr_assignment);
    auto ret = std::pair<expr, expr>(bd_mask_arr, bd_mask_arr[bd_mask_idx]);
    cur_cache[bd_mask_pair] = ret;

    return ret;
}

static expr get_brgemm_postops_setting_arg(const ir_module_ptr &mod,
        const sc_brgemm_postops_setting_t &postops,
        std::unordered_map<sc_brgemm_postops_setting_t, expr> &cache) {
    if (postops.empty()) { return get_ir_null(); }
    if (cache.find(postops) != cache.end()) { return cache[postops]; }
    size_t sz = postops.size();
    size_t tsr_sz = sz * sizeof(postop_setting_t) + sizeof(int64_t);
    std::vector<char> data(tsr_sz, 0);
    postops_setting_t *setting
            = reinterpret_cast<postops_setting_t *>(data.data());
    setting->num_ = sz;
    int c = 0;
    for (auto &op : postops) {
        setting->ops_[c++] = op;
    }
    auto init = std::make_shared<static_data_t>(data.data(), tsr_sz);
    expr tsr = builder::make_tensor("__brgemm_postops_setting", {tsr_sz},
            datatypes::u8, address_space::automatic, init);
    mod->add_global_var(builder::make_var_tensor_def_unattached(
            tsr, linkage::private_global)
                                .static_as<define>());
    cache[postops] = tsr;
    return tsr;
}

// has post ops
static expr get_brgemm_postops_data_arg(
        std::vector<stmt_c> &ret, const std::vector<expr> &ops_data) {
    assert(ops_data.size() == brgemm::postops_data_init_func_nargs);
    auto &in_bufs = ops_data;
    auto postop_data = builder::make_tensor(
            "__brgemm_postops_data", {postops_data_size}, datatypes::u8);

    expr bin_ptr = in_bufs[2];
    if (!in_bufs[2]->equals(get_ir_null())) {
        bin_ptr = builder::make_tensor(
                "__binary_rhs_ptr", {1}, datatypes::pointer);
        ret.emplace_back(builder::make_var_tensor_def_unattached(bin_ptr));
        ret.back().remove_const()->attr().set(
                attr_keys::tsr_dont_buf_sched, true);
        ret.emplace_back(builder::make_assign_unattached(
                bin_ptr[UINT64_C(0)], in_bufs[2]));
    }
    ret.emplace_back(builder::make_var_tensor_def_unattached(postop_data));
    ret.emplace_back(builder::make_evaluate_unattached(
            builder::make_call(builtin::get_brgemm_postops_data_init_func(),
                    {postop_data, in_bufs[0], in_bufs[1], bin_ptr, in_bufs[3],
                            in_bufs[4], in_bufs[5], in_bufs[6], in_bufs[7],
                            in_bufs[8], in_bufs[9], in_bufs[10], in_bufs[11],
                            in_bufs[12], in_bufs[13]})));

    return postop_data;
}

static sc_data_type_t infer_out_dtype(const sc_data_type_t &in_dtype) {
    if (utils::is_one_of(in_dtype, datatypes::u8, datatypes::s8)) {
        return datatypes::s32;
    }
    return datatypes::f32;
}

static expr brgemm_init_kernel_cache(brgemm_mode mode,
        scflags_t::brgemm_t backend, const std::vector<expr> &args,
        std::vector<expr_c> &cached_args, float beta, bool has_postop) {
    if (mode == brgemm_mode::stride) {
        // cache basic args
        const int expected_cache_org_args = brgemm_args::NUM_BASIC_ARGS_STRIDE;
        const int expected_cache_extra_start
                = brgemm_args::NUM_FULL_ARGS_STRIDE;
        const int expected_cache_extra_args
                = brgemm_args::extra_args_offset::cache_nargs;
        // +1 for context pointer
        assert(args.size()
                == brgemm_args::NUM_FULL_ARGS_STRIDE
                        + brgemm_args::extra_args_offset::nargs + 1);
        if (!check_arg_range(args, cached_args, brgemm_args::M,
                    expected_cache_org_args)) {
            return expr();
        }
        if (!check_arg_range(args, cached_args, expected_cache_extra_start,
                    expected_cache_extra_start + expected_cache_extra_args)) {
            return expr();
        }
        return get_brgemm_creator_and_call_func(mode, backend, has_postop)
                .first(args[brgemm_args::M], args[brgemm_args::N],
                        args[brgemm_args::K], args[brgemm_args::LDA],
                        args[brgemm_args::LDB], args[brgemm_args::LDC],
                        args[brgemm_args::STRIDE_A],
                        args[brgemm_args::STRIDE_B], beta,
                        args[brgemm_args::NUM_FULL_ARGS_STRIDE
                                + brgemm_args::extra_args_offset::
                                        dtypeA], // dtypeA
                        args[brgemm_args::NUM_FULL_ARGS_STRIDE
                                + brgemm_args::extra_args_offset::
                                        dtypeB], // dtypeB
                        args[brgemm_args::NUM_FULL_ARGS_STRIDE
                                + brgemm_args::extra_args_offset::
                                        brg_attrs], // attrs
                        args[brgemm_args::NUM_FULL_ARGS_STRIDE
                                + brgemm_args::extra_args_offset::
                                        bd_mask], // bd_mask
                        args[brgemm_args::NUM_FULL_ARGS_STRIDE
                                + brgemm_args::extra_args_offset::
                                        postops_setting]); // postops set
    } else {
        // cache basic args and bdmask
        // brgemm_args::LEN does not require to be cached
        // so we use NUM_BASIC_ARGS_STRIDE rather than NUM_BASIC_ARGS_LIST here
        const int expected_cache_org_args = brgemm_args::NUM_BASIC_ARGS_STRIDE;
        const int expected_cache_extra_start = brgemm_args::NUM_FULL_ARGS_LIST;
        const int expected_cache_extra_args
                = brgemm_args::extra_args_offset::cache_nargs;
        // +1 for context pointer
        assert(args.size()
                == brgemm_args::NUM_FULL_ARGS_LIST
                        + brgemm_args::extra_args_offset::nargs + 1);
        if (!check_arg_range(args, cached_args, brgemm_args::M,
                    expected_cache_org_args)) {
            return expr();
        }
        if (!check_arg_range(args, cached_args, expected_cache_extra_start,
                    expected_cache_extra_start + expected_cache_extra_args)) {
            return expr();
        }
        return get_brgemm_creator_and_call_func(mode, backend, has_postop)
                .first(args[brgemm_args::M], args[brgemm_args::N],
                        args[brgemm_args::K], args[brgemm_args::LDA],
                        args[brgemm_args::LDB], args[brgemm_args::LDC], beta,
                        args[brgemm_args::NUM_FULL_ARGS_LIST
                                + brgemm_args::extra_args_offset::
                                        dtypeA], // dtypeA
                        args[brgemm_args::NUM_FULL_ARGS_LIST
                                + brgemm_args::extra_args_offset::
                                        dtypeB], // dtypeB
                        args[brgemm_args::NUM_FULL_ARGS_LIST
                                + brgemm_args::extra_args_offset::
                                        brg_attrs], // attrs
                        args[brgemm_args::NUM_FULL_ARGS_LIST
                                + brgemm_args::extra_args_offset::
                                        bd_mask], // bd_mask
                        args[brgemm_args::NUM_FULL_ARGS_LIST
                                + brgemm_args::extra_args_offset::
                                        postops_setting]); // postops set
    }
}

static expr brgemm_run(brgemm_mode mode, scflags_t::brgemm_t backend,
        const expr &cache, const std::vector<expr> &args, bool has_postop) {
    if (mode == brgemm_mode::stride) {
        const int expected_num_args = brgemm_args::NUM_FULL_ARGS_STRIDE
                + brgemm_args::extra_args_offset::nargs;
        assert(args.size() == expected_num_args + 1);
        auto run_func
                = get_brgemm_creator_and_call_func(mode, backend, has_postop)
                          .second;
        if (has_postop) {
            return run_func(cache, args[brgemm_args::A], args[brgemm_args::B],
                    args[brgemm_args::C], args[brgemm_args::NUM],
                    /*postop data*/
                    args[brgemm_args::NUM_FULL_ARGS_STRIDE
                            + brgemm_args::extra_args_offset::postops_data],
                    /*c_buf*/
                    args[brgemm_args::NUM_FULL_ARGS_STRIDE
                            + brgemm_args::extra_args_offset::c_buf],
                    /*ctx*/ args.back());
        }
        return run_func(cache, args[brgemm_args::A], args[brgemm_args::B],
                args[brgemm_args::C], args[brgemm_args::NUM],
                /*ctx*/ args.back());
    } else {
        const int expected_num_args = brgemm_args::NUM_FULL_ARGS_LIST
                + brgemm_args::extra_args_offset::nargs;
        assert(args.size() == expected_num_args + 1);
        auto run_func
                = get_brgemm_creator_and_call_func(mode, backend, has_postop)
                          .second;
        if (has_postop) {
            return run_func(cache, args[brgemm_args::A], args[brgemm_args::B],
                    args[brgemm_args::C], args[brgemm_args::NUM],
                    args[brgemm_args::STRIDE_A], args[brgemm_args::STRIDE_B],
                    args[brgemm_args::LEN],
                    args[brgemm_args::NUM_FULL_ARGS_LIST
                            + brgemm_args::extra_args_offset::dtypeA],
                    args[brgemm_args::NUM_FULL_ARGS_LIST
                            + brgemm_args::extra_args_offset::dtypeB],
                    /*postop data*/
                    args[brgemm_args::NUM_FULL_ARGS_LIST
                            + brgemm_args::extra_args_offset::postops_data],
                    /*c_buf*/
                    args[brgemm_args::NUM_FULL_ARGS_LIST
                            + brgemm_args::extra_args_offset::c_buf],
                    /*ctx*/ args.back());
        }
        return run_func(cache, args[brgemm_args::A], args[brgemm_args::B],
                args[brgemm_args::C], args[brgemm_args::NUM],
                args[brgemm_args::STRIDE_A], args[brgemm_args::STRIDE_B],
                args[brgemm_args::LEN],
                args[brgemm_args::NUM_FULL_ARGS_LIST
                        + brgemm_args::extra_args_offset::dtypeA],
                args[brgemm_args::NUM_FULL_ARGS_LIST
                        + brgemm_args::extra_args_offset::dtypeB],
                /*ctx*/ args.back());
    }
}

static expr range_brgemm_run(brgemm_mode mode, scflags_t::brgemm_t backend,
        const expr &cache, const std::vector<expr> &args, bool has_postop) {
    if (mode == brgemm_mode::stride) {
        const int expected_num_args = brgemm_args::NUM_FULL_ARGS_STRIDE
                + brgemm_args::extra_args_offset::nargs;
        assert(args.size() == expected_num_args + 1);
        auto run_func = get_brgemm_call_range_func(mode);
        return run_func(cache, args[brgemm_args::M], args[brgemm_args::N],
                args[brgemm_args::K], args[brgemm_args::A],
                args[brgemm_args::B], args[brgemm_args::C],
                args[brgemm_args::NUM],
                /*ctx*/ args.back());
    } else {
        const int expected_num_args = brgemm_args::NUM_FULL_ARGS_LIST
                + brgemm_args::extra_args_offset::nargs;
        assert(args.size() == expected_num_args + 1);
        auto run_func = get_brgemm_call_range_func(mode);
        return run_func(cache, args[brgemm_args::M], args[brgemm_args::N],
                args[brgemm_args::K], args[brgemm_args::A],
                args[brgemm_args::B], args[brgemm_args::C],
                args[brgemm_args::NUM], args[brgemm_args::STRIDE_A],
                args[brgemm_args::STRIDE_B], args[brgemm_args::LEN],
                args[brgemm_args::NUM_FULL_ARGS_LIST
                        + brgemm_args::extra_args_offset::dtypeA],
                args[brgemm_args::NUM_FULL_ARGS_LIST
                        + brgemm_args::extra_args_offset::dtypeB],
                /*ctx*/ args.back());
    }
}

class kernel_lower_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    // the kernel parameters => kernel pointer mapping
    using param_cache_table = content_hash_map<std::vector<expr_c>, expr>;
    ir_module_ptr mod_;
    int optimize_;
    int brg_bdmask_set_num_ = 0;
    bool brg_use_bdmask_ = false;
    // brgemm init stmts includes postops_data/c_buf
    std::vector<stmt_c> brg_postop_init_;
    std::unordered_map<std::vector<int64_t>, expr> attrs_cache_;
    std::unordered_map<std::pair<sc_brgemm_bd_mask_t, int>,
            std::pair<expr, expr>>
            cur_bd_mask_cache_;
    std::unordered_map<sc_brgemm_bd_mask_t, expr> full_bd_mask_cache_;
    std::unordered_map<sc_brgemm_postops_setting_t, expr> postop_set_cache_;
    std::vector<std::vector<stmt>> sc_kernel_cache_assignment_;
    std::vector<std::vector<stmt>> brg_bd_mask_arr_assignment_;
    // the kernel name => param_cache_table mapping
    std::unordered_map<std::string, param_cache_table> kernel_cache;
    typedef expr (*init_func_t)(brgemm_mode mode, scflags_t::brgemm_t backend,
            const std::vector<expr> &args, std::vector<expr_c> &cached_args,
            float beta, bool has_postop);
    using run_func_t = expr (*)(brgemm_mode, scflags_t::brgemm_t, const expr &,
            const std::vector<expr> &, bool has_postop);

    expr_c optimize_range_kernel_call(brgemm_mode mode,
            scflags_t::brgemm_t backend, expr_c v,
            const std::vector<expr> &args, const std::string &name,
            run_func_t run_func, const sc_brgemm_attrs_t &brg_attrs, float beta,
            bool has_postop, bool use_bdmask) {
        std::vector<expr_c> cached_args;
        if (backend != scflags_t::brgemm_t::dnnl || has_postop || use_bdmask) {
            SC_MODULE_INFO << "Cannot optimize the range kernel call: " << v;
            return v;
        }
        auto M_iter = brg_attrs.find(brgemm::attr_key::M_range_upper_bound);
        auto N_iter = brg_attrs.find(brgemm::attr_key::N_range_upper_bound);
        auto K_iter = brg_attrs.find(brgemm::attr_key::K_range_upper_bound);
        if ((M_iter == brg_attrs.end() && !args[brgemm_args::M].isa<constant>())
                || (N_iter == brg_attrs.end()
                        && !args[brgemm_args::N].isa<constant>())
                || (K_iter == brg_attrs.end()
                        && !args[brgemm_args::K].isa<constant>())) {
            SC_MODULE_INFO << "Cannot optimize the range kernel call: " << v;
            return v;
        }
        auto M_tail_iter = brg_attrs.find(brgemm::attr_key::M_range_tail_value);
        auto N_tail_iter = brg_attrs.find(brgemm::attr_key::N_range_tail_value);
        auto K_tail_iter = brg_attrs.find(brgemm::attr_key::K_range_tail_value);
        auto get_tail_value
                = [](const sc_brgemm_attrs_t &m,
                          const sc_brgemm_attrs_t::const_iterator &tail_it,
                          bool is_range) {
                      if (is_range) {
                          if (tail_it != m.end()) {
                              return static_cast<int>(tail_it->second);
                          }
                          return brg_range_tail_value::dyn_tail;
                      }
                      return brg_range_tail_value::no_tail;
                  };
        bool is_M_range = M_iter != brg_attrs.end();
        bool is_N_range = N_iter != brg_attrs.end();
        bool is_K_range = K_iter != brg_attrs.end();
        int M_tail_value = get_tail_value(brg_attrs, M_tail_iter, is_M_range);
        int N_tail_value = get_tail_value(brg_attrs, N_tail_iter, is_N_range);
        int K_tail_value = get_tail_value(brg_attrs, K_tail_iter, is_K_range);
        int M_upper_bound = static_cast<int>(is_M_range
                        ? M_iter->second
                        : get_expr_as_int(args[brgemm_args::M]));
        int N_upper_bound = static_cast<int>(is_N_range
                        ? N_iter->second
                        : get_expr_as_int(args[brgemm_args::N]));
        int K_upper_bound = static_cast<int>(is_K_range
                        ? K_iter->second
                        : get_expr_as_int(args[brgemm_args::K]));
        const int expected_cache_org_args = brgemm_args::NUM_BASIC_ARGS_STRIDE;
        const int expected_cache_extra_start = mode == brgemm_mode::stride
                ? brgemm_args::NUM_FULL_ARGS_STRIDE
                : brgemm_args::NUM_FULL_ARGS_LIST;
        const int expected_cache_extra_args
                = brgemm_args::extra_args_offset::cache_nargs;
        // start from lda
        if (!check_arg_range(args, cached_args, brgemm_args::LDA,
                    expected_cache_org_args)) {
            return v;
        }
        if (!check_arg_range(args, cached_args, expected_cache_extra_start,
                    expected_cache_extra_start + expected_cache_extra_args)) {
            return v;
        }
        int LDA = static_cast<int>(get_expr_as_int(args[brgemm_args::LDA]));
        int LDB = static_cast<int>(get_expr_as_int(args[brgemm_args::LDB]));
        int LDC = static_cast<int>(get_expr_as_int(args[brgemm_args::LDC]));
        int stride_a = static_cast<int>(
                get_expr_as_int(args[brgemm_args::STRIDE_A]));
        int stride_b = static_cast<int>(
                get_expr_as_int(args[brgemm_args::STRIDE_B]));
        int dtypeA = static_cast<int>(
                get_expr_as_int(args[expected_cache_extra_start
                        + brgemm_args::extra_args_offset::dtypeA]));
        int dtypeB = static_cast<int>(
                get_expr_as_int(args[expected_cache_extra_start
                        + brgemm_args::extra_args_offset::dtypeB]));
        void *attr_ptr = nullptr;
        auto attr_arg = args[expected_cache_extra_start
                + brgemm_args::extra_args_offset::brg_attrs];
        if (attr_arg.isa<tensor>()) {
            attr_arg = attr_arg.static_as<tensor>()->init_value_->data_;
        }
        auto handle = (mode == brgemm_mode::stride
                        ? std::make_shared<brg_range_handle_t>(M_upper_bound,
                                N_upper_bound, K_upper_bound, LDA, LDB, LDC,
                                stride_a, stride_b, beta, dtypeA, dtypeB,
                                attr_ptr, M_tail_value, N_tail_value,
                                K_tail_value)
                        : std::make_shared<brg_range_handle_t>(M_upper_bound,
                                N_upper_bound, K_upper_bound, LDA, LDB, LDC,
                                beta, dtypeA, dtypeB, attr_ptr, M_tail_value,
                                N_tail_value, K_tail_value));
        mod_->get_brg_range_handle_vec().emplace_back(handle);
        // Make kernel pointer global var. will be auto-renamed
        auto init = make_expr<constant_node>(
                reinterpret_cast<uintptr_t>(handle.get()), datatypes::pointer);
        auto handlev = mod_->make_global_var(datatypes::pointer,
                "__sc_range_kernel_cache", linkage::private_global, init);
        // todo: add cache for range handle.
        expr result = run_func(mode, backend, handlev, args, has_postop);
        assert(result.defined());
        return result;
    }

    expr_c optimize_kernel_call(brgemm_mode mode, scflags_t::brgemm_t backend,
            expr_c v, const std::vector<expr> &args, const std::string &name,
            init_func_t init_func, run_func_t run_func, float beta,
            bool has_postop, bool use_bdmask) {
        std::vector<expr_c> cached_args;

        if (use_bdmask) {
            int num_full_args = (mode == brgemm_mode::stride)
                    ? brgemm_args::NUM_FULL_ARGS_STRIDE
                    : brgemm_args::NUM_FULL_ARGS_LIST;
            int bd_mask_arg_offset
                    = num_full_args + +brgemm_args::extra_args_offset::bd_mask;
            expr bd_mask = args[bd_mask_arg_offset];
            expr bd_mask_idx = args[num_full_args - 1];
            assert(bd_mask.defined() && bd_mask_idx.defined());

            expr result = get_ir_null();
            std::vector<expr> brg_args = args;
            std::vector<expr> sc_kernel_cache;
            // cached_args contains the original bd_mask with new index
            std::vector<expr_c> cached_args_iter;

            // cached_args contains the bd_mask_arr
            expr cachev = init_func(
                    mode, backend, args, cached_args, beta, has_postop);
            if (!cachev.defined()) {
                SC_MODULE_INFO << "Cannot optimize the kernel call: " << v;
                return v;
            }

            // find the param_cache_table in the kernel cache
            auto first_itr = kernel_cache.find(name);
            if (first_itr != kernel_cache.end()) {
                auto &entry = first_itr->second;
                auto second_itr = entry.find(cached_args);
                if (second_itr != entry.end()) {
                    // if the same parameters are cached in the
                    // kernel_cache, reuse the cached kernel pointer
                    return run_func(mode, backend,
                            second_itr->second[bd_mask_idx], args, has_postop);
                }
            }

            expr sc_kernel_cache_arr = mod_->make_global_tensor(
                    datatypes::pointer, "__sc_kernel_cache_arr",
                    {brg_bdmask_set_num_}, linkage::private_global);
            std::vector<stmt> kernel_cache_assignment;
            for (int i = 0; i < brg_bdmask_set_num_; ++i) {
                brg_args[bd_mask_arg_offset] = bd_mask[i];
                expr cachev = init_func(mode, backend, brg_args,
                        cached_args_iter, beta, has_postop);

                if (!cachev.defined()) {
                    SC_MODULE_INFO << "Cannot optimize the kernel call: " << v;
                    return v;
                }
                kernel_cache_assignment.emplace_back(
                        builder::make_assign_unattached(
                                builder::make_indexing(
                                        sc_kernel_cache_arr, {i}),
                                cachev));
            }
            sc_kernel_cache_assignment_.push_back(kernel_cache_assignment);
            // put the var to the kernel_cache
            kernel_cache[name][cached_args] = sc_kernel_cache_arr;
            brg_args[bd_mask_arg_offset] = bd_mask[bd_mask_idx];
            result = run_func(mode, backend, sc_kernel_cache_arr[bd_mask_idx],
                    brg_args, has_postop);
            assert(result.defined());

            return result;
        } else {
            expr cachev = init_func(
                    mode, backend, args, cached_args, beta, has_postop);

            // check if the kernel lowerer agrees to optimize the kernel call
            if (!cachev.defined()) {
                SC_MODULE_INFO << "Cannot optimize the kernel call: " << v;
                return v;
            }
            // find the param_cache_table in the kernel cache
            auto first_itr = kernel_cache.find(name);
            if (first_itr != kernel_cache.end()) {
                auto &entry = first_itr->second;
                auto second_itr = entry.find(cached_args);
                if (second_itr != entry.end()) {
                    // if the same parameters are cached in the kernel_cache,
                    // reuse the cached kernel pointer
                    return run_func(mode, backend, second_itr->second, args,
                            has_postop);
                }
            }
            // Make kernel pointer global var. will be auto-renamed
            expr cache = mod_->make_global_var(cachev->dtype_,
                    "__sc_kernel_cache", linkage::private_global, cachev);
            // put the var to the kernel_cache
            kernel_cache[name][cached_args] = cache;
            expr result = run_func(mode, backend, cache, args, has_postop);
            assert(result.defined());
            return result;
        }
    }

    expr_c visit(intrin_call_c v) override {
        brgemm_args::extra_args_t *extras;
        sc_data_type_t dtypeA, dtypeB;
        brgemm_mode mode;
        v = ir_visitor_t::visit(std::move(v)).checked_as<intrin_call_c>();
        if (v->type_ == intrin_type::brgemm) {
            mode = brgemm_mode::stride;
        } else if (v->type_ == intrin_type::list_brgemm) {
            mode = brgemm_mode::addr_list;
        } else {
            return v;
        }

        extras = &v->intrin_attrs_->get<brgemm_args::extra_args_t>(
                intrin_attr::brgemm_extras);
        COMPILE_ASSERT(extras->is_cpu_, "Found non-CPU brgemm: " << v);
        dtypeA = extras->dtype_A_;
        dtypeB = extras->dtype_B_;
        sc_brgemm_attrs_t &brg_attrs = extras->brg_attrs_;
        sc_brgemm_bd_mask_t &bd_mask = extras->bd_mask_;
        brg_bdmask_set_num_ = extras->bd_mask_set_num_;
        sc_brgemm_postops_setting_t &brg_postops_setting
                = extras->postops_setting_;
        assert(mode == brgemm_mode::stride || mode == brgemm_mode::addr_list);
        int num_basic_args = (mode == brgemm_mode::stride)
                ? brgemm_args::NUM_BASIC_ARGS_STRIDE
                : brgemm_args::NUM_BASIC_ARGS_LIST;
        int num_full_args = (mode == brgemm_mode::stride)
                ? brgemm_args::NUM_FULL_ARGS_STRIDE
                : brgemm_args::NUM_FULL_ARGS_LIST;
        COMPILE_ASSERT(v->args_.size() == static_cast<size_t>(num_full_args),
                "invalid number of brgemm args, expected to be "
                        << num_full_args << ", but got " << v->args_.size()
                        << ".");

        // layout of v->args (full args):
        //    | basic_args | postops_data list(11 elems) | c_buf | bdmask_idx
        std::vector<expr> brg_postops_data
                = std::vector<expr>(v->args_.begin() + num_basic_args,
                        v->args_.begin() + num_full_args - 2);
        COMPILE_ASSERT(
                brg_postops_data.size() == brgemm::postops_data_init_func_nargs,
                "brg_postops_data.size() is expected to be "
                        << brgemm::postops_data_init_func_nargs << ", but got "
                        << brg_postops_data.size());
        bool use_bdmask = false;
        validate_brgemm_attrs(
                brg_attrs, bd_mask, brg_bdmask_set_num_, use_bdmask);
        brg_use_bdmask_ |= use_bdmask;

        // layout of opt_args:
        //    | basic_args | postops_data list(11 elems) | c_buf | bdmask_idx
        //    | dtypeA | dtypeB | attrs | bd_mask | postops setting
        //    | postops data | c_buf | stream |
        // The 1st c_buf will be replaced by 2nd c_buf
        std::vector<expr> opt_args {v->args_.begin(), v->args_.end()};
        opt_args.emplace_back(dtypeA.as_etype_int());
        opt_args.emplace_back(dtypeB.as_etype_int());
        // brgemm attrs
        bool try_range_optimize = false;
        expr brg_attrs_arg = get_brgemm_attrs_arg(
                mod_, brg_attrs, attrs_cache_, try_range_optimize);
        opt_args.emplace_back(brg_attrs_arg);
        // bd mask
        expr bd_mask_arr = get_ir_null(), cur_bd_mask = get_ir_null();
        if (use_bdmask) {
            expr bd_mask_idx = v->args_[num_full_args - 1];
            expr bd_mask_len = v->args_[brgemm_args::M];
            auto bd_mask_arg = get_brgemm_bd_mask_arg(mod_, bd_mask,
                    bd_mask_idx, bd_mask_len, brg_bdmask_set_num_,
                    cur_bd_mask_cache_, full_bd_mask_cache_,
                    brg_bd_mask_arr_assignment_);
            bd_mask_arr = bd_mask_arg.first;
            cur_bd_mask = bd_mask_arg.second;
        }
        opt_args.emplace_back(bd_mask_arr);
        // brgemm postops setting
        expr brg_setting_arg = get_brgemm_postops_setting_arg(
                mod_, brg_postops_setting, postop_set_cache_);
        opt_args.emplace_back(brg_setting_arg);
        // brgemm postops data
        std::vector<stmt_c> &ret = brg_postop_init_;
        expr brg_data_arg;
        if (!brg_postops_setting.empty()) {
            brg_data_arg = get_brgemm_postops_data_arg(ret, brg_postops_data);
        } else {
            auto ref = create_initialed_postops_data();
            for (size_t i = 0; i < brg_postops_data.size(); i++) {
                COMPILE_ASSERT(brg_postops_data[i]->equals(ref[i]),
                        "Postops data is not empty when setting is empty.");
            }
            brg_data_arg = get_ir_null();
        }
        opt_args.emplace_back(brg_data_arg);

        // brgemm c buf, currently we create a local buffer with M*N size
        expr brg_c_buf = v->args_[num_full_args - 2];
        assert(brg_c_buf.defined());
        if (brg_c_buf->equals(get_ir_null())) {
            if (!brg_postops_setting.empty()) {
                brg_c_buf = builder::make_tensor("__brgemm_c_buf",
                        {opt_args[brgemm_args::M] * opt_args[brgemm_args::N]},
                        infer_out_dtype(dtypeA));
                ret.emplace_back(
                        builder::make_var_tensor_def_unattached(brg_c_buf));
            }
        }
        opt_args.emplace_back(brg_c_buf);

        // placeholder for the context, required by brgemm with AMX.
        opt_args.emplace_back(get_ir_null());
        brg_postop_init_ = ret;

        // layout of no_opt_args:
        //    | basic_args | dtypeA | dtypeB | attrs | bd_mask | postops setting
        //    | postops data | c_buf | stream |
        std::vector<expr> no_opt_args(
                opt_args.begin(), opt_args.begin() + num_basic_args);
        // +2 for old c_buf and bdmask_idx
        no_opt_args.insert(no_opt_args.end(),
                opt_args.begin() + num_basic_args
                        + brgemm::postops_data_init_func_nargs + 2,
                opt_args.end());
        no_opt_args[num_basic_args + 3] = cur_bd_mask;

        bool optimized = optimize_ >= 1;
        scflags_t::brgemm_t backend = mod_->ctx_->flags_.brgemm_backend_;
        auto fpair = get_brgemm_update_funcs(mode, backend);
        func_t f = extras->cpu_.init_ ? fpair.second : fpair.first;
        assert(f);

        if (!optimized) {
            return builder::make_call(f, no_opt_args);
        } else {
            if (try_range_optimize) {
                auto ret = optimize_range_kernel_call(mode, backend, v,
                        opt_args, f->name_, range_brgemm_run, brg_attrs,
                        extras->cpu_.init_ ? 0.0f : 1.0f,
                        !brg_postops_setting.empty(), use_bdmask);
                if (!ret.ptr_same(v)) { return ret; }
            }
            // try general optimization again for bd mask and postop.
            auto ret = optimize_kernel_call(mode, backend, v, opt_args,
                    f->name_, brgemm_init_kernel_cache, brgemm_run,
                    extras->cpu_.init_ ? 0.0f : 1.0f,
                    !brg_postops_setting.empty(), use_bdmask);
            if (ret.ptr_same(v)) { return builder::make_call(f, no_opt_args); }
            return ret;
        }
    }

    stmt_c visit(stmts_c v) override {
        bool changed = false;
        std::vector<stmt_c> seq;
        for (auto &st : v->seq_) {
            brg_postop_init_.clear();
            auto new_st = dispatch(st);
            changed |= !new_st.ptr_same(st);
            if (new_st.isa<evaluate>()
                    && new_st.static_as<evaluate>()->value_.isa<call>()
                    && !brg_postop_init_.empty()) {
                seq.insert(seq.end(), brg_postop_init_.begin(),
                        brg_postop_init_.end());
                changed = true;
            }
            seq.emplace_back(new_st);
        }
        if (changed) {
            return copy_attr(*v, builder::make_stmts_unattached(seq));
        }
        return v;
    }

    kernel_lower_impl_t(ir_module_ptr mod, int optimize)
        : mod_(std::move(mod)), optimize_(optimize) {}
};

const_ir_module_ptr kernel_lowering_cpu_t::operator()(const_ir_module_ptr m) {
    auto ret = m->copy();
    kernel_lower_impl_t pass(ret, optimize_);
    auto old_gval_size = ret->get_module_vars().size();
    for (auto &f : ret->get_contents()) {
        f = std::const_pointer_cast<func_base>(pass.dispatch(f));
    }
    if (auto initf = ret->get_func("__sc_init__")) {
        for (size_t i = old_gval_size; i < ret->get_module_vars().size(); i++) {
            auto pvar = ret->get_module_vars()[i];
            // attrs/bdmask/set are tensors.
            if (pvar->var_.isa<var>()) {
                auto name = pvar->var_.static_as<var>()->name_;
                // only kernel cache needs init define.
                if (name.find("kernel_cache") != std::string::npos) {
                    initf->body_.checked_as<stmts>()->seq_.emplace_back(
                            builder::make_assign_unattached(
                                    pvar->var_, pvar->init_));
                }
            }
        }
    } else {
        initf = ret->make_init_func();
        if (initf) ret->add_func({initf});
    }

    if (pass.brg_use_bdmask_) {
        auto initf = ret->get_func("__sc_init__");
        if (!initf) {
            stmts seq = make_stmt<stmts_node_t>(std::vector<stmt>());
            initf = builder::make_func("__sc_init__", std::vector<expr_c>(),
                    std::move(seq), datatypes::void_t);
            ret->add_func({initf});
        }
        assert(initf && "__sc_init__ func is expected be presented in \
            the current ir module, but not.");

        for (auto &sts : pass.brg_bd_mask_arr_assignment_) {
            for (auto &st : sts) {
                initf->body_.checked_as<stmts>()->seq_.push_back(st);
            }
        }
        for (auto &sts : pass.sc_kernel_cache_assignment_) {
            for (auto &st : sts) {
                initf->body_.checked_as<stmts>()->seq_.push_back(st);
            }
        }
    }

    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
