/*******************************************************************************
 * Copyright 2020-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_HPP

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tensor_detail.hpp"
#include <compiler/config/context.hpp>
#include <compiler/ir/ir_module.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/utils.hpp>

#if SC_MEMORY_LEAK_CHECK > 0
#include <util/leak_detector.hpp>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace reflection {
template <typename T, typename Dummy>
struct type_registry;
struct shared_general_object_t;
} // namespace reflection

struct graph_tensor;
class sc_op;
using graph_tensor_ptr = std::shared_ptr<graph_tensor>;
using sc_op_ptr = std::shared_ptr<sc_op>;

// the additional data related to fusion manager, attached to logical_tensor_t
struct tensor_slice;
struct fusion_anchor_t;

template <typename valT>
struct gt_map_t;
using gt2gt_map = gt_map_t<graph_tensor_ptr>;
using gt2axis_map = gt_map_t<std::vector<int>>;
using gt2buf_map = gt_map_t<expr>;

using slice_range = std::vector<std::pair<expr, expr>>;
using slice_range_list = std::vector<slice_range>;
using fslice_map = gt_map_t<slice_range_list>;

using format_stride_pair = std::pair<sc_data_format_t, sc_dims>;
using shape_rl_vec = std::vector<std::pair<sc_dim, sc_dim>>;

struct dispatch_key_set_base_t;
using dispatch_set_ptr = std::shared_ptr<dispatch_key_set_base_t>;
struct dyn_internal_info_t;

/** VConst struct record possible varible in constant value, e.g.
 *
 *   const int a = k * b;
 *
 *  where `k` maybe variable related on other factor such as blocking dims.
 * */
struct VConst {
    // variable value
    int64_t var_;
};

// a weak pointer which always asserts the object exists
struct sc_op_weak_ptr_t : public std::weak_ptr<sc_op> {
    using parent = std::weak_ptr<sc_op>;
    using parent::parent;
    using parent::operator=;

    sc_op_weak_ptr_t(const sc_op_weak_ptr_t &) = default;
    sc_op_weak_ptr_t(sc_op_weak_ptr_t &&) = default;

    sc_op_weak_ptr_t &operator=(sc_op *);
    sc_op_weak_ptr_t &operator=(const sc_op_weak_ptr_t &other) {
        parent::operator=(other);
        return *this;
    }
    bool operator!() const { return get(); }
    bool operator==(const sc_op_weak_ptr_t &other) const {
        return get() == other.get();
    }
    bool operator==(const sc_op_ptr &other) const {
        return get() == other.get();
    }
    sc_op *operator*() const { return get(); }
    sc_op *operator->() const { return get(); }

    sc_op_ptr get_shared() const {
        auto ret = lock();
        assert(ret);
        return ret;
    }
    sc_op *get() const { return get_shared().get(); }
    sc_op_weak_ptr_t &operator=(const sc_op *other);
    operator sc_op_ptr() const { return get_shared(); }
};

// the logical tensor for in the graph to represent a result value in the graph.
// It contains the tensor details (shape, dtype, etc.) and the connectivity of
// the value in the graph
struct SC_API graph_tensor : public std::enable_shared_from_this<graph_tensor>
                             SC_EXTENDS_LEAK_CHECK(graph_tensor) {
    logical_tensor_t details_;
    // todo(zhichen/yijie): producer_owner should be used weak pointer.
    sc_op *producer_owner_ {nullptr};
    // the op nodes that use this tensor as input
    std::vector<std::pair<int, sc_op_weak_ptr_t>> uses_;

    any_map_t attrs_;

    graph_tensor(sc_op *owner);
    graph_tensor(sc_op *owner, const logical_tensor_t &lt);
    graph_tensor(sc_op *owner, const sc_data_format_t &format,
            const sc_dims &plain_shape, const sc_data_type_t &type,
            const sc_dims &stride = {});

    // adds a node to the `uses_` list, given the index of the tensor in the
    // inputs of the op
    void attach_use(sc_op_ptr op, int index);
    // removes a node from the `uses_` list (may need a linear search)
    void detach_use(const sc_op_ptr &op);
    // removes a node to the `uses_` list, given the index of the tensor in the
    // inputs of the op
    void detach_use(const sc_op_ptr &op, int input);

    // replaces all uses of this tensors with `v`. Will call `replace_input` of
    // all uses of this
    void replace_with(const graph_tensor_ptr &v);
    graph_tensor_ptr copy();
    ~graph_tensor() = default;

    static graph_tensor_ptr make(const sc_dims &shape,
            const sc_data_format_t &fmt = sc_data_format_t(),
            sc_data_type_t dtype = sc_data_type_t(sc_data_etype::F32, 1),
            const sc_dims &stride = {}) {
        return std::make_shared<graph_tensor>(
                nullptr, fmt, shape, dtype, stride);
    }
    bool is_dynamic() const { return details_.is_dynamic(); }
};

using ltensors = std::vector<logical_tensor_t>;

struct sc_op_info_t {
    std::vector<graph_tensor_ptr> outputs_;
    std::vector<graph_tensor_ptr> inputs_;
    // set of all dynamic dispatch keys combinations of op, this field is mainly
    // prepared for dynamic dispatch during lowering and is created during
    // layout propagation.
    dispatch_set_ptr dispatch_key_set_;
    // Extra info for op who could be internal queried.
    std::shared_ptr<dyn_internal_info_t> internal_info_;
    // current used impl type
    int cur_impl_ = 0;
};

struct op_base_trait_t {};

namespace op_attr_key {
// Boolean. If true, don't fuse this Op. Default = false
constexpr const char *no_fuse = "no_fuse";
// Boolean. If true, don't break the fusion partition after this Op
// (this Op can still be fused with previous Ops). Default = false
constexpr const char *break_post_fuse = "break_post_fuse";
// Boolean. If true, don't break the fusion partition before this Op
// (this Op can still be fused with post Ops). Default = false
constexpr const char *break_pre_fuse = "break_pre_fuse";
// Fuse Anchor
constexpr const char *inner_anchor = "inner_anchor";
// Inner anchor created in fusible ops and will be add to mixed partition when
// committing fusible op.
constexpr const char *fusible_inner_anchors = "fusible_inner_anchors";
// the name of the layer. Will be used to name the IR function
constexpr const char *layer_name = "temp.name";
// op marked with not_redundant will not be removed in horizontal same op
// elimination
constexpr const char *not_redundant = "temp.not_redundant";
// binary/ternary elementwise op layout propagation source input index.
constexpr const char *layout_input_index = "layout_input_index";
// Could use mask select to process output for reduce, matmul or other memory
// movement op.
constexpr const char *use_padded_mask = "use_padded_mask";
// Boolean. If true, it will skip graph pass div_bcast_transform. The precision
// requirements are high, and division must be used in the calculation of op.
constexpr const char *must_div = "must_div";
// Boolean. If true, the optimized formula will be used when norm calculates
// mean and var.
constexpr const char *use_norm_opt = "use_norm_opt";
}; // namespace op_attr_key

class sc_graph_t;
class SC_INTERNAL_API sc_op : public virtual op_base_trait_t,
                              public std::enable_shared_from_this<sc_op>
                              SC_LEAK_CHECK(sc_op) {
public:
    sc_graph_t *owner_graph_ {nullptr};
    sc_op_info_t info_;
    any_map_t attrs_;
    // the logical op ID in the op in the manager, default is 0.
    int logical_op_id_ = 0;
    // todo(zhichen): remove this, replace by dynamic cast check
    // is or not template op
    bool is_fusible_ = false;
    // align template name
    std::string op_name_;
    // if the node is removed from the manager
    bool is_removed_ = false;
    // get op infos
    const sc_op_info_t &get_info() const { return info_; }
    // get input logical tensors
    const std::vector<graph_tensor_ptr> &get_inputs() const {
        return info_.inputs_;
    }
    // get output logcial tesnors
    const std::vector<graph_tensor_ptr> &get_outputs() const {
        return info_.outputs_;
    }

    virtual const dispatch_set_ptr &get_dispatch_key_set() const;
    virtual dispatch_set_ptr &get_dispatch_key_set();
    // internal query disaptch keys, mainly for impl kind.
    virtual dispatch_set_ptr get_internal_dispatch_key_set(
            const context_ptr &ctx);
    // call impl_func if return true, create the internal info.
    bool need_dynamic_internal_query();
    // if the op needs internal query, default false.
    virtual bool need_dynamic_internal_query_impl() const { return false; }
    void copy_dispatch_key_set_from_op(const sc_op_ptr &other);
    /**
     * Repalces an input logical tensor
     * @param index the index within get_inputs()
     * @param new_input the new logical tensor
     * @param skip_shape_check whether to check new shape vs original shape, for
     * padding op the shape could be different
     * */
    void replace_input(size_t index, const graph_tensor_ptr &new_input,
            const bool skip_shape_check = false);

    // Replaces the current Op in the graph using another Op. All other Ops
    // using the output tensors of current Op will use the corresponding tensors
    // in the replacer Op instead. Finally the current node will be removed
    // Requires that the replacer has the same number of outputs of the current
    // node. Will detach from the input tensors. The replacer should manually
    // attach to the input tensors when it is needed
    void replace_uses_with_and_remove(const sc_op_ptr &replacer);

    /**
     * Checks whether any output tensor of this op is the output of the graph.
     */
    bool has_graph_output() const;

    // the op is single output and the output is single used
    bool is_single_output_single_use();

    // the op contains dynamic shape calculation, we didn't store a boolean
    // value cache for op/graph, because they are dynamic themselves.
    bool is_dynamic() const;

    // Get relationship of shapes from input/output plain shapes. Return
    // a vector of a pair of parent shape placeholder and child shape
    // placeholder. If one of the shapes corresponding to parent and child is
    // static, the other is also inferred as static, e.g. binary elemwise op
    // input shapes [-1, 64] and [16, 64], -1 will be inferred as 16.
    virtual shape_rl_vec get_dynamic_shape_relations() const { return {}; }
    // Get calculation expressions between different dynamic vars in
    // output/input. The expressions will be used by internal dynamic vars
    // inside kernel which does not infer shape. The expression will store in
    // var expr with attribute `pass.cal_expression`.
    virtual void calculate_dynamic_shape_expression() {}
    // the op share given graph tensor with opT(except itself)
    template <typename opT>
    bool share_gt_with_op(const graph_tensor_ptr &gt) {
        auto ths = this;
        return std::any_of(gt->uses_.begin(), gt->uses_.end(),
                [&ths](const std::pair<int, sc_op_weak_ptr_t> &user) {
                    return user.second->isa<opT>()
                            && (user.second.get() != ths);
                });
    }

    // Marks this node invalid and detach_use from all input tensors
    void remove();

    // get op's owner graph
    sc_graph_t &get_owner_graph() const { return *owner_graph_; }
    void set_owner_graph(sc_graph_t *owner_graph) {
        owner_graph_ = owner_graph;
    }
    // infer output details like plain shapes, most op infer in their
    // constructors except for conv now.
    virtual void infer_out_tensor_details() {}
    /**
     * Checks if the node is of a subclass.
     * @param T the subclass
     * */

    template <typename T>
    bool isa() const {
        static_assert(is_base_of_t<sc_op, T>::value
                        || is_base_of_t<op_base_trait_t, T>::value,
                "T is not a subclass of sc_op/op_trait.");
        return dynamic_cast<const T *>(this);
    }

    /**
     * Dynamic cast the node to a subclass or a trait.
     * @param T the subclass
     * */
    template <typename T>
    T *dyn_cast() const {
        return dynamic_cast<T *>(this);
    }

    template <typename T>
    T *dyn_cast() {
        return dynamic_cast<T *>(this);
    }

    /**
     * Static cast the node to a subclass or a trait.
     * @param T the subclass
     * */
    template <typename T>
    T *stc_cast() {
        assert(isa<T>() && "check T failed.");
        return static_cast<T *>(this);
    }

    /**
     * Compares the contents (op_name/attrs/other fields in the op). The default
     * implementation only compares op_name and attrs. This function does not
     * check the op-tensor connections, which will be checked by the
     * graph_comparer.
     * @param filter the filter for the attr name. If it returns false, the attr
     * will be ignored. By default it only filters out keys starting with
     * "temp."
     * @note we ingore the attrs with keys starting with "temp."
     * @return true if the contents (not including the op connections) are the
     * same
     * */
    virtual bool compare_contents(const sc_op *other,
            const std::function<bool(const sc_op *, const std::string &)>
                    &filter
            = nullptr) const;

    /**
     * Hash the contents. The default implementation only hashs op_name and
     * attrs. The function can be used to make hash map. When hash conflict
     * happened, we can compare them with `compare_contents`.
     * @param filter the filter for the attr name. If it returns false, the attr
     * will be ignored. By default it only filters out keys starting with
     * "temp."
     * @note we ingore the attrs with keys starting with "temp."
     * @return hash value with size_t datatype.
     * */
    virtual size_t hash_contents(
            const std::function<bool(const sc_op *, const std::string &)>
                    &filter
            = nullptr) const;

    // the default implementation of hash_contents
    static size_t standard_hash_contents(const sc_op *p,
            const std::function<bool(const sc_op *, const std::string &)>
                    &filter);

    // constructor
    sc_op(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs);
    sc_op() = default;
    sc_op(const sc_op &) = delete;
    sc_op(sc_op &&) = delete;

    virtual bool is_valid(const context_ptr &ctx) { return true; }
    virtual ir_module_ptr get_func(context_ptr ctx) = 0;
    virtual ir_module_ptr get_internal_func(const context_ptr &ctx) {
        throw std::runtime_error("Unimplement.");
    }

    virtual void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            = 0;
    void format_to_dense_format_stride_pair(
            const std::vector<std::vector<sc_data_format_t>> &in_formats,
            const std::vector<std::vector<sc_data_format_t>> &out_formats,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs);

    virtual ~sc_op() = default;
    virtual float get_gflop();
    // Get op impl type candidates for dynamic dispatch. Default candidates are
    // padding/no_padding. Return the impl alg candidates vector, element is
    // int(not enum) because different ops have different impl algs.
    virtual std::vector<int> get_impl_dispatch_candidates(
            const context_ptr &ctx);
    virtual reflection::shared_general_object_t get_dynamic_runtime_info();
};

inline sc_op_weak_ptr_t &sc_op_weak_ptr_t::operator=(sc_op *other) {
    *this = other->shared_from_this();
    return *this;
}

class SC_INTERNAL_API input_op;
class SC_INTERNAL_API output_op;

std::vector<graph_tensor_ptr> copy_logical_tsr(
        const std::vector<graph_tensor_ptr> &v);

struct dynamic_lower_info_t;
class SC_API sc_graph_t {
public:
    // the attr keys for graph
    struct attr_key_t {
        static constexpr const char *gflop = "gflop";
        static constexpr const char *quantize = "quantize";
        static constexpr const char *bf16 = "bf16";
        static constexpr const char *fp16 = "fp16";
        // if false, will keep the blocking format of output tensor
        static constexpr const char *is_output_plain = "is_output_plain";
        // if false, when an input_op is plain format and has only one use, will
        // set the input's data format to the expected format of the use
        static constexpr const char *is_input_plain = "is_input_plain";
        // if true, will allow output to be in channel last format
        static constexpr const char *allow_channel_last_output
                = "allow_channel_last_output";
        static constexpr const char *fpmath_mode = "fpmath_mode";
    };

    std::vector<sc_op_ptr> ops_;
    any_map_t attrs_;
    // lowering info related to dynamic
    std::shared_ptr<dynamic_lower_info_t> dyn_info_;

    // adds an existing node to the graph
    void add(const sc_op_ptr &node);

    // return true if the graph is empty
    bool empty() const { return ops_.empty(); }

    std::shared_ptr<sc_op> make(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &inputs,
            const std::vector<graph_tensor_ptr> &outputs,
            const any_map_t &attrs);

    template <typename T, typename... Args>
    std::shared_ptr<T> make(Args &&...args) {
        static_assert(!(std::is_same<T, input_op>::value),
                "output_op should go to specialized function");
        static_assert(!(std::is_same<T, output_op>::value),
                "output_op should go to specialized function");
        auto ret = std::make_shared<T>(std::forward<Args>(args)...);
        add(ret);
        return ret;
    }

    /**
     * Hash the contents. The default implementation only hashs ops and
     * attrs. The function can be used to make hash map. When hash conflict
     * happened, we can compare them with `compare_graph`.
     * @param filter the filter for the attr name. If it returns false, the attr
     * will be ignored. By default it only filters out keys starting with
     * "temp."
     * @note we ingore the attrs with keys starting with "temp."
     * @return hash value with size_t datatype.
     * */
    size_t hash_contents(
            const std::function<bool(const sc_op *, const std::string &)>
                    &filter
            = nullptr) const;

    // This function removes the Ops with is_removed_=true. And it compresses
    // the ops_ array by removing the holes of removed ops. It finally resets
    // all ids of each ops, to keep the continuity of ids.
    void reset_op_ids();
    // This function is aimed to re-sort the ops in the graph's `ops_` array
    // using the op-ids in the op_id_map
    void resort_op_ids(const std::unordered_map<sc_op_ptr, int> &op_id_map);

    // get next valid dynamic placeholder (decreasing) in graph.
    // We use [-2, -min_of_int64_t] to represent relationships between dynamic
    // shapes. It will be called by infer shapes(constructor of each op).
    // E.g. plain shape [-2, 64, -2, 128, -3, -4], first and third dimension are
    // the same dynamic var, and the last two are separated.
    sc_dim get_next_dynamic_placeholder();
    // return a expr (vector) from input dim (vector), if dim is dynamic, first
    // find in dim2expr_map_, if not found, create a var based on placeholder.
    expr dim_to_expr(const sc_dim &);
    std::vector<expr> dims_to_expr(const sc_dims &dim);
    std::vector<expr_c> dims_to_expr_c(const sc_dims &dim);

    // Get the total gflop from all tunable ops contained in the graph.
    float get_gflop() const;
    bool is_dynamic() const;
    bool is_non_dense() const;
    std::vector<sc_op_ptr> get_output_ops();
    std::vector<sc_op_ptr> get_input_ops();
    std::vector<sc_op_ptr> get_input_or_const_ops() const;

    // sync dynamic placeholder and dim2expr map with other graph. When you
    // create a subgraph b of graph a, you need to call
    // b.sync_dynamic_info_with_graph(a).
    void sync_dynamic_info_with_graph(const sc_graph_t &other) {
        dyn_info_ = other.dyn_info_;
    }
    // Get external dynamic vars existed in inputs/outputs.
    std::unordered_set<sc_dim> get_external_dynamic_vars();
    // Judge if the ops in graph need dynamic internal query
    bool need_dynamic_internal_query();
    // output op
    std::shared_ptr<sc_op> make_output(
            const std::vector<graph_tensor_ptr> &inputs,
            const any_map_t &attrs = any_map_t());
    // input op
    std::shared_ptr<sc_op> make_input(
            const std::vector<graph_tensor_ptr> &inputs,
            const any_map_t &attrs = any_map_t());
    sc_graph_t() = default;
    sc_graph_t(sc_graph_t &&other);
    sc_graph_t &operator=(sc_graph_t &&other);
    sc_graph_t(const sc_graph_t &other) = delete;
};

std::vector<expr> get_blocking_shapes_expr(sc_graph_t &graph,
        const sc_dims &plain_shapes, const sc_data_format_t &format);

using op_factory_func = sc_op_ptr (*)(const std::vector<graph_tensor_ptr> &,
        const std::vector<graph_tensor_ptr> &, const any_map_t &);
op_factory_func get_op_factory(const std::string &name);
void set_op_factory(const std::string &name, op_factory_func f);

template <typename T>
struct register_helper_t {
    static bool op_call(const std::string &op_name) {
        auto func = [](const std::vector<graph_tensor_ptr> &ins,
                            const std::vector<graph_tensor_ptr> &outs,
                            const any_map_t &attrs) -> sc_op_ptr {
            return std::shared_ptr<T>(new T(ins, outs, attrs));
        };
        set_op_factory(op_name, func);
        return false;
    }
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#define OP_REGISTER(IDENTIFIER, NAME) \
    volatile bool __help_dummy_##NAME \
            = dnnl::impl::graph::gc::register_helper_t<IDENTIFIER>::op_call( \
                    #NAME);
#endif
