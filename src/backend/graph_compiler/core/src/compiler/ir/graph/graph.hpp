/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tensor_detail.hpp"
#include <compiler/config/context.hpp>
#include <compiler/ir/ir_module.hpp>
#include <unordered_map>
#include <util/any_map.hpp>
#include <util/utils.hpp>

#if SC_MEMORY_LEAK_CHECK > 0
#include <util/leak_detector.hpp>
#endif

namespace sc {

namespace reflection {
template <typename T, typename Dummy>
struct type_registry;
}

struct graph_tensor;
class sc_op;
using graph_tensor_ptr = std::shared_ptr<graph_tensor>;
using sc_op_ptr = std::shared_ptr<sc_op>;

// the additional data related to fusion manager, attached to logical_tensor_t
struct tensor_slice;
struct fusion_data_t;
struct fuse_state_t;

template <typename valT>
struct gt_map_t;
using fdata_map = gt_map_t<fusion_data_t>;
using gt2gt_map = gt_map_t<graph_tensor_ptr>;
using gt2axes_map = gt_map_t<std::vector<int>>;
using format_stride_pair = std::pair<sc_data_format_t, sc_dims>;

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
struct SC_API graph_tensor SC_EXTENDS_LEAK_CHECK(graph_tensor) {
    logical_tensor_t details_;
    // todo(zhichen/yijie): producer_owner should be used weak pointer.
    sc_op *producer_owner_ {nullptr};
    // the op nodes that use this tensor as input
    std::vector<std::pair<int, sc_op_weak_ptr_t>> uses_;

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
};

using ltensors = std::vector<logical_tensor_t>;
struct sc_op_info_t {
    std::vector<graph_tensor_ptr> outputs_;
    std::vector<graph_tensor_ptr> inputs_;
    // todo: move the 2 fields below to fusion data
    /* the map of <output index, input index vector> decribes the sharing
     relationship between input and output tensors */
    std::unordered_map<int, std::vector<int>> tensor_share_info_;
};

struct op_base_trait_t {};

namespace op_attr_key {
// const char, represents the mode of fusion, including total three modes as
// below
constexpr const char *fused_mode_hint = "fused_mode_hint";
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
// Batchwise fused
constexpr const char *bwise_fuse = "bwise_fuse";
constexpr const char *bwise_no_fuse = "bwise_no_fuse";
// `bwise_skip_fuse` differs with `bwise_no_fuse` in that it is often used as
// temporarily status check
constexpr const char *bwise_skip_fuse = "bwise_skip_fuse";
constexpr const char *bwise_break_pre_fuse = "bwise_break_pre_fuse";
constexpr const char *bwise_break_post_fuse = "bwise_break_post_fuse";
constexpr const char *bwise_no_strided_dims = "bwise_no_strided_dims";
// the name of the layer. Will be used to name the IR function
constexpr const char *layer_name = "temp.name";
}; // namespace op_attr_key

class SC_INTERNAL_API sc_op : public virtual op_base_trait_t,
                              public std::enable_shared_from_this<sc_op>
                              SC_LEAK_CHECK(sc_op) {
public:
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

    /**
     * Repalces an input logical tensor
     * @param index the index within get_inputs()
     * @param new_input the new logical tensor
     * */
    void replace_input(size_t index, const graph_tensor_ptr &new_input);

    // Replaces the current Op in the graph using another Op. All other Ops
    // using the output tensors of current Op will use the corresponding tensors
    // in the replacer Op instead. Finally the current node will be removed
    // Requires that the replacer has the same number of outputs of the current
    // node. Will detach from the input tensors. The replacer should manually
    // attach to the input tensors when it is needed
    void replace_uses_with_and_remove(const sc_op_ptr &replacer);

    // the op is single output and the output is single used
    bool is_single_output_single_use();

    // the op share given graph tensor with opT(except itself)
    template <typename opT>
    bool share_gt_with_op(const graph_tensor_ptr &gt) {
        auto ths = this;
        return std::any_of(gt->uses_.begin(), gt->uses_.end(),
                [&ths](const std::pair<int, sc::sc_op_weak_ptr_t> &user) {
                    return user.second->isa<opT>()
                            && (user.second.get() != ths);
                });
    }

    // Marks this node invalid and detach_use from all input tensors
    void remove();

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
     * @note we ingore the attrs with keys starting with "temp."
     * @return true if the contents (not including the op connections) are the
     * same
     * */
    virtual bool compare_contents(const sc_op *other) const;

    /**
     * Hash the contents. The default implementation only hashs op_name and
     * attrs. The function can be used to make hash map. When hash conflict
     * happened, we can compare them with `compare_contents`.
     * @note we ingore the attrs with keys starting with "temp."
     * @return hash value with size_t datatype.
     * */
    virtual size_t hash_contents() const;

    // constructor
    sc_op(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs);
    sc_op() = default;
    virtual bool is_valid(const context_ptr &ctx) { return true; }
    virtual ir_module_ptr get_func(context_ptr ctx) = 0;

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
};

inline sc_op_weak_ptr_t &sc_op_weak_ptr_t::operator=(sc_op *other) {
    *this = other->shared_from_this();
    return *this;
}

class SC_INTERNAL_API input_op;
class SC_INTERNAL_API output_op;

std::vector<graph_tensor_ptr> copy_logical_tsr(
        const std::vector<graph_tensor_ptr> &v);

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
    };

    std::vector<sc_op_ptr> ops_;
    any_map_t attrs_;

    // adds an existing node to the graph
    void add(const sc_op_ptr &node);

    // return true if the graph is empty
    bool empty() const { return ops_.empty(); }

    std::shared_ptr<sc_op> make(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &inputs,
            const std::vector<graph_tensor_ptr> &outputs,
            const any_map_t &attrs);

    template <typename T, typename... Args>
    std::shared_ptr<T> make(Args &&... args) {
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
     * @note we ingore the attrs with keys starting with "temp."
     * @return hash value with size_t datatype.
     * */
    size_t hash_contents() const;

    // This function removes the Ops with is_removed_=true. And it compresses
    // the ops_ array by removing the holes of removed ops. It finally resets
    // all ids of each ops, to keep the continuity of ids.
    void reset_op_ids();
    // This function is aimed to re-sort the ops in the graph's `ops_` array
    // using the op-ids in the op_id_map
    void resort_op_ids(const std::unordered_map<sc_op_ptr, int> &op_id_map);
    // Get the total gflop from all tunable ops contained in the graph.
    float get_gflop() const;
    std::vector<sc_op_ptr> get_output_ops();
    std::vector<sc_op_ptr> get_input_ops();
    std::vector<sc_op_ptr> get_input_or_const_ops() const;
    // output op
    std::shared_ptr<sc_op> make_output(
            const std::vector<graph_tensor_ptr> &inputs,
            const any_map_t &attrs = any_map_t());
    // input op
    std::shared_ptr<sc_op> make_input(
            const std::vector<graph_tensor_ptr> &inputs,
            const any_map_t &attrs = any_map_t());
    sc_graph_t() = default;
    sc_graph_t(sc_graph_t &&other) = default;
    sc_graph_t &operator=(sc_graph_t &&other) = default;
    sc_graph_t(const sc_graph_t &other) = delete;
};

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
} // namespace sc
#define OP_REGISTER(IDENTIFIER, NAME) \
    volatile bool __help_dummy_##NAME \
            = sc::register_helper_t<IDENTIFIER>::op_call(#NAME);
#endif
