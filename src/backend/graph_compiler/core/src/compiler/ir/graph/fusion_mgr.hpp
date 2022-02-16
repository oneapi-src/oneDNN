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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_MGR_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_MGR_HPP

#include <memory>
#include <utility>
#include <vector>
#include "brgemm_fusion.hpp"
#include "fusible_op.hpp"
#include <microkernel/cpu/brgemm_common.hpp>
#include <unordered_map>
#include <util/utils.hpp>
namespace sc {
// fusion mgr data is the set of internal data of fusion manager
struct fusion_anchor_data;
class fusible_op_t;
class fusion_manager {
protected:
    // input tensor index, increases 1 when a input_op constructs
    int input_op_count_ = 0;
    // output tensor index, increases 1 when a output_op constructs
    int output_op_count_ = 0;
    // alloc tensor index, increases 1 when a allocated_tensor constructs
    int alloc_tensor_count_ = 0;
    // the contained graph
    sc_graph_t graph_;
    // the temp tensors that this manager allocates
    std::vector<tensor> allocated_tensors_;
    // the global var/tensor definations that constant ops hold
    std::vector<define> global_defines_;
    // input op => the index of tensor slice in src of commit(...)
    std::unordered_map<sc_op *, int> input_idx_map_;
    // input op => the index of tensor slice in src of commit(...)
    std::unordered_map<sc_op *, int> output_idx_map_;
    // the maximum anchor id fusion manager use
    int max_anchor_;
    // the list record output anchor position, which will be inserted real
    // generated code commited by fusion.manager.
    // TODO(xxx): extend to the vector
    std::vector<stmts> output_anchor_position_;
    // the list record output anchor slice, which will be used by
    // compute_xxx_block of each fusible op.
    std::vector<std::pair<std::vector<tensor_slice>, std::vector<tensor_slice>>>
            output_anchor_slice_;
    // fusion manager will manager the sorted ops by rules
    std::vector<sc_op_ptr> sorted_ops_;
    // register for fusion inside brgemm.
    brgemm_fusion_register brg_fusion_reg_;
    // Get basic dfs topology sequence of graph op to initialize sorted_ops
    void init_sorted_ops();
    // calls prepare_fusion_data() on every op
    void do_query(const context_ptr &ctx,
            std::vector<fusion_anchor_data> &fdata_list, bool legacy_mode,
            int anchor_id = 0);
    // prepare fusion data
    void do_prepare_fusion_data(const context_ptr &ctx,
            std::vector<fusion_anchor_data> &fdata_list, int anchor_id = 0);
    // infer the slice range for input args.
    void do_infer_slice_ranges(std::vector<fusion_anchor_data> &fdata_list,
            std::vector<sc_op_ptr> &failed_ops, int anchor_id = 0);
    // pre-allocate tensors to logical tensors
    void do_allocate_tensor(std::vector<fusion_anchor_data> &fdata_list,
            int anchor_id = 0, const std::vector<expr> &outs = {},
            const std::vector<expr> &inargs = {});
    // schedules the real tensors to logical tensors
    void do_shedule_tensor(std::vector<fusion_anchor_data> &fdata_list);
    // define all the real tensors in suitable fusion anchor
    void do_declare_tensor(std::vector<fusion_anchor_data> &fdata_list);
    // commit compute_xxx_block
    void do_compute_block(const context_ptr &ctx,
            std::vector<fusion_anchor_data> &fdata_list);
    // dispatch fusion anchor for each fusible op, it will return tbd_op_list
    std::vector<sc_op_ptr> dispatch_fusion_anchor(const context_ptr &ctx,
            std::vector<fusion_anchor_data> &fdata_list,
            const std::vector<expr> &outs = {},
            const std::vector<expr> &inargs = {});
    // reschedules the anchor, use_one_anchor can be treated as a tuning option
    void do_reshedule_anchor(std::vector<fusion_anchor_data> &fdata_list,
            bool use_one_anchor = false);

    // allocates a temp tensor for the result of the op
    void allocate_tensor(graph_tensor_ptr, fusion_anchor_data &fdata);

public:
    int get_input_op_count() const { return input_op_count_; }
    int get_output_op_count() const { return output_op_count_; }
    const sc_graph_t &get_graph() const { return graph_; }
    sc_graph_t &get_graph() { return graph_; }
    const brgemm_fusion_register &get_brgemm_fusion_register() const {
        return brg_fusion_reg_;
    }
    void break_brgemm_fusion();
    bool can_register_brgemm_fusion(const stmt &body);

    template <typename T, typename... Args>
    std::shared_ptr<T> make(Args &&... args) {
        static_assert(!(std::is_same<T, input_op>::value),
                "input_op should go to specialized function");
        static_assert(!(std::is_same<T, output_op>::value),
                "output_op should go to specialized function");
        return graph_.make<T>(std::forward<Args>(args)...);
    }

    sc_op_ptr make_input(const std::vector<graph_tensor_ptr> &in);

    void add(fusion_op_ptr node);

    // todo: remove and use standard graph::make
    template <typename T, typename... Args>
    std::shared_ptr<T> make(
            const std::vector<graph_tensor_ptr> &vec, Args &&... args) {
        auto ret = std::make_shared<T>(vec, std::forward<Args>(args)...);
        graph_.add(ret);
        return ret;
    }

    /* Function below is all of the fusion interface for graph based */
    /** this function will query the output of fusion pattern can be inplaced by
     *  which inputs of fusion pattern.
     * @return for each output, it will return a vector, which consists of all
     * input_op idx those can be replaced.
     * */
    std::vector<std::vector<int>> query_inplace();

    /**
     * Prepares fusion infos and check if given anchors can meet demands of all
     *  ops
     * @param outs: the final fusible output buffer given by users
     * @param inrags: the addtional input args buffer given by users
     * @return if checking passes, returns empty vector. Otherwise, returns the
     * ops that cause the failure
     * */
    std::vector<sc_op_ptr> prepare_and_check(const context_ptr &ctx,
            std::vector<fusion_anchor_data> &fdata_list,
            const std::vector<expr> &outs = {},
            const std::vector<expr> &inargs = {});

    // the graph based fusion commit api
    void commit(const ir_module_ptr &modu,
            std::vector<fusion_anchor_data> &fdata_list);

    /**
     * create_output_fusion_anchor does not really generate code here at once,
     * it is just used to create an anchor to record the position which wil be
     * avaliable for fusion manager to commit real generated code there.
     * */
    void create_output_fusion_anchor(const std::vector<tensor_slice> &src,
            const std::vector<tensor_slice> &dst = {});

    // clear anchor and reset their status.
    void clear_anchor();

    void add_to_module(const ir_module_ptr &mod);

    // this function will put given inp at first input_op in fusion manager,
    // usunally used when we need to reset base input op in some cases.
    void put_input_first(input_op *inp);
    // this function will return input op which own first input idx. (also known
    // as base op in most cases)
    const sc_op *get_first_input() const;

    fusion_manager() = default;
    fusion_manager(const fusion_manager &other) = delete;
    fusion_manager(fusion_manager &&other);
    fusion_manager &operator=(const fusion_manager &other) = delete;
    // get real tensors list vector in input logical tensors
    std::vector<std::vector<const tensor_slice *>> get_input_tsr_slices_list(
            fusible_op_t *op, fusion_anchor_data &fdata) const;
    // get real tensors list vector in output logical tensors
    std::vector<std::vector<tensor_slice *>> get_output_tsr_slices_list(
            fusible_op_t *op, fusion_anchor_data &fdata) const;
    // get input op index
    int get_input_op_index(sc_op *op) const;
};

/** fusion_anchor_data vs fusion_data
 *  1. fusion_anchor_data: is from the view of fusion manager, which can manager
 * some overall information fusion_data in the *certain fusion anchor*.
 *  2. fusion_data: focus on those infos related to fusion,  especially
 * tensor_slice
 * */

// fusion_data is related to tensor slice of fusible op
struct fusion_data {
    // the number of uses of the tensor in the graph. TODO: Should be replaced
    // by uses_.size()
    int use_count_ = 0;
    /**
     * @TODO: remove shape, because it can not represent multi-slice semantics,
     * we can use original_ranges_list_ to replace shape_
     * */
    // the tensor slice shape
    std::vector<expr> shape_;
    // original_ranges_list_ means the slice range for the original whole
    // buffer, ignoring any writer_cache or temp_buffer
    std::vector<std::vector<std::pair<expr, expr>>> original_ranges_list_;
    bool need_alloc_ = true;
    // judge whether the tsr_slice is contiguous, if tsr_slice_list have
    // more than two size, return false.
    bool is_contiguous();
    fusion_data() = default;
    fusion_data(const fusion_data &) = delete;

private:
    // deal with multi-slice
    std::vector<tensor_slice> tsr_slice_list_;
    friend class fusion_manager;
    friend void set_buffer_reuse_hint(int64_t &, fusion_anchor_data &,
            const sc_op_ptr &, const expr &, bool);
};

// fusion_anchor_data is related to fusion manager in the certain fusion anchor
struct fusion_anchor_data {
    std::unordered_map<graph_tensor *, fusion_data> datamap_;
    // input op => the index of tensor slice in src of commit(...)
    std::unordered_map<sc_op *, int> *input_idx_map_ = nullptr;
    // input op => the index of tensor slice in src of commit(...)
    std::unordered_map<sc_op *, int> *output_idx_map_ = nullptr;
    // the number of src tensor slices when fusion_manager::commit is called
    int num_commit_src_ = 0;
    fusion_data &get(graph_tensor *);
    fusion_data &get(const graph_tensor_ptr &);
    int get_input_idx(sc_op *) const;
    int get_output_idx(sc_op *) const;
    // judge input_op whether is the additional arg
    bool is_arg_input(input_op *op) const {
        return get_input_idx(op) >= num_commit_src_;
    }
};

// todo: remove and use standard graph::make
template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>();
template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(
        sc_dims &&dims, const sc_data_type_t &dtype);
template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(
        const logical_tensor_t &lt);

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(logical_tensor_t &lt);

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(logical_tensor_t &&lt);

template <>
std::shared_ptr<output_op> fusion_manager::make<output_op>(
        const graph_tensor_ptr &arg);

using fusion_mgr_ptr = std::shared_ptr<fusion_manager>;

} // namespace sc

#endif
