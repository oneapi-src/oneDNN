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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_MGR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_MGR_HPP

#include <memory>
#include <utility>
#include <vector>
#include "brgemm_fusion.hpp"
#include "fusible_op.hpp"
#include "fusion_anchor.hpp"
#include "fusion_data.hpp"
#include <compiler/ir/sc_expr.hpp>
#include <runtime/microkernel/cpu/brgemm_common.hpp>
#include <unordered_map>
#include <util/utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
class fusible_op_t;
struct fuse_state_t;

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
    int max_anchor_ = -1;
    std::vector<fuse_anchor_t> fanchor_list_;
    // fusion manager will manager the sorted ops by rules
    std::vector<sc_op_ptr> sorted_ops_;
    // register for fusion inside brgemm.
    brgemm_fusion_register brg_fusion_reg_;
    // Get basic dfs topology sequence of graph op to initialize sorted_ops
    void init_sorted_ops();
    // prepare fusion data
    void do_prepare_fusion_data(fdata_map &fdmap);
    // pre-allocate tensors to logical tensors, NOTE that this stage has no
    // relation with fusion anchor.
    void do_allocate_tensor(fdata_map &fdmap,
            const std::vector<expr> &outs = {},
            const std::vector<expr> &inargs = {});
    // dispatch fusion anchor for each fusible op, it will return tbd_op_list
    std::vector<sc_op_ptr> dispatch_fusion_anchor(
            std::vector<fslice_map> &fsmap_list, const context_ptr &ctx);
    // infer the slice range for input args.
    void do_infer_slice_ranges(
            fslice_map &fsmap, int anchor_id, infer_status_map_t &stat_map);

    // commit compute_xxx_block
    void do_compute_block(const context_ptr &ctx, fuse_state_t &fstate);

    // define all the real tensors in suitable fusion anchor
    void do_declare_tensor(fuse_state_t &fstate);

    // reschedules the anchor, use_one_anchor can be treated as a tuning option
    void do_reshedule_anchor(
            std::vector<fslice_map> &fsmap_list, bool use_one_anchor = false);

    // allocates a temp tensor for the result of the op
    expr allocate_tensor(graph_tensor_ptr, fdata_map &);

    bool is_allocated_tensor(const tensor &tsr);

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
    // reset all status in brgemm fusion.
    void reset_brgemm_register_infos();

    int get_input_idx(sc_op *) const;
    int get_output_idx(sc_op *) const;

    template <typename T, typename... Args>
    std::shared_ptr<T> make(Args &&...args) {
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
            const std::vector<graph_tensor_ptr> &vec, Args &&...args) {
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
    std::vector<sc_op_ptr> prepare_and_check(
            const context_ptr &ctx, fuse_state_t &fstate);

    // the graph based fusion commit api
    void commit(const ir_module_ptr &modu, fuse_state_t &fstate,
            const std::vector<expr> &outs = {},
            const std::vector<expr> &inargs = {});

    /**
     * create_output_fusion_anchor does not really generate code here at once,
     * it is just used to create an anchor to record the position which wil be
     * avaliable for fusion manager to commit real generated code there.
     * */
    void create_output_fusion_anchor(const std::vector<tensor_slice> &src,
            const std::vector<tensor_slice> &dst = {});

    std::vector<fuse_anchor_t> input_anchor_list_;
    void create_input_fusion_anchor(const std::vector<tensor_slice> &dst,
            const std::vector<tensor_slice> &src = {});

    // iter anchor
    std::vector<iter_fuse_anchor_t> iter_anchor_list_;
    // create iter anchor
    void create_output_fusion_anchor(expr iter, expr tsr,
            slice_range_list slice_list, stmt dispatch_helper = stmt());

    // grouped anchor
    std::unordered_map<int, grouped_fuse_anchor_t> grouped_anchor_map_;
    // create grouped anchor
    void create_output_fusion_anchor(
            const std::vector<tensor_slice> &src, int group_id);

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
    void bind_graph(sc_graph_t *graph);
    fusion_manager &operator=(const fusion_manager &other) = delete;
    // get real tensors list vector in input logical tensors
    std::vector<std::vector<tensor_slice>> get_input_tsr_slices_list(
            fusible_op_t *op, fdata_map &fdmap, fslice_map &fsmap) const;
    // get real tensors list vector in output logical tensors
    std::vector<std::vector<tensor_slice>> get_output_tsr_slices_list(
            fusible_op_t *op, fdata_map &fdmap, fslice_map &fsmap) const;

    // copy fusion mgr, return shared_ptr of fusion_mgr
    std::shared_ptr<fusion_manager> copy() const;

    std::vector<std::pair<stmts, std::unordered_map<expr, slice_range_list>>>
    unpack_src_anchor();
    std::vector<std::pair<stmts, std::unordered_map<expr, slice_range_list>>>
    unpack_dst_anchor();

    /**
     * transform the graph if necessary before really committing code
     * @param has_main_op if the fusion happens after another main op
     * */
    void transform_graph(const context_ptr &ctx, bool has_main_op);
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
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
