/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GRAPH_EXAMPLE_COMPAT_COMPAT_HELPERS_HPP
#define GRAPH_EXAMPLE_COMPAT_COMPAT_HELPERS_HPP

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "../graph_example_utils.hpp"

namespace compat {

// mimic cudnnHandle_t
class Handle {
    std::shared_ptr<dnnl::engine> eng_;
    std::shared_ptr<dnnl::stream> str_;

public:
    Handle(dnnl::engine::kind ekind, int index) {
        eng_ = std::make_shared<dnnl::engine>(ekind, index);
        str_ = std::make_shared<dnnl::stream>(*eng_);
    }

    dnnl::engine::kind get_engine_kind() const { return eng_->get_kind(); }

    dnnl::engine *get_engine() { return eng_.get(); }
    dnnl::stream *get_stream() { return str_.get(); }

    void synchronize() { str_->wait(); }
};

inline void fill_random(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

// Surface is not part fo cuDNN API, but a utility for the example. Here we use
// it to allocate and initialize user data. We rely on the memory allocation in
// `dnnl::graph::tensor` inside the library, to avoid the explicit code for
// cpu/ocl/sycl memory allocation.
struct Surface {
    using lt = dnnl::graph::logical_tensor;

    int64_t n_elems_ = 0;
    Handle *handle_ = NULL;
    lt::data_type dt_ = lt::data_type::undef;
    dnnl::graph::tensor ts_;

protected:
    explicit Surface() {}

public:
    Surface(lt::data_type dt, int64_t n_elems, Handle *handle)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        std::vector<float> raw_data(n_elems);
        fill_random(raw_data);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        // write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    Surface(lt::data_type dt, int64_t n_elems, Handle *handle, float val)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        std::vector<float> raw_data(n_elems, val);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        // write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    void *get_ptr() const { return ts_.get_data_handle(); }

    ~Surface() = default;
};

} // namespace compat

namespace compat_0_x {

int64_t op_id = 0;
using lt = dnnl::graph::logical_tensor;
using op = dnnl::graph::op;
using Operation = dnnl::graph::op;
using DataType_t = logical_tensor::data_type;
//using Tensor = dnnl::graph::logical_tensor;

// mimic cudnnHandle_t
class Handle {
    std::shared_ptr<dnnl::engine> eng_;
    std::shared_ptr<dnnl::stream> str_;

public:
    Handle(dnnl::engine::kind ekind, int index) {
        eng_ = std::make_shared<dnnl::engine>(ekind, index);
        str_ = std::make_shared<dnnl::stream>(*eng_);
    }

    dnnl::engine::kind get_engine_kind() const { return eng_->get_kind(); }

    dnnl::engine *get_engine() { return eng_.get(); }
    dnnl::stream *get_stream() { return str_.get(); }

    void synchronize() { str_->wait(); }
};

inline void fill_random(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

// Surface is not part fo cuDNN API, but a utility for the example. Here we use
// it to allocate and initialize user data. We rely on the memory allocation in
// `dnnl::graph::tensor` inside the library, to avoid the explicit code for
// cpu/ocl/sycl memory allocation.
struct Surface {
    using lt = dnnl::graph::logical_tensor;

    int64_t n_elems_ = 0;
    Handle *handle_ = NULL;
    lt::data_type dt_ = lt::data_type::undef;
    dnnl::graph::tensor ts_;

protected:
    explicit Surface() {}

public:
    Surface(lt::data_type dt, int64_t n_elems, Handle *handle)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        std::vector<float> raw_data(n_elems);
        fill_random(raw_data);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        // write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    Surface(lt::data_type dt, int64_t n_elems, Handle *handle, float val)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        std::vector<float> raw_data(n_elems, val);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        // write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    void *get_ptr() const { return ts_.get_data_handle(); }

    ~Surface() = default;
};

///

class Tensor_v8 {
public:
    friend class TensorBuilder_v8;
    //     std::string describe() const override {
    //         std::stringstream ss;
    // #ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    //         ss << "CUDNN_BACKEND_TENSOR_DESCRIPTOR :"
    //            << " Datatype: " << json {data_type} << " Id: " << std::to_string(id)
    // #else
    //         ss << "CUDNN_BACKEND_TENSOR_DESCRIPTOR :"
    //            << " Datatype: " << int(data_type) << " Id: " << std::to_string(id)
    // #endif
    //            << " nDims " << nDims << " VectorCount: " << vectorCount
    //            << " vectorDimension " << vectorDimension;
    //         ss << " Dim [ ";
    //         for (auto i = 0; i < nDims; i++) {
    //             if (i != 0) { ss << ','; }
    //             ss << btensor_dimA[i];
    //         }
    //         ss << " ] Str [ ";
    //         for (auto i = 0; i < nDims; i++) {
    //             if (i != 0) { ss << ','; }
    //             ss << btensor_strA[i];
    //         }
    //         ss << " ]";
    //         ss << " isVirtual: " << isVirtual << " isByValue: " << isByValue
    //            << " Alignment: " << alignment;
    // #ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    //         ss << " reorder_type: " << json {reorder_type};
    // #else
    //         ss << " reorder_type: " << int(reorder_type);
    // #endif
    //         return ss.str();
    //     }

    int64_t getDimCount() const { return nDims; }

    lt::dim const *getDim() const { return btensor_dimA; }

    lt::dim const *getStride() const { return btensor_strA; }

    lt::data_type getDataType() const {
        return static_cast<int64_t>(data_type);
    }

    int64_t getId() const { return id; }

    int64_t getAlignment() const { return alignment; }

    bool isVirtualTensor() const { return isVirtual; }

    lt const *getInternal_lt() const { return internal_lt; }

    Tensor_v8(Tensor_v8 &&from) = default;
    Tensor_v8 &operator=(Tensor_v8 &&) = default;

    ~Tensor_v8() = default;

private:
    Tensor_v8() = default;
    Tensor_v8(Tensor_v8 const &) = delete;
    Tensor_v8 &operator=(Tensor_v8 const &) = delete;

    DataType_t data_type = DataType_t::undef; //! Datatype of the elements
    int64_t btensor_dimA[CUDNN_DIM_MAX + 1] = {-1}; //! n, g, c, d, h, w
    int64_t btensor_strA[CUDNN_DIM_MAX + 1] = {-1}; //! n, g, c, d, h, w
    int64_t id = -1; //! Unique id of the tensor
    int64_t alignment = -1; //! Alignment of the tensor.
    //! Certain engine config expect minimum alignment of 16B
    int64_t nDims = -1; //! Number of Dimensions of the tensor
    bool isVirtual
            = false; //! Whether it is an intermediate tensor of an op graph
    bool isByValue
            = false; //! Whether the tensor is in host memory that needs to be passed to the kernel by value
    lt *internal_lt;
};

/// TensorBuilder_v8 Class
/// Helper class used to build Tensor_v8 class
class TensorBuilder_v8 {
public:
    using lt = dnnl::graph::logical_tensor;
    /** @defgroup TensorBuilder_v8
     *  Set individual property of Tensor_v8 class
     *  @{
     */
    //! Set Datatype for the Tensor_v8
    auto setDataType(lt::data_type data_type) -> TensorBuilder_v8 & {
        m_tensor.data_type = data_type;
        return *this;
    }

    //! Set Dimensions of the tensor
    auto setDim(int64_t ndim, lt::dim const *dim) -> TensorBuilder_v8 & {
        std::copy((dim), dim + ndim, m_tensor.btensor_dimA);
        m_tensor.nDims = ndim;
        return *this;
    }
    //! Set Strides of the tensor
    auto setStride(int64_t ndim, lt::dim const *strides) -> TensorBuilder_v8 & {
        std::copy(strides, strides + ndim, m_tensor.btensor_strA);
        return *this;
    }
    //! Set Unique Id  of the tensor
    auto setId(int64_t id_) -> TensorBuilder_v8 & {
        m_tensor.id = id_;
        return *this;
    }
    //! Set Alignment of the tensor
    auto setAlignment(int64_t alignment_) -> TensorBuilder_v8 & {
        m_tensor.alignment = alignment_;
        return *this;
    }
    //! Set isVirtual of the tensor
    auto setVirtual(bool virtual_ = true) -> TensorBuilder_v8 & {
        m_tensor.isVirtual = virtual_;
        return *this;
    }
    //! Set isByValue of the tensor
    auto setByValue(bool isByValue_ = true) -> TensorBuilder_v8 & {
        m_tensor.isByValue = isByValue_;
        return *this;
    }

    //! constructs the Tensor_v8 by calling the cudnn API
    //! Throws the appropriate error message
    Tensor_v8 &&build() {
        auto lt_tmp = dnnl::graph::logical_tensor(m_tensor.getId(),
                m_tensor.getDataType(), m_tensor.getDim(),
                m_tensor.getStride());
        m_tensor.internal_lt = &lt_tmp;
        return std::move(m_tensor);
    }

    explicit TensorBuilder_v8() = default;
    ~TensorBuilder_v8() = default;
    TensorBuilder_v8(TensorBuilder_v8 &&) = delete;
    TensorBuilder_v8(TensorBuilder_v8 const &) = delete;
    TensorBuilder_v8 &operator=(TensorBuilder_v8 const &) = delete;

private:
    Tensor_v8 m_tensor; //! Tensor built by the TensorBuilder class.
};

using Tensor = Tensor_v8;
using TensorBuilder = TensorBuilder_v8;

class Operation_v8 {
public:
    friend class OperationBuilder_v8;
    // std::string describe() const override {
    //     std::stringstream ss;
    //     ss << "CUDNN_BACKEND_OPERATION :"
    //        << " OpMode: " << op_mode;
    //     ss << std::hex << " X " << xdesc;
    //     ss << std::hex << " Y " << ydesc;
    //     ss << std::hex << " W " << wdesc;
    //     ss << std::hex << " B " << bdesc;
    //     ss << std::hex << " T " << tdesc;
    //     ss << std::hex << " DW " << dwdesc;
    //     ss << std::hex << " DY " << dydesc;
    //     ss << std::hex << " DX " << dxdesc;
    //     ss << std::hex << " C " << cdesc;
    //     ss << std::hex << " A Mtrix " << amatdesc;
    //     ss << std::hex << " B Mtrix " << bmatdesc;
    //     ss << std::hex << " C Mtrix " << cmatdesc;
    //     ss << std::hex << " P " << pwdesc;
    //     ss << std::hex << " MatMul " << matmuldesc;
    //     ss << std::hex << " Reduction " << reductiondesc;
    //     ss << std::dec << " alphabetaType " << alphabetaType;
    //     ss << " Alpha: " << alpha_s << " " << alpha_d;
    //     ss << " Alpha2: " << alpha2_s << " " << alpha2_d;
    //     ss << " Beta: " << beta_s << " " << beta_d;
    //     return ss.str();
    // }

    Operation_v8(Operation_v8 &&from) = default;
    Operation_v8 &operator=(Operation_v8 &&from) = default;

    ~Operation_v8() = default;

private:
    Operation_v8() = default;
    Operation_v8(Operation_v8 const &) = delete;
    Operation_v8 &operator=(Operation_v8 const &) = delete;

    op::kind op_kind = op::kind::MatMul;

    lt *amatdesc = nullptr;
    lt *bmatdesc = nullptr;
    lt *cmatdesc = nullptr;

    op *internal_op = nullptr;
};

class OperationBuilder_v8 {
private:
    Operation_v8 m_operation;

    OperationBuilder_v8(op::kind op_kind) { m_operation.op_kind = op_kind; }

    Operation_v8 &&build_matmul_op() {

        // score = query x key.T
        //auto score = logical_tensor(id++, dt, score_sz, layout_type::strided);
        auto bmm1 = op(op_id++, op::kind::MatMul, "bmm1");
        bmm1.set_attr<bool>(op::attr::transpose_b, true);
        bmm1.add_inputs({m_operation.amatdesc, m_operation.bmatdesc});
        bmm1.add_outputs({m_operation.cmatdesc});
        m_operation.internal_op = &bmm1;
        return std::move(m_operation);
    }

public:
    auto setaMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.amatdesc = tensor.getInternal_lt();
        return *this;
    }
    auto setbMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.bmatdesc = tensor.getInternal_lt();
        return *this;
    }
    auto setcMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.cmatdesc = tensor.getInternal_lt();
        return *this;
    }

    //! constructs the backend Operation_v8 by calling the cudnn API
    //! Throws the appropriate error message
    Operation_v8 &&build() {
        if (m_operation.op_kind == op::kind::MatMul) {
            return build_matmul_op();
        }
        return std::move(m_operation);
    }
};

using Operation = Operation_v8;
using OperationBuilder = OperationBuilder_v8;

///
/// OperationGraph_v8 Class
/// This class tells the properties of the Tensor_v8 on which the operation will be
/// performed
/// Properties:
///    - handle
///    - operation
///
/// Use OperationGraphBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class OperationGraph_v8 : public BackendDescriptor {
public:
    friend class OperationGraphBuilder_v8;
    // std::string describe() const override {
    //     std::stringstream ss;
    //     ss << "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR has " << numOps
    //        << " operations." << std::endl;
    //     ss << "Tag: " << opGraphTag << std::endl;
    //     return ss.str();
    // }

    OperationGraph_v8(OperationGraph_v8 &&from) = default;
    OperationGraph_v8 &operator=(OperationGraph_v8 &&from) = default;

    ~OperationGraph_v8() = default;

    uint64_t getOpCount() const { return numOps; }

    const std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> &getOps() const {
        return ops;
    }

private:
    OperationGraph_v8() = default;
    OperationGraph_v8(OperationGraph_v8 const &) = delete;
    OperationGraph_v8 &operator=(OperationGraph_v8 const &) = delete;

    cudnnHandle_t handle = nullptr;
    std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> ops {};
    int64_t numOps = -1;
};

///
/// OperationGraphBuilder_v8 Class
/// Helper class used to build OperationGraph_v8 class
class OperationGraphBuilder_v8 {
public:
    /** @defgroup OperationGraphBuilder_v8
     *  Set individual property of OperationGraph_v8 class
     *  @{
     */

    auto setHandle(Handle handle_) -> OperationGraphBuilder_v8 & {
        m_operationGraph.handle = handle_;
        return *this;
    }

    //! Set numoperations and the operations
    auto setOperationGraph(int64_t numOps_, Operation_v8 const **ops_)
            -> OperationGraphBuilder_v8 & {
        m_operationGraph.numOps = numOps_;
        for (auto i = 0u; i < numOps_; i++) {
            m_operationGraph.ops[i] = ops_[i]->get_desc();
        }
        return *this;
    }

    //! Set numoperations and the operations
    auto setOperationGraph(std::vector<Operation> const &ops_)
            -> OperationGraphBuilder_v8 & {
        m_operationGraph.numOps = ops_.size();
        m_operationGraph.feature_vectors.resize(ops_.size());
        for (auto i = 0u; i < ops_.size(); i++) {
            m_operationGraph.ops[i] = ops_[i].get_desc();
            m_operationGraph.opGraphTag += ops_[i].getTag() + '_';
            m_operationGraph.feature_vectors[i] = ops_[i].getFeatureVector();
        }
        return *this;
    }

    auto addOperation(ManagedOpaqueDescriptor desc)
            -> OperationGraphBuilder_v8 & {
        m_operationGraph.ops[m_operationGraph.numOps] = desc;
        ++m_operationGraph.numOps;
        return *this;
    }
    /** @} */

    //! constructs the OperationGraph_v8 by calling the cudnn API
    //! Throws the appropriate error message
    OperationGraph_v8 &&build() { return std::move(m_operationGraph); }

    explicit OperationGraphBuilder_v8() = default;
    ~OperationGraphBuilder_v8() = default;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 &&) = delete;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 const &) = delete;
    OperationGraphBuilder_v8 &operator=(OperationGraphBuilder_v8 const &)
            = delete;

private:
    OperationGraph_v8 m_operationGraph;
};

using OperationGraph = OperationGraph_v8;
using OperationGraphBuilder = OperationGraphBuilder_v8;

} // namespace compat_0_x

#endif