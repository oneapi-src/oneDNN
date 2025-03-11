#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"
using namespace dnnl::graph;
int main() {
    allocator alloc {};
    dnnl::engine eng
            = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
    graph g {engine::kind::cpu};

    // Create dnnl::stream.
    dnnl::stream strm(eng);

    size_t lt_id = 0;
    size_t op_id = 0;

    std::vector<int64_t> dot1_in1_shape {16, 128, 256};
    std::vector<int64_t> dot1_in2_shape {16, 256, 128};
    std::vector<int64_t> dot1_out_shape {
            16, 128, 128}; // also multiply 2nd operand shape
    logical_tensor lt_dot1_in1 {lt_id++, logical_tensor::data_type::f32,
            dot1_in1_shape, logical_tensor::layout_type::strided};
    logical_tensor lt_dot1_in2 {lt_id++, logical_tensor::data_type::f32,
            dot1_in2_shape, logical_tensor::layout_type::strided};
    logical_tensor lt_dot1_out {lt_id++, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    op matmul_1_op(op_id++, op::kind::MatMul, "matmul_1");
    matmul_1_op.add_inputs({lt_dot1_in1, lt_dot1_in2});
    matmul_1_op.add_output(lt_dot1_out);

    std::vector<int64_t> reshape1_out_shape {1, 16, 128, 128, 1};
    logical_tensor lt_reshape1_out {lt_id++, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    op reshape_1_op(op_id++, op::kind::StaticReshape, "reshape_1");
    reshape_1_op.set_attr(op::attr::special_zero, false);
    reshape_1_op.set_attr<std::vector<int64_t>>(
            op::attr::shape, reshape1_out_shape);
    reshape_1_op.add_input(lt_dot1_out);
    reshape_1_op.add_output(lt_reshape1_out);

    std::vector<int64_t> tr1_out_shape {1, 16, 1, 128, 128};
    logical_tensor lt_tr1_out {lt_id++, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    op transpose_1_op(op_id++, op::kind::StaticTranspose, "transpose_1");
    transpose_1_op.set_attr<std::vector<int64_t>>(
            op::attr::order, std::vector<int64_t> {0, 1, 4, 3, 2});
    transpose_1_op.add_input(lt_reshape1_out);
    transpose_1_op.add_output(lt_tr1_out);

    logical_tensor lt_add_in2 {lt_id++, logical_tensor::data_type::f32,
            tr1_out_shape, logical_tensor::layout_type::strided};
    logical_tensor lt_add_out {lt_id++, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    op add_op(op_id++, op::kind::Add, "add");
    add_op.add_inputs({lt_tr1_out, lt_add_in2});
    add_op.add_output(lt_add_out); // output shape is {1,16,1,128,128}

    op softmax_op(op_id++, op::kind::SoftMax, "softmax");
    softmax_op.set_attr<int64_t>(
            op::attr::axis, -1); // assume -1 means last axis, to confirm
    logical_tensor lt_softmax_out {lt_id++, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    softmax_op.add_input(lt_add_out);
    softmax_op.add_output(lt_softmax_out); //output shape is {1,4,2,1,31}

    logical_tensor lt_tr2_out {lt_id++, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    op transpose_2_op(op_id++, op::kind::StaticTranspose, "transpose_2");
    transpose_2_op.set_attr<std::vector<int64_t>>(
            op::attr::order, std::vector<int64_t> {0, 1, 4, 2, 3});
    transpose_2_op.add_input(lt_softmax_out);
    transpose_2_op.add_output(lt_tr2_out); //output shape is {1,16,128,1,128}

    std::vector<int64_t> reshape2_out_shape {
            16, 128, 128}; // input shape {1,16,128,1,128}
    logical_tensor lt_reshape2_out {lt_id++, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};

    op reshape_2_op(op_id++, op::kind::StaticReshape, "reshape_2");
    reshape_2_op.set_attr(op::attr::special_zero, false);
    reshape_2_op.set_attr<std::vector<int64_t>>(
            op::attr::shape, reshape2_out_shape);
    reshape_2_op.add_input(lt_tr2_out);
    reshape_2_op.add_output(lt_reshape2_out);

    std::vector<int64_t> dot2_in1_shape {16, 256, 128};
    std::vector<int64_t> dot2_out_shape {16, 256, 128};
    logical_tensor lt_dot2_in1 {lt_id++, logical_tensor::data_type::f32,
            dot2_in1_shape, logical_tensor::layout_type::strided};
    logical_tensor lt_dot2_out {lt_id++, logical_tensor::data_type::f32,
            dot2_out_shape, logical_tensor::layout_type::strided};

    op matmul_2_op(op_id++, op::kind::MatMul, "matmul_2");
    matmul_2_op.add_inputs({lt_dot2_in1, lt_reshape2_out});
    matmul_2_op.add_output(lt_dot2_out);

    g.add_op(matmul_1_op);
    g.add_op(reshape_1_op);
    g.add_op(transpose_1_op);
    g.add_op(add_op);
    g.add_op(softmax_op);
    g.add_op(transpose_2_op);
    g.add_op(reshape_2_op);
    g.add_op(matmul_2_op);

    g.finalize();
    auto partitions = g.get_partitions();

    assert(partitions.size() == 1);

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {lt_dot1_in1, lt_dot1_in2, lt_add_in2, lt_dot2_in1}, {lt_dot2_out},
            eng);

    // Create tensor objects
    auto ts_dot1_in1 = tensor(lt_dot1_in1, eng);
    auto ts_dot1_in2 = tensor(lt_dot1_in2, eng);
    auto ts_add_in2 = tensor(lt_add_in2, eng);
    auto ts_dot2_in1 = tensor(lt_dot2_in1, eng);
    auto ts_dot2_out = tensor(lt_dot2_out, eng);

    // Warmup run.
    // Execute the compiled partition of mqa.
    cp.execute(strm, {ts_dot1_in1, ts_dot1_in2, ts_add_in2, ts_dot2_in1},
            {ts_dot2_out});

    //     Wait for the computation to finish.
    strm.wait();
}
