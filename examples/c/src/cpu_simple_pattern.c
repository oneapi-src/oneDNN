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

/// @example cpu_simple_pattern.c
/// @copybrief cpu_simple_pattern_c
/// > Annotated version: @ref cpu_simple_pattern_c

/// @page cpu_simple_pattern_c A CPU example for conv+relu+conv+relu pattern
///
/// > Example code: @ref cpu_simple_pattern.c

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/allocator.h"
#include "common/graph.h"
#include "common/op.h"
#include "common/op_def.h"
#include "common/tensor.h"
#include "common/utils.h"

#include "oneapi/dnnl/dnnl_graph.h"

typedef struct {
    example_op_t *example_op_;
    dnnl_graph_op_t dnnl_graph_op_;
} op_map;

typedef struct {
    example_op_t *e_op_;
    dnnl_graph_partition_t l_p_;
    dnnl_graph_compiled_partition_t l_cp_;

} partition_map;

op_map o_map[100];
int64_t o_map_num = 0;

partition_map p_map[100];
int64_t p_map_num = 0;

void op_map_add(example_op_t *e_op, dnnl_graph_op_t l_op) {
    o_map[o_map_num].example_op_ = e_op;
    o_map[o_map_num].dnnl_graph_op_ = l_op;
    o_map_num++;
}

example_result_t find_example_op_by_dnnl_graph_op_id(
        example_op_t **op, size_t lp_id) {
    for (int k = 0; k < o_map_num; k++) {
        size_t temp_id;
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_get_id(o_map[k].dnnl_graph_op_, &temp_id));
        if (temp_id == lp_id) {
            *op = o_map[k].example_op_;
            if (*op == NULL) return example_result_error_common_fail;
            return example_result_success;
        }
    }
    return example_result_error_common_fail;
}

void partition_map_add(example_op_t *e_op, dnnl_graph_partition_t l_p) {
    p_map[p_map_num].e_op_ = e_op;
    p_map[p_map_num].l_p_ = l_p;
    p_map[p_map_num].l_cp_ = NULL;
    p_map_num++;
}

int64_t find_by_dnnl_graph_partition(dnnl_graph_partition_t l_p) {
    for (int k = 0; k < p_map_num; k++) {
        if (l_p == p_map[k].l_p_) { return k; }
    }
    return -1;
}

int64_t find_by_example_op(example_op_t *e_op) {
    for (int k = 0; k < p_map_num; k++) {
        if (e_op == p_map[k].e_op_) { return k; }
    }
    return -1;
}

/*! \brief simple conv->relu->conv->relu pattern */
example_result_t create_simple_pattern_graph(
        example_graph_t **graph, example_tensor_t *input) {
    CHECK_EXAMPLE(example_graph_create(graph));

    example_tensor_t *net = NULL;
    net = createConv2d(*graph, "conv_0", input, 96, 11, 4, 0, 1, 1);
    net = createRelu(*graph, "relu_0", net);
    net = createConv2d(*graph, "conv_1", net, 96, 3, 1, 0, 1, 1);
    net = createRelu(*graph, "relu_1", net);

    if (net == NULL) return example_result_error_common_fail;

    return example_result_success;
}

dnnl_graph_op_attr_t convert_attr(const char *name) {
    if (strcmp(name, "strides") == 0) {
        return dnnl_graph_op_attr_strides;
    } else if (strcmp(name, "pads_begin") == 0) {
        return dnnl_graph_op_attr_pads_begin;
    } else if (strcmp(name, "pads_end") == 0) {
        return dnnl_graph_op_attr_pads_end;
    } else if (strcmp(name, "dilations") == 0) {
        return dnnl_graph_op_attr_dilations;
    } else if (strcmp(name, "data_format") == 0) {
        return dnnl_graph_op_attr_data_format;
    } else if (strcmp(name, "filter_format") == 0) {
        return dnnl_graph_op_attr_filter_format;
    } else if (strcmp(name, "groups") == 0) {
        return dnnl_graph_op_attr_groups;
    } else {
        return dnnl_graph_op_attr_undef;
    }
}

int is_s64_attr(dnnl_graph_op_attr_t attr) {
    if (attr == dnnl_graph_op_attr_strides
            || attr == dnnl_graph_op_attr_pads_begin
            || attr == dnnl_graph_op_attr_pads_end
            || attr == dnnl_graph_op_attr_dilations
            || attr == dnnl_graph_op_attr_groups)
        return 1;
    else
        return 0;
}

int is_str_attr(dnnl_graph_op_attr_t attr) {
    if (attr == dnnl_graph_op_attr_data_format
            || attr == dnnl_graph_op_attr_filter_format)
        return 1;
    else
        return 0;
}

dnnl_graph_op_t convert_op(example_op_t *e_op) {
    dnnl_graph_op_t l_op = NULL;

    // dispatch to different op kind
    if (e_op->kind_ == e_kconv2d) {
        DNNL_GRAPH_CHECK(dnnl_graph_op_create(
                &l_op, e_op->id_, dnnl_graph_op_convolution, e_op->name_));
    } else if (e_op->kind_ == e_krelu) {
        DNNL_GRAPH_CHECK(dnnl_graph_op_create(
                &l_op, e_op->id_, dnnl_graph_op_relu, e_op->name_));
    } else {
        // TODO(qun) support more op kind
    }

    // add op attrs
    if (l_op != NULL) {
        for (int i = 0; i < e_op->attrs_num_; i++) {
            example_attr_t *attr = e_op->attrs_[i];
            dnnl_graph_op_attr_t val = convert_attr(attr->name_);
            if (is_s64_attr(val)) {
                DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
                        l_op, val, attr->data_, attr->data_num_));
            } else if (is_str_attr(val)) {
                DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_str(
                        l_op, val, attr->data_, attr->data_num_));
            } else {
                // TODO(xxx) support more op attr
            }
        }
    }

    return l_op;
}

int main(int argc, char **argv) {
    // Get input args
    dnnl_graph_engine_kind_t engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == dnnl_graph_gpu) {
        printf("Don't support gpu now\n");
        return -1;
    }

    /*
     * Part 1:
     * We construct a graph in example level, which simulates the framework graph.
     */

    // Step 1: create example input tensor
    int64_t input_dims[] = {8, 3, 227, 227};

    example_tensor_t *example_input = NULL;
    CHECK_EXAMPLE(
            example_tensor_create(&example_input, f32, 4, input_dims, any));

    // Step 2: construct a example graph
    example_graph_t *example_graph = NULL;
    CHECK_EXAMPLE(create_simple_pattern_graph(&example_graph, example_input));

    /*
     * Part 2:
     * We show how to use dnnl graph to optimize the example graph, and run it.
     */

    // Step 1: create allocator by using example's memory management function
    printf("Step 1: Create allocator----------------");
    dnnl_graph_allocator_t allocator = NULL;
    DNNL_GRAPH_CHECK(
            dnnl_graph_allocator_create(&allocator, allocate, deallocate));
    printf("Success!\n");

    // Step 2: create an engine and set the allocator to it,
    // the engine and allocator will used by dnnl graph backend to manage memory resource
    printf("Step 2: Create engine-------------------");
    dnnl_graph_engine_t engine = NULL;
    int32_t device_id = 0;
    DNNL_GRAPH_CHECK(dnnl_graph_engine_create_with_allocator(
            &engine, engine_kind, device_id, allocator));
    printf("Success!\n");

    // Step 3: traverse the example graph, convert each op to
    // dnnl graph op, create corresponding logical tensor edge and
    // then add supported ops into the graph
    printf("Step 3: Add OP to graph-----------------");
    dnnl_graph_graph_t graph = NULL;
    DNNL_GRAPH_CHECK(dnnl_graph_graph_create(&graph, engine_kind));

    for (int i = 0; i < example_graph->op_num_; i++) {
        example_op_t *e_op = example_graph->ops_[i];

        // convert example op to dnnl graph op and merge attrs
        dnnl_graph_op_t l_op = convert_op(e_op);
        if (l_op == NULL) {
            example_graph_destroy(example_graph);
            example_tensor_destroy_all();
            DNNL_GRAPH_CHECK(dnnl_graph_graph_destroy(graph));
            DNNL_GRAPH_CHECK(dnnl_graph_engine_destroy(engine));
            DNNL_GRAPH_CHECK(dnnl_graph_allocator_destroy(allocator));
            return -1;
        }
        op_map_add(e_op, l_op);

        // add in/outputs
        for (int j = 0; j < e_op->inputs_num_; j++) {
            example_tensor_t *e_t = e_op->inputs_[j];

            // create a logical tensor to represent edge according
            // to example.
            dnnl_graph_logical_tensor_t l_lt;
            DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&l_lt,
                    e_t->id_, (dnnl_graph_data_type_t)e_t->dtype_, e_t->ndims_,
                    e_t->dims_, dnnl_graph_layout_type_any,
                    dnnl_graph_tensor_property_undef));
            DNNL_GRAPH_CHECK(
                    dnnl_graph_op_add_input(l_op, &l_lt)); // value copy
        }

        for (int j = 0; j < e_op->outputs_num_; j++) {
            example_tensor_t *e_t = e_op->outputs_[j];
            dnnl_graph_logical_tensor_t l_lt;
            DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&l_lt,
                    e_t->id_, (dnnl_graph_data_type_t)e_t->dtype_, e_t->ndims_,
                    e_t->dims_, dnnl_graph_layout_type_any,
                    dnnl_graph_tensor_property_undef));
            DNNL_GRAPH_CHECK(dnnl_graph_op_add_output(l_op, &l_lt));
        }

        // add the op into a graph (internal graph)
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, l_op));
    }
    printf("Success!\n");

    // Step 4: filter and get partition
    // Run pass to optimize the added graph. this will fuse some op into
    // one fused op, so the graph will be rewrited after this process
    printf("Step 4: Filter and get partition--------");
    DNNL_GRAPH_CHECK(
            dnnl_graph_graph_filter(graph, dnnl_graph_partition_policy_fusion));

    // Get partition from the optimized graph. Each partition will be composed
    // of a single op (fused or unfused op)
    size_t partitions_num;
    DNNL_GRAPH_CHECK(
            dnnl_graph_graph_get_partition_num(graph, &partitions_num));
    if (partitions_num != 2) {
        printf("Error: partitions number is not equal to %llu\n",
                (unsigned long long)partitions_num);
        example_graph_destroy(example_graph);
        example_tensor_destroy_all();
        DNNL_GRAPH_CHECK(dnnl_graph_graph_destroy(graph));
        DNNL_GRAPH_CHECK(dnnl_graph_engine_destroy(engine));
        return -1;
    }
    dnnl_graph_partition_t partitions[2];
    for (int i = 0; i < partitions_num; i++) {
        DNNL_GRAPH_CHECK(dnnl_graph_partition_create(&partitions[i]));
    }
    DNNL_GRAPH_CHECK(
            dnnl_graph_graph_get_partitions(graph, partitions_num, partitions));
    printf("Success!\n");
    printf("Partition number: %llu\n", (unsigned long long)partitions_num);

    // Step 5: rewrite the example graph
    // replace the example graph's op with fake op, which will be bind to a
    // partition and run by dnnl graph in execution
    printf("Step 5: Rewrite the example graph-------");
    int64_t unoptimized_graph_op_num = example_graph->op_num_;
    for (int i = 0; i < partitions_num; i++) {
        // create a fake op to represent the partition in example graph
        // and cache the map between fake op and partition
        example_op_t *e_op_fake = NULL;
        CHECK_EXAMPLE(example_op_create_base(
                &e_op_fake, "DNNL_GRAPH_EXE_OP", e_kfake));
        partition_map_add(e_op_fake, partitions[i]);

        // get partition's op
        size_t ops_num;
        DNNL_GRAPH_CHECK(
                dnnl_graph_partition_get_op_num(partitions[i], &ops_num));
        size_t partition_ops_ids[2];
        DNNL_GRAPH_CHECK(dnnl_graph_partition_get_ops(
                partitions[i], ops_num, partition_ops_ids));

        // for every op_id in partition, we need to find the corresponding example
        // op, cut off its connection in the example graph and re-connect its inputs
        // and outputs to the new fake op.
        for (int j = 0; j < ops_num; j++) {
            example_op_t *e_op_erase;
            CHECK_EXAMPLE(find_example_op_by_dnnl_graph_op_id(
                    &e_op_erase, partition_ops_ids[j]));

            for (int k = 0; k < e_op_erase->inputs_num_; k++) {
                example_tensor_t *input = e_op_erase->inputs_[k];
                CHECK_EXAMPLE(example_tensor_erase_user(input, e_op_erase));

                e_op_fake->inputs_[e_op_fake->inputs_num_] = input;
                CHECK_EXAMPLE(example_tensor_add_user(
                        input, e_op_fake, e_op_fake->inputs_num_));
                e_op_fake->inputs_num_++;
            }

            for (int k = 0; k < e_op_erase->outputs_num_; k++) {
                example_tensor_t *output = e_op_erase->outputs_[k];
                CHECK_EXAMPLE(
                        example_tensor_erase_producer(output, e_op_erase));

                e_op_fake->outputs_[e_op_fake->outputs_num_] = output;
                CHECK_EXAMPLE(example_tensor_set_producer(
                        output, e_op_fake, e_op_fake->outputs_num_));
                e_op_fake->outputs_num_++;
            }
        }

        // add the new fake op to example graph, and destroy the fused op
        int32_t do_replace = 1;
        for (int64_t j = example_graph->op_num_ - 1; j >= 0; j--) {
            int32_t found = 0;

            for (int k = 0; k < ops_num; k++) {
                example_op_t *e_op_erase;
                CHECK_EXAMPLE(find_example_op_by_dnnl_graph_op_id(
                        &e_op_erase, partition_ops_ids[k]));
                if (example_graph->ops_[j] == e_op_erase) {
                    found = 1;
                    break;
                }
            }

            if (!found) continue;

            example_op_destroy(example_graph->ops_[j]);

            // we replace the first found fused op with the fake op
            if (do_replace) {
                example_graph->ops_[j] = e_op_fake;
                do_replace = 0;
                continue;
            }

            // just erase the non-first found fused op
            for (int64_t k = j; k < example_graph->op_num_ - 1; k++) {
                example_graph->ops_[k] = example_graph->ops_[k + 1];
            }
            example_graph->op_num_--;
        }

    } // for (int i = 0; i < partitions_num; i++)
    printf("Success!\n");
    printf("Op num in unoptimized graph: %lld\n",
            (long long int)unoptimized_graph_op_num);
    printf("Op num in optimized graph: %lld\n",
            (long long int)example_graph->op_num_);

    // Step 6: compile the partitions
    // do layout propagation and mount backend's execution kernel
    printf("Step 6: Compile the partitions----------");
    dnnl_graph_compiled_partition_t compiled_partitions[2];
    for (int i = 0; i < partitions_num; i++) {
        // we find the corresponding fake op in example graph, create
        // logical tensor according to its inputs and outputs example
        // tensor.
        int64_t idx = find_by_dnnl_graph_partition(partitions[i]);
        example_op_t *e_op = p_map[idx].e_op_;

        DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_create(
                &compiled_partitions[i], partitions[i]));
        p_map[idx].l_cp_ = compiled_partitions[i];

        dnnl_graph_logical_tensor_t l_lts_in[3];
        for (int j = 0; j < e_op->inputs_num_; j++) {
            example_tensor_t *e_t = e_op->inputs_[j];

            if (!e_t->producer_.producer_) {
                // this is the graph's input, we set the layout id to plain
                DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
                        l_lts_in + j, e_t->id_,
                        (dnnl_graph_data_type_t)e_t->dtype_, e_t->ndims_,
                        e_t->dims_, dnnl_graph_layout_type_strided,
                        dnnl_graph_tensor_property_undef));

            } else {
                // we need to query logical tensor id from the producer compiled partition
                int64_t temp_idx = find_by_example_op(e_t->producer_.producer_);
                dnnl_graph_compiled_partition_t l_cp = p_map[temp_idx].l_cp_;
                DNNL_GRAPH_CHECK(
                        dnnl_graph_compiled_partition_query_logical_tensor(
                                l_cp, e_t->id_, l_lts_in + j));
            }
        }

        dnnl_graph_logical_tensor_t l_lts_out[2];
        for (int j = 0; j < e_op->outputs_num_; j++) {
            example_tensor_t *e_t = e_op->outputs_[j];

            dnnl_graph_layout_type_t layout_type;
            if (!e_t->users_num_) {
                // this is the graph's output, we set the layout type to plain
                layout_type = dnnl_graph_layout_type_strided;
            } else {
                layout_type = dnnl_graph_layout_type_any;
            }
            DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
                    l_lts_out + j, e_t->id_,
                    (dnnl_graph_data_type_t)e_t->dtype_, e_t->ndims_,
                    e_t->dims_, layout_type, dnnl_graph_tensor_property_undef));
        }

        // compile the partition
        const dnnl_graph_logical_tensor_t *l_lts_in_ptr[3];
        const dnnl_graph_logical_tensor_t *l_lts_out_ptr[2];
        for (int j = 0; j < e_op->inputs_num_; j++) {
            l_lts_in_ptr[j] = l_lts_in + j;
        }
        for (int j = 0; j < e_op->outputs_num_; j++) {
            l_lts_out_ptr[j] = l_lts_out + j;
        }

        DNNL_GRAPH_CHECK(dnnl_graph_partition_compile(partitions[i],
                compiled_partitions[i], e_op->inputs_num_, l_lts_in_ptr,
                e_op->outputs_num_, l_lts_out_ptr, engine));

        // partitions are all copied to compiled partition,
        // now we can destroy them safely
        DNNL_GRAPH_CHECK(dnnl_graph_partition_destroy(partitions[i]));
    }
    printf("Success!\n");

    // Step 7: alloc memory for example tensor
    // for opaque tensors, we need to get their actual memory size from compiled partition
    printf("Step 7: Alloc memory--------------------");
    for (int i = 0; i < example_graph->op_num_; i++) {
        example_op_t *e_op = example_graph->ops_[i];

        if (strcmp(e_op->name_, "DNNL_GRAPH_EXE_OP") != 0) {
            // this example op is not fake op, it's input and output memory should be
            // prepared in other ways
            continue;
        }

        int64_t idx = find_by_example_op(e_op);
        dnnl_graph_compiled_partition_t l_cp = p_map[idx].l_cp_;

        for (int j = 0; j < e_op->inputs_num_; j++) {
            example_tensor_t *e_t = e_op->inputs_[j];
            if (e_t->data_) continue;

            dnnl_graph_logical_tensor_t temp;
            size_t mem_size = 0;
            DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
                    l_cp, e_t->id_, &temp));
            DNNL_GRAPH_CHECK(
                    dnnl_graph_logical_tensor_get_mem_size(&temp, &mem_size));
            if (mem_size == 0 || mem_size == (size_t)-1) { continue; }
            e_t->data_ = malloc(mem_size);
        }

        for (int j = 0; j < e_op->outputs_num_; j++) {
            example_tensor_t *e_t = e_op->outputs_[j];
            if (e_t->data_) continue;

            dnnl_graph_logical_tensor_t temp;
            size_t mem_size = 0;
            DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
                    l_cp, e_t->id_, &temp));
            DNNL_GRAPH_CHECK(
                    dnnl_graph_logical_tensor_get_mem_size(&temp, &mem_size));
            if (mem_size == 0 || mem_size == (size_t)-1) { continue; }
            e_t->data_ = malloc(mem_size);
        }
    }
    printf("Success!\n");

    // Step 8: Execute compiled partitions
    printf("Step 8: Execute compiled partitions-----");
    dnnl_graph_stream_t stream = NULL;
    DNNL_GRAPH_CHECK(dnnl_graph_stream_create(&stream, engine));
    for (int i = 0; i < example_graph->op_num_; i++) {
        example_op_t *e_op = example_graph->ops_[i];

        if (strcmp(e_op->name_, "DNNL_GRAPH_EXE_OP") != 0) {
            // this example op is not a fake op, need to be executed
            // in other ways
            continue;
        }

        int64_t idx = find_by_example_op(e_op);
        dnnl_graph_compiled_partition_t l_cp = p_map[idx].l_cp_;

        dnnl_graph_tensor_t l_ts_in[3];
        for (int j = 0; j < e_op->inputs_num_; j++) {
            example_tensor_t *e_t = e_op->inputs_[j];

            dnnl_graph_logical_tensor_t l_lt;
            DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
                    l_cp, e_t->id_, &l_lt));

            DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
                    &l_ts_in[j], &l_lt, engine, e_t->data_));
        }

        dnnl_graph_tensor_t l_ts_out[2];
        for (int j = 0; j < e_op->outputs_num_; j++) {
            example_tensor_t *e_t = e_op->outputs_[j];

            dnnl_graph_logical_tensor_t l_lt;
            DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
                    l_cp, e_t->id_, &l_lt));

            DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
                    &l_ts_out[j], &l_lt, engine, e_t->data_));
        }

        // execute the compiled partition
        DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_execute(l_cp, stream,
                e_op->inputs_num_, (const_dnnl_graph_tensor_t *)l_ts_in,
                e_op->outputs_num_, (const_dnnl_graph_tensor_t *)l_ts_out));

        // destroy tensor, which will not free data_ memory
        for (int j = 0; j < e_op->inputs_num_; j++) {
            DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(l_ts_in[j]));
        }
        for (int j = 0; j < e_op->outputs_num_; j++) {
            DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(l_ts_out[j]));
        }
    }
    DNNL_GRAPH_CHECK(dnnl_graph_stream_destroy(stream));
    printf("Success!\n");

    // release resource
    for (int i = 0; i < partitions_num; i++) {
        DNNL_GRAPH_CHECK(
                dnnl_graph_compiled_partition_destroy(compiled_partitions[i]));
    }
    for (int i = 0; i < o_map_num; i++) {
        DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(o_map[i].dnnl_graph_op_));
    }
    example_graph_destroy(example_graph);
    example_tensor_destroy_all();
    DNNL_GRAPH_CHECK(dnnl_graph_graph_destroy(graph));
    DNNL_GRAPH_CHECK(dnnl_graph_engine_destroy(engine));
    DNNL_GRAPH_CHECK(dnnl_graph_allocator_destroy(allocator));

    return 0;
}
