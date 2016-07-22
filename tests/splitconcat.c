#include "mkl_dnn.h"

int main(int argc, char **argv)
{
    uint32_t full_buffer_sizes[4] = {256, 384, 13, 13};

    tensor_desc_t td;
    memory_desc_t md;
    memory_primitive_desc_t mpd;
    primitive_t mem_in, mem_out;

    tensor_desc_init(&td, 1, 1, 2, full_buffer_sizes);
    memory_desc_init(&md, &td, memory_format_nchw_f32);
    memory_primitive_desc_init(&mpd, &md, cpu_engine);
    memory_create(&mem_in, &mpd, NULL);
    memory_create(&mem_out, &mpd, NULL);

    ///////// SPLIT

    dims_t dims[2] = {{256, 256, 13, 13}, {256, 128, 13, 13}};
    nd_offset_t offts[2] = {{0, 0, 0, 0}, {0, 256, 0, 0}};

    split_desc_t sd;
    split_desc_init(&sd, &md, 2, dims, offts);

    // iterate over outputs creating memory primitives
    primitive_t mem_split[2];
    for (int i = 0; i < 2; i++) {
        // computes on the fly; makes it unnecessary to store sub-descs in the
        // structure
        memory_desc_t sub_md;
        split_desc_get_output_memory_desc(&sub_md, i);
        memory_primitive_desc_t sub_mpd;
        memory_primitive_desc_init(&sub_mpd, &sub_md, cpu_engine);
        // this requires memory_create() to be able to distinguish between full
        // and sub-memory creation; this is going to be based on memory_desc_t
        // definition; user is also responsible to pass the parent memory
        // primitive if a sub-memory is being created
        memory_create(&mem_split[i], &sub_mpd, sub_md.is_submemory ? mem_in : NULL);
    }

    split_primitive_desc_t spd;
    split_primitive_desc_init(&spd, cpu_engine);

    primitive_t split;
    split_create(&split, &spd, mem);

    ///////// CONCAT

    concat_desc_t cd;
    concat_desc_init(&cd, &md, 2, dims, offts);
    // same shit as with split
    primitive_t mem_concat[2];
    for (int i = 0; i < 2; i++) {
        memory_desc_t sub_md;
        concat_desc_get_input_memory_desc(&sub_md, i);
        memory_primitive_desc_t sub_mpd;
        memory_primitive_desc_init(&sub_mpd, &sub_md, cpu_engine);
        memory_create(&mem_concat[i], &sub_mpd, sub_md.is_submemory ? mem_in : NULL);
    }

    ///////// REORDER!

    // create reorders from mem_split to mem_concat

    return 0;
}
