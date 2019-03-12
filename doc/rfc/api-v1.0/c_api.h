/*
 * mkldnn_types.h
 *
 * NOTE: in some places `mkldnn_` prefix is omitted to simplify reading.
 */

#define MKLDNN_MAX_NDIMS 12 // previously 16, 12 should be enough

// previously typeof(dims_t[0]) == int, now ptrdiff_t to handle big 1D tensors
typedef int64_t dims_t[MKLDNN_MAX_NDIMS];

// enum that helps to specify the memory format
// (used in auxiliary functions only)
typedef enum {
    mkldnn_format_tag_any,      // special indicator

    mkldnn_a,
    mkldnn_ab,
    mkldnn_ba,
    //...
    mkldnn_abcd,     // corresponds to nchw, oihw
    mkldnn_acdb,     // corresponds to nhwc, ohwi
    // ...
    mkldnn_aBcd16b,  // corresponds to nChw16c, oIhw16c
    // ...

    // aliases below
    mkldnn_nchw = mkldnn_abcd,
    mkldnn_oihw = mkldnn_abcd,
    mkldnn_nChw16c = mkldnn_aBcd16b,
    // ...
} mkldnn_format_tag_t;

// enum that identifies how data is described in memory descriptor, i.e. at
// what field to look at in memory_desc_t.format_desc union.
typedef enum {
    mkldnn_format_kind_undef,
    mkldnn_format_kind_any,
    mkldnn_blocked,
    mkldnn_format_kind_wino,
} mkldnn_memory_format_kind_t

// new blocking structure that handles double+ blocking
typedef struct {
    dims_t strides;     // the strides between the *major* dimensions, i.e.
                        // between `O`, `I`, `h`, and `w`, in `OIhw4i16o4i`

    // tail section.
    // ASSUMPTION: the tail is always dense, unlike the *major* dimensions
    int tail_nblks;     // number of tail blocks, e.g. 3 in case OIhw_4i16o4i_
    dims_t tail_blks;   // the size of blocks, {4, 16, 4} in case OIhw4i16o4i
    dims_t tail_idxs;   // the logical indices of the tail blocks, e.g.
                        // {1, 0, 1} in case of 4i16o4i, because `i` is 1st dim
                        // and `o` is 0st dim
} blocking_desc_t;

// structure that holds *extra* information
typedef struct {
    // flags contain arbitrary extra info, such as compensation, e.g.:
    // - MKLDNN_MDEF_NONE (0u)
    // - MKLDNN_MDEF_COMP_CONV_S8S8 (1u) - tensor contains compensation
    //                                     information (along what dim?)
    // - MKLDNN_MDEF_ADJ_SCALE (2u) - tensor is adjusted by the given value
    //                                (used in s8s8-conv and wino2x3)
    uint64_t flags;     // flags contain arbitrary extra info,
                        // such as compensation
    float scale_adjust; // scale applied to the data (used on SKX)
    char reserved[64];  // for future backwards compatibility
} mkldnn_md_extra_t;

// reworked memory_desc_t
typedef struct {
    // logical description of the tensor
    int ndims;              // number of logical dimension
    dims_t dims;            // the dimensions themselves
    data_type_t data_type;  // (main) data type

    // information about padded dimensions
    dims_t padded_dims;     // the dimensions of the parent tensor incl. padding
    dims_t padded_offsets;  // the offsets of parent tensor with padded tensor

    // basic offset (useful in the cases when there is no pointer arithmetic)
    ptrdiff_t offset0;      // to access the first element of the padded
                            // tensor, one should dereference ptr + offset0

    // physical description
    memory_format_kind_t format_kind; // { undef, any, blocking, wino }
    union {
        blocking_desc_t;    // must be able to handle double+ blocking
        wino_desc_t;
    } format_desc;

    // section with *extra* information
    mkldnn_md_extra_t extra;
} mkldnn_memory_desc_t;


#define MKLDNN_NATIVE_HANDLE_ALLOCATE  ((void *)-1)
#define MKLDNN_NATIVE_HANDLE_NONE      ((void *)0)

struct mkldnn_memory_pd_t; // previous primitive descriptor
typedef const mkldnn_memory_pd_t const_mkldnn_memory_pd;

struct mkldnn_memory_t; // previously primitive


// ...

#define MKLDNN_ARG_SRC_0                1
#define MKLDNN_ARG_SRC                  MKLDNN_ARG_SRC_0
#define MKLDNN_ARG_SRC_LAYER            MKLDNN_ARG_SRC_0

#define MKLDNN_ARG_SRC_1                2
#define MKLDNN_ARG_SRC_ITER             MKLDNN_ARG_SRC_1

#define MKLDNN_ARG_DST_0                17
#define MKLDNN_ARG_DST                  MKLDNN_ARG_DST_0
#define MKLDNN_ARG_TO                   MKLDNN_ARG_DST_0
#define MKLDNN_ARG_DST_LAYER            MKLDNN_ARG_DST_0

#define MKLDNN_ARG_DST_1                18
#define MKLDNN_ARG_DST_ITER             MKLDNN_ARG_DST_1

#define MKLDNN_ARG_WEIGHTS_0            33
#define MKLDNN_ARG_WEIGHTS              MKLDNN_ARG_WEIGHTS_0
#define MKLDNN_ARG_SCALE_SHIFT          MKLDNN_ARG_WEIGHTS_0
#define MKLDNN_ARG_WEIGHTS_LAYER        MKLDNN_ARG_WEIGHTS_0

#define MKLDNN_ARG_WEIGHTS_1            34
#define MKLDNN_ARG_BIAS                 MKLDNN_ARG_WEIGHTS_1
#define MKLDNN_ARG_WEIGHTS_ITER         MKLDNN_ARG_WEIGHTS_1

#define MKLDNN_ARG_MEAN                 49
#define MKLDNN_ARG_VARIANCE             50

#define MKLDNN_ARG_WORKSPACE            64
#define MKLDNN_ARG_SCRATCHPAD           80

#define MKLDNN_ARG_DIFF_SRC_0           129
#define MKLDNN_ARG_DIFF_SRC             MKLDNN_ARG_DIFF_SRC_0
#define MKLDNN_ARG_DIFF_SRC_LAYER       MKLDNN_ARG_DIFF_SRC_0

#define MKLDNN_ARG_DIFF_SRC_1           130
#define MKLDNN_ARG_DIFF_SRC_ITER        MKLDNN_ARG_DIFF_SRC_1

#define MKLDNN_ARG_DIFF_DST_0           145
#define MKLDNN_ARG_DIFF_DST             MKLDNN_ARG_DIFF_DST_0
#define MKLDNN_ARG_DIFF_DST_LAYER       MKLDNN_ARG_DIFF_DST_0

#define MKLDNN_ARG_DIFF_DST_1           146
#define MKLDNN_ARG_DIFF_DST_ITER        MKLDNN_ARG_DIFF_DST_1

#define MKLDNN_ARG_DIFF_WEIGHTS_0       161
#define MKLDNN_ARG_DIFF_WEIGHTS         MKLDNN_ARG_DIFF_WEIGHTS_0
#define MKLDNN_ARG_DIFF_SCALE_SHIFT     MKLDNN_ARG_DIFF_WEIGHTS_0
#define MKLDNN_ARG_DIFF_WEIGHTS_LAYER   MKLDNN_ARG_DIFF_WEIGHTS_0

#define MKLDNN_ARG_DIFF_WEIGHTS_1       162
#define MKLDNN_ARG_DIFF_BIAS            MKLDNN_ARG_DIFF_WEIGHTS_1
#define MKLDNN_ARG_DIFF_WEIGHTS_ITER    MKLDNN_ARG_DIFF_WEIGHTS_1

#define MKLDNN_ARG_MULTIPLE_SRC         1024
#define MKLDNN_ARG_MULTIPLE_DST         2048

// describes an argument
typedef struct {
    int arg; // MKLDNN_ARG_SRC, ...
    mkldnn_memory_t memory;
} mkldnn_exec_arg_t;


// ...


typedef enum {
    // ...
    mkldnn_query_workspace_md,  /**< workspace memory primitive desc */
    mkldnn_query_scratchpad_md, /**< scratchpad memory primitive desc */
} mkldnn_query_t;




/*
 * mkldnn.h
 */

// inits memory desc for the given dims and strides between them
mkldnn_memory_desc_init_by_strides(mkldnn_memory_desc_t *md,
        int ndims, const dims_t dims, const dims_t strides,
        mkldnn_data_type_t data_type);

// inits memory desc for the given dims and memory format
// for those who is used to the previous versions
mkldnn_memory_desc_init_by_tag(mkldnn_memory_desc_t *md,
        int ndims, const dims_t dims, mkldnn_format_tag_t tag,
        mkldnn_data_type_t data_type);

// creates a memory
// native_handle can:
//  - point to the user allocated memory, i.e. valid handle. In this case the
//    library doesn't own allocated memory.
//  - be MKLDNN_NATIVE_HANDLE_ALLOCATE to ask the library to allocate and
//    attach memory. In this case the library owns allocated memory.
//  - be MKLDNN_NATIVE_HANDLE_NONE to create mkldnn_memory w/o attached memory.
mkldnn_status_t mkldnn_memory_create(mkldnn_memory_t *memory,
    const mkldnn_memory_dest_t *md, mkldnn_engine_t engine, void *handle);

// attaches a native handle to memory
// perform_zero_padding is flag that indicates whether the library should
// zero pad the padded area (if any). It is recommended to always pass 1 (true)
// to have a consistent memory (i.e. memory that corresponds to the memory
// descriptor structure).
mkldnn_status_t mkldnn_memory_set_data_handle(mkldnn_memory_t *mem,
        void *handle, int perform_zero_padding);

// creates a primitive based on a primitive descriptor. no input/outputs
mkldnn_status_t mkldnn_primitive_create(mkldnn_primitive_t *primitive,
        const_mkldnn_primitive_desc_t *pd);

// executes primitive on a given stream and given set of inputs and outputs
mkldnn_status_t mkldnn_primitive_exec(mkldnn_primitive_t prim,
        mkldnn_stream_t stream, int nargs, const mkldnn_exec_arg_t *args);
