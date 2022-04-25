/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

/*
 Fast, generic reorder kernel.

 Reorder kernel is strictly memory-bound. Performance is determined only by
 memory access efficiency.

 There's no perf difference between using single word reads and block read
 functions (intel_sub_group_read8 etc.). Only thing that matters is that when
 data is read/written from/to global memory, code must utilize whole cache
 lines. Using smaller chunks causes cache eviction and requires the same data
 to be accessed later again, wasting bandwidth.

 This kernel tries to load/store data in packets that are at least as large as
  cache line.

  Example: abc -> bca, 32x32x32, data type f32
  Assume SIMD16 and cache line size = 64B
  To fill cache line kernel must load 16 consecutive items (16c) and
  store 16 consecutive items (16a). So it needs to operate on a matrix of
  16a x 16c.
  It will load 16 non-adjacent (strided by A) sets of 16 adjacent data
  (strided by C, src' innermost dimension), perform internal transposition,
   then store 16 non-adjacent (strided by C) sets of 16 adjacent data (strided
   by A, dst's innermost dimension).

Difficulty is in determining how to achieve the above goal for
  any combination of tensor size and format tags.
*/

#include <algorithm>
#include "gpu/ocl/generic_reorder.hpp"

#include "common/utils.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::memory_tracking::names;

using dimension_t = struct {
    dim_t size;
    dim_t step;
    int idx;
};

using dimensions_t = std::vector<dimension_t>;

bool is_power_of_2(int n) {
    return ((n & (n - 1)) == 0);
}

using stride_t = struct {
    dim_t stride;
    dim_t size;
    int idx;
};

// Stride sorter. Smaller stride = inner dim, bigger stride = outer dim.
// Dimensions of size 1 are considered outermost regardless of strides and
// they are sorted by index.
bool stride_less(const stride_t &a, const stride_t &b) {
    if (a.size == 1 && b.size == 1) {
        return a.idx > b.idx;
    } else if (a.size != 1 && b.size == 1) {
        return true;
    } else if (a.size == 1 && b.size != 1) {
        return false;
    } else {
        return a.stride < b.stride;
    }
}

// Returns description of blocks and dimensions that constitute the format tag
// of tensor, starting from innermost. Blocks, if exist, take precedence before
// dimensions. Order of dimensions is determined by sorting strides; smallest
// stride is innermost dimension. Dimensions of size 1 are ignored (treated as
// outermost). Dimensions are treated like blocks, that is they don't report
// whole tensor size across given axis but rather number of underlying blocks
// for given dimension.
// Example: ABcd8b8a2b 32x48x1x7 will return description that amounts to...
// outermost-> 1c:4a:3b:7d:8b:8a:2b <-innermost
dimensions_t query_dims_and_blocks(
        const memory_desc_wrapper &md, int distance = 0) {
    const int nblks = md.blocking_desc().inner_nblks;
    const int ndims = md.ndims();

    std::vector<stride_t> strides(ndims);
    for (int d = 0; d < ndims; ++d) {
        strides[d].idx = d;
        strides[d].stride = md.blocking_desc().strides[d];
        strides[d].size = md.padded_dims()[d];
    }
    std::sort(strides.begin(), strides.end(), stride_less);
    for (int i = 0; i < nblks; i++) {
        stride_t blk;
        blk.idx = md.blocking_desc().inner_idxs[i];
        blk.size = md.blocking_desc().inner_blks[i];
        if (i == 0 && blk.idx == strides[0].idx) { continue; }
        strides.insert(strides.begin(), blk);
    }
    // calculate step sizes
    int steps[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
    dimensions_t dims(strides.size());
    for (size_t i = 0; i < strides.size(); i++) {
        dims[i].idx = strides[i].idx;
        dims[i].size = strides[i].size;
        dims[i].step = steps[dims[i].idx];
        steps[strides[i].idx] *= strides[i].size;
    }
    // divide last instance of given dim (it's true dim, not a block)
    // by its step
    bool idx_done[MAX_NDIMS] = {false, false, false, false, false, false};
    for (int i = (int)(dims.size() - 1); i >= 0; i--) {
        if (!idx_done[dims[i].idx]) {
            dims[i].size = utils::div_up(dims[i].size, dims[i].step);
            idx_done[dims[i].idx] = true;
        }
    }
    return dims;
}

dimensions_t query_dims_and_blocks(const memory_desc_t &m, int distance = 0) {
    memory_desc_wrapper mdw(m);
    return query_dims_and_blocks(mdw, distance);
}

void fix_steps(dimensions_t &blk, dimensions_t pkt) {
    int steps[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
    for (size_t i = 0; i < pkt.size(); i++) {
        steps[pkt[i].idx] *= pkt[i].size;
    }
    for (size_t i = 0; i < blk.size(); i++) {
        blk[i].step = steps[blk[i].idx];
        steps[blk[i].idx] *= blk[i].size;
    }
}

// Returns vector of blocks that were present in a but missing from b
dimensions_t find_missing_blocks(
        dimensions_t all, dimensions_t subset, bool round_up) {
    dimensions_t ret;
    for (size_t ia = 0; ia < all.size(); ia++) {
        dimension_t from_a = all[ia];
        for (size_t ib = 0; ib < subset.size(); ib++) {
            if (subset[ib].idx == from_a.idx) {
                auto smaller = std::min(from_a.size, subset[ib].size);
                if (round_up) {
                    from_a.size = utils::div_up(from_a.size, smaller);
                    subset[ib].size = utils::div_up(subset[ib].size, smaller);
                } else {
                    from_a.size /= smaller;
                    subset[ib].size /= smaller;
                }
            }
        }
        if (from_a.size > 1) { ret.push_back(from_a); }
    }
    return ret;
}

enum order_t { none, a_then_b, b_then_a };

dimensions_t remainder(dimensions_t all, dimensions_t subset) {
    dimensions_t ret;
    for (size_t i = 0; i < all.size(); i++) {
        if (i < subset.size()) {
            if (all[i].size == subset[i].size) {
                continue;
            } else {
                dimension_t item;
                item.idx = all[i].idx;
                item.size = all[i].size / subset[i].size;
                item.step = all[i].step * subset[i].size;
                ret.push_back(item);
            }
        } else {
            ret.push_back(all[i]);
        }
    }
    return ret;
}

// Given format description, try to find formula for 16 adjacent items to
// vectorize across.
// Examples:
// For 1024x1024 ab, it will be (16b)
// for 16x16x16 ABc2a2b, it will be (4c2a2b)
bool fill_to_vect(int simd_size, dimensions_t all, dimensions_t &subset) {
    int need = simd_size;
    int current_size = 1;
    subset.clear();
    for (size_t i = 0; i < all.size(); i++) {
        if (i == 0 && all[i].size > 3 * simd_size) {
            // Vectorize innermost dim. Since it's not divisible by simd size,
            // it will need to be padded. And for that the vectorised dim
            // should be large enough because otherwise the padding would be
            // too significant fraction of tensor and it would hurt perf.
            dimension_t tmp = all[i];
            tmp.size = simd_size;
            subset.push_back(tmp);
            return true;
        }
        if (all[i].size % 2 != 0) {
            // can't fill packet to exactly 16
            return false;
        }
        if (all[i].size <= need) {
            if (2 * all[i].size <= simd_size && need % all[i].size != 0) {
                // Avoid the problematic case where multiple reads are necessary
                // to read 16 entries and the trailing packet has fewer than
                // all[i].size entries.
                return false;
            }
            subset.push_back(all[i]);
            current_size *= all[i].size;
            need /= all[i].size;
        } else if ((all[i].size / need) * need == all[i].size) {
            dimension_t tmp = all[i];
            tmp.size = need;
            subset.push_back(tmp);
            current_size *= tmp.size;
            need /= tmp.size;
        } else {
            // can't fill packet exactly to requested value; either size or
            // simd was not power-of-2?
            // TODO: cases where innermost dim is small and odd are not
            // supported by this kernel because there's no logic to construct
            // 16-item packet in that case. However, it should be possible to
            // support cases like ...16a3b: declare packet=5.3a3b, burst=3a,
            // then on opencl side add logic that will decode it into integer
            // a,b coordinates. Such change would almost eliminate use of slow,
            // reference reorder.
            return false;
        }
        if (need == 1) { return true; }
        if (current_size == 1) { return false; }
    }
    // there was not enough data in tensor to fill even a single packet
    return false;
}

bool add_to_vector(dimensions_t &v, dimension_t item) {
    if (v.empty() || item.idx != v.back().idx) {
        if (v.size() >= LOOP_NEST_LEVEL) { return false; }
        v.push_back(item);
        v.back().size = item.size;
    } else {
        v.back().size *= item.size;
    }
    return true;
}

bool no_more_such_idx(dimensions_t &vect, size_t iter) {
    const int idx_to_search_for = vect[iter].idx;
    for (size_t i = iter + 1; i < vect.size(); i++) {
        if (vect[i].idx == idx_to_search_for) { return false; }
    }
    return true;
}

// Given full description of tensor and subset of description,
// sort the subset in such way that it will describe longest possible
// sequence of continuous memory addresses.
// Example: full 32a32b4c4a, subset 12a2b4c,
// result = 3a2b4c4a, it gives 3 distant sets of 2*4*4 adjacent items
dimensions_t fix_order_to(dimensions_t input, dimensions_t ref) {
    dimensions_t ret;
    for (size_t i = 0; i < ref.size(); i++) {
        for (size_t j = 0; j < input.size(); j++) {
            if (ref[i].size != 1 && input[j].size != 1
                    && ref[i].idx == input[j].idx) {
                int smaller = std::min(ref[i].size, input[j].size);
                if (no_more_such_idx(ref, i) || j == input.size() - 1) {
                    smaller = input[j].size;
                }
                dimension_t item = ref[i];
                item.size = smaller;
                ref[i].size = utils::div_up(ref[i].size, smaller);
                input[j].size = utils::div_up(input[j].size, smaller);
                add_to_vector(ret, item);
            }
        }
    }
    // It is possible that requested block on a dimension of src is bigger than
    // whole dimension in src. That happens when there's large padding in dst.
    // Add this block at the end, it will be handled by padding in opencl code.
    for (size_t i = 0; i < input.size(); i++) {
        if (input[i].size > 1) { add_to_vector(ret, input[i]); }
    }
    return ret;
}

int check_size(dimensions_t block) {
    int length = 1;
    for (size_t i = 0; i < block.size(); i++) {
        length *= block[i].size;
    }
    return length;
}

// Given full tensor description and subset of that description, find
// how many items are adjacent in memory
size_t check_burst_length(dimensions_t all, dimensions_t subset) {
    size_t length = 1;
    for (size_t i = 0; i < all.size(); i++) {
        for (size_t j = 0; j < subset.size(); j++) {
            if (all[i].idx == subset[j].idx) {
                auto smaller = std::min(all[i].size, subset[j].size);
                length *= (int)smaller;
                all[i].size /= smaller;
                subset[j].size /= smaller;
            }
        }
        if (all[i].size != 1) {
            return length;
        } // dim not covered in block, so burst ends
    }
    return length;
}

// Given full tensor description and subset of that description which
// determines how many items will be read in a burst, try to enlarge subset
// to increase burst size to achieve better cache line utilizaton.
// Example: full 32a32b4c4a, subset 12a2b2c,
// current burst size = 8 (2c*4a); enlarge subset to 12a2b4c to achieve
// burst size = 32 (2b*4c*4a)
bool increase_burst(dimensions_t all, dimensions_t &subset, dimensions_t &other,
        size_t itemlimit, size_t current_size, size_t optimal_size) {
    const dim_t space_coeff = itemlimit / check_size(subset);
    const dim_t request_coeff = utils::div_up(optimal_size, current_size);
    dimensions_t subset_copy = subset;
    if (space_coeff < 2) { return false; }
    for (size_t i = 0; i < all.size(); i++) {
        for (size_t j = 0; j < subset_copy.size(); j++) {
            if (all[i].idx == subset_copy[j].idx) {
                auto smaller = std::min(all[i].size, subset_copy[j].size);
                all[i].size /= smaller;
                subset_copy[j].size /= smaller;
            }
        }
        if (all[i].size != 1) {
            // add to subset new item or enlarge last item, if it was the same dim
            auto incr = std::min(space_coeff, all[i].size);
            incr = std::min(incr, request_coeff);
            all[i].size = incr;
            bool success = add_to_vector(subset, all[i]);
            if (!success) { return false; }
            add_to_vector(other, all[i]);
            return true;
        }
    }
    return false;
}

// "packet" - set of 16 adjacent data to be read in one go by a subgroup
// "block"  - how many iterations of packet read should a subgroup do
// This function splits tensor description into blocks and packets in such way
// that optimizes burst length.
bool split_into_blocks_and_packets(size_t vect, size_t optimal_burst_bytes,
        size_t memlimit_bytes, size_t sizeof_src, size_t sizeof_dst,
        const dimensions_t &src, const dimensions_t &dst,
        dimensions_t &src_packet, dimensions_t &src_block,
        dimensions_t &dst_packet, dimensions_t &dst_block) {

    // 1. determine composition of src and dst packet
    if (!fill_to_vect((int)vect, src, src_packet)) { return false; }
    if (!fill_to_vect((int)vect, dst, dst_packet)) { return false; }
    // 2. determine which parts of tensor format tag are left after taking away packet
    dimensions_t sremainder = remainder(src, src_packet);
    dimensions_t dremainder = remainder(dst, dst_packet);
    // 3. The same amount of data will be read and written. So, every dimension
    // that's in src packet and not in dst packet must be in dst block.
    src_block = find_missing_blocks(dst_packet, src_packet, true);
    dst_block = find_missing_blocks(src_packet, dst_packet, false);
    // 4a. Check how much continuous data will be read/written...
    size_t burst_size_src
            = vect * sizeof_src * check_burst_length(sremainder, src_block);
    size_t burst_size_dst
            = vect * sizeof_dst * check_burst_length(dremainder, dst_block);
    bool success = true;
    // TODO: use smaller of SRC_T, DST_T type to conserve local mem
    size_t itemlimit = memlimit_bytes / (vect * sizeof_src);
    // 4b. ... and determine if that's long enough to achieve good performance
    while (success
            && (burst_size_src < optimal_burst_bytes
                    || burst_size_dst < optimal_burst_bytes)) {
        // 5. If burst needs to be longer, attempt to increase block size (but
        // don't exceed local memory limits as that would hurt performance)
        if (burst_size_src < burst_size_dst) {
            success = increase_burst(sremainder, src_block, dst_block,
                    itemlimit, burst_size_src, optimal_burst_bytes);
        } else {
            success = increase_burst(dremainder, dst_block, src_block,
                    itemlimit, burst_size_dst, optimal_burst_bytes);
        }
        burst_size_src
                = vect * sizeof_src * check_burst_length(sremainder, src_block);
        burst_size_dst
                = vect * sizeof_dst * check_burst_length(dremainder, dst_block);
    }
    // 6. At this point contents of src block and dst blocks are not sorted.
    // Sort each of them according to tensor format tag to make longest
    // possible continuous memory accesses.
    src_block = fix_order_to(src_block, sremainder);
    dst_block = fix_order_to(dst_block, dremainder);
    fix_steps(src_block, src_packet);
    fix_steps(dst_block, dst_packet);
    return true;
}

bool fill_conf_vld(const memory_desc_wrapper &src,
        const memory_desc_wrapper &dst, int scale_mask, size_t memlimit_bytes,
        size_t optimal_burst_bytes, vectorize_last_dim_t &cfg, int &vect_dim,
        int &vect_size, dim_t *blocks) {

    const dimensions_t src_dims = query_dims_and_blocks(src);
    const dimensions_t dst_dims = query_dims_and_blocks(dst);
    dimensions_t src_packet, src_block, dst_packet, dst_block;
    bool success = split_into_blocks_and_packets(16, memlimit_bytes,
            optimal_burst_bytes, src.data_type_size(), dst.data_type_size(),
            src_dims, dst_dims, src_packet, src_block, dst_packet, dst_block);
    if (!success) { return false; }
    // Below: unpack std vectors into POD arrays

    cfg.src_vect_limit = (int)check_burst_length(src_packet, src_packet);
    cfg.dst_vect_limit = (int)check_burst_length(dst_packet, dst_packet);

    // reset packet and loop
    for (size_t i = 0; i < LOOP_NEST_LEVEL; i++) {
        cfg.src_vct[i].blk_size = 1;
        cfg.dst_vct[i].blk_size = 1;
        cfg.src_blk[i].blk_size = 1;
        cfg.dst_blk[i].blk_size = 1;
        cfg.src_vct[i].step_size = 1;
        cfg.dst_vct[i].step_size = 1;
        cfg.src_blk[i].step_size = 1;
        cfg.dst_blk[i].step_size = 1;
        cfg.src_vct[i].dim_idx = 0;
        cfg.dst_vct[i].dim_idx = 0;
        cfg.src_blk[i].dim_idx = 0;
        cfg.dst_blk[i].dim_idx = 0;
    }
    cfg.src_vct[0].blk_size = src_packet[0].size;
    cfg.src_vct[0].dim_idx = src_packet[0].idx;
    cfg.dst_vct[0].blk_size = dst_packet[0].size;
    cfg.dst_vct[0].dim_idx = dst_packet[0].idx;
    for (size_t i = 0; i < src_packet.size(); i++) {
        cfg.src_vct[i].dim_idx = src_packet[i].idx;
        cfg.src_vct[i].blk_size = src_packet[i].size;
        cfg.src_vct[i].step_size = src_packet[i].step;
    }
    for (size_t i = 0; i < dst_packet.size(); i++) {
        cfg.dst_vct[i].dim_idx = dst_packet[i].idx;
        cfg.dst_vct[i].blk_size = dst_packet[i].size;
        cfg.dst_vct[i].step_size = dst_packet[i].step;
    }

    // fill src's and dst's loop recipe
    for (size_t i = 0; i < src_block.size(); i++) {
        cfg.src_blk[i].dim_idx = src_block[i].idx;
        cfg.src_blk[i].blk_size = src_block[i].size;
        cfg.src_blk[i].step_size = src_block[i].step;
    }
    for (size_t i = 0; i < dst_block.size(); i++) {
        cfg.dst_blk[i].dim_idx = dst_block[i].idx;
        cfg.dst_blk[i].blk_size = dst_block[i].size;
        cfg.dst_blk[i].step_size = dst_block[i].step;
    }
    cfg.vector_dim = dst_packet[0].idx;
    vect_dim = dst_packet[0].idx;
    vect_size = 16;
    for (int i = 0; i < LOOP_NEST_LEVEL; i++) {
        if (cfg.dst_blk[i].blk_size != 1) {
            blocks[cfg.dst_blk[i].dim_idx] *= cfg.dst_blk[i].blk_size;
        }
    }
    // Multiply by 16 the size of the dimension that will be vectorized.
    // This is workaround for 2 dispatcher problems:
    // - it doesn't allow vectorization of dims that are not divisible by 16
    // - vectorized dim's coordinate returned in openCL side is rounded to 16
    // Here we multiply the dim-to-be-vectorized by 16 and it immediately
    // solves 1st issue; we declare larger block on dim-to-be-vectorized to
    // prevent dispatcher from spawning too many work items over this enlarged
    // dim; and later on openCL side we'll divide this dim's coordinate by 16
    // to get fine-grained coordinates not rounded to 16.
    cfg.rescale_coeff = 16;

    for (int i = 0; i < LOOP_NEST_LEVEL; i++) {
        auto db = cfg.dst_vct[i];
        blocks[db.dim_idx] *= db.blk_size;
    }

    return true;
}

#define IRRELEVANT_DIM 7

bool is_relevant(const dimension_t dim, int idxa, int idxb) {
    return (dim.idx == idxa || dim.idx == idxb) && (dim.size > 1);
}

dimension_t get_idx(
        const dimensions_t &dims, int idxa, int idxb, size_t &iter) {
    dimension_t ret;
    if (iter >= dims.size()) {
        ret.idx = IRRELEVANT_DIM;
        ret.size = 0;
        return ret;
    }
    // found relevant dim? advance iterator and return that dim
    if (is_relevant(dims[iter], idxa, idxb)) {
        ret = dims[iter];
        iter++;
        return ret;
    } else {
        // found irrelevant dim? consume all subsequent irrelevant dims
        ret.idx = IRRELEVANT_DIM;
        ret.size = 0;
        iter++;
        while (iter < dims.size() && !is_relevant(dims[iter], idxa, idxb)) {
            iter++;
        }
        return ret;
    }
}

bool doesnt_fit(dimension_t a, dimension_t b) {
    return (a.idx != b.idx) || (a.size != b.size);
}

// Mode of finding adjacent dimensions to be combined together.
// adjacent = Every instance of given dim must be a neighbor of the other dim,
//            this combination doesn't block any other valid combination.
// any      = Dims don't have to be adjacent. Any such combination may prevent
//            other 'any' combination, they are mutually exclusive. Need to
//            write logic to choose best from such possible combinations.
//            NOT IMPLEMENTED.
enum mode_t { adjacent, any };

bool check_adjacent(int dim, int &prev_dim, bool &adjacent) {
    if (prev_dim != IRRELEVANT_DIM && dim != IRRELEVANT_DIM) {
        adjacent = true;
    } else if (adjacent == true && dim == IRRELEVANT_DIM) {
        adjacent = false;
    } else if (adjacent == false && prev_dim != IRRELEVANT_DIM
            && dim == IRRELEVANT_DIM) {
        return false;
    }
    return true;
}

bool compare_formats(const dimensions_t &dims_a, const dimensions_t &dims_b,
        int dim1, int dim2, mode_t mode) {
    size_t iter_a = 0;
    size_t iter_b = 0;
    dimension_t dim_a;
    dimension_t dim_b;
    int prev_dim_a = IRRELEVANT_DIM;
    int prev_dim_b = IRRELEVANT_DIM;
    bool a_adj = false;
    bool b_adj = false;
    do {
        dim_a = get_idx(dims_a, dim1, dim2, iter_a);
        dim_b = get_idx(dims_b, dim1, dim2, iter_b);
        if (doesnt_fit(dim_a, dim_b)) { return false; }
        if (mode == mode_t::adjacent) {
            if (!check_adjacent(dim_a.idx, prev_dim_a, a_adj)) { return false; }
            if (!check_adjacent(dim_b.idx, prev_dim_b, b_adj)) { return false; }
        }
        prev_dim_a = dim_a.idx;
        prev_dim_b = dim_b.idx;
    } while (iter_a < dims_a.size() && iter_b < dims_b.size());
    if (mode == mode_t::adjacent) {
        if (!check_adjacent(IRRELEVANT_DIM, prev_dim_a, a_adj)) {
            return false;
        }
        if (!check_adjacent(IRRELEVANT_DIM, prev_dim_b, b_adj)) {
            return false;
        }
    }
    return true;
}

// Takes memory descriptor and information which two dimension to combine.
// Returns new memory descriptor with those two dimensions combined into one.
// TODO: it doesn't understand padded_offsets
void combine_dims(memory_desc_t &new_a, int dim1, int dim2) {
    // shorter names
    dnnl_dim_t *str = &new_a.format_desc.blocking.strides[0];
    dnnl_dim_t *blks = &new_a.format_desc.blocking.inner_blks[0];
    dnnl_dim_t *idxs = &new_a.format_desc.blocking.inner_idxs[0];

    for (int i = 0; i < new_a.ndims; i++) {
        if (i == dim1) {
            new_a.dims[i] = new_a.padded_dims[i];
        } else if (i == dim2) {
            new_a.dims[dim1] *= new_a.padded_dims[i];
        } else if (i < dim2) {
            new_a.dims[i] = new_a.dims[i];
        } else {
            new_a.dims[i - 1] = new_a.dims[i];
        }
    }
    for (int i = new_a.ndims - 1; i < MAX_NDIMS; i++) {
        new_a.dims[i] = 0;
    }

    for (int i = 0; i < new_a.ndims; i++) {
        if (i == dim1) {
            new_a.padded_dims[i] = new_a.padded_dims[i];
        } else if (i == dim2) {
            new_a.padded_dims[dim1] *= new_a.padded_dims[i];
        } else if (i < dim2) {
            new_a.padded_dims[i] = new_a.padded_dims[i];
        } else {
            new_a.padded_dims[i - 1] = new_a.padded_dims[i];
        }
    }
    for (int i = new_a.ndims - 1; i < MAX_NDIMS; i++) {
        new_a.padded_dims[i] = 0;
    }

    for (int i = 0; i < new_a.ndims; i++) {
        if (i == dim1) {
            str[i] = std::min(str[dim1], str[dim2]);
        } else if (i == dim2) { // already accounted for in the line above
        } else if (i < dim2) { // do nothing: str[i] = str[i];
        } else {
            str[i - 1] = str[i];
        }
    }
    for (int i = new_a.ndims - 1; i < MAX_NDIMS; i++) {
        str[i] = 0;
    }

    bool last_is_combined = false;
    int out_iter = 0;
    for (int i = 0; i < new_a.format_desc.blocking.inner_nblks; i++) {
        if (idxs[i] == dim1 || idxs[i] == dim2) {
            if (last_is_combined) {
                blks[out_iter - 1] *= blks[i];
            } else {
                last_is_combined = true;
                blks[out_iter] = blks[i];
                idxs[out_iter] = dim1;
                out_iter++;
            }
        } else {
            last_is_combined = false;
            blks[out_iter] = blks[i];
            idxs[out_iter] = (idxs[i] > dim2 ? idxs[i] - 1 : idxs[i]);
            out_iter++;
        }
    }
    new_a.format_desc.blocking.inner_nblks = out_iter;
    new_a.ndims = new_a.ndims - 1;
}

int shift_mask(int &mask, int dim) {
    int ret = 0;
    for (int i = 0; i < dim; i++) {
        ret |= (mask & (1 << i));
    }
    for (int i = dim; i < MAX_NDIMS - 1; i++) {
        ret |= ((mask & (1 << i)) >> 1);
    }
    return ret;
}

// Returns bit mask with each bit corresponding to dimension that has strides
// larger than what could be expected according to dimensions sizes.
int is_padded_by_strides(const memory_desc_wrapper &md) {
    int ret = 0;
    const int nblks = md.blocking_desc().inner_nblks;
    const int ndims = md.ndims();

    std::vector<stride_t> strides(ndims);
    for (int d = 0; d < ndims; ++d) {
        strides[d].idx = d;
        strides[d].stride = md.blocking_desc().strides[d];
        strides[d].size = md.padded_dims()[d];
    }
    std::sort(strides.begin(), strides.end(), stride_less);
    int blocks_size = 1;
    for (int i = 0; i < nblks; i++) {
        blocks_size *= md.blocking_desc().inner_blks[i];
    }
    int expected_stride = blocks_size;
    for (size_t i = 0; i < strides.size(); i++) {
        if (strides[i].stride != expected_stride) {
            ret |= 0x1 << strides[i].idx;
            expected_stride = strides[i].stride * strides[i].size;
        } else {
            expected_stride *= strides[i].size;
        }
    }
    return ret;
}

// Given description of src and dst tensors, try to find a subset of dimensions
// that are not reordered between src and dst, then try to treat that subset as
// a single dmension.
// Example: abcd -> acdb 128x3x7x7; the CD dimensions stay in the same order
// relative to each other, so operation can be redefined as abz -> azb 128x3x49
bool try_combine_dims(
        memory_desc_t &a, memory_desc_t &b, int &mask, mode_t mode) {
    const dimensions_t a_dims = query_dims_and_blocks(a);
    const dimensions_t b_dims = query_dims_and_blocks(b);
    int padded_strides = is_padded_by_strides(a) | is_padded_by_strides(b);

    if (mask != 0) return false;

    for (int i = 0; i < a.ndims; i++) {
        for (int j = i + 1; j < b.ndims; j++) {
            if ((mask >> i) & 0x1) { continue; }
            if ((mask >> j) & 0x1) { continue; }
            if ((padded_strides >> i) & 0x1) { continue; }
            if ((padded_strides >> j) & 0x1) { continue; }
            if (i == j) { continue; }
            if (compare_formats(a_dims, b_dims, i, j, mode)) {
                combine_dims(a, i, j);
                combine_dims(b, i, j);
                mask = shift_mask(mask, j);
                return true;
            }
        }
    }
    return false;
}

status_t generic_reorder_t::pd_t::init_conf(engine_t *engine) {
    using namespace format_tag;

    size_t memlimit_bytes;
    size_t optimal_burst_bytes;

    conf.scale_quant = attr()->output_scales_.mask_ != 0;
    conf.scale_mask = conf.scale_quant ? attr()->output_scales_.mask_ : 0;
    conf.scales_num = conf.scale_quant ? attr()->output_scales_.count_ : 0;

    memory_desc_t new_a;
    memory_desc_t new_b;
    memcpy(&new_a, src_md(), sizeof(new_a));
    memcpy(&new_b, dst_md(), sizeof(new_b));

    while (try_combine_dims(new_a, new_b, conf.scale_mask, adjacent)) {}

    const memory_desc_wrapper src_mdw(new_a);
    const memory_desc_wrapper dst_mdw(new_b);
    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);

    status_t status = status::success;

    const auto &padded_dims = dst_mdw.padded_dims();
    conf.with_sum_ab = (alpha() != 1.f || beta() != 0.f);
    conf.with_sum_a = conf.with_sum_ab && beta() == 0.f;
    conf.has_padding = !src_mdw.is_dense() || !dst_mdw.is_dense();
    conf.ndims = src_mdw.ndims();
    conf.nelems = utils::array_product(padded_dims, conf.ndims);

    conf.sub_group_size = 1;
    if (conf.nelems == 0) { return status::success; }
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    // Theoretically, bursts should be at least big enough to span whole
    // cache line and bigger bursts should give better perf as long as
    // local mem capacity is not exceeded. However, all tests show that
    // burst size 64 gives best performance regardless of cache line size.
    memlimit_bytes = 2048;
    optimal_burst_bytes = 64;

    dim_t blocks[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
    int vect_size = 1;
    int vect_dim = 0;

    if (!fill_conf_vld(src_mdw, dst_mdw, conf.scale_mask, memlimit_bytes,
                optimal_burst_bytes, conf.aux_data.vld, vect_dim, vect_size,
                &blocks[0])) {
        return status::unimplemented;
    }

    conf.sub_group_size = vect_size;

    conf.dispatch = compute_engine->create_dispatch(dst_mdw.md_);

    for (int i = 0; i < MAX_NDIMS; ++i) {
        auto dim_str = utils::format("D%d", i);
        if (i < dst_mdw.ndims()) {
            uint64_t dim = padded_dims[i];
            // Pad vectorized dim to multiple of block size (to make sure that
            // enough work items will be generated to have only full subgroups,
            // no fractions) then multiply it by vector size (to work around
            // dispatcher's limitation that vectorized dim must be divisible by
            // vector size).
            if (i == vect_dim) {
                dim = utils::rnd_up(dim, blocks[i]);
                dim *= 16;
            }
            conf.dispatch.define_dim(dim_str, i, dim, blocks[i]);
        } else {
            conf.dispatch.define_dim(dim_str, 1);
        }
    }
    if (vect_size != 1) {
        const auto dim_str = utils::format("D%d", vect_dim);
        conf.dispatch.vectorize_dim(dim_str, vect_size);
    }

    conf.dispatch.generate();

    return status;
}

status_t generic_reorder_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    using namespace format_tag;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    if (conf.nelems == 0) return status::success;

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.add_option("-cl-std=CL2.0");

    if (conf.with_sum_a)
        kernel_ctx.define_int("WITH_SUM_A", 1);
    else if (conf.with_sum_ab)
        kernel_ctx.define_int("WITH_SUM_AB", 1);

    if (conf.scale_quant) {
        kernel_ctx.define_int("SCALE_QUANT", 1);
        kernel_ctx.define_int("SCALE_MASK", conf.scale_mask);
    }

    def_dispatch(kernel_ctx, conf.dispatch);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    kernel_ctx.define_int("PAD_FILL_ZERO", conf.has_padding);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    kernel_ctx.define_int("GENERIC_REORDER", 1);
    kernel_ctx.define_int("VECT_DIM", conf.aux_data.vld.vector_dim);
    kernel_ctx.define_int("VECT_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("RESCALE_COEFF", conf.aux_data.vld.rescale_coeff);
    kernel_ctx.define_int("LIMIT_SSGID", conf.aux_data.vld.src_vect_limit);
    kernel_ctx.define_int("LIMIT_DSGID", conf.aux_data.vld.dst_vect_limit);
    auto r = conf.dispatch.nd_range();
    auto *lr = r.local_range();
    kernel_ctx.define_int(
            "SG_PER_WG", (lr[0] * lr[1] * lr[2]) / conf.sub_group_size);
    int i = 0;
    int cache_dim[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
    while (i < LOOP_NEST_LEVEL) {
        cache_dim[conf.aux_data.vld.dst_vct[i].dim_idx]
                *= conf.aux_data.vld.dst_vct[i].blk_size;
        cache_dim[conf.aux_data.vld.dst_blk[i].dim_idx]
                *= conf.aux_data.vld.dst_blk[i].blk_size;
        kernel_ctx.define_int(std::string("S_BLK_SIZE_") + std::to_string(i),
                conf.aux_data.vld.src_blk[i].blk_size);
        kernel_ctx.define_int(std::string("S_BLK_STEP_") + std::to_string(i),
                conf.aux_data.vld.src_blk[i].step_size);
        kernel_ctx.define_int(std::string("S_BLK_IDX_") + std::to_string(i),
                conf.aux_data.vld.src_blk[i].dim_idx);
        kernel_ctx.define_int(std::string("D_BLK_SIZE_") + std::to_string(i),
                conf.aux_data.vld.dst_blk[i].blk_size);
        kernel_ctx.define_int(std::string("D_BLK_STEP_") + std::to_string(i),
                conf.aux_data.vld.dst_blk[i].step_size);
        kernel_ctx.define_int(std::string("D_BLK_IDX_") + std::to_string(i),
                conf.aux_data.vld.dst_blk[i].dim_idx);
        i++;
    }
    int cache_stride = 1;
    for (int i = 0; i < MAX_NDIMS; i++) {
        kernel_ctx.define_int(
                std::string("CACHE_STRIDE_") + std::to_string(i), cache_stride);
        cache_stride *= cache_dim[i];
    }
    int s_size_so_far = 1;
    int d_size_so_far = 1;
    for (int i = 0; i < LOOP_NEST_LEVEL; i++) {
        auto s = conf.aux_data.vld.src_vct[i];
        auto d = conf.aux_data.vld.dst_vct[i];
        kernel_ctx.define_int(
                std::string("S_MOD_") + std::to_string(i), s.blk_size);
        kernel_ctx.define_int(
                std::string("S_DIV_") + std::to_string(i), s_size_so_far);
        kernel_ctx.define_int(
                std::string("S_MUL_") + std::to_string(i), s.step_size);
        kernel_ctx.define_int(
                std::string("S_IDX_") + std::to_string(i), s.dim_idx);
        kernel_ctx.define_int(
                std::string("D_MOD_") + std::to_string(i), d.blk_size);
        kernel_ctx.define_int(
                std::string("D_DIV_") + std::to_string(i), d_size_so_far);
        kernel_ctx.define_int(
                std::string("D_MUL_") + std::to_string(i), d.step_size);
        kernel_ctx.define_int(
                std::string("D_IDX_") + std::to_string(i), d.dim_idx);

        s_size_so_far *= s.blk_size;
        d_size_so_far *= d.blk_size;
    }

    kernel_ctx.print_options();
    return status::success;
}

void generic_reorder_t::pd_t::init_scratchpad() {
    if (conf.scales_num > 0) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_reorder_scales,
                conf.scales_num, sizeof(float), OCL_BUFFER_ALIGNMENT);
    }
}

status_t generic_reorder_t::execute(const exec_ctx_t &ctx) const {

    status_t status = status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_TO);
    CHECK(status);

    const auto &conf = pd()->conf;
    if (conf.nelems == 0) { return status::success; }
    const float alpha = pd()->alpha();
    const float beta = pd()->beta();

    std::unique_ptr<memory_storage_t> scales;
    if (conf.scale_quant) {
        scales = ctx.get_scratchpad_grantor().get_memory_storage(
                key_reorder_scales);

        void *tmp_ptr = nullptr;
        status = scales->map_data(&tmp_ptr, ctx.stream(),
                sizeof(float) * pd()->attr()->output_scales_.count_);
        if (status != status::success) { return status; }
        utils::array_copy((float *)tmp_ptr,
                pd()->attr()->output_scales_.scales_,
                pd()->attr()->output_scales_.count_);
        status = scales->unmap_data(tmp_ptr, ctx.stream());
        if (status != status::success) { return status; }
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.set(2, alpha);
    arg_list.set(3, beta);
    arg_list.set(4, scales ? *scales : memory_storage_t::empty_storage());

    auto nd_range = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
