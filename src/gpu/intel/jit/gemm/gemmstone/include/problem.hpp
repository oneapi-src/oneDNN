/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GEMMSTONE_GUARD_PROBLEM_HPP
#define GEMMSTONE_GUARD_PROBLEM_HPP

#include "config.hpp"

#include <bitset>
#include "gpu/intel/gpu_post_ops.hpp"

#include "internal/ngen_includes.hpp"
#include "internal/utils.hpp"
#include "type.hpp"

#include "internal/namespace_start.hxx"

// Matrix layouts in memory.
enum class MatrixLayout : uint8_t {
    N = 0,  Nontranspose = N,
    T = 1,  Transpose = T,
    Pc = 2, PackedColumns = Pc,
    Pr = 3, PackedRows = Pr
};

static inline bool isPacked(MatrixLayout l) {
    return (l == MatrixLayout::PackedRows) || (l == MatrixLayout::PackedColumns);
}

static inline bool isColMajor(MatrixLayout l) {
    return (l == MatrixLayout::N || l == MatrixLayout::Pc);
}

static inline MatrixLayout transposeLayout(MatrixLayout l) {
    return static_cast<MatrixLayout>(static_cast<uint8_t>(l) ^ 0x1);
}

inline char layoutChar(MatrixLayout layout)
{
    switch (layout) {
        case MatrixLayout::N:  return 'N';
        case MatrixLayout::T:  return 'T';
        case MatrixLayout::Pc: return 'A';
        case MatrixLayout::Pr: return 'B';
        default:               return '?';
    }
}

struct MatrixAddressing {
    MatrixLayout layout;            // Layout type (N/T/Pr/Pc)
    uint8_t packSize = 0;           // # of elements in a packed row/column for packed layouts.
    uint8_t crosspack = 1;          // Crosspack for packed layouts.
    uint8_t alignment;              // Alignment for all addresses, offsets, and leading dimensions.
    uint8_t panelLength = 0;        // Length of the panel for packed layouts = #cols/rows for Pc/Pr respectively.
    bool needA64 = false;
    uint16_t tileR = 0, tileC = 0;  // Tiling (0 if none) for packed layouts.

    void setAlignment(int align) { alignment = static_cast<uint8_t>(sanitizeAlign(align)); }
    int defaultAlignment(Type T) const {
        return sanitizeAlign((isPacked(layout) ? (packSize * crosspack) : 1) * T);
    }

    void transpose() {
        layout = transposeLayout(layout);
        std::swap(tileR, tileC);
    }

private:
    static int sanitizeAlign(int align) { return std::min(128, largest_pow2_divisor(align)); }
};

// Information on a scalar argument (e.g. alpha/beta).
class Scalar {
public:
    enum ScalarType {Fixed, Variable, Pointer, RealPointer};

private:
    int value;
    ScalarType type;

public:
    Scalar() : Scalar(Variable) {}
    explicit Scalar(ScalarType type_) : value(0), type(type_) {}
    explicit Scalar(int value_) : value(value_), type(Fixed) {}

    Scalar &operator=(int value_) { type = Fixed; value = value_; return *this; }
    Scalar &operator=(ScalarType type_) { type = type_; value = 0; return *this; }

    template <typename U> bool operator==(U value_) const   { return fixed() && (value == value_); }
    bool operator==(ScalarType type_) const                 { return (type == type_); }
    template <typename U> bool operator!=(U value_) const   { return !operator==(value_); }

    operator int() const {
        if (!fixed()) throw std::runtime_error("Scalar is not fixed.");
        return value;
    }
    operator double() const { return int(*this); }

    bool fixed() const { return (type == Fixed); }
    bool pointer() const { return (type == Pointer) || (type == RealPointer); }
    ScalarType getType() const { return type; }
};

// Types of updates for GEMM kernels.
enum class UpdateType {
    Full,
};

// A/B offset mode.
enum class ABOffset {
    None,       // No A/B offsets.
    Calc,       // Calculate A/B row/column sums in kernel.
    Load,       // Use precalculated row/column sums.
};

// C offset mode.
enum class COffset {
    None,       // No C offsets.
    Post,       // C offset after all other updates.
    Pre,        // C offset before all other updates (bias).
};

// Batch mode.
enum class BatchMode {
    None, Strided, Nonstrided, Variable
};

// Binary operations.
enum class BinaryOp {
    Add, Sub, Mul, Div,
    Min, Max,
    Prelu,
    ScaleSub    /* internal use only */
};

// Problem parameters shared between kernel types.
struct CommonProblem {
    bool nonuniformWGs = false;                 // Support nonuniform workgroups?
    bool gtpinSupport = false;                  // Support GT-Pin?
};

// GEMM kernel problem description.
struct GEMMProblem : public CommonProblem {
    Type Ta, Tb, Tc, Ts;                            // Types for A/B/C/scalars in registers.
    Type Ta_ext, Tb_ext, Tc_ext;                    // Types for A/B/C data in memory.
    Type Tao, Tbo, Tco;                             // Types for A/B/C offsets.
    Type Ta_scale, Tb_scale;                        // Types for A/B scales.

    Scalar alpha, beta;                             // Scaling factors for A*B and C, respectively.
    MatrixAddressing A, B, C;                       // Addressing information for A/B/C matrices.
    MatrixAddressing AO, BO, CO;                    // Addressing information for A/B/C offsets (if 2D).
    MatrixAddressing A_scale, B_scale;              // Addressing information for A/B/C scales (if 2D).
    MatrixAddressing sroundSeed;              // Addressing information for A/B/C scales (if 2D).
    bool checkBeta0 = true;                         // If true, check for beta = 0 and handle specially.
    ABOffset aOffset = ABOffset::None;              // A/B offset modes.
    ABOffset bOffset = ABOffset::None;              //
    int aoPtrDims = -1, boPtrDims = -1;             // A/B offset dimensionality (-1: none; 0: scalar; 1: vector, 2: matrix)
    bool aScale2D = false, bScale2D = false;        // A/B 2D scaling.
    int aqGroupM = 0, aqGroupK = 0;                 // Group sizes for A quantization parameters (offsets and scales)
    int bqGroupN = 0, bqGroupK = 0;                 // Group sizes for B quantization parameters (offsets and scales)
    COffset cOffset = COffset::None;                // C offset mode.
    BatchMode batch = BatchMode::None;              // Batch mode.
    int batchDims = 0;                              // # of batch dimensions (strided batch only).
    bool sumA = false, sumB = false;                // If true, calculate A row sums/B column sums and store in CO.
    bool postOpFwd = true;                          // Eltwise parameters
    bool cStochasticRound = false;

    gpu_post_ops_t postOps;                         // Fused post operations to apply
    std::bitset<post_ops_t::post_ops_limit> binaryRow;      // Binary op broadcasts row data if false
    std::bitset<post_ops_t::post_ops_limit> binaryCol;      // Binary op broadcasts column data if false
    std::bitset<post_ops_t::post_ops_limit> binaryBatch;    // Binary op broadcasts in batch dimension if false
    std::bitset<post_ops_t::post_ops_limit> binaryTrans;    // Used to compute GEMMProblem::binary

    // The following data is derived from the postOps and does not need
    //   to be considered for equality/hashing purposes.
    std::vector<MatrixAddressing> binary;                   // Binary postop data
    std::vector<Type> Tbinary;                              // Binary types

    bool hasPostOp() const { return !postOps.empty(); }
    bool hasNonSum1PostOp() const {
        for (const auto &e : postOps)
            if (!e.is_sum()) return true;
        return false;
    }
    bool hasBinaryPostOp() const {
        for (auto &e : postOps)
            if (e.is_binary()) return true;
        return false;
    }
    bool hasSum1PostOpAtEnd() const {
        return !postOps.empty() && postOps.back().is_sum();
    }
    void removeFinalSumPostOp() {
        if (hasSum1PostOpAtEnd())
            postOps.pop_back();
    }

    bool beta0() const   { return (beta  ==  0); }
    bool beta1() const   { return (beta  ==  1); }
    bool alpha1() const  { return (alpha ==  1); }
    bool alphaM1() const { return (alpha == -1); }

    bool needsTsConvert() const {
        if (!(alpha1() || alphaM1())) return true;
        if (!(beta0() || beta1())) return true;
        if (beta1() && !Tc_ext.isSubsetOf(Tc)) return true;
        if ((Tc == Type::s32 || Tc == Type::u32) && Tc_ext == Type::bf16) return true;
        if (hasNonSum1PostOp()) return true;
        return false;
    }

    bool isIGEMM() const {
        return (Ta.real().isInt8() && Tb.real().isInt8() && Tc.real().paddedSize() == 4);
    }

    bool gemmt() const { return false; }
    bool backward() const { return false; }

    bool needsASums() const { return sumA || (bOffset == ABOffset::Calc && !earlyDequantizeB()); }
    bool needsBSums() const { return sumB || (aOffset == ABOffset::Calc && !earlyDequantizeA()); }
    bool usesCO() const { return (cOffset != COffset::None) || sumA || sumB; }
    bool allowMatrixOffset() const { return (cOffset == COffset::Pre); }

    bool quantized2DA() const { return (aoPtrDims == 2) || aScale2D; }
    bool quantized2DB() const { return (boPtrDims == 2) || bScale2D; }

    bool downconvertAScales() const { return Ta == Type::f16 && Ta_scale == Type::f32; }
    bool downconvertBScales() const { return Tb == Type::f16 && Tb_scale == Type::f32; }

    bool earlyDequantizeA() const {
        return (aOffset == ABOffset::Calc && Tao.asSigned().isSubsetOf(Ta)
                    && (Ta_ext.bits() < Ta.bits() || Ta.isFP()))
            || (aScale2D && (Ta_scale.isSubsetOf(Ta) || downconvertAScales()));
    }
    bool earlyDequantizeB() const {
        return (bOffset == ABOffset::Calc && Tbo.asSigned().isSubsetOf(Tb)
                    && (Tb_ext.bits() < Tb.bits() || Tb.isFP()))
            || (bScale2D && (Tb_scale.isSubsetOf(Tb) || downconvertBScales()));
    }

    Type Tc_compute() const {
        if (Ta.isInteger() && Tb.isInteger() && Tc == Type::f32)
            return Type::s32;
        else
            return Tc;
    }

    inline void autoTypeConversions(ngen::HW hw, bool systolicAvailable);
    void transpose();

    std::string toString() const;
    std::string scalarsToString() const;

    /* Serialization for kernel cache. */
    void serialize(serialized_data_t &s) const
    {
        s.append(Ta, Tb, Tc, Ts);
        s.append(Ta_ext, Tb_ext, Tc_ext);
        s.append(Tao, Tbo, Tco);
        s.append(Ta_scale, Tb_scale);
        s.append(alpha);
        s.append(beta);
        s.append(A, B, C, CO);
        s.append(checkBeta0);
        s.append(aOffset, bOffset);
        s.append(aoPtrDims, boPtrDims);
        s.append(aScale2D, bScale2D);
        s.append(aqGroupK, bqGroupK);
        s.append(cOffset);
        s.append(batch);
        s.append(batchDims);
        s.append(sumA, sumB);
        s.append(postOpFwd);
        s.append(postOps);
        s.append(binaryRow);
        s.append(binaryCol);
        s.append(binaryBatch);
        s.append(binaryTrans);
        s.append(cStochasticRound);
    }
};


// Apply automatic internal type conversions to a problem.
void GEMMProblem::autoTypeConversions(ngen::HW hw, bool systolicAvailable)
{
    using namespace ngen;

    // Weights decompression
    if ((Ta.isInt8() || Ta.isInt4()) && Tb.isFP() && Tc.isFP()) Ta = Tb;
    if ((Tb.isInt8() || Tb.isInt4()) && Ta.isFP() && Tc.isFP()) Tb = Ta;

    if (Ta == Ta_ext.asSigned()) Ta = Ta_ext;
    if (Tb == Tb_ext.asSigned()) Tb = Tb_ext;

        if (Ta.isF8()) Ta = Type::f16;
        if (Tb.isF8()) Tb = Type::f16;

    if (hw > HW::Gen9 && !systolicAvailable && Tc == Type::f32) {
        if (Ta == Type::f16) Ta = Type::f32;
        if (Tb == Type::f16) Tb = Type::f32;
    }

    if (hw < HW::XeHP || (hw > HW::XeHP && !systolicAvailable)) {
        if (Ta == Type::bf16) Ta = Type::f32;
        if (Tb == Type::bf16) Tb = Type::f32;
    }

    if (hw == HW::Gen11) {
        if (one_of(Ta, Type::s8, Type::u8)) Ta = Type::s16;
        if (one_of(Tb, Type::s8, Type::u8)) Tb = Type::s16;
    }
}

#include "internal/namespace_end.hxx"

#endif /* header guard */
