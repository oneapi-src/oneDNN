/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "kernel_info.hpp"

#include "benchstone_common.hpp"

namespace {
void parseQuantization(std::istream &s, char &type, std::array<int32_t, 2> &group, char &precision, bool *precalc = nullptr)
{
    char eat;

    type = 's';
    precision = 'I';
    if (s.peek() == '[') {
        s >> eat >> type;
        if (precalc) *precalc = std::isupper(type);
        type = std::tolower(type);
        if (type == 'g') {
            s >> group[0] >> eat >> group[1];
            group[0] = std::max(group[0], 1);
            group[1] = std::max(group[1], 1);
        }
        s >> eat >> precision >> eat;
    }
}

void inputPrecision(std::istream &s, char &p1, char &p2, int &components)
{
    components = 1;
    s >> p1;
    if (p1 == '[') {
        char rbracket;
        s >> p1 >> p2;
        if (s.peek() == 'x') {
            char x;
            s >> x >> components;
        }
        s >> rbracket;
        if (rbracket != ']') throw std::runtime_error("Error parsing precisions, expected ]");
    } else
        p2 = p1;
}

void inputPrecision(std::istream &s, char &p1, char &p2)
{
    int components;
    inputPrecision(s, p1, p2, components);
    if (components != 1) throw std::runtime_error("Error parsing precisions: components not supported here.");
}

void inputLayout(std::istream &s, char precision, char &layout, int &crosspack, int &tileR, int &tileC, int &packSize, int &align)
{
    crosspack = 1;
    tileR = 0;
    tileC = 0;
    align = charToType(precision).paddedSize();
    packSize = 0;
    s >> layout;
    if (std::isdigit(s.peek()))
        s >> crosspack;
    if (s.peek() == '#') {
        char eat;
        s >> eat >> tileR >> eat >> tileC;
    }
    if (s.peek() == '%') {
        char eat;
        s >> eat >> packSize;
    }
    if (s.peek() == '@') {
        char eat;
        s >> eat >> align;
    }
}

double inputRealScalar(std::istream &s)
{
    bool negative = false;

    s >> std::ws;
    if (s.peek() == 'n') {
        std::string nan;
        s >> nan;
        return qnan<double>();
    } else if (s.peek() == '-') {
        char eat;
        s >> eat;
        if (s.peek() == ' ' || s.eof())
            return qnan<double>();
        negative = true;
    }

    double scalar;
    s >> scalar;
    if (negative)
        scalar *= -1;
    return scalar;
}

void processProblemInfo(Level3KernelInfo &info)
{
    // Interpret problem string.
    info.readCheck = false;
    info.strongReadCheck = false;
    info.hpl = false;
    info.superkernel = false;
    info.scaleA = false;
    info.aqGroup.fill(0);
    info.scaleB = false;
    info.bqGroup.fill(0);
    info.aOffset = false;
    info.bOffset = false;
    info.cOffset = false;
    info.cBias = false;
    info.sumA = info.sumB = false;
    info.fused = false;
    info.batch = false;
    info.systolic = false;

    std::stringstream ss(info.problem);
    std::string word;
    char eat;

    ss >> word;
    if (word == "rc")         { info.readCheck = true;   ss >> word; }
    if (word == "super")      { info.superkernel = true; ss >> word; }
    if (word == "fused")      { info.fused = true;       ss >> word; }

    if (word.substr(0,5) == "batch") {
        if (word == "batchnv")
            info.batch = 'v';
        else
            info.batch = word[5];
        ss >> word;
    }

    if (word.substr(0,6) == "offset") {
        std::stringstream ws(word.substr(6));
        while (!ws.eof()) {
            eat = 0, ws >> eat;
            if (eat == 'a')
                parseQuantization(ws, info.aOffset, info.aqGroup, info.precisionAO);
            else if (eat == 'b')
                parseQuantization(ws, info.bOffset, info.bqGroup, info.precisionBO);
            else if (eat == 'c')
                info.cOffset = true;
            else break;
        }
        ss >> word;
    }

    if (word.find("scale") == 0) {
        std::stringstream ws(word.substr(5));
        while (!ws.eof()) {
            eat = 0, ws >> eat;
            if (eat == 'a') {
                info.scaleA = true;
                parseQuantization(ws, eat, info.aqGroup, info.precisionAScale);
            } else if (eat == 'b') {
                info.scaleB = true;
                parseQuantization(ws, eat, info.bqGroup, info.precisionBScale);
            } else break;
        }
        ss >> word;
    }
    if (word == "suma") {
        info.sumA = true;
        ss >> word;
    } else if (word == "sumb") {
        info.sumB = true;
        ss >> word;
    }
    if (word == "bias") {
        info.cBias = true;
        ss >> word;
    }
    if (word == "hpl") {
        info.hpl = true;
        ss >> word;
    }

    info.left = false;
    info.lower = false;
    info.unit = false;
    info.checkUnit = false;
    info.hermitian = false;
    info.triangular = false;
    info.trsm = (word.substr(0, 4) == "trsm");
    info.trmm = (word.substr(0, 4) == "trmm");
    info.symm = (word.substr(0, 4) == "symm");
    info.hemm = (word.substr(0, 4) == "hemm");

    if (info.trsm || info.trmm) {
        int cn = 4;
        info.inplace = true;
        if (info.trsm && (word[cn] == '3')) {
            info.inplace = false;
            cn++;
        }
        info.left      = (word[cn++] == 'l');
        info.lower     = (word[cn++] == 'l');
        info.unit      = (word[cn] == 'u');
        info.checkUnit = (word[cn] == '\0');
    } else if (info.symm || info.hemm) {
        info.left      = (word[4] == 'l');
        info.lower     = (word[5] == 'l');
    } else {
        info.triangular = (word.length() > 4);
        if (info.triangular) {
            info.lower = (word[4] == 'l');
            info.hermitian = (word.length() > 5) && (word[5] == 'h');
        }
    }

    inputPrecision(ss, info.precisionA, info.precisionAInt, info.A_components);
    inputPrecision(ss, info.precisionB, info.precisionBInt, info.B_components);
    inputPrecision(ss, info.precisionCInt, info.precisionC);
    if (isalpha(ss.peek()))
        ss >> info.precisionS;
    else
        info.precisionS = info.precisionCInt;

    inputLayout(ss, info.precisionA, info.layoutA, info.crosspackA, info.A_tileR, info.A_tileC, info.packSizeA, info.alignA);
    inputLayout(ss, info.precisionB, info.layoutB, info.crosspackB, info.B_tileR, info.B_tileC, info.packSizeB, info.alignB);
    inputLayout(ss, info.precisionC, info.layoutC, info.crosspackC, info.C_tileR, info.C_tileC, info.packSizeC, info.alignC);

    if (info.packSizeA == 0) info.packSizeA = info.unroll[LoopM];
    if (info.packSizeB == 0) info.packSizeB = info.unroll[LoopN];

    info.alpha = inputRealScalar(ss);
    info.beta  = inputRealScalar(ss);

    std::stringstream ssExtra(info.extra);
    while (!ssExtra.eof()) {
        ssExtra >> std::ws >> word;
        if (word == "sys")
            info.systolic = true;
        else if (word == "track")
            info.tracking = true;
        else if (word == "src")
            info.strongReadCheck = true;
        else if (word == "alocal")
            info.localA = true;
        else if (word == "blocal")
            info.localB = true;
        else if (word == "clocal")
            info.localC = true;
        else if (word == "rx")
            info.relaxedAccumulation = true;
        else if (word == "a2d")
            info.a2D = true;
        else if (word == "b2d")
            info.b2D = true;
        else if (word == "c2d")
            info.c2D = true;
    }

    if (info.needsAGroupSums() && info.aqGroup[1] == 0)
        info.aqGroup = {1, info.bqGroup[0]};
    if (info.needsBGroupSums() && info.bqGroup[0] == 0)
        info.bqGroup = {info.aqGroup[1], 1};
}

void processProblemInfo(CopyKernelInfo &info)
{
    info.conjugate = false;
    info.lower = false;
    info.unit = false;
    info.trsm = false;
    info.symmetric = false;
    info.hermitian = false;
    info.triangular = false;
    info.sum = false;
    info.crosspackS = info.crosspackD = 1;
    info.alignS = info.alignD = 0;
    info.targetWG = 1;

    std::stringstream ss(info.problem);
    std::string word;
    char eat;

    ss >> word;

    if (word == "rc") { info.readCheck = true; ss >> word; }

    for (auto c = word.begin() + 4; c < word.end(); c++) {
        switch (*c) {
            case 'c': info.conjugate = true; break;
            case 's':
                if (info.triangular)
                    info.trsm = true;
                else
                    info.symmetric = true;
                break;
            case 'h': info.hermitian = true; break;
            case 'l': info.lower = true;     break;
            case 'n': info.unit = false;     break;
            case 'u':
                if (info.triangular)
                    info.unit = true;
                else
                    info.lower = false;
                break;
            case 't': info.triangular = true; break;
        }
    }

    ss >> info.precisionS >> info.precisionD;
    if (ss.peek() == '+') {
        ss >> eat >> info.precisionSum;
        info.sum = true;
    }
    inputLayout(ss, info.precisionS, info.layoutS, info.crosspackS, info.S_tileR, info.S_tileC, info.packSize, info.alignS);
    inputLayout(ss, info.precisionD, info.layoutD, info.crosspackD, info.D_tileR, info.D_tileC, info.packSize, info.alignD);

    ss >> info.packSize;

    info.alpha = inputRealScalar(ss);
}
} // anonymous namespace

KernelInfo::KernelInfo(const std::string& info_str) {
    std::stringstream s(info_str);
    while (!s.eof()) {
        char comma;
        s >> std::ws;
        if (s.peek() == ',')
            s >> comma >> std::ws;

        auto dinfo = gemmstone::CommonDriverInfo::undump(s);

        bool isCopy = (dinfo.problem.find("copy") != std::string::npos);

        if (isCopy) {
            copy.resize(copy.size() + 1);
            auto &last = copy[copy.size() - 1];
            static_cast<gemmstone::CommonDriverInfo &>(last) = dinfo;
            processProblemInfo(last);
        } else {
            l3.resize(l3.size() + 1);
            auto &last = l3[l3.size() - 1];
            static_cast<gemmstone::CommonDriverInfo &>(last) = dinfo;
            processProblemInfo(last);
        }

        s >> std::ws;
    }

    if (isL3() && l3[0].fused) {
        // Get external GEMM problem.
        auto &l3Info = l3[0];
        l3Info.layoutA = '?';
        l3Info.layoutB = '?';
        if (!copy[0].empty()) {
            l3Info.precisionA = copy[0].precisionS;
            l3Info.crosspackA = copy[0].crosspackS;
            l3Info.alignA     = copy[0].alignS;
            l3Info.A_tileR    = copy[0].S_tileR;
            l3Info.A_tileC    = copy[0].S_tileC;
        }
        if (!copy[2].empty()) {
            l3Info.precisionB = copy[2].precisionS;
            l3Info.crosspackB = copy[2].crosspackS;
            l3Info.alignB     = copy[2].alignS;
            l3Info.B_tileR    = copy[2].S_tileR;
            l3Info.B_tileC    = copy[2].S_tileC;
        }
    }
}
