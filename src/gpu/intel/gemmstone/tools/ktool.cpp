/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gemmstone/config.hpp"
#include "tools/benchstone/benchstone.hpp"
#include "tools/benchstone/kernel_info.hpp"
#include "tools/device_utils.hpp"
#include "tools/frontend.hpp"
#include "tools/utils.hpp"
#include "tools/getopt_compat.hpp"

#define KERNEL_CATALOG_MUTABLE
#include "gemmstone/kernel_catalog.hpp"
#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/kernel_evaluator.hpp"

#ifndef GEMMSTONE_WITH_MKL
#include "mini_mkl.hpp"
#elif __has_include("mkl/mkl.h")
#include "mkl/mkl.h"
#else
#include "mkl.h"
#endif

#ifdef GEMMSTONE_WITH_L0_RUNTIME
#include <level_zero/ze_api.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <climits>
#include <unistd.h>
#include <sys/wait.h>
#endif

#include <regex>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

using namespace gemmstone;

bool Sorted = true;
bool CSVOutput = false;
bool EvalEffective = false;

bool DumpData = false;
bool ShowFitData = false;
bool Verbose = false;
bool UseL0 = false;

std::unordered_map<std::string, double> DefaultModelParams = {
    {"priority", 0.}
};
char FitModel = '\0';
char Arch = '\0';
std::string Family;
int Stepping = 0;

int EUCount = 0;
size_t LLCSize = 0;
double NominalFrequency = 0.;
bool SystolicAvailable = true;

bool CatalogUpdated = false;
bool ReadKernelFromArgv = false;

// Directory containing the default kernel catalogs, next to the executable.
static const std::filesystem::path &catalogDbDir()
{
    static const std::filesystem::path dir = []() -> std::filesystem::path {
        std::string exePath;
#ifdef _WIN32
        char buf[MAX_PATH];
        DWORD len = GetModuleFileNameA(nullptr, buf, sizeof(buf));
        if (len > 0 && len < sizeof(buf))
            exePath.assign(buf, len);
#else
        char buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (len > 0 && (size_t)len < sizeof(buf) - 1)
            exePath.assign(buf, len);
#endif
        if (exePath.empty())
            return "selector/db";
        return std::filesystem::path(exePath).parent_path() / "selector" / "db";
    }();
    return dir;
}

// Resolve a user-supplied catalog argument. Bare filenames (no path separator)
// are resolved against the catalog db dir next to the executable, so that
// `--catalog symm.db` refers to the build directory catalog regardless of cwd.
// Paths containing '/' or '\\' (including absolute paths and './foo') are used verbatim.
static std::string resolveCatalogArg(const char *arg)
{
    if (!arg || !*arg) return {};
    std::filesystem::path p(arg);
    if (p.has_parent_path()) return arg;
    return (catalogDbDir() / p).string();
}

std::string CatalogPath = (catalogDbDir() / "kernel.db").string();
// True when CatalogPath points into the build-tree <exe_dir>/selector/db/ dir
// (i.e. default, or `-c` with a bare filename like `symm.db`).
bool CatalogInDbDir = true;
std::string OtherCatalogPath;
std::string BaseDir = "./";
std::string UpgradeType = "";
#ifndef _WIN32
std::string ModifyCommand = "";
#endif
std::string TestAlpha = "";
std::string TestBeta = "";
bool TestTightLD = false;
bool NoDerived = false;
bool TestVariations = false;
std::string DerivedType = "";

std::string LastFrontendInput = "";
bool BatchActive = false;

enum class TestKind { Generation, Accuracy, Compare };
#ifndef _WIN32
std::string CompareOldGenerator = "ofrontend-stable";
#endif

enum class Mode {
    None,
    Init,
    Fit,
    Add,
    Clone,
    Delete,
#ifndef _WIN32
    Modify,
#endif
    Refit,
    Reinfo,
    Replace,
    List,
    ListIDs,
    Evaluate,
    Restrict,
    Append,
    Extract,
    TestGeneration,
    Test,
#ifndef _WIN32
    TestCompare,
#endif
    Upgrade,
    Downgrade,
    RemainderReport,
    LDReport,
    WipeConstant,
    WipeMemory,
} CurrentMode = Mode::None;

const char *ModeArgument = "";

SizeParams EvalSize;
bool EvalCheckPerformance;

kcatalog::Catalog Catalog;

#define FLAT_CATALOG_SENTINEL "/*@kcatalog@*/"

struct RestrictionSpec {
    bool clear = false;
    bool negative = false;
    enum {
        None, Tag, Allowed, Accept, Steppings
    } mode = None;
    char tag = '\0';
    int mnkMin[3] = {-1, -1, -1};
    int mnkMax[3] = {-1, -1, -1};
    int steppingMin = -1, steppingMax = -1;
};

std::vector<RestrictionSpec> Restrictions;

struct DataPoint {
    int64_t m, n, k, batch;
    int64_t lda = -1, ldb = -1, ldc = -1;
    double nominal, effective;
};

using Evaluator = std::function<double(int64_t, int64_t, int64_t)>;

/******************/
/* Error Handling */
/******************/

#define VT_HIGHLIGHT "\x1B[1m"
#define VT_RESET "\x1B[0m"

[[noreturn]] void fatal(const char *str)
{
    fprintf(stderr, "Fatal error: %s\n", str);
    exit(1);
}

[[noreturn]] void fatal(const char *child, const char *str2)
{
    bool feFailed = strstr(child, "frontend") != nullptr;

    fprintf(stderr, "Fatal error: %s %s\n", child, str2);

    if (!LastFrontendInput.empty() && feFailed)
        fprintf(stderr, "Last kernel: %s\n", LastFrontendInput.c_str());

    exit(1);
}

#ifdef GEMMSTONE_WITH_L0_RUNTIME
void handle(ze_result_t result)
{
    if (result != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "Fatal error: Level Zero returned 0x%x\n", (uint32_t) result);
        exit(result);
    }
}
#endif

/*************/
/* Utilities */
/*************/

template <typename T1, typename T2>
static inline auto divUp(T1 x, T2 y)
{
    return (x + y - 1) / y;
}

template <typename T1, typename T2>
static inline auto alignUp(T1 x, T2 y)
{
    return divUp(x, y) * y;
}

bool contains(const std::string &haystack, const std::string &needle)
{
    return (haystack.find(needle) != std::string::npos);
}

bool contains(const std::string &haystack, char needle)
{
    return (haystack.find(needle) != std::string::npos);
}

void remove(std::string &haystack, const std::string &needle)
{
    if (needle.empty()) return;

    auto needleChars = needle.length();
    while (true) {
        auto pos = haystack.rfind(needle);
        if (pos == std::string::npos) break;
        haystack.erase(pos, needleChars);
    }
}

bool isPackedChar(const kcatalog::string &layout)
{
    return one_of(layout[0], {'A', 'B'});
}

bool isLDRow(const kcatalog::string &layout)
{
    return one_of(layout[0], {'B', 'N'});
}

int64_t niceLD(const kcatalog::string &layout, double elementSize, int64_t r, int64_t c)
{
    int64_t x = isLDRow(layout) ? r : c;

    x *= elementSize;
    x = alignUp(x, 128);
    if (x % 4096 == 0)
        x += 128;
    x /= elementSize;

    return x;
}

void tableOutput(std::ostream &stream, const std::string &str, int gutter = 2)
{
    std::vector<int> stops;
    int column = 0;
    int tab = 0;

    for (auto c: str) {
        switch (c) {
            case '\t':
                if (tab >= int(stops.size()))
                    stops.resize(tab + 1);
                stops[tab] = std::max(stops[tab], column);
                tab++;
                column = 0;
                break;
            case '\n':
                column = tab = 0;
                break;
            default:
                column++;
                break;
        }
    }

    column = 0;
    for (auto &stop: stops) {
        column += stop + gutter;
        stop = column;
    }

    column = tab = 0;

    for (auto c: str) {
        switch (c) {
            case '\t': {
                int nspace = stops[tab++] - column;
                for (int i = 0; i < nspace; i++)
                    stream << ' ';
                column += nspace;
                break;
            }
            case '\n':
                column = tab = 0;
                stream << c;
                break;
            default:
                column++;
                stream << c;
                break;
        }
    }
}

#ifndef _WIN32

#define PIPE_READ 0
#define PIPE_WRITE 1

std::string outputFromProcess(const std::string &name, const std::vector<std::string> &args = {}, int fileNo = STDOUT_FILENO)
{
    auto fatalWithPerror = [](const char *str) -> void {
        fprintf(stderr, "Fatal error ");
        perror(str);
        exit(1);
    };

    // Create pipe from process.
    int stdoutPipe[2];

    if (pipe(stdoutPipe) < 0)
        fatalWithPerror("allocating pipe");

    // Launch.
    auto childPID = fork();
    if (childPID == 0) {
        if (dup2(stdoutPipe[PIPE_WRITE], fileNo) == -1)
            fatalWithPerror("duplicating fd");
        close(stdoutPipe[PIPE_READ]);
        close(stdoutPipe[PIPE_WRITE]);

        std::string path = BaseDir + name;
        std::vector<char *> argsC(args.size() + 1);
        argsC[0] = const_cast<char *>(name.c_str());
        for (size_t i = 0; i < args.size(); i++)
            argsC[i + 1] = const_cast<char *>(args[i].c_str());
        argsC.push_back(nullptr);

        (void) execv(path.c_str(), argsC.data());
        fatalWithPerror("launching child process");
    } else if (childPID < 0)
        fatal("could not fork");

    auto pipeOut = fdopen(stdoutPipe[PIPE_READ], "r");
    close(stdoutPipe[PIPE_WRITE]);

    // Wait for completion.
    int wstatus;
    if (waitpid(childPID, &wstatus, 0) == -1)
        fatalWithPerror("waiting for child process to complete");

    if (WIFEXITED(wstatus)) {
        if (WEXITSTATUS(wstatus) != 0)
            fatal(name.c_str(), "exited with non-zero return value");
    } else if (WIFSIGNALED(wstatus) && WTERMSIG(wstatus) == SIGINT)
        fatal("aborted");
    else
        fatal(name.c_str(), "terminated abnormally");

    // Nab output.
    char *output = nullptr;
    size_t osize = 0;
    (void) getdelim(&output, &osize, '\0', pipeOut);

    fclose(pipeOut);

    std::string result = output;
    free(output);

    return result;
}

#endif // !_WIN32

template <typename T>
void getenvInt(const char *name, T &var)
{
    auto env = getenv(name);
    if (env) var = T(std::atol(env));
}

double extPrecisionBytes(const kcatalog::string &precision, bool isC)
{
    char ext = precision.at(0);
    if (ext == '[')
        ext = precision.at(isC ? 2 : 1);
    auto T = charPrecision(ext);
    return double(T.paddedSize()) / double(T.perByte());
}

int defaultAlignment(char c)
{
    switch (std::toupper(c)) {
        case 'E':
        case 'F':
        case 'O': return 1;
        case 'W':
        case 'B':
        case 'H': return 2;
        case 'I':
        case 'J':
        case 'T':
        case 'S': return 4;
        case 'D':
        case 'C': return 8;
        case 'Z': return 16;
        default: fatal("unkown precision");
    }
}

int defaultAlignment(std::string &s)
{
    return defaultAlignment(s[0] == '[' ? s[1] : s[0]);
}

/*********************/
/* Catalog Handling */
/*********************/
void expect(std::istream &s, char c)
{
    char in;
    s >> std::ws;

    if (s.eof()) {
        fprintf(stderr, "Fatal error parsing: expected '%c', but found EOF\n", c);
        exit(1);
    }

    s >> in;

    if (in != c) {
        fprintf(stderr, "Fatal error parsing: expected '%c', but found '%c'\n", c, in);
        exit(1);
    }
}

void deserializeCatalog(std::istream &s, int &i)
{
    s >> i;
}

void deserializeCatalog(std::istream &s, char &c)
{
    expect(s, '\'');
    s >> c;
    expect(s, '\'');
}

void deserializeCatalog(std::istream &s, std::string &str)
{
    expect(s, '"');
    std::getline(s, str, '"');
}

template <typename T, int n>
void deserializeCatalog(std::istream &s, T (&array)[n])
{
    expect(s, '{');
    deserializeCatalog(s, array[0]);
    for (int i = 1; i < n; i++) {
        expect(s, ',');
        deserializeCatalog(s, array[i]);
    }
    expect(s, '}');
}

void deserializeCatalog(std::istream &s, kcatalog::Selector &sel)
{
    expect(s, '{');
    deserializeCatalog(s, sel.hw);
    expect(s, ',');
    deserializeCatalog(s, sel.kernelType);
    expect(s, ',');
    deserializeCatalog(s, sel.precisions);
    expect(s, ',');
    deserializeCatalog(s, sel.layouts);
    expect(s, '}');
}

void deserializeCatalog(std::istream &s, kcatalog::Restrictions &r)
{
    expect(s, '{');
    s >> r.steppingMin;
    expect(s, ',');
    s >> r.steppingMax;
    expect(s, ',');
    deserializeCatalog(s, r.acceptSizesMin);
    expect(s, ',');
    deserializeCatalog(s, r.acceptSizesMax);
    expect(s, ',');
    deserializeCatalog(s, r.allowedSizesMin);
    expect(s, ',');
    deserializeCatalog(s, r.allowedSizesMax);
    expect(s, ',');
    deserializeCatalog(s, r.alignment);
    expect(s, ',');
    deserializeCatalog(s, r.tags);
    expect(s, '}');
}

void deserializeCatalog(std::istream &s, kcatalog::Model &m)
{
    expect(s, '{');
    expect(s, '\'');
    s >> m.id;
    expect(s, '\'');
    expect(s, ',');
    s >> m.paramCount;
    if (m.paramCount < 0 || m.paramCount > kcatalog::MaxParamCount)
        fatal("invalid parameter count in catalog model");
    expect(s, ',');
    expect(s, '{');
    for (int i = 0; i < m.paramCount; i++) {
        if (i > 0)
            expect(s, ',');
        s >> m.params[i];
    }
    expect(s, '}');
    expect(s, '}');
}

void deserializeCatalog(std::istream &s, kcatalog::Entry &e)
{
    expect(s, '{');
    deserializeCatalog(s, e.selector);
    expect(s, ',');
    deserializeCatalog(s, e.restrictions);
    expect(s, ',');
    deserializeCatalog(s, e.strategy);
    expect(s, ',');
    e.driverInfo = CommonDriverInfo::undump(s, false);
    expect(s, ',');
    deserializeCatalog(s, e.model);
    expect(s, '}');
}

void deserializeCatalog(std::istream &s)
{
    std::string line;
    while (true) {
        if (!s) fatal("could not find sentinel at beginning of catalog");
        std::getline(s, line);
        if (line == FLAT_CATALOG_SENTINEL)
            break;
        Catalog.header += line;
        Catalog.header += '\n';
    }

    std::getline(s, line);

    while (true) {
        s >> std::ws;
        if (!s || s.eof())
            fatal("could not parse catalog");

        if (s.peek() == '}') break;

        if (!Catalog.entries.empty())
            expect(s, ',');

        kcatalog::Entry e;
        deserializeCatalog(s, e);

        Catalog.entries.push_back(std::move(e));
    }
    Catalog.entryCount = Catalog.entries.size();

    expect(s, '}');
    expect(s, ')');
    expect(s, ';');

    if (!std::is_sorted(Catalog.entries.begin(), Catalog.entries.end())) {
        fprintf(stderr, "Warning: catalog is not sorted; fixing.\n");
        std::stable_sort(Catalog.entries.begin(), Catalog.entries.end());
        CatalogUpdated = true;
    }
}

void serializeCatalog(std::ostream &s, int i)
{
    s << i;
}

void serializeCatalog(std::ostream &s, char c)
{
    s << '\'' << c << '\'';
}

void serializeCatalog(std::ostream &s, const std::string &str)
{
    s << '"' << str << '"';
}

template <typename T, int n>
void serializeCatalog(std::ostream &s, const T (&array)[n])
{
    s << '{';
    serializeCatalog(s, array[0]);
    for (int i = 1; i < n; i++) {
        s << ", ";
        serializeCatalog(s, array[i]);
    }
    s << '}';
}

void serializeCatalog(std::ostream &s, const kcatalog::Selector &sel)
{
    s << '{';
    serializeCatalog(s, sel.hw);
    s << ", ";
    serializeCatalog(s, sel.kernelType);
    s << ", ";
    serializeCatalog(s, sel.precisions);
    s << ", ";
    serializeCatalog(s, sel.layouts);
    s << "}";
}

void serializeCatalog(std::ostream &s, const kcatalog::Restrictions &r)
{
    s << "{";
    s << r.steppingMin << ", " << r.steppingMax << ", ";
    serializeCatalog(s, r.acceptSizesMin);
    s << ", ";
    serializeCatalog(s, r.acceptSizesMax);
    s << ", ";
    serializeCatalog(s, r.allowedSizesMin);
    s << ", ";
    serializeCatalog(s, r.allowedSizesMax);
    s << ", ";
    serializeCatalog(s, r.alignment);
    s << ", ";
    serializeCatalog(s, r.tags);
    s << '}';
}

void serializeCatalog(std::ostream &s, const kcatalog::Model &m)
{
    s << '{';
    serializeCatalog(s, m.id);
    s << ", " << m.paramCount << ", {";
    if (m.paramCount > 0) s << m.params[0];
    for (int i = 1; i < m.paramCount; i++)
        s << ", " << m.params[i];
    s << "}}";
}

void serializeCatalog(std::ostream &s, const kcatalog::Entry &e)
{
    s << '{';
    serializeCatalog(s, e.selector);
    s << ", ";
    serializeCatalog(s, e.restrictions);
    s << ", ";
    serializeCatalog(s, e.strategy);
    s << ", ";
    e.driverInfo.dump(s, false);
    s << ", ";
    serializeCatalog(s, e.model);
    s << "}";
}

void serializeCatalog(std::ostream &s)
{
    s << Catalog.header;
    s << FLAT_CATALOG_SENTINEL "\nauto _CATALOG_ = kcatalog::toArray({\n";

    bool first = true;
    for (const auto &e : Catalog.entries) {
        if (!first)
            s << ",\n";
        first = false;
        serializeCatalog(s, e);
    }

    s << "\n});\n";
}

void catalogUpdated()
{
    CatalogUpdated = true;
}

int addToCatalog(kcatalog::Entry &&e)
{
    auto pos = std::upper_bound(Catalog.entries.begin(), Catalog.entries.end(), e);
    pos = Catalog.entries.insert(pos, e);
    Catalog.entryCount++;
    catalogUpdated();
    return std::distance(Catalog.entries.begin(), pos);
}

void removeFromCatalog(int ientry)
{
    Catalog.entries.erase(Catalog.entries.begin() + ientry);
    Catalog.entryCount--;
    catalogUpdated();
}

void addDefaultRestrictions(kcatalog::Entry &entry)
{
    auto &tags = entry.restrictions.tags;

    if (!entry.driverInfo.support4GB[0]) tags += kcatalog::ReqNo64BitA;
    if (!entry.driverInfo.support4GB[1]) tags += kcatalog::ReqNo64BitB;
    if (!entry.driverInfo.support4GB[2]) tags += kcatalog::ReqNo64BitC;

    if (entry.driverInfo.kParallel())
        tags += kcatalog::ReqNoBatch;

    if (contains(entry.strategy, "sys"))
        tags += kcatalog::ReqSystolic;

    // Note we only get 'extra' on a freshly added entry, with information from frontend.
    // The catalog does not store 'extra'.
    if (contains(entry.driverInfo.extra, "a2d"))
        tags += kcatalog::ReqBlock2DA;
    if (contains(entry.driverInfo.extra, "b2d"))
        tags += kcatalog::ReqBlock2DB;
    if (contains(entry.driverInfo.extra, "c2d"))
        tags += kcatalog::ReqBlock2DC;

    if (entry.selector.hw == kcatalog::HWTagXeHPC) {
        if (contains(entry.selector.precisions[0], 'T') || contains(entry.selector.precisions[1], 'T')) {
            entry.restrictions.steppingMin = 5; /* PVC-XT B0 */
        }
    }

    if (entry.selector.hw == kcatalog::HWTagXeHPC) {
        bool noXe2 = false;
        if (contains(entry.restrictions.tags, kcatalog::ReqBlock2DA) && entry.selector.layouts[0][0] == 'N')
            noXe2 |= (int(entry.driverInfo.unroll[LoopM] * extPrecisionBytes(entry.selector.precisions[0], false)) % 16) != 0;
        if (contains(entry.restrictions.tags, kcatalog::ReqBlock2DB) && entry.selector.layouts[1][0] == 'T')
            noXe2 |= (int(entry.driverInfo.unroll[LoopN] * extPrecisionBytes(entry.selector.precisions[1], false)) % 16) != 0;
        if (noXe2)
            tags += kcatalog::ReqNoXe2Block2D;
    }

    auto wgTileM = entry.driverInfo.wgTile(LoopM);
    auto wgTileN = entry.driverInfo.wgTile(LoopN);

    std::sort(tags.begin(), tags.end());
}

void applyRestriction(kcatalog::Entry &entry, RestrictionSpec spec)
{
    switch (spec.mode) {
        case RestrictionSpec::Tag: {
            auto &tags = entry.restrictions.tags;
            for (auto citer = tags.begin(); citer < tags.end(); citer++) {
                if (std::toupper(*citer) == spec.tag) {
                    tags.erase(citer);
                    break;
                }
            }
            if (!spec.clear) {
                if (spec.negative)
                    tags += std::tolower(spec.tag);
                else
                    tags += spec.tag;
                std::sort(tags.begin(), tags.end());
            }
            break;
        }
        case RestrictionSpec::Accept:
            for (int dim = 0; dim < 3; dim++) {
                entry.restrictions.acceptSizesMin[dim] = spec.mnkMin[dim];
                entry.restrictions.acceptSizesMax[dim] = spec.mnkMax[dim];
            }
            break;
        case RestrictionSpec::Allowed:
            for (int dim = 0; dim < 3; dim++) {
                entry.restrictions.allowedSizesMin[dim] = spec.mnkMin[dim];
                entry.restrictions.allowedSizesMax[dim] = spec.mnkMax[dim];
            }
            break;
        case RestrictionSpec::Steppings:
            entry.restrictions.steppingMin = spec.steppingMin;
            entry.restrictions.steppingMax = spec.steppingMax;
            break;
        case RestrictionSpec::None: break;
    }
}

bool hasRestriction(kcatalog::Entry &entry, RestrictionSpec spec)
{
    auto restrictions0 = entry.restrictions;
    std::sort(restrictions0.tags.begin(), restrictions0.tags.end());  /* in case of unsorted tags */

    auto clone = entry;
    applyRestriction(clone, spec);
    return (clone.restrictions == restrictions0);
}

/*********************/
/* Catalog filtering */
/*********************/

std::string frontendInputString(const kcatalog::Entry &e,
                                const std::string &alpha = "1", const std::string &beta = "0",
                                const std::string &additions = "",
                                const kcatalog::Selector *selectorOverride = nullptr);

enum class Client {None, oneMKL, oneDNN};

class ExtendedEntryIterator {
public:
    explicit ExtendedEntryIterator(kcatalog::Catalog &catalog_, char arch_ = '\0') : catalog(catalog_), entryID(-1), arch(arch_) {
        current = begin = catalog_.entries.begin();
        end = catalog_.entries.end();
        findNextMatch();
    }

    ExtendedEntryIterator(kcatalog::Catalog &catalog_, Client client_, bool matchClient_, char arch_ = '\0') : catalog(catalog_), entryID(-1), arch(arch_), client(client_), matchClient(matchClient_) {
        current = begin = catalog_.entries.begin();
        end = catalog_.entries.end();
        findNextMatch();
    }

    ExtendedEntryIterator(kcatalog::Catalog &catalog_, const std::string &matchString_, char arch_ = '\0') : catalog(catalog_), entryID(-1), matchString(matchString_), arch(arch_) {
        current = begin = catalog_.entries.begin();
        end = catalog_.entries.end();
        findNextMatch();
    }

    ExtendedEntryIterator(kcatalog::Catalog &catalog_, int entryID_, char arch_ = '\0') : catalog(catalog_), entryID(entryID_), arch(arch_) {
        if (entryID < 0 || entryID >= catalog.entryCount)
            begin = current = end = catalog.entries.end();
        else {
            current = begin = catalog.entries.begin() + entryID;
            end = begin + 1;
        }
    }

    operator bool() const { return current < end; }

    ExtendedEntryIterator &operator++() {
        ++current;
        findNextMatch();
        return *this;
    }

    ExtendedEntryIterator operator++(int) {
        auto old = *this;
        operator++();
        return old;
    }

    kcatalog::Entry &operator*()  { return  *current; }
    kcatalog::Entry *operator->() { return &*current; }

    friend bool operator==(const ExtendedEntryIterator &i1, const ExtendedEntryIterator &i2) {
        return (i1.current == i2.current);
    }
    friend bool operator!=(const ExtendedEntryIterator &i1, const ExtendedEntryIterator &i2) {
        return !(i1 == i2);
    }

    int index() const {
        return std::distance(catalog.entries.begin(), current);
    }

protected:
    kcatalog::Catalog &catalog;
    int entryID = -1;
    std::string matchString;
    char arch = '\0';
    Client client = Client::None;
    bool matchClient = false;

    std::vector<kcatalog::Entry>::iterator begin, end, current;

    static bool isOneDNNPrecision(const std::string &s) {
        char c = s[0];
        return (c != 'C' && c != 'Z' && c != 'J' && !contains(s, 'x'));
    }

    static bool isOneDNN(const kcatalog::Entry &e) {
        using RT = kcatalog::RestrictionTags;

        for (int dim = 0; dim < 3; dim++)
            if (!isOneDNNPrecision(e.selector.precisions[dim]))
                return false;
        for (auto tag: {RT::ReqBatchN})
            if (contains(e.restrictions.tags, tag))
                return false;
        return true;
    }

    static bool isOneMKLPrecision(const std::string &s) {
        char c = s[0];
        return (c != 'F' && c != 'Q');
    }

    static bool isOneMKL(const kcatalog::Entry &e) {
        using RT = kcatalog::RestrictionTags;

        if (e.selector.precisions[0] != e.selector.precisions[1])
            return false;
        for (int dim = 0; dim < 3; dim++)
            if (!isOneMKLPrecision(e.selector.precisions[dim]))
                return false;
        for (auto tag: {RT::ReqBatchMultiDim, RT::ReqOffsetMultiDim, RT::ReqSumA, RT::ReqSumB})
            if (contains(e.restrictions.tags, tag))
                return false;
        return true;
    }

    void findNextMatch() {
        for (; current < end; current++) {
            if (arch && current->selector.hw != arch)
                continue;
            if (!matchString.empty() && !contains(frontendInputString(*current), matchString))
                continue;
            if (client == Client::oneDNN)
                if (isOneDNN(*current) != matchClient)
                    continue;
            if (client == Client::oneMKL)
                if (isOneMKL(*current) != matchClient)
                    continue;
            break;
        }
    }
};

/************/
/* GPU info */
/************/

#ifdef GEMMSTONE_WITH_L0_RUNTIME
ze_device_handle_t l0Device()
{
    static ze_device_handle_t dev = nullptr;

    if (!dev) {
        handle(zeInit(ZE_INIT_FLAG_GPU_ONLY));

        uint32_t driverCount = 0;
        handle(zeDriverGet(&driverCount, nullptr));

        std::vector<ze_driver_handle_t> allDrivers(driverCount);
        std::vector<ze_device_handle_t> allDevices;

        handle(zeDriverGet(&driverCount, allDrivers.data()));

        for (uint32_t i = 0; i < driverCount; i++) {
            uint32_t deviceCount = 0;
            handle(zeDeviceGet(allDrivers[i], &deviceCount, nullptr));

            allDevices.resize(deviceCount);
            handle(zeDeviceGet(allDrivers[i], &deviceCount, allDevices.data()));

            dev = allDevices[0];
            break;
        }
    }

    if (!dev) fatal("could not find Level Zero GPU device");

    return dev;
}

const ze_device_properties_t &l0DeviceProperties()
{
    static ze_device_properties_t props;
    static bool gotProps = false;

    if (!gotProps) {
        auto dev = l0Device();

        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        props.pNext = nullptr;

        handle(zeDeviceGetProperties(dev, &props));
        gotProps = true;
    }

    return props;
}

const std::vector<ze_device_cache_properties_t> &l0CacheProperties()
{
    static std::vector<ze_device_cache_properties_t> props;
    static bool gotProps = false;

    if (!gotProps) {
        auto dev = l0Device();

        uint32_t count = 0;
        handle(zeDeviceGetCacheProperties(dev, &count, nullptr));

        props.resize(count);
        for (auto &prop: props) {
            prop.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
            prop.pNext = nullptr;
        }

        handle(zeDeviceGetCacheProperties(dev, &count, props.data()));
        gotProps = true;
    }

    return props;
}
#endif // GEMMSTONE_WITH_L0_RUNTIME

bool aubCaptureEnabled()
{
    int csr = 0;
    if (auto e = getenv("SetCommandStreamReceiver")) csr = atoi(e);

    return csr & 1;
}

const char *tagToGen(char c, bool longNames = false)
{
    using namespace kcatalog;
    switch (c) {
        case HWTagGen12LP: return longNames ? "Gen12LP" : "12LP";
        case HWTagXeHPG:   return longNames ? "XeHPG"   : "12p7";
        case HWTagXeHPC:   return longNames ? "XeHPC"   : "12p8";
        case HWTagXe2:     return "Xe2";
        case HWTagXe3:     return "Xe3";
        case HWTagXe3p:    return "Xe3p";

        default: fatal("unknown architecture");
    }
    return nullptr;
}

char hwToTag(ngen::HW hw)
{
    using namespace kcatalog;
    switch (hw) {
        case ngen::HW::Gen12LP: return HWTagGen12LP;
        case ngen::HW::XeHPG:  return HWTagXeHPG;
        case ngen::HW::XeHPC:  return HWTagXeHPC;
        case ngen::HW::Xe2:    return HWTagXe2;
        case ngen::HW::Xe3:    return HWTagXe3;
        case ngen::HW::Xe3p:   return HWTagXe3p;
        default:                return '?';
    }
}

char genToTag(const std::string &s, bool okIfUnknown = false)
{
    // Try parseGen for standard gen names (12LP, 12HP, 12p7, 12p8, Xe2, ...).
    char tag = hwToTag(parseGen(s));
    if (tag != '?') return tag;

    if (okIfUnknown) return '?';
    fatal("unknown architecture");
    return 0;
}

char hostArch(bool okIfUnknown = false)
{
    static char savedArch;

    if (!savedArch) {
        savedArch = hwToTag(ngen::getCore(detectDevice(false).family));
        if (savedArch == '?' && !okIfUnknown)
            fatal("host architecture could not be detected");
    }

    return savedArch;
}

char arch(bool okIfUnknown = false)
{
    return Arch ? Arch : hostArch(okIfUnknown);
}

int stepping();

// Device to generate kernels for. Without -A this is the host GPU; with it,
// kernels are generated for the requested target (and optional family), which
// is what makes catalog entries created on one machine valid on another.
// Arch/Family are fixed for the run, so detect once and cache.
const ngen::Product &targetProduct()
{
    static ngen::Product product = [] {
        if (!Arch)
            return detectDevice(true);

        ngen::Product p = {};
        p.family = Family.empty()
                ? ngen::genericProductFamily(parseGen(tagToGen(arch(), true)))
                : parseFamily(Family);
        p.stepping = stepping();
        p.type = ngen::getPlatformType(p.family);
        return p;
    }();
    return product;
}

// Verify that the target architecture matches the host GPU.
// Call before any code path that will execute a generated kernel.
void requireNativeArch(char entryArch)
{
    if (entryArch != hostArch()) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "kernel architecture (%s) does not match host GPU (%s)."
                 "\n         Remove -A/--arch to auto-detect, or use a generation-only mode (-g).",
                 tagToGen(entryArch, true), tagToGen(hostArch(), true));
        fatal(msg);
    }
}

int defaultStepping(char arch)
{
    switch (arch) {
        case kcatalog::HWTagXeHPC: return 7; /* PVC-XT B4 */
        default:                   return 0;
    }
}

int hostStepping()
{
    static int savedStepping = -1;

    if (savedStepping < 0) {
        auto env = getenv("STEPPING");
        if (env)
            savedStepping = atoi(env);
        else
            savedStepping = defaultStepping(hostArch(true));
    }

    return savedStepping;
}

int stepping()
{
    return Stepping ? Stepping : hostStepping();
}

int euCount()
{
    if (EUCount <= 0) {
#ifdef GEMMSTONE_WITH_L0_RUNTIME
        if (UseL0) {
            ze_eu_count_ext_t euCountExt = {};
            euCountExt.stype = ZE_STRUCTURE_TYPE_EU_COUNT_EXT;

            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            props.pNext = &euCountExt;

            handle(zeDeviceGetProperties(l0Device(), &props));
            EUCount = euCountExt.numTotalEUs;
        } else
#endif
        {
            EUCount = getDeviceEUCount();
        }
        if (Verbose)
            fprintf(stderr, "Note: detected %d EUs.\n", EUCount);
    }
    return EUCount;
}

double nominalFrequency()
{
    if (NominalFrequency <= 0) {
        unsigned clockRateMHz = 0;
#ifdef GEMMSTONE_WITH_L0_RUNTIME
        if (UseL0) {
            auto &props = l0DeviceProperties();
            clockRateMHz = props.coreClockRate;
            NominalFrequency = clockRateMHz * 1e6;
        } else
#endif
        {
            clockRateMHz = getDeviceMHz();
            NominalFrequency = clockRateMHz * 1e6;
        }
        if (NominalFrequency == 0 && simulation() && (arch(true) == kcatalog::HWTagXe3p)) {
            NominalFrequency = 1.6e9;
            if (Verbose)
                fprintf(stderr, "Note: assuming %uMHz nominal frequency.\n", unsigned(NominalFrequency * 1e-6));
        } else {
            if (NominalFrequency == 0)
                fatal("could not detect frequency. Use --frequency=<MHz> to manually specify frequency.");
            if (Verbose)
                fprintf(stderr, "Note: detected %uMHz nominal frequency.\n", clockRateMHz);
        }
    }
    return NominalFrequency;
}

size_t llcSize()
{
    const auto MB = 1048576;
    if (LLCSize <= 0) {
#ifdef GEMMSTONE_WITH_L0_RUNTIME
        if (UseL0) {
            auto props = l0CacheProperties();
            for (auto &prop: props)
                LLCSize = std::max(LLCSize, prop.cacheSize);
        } else
#endif
        {
            cl_device_id dev = findDevice();
            cl_ulong cacheSize = 0;
            clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                            sizeof(cacheSize), &cacheSize, nullptr);
            LLCSize = cacheSize;
        }
        if (LLCSize < MB)
            LLCSize = 0;    /* ignore value if too low */
        if (Verbose && LLCSize > 0)
            fprintf(stderr, "Note: detected %uMB last-level cache.\n", unsigned(LLCSize / MB));
    }

    if (LLCSize <= 0) {
        switch (arch()) {
            case kcatalog::HWTagGen12LP: LLCSize = 12 * MB; break;   // assume DG1
            case kcatalog::HWTagXeHPG:   LLCSize = (euCount() > 256) ? 16 * MB :
                                                   (euCount() > 128) ?  8 * MB : 4 * MB; break;
            case kcatalog::HWTagXeHPC:   LLCSize = 256 * MB; break;  // huge
            case kcatalog::HWTagXe2:     LLCSize = 8 * MB; break;    // LNL guess; FIXME: get correct value
            case kcatalog::HWTagXe3:     LLCSize = 8 * MB; break;    // TBD
            case kcatalog::HWTagXe3p:    LLCSize = 192 * MB; break;

            default: fatal("unknown architecture");
        }
        if (Verbose)
            fprintf(stderr, "Note: assuming %uMB last-level cache.\n", unsigned(LLCSize / MB));
    }
    return LLCSize;
}

/*******************/
/* Data Collection */
/*******************/

bool cleanUpEffectiveData(std::vector<DataPoint> &data, double minGoodFraction = 0.8)
{
    double maxFreqDrop = 1.0f;

    int total = int(data.size()), good = 0;
    for (const auto &dp: data) {
        if (dp.effective > 0.) {
            good++;
            maxFreqDrop = std::max(maxFreqDrop, dp.nominal / dp.effective);
        }
    }

    if (good == 0)
        fatal("unable to measure GPU frequency");
    else if (good == total)
        return true;

    /* If there wasn't much frequency throttling, pretend there was none for data
       points without frequency information. */
    for (auto &dp: data)
        if (dp.effective == 0.)
            dp.effective = dp.nominal;

    return (good >= total * minGoodFraction) && (maxFreqDrop <= 1.05);
}

// Run benchstone to collect data.
// On entry, data holds m/n/k values.
// On exit, nominal/effective times will be filled in.
namespace {

benchstone::Driver &benchstoneDriver()
{
    static std::unique_ptr<benchstone::Driver> driver;
    if (!driver) {
        auto backend =
#ifdef GEMMSTONE_WITH_L0_RUNTIME
                       UseL0 ? benchstone::RuntimeBackend::LevelZero :
#endif
                       benchstone::RuntimeBackend::OpenCL;
        driver = benchstone::createDriver(backend);
        driver->Quiet = true;
        driver->init();
        driver->enableMetrics();
    }
    return *driver;
}

TestResult safeBenchstoneTest(benchstone::Driver &driver, KernelInfo &info,
                                     TestMode mode, uint32_t b, int32_t m, int32_t n,
                                     int32_t k, int32_t lda, int32_t ldb, int32_t ldc)
{
    try {
        return driver.test(info, mode, b, m, n, k, lda, ldb, ldc,
                           false, false, false, 'F', false, false);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        TestResult result;
        result.markAccuracy(TestAccuracy::RuntimeError);
        return result;
    }
}

} // namespace

void collectData1(const frontend::Result &res, std::vector<DataPoint> &data, bool noBlock = false, bool eval_mode = false)
{
    auto &bench = benchstoneDriver();
    if (bench.isKernelLoaded())
        bench.finishKernel();

    bench.Progress = false;
    bench.TimingOnly = true;
    bench.NoBlocking = noBlock;
    bench.TestBeta0 = false;
    bench.FixWGK = false;
    if (!bench.isSimulation()) {
        bench.TimeMaxReps = 100;
        bench.RelativeMeanStdDevGoal = 0.001;
    }
    if (eval_mode) {
        bench.TestBeta0 = true;
        bench.FixWGK = true;
    }

    if (res.binary.empty())
        fatal("Could not load kernel");

    KernelInfo info(res.info);
    if (!bench.initKernel(info, res.binary))
        fatal("Could not initialize kernel");

    int64_t maxM = 1, maxN = 1, maxK = 1, maxBatch = 1;
    int64_t single_elems = 0;
    for (const auto &dp : data) {
        maxM = std::max(maxM, dp.m);
        maxN = std::max(maxN, dp.n);
        maxK = std::max(maxK, dp.k);
        int64_t batch = info.isBatched() ? dp.batch : 1;
        maxBatch = std::max(maxBatch, batch);
        single_elems += batch * (dp.m * dp.n + dp.m * dp.k + dp.n * dp.k);
    }

    // Use series mode if it decreases the total amount of elements required to initialize
    int64_t series_elems = maxBatch * (maxM * maxN + maxM * maxK + maxN * maxK);
    bool series = series_elems < single_elems;
    if (series) {
        (void) safeBenchstoneTest(bench, info, TestMode::BeginSeries,
                                  maxBatch, maxM, maxN, maxK, -1, -1, -1);
    }

    for (auto &dp : data) {
        int32_t lda = dp.lda;
        int32_t ldb = dp.ldb;
        int32_t ldc = dp.ldc;
        if (TestTightLD && lda < 0 && ldb < 0 && ldc < 0)
            lda = ldb = ldc = 0;

        auto mode = series ? TestMode::TimeOneOfSeries : TestMode::Time;
        auto result = safeBenchstoneTest(bench, info, mode, dp.batch,
                                         dp.m, dp.n, dp.k, lda, ldb, ldc);
        if (result.accuracy == TestAccuracy::RuntimeError)
            fatal("benchstone runtime error during benchmark");

        dp.nominal = result.time;
        dp.effective = result.effTime;
    }

    if (series) {
        (void) safeBenchstoneTest(bench, info, TestMode::EndSeries,
                                  maxBatch, maxM, maxN, maxK, -1, -1, -1);
    }

    bench.finishKernel();

    // Optionally dump collected data to disk.
    if (DumpData) {
        static int fnum = 0;
        std::ofstream dumpfile(std::string("ktool_data") + std::to_string(fnum) + ".csv");
        if (BatchActive)
            dumpfile << "batch, ";
        dumpfile << "m, n, k, nominal time, effective time\n";
        for (const auto &dp: data) {
            if (BatchActive)
                dumpfile << dp.batch << ", ";
            dumpfile << dp.m << ", " << dp.n << ", " << dp.k << ", " << dp.nominal << ", " << dp.effective << std::endl;
        }
        fnum++;
    }

}

void collectData(const frontend::Result &res, std::vector<DataPoint> &data, bool noBlock = false, bool needEffective = false, bool eval_mode = false)
{
    if (data.empty()) return;
    if (aubCaptureEnabled()) fatal("Cannot run benchmarks, aub capture is enabled.");

    collectData1(res, data, noBlock, eval_mode);

    if (!needEffective) return;

    if (simulation()) {
        for (auto &dp: data)
            dp.effective = dp.nominal;
        return;
    }

    // Check if we have enough frequency data, and remove data points that are missing frequency information.
    auto ok = cleanUpEffectiveData(data);

    // If not enough frequency data, available, try again.
    if (!ok) {
        collectData1(res, data, noBlock, eval_mode);
        ok = cleanUpEffectiveData(data);
        if (!ok)
            fatal("unable to collect enough frequency data.");
    }
}

void verboseFit(const char *params, const char *method, const std::vector<DataPoint> &data,
                const double *values, int nvalues, double fitQuality = -1.)
{
    if (!Verbose) return;

    fprintf(stderr, "  %-12s %-26s =", params, method);
    for (int i = 0; i < nvalues; i++)
        fprintf(stderr, " %12.6g", values[i]);

    if (fitQuality >= 0.)
        fprintf(stderr, "  (relerr=%.1f%%)", fitQuality);
    fprintf(stderr, "\n");
}

void verboseFit(const char *params, const char *method, const std::vector<DataPoint> &data,
                double v0, double fitQuality = -1.)
{
    verboseFit(params, method, data, &v0, 1, fitQuality);
}

void verboseFit(const char *params, const char *method, const std::vector<DataPoint> &data,
                double v0, double v1, double fitQuality)
{
    double vals[2] = {v0, v1};
    verboseFit(params, method, data, vals, 2, fitQuality);
}

// Compute relative RMSE (%) of a fit: 100 * sqrt(mean(((y - yhat) / y)^2)).
// Skips data points where y is near zero.
// When ShowFitData is set, also prints per-data-point details.
double relativeRMSE(const std::vector<DataPoint> &data,
                    const std::function<double(const DataPoint &)> &predict,
                    bool effective,
                    const Evaluator &eval = nullptr)
{
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;
    double sumRelSq = 0.;
    int count = 0;

    if (ShowFitData) {
        if (eval)
            fprintf(stderr, "    %8s %8s %8s %12s %12s %12s %11s\n",
                    "m", "n", "k", "x", "y", "predicted", "relerr%");
        else
            fprintf(stderr, "    %8s %8s %8s %12s %12s %11s\n",
                    "m", "n", "k", "y", "predicted", "relerr%");
    }

    for (auto &dp : data) {
        double y = dp.*fit;
        double yhat = predict(dp);
        double x = eval ? eval(dp.m, dp.n, dp.k) : 0.;
        bool ignore = (x == 0.) && (std::abs(yhat) < 1e-20);
        double rel = (std::abs(y) < 1e-20 || ignore) ? 0. : (y - yhat) / y;

        if (ShowFitData) {
            if (eval)
                fprintf(stderr, "    %8" PRId64 " %8" PRId64 " %8" PRId64 " %12.6g %12.6g %12.6g %+10.1f%%\n",
                        dp.m, dp.n, dp.k, x, y, yhat, 100. * rel);
            else
                fprintf(stderr, "    %8" PRId64 " %8" PRId64 " %8" PRId64 " %12.6g %12.6g %+10.1f%%\n",
                        dp.m, dp.n, dp.k, y, yhat, 100. * rel);
        }

        if (std::abs(y) < 1e-20) continue;
        sumRelSq += rel * rel;
        count++;
    }
    return count > 0 ? 100. * std::sqrt(sumRelSq / count) : -1.;
}

double linearRegression(const std::vector<DataPoint> &data, const Evaluator &eval, double &c0, double &c1, bool effective = false)
{
    size_t n = data.size();

    std::vector<double> A(2*n);
    std::vector<double> y(n);
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    for (size_t i = 0; i < n; i++) {
        A[i] = 1.;
        A[i + n] = eval(data[i].m, data[i].n, data[i].k);
        y[i] = data[i].*fit;
    }

    auto result = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', n, 2, 1, A.data(), n, y.data(), n);

    if (result != 0)
        fatal("linear regression failed. Collected performance data may be invalid.");

    c0 = y[0];
    c1 = y[1];

    return relativeRMSE(data, [&](const DataPoint &dp) {
        return c0 + c1 * eval(dp.m, dp.n, dp.k);
    }, effective, eval);
}

// Fit a linear function c0*x + c1, clamped to the range [low, high], to given data.
// O(n^3) for n data points.
double clampedLinearRegression(const std::vector<DataPoint> &data, const Evaluator &eval, double &c0, double &c1, double low, double high, bool effective = false)
{
    size_t n = data.size();

    std::vector<double> x(n);
    std::vector<double> y(n);
    std::vector<double> A(2*n), b(n);

    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    double bestC0 = 0., bestC1 = 0.;
    double bestMSE = std::numeric_limits<double>::infinity();

    auto consider = [&](double tc0, double tc1) {
        double mse = 0.;
        for (size_t i = 0; i < n; i++) {
            double ypred = std::max(low, std::min(high, tc0 + tc1*x[i]));
            mse += (y[i] - ypred) * (y[i] - ypred);
        }
        if (mse < bestMSE) {
            bestC0 = tc0;
            bestC1 = tc1;
            bestMSE = mse;
        }
    };

    for (size_t i = 0; i < n; i++) {
        x[i] = eval(data[i].m, data[i].n, data[i].k);
        y[i] = data[i].*fit;
    }

    for (size_t i0 = 0; i0 < n - 1; i0++) {
        for (size_t i1 = i0 + 1; i1 < n; i1++) {
            if (x[i0] == x[i1]) continue;

            // Check case where best linear segment goes through two data points.
            double tc1 = (y[i0] - y[i1]) / (x[i0] - x[i1]);
            double tc0 = y[i0] - (tc1 * x[i0]);

            consider(tc0, tc1);

            // Check case where best linear segment is inside (x(i0), x(i1)).
            int nn = i1 - i0 + 1;
            if (nn > 2) {
                for (int ii = 0; ii < nn; ii++) {
                    A[ii] = 1.;
                    A[ii + nn] = x[i0 + ii];
                    b[ii] = y[i0 + ii];
                }
                auto result = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', nn, 2, 1, A.data(), nn, b.data(), nn);
                if (result != 0)
                    fatal("linear regression failed. Collected performance data may be invalid.");

                consider(b[0], b[1]);
            }

            // To be rigorous, should also check cases where one endpoint of the linear segment
            //  goes through a data point and the other does not. Probably this is good enough though.
        }
    }

    if (!std::isfinite(bestMSE))
        fatal("clamped linear regression failed. Not enough data was collected.");
    c0 = bestC0;
    c1 = bestC1;

    return relativeRMSE(data, [&](const DataPoint &dp) {
        return std::max(low, std::min(high, c0 + c1 * eval(dp.m, dp.n, dp.k)));
    }, effective, eval);
}

double minimumFit(const std::vector<DataPoint> &data, const Evaluator &eval, double &c, bool effective = false, double *outMinVal = nullptr)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    double minVal = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; i++) {
        auto val = eval(data[i].m, data[i].n, data[i].k);
        minVal = std::min(minVal, val);
    }

    double sum = 0.;
    int count = 0;
    for (size_t i = 0; i < n; i++) {
        auto val = eval(data[i].m, data[i].n, data[i].k);
        if (val == minVal) {
            sum += data[i].*fit;
            count++;
        }
    }

    if (count > 0)
        c = (sum / count);
    else
        c = -std::numeric_limits<double>::infinity();

    if (outMinVal) *outMinVal = minVal;

    return relativeRMSE(data, [&](const DataPoint &dp) {
        return c;
    }, effective, eval);
}

double scaledMinimumFit(const std::vector<DataPoint> &data, const Evaluator &eval, double &c1, bool effective = false)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    bool ok = false;
    c1 = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; i++) {
        auto val = eval(data[i].m, data[i].n, data[i].k);
        if (val != 0.) {
            c1 = std::min(c1, data[i].*fit / val);
            ok = true;
        }
    }
    if (!ok)
        fatal("max-fit failed. Collected performance data may be invalid.");

    return relativeRMSE(data, [&](const DataPoint &dp) {
        return c1 * eval(dp.m, dp.n, dp.k);
    }, effective, eval);
}

double mixCoeffFit(const std::vector<DataPoint> &data, const Evaluator &eval0, const Evaluator &eval1, double &c, bool effective = false)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    int samples = 0;
    double sum = 0.;

    for (size_t i = 0; i < n; i++) {
        auto val0 = eval0(data[i].m, data[i].n, data[i].k);
        auto val1 = eval1(data[i].m, data[i].n, data[i].k);
        if (val1 != val0) {
            auto mix = (data[i].*fit - val0) / (val1 - val0);
            sum += std::max(0., std::min(1., mix));
            samples++;
        }
    }

    if (samples > 0)
        c = sum / samples;
    else
        c = 0.;

    return relativeRMSE(data, [&](const DataPoint &dp) {
        return (1. - c) * eval0(dp.m, dp.n, dp.k) + c * eval1(dp.m, dp.n, dp.k);
    }, effective);
}

// Linear "regression" that intercepts the data point with lowest x value,
//   and puts more weight on lower x values.
double fixedMinLinearRegression(const std::vector<DataPoint> &data, const Evaluator &eval, double &c0, double &c1, bool effective = false)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    double minVal = 0.;
    minimumFit(data, eval, c0, effective, &minVal);

    double c1Total = 0.;
    int count = 0;
    for (size_t i = 0; i < n; i++) {
        auto val = eval(data[i].m, data[i].n, data[i].k);
        if (val != minVal) {
            c1Total += (data[i].*fit - c0) / (val - minVal);
            count++;
        }
    }

    if (count == 0)
        fatal("linear fit failed. Collected performance data may be invalid.");

    c1 = c1Total / count;
    c0 -= c1 * minVal;

    return relativeRMSE(data, [&](const DataPoint &dp) {
        return c0 + c1 * eval(dp.m, dp.n, dp.k);
    }, effective, eval);
}

double percentile(const std::vector<DataPoint> &data, double frac, bool effective = false)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    std::vector<double> all;
    all.reserve(data.size());

    for (size_t i = 0; i < n; i++)
        all.push_back(data[i].*fit);

    std::sort(all.begin(), all.end());

    auto idx = size_t(std::round(n * frac));
    idx = std::min(idx, n - 1);

    return all[idx];
}

/**********************/
/* Kernel Preparation */
/**********************/
std::vector<std::string> extractWords(const std::string &str)
{
    std::vector<std::string> result;
    std::string current;

    auto recordWord = [&] {
        if (!current.empty()) {
            result.push_back(std::move(current));
            current.clear();
        }
    };

    for (auto &c: str) {
        if (std::isspace(c))
            recordWord();
        else
            current += c;
    }

    recordWord();

    return result;
}

const char *scalarString(double s)
{
    if (s == 1)
        return "1";
    else if (s == 0)
        return "0";
    else if (s == -1)
        return "-1";
    else
        return "-";
}

bool mustUseBeta1(const std::string &strategy)
{
    return ((contains(strategy, "kb") || contains(strategy, "kv")))
        && !contains(strategy, "fb");
}

std::string frontendInputString(const kcatalog::Entry &e,
                                const std::string &alpha, const std::string &beta,
                                const std::string &additions,
                                const kcatalog::Selector *selectorOverride /*= nullptr*/)
{
    auto &selector = selectorOverride ? *selectorOverride : e.selector;

    std::string result;
    result += selector.kernelType;

    result += ' ';
    for (int i = 0; i < 3; i++)
        result += selector.precisions[i];

    int alignmentIn[3] = {e.restrictions.alignment[0], e.restrictions.alignment[1], e.restrictions.alignment[2]};

    if (e.selector.hw == kcatalog::HWTagXeHPC && selectorOverride
            && one_of(arch(), {kcatalog::HWTagXe2, kcatalog::HWTagXe3})) {
        // Bump up alignment when reusing XeHPC strategies on Xe2/Xe3.
        for (auto &tag: e.restrictions.tags) switch (tag) {
            case kcatalog::ReqBlock2DA: alignmentIn[0] = std::max(alignmentIn[0], 16); break;
            case kcatalog::ReqBlock2DB: alignmentIn[1] = std::max(alignmentIn[1], 16); break;
            case kcatalog::ReqBlock2DC: alignmentIn[2] = std::max(alignmentIn[2], 16); break;
            default: break;
        }
    }

    result += ' ';
    for (int i = 0; i < 3; i++) {
        result += selector.layouts[i];
        if (e.restrictions.alignment[i] > 1) {
            result += '@';
            result += std::to_string(alignmentIn[i]);
        }
    }

    result += ' ';
    result += std::to_string(e.driverInfo.unroll[LoopM]);
    result += ' ';
    result += std::to_string(e.driverInfo.unroll[LoopN]);
    result += ' ';
    result += alpha;
    result += ' ';
    result += beta;

    auto appendStrategy = [&](const std::string &s) {
        std::istringstream ss(s);
        std::string word;
        while (ss >> word) {
            result += ' ';
            result += word;
        }
    };
    appendStrategy(e.strategy);
    appendStrategy(additions);

    return result;
}

std::optional<frontend::Result> prepareKernel(const kcatalog::Entry &e,
                   const std::string &alpha = "1", const std::string &beta = "0",
                   const std::string &additions = "",
                   const kcatalog::Selector *selectorOverride = nullptr,
                   bool errorsFatal = true)
{
    LastFrontendInput = frontendInputString(e, alpha, beta, additions, selectorOverride);

    requireNativeArch(e.selector.hw);
    auto generator = frontend::Generator::Create();
    auto res = generator->generate(LastFrontendInput);

    if (!res.success) {
        if (errorsFatal)
            fatal("frontend", res.info.c_str());
        return std::nullopt;
    }

    auto &selector = selectorOverride ? *selectorOverride : e.selector;
    BatchActive = contains(selector.kernelType, "batch");

    return res;
}

std::string mock(const std::string &s)
{
    return "-- mock " + s;
}

void extractStrategyParameters(kcatalog::Entry &e)
{
    bool beta1 = mustUseBeta1(e.strategy);
    LastFrontendInput = frontendInputString(e, "1", beta1 ? "1" : "0");

    std::unique_ptr<frontend::Generator> generator
            = frontend::Generator::Create(targetProduct());
    auto res = generator->generate(LastFrontendInput, true);
    if (!res.success)
        fatal("frontend", res.info.c_str());

    std::stringstream params(res.info);
    e.driverInfo = CommonDriverInfo::undump(params);
    if (contains(e.strategy, "fp") || contains(e.strategy, "kv"))
        e.driverInfo.flags |= FlagFusedPostOps;
}

bool tagCheck(bool hasTag, const kcatalog::string &entryTags, kcatalog::RestrictionTags tag)
{
    char ctag = static_cast<char>(tag);

    return !contains(entryTags, hasTag ? std::tolower(ctag) : std::toupper(ctag));
}

std::vector<std::string> getDerivedKernelTypes(const kcatalog::Entry &e, bool accuracy)
{
    std::vector<std::string> derived;

    auto &ktype = e.selector.kernelType;
    auto &tags = e.restrictions.tags;
    char basicPrecA = e.selector.precisions[0][0];
    char basicPrecB = e.selector.precisions[1][0];
    char basicPrecC = e.selector.precisions[2][0];
    if (basicPrecA == '[') basicPrecA = e.selector.precisions[0][1];
    if (basicPrecB == '[') basicPrecB = e.selector.precisions[1][1];
    if (basicPrecC == '[') basicPrecC = e.selector.precisions[2][2];
    bool packed = (e.selector.layouts[0][0] == 'A' || e.selector.layouts[1][0] == 'B');

    bool aSmallInt = contains("OF", basicPrecA);
    bool bSmallInt = contains("OF", basicPrecB);
    bool aSmallType = contains("OFNEQ", basicPrecA);
    bool bSmallType = contains("OFNEQ", basicPrecB);
    bool aCompressed = aSmallType && !bSmallType;
    bool bCompressed = bSmallType && !aSmallType;
    bool dynamicQ = aSmallInt && bSmallInt && basicPrecC == 'S';    /* dynamic quantization */

    auto offsetType = [](char precision) {
        if (precision == 'O') return "[v]";
        if (precision == 'F') return "[g32]";
        return "";
    };

    if (!DerivedType.empty())
        derived.push_back(DerivedType);
    else if (ktype == "gemm" || ktype == "ugemm") {
        for (bool withQ: {false, true}) {
            for (char batch: {'\0', 's', 'n'}) {
                for (char triC: {'\0', 'l'}) {
                    if (withQ && !aSmallType && !bSmallType) continue;
                    if (withQ && batch) continue;
                    if (triC && withQ) continue;
                    if (triC && !contains("SDCZUJK", basicPrecA)) continue;
                    if (ktype == "ugemm" && (triC || batch)) continue;
                    if (batch == 'n' && !UseL0) continue;
                    if (batch == 'n' && packed) continue;
                    if (batch == 'n' && basicPrecA != basicPrecB) continue;
                    if (batch == 'n' && basicPrecA == 'Q') continue;
                    if (batch == 's' && accuracy) continue;
                    if (batch == 'n' && accuracy && e.strategy[0] == 'a') continue;

                    std::string offsets;
                    std::string scales;
                    if (withQ) {
                        if (aSmallType && (aCompressed || !contains(tags, kcatalog::ReqNoSumB))) {
                            std::string otype = "[p]";
                            if (aCompressed) {
                                if (aSmallInt && !contains(tags, kcatalog::ReqNoOffsetMultiDim))
                                    otype = offsetType(basicPrecA);
                                if (!bSmallType && contains("FNEQ", basicPrecA) && basicPrecB != 'S')
                                    scales += "a[g32]";
                            }
                            if (aSmallInt) {
                                offsets += 'a';
                                offsets += otype;
                            }
                        }
                        if (bSmallType && (bCompressed || !contains(tags, kcatalog::ReqNoSumA))) {
                            std::string otype = "[p]";
                            if (bCompressed) {
                                if (bSmallInt && !contains(tags, kcatalog::ReqNoOffsetMultiDim))
                                    otype = offsetType(basicPrecB);
                                if (!aSmallType && contains("FNEQ", basicPrecB) && basicPrecA != 'S')
                                    scales += "b[g32]";
                            }
                            if (bSmallInt) {
                                offsets += 'b';
                                offsets += otype;
                            }
                        }
                        if (dynamicQ) {
                            offsets = "";
                            if (e.selector.precisions[0] == "[FO]") offsets += "a[g32]";
                            if (e.selector.precisions[0] == "O")    offsets += "a[g32]";
                            if (e.selector.precisions[1] == "[FO]") offsets += "b[g32]";
                            scales = "a[g32]b[g32]";
                        }
                        if (contains(tags, kcatalog::ReqNoABOffset)) offsets.clear();
                        if (offsets.empty() && scales.empty()) continue;
                        if (basicPrecC == 'I' && !e.driverInfo.kParallel()) offsets += 'c';
                    }

                    if (!tagCheck(batch,           tags, kcatalog::ReqBatch))    continue;
                    if (!tagCheck(batch == 'n',    tags, kcatalog::ReqBatchN))   continue;
                    if (triC) continue;

                    std::string d;
                    if (!offsets.empty()) {
                        d = "offset";
                        d += offsets;
                        d += ' ';
                    }
                    if (!scales.empty()) {
                        d += "scale";
                        d += scales;
                        d += ' ';
                    }
                    if (batch) {
                        d += "batch";
                        d += batch;
                        d += ' ';
                    }
                    d += ktype;
                    if (triC)
                        d += triC;

                    derived.push_back(d);
                }
            }
        }
        bool doSum = contains("SHBTI", basicPrecC) && basicPrecA == basicPrecB;
        bool doSumA = doSum || (basicPrecB == 'O' && basicPrecA != 'O');
        bool doSumB = (doSum || (basicPrecA == 'O' && basicPrecB != 'O')) && !dynamicQ;
        if (doSumA && !contains(tags, kcatalog::ReqNoSumA))
            derived.push_back("suma gemm");
        if (doSumB && !contains(tags, kcatalog::ReqNoSumB))
            derived.push_back("sumb gemm");
    } else if (ktype.substr(0, 4) == "trsm") {
        bool trsm3 = (ktype.substr(0, 5) == "trsm3");
        for (char batch: {'\0', 's', 'n'}) {
            for (char side: {'l', 'r'}) {
                for (char uplo: {'l', 'u'}) {
                    if (batch && contains(tags, kcatalog::ReqNoBatch)) continue;
                    if (batch == 'n' && contains(tags, kcatalog::ReqNoBatchN)) continue;
                    if (batch && !trsm3) continue;
                    if (batch == 'n' && !UseL0) continue;
                    if (batch == 'n' && packed) continue;

                    std::string d;
                    if (batch) {
                        d = "batch";
                        d += batch;
                        d += ' ';
                    }
                    d += "trsm";
                    if (trsm3)
                        d += '3';
                    d += side;
                    d += uplo;

                    derived.push_back(d);
                }
            }
        }
    } else if (ktype.substr(0, 4) == "symm") {
        derived.push_back("symmll");
        derived.push_back("symmlu");
        derived.push_back("symmrl");
        derived.push_back("symmru");
        if (contains("CZJKU", basicPrecC)) {
            derived.push_back("hemmll");
            derived.push_back("hemmlu");
            derived.push_back("hemmrl");
            derived.push_back("hemmru");
        }
    } else
        derived.push_back(ktype);

    return derived;
}

bool testGeneration(const std::vector<std::string> &kernelDescs,
                    const std::regex &filterRegex, bool hasFilter)
{
    bool allOk = true;
    for (auto &desc : kernelDescs) {
        auto trimmed = desc.substr(desc.find_first_not_of(' '));
        if (hasFilter && !std::regex_search(trimmed, filterRegex)) continue;

        printf("%s ", trimmed.c_str());

        auto generator = frontend::Generator::Create();
        auto res = generator->generate(trimmed, true);
        if (res.success) {
            printf("\x1B[1;32mOK\x1B[0m\n");
        } else {
            printf("\x1B[1;31mFailed\x1B[0m\n");
            allOk = false;
        }
    }
    return allOk;
}

bool testAccuracy(const std::vector<std::string> &kernelDescs,
                  const std::regex &filterRegex, bool hasFilter)
{
    bool allOk = true;

    std::string appendStrategy;
    if (auto env = getenv("APPEND_STRATEGY")) appendStrategy = std::string(" ") + env;

    bool rcOnly = false, noRc = false;
    if (auto env = getenv("RC_ONLY")) rcOnly = atoi(env) != 0;
    if (auto env = getenv("NO_RC")) noRc = atoi(env) != 0;

    // Separate kernel descriptions into main and rc lists.
    std::vector<std::string> mainDescs, rcDescs;
    std::regex transposePattern(" [NT](@[0-9]+)?[NT]");

    for (auto &desc : kernelDescs) {
        auto trimmed = desc.substr(desc.find_first_not_of(' '));
        if (hasFilter && !std::regex_search(trimmed, filterRegex)) continue;

        if (contains(trimmed, "rc")) {
            rcDescs.push_back(trimmed);
        } else {
            if (contains(trimmed, "copy") || std::regex_search(desc, transposePattern)) {
                if (!contains(trimmed, "pab") || contains(trimmed, "batchn")) {
                    rcDescs.push_back("rc " + trimmed);
                }
            }
            mainDescs.push_back(trimmed);
        }
    }

    auto runOneAccuracy = [&](const std::string &kerneldesc) -> bool {
        std::string kd = kerneldesc + appendStrategy;
        bool isRc = contains(kd, "rc");

        if (rcOnly && !isRc) return true;
        if (noRc && isRc) return true;

        printf("%s ", kd.c_str());

        // For rc kernels, try with "pl" appended first.
        std::string kdAdjusted = kd;
        if (isRc) kdAdjusted += " pl";

        auto generator = frontend::Generator::Create();
        auto res = generator->generate(kdAdjusted);

        if (isRc && !res.success) {
            // Retry without "pl".
            generator = frontend::Generator::Create();
            res = generator->generate(kd);
        } else if (isRc && res.success) {
            printf("pl ");
        }

        if (!res.success) {
            printf("\x1B[1;31m[Kernel generation failed]\x1B[0m %s\n", res.info.c_str());
            return false;
        }

        // Initialize benchstone with the generated kernel.
        auto &bench = benchstoneDriver();
        if (bench.isKernelLoaded())
            bench.finishKernel();

        KernelInfo info(res.info);
        if (!bench.initKernel(info, res.binary)) {
            printf("\x1B[1;31m[Could not initialize kernel]\x1B[0m\n");
            return false;
        }

        // Run accuracy tests in-process.
        auto result = bench.runAccuracyTests(info);

        // Print outcome.
        if (result.failed()) {
            printf("\x1B[1;31m[%s]\x1B[0m", result.failingCase.c_str());
        }
        switch (result.accuracy) {
            case TestAccuracy::OK:           printf("\x1B[1;32m OK \x1B[0m"); break;
            case TestAccuracy::Unchecked:    printf("\x1B[1;33m Unchecked \x1B[0m"); break;
            case TestAccuracy::Inaccurate:   printf("\x1B[1;31m %s \x1B[0m", info.readCheck() ? "Read overrun" : "Inaccurate"); break;
            case TestAccuracy::Overrun:      printf("\x1B[1;31m Overrun \x1B[0m"); break;
            case TestAccuracy::RuntimeError: printf("\x1B[1;31m Runtime error \x1B[0m"); break;
        }
        printf("\n");

        bench.finishKernel();

        return !result.failed();
    };

    for (auto &desc : mainDescs) {
        if (!runOneAccuracy(desc)) allOk = false;
    }
    for (auto &desc : rcDescs) {
        if (!runOneAccuracy(desc)) allOk = false;
    }

    return allOk;
}

#ifndef _WIN32
bool testCompare(const std::vector<std::string> &kernelDescs,
                 const std::regex &filterRegex, bool hasFilter)
{
    bool allOk = true;

    bool changesOnly = false;
    if (auto env = getenv("CHANGES_ONLY")) changesOnly = atoi(env) != 0;

    for (auto &desc : kernelDescs) {
        auto trimmed = desc.substr(desc.find_first_not_of(' '));
        if (hasFilter && !std::regex_search(trimmed, filterRegex)) continue;

        // Generate old assembly via external process.
        std::string oldCmd = BaseDir + CompareOldGenerator + " " + trimmed + " 2>/dev/null";
        FILE *pipe = popen(oldCmd.c_str(), "r");
        if (!pipe) {
            printf("%s \x1B[1;35m[Could not launch old generator]\x1B[0m\n", trimmed.c_str());
            allOk = false;
            continue;
        }

        std::vector<uint8_t> oldBinary;
        uint8_t buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0)
            oldBinary.insert(oldBinary.end(), buf, buf + n);
        int status = pclose(pipe);

        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            if (WIFSIGNALED(status) && WTERMSIG(status) == SIGINT)
                fatal("aborted");
            printf("%s \x1B[1;35m[Kernel generation failed (old)]\x1B[0m\n", trimmed.c_str());
            allOk = false;
            continue;
        }

        if (oldBinary.empty()) {
            printf("%s \x1B[1;31m[No binary produced (old)]\x1B[0m\n", trimmed.c_str());
            allOk = false;
            continue;
        }

        // Generate new assembly in-process.
        auto generator = frontend::Generator::Create();
        auto newRes = generator->generate(trimmed);
        if (!newRes.success) {
            printf("%s \x1B[1;31m[Kernel generation failed (new)]\x1B[0m\n", trimmed.c_str());
            allOk = false;
            continue;
        }

        auto &newBinary = newRes.binary;
        if (newBinary.empty()) {
            printf("%s \x1B[1;31m[No binary produced (new)]\x1B[0m\n", trimmed.c_str());
            allOk = false;
            continue;
        }

        if (oldBinary != newBinary) {
            if (changesOnly)
                printf("%s\n", trimmed.c_str());
            else
                printf("%s \x1B[1;33mChanged\x1B[0m\n", trimmed.c_str());
        } else if (!changesOnly) {
            printf("%s \x1B[1;32mOK\x1B[0m\n", trimmed.c_str());
        }
    }

    return allOk;
}
#endif

bool testEntry(const kcatalog::Entry &e, TestKind kind)
{
    bool accuracy = (kind == TestKind::Accuracy);
    auto derivedList = getDerivedKernelTypes(e, accuracy);
    std::vector<std::string> kernelDescs;

    auto aBytes = extPrecisionBytes(e.selector.precisions[0], false);
    auto bBytes = extPrecisionBytes(e.selector.precisions[1], false);
    auto cBytes = extPrecisionBytes(e.selector.precisions[2], true);

    bool tryConvertC = contains("BHOQ", e.selector.precisions[0][0])
                   && !contains("BHOQ", e.selector.precisions[2][0])
                   && e.selector.kernelType[0] != 'u'
                   && !mustUseBeta1(e.strategy)
                   && aBytes < cBytes && bBytes < cBytes;

    auto downconvertedC = e.selector.precisions[(bBytes > aBytes) ? 1 : 0];

    if (downconvertedC == "O" && e.selector.precisions[2] == "S")
        tryConvertC = false;

    bool convertCFirst = (e.selector.precisions[0][0] != 'Q');
    bool firstDerived = true;

    for (auto &derivedKType: derivedList) {
        for (bool convertC: {convertCFirst, !convertCFirst}) {
            if (convertC && !tryConvertC) continue;

            auto alpha = (contains(derivedKType, "trsml") || contains(derivedKType, "trsmr")) ? "1" : "-";
            auto beta = (contains(derivedKType, "tr")) ? "0" : "-";
            std::string scalarType = "";

            auto emod = e;
            emod.selector.kernelType = derivedKType;
            if (emod.selector.precisions[2] == "I") {
                if (mustUseBeta1(e.strategy))
                    alpha = beta = "1";
                else if (contains(derivedKType, "sum"))
                    alpha = "1", beta = "0";
                else
                    scalarType = "S";
            }

            if (emod.selector.precisions[0] == "F" || emod.selector.precisions[0] != emod.selector.precisions[1])
                alpha = "1";

            if (contains(derivedKType, "sum"))
                alpha = "1";

            if (!TestAlpha.empty()) alpha = TestAlpha.c_str();
            if (!TestBeta.empty())  beta  = TestBeta.c_str();

            if (derivedKType[0] == 'u')
                alpha = "1", beta = "0", scalarType = "";

            if (convertC) {
                emod.selector.precisions[2] = '[' + emod.selector.precisions[2]
                                                  + downconvertedC
                                                  + ']';
            }

            emod.selector.precisions[2] += scalarType;

            auto &ktype = emod.selector.kernelType;
            if (convertC && contains(ktype, "offset")) {
                auto pos = ktype.begin() + ktype.find("offset") + 6;
                for (; pos < ktype.end() && *pos != ' '; ++pos) {
                    if (*pos == 'c') {
                        ktype.erase(pos);
                        break;
                    }
                }
            }

            kernelDescs.push_back(frontendInputString(emod, alpha, beta));
            if (NoDerived) goto done;
            if (!firstDerived) break;   /* only try conversion variant once */
        }
        firstDerived = false;
    }
done:

    utils::setenv("GEN", tagToGen(e.selector.hw), true);
    /* Generic Xe3p lacks fp8/fp4 dpas; test against a full-featured family. */
    const char *family = !Family.empty() ? Family.c_str()
                       : (e.selector.hw == kcatalog::HWTagXe3p) ? "CRI"
                       : tagToGen(e.selector.hw);
    utils::setenv("FAMILY", family, true);
    utils::setenv("FORMAT", accuracy ? "o" : "", true);

    // For compare tests, use FORCE_ZEBIN since it doesn't rely on OpenCL to agree
    // on hardware features or possible associated failures
    if (kind == TestKind::Compare)
        utils::setenv("FORCE_ZEBIN", "1", true);

    if (accuracy && aubCaptureEnabled()) fatal("Cannot run accuracy test, aub capture is enabled.");

    // Apply FILTER env var.
    std::string filterStr;
    if (auto env = getenv("FILTER")) filterStr = env;
    std::regex filterRegex;
    bool hasFilter = !filterStr.empty();
    if (hasFilter) filterRegex = std::regex(filterStr);

    switch (kind) {
        case TestKind::Generation:
            return testGeneration(kernelDescs, filterRegex, hasFilter);
        case TestKind::Accuracy:
            return testAccuracy(kernelDescs, filterRegex, hasFilter);
#ifndef _WIN32
        case TestKind::Compare:
            return testCompare(kernelDescs, filterRegex, hasFilter);
#endif
    }
    return true;
}

/**********/
/* Models */
/**********/

// Model 'W': old kernel selection logic, based on simple heuristic that chooses the
//            smallest unrolls that fit in 1 wave on the GPU, if possible.
void fitModelW(kcatalog::Entry &e)
{
    using namespace kcatalog;
    double priority = DefaultModelParams["priority"];
    if (priority == 0.)
        priority = e.driverInfo.unroll[LoopM] * e.driverInfo.unroll[LoopN];
    e.model.paramCount = ParamWCount;
    e.model.params[ParamWPriority] = priority;
}

void displayModelW(std::ostream &stream, const kcatalog::Entry &e, bool brief = false)
{
    using namespace kcatalog;

    if (brief)
        stream << "priority = " << int(e.model.params[ParamWPriority]);
    else
        stream << "  priority: " << int(e.model.params[ParamWPriority]) << std::endl;
}

// Model 'S': Large WG-oriented kernel selection, with more sophisticated load-balancing terms.
#define PARAM(p) e.model.params[kcatalog::ParamS_ ## p]

int64_t mnPerformanceSize(char precision)
{
    return simulation() ? 128 : 4096;
}

void findSquarishSize(int64_t &outM, int64_t &outN, int64_t wgTileM, int64_t wgTileN, int wgPerWave)
{
    outM = -1, outN = -1;
    for (int nwgM = 1; nwgM <= wgPerWave; nwgM++) {
        if (wgPerWave % nwgM) continue;
        int64_t candidateM = nwgM * wgTileM;
        int64_t candidateN = (wgPerWave / nwgM) * wgTileN;
        if (outM == -1 || (std::abs(candidateM - candidateN) < std::abs(outM - outN))) {
            outM = candidateM;
            outN = candidateN;
        }
    }
}

void fitModelS(kcatalog::Entry &e)
{
    int unrollM = e.driverInfo.unroll[LoopM];
    int unrollN = e.driverInfo.unroll[LoopN];
    int unrollK = e.driverInfo.unroll[LoopK];
    int wgTileM = e.driverInfo.wgTile(LoopM);
    int wgTileN = e.driverInfo.wgTile(LoopN);
    int wgK = e.driverInfo.wg[LoopK];
    int threadsPerWG = e.driverInfo.threadsPerWG();

    auto aElemBytes = extPrecisionBytes(e.selector.precisions[0], false);
    auto bElemBytes = extPrecisionBytes(e.selector.precisions[1], false);
    auto cElemBytes = extPrecisionBytes(e.selector.precisions[2], true);

    int eus = euCount();
    double freq = nominalFrequency();
    bool largeL3 = (e.selector.hw == kcatalog::HWTagXeHPC);

    EvaluateParams p = {{1, 65536, 65536, 65536}, 1., 0., eus};
    p.product = targetProduct();
    auto dp = getDerivedParams(e, p);

    auto threadCount = [&](int64_t m, int64_t n, int64_t k) {
        return divUp(m, wgTileM) * divUp(n, wgTileN) * threadsPerWG;
    };
    auto matrixAElems = [](int64_t m, int64_t n, int64_t k) { return static_cast<double>(m * k); };
    auto matrixBElems = [](int64_t m, int64_t n, int64_t k) { return static_cast<double>(n * k); };
    auto fmaCount     = [](int64_t m, int64_t n, int64_t k) { return static_cast<double>(m * n * k); };
    auto partialWaveCount    = [&](int64_t m, int64_t n, int64_t k) { return static_cast<double>(divUp(threadCount(m, n, k), dp.hwThreadsPartialWave)); };
    auto partialWaveQuotient = [&](int64_t m, int64_t n, int64_t k) {
        return double(dp.partialWaveCount) / partialWaveCount(m, n, k);
    };

    // SYMM/HEMM: fit the corresponding GEMM kernel instead.
    auto e_bm = e;
    if (contains(e.selector.kernelType, "symm") || contains(e.selector.kernelType, "hemm"))
        e_bm.selector.kernelType = "gemm";

    // Test cases:
    //  - Constant costs: k = 0 cases (fitting in L3 for PVC)
    std::vector<DataPoint> k0Sizes;
    for (double nwave = 1./4; nwave <= 32; nwave += 1./4) {
        DataPoint p;
        p.m = wgTileM;
        p.n = wgTileN * std::round((nwave * dp.hwThreadCapacity) / threadsPerWG);
        p.k = 0;

        if (p.n <= 0) continue;
        if (largeL3) {
            auto cBytes = p.m * p.n * cElemBytes;
            if (cBytes > (llcSize() / 2))
                continue;
        }

        k0Sizes.push_back(p);
    }

    //  - Memory costs: gemv cases fitting in L3 for PVC, outside L3 for others.
    std::vector<DataPoint> maSizes, mbSizes;
    size_t memMin, memMax;
    bool gemvM = (unrollM == 1 && unrollN > 1);
    bool gemvN = (unrollM > 1 && unrollN == 1);
    if (largeL3) {
        memMin = 1048576;
        memMax = llcSize() / 3;
    } else {
        memMin = llcSize() * 2;
        memMax = llcSize() * 8;
    }
    memMin /= std::max(aElemBytes, bElemBytes);
    memMax /= std::max(aElemBytes, bElemBytes);

    for (int64_t k0: {256, 1024}) {
        DataPoint p;
        p.k = k0 * wgK;
        for (double mn = memMin / p.k; mn <= memMax / p.k; mn *= 1.25) {
            int mnInt = std::max(mn, 1.);
            p.m = mnInt;
            p.n = 1;
            maSizes.push_back(p);
            p.m = 1;
            p.n = mnInt;
            mbSizes.push_back(p);
        }
    }

    auto wgCapacity = dp.hwThreadCapacity / threadsPerWG;
    auto gemvUnrollM = wgTileM * wgCapacity;
    auto gemvUnrollN = wgTileN * wgCapacity;
    /* Add some cases with nice m/n. */
    {
        DataPoint p;
        p.m = gemvUnrollM;
        p.n = 1;
        for (double k = memMin / gemvUnrollM; k <= memMax / gemvUnrollM; k *= 2) {
            p.k = std::max(k, 1.);
            maSizes.push_back(p);
        }

        p.n = gemvUnrollN;
        p.m = 1;
        for (double k = memMin / gemvUnrollN; k <= memMax / gemvUnrollN; k *= 2) {
            p.k = std::max(k, 1.);
            mbSizes.push_back(p);
        }
    }

    //  - Compute costs: peak performance: large near-square C, good k
    std::vector<DataPoint> efSizes;
    int wgPerWave = dp.hwThreadCapacity / threadsPerWG;
    int64_t bestM, bestN;
    findSquarishSize(bestM, bestN, wgTileM, wgTileN, wgPerWave);
    int64_t tileK = alignUp(2048, unrollK) * wgK;
    int64_t minTileMN = mnPerformanceSize(e.selector.precisions[0][0]) / std::sqrt(double(wgK));
    int64_t tileM = std::max(2 * bestM, alignUp(minTileMN, bestM));
    int64_t tileN = std::max(2 * bestN, alignUp(minTileMN, bestN));

    for (int expandM: {1, 2}) {
        for (int expandN: {1, 2}) {
            for (int expandK: {1, 2}) {
                DataPoint p;
                p.m = tileM * expandM;
                p.n = tileN * expandN;
                p.k = tileK * expandK;
                efSizes.push_back(p);
            }
        }
    }

    //  - Compute costs: partial waves: integral numbers of partial waves, up to 1 full wave
    std::vector<DataPoint> epSizes;
    if (dp.hwThreadsPartialWave < dp.hwThreadCapacity) {
        for (int pw = 1; pw * dp.hwThreadsPartialWave <= dp.hwThreadCapacity; pw++) {
            DataPoint p;
            findSquarishSize(p.m, p.n, wgTileM, wgTileN, (dp.hwThreadsPartialWave * pw) / threadsPerWG);
            p.k = tileK;
            epSizes.push_back(p);
        }
    }

    //  - Compute costs: mixing factor: 1-4 full waves + a little extra
    std::vector<DataPoint> emSizes;
    for (int expandM: {1, 2}) {
        for (int expandN: {1, 2}) {
            for (bool adjustM: {true, false}) {
                DataPoint p;
                p.m = bestM * expandM;
                p.n = bestN * expandN;
                p.k = tileK;
                if (adjustM)
                    p.m += wgTileM;
                else
                    p.n += wgTileN;
                emSizes.push_back(p);
            }
        }
    }

    // Frequency throttling: varying k
    std::vector<DataPoint> frSizes;
    for (double k = 64.; k < 2 * tileK; k *= 1.1) {
        DataPoint p;
        p.m = 2 * tileM;
        p.n = 2 * tileN;
        p.k = alignUp(int64_t(k), unrollK);
        frSizes.push_back(p);
    }

    // Time beta = 1 kernel with k = 0 to measure constant terms C01, C11.
    // Use effective efficiency to help stabilize timing.
    if (Verbose)
        fprintf(stderr, "Running benchmarks, please wait...\n");
    auto res = prepareKernel(e_bm, "1", "1");
    collectData(*res, k0Sizes, true, true);
    auto q = linearRegression(k0Sizes, partialWaveCount, PARAM(C01), PARAM(C11), true);
    verboseFit("C01, C11", "linearRegression", k0Sizes, PARAM(C01), PARAM(C11), q);
    q = minimumFit(k0Sizes, partialWaveCount, PARAM(Cm1), true);
    verboseFit("Cm1", "minimumFit", k0Sizes, PARAM(Cm1), q);

    // Same for beta = 0 kernel.
    // Use variable beta for kb kernels to enable the non-atomic beta = 0 path
    //   (benchstone will test with beta = 0).
    auto beta0 = e.driverInfo.kParallel() ? "-" : "0";
    res = prepareKernel(e_bm, "1", beta0);
    collectData(*res, k0Sizes, true, true);
    q = linearRegression(k0Sizes, partialWaveCount, PARAM(C00), PARAM(C10), true);
    verboseFit("C00, C10", "linearRegression", k0Sizes, PARAM(C00), PARAM(C10), q);
    q = minimumFit(k0Sizes, partialWaveCount, PARAM(Cm0), true);
    verboseFit("Cm0", "minimumFit", k0Sizes, PARAM(Cm0), q);

    // Baseline for the remaining fitting: remove C update to reduce variance.
    double c0, c1;
    res = prepareKernel(e_bm, "1", beta0, mock("c"));
    collectData(*res, k0Sizes, true, true);
    linearRegression(k0Sizes, partialWaveCount, c0, c1, true);

    auto subtractConstantCosts = [&](std::vector<DataPoint> &data) {
        for (auto &dp: data) {
            auto overhead = std::max(0., c0 + partialWaveCount(dp.m, dp.n, dp.k) * c1);
            dp.nominal -= overhead;
            dp.effective -= overhead;
        }
    };

    // Estimate memory loading costs for A/B separately. Skip for packed kernels.
    // Remove computation for this part.
    bool doMemA = !isPackedChar(e.selector.layouts[0]);
    bool doMemB = !isPackedChar(e.selector.layouts[1]);

    if (doMemA || doMemB)
        res = prepareKernel(e_bm, "1", beta0, mock("c m"));

    if (doMemA) {
        collectData(*res, maSizes, true, true);
        subtractConstantCosts(maSizes);
        q = scaledMinimumFit(maSizes, matrixAElems, PARAM(Ma), true);
        PARAM(Ma) = std::max(PARAM(Ma), 0.);
        verboseFit("Ma", "scaledMinimumFit", maSizes, PARAM(Ma), q);
    } else
        PARAM(Ma) = 0.;

    if (doMemB) {
        collectData(*res, mbSizes, true, true);
        subtractConstantCosts(mbSizes);
        q = scaledMinimumFit(mbSizes, matrixBElems, PARAM(Mb), true);
        PARAM(Mb) = std::max(PARAM(Mb), 0.);
        verboseFit("Mb", "scaledMinimumFit", mbSizes, PARAM(Mb), q);
    } else
        PARAM(Mb) = 0.;

    e.model.paramCount = kcatalog::ParamSCount;

    // Compute parameters. Use beta = 1 so benchstone can enable k blocking.
    res = prepareKernel(e_bm, "1", "1", mock("c"));
    collectData(*res, efSizes, false, true);
    subtractConstantCosts(efSizes);
    q = scaledMinimumFit(efSizes, fmaCount, PARAM(Ef), true);
    verboseFit("Ef", "scaledMinimumFit", efSizes, PARAM(Ef), q);

    double maxFreqDrop = 1.0;
    for (const auto &dp: efSizes) {
        auto drop = dp.nominal / dp.effective;
        if (drop > 4.) continue;    /* discard unreasonably low measured frequencies (likely MDAPI bug) */
        maxFreqDrop = std::max(maxFreqDrop, drop);
    }
    PARAM(Fp) = maxFreqDrop;
    verboseFit("Fp", "max(nom/eff)", efSizes, PARAM(Fp));

    if (gemvM || gemvN)
        PARAM(Fr0) = PARAM(Fr1) = 0;
    else {
        collectData(*res, frSizes, false, true);
        for (auto &dp: frSizes)
            dp.nominal /= dp.effective;
        q = clampedLinearRegression(frSizes, fmaCount, PARAM(Fr0), PARAM(Fr1), 1, maxFreqDrop, false);
        verboseFit("Fr0, Fr1", "clampedLinearRegression", frSizes, PARAM(Fr0), PARAM(Fr1), q);
    }

    if (epSizes.size() > 1) {
        collectData(*res, epSizes, false, true);
        subtractConstantCosts(epSizes);
        for (auto &dp: epSizes)
            dp.effective /= fmaCount(dp.m, dp.n, dp.k);
        q = linearRegression(epSizes, partialWaveQuotient, PARAM(Ep0), PARAM(Ep1), true);
        PARAM(Ep1) = std::max(0., PARAM(Ep1));
        verboseFit("Ep0, Ep1", "linearRegression", epSizes, PARAM(Ep0), PARAM(Ep1), q);
    } else {
        PARAM(Ep0) = PARAM(Ef);
        PARAM(Ep1) = 0.;
    }

    auto tPredictNoLB = [=](int64_t m, int64_t n, int64_t k) { return PARAM(Ef) * fmaCount(m, n, k); };
    auto tPredictLB   = [=](int64_t m, int64_t n, int64_t k) {
        int64_t threads = threadCount(m, n, k);
        int64_t kthread = divUp(k, e.driverInfo.wg[LoopK]);
        int partialWaveThreads = threads % dp.hwThreadCapacity;
        int partialWaves = divUp(partialWaveThreads, dp.hwThreadsPartialWave);
        double Ef = PARAM(Ef);
        double Ep = std::max(Ef, PARAM(Ep0) + (PARAM(Ep1) * dp.partialWaveCount) / partialWaves);
        double Tf = PARAM(Ef) * (threads - partialWaveThreads);
        double Tp = Ep * partialWaves * dp.hwThreadsPartialWave;
        return (Tf + Tp) * unrollM * unrollN * kthread;
    };

    collectData(*res, emSizes, false, true);
    subtractConstantCosts(emSizes);
    q = mixCoeffFit(emSizes, tPredictNoLB, tPredictLB, PARAM(Em), true);
    verboseFit("Em", "mixCoeffFit", emSizes, PARAM(Em), q);

    // Normalize data to remove frequency/EU count dependency.
    // Model outputs total EU cycles.
    for (int i = 0; i < kcatalog::ParamS_Em; i++)
        e.model.params[i] *= freq * eus;
}

void displayModelS(std::ostream &stream, const kcatalog::Entry &e, bool brief)
{
    if (brief) {
        stream << "peak FMAs/cycle = " << 1./PARAM(Ef);
        return;
    }

    bool haveEUCount = false;
    double nominalFreq = 0.;

    EvaluateParams p = {{1, 8192, 8192, 8192}, 1., 0., 128};
    p.product = targetProduct();
    if (e.selector.hw == arch(true)) {
        p.euCount = euCount();
        haveEUCount = true;
        nominalFreq = nominalFrequency();
    }

    auto dp = getDerivedParams(e, p);
    int unrollM = e.driverInfo.unroll[LoopM];
    int unrollN = e.driverInfo.unroll[LoopN];
    auto aElemBytes = extPrecisionBytes(e.selector.precisions[0], false);
    auto bElemBytes = extPrecisionBytes(e.selector.precisions[1], false);
    int cElemsPerPartialWave = dp.hwThreadsPartialWave * unrollM * unrollN / e.driverInfo.wg[LoopK];

    std::stringstream ss;

    ss << "  Cm: " << PARAM(Cm0) << " [beta = 0]\t" << PARAM(Cm1) << " [beta = 1]\n";
    ss << "  C0: " << PARAM(C00) << " [beta = 0]\t" << PARAM(C01) << " [beta = 1]\n";
    ss << "  C1: " << PARAM(C10) << " [beta = 0]\t" << PARAM(C11) << " [beta = 1]\t";
    if (haveEUCount) {
        ss << "(" << PARAM(C10) / cElemsPerPartialWave;
        ss << '/' << PARAM(C11) / cElemsPerPartialWave << " cycles/C element)";
    }
    ss << std::endl;
    ss << "  Ma: " << PARAM(Ma);
    if (nominalFreq > 0. && PARAM(Ma) > 0)
        ss << "\t\t(" << (1e-9 * nominalFreq * aElemBytes * p.euCount) / PARAM(Ma) << " GB/s)";
    ss << std::endl;
    ss << "  Mb: " << PARAM(Mb);
    if (nominalFreq > 0. && PARAM(Mb) > 0)
        ss << "\t\t(" << (1e-9 * nominalFreq * bElemBytes * p.euCount) / PARAM(Mb) << " GB/s)";
    ss << std::endl;
    ss << "  Fp: " << PARAM(Fp) << "\t\t(" << int(100/PARAM(Fp)) << "% of nominal frequency)\n";
    ss << "  Fr: " << PARAM(Fr0) << ", " << PARAM(Fr1);
    if (PARAM(Fr1) > 0.) {
        ss << "\t\t(throttling region " << 1e-9 * (1 - PARAM(Fr0)) / PARAM(Fr1);
        ss << "-" << 1e-9 * (PARAM(Fp) - PARAM(Fr0)) / PARAM(Fr1) << " GFMAs)";
    }
    ss << std::endl;
    ss << "  Ef: " << PARAM(Ef) << "\t\t(" << 1./PARAM(Ef) << " FMA/cycle)\n";
    ss << "  Ep: " << PARAM(Ep0) << ", " << PARAM(Ep1);
    ss << "\t\t(" << 1./(PARAM(Ep0) + PARAM(Ep1)*dp.partialWaveCount);
    ss << "-" << 1./(PARAM(Ep0) + PARAM(Ep1)) << " FMA/cycle)\n";
    ss << "  Em: " << PARAM(Em) << std::endl;

    tableOutput(stream, ss.str());
}
#undef PARAM

// Model 'E': Updated version of 'S'
#define PARAM(p) e.model.params[kcatalog::ParamE_ ## p]

void fitModelE(kcatalog::Entry &e)
{
    int unrollM = e.driverInfo.unroll[LoopM];
    int unrollN = e.driverInfo.unroll[LoopN];
    int unrollK = e.driverInfo.unroll[LoopK];
    int wgTileM = e.driverInfo.wgTile(LoopM);
    int wgTileN = e.driverInfo.wgTile(LoopN);
    int wgM = e.driverInfo.wg[LoopM];
    int wgN = e.driverInfo.wg[LoopN];
    int wgK = e.driverInfo.wg[LoopK];
    if (e.driverInfo.kParallelVariable())
        wgK = 1;
    int threadsPerWG = wgM * wgN * wgK * e.driverInfo.wgExpand;

    auto aElemBytes = extPrecisionBytes(e.selector.precisions[0], false);
    auto bElemBytes = extPrecisionBytes(e.selector.precisions[1], false);
    auto cElemBytes = extPrecisionBytes(e.selector.precisions[2], true);

    int eus = euCount();
    double freq = nominalFrequency();
    bool sim = simulation();

    bool largeL3 = (e.selector.hw == kcatalog::HWTagXeHPC);

    EvaluateParams p = {{1, 65536, 65536, 65536}, 1., 0., eus};
    p.product = targetProduct();
    auto dp = getDerivedParams(e, p);

    auto threadCount = [&](int64_t m, int64_t n, int64_t k) {
        return divUp(m, wgTileM) * divUp(n, wgTileN) * threadsPerWG;
    };
    auto matrixAElems = [](int64_t m, int64_t n, int64_t k) { return static_cast<double>(m * k); };
    auto matrixBElems = [](int64_t m, int64_t n, int64_t k) { return static_cast<double>(n * k); };
    auto fmaCount     = [](int64_t m, int64_t n, int64_t k) { return static_cast<double>(m * n * k); };
    auto partialWaveCount    = [&](int64_t m, int64_t n, int64_t k) { return static_cast<double>(divUp(threadCount(m, n, k), dp.hwThreadsPartialWave)); };
    auto partialWaveQuotient = [&](int64_t m, int64_t n, int64_t k) {
        return double(dp.partialWaveCount) / partialWaveCount(m, n, k);
    };

    int wgPerWave = dp.hwThreadCapacity / threadsPerWG;

    size_t memMin, memMax;
    bool gemvM = (unrollM == 1 && unrollN > 1);
    bool gemvN = (unrollM > 1 && unrollN == 1);
    if (largeL3) {
        memMin = 1048576;
        memMax = llcSize() / 3;
    } else {
        memMin = llcSize() * 2;
        memMax = llcSize() * 8;
    }

    // SYMM/HEMM: fit the corresponding GEMM kernel instead.
    auto e_bm = e;
    if (contains(e.selector.kernelType, "symm") || contains(e.selector.kernelType, "hemm"))
        e_bm.selector.kernelType = "gemm";

    // Global k-parallel kernels: remove global k-parallelization for most testing.
    bool globalKParallel = e.driverInfo.kParallel() || e.driverInfo.kParallelVariable();
    if (globalKParallel) {
        remove(e_bm.strategy, " kb");
        remove(e_bm.strategy, " kv");
        if (e.driverInfo.kParallelVariable())
            remove(e_bm.strategy, " kr");
    }

    // Test cases:
    //  - Constant costs: k = 0 cases (fitting in L3 for PVC)
    std::vector<DataPoint> k0Sizes;
    double wstep  = sim ? 4 : 1./4;
    double wlimit = sim ? 8 : 32;
    for (double nwave = wstep; nwave <= wlimit; nwave += wstep) {
        DataPoint p;
        p.m = wgTileM;
        p.n = wgTileN * std::round((nwave * dp.hwThreadCapacity) / threadsPerWG);
        p.k = 0;

        if (p.n <= 0) continue;
        if (largeL3) {
            auto cBytes = p.m * p.n * cElemBytes;
            if (cBytes > (llcSize() / 2))
                continue;
        }

        k0Sizes.push_back(p);
    }

    if (k0Sizes.back().m * k0Sizes.back().n * cElemBytes <= memMin) {
        /* Need larger sizes to reach memMin (and a bit beyond for extra data points). */
        for (double scale = 1; scale <= 2; scale += 1./8) {
            DataPoint p;
            p.m = wgTileM;
            p.n = align_up(std::ceil((scale * memMin) / (p.m * cElemBytes)), wgTileN);
            p.k = 0;

            k0Sizes.push_back(p);
            if (sim) break;
        }
    }

    //  - Alternate fused beta overheads: small squares, up to 1 wave.
    std::vector<DataPoint> afbSizes;

    auto mnStep = std::min(wgTileM, wgTileN);

    for (int64_t mn = mnStep; mn < 4096; mn += mnStep) {
        DataPoint p;
        p.m = div_up(mn, wgTileM);
        p.n = div_up(mn, wgTileN);
        if (p.m * p.n >= wgPerWave) break;
        p.m *= wgTileM;
        p.n *= wgTileN;
        p.k = 128;
        if (sim)
            afbSizes.push_back(p);
        else for (int64_t k: {64, 96, 128, 192}) {
            p.k = k;
            afbSizes.push_back(p);
        }
    }

    if (sim) {
        std::vector<DataPoint> selected;
        auto n = int(afbSizes.size());
        auto stride = std::max(1, n / 4);
        for (int i = 0; i < 4 && i*stride < n; i++)
            selected.push_back(afbSizes[i*stride]);
        std::swap(selected, afbSizes);
    }

    //  - Memory costs: gemv cases fitting in L3 for PVC, outside L3 for others.
    std::vector<DataPoint> maSizes, mbSizes;
    auto memMinAB = memMin / std::max(aElemBytes, bElemBytes);
    auto memMaxAB = memMax / std::max(aElemBytes, bElemBytes);

    for (int64_t k0: {256, 1024}) {
        DataPoint p;
        p.k = k0 * wgK;
        for (double mn = memMinAB / p.k; mn <= memMaxAB / p.k; mn *= 1.25) {
            int mnInt = std::max(mn, 1.);
            p.m = mnInt;
            p.n = 1;
            maSizes.push_back(p);
            p.m = 1;
            p.n = mnInt;
            mbSizes.push_back(p);
        }
    }

    auto wgCapacity = dp.hwThreadCapacity / threadsPerWG;
    auto gemvUnrollM = wgTileM * wgCapacity;
    auto gemvUnrollN = wgTileN * wgCapacity;
    /* Add some cases with nice m/n. */
    {
        DataPoint p;
        p.m = gemvUnrollM;
        p.n = 1;
        for (double k = memMinAB / gemvUnrollM; k <= memMaxAB / gemvUnrollM; k *= 2) {
            p.k = std::max(k, 1.);
            maSizes.push_back(p);
        }

        p.n = gemvUnrollN;
        p.m = 1;
        for (double k = memMinAB / gemvUnrollN; k <= memMaxAB / gemvUnrollN; k *= 2) {
            p.k = std::max(k, 1.);
            mbSizes.push_back(p);
        }
    }

    if (sim) {
        maSizes.resize(1);
        mbSizes.resize(1);
    }

    //  - Compute costs: peak performance: large near-square C, good k
    std::vector<DataPoint> efSizes;
    int64_t bestM, bestN;
    findSquarishSize(bestM, bestN, wgTileM, wgTileN, wgPerWave);
    int64_t perThreadK = sim ? 1024 : 2048;
    int64_t tileK = alignUp(perThreadK, unrollK) * wgK;
    int64_t minTileMN = mnPerformanceSize(e.selector.precisions[0][0]) / std::sqrt(double(wgK));
    int64_t tileM = std::max(2 * bestM, alignUp(minTileMN, bestM));
    int64_t tileN = std::max(2 * bestN, alignUp(minTileMN, bestN));

    for (int expandM: {1, 2}) {
        for (int expandN: {1, 2}) {
            for (int expandK: {1, 2}) {
                DataPoint p;
                p.m = tileM * expandM;
                p.n = tileN * expandN;
                p.k = tileK * expandK;
                efSizes.push_back(p);
            }
        }
    }

    if (sim) efSizes.resize(1);

    //  - Compute costs: partial waves: integral numbers of partial waves, up to 1 full wave
    std::vector<DataPoint> epSizes;
    if (dp.hwThreadsPartialWave < dp.hwThreadCapacity) {
        for (int pw = 1; pw * dp.hwThreadsPartialWave <= dp.hwThreadCapacity; pw++) {
            DataPoint p;
            findSquarishSize(p.m, p.n, wgTileM, wgTileN, (dp.hwThreadsPartialWave * pw) / threadsPerWG);
            p.k = tileK;
            epSizes.push_back(p);
        }
    }

    if (sim && epSizes.size() > 4) {    /* Choose at most 4 test cases under simulation */
        std::vector<DataPoint> epSizes1;
        for (int i = 1; i < 4; i++) {
            auto idx = size_t(std::round(epSizes.size() * i * 0.25) - 1);
            epSizes1.push_back(epSizes[idx]);
        }
        std::swap(epSizes, epSizes1);
    }

    //  - Compute costs: mixing factor: 1-4 full waves + a little extra
    std::vector<DataPoint> emSizes;
    for (int expandM: {1, 2}) {
        for (int expandN: {1, 2}) {
            for (bool adjustM: {true, false}) {
                DataPoint p;
                p.m = bestM * expandM;
                p.n = bestN * expandN;
                p.k = tileK;
                if (adjustM)
                    p.m += wgTileM;
                else
                    p.n += wgTileN;
                emSizes.push_back(p);
            }
        }
    }

    if (sim) emSizes.resize(1);

    // Frequency throttling: varying k
    std::vector<DataPoint> frSizes;
    for (double k = 64.; k < 2 * tileK; k *= 1.1) {
        DataPoint p;
        p.m = 2 * tileM;
        p.n = 2 * tileN;
        p.k = alignUp(int64_t(k), unrollK);
        frSizes.push_back(p);
    }

    if (Verbose)
        fprintf(stderr, "Running benchmarks, please wait...\n");

    // Time constant overheads: k = 0, and C update removed. Keep any local k reduction.
    double c0, c1;
    auto res = prepareKernel(e_bm, "1", "1", mock("c kkr"));
    collectData(*res, k0Sizes, true);
    auto q = fixedMinLinearRegression(k0Sizes, partialWaveCount, c0, c1);
    verboseFit("C0, C1", "fixedMinLinearRegression", k0Sizes, c0, c1, q);

    PARAM(C0) = c0;
    PARAM(C1) = c1;

    auto subtractConstantCosts = [&](std::vector<DataPoint> &data) {
        for (auto &dp: data) {
            auto overhead = std::max(0., c0 + partialWaveCount(dp.m, dp.n, dp.k) * c1);
            dp.nominal -= overhead;
            dp.effective -= overhead;
        }
    };

    // Now time C updates: first load + store, using beta = 1 and atomics disabled.
    // Use effective efficiency to help stabilize timing.
    // Only consider data points above memMin bytes, for greater stability.
    auto matrixCElemsAboveMin = [=](int64_t m, int64_t n, int64_t k) {
        return (m * n * cElemBytes >= memMin) ? static_cast<double>(m * n) : 0.;
    };

    res = prepareKernel(e_bm, "1", "1", "nau");    // FIXME
    collectData(*res, k0Sizes, true);
    subtractConstantCosts(k0Sizes);
    q = scaledMinimumFit(k0Sizes, matrixCElemsAboveMin, PARAM(Mcu));
    verboseFit("Mcu", "scaledMinimumFit", k0Sizes, PARAM(Mcu), q);

    // Same for beta = 0 kernel, to time only C store.
    res = prepareKernel(e_bm, "1", "0");
    collectData(*res, k0Sizes, true);
    subtractConstantCosts(k0Sizes);
    q = scaledMinimumFit(k0Sizes, matrixCElemsAboveMin, PARAM(Mc));
    verboseFit("Mc", "scaledMinimumFit", k0Sizes, PARAM(Mc), q);

    // Global k-parallel kernels with fused beta: test beta = 0 again
    //  with fused beta scaling.
    PARAM(Cb0) = PARAM(Cb1) = 0;
    if (e.driverInfo.fusedBeta() && !e.driverInfo.altFusedBeta()) {
        res = prepareKernel(e_bm, "1", "0", "kb fb " + mock("c kkr"));
        collectData(*res, k0Sizes, true);
        q = fixedMinLinearRegression(k0Sizes, partialWaveCount, PARAM(Cb0), PARAM(Cb1));
        verboseFit("Cb0, Cb1", "fixedMinLinearRegression", k0Sizes, PARAM(Cb0), PARAM(Cb1), q);
    }

    // Alternate fused beta: measure difference of beta = 0 vs. beta = 1,
    //   and take the 90th percentile.
    if (e.driverInfo.altFusedBeta()) {
        auto afbSizes1 = afbSizes;
        res = prepareKernel(e, "1", "0");
        collectData(*res, afbSizes, true);
        res = prepareKernel(e, "1", "1");
        collectData(*res, afbSizes1, true);
        for (size_t i = 0; i < afbSizes.size(); i++) {
            afbSizes1[i].nominal   = afbSizes[i].nominal   - afbSizes1[i].nominal;
            afbSizes1[i].effective = afbSizes[i].effective - afbSizes1[i].effective;
        }
        PARAM(Cb0) = std::max(0., percentile(afbSizes1, 0.9));
        verboseFit("Cb0", "percentile(0.9)", afbSizes1, PARAM(Cb0));

        if (e.driverInfo.fusedPostOps()) {
            auto afpSizes = afbSizes;
            res = prepareKernel(e, "1", "0", "ffp");
            collectData(*res, afpSizes, true);
            for (size_t i = 0; i < afbSizes.size(); i++) {
                afpSizes[i].nominal   -= afbSizes[i].nominal;
                afpSizes[i].effective -= afbSizes[i].effective;
            }
            PARAM(Cp0) = std::max(0., percentile(afpSizes, 0.9));
            verboseFit("Cp0", "percentile(0.9)", afpSizes, PARAM(Cp0));
        }
     }

    // For kernels with local k reduction, re-measure constant overheads without local k reduction
    //   ... for variable k-sliced kernels, measure _with_ local k reduction (it was disabled).
    PARAM(Ck0) = PARAM(Ck1) = 0.;
    if (contains(e.strategy, "kr")) {
        bool withKR = e.driverInfo.kParallelVariable();
        res = prepareKernel(e_bm, "1", "1", "kr " + mock(withKR ? "c kkr" : "c"));
        collectData(*res, k0Sizes, true);
        q = fixedMinLinearRegression(k0Sizes, partialWaveCount, c0, c1);

        int maxWGK = e_bm.driverInfo.wg[LoopK];
        if (maxWGK > 1) {
            if (withKR) {
                std::swap(PARAM(C0), c0);
                std::swap(PARAM(C1), c1);
            }
            PARAM(Ck0) = (PARAM(C0) - c0) / (maxWGK - 1);
            PARAM(Ck1) = (PARAM(C1) - c1) / (maxWGK - 1);
            PARAM(C0) = c0 - PARAM(Ck0);
            PARAM(C1) = c1 - PARAM(Ck1);
        }
        verboseFit("Ck0, Ck1", "fixedMinLinearRegression", k0Sizes, PARAM(Ck0), PARAM(Ck1), q);
    }

    // Estimate memory loading costs for A/B separately. Skip for packed kernels.
    // Remove computation for this part.
    bool doMemA = !isPackedChar(e.selector.layouts[0]);
    bool doMemB = !isPackedChar(e.selector.layouts[1]);

    if (doMemA || doMemB)
        res = prepareKernel(e_bm, "1", "0", mock("c m"));

    if (doMemA) {
        collectData(*res, maSizes, true);
        subtractConstantCosts(maSizes);
        q = scaledMinimumFit(maSizes, matrixAElems, PARAM(Ma));
        PARAM(Ma) = std::max(PARAM(Ma), 0.);
        verboseFit("Ma", "scaledMinimumFit", maSizes, PARAM(Ma), q);
    } else
        PARAM(Ma) = 0.;

    if (doMemB) {
        collectData(*res, mbSizes, true);
        subtractConstantCosts(mbSizes);
        q = scaledMinimumFit(mbSizes, matrixBElems, PARAM(Mb));
        PARAM(Mb) = std::max(PARAM(Mb), 0.);
        verboseFit("Mb", "scaledMinimumFit", mbSizes, PARAM(Mb), q);
    } else
        PARAM(Mb) = 0.;

    // Compute parameters. Use beta = 1 so benchstone can enable k blocking.
    res = prepareKernel(e_bm, "1", "1", mock("c"));
    collectData(*res, efSizes, false, true);
    subtractConstantCosts(efSizes);
    q = scaledMinimumFit(efSizes, fmaCount, PARAM(Ef), true);
    verboseFit("Ef", "scaledMinimumFit", efSizes, PARAM(Ef), q);

    double maxFreqDrop = 1.0;
    for (const auto &dp: efSizes) {
        auto drop = dp.nominal / dp.effective;
        if (drop > 4.) continue;    /* discard unreasonably low measured frequencies */
        maxFreqDrop = std::max(maxFreqDrop, drop);
    }
    PARAM(Fp) = maxFreqDrop;
    verboseFit("Fp", "max(nom/eff)", efSizes, PARAM(Fp));

    if (gemvM || gemvN || sim)
        PARAM(Fr0) = PARAM(Fr1) = 0;
    else {
        collectData(*res, frSizes, false, true);
        for (auto &dp: frSizes)
            dp.nominal /= dp.effective;
        q = clampedLinearRegression(frSizes, fmaCount, PARAM(Fr0), PARAM(Fr1), 1, maxFreqDrop, false);
        verboseFit("Fr0, Fr1", "clampedLinearRegression", frSizes, PARAM(Fr0), PARAM(Fr1), q);
    }

    if (epSizes.size() > 1) {
        collectData(*res, epSizes, false, true);
        subtractConstantCosts(epSizes);
        for (auto &dp: epSizes)
            dp.effective /= fmaCount(dp.m, dp.n, dp.k);
        q = linearRegression(epSizes, partialWaveQuotient, PARAM(Ep0), PARAM(Ep1), true);
        PARAM(Ep1) = std::max(0., PARAM(Ep1));
        verboseFit("Ep0, Ep1", "linearRegression", epSizes, PARAM(Ep0), PARAM(Ep1), q);
    } else {
        PARAM(Ep0) = PARAM(Ef);
        PARAM(Ep1) = 0.;
    }

    auto tPredictNoLB = [=](int64_t m, int64_t n, int64_t k) { return PARAM(Ef) * fmaCount(m, n, k); };
    auto tPredictLB   = [=](int64_t m, int64_t n, int64_t k) {
        int64_t threads = threadCount(m, n, k);
        int64_t kthread = divUp(k, wgK);
        int partialWaveThreads = threads % dp.hwThreadCapacity;
        int partialWaves = divUp(partialWaveThreads, dp.hwThreadsPartialWave);
        double Ef = PARAM(Ef);
        double Ep = std::max(Ef, PARAM(Ep0) + (PARAM(Ep1) * dp.partialWaveCount) / partialWaves);
        double Tf = PARAM(Ef) * (threads - partialWaveThreads);
        double Tp = Ep * partialWaves * dp.hwThreadsPartialWave;
        return (Tf + Tp) * unrollM * unrollN * kthread;
    };

    collectData(*res, emSizes, false, true);
    subtractConstantCosts(emSizes);
    q = mixCoeffFit(emSizes, tPredictNoLB, tPredictLB, PARAM(Em), true);
    verboseFit("Em", "mixCoeffFit", emSizes, PARAM(Em), q);

    // Normalize data to remove frequency/EU count dependency.
    // Model outputs total EU cycles.
    for (int i = 0; i < kcatalog::ParamE_Em; i++)
        e.model.params[i] *= freq * eus;
    e.model.paramCount = kcatalog::ParamECount;
}

void displayModelE(std::ostream &stream, const kcatalog::Entry &e, bool brief)
{
    if (brief) {
        stream << "peak FMAs/cycle = " << 1./PARAM(Ef);
        return;
    }

    bool haveEUCount = false;
    double nominalFreq = 0.;

    EvaluateParams p = {{1, 8192, 8192, 8192}, 1., 0., 128};
    p.product = targetProduct();
    if (e.selector.hw == arch(true)) {
        p.euCount = euCount();
        haveEUCount = true;
        nominalFreq = nominalFrequency();
    }

    auto dp = getDerivedParams(e, p);
    int unrollM = e.driverInfo.unroll[LoopM];
    int unrollN = e.driverInfo.unroll[LoopN];
    int wgK = e.driverInfo.wg[LoopK];
    auto aElemBytes = extPrecisionBytes(e.selector.precisions[0], false);
    auto bElemBytes = extPrecisionBytes(e.selector.precisions[1], false);
    auto cElemBytes = extPrecisionBytes(e.selector.precisions[2], true);
    int cElemsPerPartialWave = dp.hwThreadsPartialWave * unrollM * unrollN / e.driverInfo.wg[LoopK];

    std::stringstream ss;

    auto showGBps = [&](double elemBytes, double M) {
        if (nominalFreq > 0. && M > 0.)
            ss << "\t\t(" << (1e-9 * nominalFreq * elemBytes * p.euCount) / M << " GB/s)";
    };

    double c0Min = PARAM(C0) + PARAM(Ck0);
    double c1Min = PARAM(C1) + PARAM(Ck1);
    double c0Max = PARAM(C0) + wgK * PARAM(Ck0);
    double c1Max = PARAM(C1) + wgK * PARAM(Ck1);

    ss << "  C0: " << c0Min;
    if (wgK > 1)
        ss << '-' << c0Max;
    ss << std::endl;
    ss << "  C1: " << c1Min;
    if (wgK > 1)
        ss << '-' << c1Max;
    if (haveEUCount) {
        ss << "\t\t(" << c1Min / cElemsPerPartialWave;
        if (wgK > 1)
            ss << '-' << c1Max / cElemsPerPartialWave;
        ss << " cycles/C element)";
    }
    ss << std::endl;
    if (PARAM(Cb0) != 0)
        ss << " Cb0: " << PARAM(Cb0) << std::endl;
    if (PARAM(Cb1) != 0 && !e.driverInfo.altFusedBeta()) {
        ss << " Cb1: " << PARAM(Cb1);
        if (haveEUCount)
            ss << "\t\t(" << PARAM(Cb1) / cElemsPerPartialWave << " cycles/C element)";
        ss << std::endl;
    }
    if (PARAM(Cp0) != 0 && e.driverInfo.altFusedBeta() && e.driverInfo.fusedPostOps())
        ss << " Cp0: " << PARAM(Cp0) << std::endl;
    ss << "  Ma: " << PARAM(Ma);
    showGBps(aElemBytes, PARAM(Ma));
    ss << std::endl;
    ss << "  Mb: " << PARAM(Mb);
    showGBps(bElemBytes, PARAM(Mb));
    ss << std::endl;
    ss << "  Mc: " << PARAM(Mc);
    showGBps(cElemBytes, PARAM(Mc));
    ss << std::endl;
    ss << " Mcu: " << PARAM(Mcu);
    showGBps(cElemBytes, PARAM(Mcu));
    ss << std::endl;
    ss << "  Fp: " << PARAM(Fp) << "\t\t(" << int(100/PARAM(Fp)) << "% of nominal frequency)\n";
    ss << "  Fr: " << PARAM(Fr0) << ", " << PARAM(Fr1);
    if (PARAM(Fr1) > 0.) {
        ss << "\t\t(throttling region " << 1e-9 * (1 - PARAM(Fr0)) / PARAM(Fr1);
        ss << "-" << 1e-9 * (PARAM(Fp) - PARAM(Fr0)) / PARAM(Fr1) << " GFMAs)";
    }
    ss << std::endl;
    ss << "  Ef: " << PARAM(Ef) << "\t\t(" << 1./PARAM(Ef) << " FMA/cycle)\n";
    ss << "  Ep: " << PARAM(Ep0) << ", " << PARAM(Ep1);
    ss << "\t\t(" << 1./(PARAM(Ep0) + PARAM(Ep1)*dp.partialWaveCount);
    ss << "-" << 1./(PARAM(Ep0) + PARAM(Ep1)) << " FMA/cycle)\n";
    ss << "  Em: " << PARAM(Em) << std::endl;

    tableOutput(stream, ss.str());
}
#undef PARAM

void fitModel(kcatalog::Entry &e)
{
    switch (e.model.id) {
        case 'E':
            fitModelE(e);
            break;
        case 'S':
            fitModelS(e);
            break;
        case 'W':
            fitModelW(e);
            break;
        default: fatal("unknown model");
    }
}

void refitModel(kcatalog::Entry &e)
{
    fitModel(e);
}

void displayModel(std::ostream &stream, const kcatalog::Entry &e, bool brief = false)
{
    if (brief)
        stream << "Model " << e.model.id << " [";
    else
        stream << "Model Parameters (" << e.model.id << ")\n";

    switch (e.model.id) {
        case 'E':
            displayModelE(stream, e, brief);
            break;
        case 'S':
            displayModelS(stream, e, brief);
            break;
        case 'W':
            displayModelW(stream, e, brief);
            break;
        default: fatal("unknown model");
    }

    if (brief) stream << ']';
}

std::unordered_map<char, const char *> TagNames = {
    {kcatalog::ReqDisabled,       "disabled"},
    {kcatalog::ReqAlignFallback,  "alignfb"},
    {kcatalog::ReqBlock2DA,       "block2DA"},
    {kcatalog::ReqBlock2DB,       "block2DB"},
    {kcatalog::ReqBlock2DC,       "block2DC"},
    {kcatalog::Req64BitA,         "64bitA"},
    {kcatalog::Req64BitB,         "64bitB"},
    {kcatalog::Req64BitC,         "64bitC"},
    {kcatalog::ReqBatch,          "batch"},
    {kcatalog::ReqBatchMultiDim,  "batchm"},
    {kcatalog::ReqBatchN,         "batchn"},
    {kcatalog::ReqSumA,           "suma"},
    {kcatalog::ReqSumB,           "sumb"},
    {kcatalog::ReqABOffset,       "aboffset"},
    {kcatalog::ReqOffsetMultiDim, "offsetm"},
    {kcatalog::ReqSystolic,       "systolic"},
    {kcatalog::ReqIntegrated,     "integrated"},
    {kcatalog::ReqDiscrete,       "discrete"},
    {kcatalog::ReqXe2Block2D,     "xe2block2D"},
    {kcatalog::ReqCustom1,        "custom1"}
};

void displayRestrictions(std::ostream &stream, const kcatalog::Restrictions &r, bool brief = false)
{
    bool hasAcceptSizes = false;
    bool hasAllowedSizes = false;
    for (int dim = 0; dim < 3; dim++) {
        hasAcceptSizes |= (r.acceptSizesMin[dim] >= 0)
                       || (r.acceptSizesMax[dim] >= 0);
        hasAllowedSizes |= (r.allowedSizesMin[dim] >= 0)
                        || (r.allowedSizesMax[dim] >= 0);
    }

    bool hasSizeRestrictions = hasAcceptSizes || hasAllowedSizes;
    bool hasSteppingRestrictions = r.steppingMin >= 0 || r.steppingMax >= 0;

    if (!hasSizeRestrictions && !hasSteppingRestrictions && r.tags.empty())
        return;

    if (brief)
        stream << "Restrictions: " << r.tags << char((hasSizeRestrictions || hasSteppingRestrictions) ? '*' : ' ');
    else {
        stream << "Restrictions\n";

        if (hasSteppingRestrictions) {
            stream << "  steppings ";
            if (r.steppingMin == -1)
                stream << "any";
            else
                stream << r.steppingMin;
            stream << '-';
            if (r.steppingMax == -1)
                stream << "any";
            else
                stream << r.steppingMax;
            stream << std::endl;
        }

        if (!r.tags.empty()) {
            bool first = true;
            for (auto &tag: r.tags) {
                auto tagName = TagNames[std::toupper(tag)];

                stream << (first ? "  " : ", ");
                if (std::isalpha(tag))
                    stream << (std::isupper(tag) ? "only " : "no ");
                stream << (tagName ? tagName : "<unknown tag>");

                first = false;
            }
            stream << std::endl;
        }

        auto outputBound = [&](int bound) {
            if (bound >= 0)
                stream << bound;
            else
                stream << "any";
        };

        auto showMNK = [&](const int (&mnk)[3]) {
            stream << '(';
            outputBound(mnk[0]);
            stream << ',';
            outputBound(mnk[1]);
            stream << ',';
            outputBound(mnk[2]);
            stream << ')';
        };

        auto showBox = [&](const int (&minMNK)[3], const int (&maxMNK)[3]) {
            showMNK(minMNK);
            stream << " <= (m,n,k) <= ";
            showMNK(maxMNK);
            stream << std::endl;
        };

        if (hasAcceptSizes) {
            stream << "  always use if ";
            showBox(r.acceptSizesMin, r.acceptSizesMax);
        }

        if (hasAllowedSizes) {
            stream << "  only use if ";
            showBox(r.allowedSizesMin, r.allowedSizesMax);
        }
    }
}

/**************************/
/* Command Line Interface */
/**************************/

void readPrecisions(const char *str, std::string (&precisions)[3])
{
    int pidx = -1;
    bool inBracket = false;
    for (const char *c = str; *c; c++) {
        if (!inBracket)
            pidx++;
        if (pidx < 0 || pidx >= 3)
            fatal("could not parse precisions");
        if (*c == '[' || *c == ']') {
            bool newInBracket = (*c == '[');
            if (inBracket == newInBracket) fatal("could not parse precisions");
            inBracket = newInBracket;
        }

        precisions[pidx] += *c;
    }
}

kcatalog::Selector readSelector(int argc, char **argv, int *outAlignments)
{
    kcatalog::Selector s;
    int ignoredAlignments[3];
    if (!outAlignments)
        outAlignments = ignoredAlignments;

    auto checkArgc = [=]() {
        if (optind >= argc)
            fatal("unexpected or ill-formed kernel description");
    };

    s.hw = arch();

    checkArgc();
    s.kernelType = argv[optind++];
    if (s.kernelType == "fused"
            || s.kernelType == "super"
            || contains(s.kernelType, "batch")
            || contains(s.kernelType, "sum")
            || contains(s.kernelType, "offset")) {
        checkArgc();
        s.kernelType += ' ';
        s.kernelType += argv[optind++];
    }

    checkArgc();
    readPrecisions(argv[optind++], s.precisions);

    checkArgc();
    int lidx = -1;
    for (char *c = argv[optind]; *c; c++) {
        if (isalpha(*c))
            lidx++;
        if (lidx < 0 || lidx >= 3)
            fatal("could not parse layouts");
        if (*c == '@') {
            outAlignments[lidx] = strtol(c + 1, &c, 10);
            if (outAlignments == 0) fatal("could not parse alignments");
            c--; continue;
        } else
            outAlignments[lidx] = 1;
        s.layouts[lidx] += *c;
    }
    optind++;

    return s;
}

kcatalog::Entry readKernelDesc(int argc, char **argv)
{
    kcatalog::Entry entry;

    auto checkArgc = [=]() {
        if (optind >= argc)
            fatal("unexpected or ill-formed kernel description");
    };

    entry.selector = readSelector(argc, argv, entry.restrictions.alignment);

    checkArgc();
    entry.driverInfo.unroll[LoopM] = atoi(argv[optind++]);
    checkArgc();
    entry.driverInfo.unroll[LoopN] = atoi(argv[optind++]);

    // Ignore alpha/beta for now.
    if (!std::isalpha(argv[optind][0]))
        optind += 2;

    while (optind < argc) {
        if (!entry.strategy.empty())
            entry.strategy += ' ';
        entry.strategy += argv[optind++];
    }

    // Ask frontend for additional strategy information.
    extractStrategyParameters(entry);

    return entry;
}

SizeParams readSizeParams(const char *str)
{
    std::stringstream ss(str);
    SizeParams sizes;

    char x;
    ss >> sizes.m;
    ss >> x; if (x != 'x') fatal("could not parse problem size");
    ss >> sizes.n;
    ss >> x; if (x != 'x') fatal("could not parse problem size");
    ss >> sizes.k >> std::ws;
    if (!ss.eof()) {
        sizes.batch = sizes.m;
        sizes.m = sizes.n;
        sizes.n = sizes.k;
        ss >> x; if (x != 'x') fatal("could not parse problem size");
        ss >> sizes.k;
    } else
        sizes.batch = 1;

    return sizes;
}

ExtendedEntryIterator readKernelFilter(int argc, char **argv, bool allowAll = true)
{
    if (optind >= argc) {
        if (!allowAll)
            fatal("expected a kernel ID or filter");
        return ExtendedEntryIterator(Catalog, Arch);
    }

    if (!strcmp(argv[optind], "*"))
        return ExtendedEntryIterator(Catalog, Arch);
    else if (!strcmp(argv[optind], "-onednn"))
        return ExtendedEntryIterator(Catalog, Client::oneDNN, false, Arch);
    else if (!strcmp(argv[optind], "+onednn"))
        return ExtendedEntryIterator(Catalog, Client::oneDNN, true, Arch);
    else if (!strcmp(argv[optind], "-onemkl"))
        return ExtendedEntryIterator(Catalog, Client::oneMKL, false, Arch);
    else if (!strcmp(argv[optind], "+onemkl"))
        return ExtendedEntryIterator(Catalog, Client::oneMKL, true, Arch);
    else if (argv[optind][0] == '#')
        return ExtendedEntryIterator(Catalog, std::atoi(argv[optind++] + 1), Arch);
    else if (std::isdigit(argv[optind][0]))
        return ExtendedEntryIterator(Catalog, std::atoi(argv[optind++]), Arch);
    else {
        const char *pattern = argv[optind++];
        if (pattern[0] == '*') pattern++;

        std::string spattern(pattern);
        while (optind < argc) {
            spattern += ' ';
            spattern += argv[optind++];
        }

        return ExtendedEntryIterator(Catalog, spattern, Arch);
    }
}

template <typename T>
bool contains(std::string haystack, T needle)
{
    return (haystack.find(needle) != haystack.npos);
}

MatchParams readMatchParams(int argc, char **argv)
{
    MatchParams params;

    params.selector = readSelector(argc, argv, params.alignment);
    params.sizes = EvalSize;
    params.stepping = stepping();

    auto &ktype = params.selector.kernelType;
    if (contains(ktype, "batch"))
        params.tags += kcatalog::ReqBatch;
    if (contains(ktype, "batchn"))
        params.tags += kcatalog::ReqBatchN;
    if (contains(ktype, "suma"))
        params.tags += kcatalog::ReqSumA;
    if (contains(ktype, "sumb"))
        params.tags += kcatalog::ReqSumB;
    if (SystolicAvailable)
        params.tags += kcatalog::ReqSystolic;

    /* Block 2D checks */
    auto lda = isLDRow(params.selector.layouts[0]) ? params.sizes.m : params.sizes.k;
    auto ldb = isLDRow(params.selector.layouts[1]) ? params.sizes.k : params.sizes.n;
    auto ldc = isLDRow(params.selector.layouts[2]) ? params.sizes.m : params.sizes.n;

    lda *= extPrecisionBytes(params.selector.precisions[0], false);
    ldb *= extPrecisionBytes(params.selector.precisions[1], false);
    ldc *= extPrecisionBytes(params.selector.precisions[2], true);

    bool ok2DA = (lda <= 16777216);
    bool ok2DB = (ldb <= 16777216);
    bool ok2DC = (ldc <= 16777216);

    if (one_of(params.selector.hw, {kcatalog::HWTagXe2, kcatalog::HWTagXe3})) {
        ok2DA &= (params.alignment[0] % 16 == 0);
        ok2DB &= (params.alignment[1] % 16 == 0);
        ok2DC &= (params.alignment[2] % 16 == 0);
    }

    if (ok2DA) params.tags += kcatalog::ReqBlock2DA;
    if (ok2DB) params.tags += kcatalog::ReqBlock2DB;
    if (ok2DC) params.tags += kcatalog::ReqBlock2DC;

    params.lateTags = params.tags;

    return params;
}

std::tuple<double, double> readScalars(int argc, char **argv, bool optional = true)
{
    auto checkArgc = [=]() {
        if (!optional && optind >= argc)
            fatal("expected alpha/beta parameters");
    };

    double alpha = 1., beta = 0.;

    checkArgc();
    if (optind < argc)
        alpha = std::atof(argv[optind++]);

    checkArgc();
    if (optind < argc)
        beta = std::atof(argv[optind++]);

    return std::make_tuple(alpha, beta);
}

int readBound(std::stringstream &ss)
{
    int bound = -1;
    if (ss.peek() == 'a') {
        expect(ss, 'a');
        expect(ss, 'n');
        expect(ss, 'y');
    } else
        ss >> bound;
    return bound;
};

RestrictionSpec readRestrictions(const char *s)
{
    RestrictionSpec spec;

    if (s[0] == 'x') {
        spec.clear = true;
        spec.mode = RestrictionSpec::Tag;
        s++;
    }

    switch (s[0]) {
        case '+':
        case '-':
            spec.mode = RestrictionSpec::Tag;
            spec.negative = (s[0] == '-');
            s++;
            break;
        case 'a':
            if (!strncmp(s, "accept", 6)) {
                s += 6;
                spec.mode = RestrictionSpec::Accept;
            } else if (!strncmp(s, "allowed", 7)) {
                s += 7;
                spec.mode = RestrictionSpec::Allowed;
            }
            break;
        case 's':
            if (!strncmp(s, "steppings", 9)) {
                s += 9;
                spec.mode = RestrictionSpec::Steppings;
            }
            break;
        default: break;
    }

    if (spec.mode == RestrictionSpec::None)
        fatal("unknown restriction requested");

    if (spec.mode == RestrictionSpec::Tag) {
        for (auto &entry: TagNames) {
            if (!utils::strcasecmp(entry.second, s))
                spec.tag = entry.first;
        }
        if (!spec.tag)
            fatal("unknown tag");
    } else if (!spec.clear) {
        std::stringstream ss(s);

        expect(ss, ':');

        if (spec.mode == RestrictionSpec::Steppings) {
            spec.steppingMin = readBound(ss);
            expect(ss, '-');
            spec.steppingMax = readBound(ss);
        } else {
            expect(ss, '(');
            spec.mnkMin[0] = readBound(ss);
            expect(ss, ',');
            spec.mnkMin[1] = readBound(ss);
            expect(ss, ',');
            spec.mnkMin[2] = readBound(ss);
            expect(ss, ')');
            expect(ss, '-');
            expect(ss, '(');
            spec.mnkMax[0] = readBound(ss);
            expect(ss, ',');
            spec.mnkMax[1] = readBound(ss);
            expect(ss, ',');
            spec.mnkMax[2] = readBound(ss);
            expect(ss, ')');
        }
    }

    return spec;
}

void displayEntry(std::ostream &stream, const kcatalog::Entry &e, bool brief = false, int ientry = -1)
{
    if (ientry == -1) {
        for (int i = 0; i < Catalog.entryCount; i++) {
            if (!memcmp(&Catalog.entries[i], &e, sizeof(kcatalog::Entry))) {
                ientry = i;
                break;
            }
        }
    }

    if (!brief)
        stream << VT_HIGHLIGHT;

    if (ientry >= 0)
        stream << '#' << ientry << '\t';

    stream << tagToGen(e.selector.hw, true) << '\t';

    stream << e.selector.kernelType << ' ';

    for (int i = 0; i < 3; i++)
        stream << e.selector.precisions[i];

    stream << ' ';

    for (int i = 0; i < 3; i++) {
        stream << e.selector.layouts[i];
        if (e.restrictions.alignment[i] > 1)
            stream << '@' << e.restrictions.alignment[i];
    }

    stream << ' ' << e.driverInfo.unroll[LoopM];
    stream << ' ' << e.driverInfo.unroll[LoopN];

    if (!e.strategy.empty())
        stream << ' ' << e.strategy;

    if (brief) {
        stream << '\t';
        displayModel(stream, e, true);
        stream << '\t';
        displayRestrictions(stream, e.restrictions, true);
    } else {
        stream << VT_RESET "\n";
        displayModel(stream, e, false);
        displayRestrictions(stream, e.restrictions, false);
    }
}

void readCatalog()
{
    if (ReadKernelFromArgv) return;

    std::ifstream fs(CatalogPath);
    if (fs.is_open())
        deserializeCatalog(fs);
}

void writeCatalog()
{
    if (ReadKernelFromArgv) return;
    if (CurrentMode != Mode::Init && !std::filesystem::exists(CatalogPath))
        fprintf(stderr, "Note: creating new catalog at %s\n", CatalogPath.c_str());
    std::ofstream fs(CatalogPath);
    serializeCatalog(fs);

    // Only emit copy-back hint when the catalog lives in the build-tree db dir.
    if (!CatalogInDbDir) return;

    auto exeDir = catalogDbDir().parent_path().parent_path();
    auto srcDst = exeDir.parent_path() / "selector" / "db" / std::filesystem::path(CatalogPath).filename();

    fprintf(stderr,
            "Note: updated catalog written to %s.\n"
            "      To persist changes, copy it back to the source tree:\n"
            "          cp %s %s\n",
            CatalogPath.c_str(), CatalogPath.c_str(), srcDst.c_str());
}

// Upgrade/downgrade for the change in prefetch distance interpretation
//  old: @2 -> prefetch distance is 2 k loops
//  new: @2 -> prefetch distance is k = 2 rows/columns
void upgradePrefetchDistance(int argc, char **argv, bool downgrade)
{
    for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
        int unrollK = iter->driverInfo.unroll[LoopK];

        if (!contains(iter->strategy, '@'))
            continue;

        std::stringstream oldStrategy(iter->strategy);
        std::stringstream newStrategy;

        oldStrategy >> std::noskipws;
        while (oldStrategy) {
            char c;
            oldStrategy >> c;
            if (oldStrategy.eof())
                break;
            newStrategy << c;
            if (c == '@') {
                int dist;
                oldStrategy >> dist;
                if (downgrade)
                    dist /= unrollK;
                else
                    dist *= unrollK;
                newStrategy << dist;
            }
        }

        std::cout << "Updated #" << iter.index();
        std::cout << ": \"" << iter->strategy << "\" -> \"";
        iter->strategy = newStrategy.str();
        std::cout << iter->strategy << "\"\n";

        CatalogUpdated = true;
    }
}

// Upgrade/downgrade for the change in prefetch distance interpretation:
//  old: aB8x2+B8@16 -> prefetch distance is 16 rows/columns before multiply
//  new: aB8x2+B8@16 -> prefetch distance is 16 rows/columns before load
void upgradePrefetchBase(int argc, char **argv, bool downgrade)
{
    for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
        if (!contains(iter->strategy, '@'))
            continue;

        std::stringstream oldStrategy(iter->strategy);
        std::stringstream newStrategy;

        int copies = 1;
        int k_load = 1;
        int lastInt = 0;
        bool updated = false;

        oldStrategy >> std::noskipws;
        while (oldStrategy) {
            if (std::isdigit(oldStrategy.peek())) {
                oldStrategy >> lastInt;
                newStrategy << lastInt;
                if (oldStrategy.eof())
                    break;
            } else {
                char c;
                oldStrategy >> c;
                if (oldStrategy.eof())
                    break;
                newStrategy << c;
                if (c == 'x' && std::isdigit(oldStrategy.peek())) {
                    k_load = lastInt;
                    oldStrategy >> copies;
                    if (oldStrategy.eof())
                        break;
                    newStrategy << copies;
                }
                if (c == '@' && copies > 1) {
                    int dist;
                    oldStrategy >> dist;
                    if (downgrade)
                        dist += (copies - 1) * k_load;
                    else
                        dist -= (copies - 1) * k_load;
                    newStrategy << dist;
                    updated = true;
                }
                if (c == ' ') {
                    copies = 1;
                }
            }
        }

        if (updated) {
            std::cout << "Updated #" << iter.index();
            std::cout << ": \"" << iter->strategy << "\" -> \"";
            iter->strategy = newStrategy.str();
            std::cout << iter->strategy << "\"\n";

            CatalogUpdated = true;
        }
    }
}

// Upgrade for relaxing alignment requirements on block 2D loads
void upgradeBlock2DRelax(int argc, char **argv, bool downgrade)
{
    if (downgrade)
        fatal("cannot undo this upgrade");

    auto msgAlign = [](const std::string &s) -> int {
        if (s[0] != 'a') return 0;
        switch (s[1]) {
            case 'm':
            case 'v': return 8;
            case 't': return 4;
            default:  return 0;
        }
    };

    for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
        auto &tags = iter->restrictions.tags;
        auto &strategy = iter->strategy;

        if (iter->selector.hw != kcatalog::HWTagXeHPC) continue;

        std::istringstream ss(strategy);
        bool changed = false;

        char block2DTags[3] = {'A', 'B', 'C'};  // ReqBlock2D{A,B,C}
        int newOptABAlign = 0;

        for (int i = 0; i < 3; i++) {
            std::string s;
            ss >> s;
            int block2DMsgAlign = msgAlign(s);

            if (block2DMsgAlign <= 0)
                continue;

            int defaultAlign = defaultAlignment(iter->selector.precisions[i]);
            int minAlign = std::max(block2DMsgAlign, defaultAlign);

            auto pos = tags.find(block2DTags[i]);
            if (pos != std::string::npos) {
                changed = true;
                tags.erase(pos, 1);
            }

            if (minAlign <= iter->restrictions.alignment[i])
                iter->restrictions.alignment[i] = minAlign;
            else if (i < 2 && minAlign > defaultAlign)
                newOptABAlign = std::max(newOptABAlign, minAlign);
        }

        auto pos = strategy.find(" l128");
        if (pos != std::string::npos) {
            strategy.erase(pos, 5);
            changed = true;
        }

        if (newOptABAlign > 0) {
            strategy += " l";
            strategy += std::to_string(newOptABAlign);
            changed = true;
        }

        if (changed) {
            std::cout << "Updated #" << iter.index() << std::endl;
            CatalogUpdated = true;
        }
    }
}

// Upgrade block 2D kernel fallbacks:
//   Change l8 -> l2d
//   Nonsystolic kernels: add l2d if there are block 2D loads
//   Remove ReqBlock2DA/B unless systolic + block 2D loads
void upgradeBlock2DFallbacks(int argc, char **argv, bool downgrade)
{
    if (downgrade)
        fatal("cannot undo this upgrade");

    for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
        auto &tags = iter->restrictions.tags;
        auto &strategy = iter->strategy;
        auto oldStrategy = strategy;

        if (iter->selector.hw != kcatalog::HWTagXeHPC) continue;

        std::istringstream ss(strategy);
        bool changed = false;
        bool block2DAB = false;
        bool addL2D = false;
        bool sys = contains(strategy, "sys");

        for (int i = 0; i < 2; i++) {
            std::string s;
            ss >> s;
            block2DAB |= (s[0] == 'a' && one_of(s[1], {'m', 'v', 't'}));
        }

        if (!sys)
            addL2D |= block2DAB;

        auto pos = strategy.find(" l8");
        if (pos != std::string::npos) {
            strategy.erase(pos, 3);
            addL2D = true;
            changed = true;
        }

        bool hasL2D = contains(strategy, "l2d");
        if (addL2D && !hasL2D) {
            strategy += " l2d";
            changed = true;
            hasL2D = true;
        }

        if (!(sys && block2DAB && !hasL2D)) {
            for (auto tag: {kcatalog::ReqBlock2DA, kcatalog::ReqBlock2DB}) {
                auto pos = tags.find(tag);
                if (pos != std::string::npos) {
                    tags.erase(pos, 1);
                    changed = true;
                }
            }
        }

        if (changed) {
            std::cout << "Updated #" << iter.index();
            std::cout << ": \"" << oldStrategy << "\" -> \"" << strategy << "\"\n";
            CatalogUpdated = true;
        }
    }
}

// Upgrade for adding local k-parallelization to variable k-sliced kernels
void upgradeKVWithKR(int argc, char **argv, bool downgrade)
{
    if (downgrade)
        fatal("cannot undo this upgrade");

    for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
        auto &strategy = iter->strategy;

        if (!contains(strategy, " kv") || contains(strategy, " kr")) continue;

        if (iter->selector.hw != kcatalog::HWTagXeHPC) continue;

        EvaluateParams eparams = {{0, 0, 0, 0}, 1., 0., euCount()};
        eparams.product = targetProduct();
        auto dp = getDerivedParams(*iter, eparams);

        int wgK = dp.hwThreadCapacity / dp.hwThreadsPartialWave;

        if (wgK > 1) {
            std::istringstream ss(strategy);
            std::ostringstream newStrategy;
            std::string word;
            bool first = true;

            do {
                if (!first)
                    newStrategy << ' ';
                first = false;
                ss >> word >> std::ws;
                newStrategy << word;
                if (word == "wg") {
                    ss >> word >> std::ws;
                    word += 'x';
                    word += std::to_string(wgK);
                    word += " kr";
                    newStrategy << ' ' << word;
                }
            } while (!ss.eof());

            std::cout << "Updated #" << iter.index();
            std::cout << ": \"" << iter->strategy << "\" -> \"";
            iter->strategy = newStrategy.str();
            std::cout << iter->strategy << "\"\n";

            CatalogUpdated = true;
        }
    }
}

// "Upgrade" that re-applies default restrictions
void upgradeRetag(int argc, char **argv, bool downgrade)
{
    if (downgrade)
        fatal("cannot undo this upgrade");

    for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
        auto copy = *iter;
        Arch = copy.selector.hw;
        extractStrategyParameters(copy);
        copy.restrictions.tags.clear();
        addDefaultRestrictions(copy);

        for (auto tag: copy.restrictions.tags) {
            auto &tags = iter->restrictions.tags;
            if (!contains(tags, tag)) {
                tags.insert(std::upper_bound(tags.begin(), tags.end(), tag), tag);
                CatalogUpdated = true;
            }
        }
    }
}

// Upgrade to convert a regular kernel to a microkernel
void upgradeMicrokernel(int argc, char **argv, bool downgrade)
{
    if (downgrade)
        fatal("cannot undo this upgrade");

    for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
        auto &kernelType = iter->selector.kernelType;

        if (kernelType[0] != 'u') {
            kernelType = 'u' + kernelType;

            std::istringstream oldStrategy(iter->strategy);
            std::ostringstream newStrategy;

            std::string word;

            for (int i = 0; i < 2; i++) {
                oldStrategy >> word;
                if (word[0] == 's') {
                    word[0] = 'a';
                    if (word[1] == 'u')
                        word[1] = 's';
                }
                if (i > 0) newStrategy << ' ';
                newStrategy << word;
            }

            oldStrategy >> word;
            word[0] = 'r';
            newStrategy << ' ' << word;

            while (!oldStrategy.eof()) {
                oldStrategy >> word;
                if (word == "pab" || word == "kv" || word == "kb" || word == "afb" || word == "fb"
                        || word == "ar" || word == "sr" || word == "br"
                        || word == "hi" || word == "bo" || word == "li" || word == "pt")
                    continue;
                if (word.length() > 2) {
                    auto prefix = word.substr(0, 2);
                    if (prefix == "bm" || prefix == "bn" || prefix == "bk")
                        continue;
                }
                newStrategy << ' ' << word;
            }

            std::cout << "Updated #" << iter.index();
            std::cout << ": \"" << iter->strategy << "\" -> \"";
            iter->strategy = newStrategy.str();
            std::cout << iter->strategy << "\"\n";

            CatalogUpdated = true;
        }
    }
}

void usage(const char *pname)
{
    std::cerr << "Usage: " << pname << " action [flags] [filter | problem strategy...]\n"
                 "\n"
                 "Actions:\n"
                 "    -i, --init               Initialize kernel catalog.\n"
                 "    -f, --fit                Fit performance model to the given kernel.\n"
                 "    -a, --add                Like --fit, also adding kernel to the catalog.\n"
                 "                               -r <limits> (see below) can also be used after -a to apply restrictions.\n"
                 "    -l, --list               List kernels in catalog, optionally matching a given filter.\n"
                 "                               Use -v for longer description.\n"
                 "                               -r <limits> (see below) can also be used after -l to filter by restrictions.\n"
                 "        --list-ids           Similar to --list but outputs matched kernel IDs only, separated by spaces.\n"
                 "    -e, --eval <size>        Evaluate and rank performance models for matching kernels.\n"
                 "                               Size is specified as MxNxK.\n"
                 "    -y, --evaltest <size>    Like --eval, but also measures and reports each kernel's actual performance.\n"
                 "    -d, --delete             Delete kernel(s) from the catalog by ID, or matching a given filter.\n"
                 "    -k, --clone <types|arch> Clone kernel(s) with different data type or different architecture.\n"
#ifndef _WIN32
                 "    -M, --modify <command>   Modify strategies for kernels matching a given filter.\n"
                 "                               Old strategies are passed one-by-one as arguments to <command>, and replaced\n"
                 "                                 with the first line of the command's output.\n"
#endif
                 "    -R, --refit              Re-fit performance model to kernels matching a given filter.\n"
                 "    -I, --reinfo             Regenerate DriverInfo information for kernels matching a given filter.\n"
                 "    -o, --replace <filter>   Replace the strategy of the kernel matching <filter> (a single kernel ID\n"
                 "                               or #<id>) with the given <unrollM> <unrollN> <strategy...>. DriverInfo\n"
                 "                               is regenerated, and the performance model is re-fit if it is model E.\n"
                 "    -r, --restrict <limits>  Add or modify restrictions on kernel.\n"
                 "            -r steppings:s0-s1                 Only allow given steppings (inclusive).\n"
                 "            -r allowed:(m0,n0,k0)-(m1,n1,k1)   Only allow kernel in the given range of problem sizes (inclusive).\n"
                 "            -r accept:(m0,n0,k0)-(m1,n1,k1)    Always choose kernel in the given range of problem sizes (inclusive).\n"
                 "            -r +<tag>                          Only use kernel in cases described by <tag>.\n"
                 "            -r -<tag>                          Never use kernel in cases described by <tag>.\n"
                 "            -r xsteppings                      Remove stepping restrictions.\n"
                 "            -r xallowed                        Remove allowed/accept settings.\n"
                 "            -r xaccept\n"
                 "            -r x<tag>                          Remove given tag.\n"
                 "                             A list of supported tags is shown below. m/n/k/stepping values may be specified as 'any'\n"
                 "                               (without quotes) to specify no bounds in that dimension.\n"
                 "                             -r may be repeated multiple times.\n"
                 "    -p, --append <catalog>   Append the given catalog file to this one.\n"
                 "    -x, --extract <catalog>  Extract kernel(s) matching a given filter to a new catalog file.\n"
                 "                               If <catalog> already exists, the extracted kernels are appended to it.\n"
                 "    -g, --gentest            Check that kernels can be successfully generated. An optional filter may be applied.\n"
                 "                               All allowed variants (e.g. syrk, batch, etc.) will be tested.\n"
                 "    -t, --test               Like --gentest, but also checks accuracy. Only kernels matching the\n"
                 "                               current GPU will be tested.\n"
#ifndef _WIN32
                 "    -j, --compare-test       Compare kernels from two different frontend builds, checking for changes.\n"
#endif
                 "        --remainder-report   Check remainder handling performance for selected kernel(s).\n"
                 "    -J, --remainder-variations        Like --remainder-report, but also checks some potentially\n"
                 "                                        beneficial strategy variations.\n"
                 "        --ld-report          Check performance with misaligned leading dimensions for selected kernel(s).\n"
                 "    -L, --ld-variations      Like --ld-report, but also checks some potentially beneficial strategy variations.\n"
                 "    -?, --help               Display this help message.\n"
                 "\n"
                 "Flags:\n"
                 "    -m, --model=<char>       Specify performance model to use (default: E).\n"
                 "    -P <param>=<value>       Set model parameter <param> to <value>.\n"
                 "                               A list of valid parameter names is below.\n"
                 "    -A, --arch=<arch>[:<fam>]  GPU architecture (default: host architecture),\n"
                 "                               with an optional product family for generation.\n"
                 "                               Arch: Gen12LP/XeLP, XeHP, XeHPG, XeHPC, Xe2 (or\n"
                 "                               12LP, 12HP, 12p7, 12p8, 2). Family: e.g. CRI, BMG,\n"
                 "                               PVC; defaults to the generic family of the arch.\n"
                 "    -S, --stepping=<n>       Specify stepping (default: 0).\n"
                 "    -E, --eu=<count>         Specify EU count (default: autodetect).\n"
                 "    -C, --llc=<MB>           Specify last-level cache size, in MB (default: autodetect).\n"
                 "    -F, --frequency=<MHz>    Specify nominal frequency (default: autodetect).\n"
                 "    -c, --catalog=<path>     Kernel catalog file (default: kernel.db).\n"
                 "                               Bare filenames resolve to <exe_dir>/selector/db/<name>;\n"
                 "                               paths with '/' or '\\\\' are used as-is (use './name.db' for cwd).\n"
                 "    -K, --kernel             Read kernel to test from command line instead of catalog.\n"
                 "                               (--test/--gentest/--remainder-variations/--ld-variations, etc.)\n"
                 "    -T, --effective          Do not include frequency throttling effects.\n"
                 "    -U, --unsorted           Do not sort output.\n"
                 "    -Y, --csv                Output in CSV format.\n"
                 "    -O, --opencl             Use OpenCL runtime for benchmarking (default).\n"
#ifdef GEMMSTONE_WITH_L0_RUNTIME
                 "    -0, --l0                 Use L0 runtime for benchmarking.\n"
#endif
                 "    -v, --verbose            Enable extra output.\n"
                 "    -Z, --dumpdata           Dump collected benchmark data. Creates file(s) named\n"
                 "                               ktool_data{0,1,2,...}.csv in the current directory.\n"
                 "    -N, --no-derived         With -t/-g, only test a single kernel per entry.\n"
                 "    -D, --derived=<type>     With -t/-g, test the given kernel variant (e.g. `batchn gemm`).\n"
                 "    --alpha=<val>            Set alpha/beta values for kernel testing (-t/-g).\n"
                 "    --beta=<val>\n"
                 "    --tight-ld               Use tight leading dimensions for kernel evaluation (-y).\n"
                 "\n"
                 "Filters:\n"
                 "    <n>, #<n>                Specific kernel ID.\n"
                 "    <string>, *<string>      Any kernel containing <string> in the full description,\n"
                 "                               e.g. 'gemm SSS' or 'kb'.\n"
                 "                             If <string> begins with a digit, add * before it to avoid\n"
                 "                               confusion with kernel IDs.\n"
                 "    +onemkl                  Kernels relevant to oneMKL.\n"
                 "    +onednn                  Kernels relevant to oneDNN.\n"
                 "    -- -onemkl               Kernels not relevant to oneMKL. The '--' is needed beforehand.\n"
                 "    -- -onednn               Kernels not relevant to oneDNN. The '--' is needed beforehand.\n"
                 "\n"
                 "    -A, --arch=<arch>        The -A/--arch option can be used to filter by architecture.\n"
                 "\n";

    std::cerr << "Tags (with one-character abbreviations):\n";
    std::stringstream tagTable;
    for (char a = '#'; a <= 'Z'; a++)
        if (TagNames.find(a) != TagNames.end())
            tagTable << "  " << TagNames[a] << "\t(" << a << ")\n";
    tableOutput(std::cerr, tagTable.str());

    std::cerr << "\n\nModel Parameters:\n   ";
    for (const auto &keyval: DefaultModelParams)
        std::cerr << ' ' << keyval.first;
    std::cerr << std::endl;

    exit(2);
}

void parseArgs(int argc, char **argv)
{
    option longopts[] = {
        {"add", no_argument, nullptr, 'a'},
        {"fit", no_argument, nullptr, 'f'},
        {"clone", required_argument, nullptr, 'k'},
        {"init", no_argument, nullptr, 'i'},
        {"list", no_argument, nullptr, 'l'},
        {"list-ids", no_argument, nullptr, '&'},
        {"append", required_argument, nullptr, 'p'},
        {"extract", required_argument, nullptr, 'x'},
        {"eval", required_argument, nullptr, 'e'},
        {"restrict", required_argument, nullptr, 'r'},
        {"delete", no_argument, nullptr, 'd'},
#ifndef _WIN32
        {"modify", required_argument, nullptr, 'M'},
#endif
        {"refit", no_argument, nullptr, 'R'},
        {"reinfo", no_argument, nullptr, 'I'},
        {"replace", no_argument, nullptr, 'o'},
        {"gentest", no_argument, nullptr, 'g'},
#ifndef _WIN32
        {"compare-test", no_argument, nullptr, 'j'},
#endif
        {"test", no_argument, nullptr, 't'},
        {"evaltest", no_argument, nullptr, 'y'},
        {"upgrade", required_argument, nullptr, 'u'},
        {"downgrade", required_argument, nullptr, 'w'},
        {"remainder-report", no_argument, nullptr, '@'},
        {"remainder-variations", no_argument, nullptr, 'J'},
        {"ld-report", no_argument, nullptr, '#'},
        {"ld-variations", no_argument, nullptr, 'L'},
        {"wipe-constant", no_argument, nullptr, '!'},
        {"wipe-memory", no_argument, nullptr, '\\'},
        {"help", no_argument, nullptr, '?'},
        {"arch", required_argument, nullptr, 'A'},
        {"stepping", required_argument, nullptr, 'S'},
        {"eu", required_argument, nullptr, 'E'},
        {"frequency", required_argument, nullptr, 'F'},
        {"llc-mb", required_argument, nullptr, 'C'},
        {"catalog", required_argument, nullptr, 'c'},
        {"kernel", no_argument, nullptr, 'K'},
        {"model", required_argument, nullptr, 'm'},
        {"unsorted", no_argument, nullptr, 'U'},
        {"effective", no_argument, nullptr, 'T'},
        {"csv", no_argument, nullptr, 'Y'},
        {"opencl", no_argument, nullptr, 'O'},
#ifdef GEMMSTONE_WITH_L0_RUNTIME
        {"l0", no_argument, nullptr, '0'},
#endif
        {"verbose", no_argument, nullptr, 'v'},
        {"dumpdata", no_argument, nullptr, 'Z'},
        {"no-derived", no_argument, nullptr, 'N'},
        {"derived", required_argument, nullptr, 'D'},
        {"alpha", required_argument, nullptr, '{'},
        {"beta", required_argument, nullptr, '}'},
        {"tight-ld", no_argument, nullptr, '*'},
        {nullptr, 0, nullptr, 0}
    };

    opterr = 0;

    while (true) {
        int opt =
            getopt_long(argc, argv,
#ifdef _WIN32
                        "+ade:fgik:lop:r:tu:vw:x:y:JKLNORUTYZA:C:D:E:F:P:S:c:m:0?",
#else
                        "+ade:fgijk:lop:r:tu:vw:x:y:JKLNORUTYZA:C:D:E:F:M:P:S:c:m:0?",
#endif
                        longopts, nullptr);
        if (opt == -1) break;

        switch (opt) {
            case 'a':
                CurrentMode = Mode::Add;
                break;
            case 'f':
                CurrentMode = Mode::Fit;
                break;
            case 'k':
                CurrentMode = Mode::Clone;
                ModeArgument = optarg;
                break;
            case 'y':
                EvalCheckPerformance = true;
                /* fall through */
            case 'e':
                CurrentMode = Mode::Evaluate;
                EvalSize = readSizeParams(optarg);
                break;
            case 'i':
                CurrentMode = Mode::Init;
                break;
            case 'l':
                CurrentMode = Mode::List;
                break;
            case '&':
                CurrentMode = Mode::ListIDs;
                break;
            case 'p':
                CurrentMode = Mode::Append;
                OtherCatalogPath = resolveCatalogArg(optarg);
                break;
            case 'x':
                CurrentMode = Mode::Extract;
                OtherCatalogPath = resolveCatalogArg(optarg);
                break;
            case 'r':
                if (CurrentMode == Mode::None)
                    CurrentMode = Mode::Restrict;
                Restrictions.push_back(readRestrictions(optarg));
                break;
            case 'd':
                CurrentMode = Mode::Delete;
                break;
#ifndef _WIN32
            case 'M':
                CurrentMode = Mode::Modify;
                ModifyCommand = optarg;
                break;
#endif
            case 'R':
                CurrentMode = Mode::Refit;
                break;
            case 'I':
                CurrentMode = Mode::Reinfo;
                break;
            case 'o':
                CurrentMode = Mode::Replace;
                break;
            case 'g':
                CurrentMode = Mode::TestGeneration;
                break;
#ifndef _WIN32
            case 'j':
                CurrentMode = Mode::TestCompare;
                break;
#endif
            case 't':
                CurrentMode = Mode::Test;
                break;
            case 'u':
                CurrentMode = Mode::Upgrade;
                UpgradeType = optarg;
                break;
            case 'w':
                CurrentMode = Mode::Downgrade;
                UpgradeType = optarg;
                break;
            case 'J':
                TestVariations = true;
                /* fall through */
            case '@':
                CurrentMode = Mode::RemainderReport;
                break;
            case 'L':
                TestVariations = true;
                /* fall through */
            case '#':
                CurrentMode = Mode::LDReport;
                break;
            case '!':
                CurrentMode = Mode::WipeConstant;
                break;
            case '\\':
                CurrentMode = Mode::WipeMemory;
                break;
            case 'm':
                FitModel = std::toupper(*optarg);
                break;
            case 'P': {
                auto eq = strchr(optarg, '=');
                if (!eq || !eq[1])
                    fatal("invalid parameter setting");
                std::string key(optarg, eq);
                double val = atof(eq + 1);
                auto loc = DefaultModelParams.find(key);
                if (loc == DefaultModelParams.end())
                    fatal("unknown parameter name");
                else
                    loc->second = val;
                break;
            }
            case 'v':
                Verbose = true;
                break;
            case 'A': {
                std::string a = optarg;
                auto colon = a.find(':');
                if (colon != std::string::npos) {
                    Family = a.substr(colon + 1);
                    a.resize(colon);
                }
                Arch = genToTag(a);
                if (!Family.empty()) {
                    auto fam = parseFamily(Family);
                    if (fam == ngen::ProductFamily::Unknown)
                        fatal("unknown product family");
                    if (ngen::getCore(fam) != parseGen(tagToGen(Arch, true)))
                        fatal("product family does not belong to the requested architecture");
                }
                break;
            }
            case 'S':
                Stepping = strtol(optarg, nullptr, 10);
                break;
            case 'E':
                EUCount = strtol(optarg, nullptr, 10);
                break;
            case 'F':
                NominalFrequency = strtod(optarg, nullptr) * 1e6;
                break;
            case 'C':
                LLCSize = strtod(optarg, nullptr) * 1048576;
                break;
            case 'c':
                CatalogPath = resolveCatalogArg(optarg);
                CatalogInDbDir = !std::filesystem::path(optarg).has_parent_path();
                break;
            case 'K':
                ReadKernelFromArgv = true;
                break;
            case 'U':
                Sorted = false;
                break;
            case 'T':
                EvalEffective = true;
                break;
            case 'O':
                UseL0 = false;
                break;
#ifdef GEMMSTONE_WITH_L0_RUNTIME
            case '0':
                UseL0 = true;
                break;
#endif
            case 'Y':
                CSVOutput = true;
                break;
            case 'Z':
                DumpData = true;
                break;
            case 'N':
                NoDerived = true;
                break;
            case 'D':
                DerivedType = optarg;
                break;
            case '{':
                TestAlpha = optarg;
                break;
            case '}':
                TestBeta = optarg;
                break;
            case '*':
                TestTightLD = true;
                break;
            default:
                usage(argv[0]);
                break;
        }
    }

    if (CurrentMode == Mode::None)
        usage(argv[0]);
}

int main(int argc, char **argv)
{
    bool test_ok = true;

    auto showFitDataEnv = getenv("SHOW_FIT_DATA");
    ShowFitData = showFitDataEnv && strcmp(showFitDataEnv, "0") != 0;

    parseArgs(argc, argv);

    if (Verbose && !ReadKernelFromArgv && std::filesystem::exists(CatalogPath))
        fprintf(stderr, "Note: using kernel catalog at %s\n", CatalogPath.c_str());

    if (ReadKernelFromArgv) {
        auto entry = readKernelDesc(argc, argv);
        entry.model.id = 'W';
        fitModel(entry);
        addDefaultRestrictions(entry);
        addToCatalog(std::move(entry));
    }

    switch (CurrentMode) {
        case Mode::Init:
            CatalogUpdated = true;
            break;
        case Mode::Fit: {
            auto entry = readKernelDesc(argc, argv);
            if (!FitModel)
                FitModel = 'E';
            entry.model.id = FitModel;
            fitModel(entry);
            displayModel(std::cout, entry);
            break;
        }
        case Mode::Add: {
            auto entry = readKernelDesc(argc, argv);
            entry.model.id = FitModel;
            fitModel(entry);
            addDefaultRestrictions(entry);
            for (auto &spec: Restrictions)
                applyRestriction(entry, spec);
            readCatalog();
            int ientry = addToCatalog(std::move(entry));
            displayEntry(std::cout, entry, false, ientry);
            break;
        }
        case Mode::Clone: {
            readCatalog();

            std::string newPrecisions[3];
            std::vector<kcatalog::Entry> clones;

            auto newArch = genToTag(ModeArgument, true);
            if (newArch == '?')
                readPrecisions(ModeArgument, newPrecisions);

            for (auto iter = readKernelFilter(argc, argv); iter; iter++)
                clones.push_back(*iter);

            for (auto &clone: clones) {
                if (newArch != '?')
                    clone.selector.hw = newArch;
                else for (int i = 0; i < 3; i++)
                    clone.selector.precisions[i] = newPrecisions[i];
                int ientry = addToCatalog(std::move(clone));
                displayEntry(std::cout, clone, false, ientry);
            }
            break;
        }
        case Mode::Delete: {
            readCatalog();
            std::vector<int> removals;
            for (auto iter = readKernelFilter(argc, argv, false); iter; iter++)
                removals.push_back(iter.index());
            for (auto idx = removals.rbegin(); idx < removals.rend(); idx++)
                removeFromCatalog(*idx);
            if (removals.empty())
                std::cout << "No matching kernels.\n";
            else {
                std::cout << "Removed kernel ";
                bool first = true;
                for (auto idx: removals) {
                    if (!first) std::cout << ", ";
                    first = false;
                    std::cout << '#' << idx;
                }
                std::cout << std::endl;
            }
            break;
        }
        case Mode::Refit:
            readCatalog();
            Arch = hostArch();

            for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
                displayEntry(std::cout, *iter, false, iter.index());
                if (FitModel) iter->model.id = FitModel;
                refitModel(*iter);
                std::cout << "New ";
                displayModel(std::cout, *iter);
                catalogUpdated();
            }
            break;
        case Mode::Reinfo: {
            int updated = 0;
            readCatalog();

            for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
                Arch = iter->selector.hw;
                extractStrategyParameters(*iter);
                updated++;
            }

            if (updated > 0) catalogUpdated();
            std::cout << updated << " entry(s) updated.\n";
            break;
        }
        case Mode::Replace: {
            readCatalog();
            auto iter = readKernelFilter(argc, argv, false);
            if (!iter)
                fatal("no matching kernel to replace");

            auto &entry = *iter;
            Arch = entry.selector.hw;

            std::cout << "Old: ";
            displayEntry(std::cout, entry, true, iter.index());
            std::cout << std::endl;

            auto checkArgc = [=]() {
                if (optind >= argc)
                    fatal("unexpected or ill-formed replacement strategy");
            };
            checkArgc();
            entry.driverInfo.unroll[LoopM] = atoi(argv[optind++]);
            checkArgc();
            entry.driverInfo.unroll[LoopN] = atoi(argv[optind++]);

            // Ignore alpha/beta for now.
            if (optind < argc && !std::isalpha(argv[optind][0]))
                optind += 2;

            entry.strategy.clear();
            while (optind < argc) {
                if (!entry.strategy.empty())
                    entry.strategy += ' ';
                entry.strategy += argv[optind++];
            }
            if (entry.strategy.empty())
                fatal("no replacement strategy given");

            extractStrategyParameters(entry);
            if (entry.model.id == 'E')
                refitModel(entry);

            std::cout << "New: ";
            displayEntry(std::cout, entry, true, iter.index());
            std::cout << std::endl;

            catalogUpdated();
            break;
        }
        case Mode::List:
        case Mode::ListIDs: {
            readCatalog();

            bool idsOnly = (CurrentMode == Mode::ListIDs);
            std::stringstream save;
            std::ostream &stream = (Verbose || idsOnly) ? std::cout : save;
            bool first = true;

            for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
                bool ok = true;
                for (auto &spec: Restrictions)
                    ok &= hasRestriction(*iter, spec);
                if (!ok) continue;

                if (!first)
                    stream << (idsOnly ? ' ' : '\n');
                first = false;
                if (idsOnly)
                    stream << iter.index();
                else
                    displayEntry(stream, *iter, !Verbose, iter.index());
            }

            if (first) {
                if (Catalog.entryCount > 0)
                    stream << "No matching kernels.\n";
                else
                    stream << "Empty or nonexistent catalog.\n";
            }

            if (!Verbose) {
                stream << std::endl;
                tableOutput(std::cout, save.str());
            }
            break;
        }
        case Mode::Append: {
            kcatalog::Catalog otherCatalog;

            std::swap(CatalogPath, OtherCatalogPath);
            readCatalog();
            std::swap(Catalog, otherCatalog);
            std::swap(CatalogPath, OtherCatalogPath);
            readCatalog();

            std::vector<int> added;

            for (auto &entry: otherCatalog.entries) {
                int ientry = addToCatalog(std::move(entry));
                for (auto &i: added)
                    if (i >= ientry) i++;
                added.push_back(ientry);
            }

            std::cout << "Added " << otherCatalog.entryCount << " kernel(s) to catalog.\n";
            for (auto ientry: added) {
                displayEntry(std::cout, Catalog.entries[ientry], true, ientry);
                std::cout << std::endl;
            }
            break;
        }
        case Mode::Extract: {
            readCatalog();

            std::vector<int> extracts;
            for (auto iter = readKernelFilter(argc, argv, false); iter; iter++)
                extracts.push_back(iter.index());

            if (extracts.empty())
                std::cout << "No matching kernels.\n";
            else {
                kcatalog::Catalog otherCatalog;
                std::swap(Catalog, otherCatalog);
                std::swap(CatalogPath, OtherCatalogPath);
                readCatalog();

                std::vector<int> added;
                for (auto idx: extracts) {
                    int ientry = addToCatalog(std::move(otherCatalog.entries[idx]));
                    for (auto &i: added)
                        if (i >= ientry) i++;
                    added.push_back(ientry);
                }

                std::cout << "Added " << added.size() << " kernel(s) to catalog.\n";
                for (auto ientry: added) {
                    displayEntry(std::cout, Catalog.entries[ientry], true, ientry);
                    std::cout << std::endl;
                }
            }
            break;
        }
        case Mode::Restrict:
            readCatalog();
            for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
                CatalogUpdated = true;
                for (auto &spec: Restrictions)
                    applyRestriction(*iter, spec);
            }
            break;
        case Mode::Evaluate: {
            double alpha, beta;

            auto mparams = readMatchParams(argc, argv);
            std::tie(alpha, beta) = readScalars(argc, argv);

            auto alphaString = scalarString(alpha);
            auto betaString = scalarString(beta);
            auto &ktype = mparams.selector.kernelType;

            std::vector<DataPoint> esizes(1);
            auto &p = esizes[0];
            p.m = mparams.sizes.m;
            p.n = mparams.sizes.n;
            p.k = mparams.sizes.k;
            p.batch = mparams.sizes.batch;

            EvaluateParams eparams;
            eparams.sizes = mparams.sizes;
            eparams.alpha = alpha;
            eparams.beta = beta;
            eparams.euCount = euCount();
            eparams.product = targetProduct();
            eparams.effective = EvalEffective;
            eparams.cConvert = (mparams.selector.precisions[2][0] == '[');
            eparams.postOps = contains(ktype, "offsetc") || contains(ktype, "offsetac")
                           || contains(ktype, "offsetbc") || contains(ktype, "offsetabc");  /* klunky */
            eparams.batch = contains(ktype, "batch");

            readCatalog();

            using ResultType = std::tuple<double, const kcatalog::Entry *, EvaluateAuxOutput>;
            std::vector<ResultType> results;

            int bestAlignA = 1, bestAlignB = 1;
            std::function observer = [&](const kcatalog::Entry *entry, double score, EvaluateAuxOutput aux) {
                results.push_back({score, entry, aux});
                bestAlignA = std::max(bestAlignA, std::max(entry->restrictions.alignment[0], 4));
                bestAlignB = std::max(bestAlignB, std::max(entry->restrictions.alignment[1], 4));
            };

            EvaluateAuxOutput _;
            select(Catalog, mparams, eparams, _, &observer);

            results.erase(std::remove_if(results.begin(), results.end(), [=](const auto &result) {
                auto &entry = *std::get<1>(result);
                return (entry.restrictions.tags[0] == kcatalog::ReqAlignFallback
                            && lessAligned(std::max(entry.restrictions.alignment[0], 4),
                                           std::max(entry.restrictions.alignment[1], 4),
                                           bestAlignA, bestAlignB));
            }), results.end());

            if (Sorted) {
                std::sort(results.begin(), results.end(), [](const ResultType &r1, const ResultType &r2) {
                    return (std::get<0>(r1) < std::get<0>(r2));
                });
            }

            auto run1 = [&](const kcatalog::Entry &entry, const EvaluateAuxOutput &aux) -> double {
                if (aux.kParallelVariable || (aux.kParallel && entry.driverInfo.kParallelVariable()))
                    utils::unsetenv("K_SLICED_GROUPS");
                else
                    utils::setenv("K_SLICED_GROUPS", "0", true);
                if (entry.driverInfo.kParallelLocal() && !aux.kParallel)
                    utils::setenv("K_BLOCKS", std::to_string(aux.wgK).c_str(), true);
                else
                    utils::unsetenv("K_BLOCKS");
                if (mustUseBeta1(entry.strategy) && beta != 1)
                    return std::numeric_limits<double>::quiet_NaN();
                auto res = prepareKernel(entry, alphaString, betaString, "", &mparams.selector);
                collectData(*res, esizes, false, false, true);
                return p.nominal;
            };

            if (CSVOutput) {
                std::cout << (EvalCheckPerformance ? "Kernel,Time,Cost\n"
                                                   : "Kernel,Cost\n");
                for (const auto &result: results) {
                    auto &entry = *std::get<1>(result);
                    auto &aux = std::get<2>(result);
                    std::cout << '"' << frontendInputString(entry, alphaString, betaString, "", &mparams.selector) << "\",";
                    if (EvalCheckPerformance)
                        std::cout << run1(entry, aux) << ',';
                    std::cout << std::get<0>(result) << std::endl;
                }
            } else if (results.empty())
                std::cout << "No matching kernels.\n";
            else {
                std::stringstream save;
                std::ostream &stream = EvalCheckPerformance ? std::cout : save;

                auto e0 = std::get<1>(results[0]);
                stream << "Using " << tagToGen(e0->selector.hw, true) << ' ';
                for (int d = 0; d < 3; d++)
                    stream << e0->selector.precisions[d];
                stream << " strategies.\n\n";

                stream << (EvalCheckPerformance ? "Cost\tTime\tKernel\n"
                                                : "Cost\tKernel\n");
                for (const auto &result: results) {
                    auto score = std::get<0>(result);
                    auto &entry = *std::get<1>(result);
                    auto &aux = std::get<2>(result);

                    stream << score;
                    if (Verbose && one_of(entry.model.id, {'S', 'E'})) {
                        stream << " (c = " << aux.ctime;
                        stream << ", m = " << aux.mtime;
                        stream << ", e = " << aux.etime << ')';
                    }
                    if (EvalCheckPerformance)
                        stream << '\t' << Duration{run1(entry, aux)};
                    stream << "\t#";
                    stream << (&entry - &Catalog.entries[0]) << ' ';
                    stream << frontendInputString(entry, alphaString, betaString, "", &mparams.selector);
                    if (entry.driverInfo.kParallelVariable()) {
                        stream << "\t(global k-slicing: " ;
                        if (aux.kParallelVariable) stream << "variable";
                        else if (aux.kParallel) stream << "fixed";
                        else stream << "none";
                        stream << ')';
                    }
                    if (!aux.kParallelVariable && (aux.kParallel || entry.driverInfo.kParallelLocal()))
                        stream << "\t(k block size = " << aux.k0 << ')';
                    stream << std::endl;
                }

                if (!EvalCheckPerformance)
                    tableOutput(std::cout, save.str());
            }
            break;
        }
        case Mode::Test:
            Arch = hostArch();
            // fall through
#ifndef _WIN32
        case Mode::TestCompare:
#endif
        case Mode::TestGeneration: {
            readCatalog();

            if (UseL0)
                utils::setenv("TARGET", "l0", false);

            auto testKind = (CurrentMode == Mode::Test) ? TestKind::Accuracy
#ifndef _WIN32
                          : (CurrentMode == Mode::TestCompare) ? TestKind::Compare
#endif
                          : TestKind::Generation;

            std::set<char> unsupportedHWWarned;
            for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
                if (contains(iter->restrictions.tags, kcatalog::ReqDisabled))
                    continue;
                if (contains(iter->selector.kernelType, "trsm") || contains(iter->selector.kernelType, "trmm"))
                    continue;
                auto hw = iter->selector.hw;
                if (!frontend::Generator::IsSupported(parseGen(tagToGen(hw)))) {
                    if (unsupportedHWWarned.insert(hw).second)
                        std::cout << "\x1B[1;33m[Skipping " << tagToGen(hw, true)
                                  << " kernels: support not compiled for this hardware]\x1B[0m" << std::endl;
                    continue;
                }
                displayEntry(std::cout, *iter, true, iter.index());
                std::cout << std::endl;
                test_ok &= testEntry(*iter, testKind);
            }

            break;
        }
        case Mode::Downgrade:
        case Mode::Upgrade: {
            bool downgrade = (CurrentMode == Mode::Downgrade);

            readCatalog();

            if (UpgradeType == "pfdist")
                upgradePrefetchDistance(argc, argv, downgrade);
            else if (UpgradeType == "pfbase")
                upgradePrefetchBase(argc, argv, downgrade);
            else if (UpgradeType == "blockrelax")
                upgradeBlock2DRelax(argc, argv, downgrade);
            else if (UpgradeType == "blockfb")
                upgradeBlock2DFallbacks(argc, argv, downgrade);
            else if (UpgradeType == "kvkr")
                upgradeKVWithKR(argc, argv, downgrade);
            else if (UpgradeType == "retag")
                upgradeRetag(argc, argv, downgrade);
            else if (UpgradeType == "micro")
                upgradeMicrokernel(argc, argv, downgrade);
            else
                fatal("unknown upgrade type requested");

            break;
        }
        case Mode::RemainderReport:
        case Mode::LDReport: {
            bool rem_report = (CurrentMode == Mode::RemainderReport);
            bool ld_report = (CurrentMode == Mode::LDReport);

            readCatalog();
            Arch = hostArch();

            utils::setenv("A", "ZERO", false);
            utils::setenv("B", "ZERO", false);
            utils::setenv("ENABLE_METRICS", "0", false);

            for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
                auto &e = *iter;
                auto modified = e;
                DataPoint p;

                // Find a nice size to fill the GPU.
                EvaluateParams eparams = {{0, 0, 0, 0}, 1., 0., euCount()};
                eparams.product = targetProduct();
                auto dp = getDerivedParams(e, eparams);

                int wgK = e.driverInfo.wg[LoopK];
                int wgTileM = e.driverInfo.wgTile(LoopM);
                int wgTileN = e.driverInfo.wgTile(LoopN);
                int wgPerWave = dp.hwThreadCapacity / e.driverInfo.threadsPerWG();

                findSquarishSize(p.m, p.n, wgTileM, wgTileN, wgPerWave);
                p.k = alignUp(256, 2 * e.driverInfo.unroll[LoopK]) * wgK;

                auto aElemBytes = extPrecisionBytes(e.selector.precisions[0], false);
                auto bElemBytes = extPrecisionBytes(e.selector.precisions[1], false);
                auto cElemBytes = extPrecisionBytes(e.selector.precisions[2], true);

                if (ld_report) {
                    auto targetMN = mnPerformanceSize(e.selector.precisions[0][0]);
                    p.m = alignUp(targetMN, p.m);
                    p.n = alignUp(targetMN, p.n);
                    p.k = alignUp(4096, p.k);
                }

                getenvInt("BASE_M", p.m);
                getenvInt("BASE_N", p.n);
                getenvInt("BASE_K", p.k);

                if (e.restrictions.allowedSizesMax[LoopM] > 0)
                    p.m = std::min<int64_t>(p.m, alignUp(e.restrictions.allowedSizesMax[LoopM], e.driverInfo.unroll[LoopM]));
                if (e.restrictions.allowedSizesMax[LoopN] > 0)
                    p.n = std::min<int64_t>(p.n, alignUp(e.restrictions.allowedSizesMax[LoopN], e.driverInfo.unroll[LoopN]));
                if (e.restrictions.allowedSizesMax[LoopK] > 0)
                    p.k = std::min<int64_t>(p.k, alignUp(e.restrictions.allowedSizesMax[LoopK], e.driverInfo.unroll[LoopK]));

                if (ld_report) {
                    p.lda = niceLD(e.selector.layouts[0], aElemBytes, p.m, p.k);
                    p.ldb = niceLD(e.selector.layouts[1], bElemBytes, p.k, p.n);
                    p.ldc = niceLD(e.selector.layouts[2], cElemBytes, p.m, p.n);
                }

                if (Verbose)
                    fprintf(stderr, "Note: base testing size %d x %d x %d.\n", int(p.m), int(p.n), int(p.k));

                // Get test sizes.
                int ncases = ld_report ? 6 : 3;
                std::vector<DataPoint> cases(ncases, p);
                std::vector<const char *> caseNames(ncases);

                if (rem_report) {
                    cases[1].m--;
                    cases[1].n--;
                    cases[2].k--;
                    caseNames[0] = " no remainder";
                    caseNames[1] = "m/n remainder";
                    caseNames[2] = "  k remainder";
                } else if (ld_report) {
                    cases[1].lda += std::max<int>(1, e.restrictions.alignment[0] / aElemBytes);
                    cases[2].ldb += std::max<int>(1, e.restrictions.alignment[1] / bElemBytes);
                    cases[3].ldc += std::max<int>(1, e.restrictions.alignment[2] / cElemBytes);
                    cases[4] = cases[1];
                    cases[4].m -= std::max<int>(1, 4 / aElemBytes);
                    cases[5] = cases[2];
                    cases[5].n -= std::max<int>(1, 4 / bElemBytes);
                    caseNames[0] = "                 aligned LDs";
                    caseNames[1] = "              misaligned lda";
                    caseNames[2] = "              misaligned ldb";
                    caseNames[3] = "              misaligned ldc";
                    caseNames[4] = "A remainder + misaligned lda";
                    caseNames[5] = "B remainder + misaligned ldb";
                }

                for (auto &c : cases) {
                    c.m = std::max<int64_t>(c.m, 1);
                    c.n = std::max<int64_t>(c.n, 1);
                    c.k = std::max<int64_t>(c.k, 0);
                }

                // Get alternate configurations to test.
                std::vector<std::string> configs;
                configs.push_back("");
                if (rem_report) {
                    if (!contains(e.strategy, "sr")) configs.push_back(" sr");
                    if (Arch >= kcatalog::HWTagXeHPC && !contains(e.strategy, "br")) configs.push_back(" br");
                    if (!contains(e.strategy, "dm")) configs.push_back(" dm");
                    if (!contains(e.strategy, "kd")) configs.push_back(" kd");
                    if (e.strategy[0] == 's') configs.push_back(" pab");
                    if (Arch >= kcatalog::HWTagXeHPC && one_of(e.selector.precisions[0][0], {'S', 'D', 'C', 'Z'})) {
                        auto words = extractWords(e.strategy);
                        std::string base;
                        if (extPrecisionBytes(e.selector.precisions[0], false) < 8)
                            base = " l8";
                        switch (std::tolower(words[0][1])) {
                            case 's': configs.push_back(base + " lat"); break;
                            case 'b':
                                if (!contains(e.strategy, "xa"))
                                    configs.push_back(base + " lam");
                                break;
                        }
                        switch (std::tolower(words[1][1])) {
                            case 's': configs.push_back(base + " lbt"); /* fall through */
                            case 'b':
                                if (!contains(e.strategy, "xb"))
                                    configs.push_back(base + " lbm");
                                break;
                        }
                    }
                    if (Arch >= kcatalog::HWTagXeHPC && contains(e.strategy, '@'))
                        configs.push_back(" p2d");
                }

                if (ld_report) {
                    auto words = extractWords(e.strategy);
                    if (!contains(e.strategy, " ca") && !contains(e.strategy, " cb")) {
                        if (!contains(e.strategy, "sb32")) configs.push_back(" sb32");
                        if (!contains(e.strategy, "sb256")) configs.push_back(" sb256");
                    }
                    if (contains(words[0], '@')) configs.push_back(" xpfa8");
                    if (contains(words[1], '@')) configs.push_back(" xpfb8");
                }

                if (!TestVariations)
                    configs.resize(1);

                displayEntry(std::cout, e, true, iter.index());
                std::cout << std::endl;

                double baseMin0 = std::numeric_limits<double>::infinity();
                bool gotBaseMin0 = false;

                std::vector<std::vector<double>> diffs(ncases);

                for (auto config: configs) {
                    modified.strategy = e.strategy + config;
                    auto res = prepareKernel(modified, "1", mustUseBeta1(e.strategy) ? "1" : "0", "", nullptr, false);
                    if (res) {
                        std::vector<double> mins(ncases, std::numeric_limits<double>::infinity());
                        const int trials = 3;

                        for (int i = 0; i < trials; i++) {
                            collectData(*res, cases, false, false);
                            for (int j = 0; j < ncases; j++)
                                mins[j] = std::min(mins[j], cases[j].nominal);
                        }

                        if (!gotBaseMin0) {
                            baseMin0 = mins[0];
                            gotBaseMin0 = true;
                        }

                        for (int j = 0; j < ncases; j++)
                            diffs[j].push_back(baseMin0 / mins[j] - 1);
                    } else {
                        for (int j = 0; j < ncases; j++)
                            diffs[j].push_back(std::numeric_limits<double>::quiet_NaN());
                    }
                }

                std::stringstream save;
                save << std::setprecision(3) << std::showpos;
                if (TestVariations) for (auto config: configs)
                    save << '\t' << config << ' ';
                for (int j = 0; j < ncases; j++) {
                    save << "\n  " << caseNames[j] << ':';
                    for (auto d: diffs[j]) {
                        if (!std::isnan(d))
                            save << '\t' << 100 * d << '%';
                        else
                            save << "\t --";
                    }
                }
                save << std::endl;

                tableOutput(std::cout, save.str());
            }

            break;
        }
#ifndef _WIN32
        case Mode::Modify:
            readCatalog();
            for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
                auto newStrategy = outputFromProcess(ModifyCommand, {std::string(iter->strategy)});
                auto lastNewline = newStrategy.find('\n');
                if (lastNewline != std::string::npos)
                    newStrategy.erase(lastNewline);
                if (newStrategy.empty())
                    fatal("new strategy is empty -- check modify command");

                std::cout << "Old: ";
                displayEntry(std::cout, *iter, true, iter.index());
                std::cout << std::endl;

                iter->strategy = newStrategy;

                if (contains(newStrategy, "l4")) {
                    if (iter->restrictions.alignment[0] == 4)
                        iter->restrictions.alignment[0] = 1;
                    if (iter->restrictions.alignment[1] == 4)
                        iter->restrictions.alignment[1] = 1;
                }

                std::cout << "New: ";
                displayEntry(std::cout, *iter, true, iter.index());
                std::cout << std::endl;
                CatalogUpdated = true;
            }
            break;
#endif
        case Mode::WipeConstant:
            readCatalog();
            for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
                if (iter->model.id != 'S') continue;
                iter->model.params[kcatalog::ParamS_C00] = 0.;
                iter->model.params[kcatalog::ParamS_C01] = 0.;
                iter->model.params[kcatalog::ParamS_C10] = 0.;
                iter->model.params[kcatalog::ParamS_C11] = 0.;
                iter->model.params[kcatalog::ParamS_Cm0] = 0.;
                iter->model.params[kcatalog::ParamS_Cm1] = 0.;
                CatalogUpdated = true;
            }
            break;
        case Mode::WipeMemory:
            readCatalog();
            for (auto iter = readKernelFilter(argc, argv); iter; iter++) {
                if (iter->model.id != 'S') continue;
                iter->model.params[kcatalog::ParamS_Ma] = 0.;
                iter->model.params[kcatalog::ParamS_Mb] = 0.;
                CatalogUpdated = true;
            }
            break;
        default: break;
    }

    if (CatalogUpdated)
        writeCatalog();

    return test_ok ? 0 : -1;
}
