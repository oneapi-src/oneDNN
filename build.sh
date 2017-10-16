#!/bin/bash
# vim: et ts=4 sw=4
#
# Usage: ./build.sh -h
#   Running cmake by hand is all you really need.
#   Something basic like that can be done with:
#      ./build.sh -jq
#                 (jit, quick=no docs, built under build-jit/)
#
#   This script began to support my cross-compiling environment,
#   with an alternate engine with JIT support removed.
#
ORIGINAL_CMD="$0 $*"
usage() {
    echo "$0 usage:"
    #head -n 30 "$0" | grep "^[^#]*.)\ #"
    awk '/getopts/{flag=1;next} /done/{flag=0} flag&&/^[^#]+\) #/; flag&&/^ *# /' $0
    echo "Example: time a full test run for a debug compilation --- time $0 -dtt"
    echo "         SX debug compile, quick (no doxygen)         --- time $0 -Sdq"
    echo "         *just* run cmake, for SX debug compile       ---      $0 -SdQ"
    echo "         *just* create doxygen docs                   ---      $0 -D"
    echo "Debug: Individual tests can be run like build-sx/tests/gtests/test_relu"
    exit 0
}
if [ "${CC##sx}" == "sx" -o "${CXX##sx}" == "sx" ]; then
    DOTARGET="s" # s for SX (C/C++ code, cross-compile)
elif [ -d src/vanilla ]; then
    DOTARGET="v" # v for vanilla (C/C++ code)
else
    DOTARGET="j" # j for JIT (Intel assembler)
fi
DOTEST=0
DODEBUG="n"
DODOC="y"
DONEEDMKL="y"
DOJUSTDOC="n"
DOWARN="y"
BUILDOK="y"
SIZE_T=32 # or 64, for -s or -S SX compile
JOBS="-j8"
#JOBS="-j1"
CMAKETRACE=""
USE_CBLAS=1
while getopts ":htvjdDqQpsSTb" arg; do
    #echo "arg = ${arg}, OPTIND = ${OPTIND}, OPTARG=${OPTARG}"
    case $arg in
        t) # [0] increment test level: (1) examples, (2) tests (longer), ...
            # Apr-14-2017 build timings:
            # 0   : build    ~ ?? min  (jit), 1     min  (vanilla)
            # >=1 : examples ~  1 min  (jit), 13-16 mins (vanilla)
            # >=2 : test_*   ~ 10 mins (jit), 108   mins (vanilla)
            # >=3 : benchdnn (bench.sh) performance/correctness tests (long)
            DOTEST=$(( DOTEST + 1 ))
            ;;
        v) # [yes] (vanilla C/C++ only: no src/cpu/ JIT assembler)
            if [ -d src/vanilla ]; then DOTARGET="v"; fi
            ;;
        j) # force Intel JIT (src/cpu/ JIT assembly code)
            DOTARGET="j"; DOJIT=100 # 100 means all JIT funcs enabled
            ;;
        d) # [no] debug release
            DODEBUG="y"
            ;;
        D) # [no] Doxygen-only : build documentation and then stop
            DOJUSTDOC="y"
            ;;
        q) # quick: skip doxygen docs [default: run doxygen if build OK]
            DODOC="n"
            ;;
        Q) # really quick: skip build and doxygen docs [JUST run cmake and stop]
            BUILDOK="n"; DODOC="n"
            ;;
        p) # permissive: disable the FAIL_WITHOUT_MKL switch
            DONEEDMKL="n"
            ;;
        S) # SX cross-compile (size_t=64, built in build-sx/, NEW: default if $CC==sxcc)
            DOTARGET="s"; DOJIT=0; SIZE_T=64; JOBS="-j4"
            ;;
        s) # SX cross-compile (size_t=32, built in build-sx/) DISCOURAGED
            # -s is NOT GOOD: sizeof(ptrdiff_t) is still 8 bytes!
            DOTARGET="s"; DOJIT=0; SIZE_T=32; JOBS="-j4"
            echo "*** WARNING ***"
            echo "-s --> -size_t32 compilation NOT SUPPORTED (-S is recommended)"
            echo "***************"
            ;;
        r) # reference impls only: no -DUSE_CBLAS compile flag (->no im2col gemm)
            USE_CBLAS=0
            ;;
        w) # reduce compiler warnings
            DOWARN=0
            ;;
        W) # lots of compiler warnings (default)
            DOWARN=1
            ;;
        T) # cmake --trace
            CMAKETRACE="--trace"
            ;;
    h | *) # help
            usage
            ;;
    esac
done
DOJIT=0
INSTALLDIR=install
BUILDDIR=build
if [ "`echo ${CC}`" == 'sxcc' -a ! "$DOTARGET" == "s" ]; then
    echo 'Detected $CC == sxcc --> SX compilation with 64-bit size_t'
    DOTARGET="s"; DOJIT=0; SIZE_T=64; JOBS="-j4"
fi
#
# I have not yet tried icc.
# For gcc, we will avoid the full MKL (omp issues)
#
if [ "${MKLROOT}" != "" ]; then
	module unload icc >& /dev/null || echo "module icc unloaded"
	if [ "${MKLROOT}" != "" ]; then
		echo "Please compile in an environment without MKLROOT"
		exit -1;
	fi
	# export -n MKLROOT
	# export MKL_THREADING_LAYER=INTEL # maybe ???
fi
#
#
#
if [ "$DOTARGET" == "j" ]; then DOJIT=100; INSTALLDIR='install-jit'; BUILDDIR='build-jit'; fi
if [ "$DOTARGET" == "s" ]; then DONEEDMKL="n"; DODOC="n"; DOTEST=0; INSTALLDIR='install-sx'; BUILDDIR='build-sx'; fi
#if [ "$DOTARGET" == "v" ]; then ; fi
if [ "$DODEBUG" == "y" ]; then INSTALLDIR="${INSTALLDIR}-dbg"; BUILDDIR="${BUILDDIR}d"; fi
if [ "$DOJUSTDOC" == "y" ]; then
    (
        if [ ! -d build ]; then mkdir build; fi
        if [ ! -f build/Doxyfile ]; then
            # doxygen does not much care HOW to build, just WHERE
            (cd build && cmake -DCMAKE_INSTALL_PREFIX=../${INSTALL_DIR} -DFAIL_WITHOUT_MKL=OFF ..)
        fi
        echo "Doxygen (please be patient) logging to doxygen.log"
        rm -rf build/doc*stamp build/reference "${INSTALL_DIR}/share/doc"
        #cd build \
        #&& make VERBOSE=1 doc \
        #&& cmake -DCOMPONENT=doc -P cmake_install.cmake
        cd build && make VERBOSE=1 install-doc # Doxygen.cmake custom target
        echo "doxygen.log ends up in gen-dnn project root"
        echo "Documentation installed under ${INSTALL_DIR}/share/doc/"
    ) 2>&1 | tee ../doxygen.log
    exit 0
fi
timeoutPID() { # unused
    PID="$1"
    timeout="$2"
    interval=1
    delay=1
    (
        ((t = timeout))

        while ((t > 0)); do
            sleep $interval
            kill -0 $$ || exit 0
            ((t -= interval))
        done

        # Be nice, post SIGTERM first.
        # The exit 0 below will be executed if any preceeding command fails.
        kill -s SIGTERM $$ && kill -0 $$ || exit 0
        sleep $delay
        kill -s SIGKILL $$
    ) 2> /dev/null &
}
if [ -d "${BUILDDIR}" ]; then
    rm -rf "${BUILDDIR}".bak && mv -v "${BUILDDIR}" "${BUILDDIR}".bak
    if [ -f "${BUILDDIR}.log" ]; then
       mv "${BUILDDIR}.log" "${BUILDDIR}".bak/
    fi
fi
if [ -d "$INSTALLDIR}" ]; then
    rm -rf "$INSTALLDIR}".bak && mv -v "$INSTALLDIR}" "$INSTALLDIR}".bak
fi
(
    echo "DOTARGET   $DOTARGET"
    echo "DOJIT      $DOJIT"
    echo "DOTEST     $DOTEST"
    echo "DODEBUG    $DODEBUG"
    echo "DODOC      $DODOC"
    echo "BUILDDIR   ${BUILDDIR}"
    echo "INSTALLDIR ${INSTALLDIR}"
    mkdir "${BUILDDIR}"
    cd "${BUILDDIR}"
    #
    CMAKEOPT=""
    if [ -f "include/mkldnn_io.hpp" ]; then # necla-ml fork
        CMAKEOPT="${CMAKEOPT} -DCMAKE_CCXX_FLAGS=-DJITFUNCS=${DOJIT}"
    fi
    if [ $USE_CBLAS -ne 0 ]; then
        export CFLAGS="${CFLAGS} -DUSE_CBLAS"
        export CXXFLAGS="${CXXFLAGS} -DUSE_CBLAS"
    fi
    if [ ! "$DOTARGET" == "j" ]; then
        CMAKEOPT="${CMAKEOPT} -DTARGET_VANILLA=ON"
        export CFLAGS="${CFLAGS} -DTARGET_VANILLA"
        export CXXFLAGS="${CXXFLAGS} -DTARGET_VANILLA"
    fi
    if [ ${DOWARN} == 'y' ]; then
        DOWARNFLAGS=""
        if [ "$DOTARGET" == "s" ]; then DOWARNFLAGS="-wall"
        else DOWARNFLAGS="-Wall"; fi
        export CFLAGS="${CFLAGS} ${DOWARNFLAGS}"
        export CXXFLAGS="${CXXFLAGS} ${DOWARNFLAGS}"
        #echo "DOWARN --> CFLAGS   = ${CFLAGS}"
        #echo "DOWARN --> CXXFLAGS = ${CXXFLAGS}"
    fi
    if [ "$DOTARGET" == "s" ]; then
        TOOLCHAIN=../cmake/sx.cmake
        if [ ! -f "${TOOLCHAIN}" ]; then echo "Ohoh. ${TOOLCHAIN} not found?"; BUILDOK="n"; fi
        CMAKEOPT="${CMAKEOPT} -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN}"
        CMAKEOPT="${CMAKEOPT} --debug-trycompile --trace -LAH" # long debug of cmake
        #  ... ohoh no easy way to include the spaces and expand variable properly ...
        #      Solution: do these changes within CMakeLists.txt
        #CMAKEOPT="${CMAKEOPT} -DCMAKE_C_FLAGS=-g\ -ftrace\ -Cdebug" # override Cvopt
        SXOPT="-DTARGET_VANILLA -D__STDC_LIMIT_MACROS"
        SXOPT="${SXOPT} -woff=1097 -woff=4038" # turn off warnings about not using attributes
        SXOPT="${SXOPT} -woff=1901"  # turn off sxcc warning defining arr[len0] for constant len0
        SXOPT="${SXOPT} -wnolongjmp" # turn off warnings about setjmp/longjmp (and tracing)

        SXOPT="${SXOPT} -Pauto -acct" # enable parallelization (and run with C_PROGINF=YES)
        #SXOPT="${SXOPT} -Pstack" # disable parallelization

        # Generate 'ftrace.out' profiling that can be displayed with ftrace++
        #  BUT not compatible with POSIX threads
        #SXOPT="${SXOPT} -Nftrace"
        SXOPT="${SXOPT} -ftrace demangled"

        # REMOVE WHEN FINISHED SX DEBUGGING
        SXOPT="${SXOPT} -g -traceback" # enable source code tracing ALWAYS
        SXOPT="${SXOPT} -DVERBOSE_PRIMITIVE_CREATE"

        export CFLAGS="${CFLAGS} -size_t${SIZE_T} -Kc99,gcc ${SXOPT}"
        # An object file that is generated with -Kexceptions and an object file
        # that is generated with -Knoexceptions must not be linked together. In
        # such conditions the exception may not be thrown correctly Therefore, do
        # not specify -Kexceptions if the program does not use the try, catch
        # and throw keywords.
        #export CXXFLAGS="${CXXFLAGS} -size_t${SIZE_T} -Kcpp11,gcc,rtti,exceptions ${SXOPT}"
        export CXXFLAGS="${CXXFLAGS} -size_t${SIZE_T} -Kcpp11,gcc,exceptions ${SXOPT}"
        #export CXXFLAGS="${CXXFLAGS} -size_t${SIZE_T} -Kcpp11,gcc,rtti"
        # __STDC_LIMIT_MACROS is a way to force definitions like INT8_MIN in stdint.h (cstdint)
        #    (it **should** be autmatic in C++11, imho)
    fi
    CMAKEOPT="${CMAKEOPT} -DCMAKE_INSTALL_PREFIX=../${INSTALLDIR}"
    if [ "$DODEBUG" == "y" ]; then
        CMAKEOPT="${CMAKEOPT} -DCMAKE_BUILD_TYPE=Debug"
    else
        CMAKEOPT="${CMAKEOPT} -DCMAKE_BUILD_TYPE=Release"
        #CMAKEOPT="${CMAKEOPT} -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    fi
    if [ "$DONEEDMKL" == "y" ]; then
        CMAKEOPT="${CMAKEOPT} -DFAIL_WITHOUT_MKL=ON"
    fi
    # Remove leading whitespace from CMAKEENV (bash magic)
    shopt -s extglob; CMAKEENV=\""${CMAKEENV##*([[:space:]])}"\"; shopt -u extglob
    # Without MKL, unit tests take **forever**
    #    TODO: cblas / mathkeisan alternatives?
    if [ "$BUILDOK" == "y" ]; then
        BUILDOK="n"
        rm -f ./stamp-BUILDOK ./CMakeCache.txt
        echo "${CMAKEENV}; cmake ${CMAKEOPT} ${CMAKETRACE} .."
        set -x
        { if [ x"${CMAKEENV}" == x"" ]; then ${CMAKEENV}; fi; \
            cmake ${CMAKEOPT} ${CMAKETRACE} .. \
                && make VERBOSE=1 ${JOBS} \
                && BUILDOK="y"; }
        set +x
    else # skip the build, just run cmake ...
        echo "CMAKEENV   <${CMAKEENV}>"
        echo "CMAKEOPT   <${CMAKEOPT}>"
        echo "CMAKETRACE <${CMAKETRACE}>"
        set -x
        { if [ x"${CMAKEENV}" == x"" ]; then ${CMAKEENV}; fi; \
            cmake ${CMAKEOPT} ${CMAKETRACE} .. ; }
        set +x
    fi
    set -x
    if [ "$BUILDOK" == "y" -a ! "$DOTARGET" == "s" ]; then
        echo "DOTARGET  $DOTARGET"
        echo "DOJIT     $DOJIT"
        echo "DOTEST    $DOTEST"
        echo "DODEBUG   $DODEBUG"
        echo "DODOC     $DODOC"
        # Whatever you are currently debugging (and is a quick sanity check) can go here
        if [ -x tests/api-io-c ]; then
            { echo "api-io-c                ..."; time tests/api-io-c || BUILDOK="n"; }
        else
            { echo "api-c                ..."; time tests/api-c || BUILDOK="n"; }
        fi
        if [ $DOTEST -eq 0 -a "$DOJIT" -gt 0 ]; then # this is fast ONLY with JIT (< 5 secs vs > 5 mins)
            { echo "simple-training-net-cpp ..."; time examples/simple-training-net-cpp || BUILDOK="n"; }
        fi
    fi
    if [ "$BUILDOK" == "y" -a "$DOTARGET" == "s" ]; then
        # make SX build dirs all-writable so SX runs can store logs etc.
        #find "${BUILDDIR}" -type d -exec chmod o+w {} \;
        { cd ..; find "${BUILDDIR}" -type d -exec chmod o+w {} \; ; }
    fi
    if [ "$BUILDOK" == "y" ]; then
        touch ./stamp-BUILDOK
        if [ "$DODOC" == "y" ]; then
            echo "Build OK... Doxygen (please be patient)"
            make VERBOSE=1 doc >& ../doxygen.log
        fi
    fi
    set +x
) 2>&1 | tee "${BUILDDIR}".log
ls -l "${BUILDDIR}"
BUILDOK="n"; if [ -f "${BUILDDIR}/stamp-BUILDOK" ]; then BUILDOK="y"; fi

echo "BUILDDIR   ${BUILDDIR}"
echo "INSTALLDIR ${INSTALLDIR}"
echo "DOTARGET=${DOTARGET}, DOJIT=${DOJIT}, DODEBUG=${DODEBUG}, DOTEST=${DOTEST}, DODOC=${DODOC}, DONEEDMKL=${DONEEDMKL}"
LOGDIR="log-${DOTARGET}${DOJIT}${DODEBUG}${DOTEST}${DODOC}${DONEEDMKL}"
if [ "$BUILDOK" == "y" ]; then
    echo "BUILDOK !"
    (
        cd "${BUILDDIR}"
        # trouble with cmake COMPONENTs ...
        echo "Installing :"; make install;
        #if [ "$DODOC" == "y" ]; then { echo "Installing docs ..."; make install-doc; } fi
    ) 2>&1 >> "${BUILDDIR}".log || { echo "'make install' in ${BUILDDIR} had issues"; }
    echo "Testing ?"
    if [ ! $DOTEST -eq 0 -a ! "$DOTARGET" == "s" ]; then
        rm -f test1.log test2.log test3.log
        echo "Testing ... test1"
        if [ true ]; then
            (cd "${BUILDDIR}" && ARGS='-VV -E .*test_.*' /usr/bin/time -v make test) 2>&1 | tee "${BUILDDIR}/test1.log" || true
        fi
        if [ $DOTEST -ge 2 ]; then
            echo "Testing ... test2"
            (cd "${BUILDDIR}" && ARGS='-VV -N' make test \
            && ARGS='-VV -R .*test_.*' /usr/bin/time -v make test) 2>&1 | tee "${BUILDDIR}/test2.log" || true
        fi
        if [ $DOTEST -ge 3 ]; then
            if [ -x ./bench.sh ]; then
                # all non-ref impls, performance and correctness
                /usr/bin/time -v ./bench.sh -${DOTARGET}mAPC -sref2>&1 | tee "${BUILDDIR}/test3.log" || true
            fi
        fi
        echo "Tests done"
    fi
    if [ ! $DOTEST -eq 0 -a "$DOTARGET" == "s" ]; then
        echo 'SX testing should be done manually (ex. ~/tosx script to log in to SX)'
    fi
else
    echo "Build NOT OK..."
fi
echo "BUILDDIR   ${BUILDDIR}"
echo "INSTALLDIR ${INSTALLDIR}"
echo "DOTARGET=${DOTARGET}, DOJIT=${DOJIT}, DODEBUG=${DODEBUG}, DOTEST=${DOTEST}, DODOC=${DODOC}, DONEEDMKL=${DONEEDMKL}"
if [ "${BUILDOK}" == "y" ]; then
    if [ $DOTEST -gt 0 ]; then
        echo "LOGDIR:       ${LOGDIR}" 2>&1 >> "${BUILDDIR}".log
    fi
    if [ $DOTEST -gt 0 ]; then
        if [ -d "${LOGDIR}" ]; then rm -rf "${LOGIDR}.bak"; mv -v "${LOGDIR}" "${LOGDIR}.bak"; fi
        mkdir ${LOGDIR}
        pwd -P
        ls "${BUILDDIR}/*log"
        for f in "${BUILDDIR}/"*log doxygen.log; do
            cp -av "${f}" "${LOGDIR}/" || true
        done
    fi
fi
echo "FINISHED:     $ORIGINAL_CMD" 2>&1 >> "${BUILDDIR}".log
# for a debug compile  --- FIXME
#(cd "${BUILDDIR}" && ARGS='-VV -R .*simple_training-net-cpp' /usr/bin/time -v make test) 2>&1 | tee test1-dbg.log
#(cd "${BUILDDIR}" && ARGS='-VV -R .*simple_training-net-cpp' valgrind make test) 2>&1 | tee test1-valgrind.log
