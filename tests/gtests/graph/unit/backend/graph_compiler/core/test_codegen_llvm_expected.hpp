/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_TEST_CODEGEN_LLVM_EXPECTED_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_TEST_CODEGEN_LLVM_EXPECTED_HPP

static const char *expected_base = R"(; ModuleID = 'name'
source_filename = "name"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind
define i32 @ccc(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, float* noalias nocapture nonnull %A_arg, i32 %len_arg) #0 {
entry:
  %len = alloca i32, align 4
  store i32 %len_arg, i32* %len, align 4
  %i = alloca i64, align 8
  store i64 0, i64* %i, align 8
  br label %for_check

for_check:                                        ; preds = %for_cont3, %entry
  %0 = load i64, i64* %i, align 8
  %1 = icmp ult i64 %0, 128
  br i1 %1, label %for_body, label %for_cont

for_body:                                         ; preds = %for_check
  %j = alloca i64, align 8
  store i64 0, i64* %j, align 8
  br label %for_check1

for_check1:                                       ; preds = %for_body2, %for_body
  %2 = load i64, i64* %j, align 8
  %3 = icmp ult i64 %2, 128
  br i1 %3, label %for_body2, label %for_cont3

for_body2:                                        ; preds = %for_check1
  %i_v = load i64, i64* %i, align 8
  %4 = mul i64 %i_v, 128
  %j_v = load i64, i64* %j, align 8
  %5 = add i64 %4, %j_v
  %6 = getelementptr float, float* %A_arg, i64 %5
  store float 0.000000e+00, float* %6, align 4
  %7 = load i64, i64* %j, align 8
  %8 = add i64 %7, 1
  store i64 %8, i64* %j, align 8
  br label %for_check1

for_cont3:                                        ; preds = %for_check1
  %len_v = load i32, i32* %len, align 4
  %9 = getelementptr float, float* %A_arg, i64 100
  %10 = bitcast float* %9 to i8*
  call void @bbb(float* %A_arg, i32 %len_v, i8* %10)
  %11 = load i64, i64* %i, align 8
  %12 = add i64 %11, 1
  store i64 %12, i64* %i, align 8
  br label %for_check

for_cont:                                         ; preds = %for_check
  %len_v4 = load i32, i32* %len, align 4
  call void @aaa(i8* %__stream_arg, i8* %__module_data_arg, float* %A_arg, float* %A_arg, float* %A_arg, i32 %len_v4)
  ret i32 12
}

; Function Attrs: nounwind
declare void @bbb(float*, i32, i8*) #1

; Function Attrs: nounwind
define void @aaa(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, float* noalias nocapture nonnull %A_arg, float* noalias nocapture nonnull %B_arg, float* noalias nocapture nonnull %C_arg, i32 %len_arg) #0 {
entry:
  %len = alloca i32, align 4
  store i32 %len_arg, i32* %len, align 4
  %0 = getelementptr i8, i8* %__module_data_arg, i64 64
  %gtsr = bitcast i8* %0 to float*
  %1 = getelementptr i8, i8* %__module_data_arg, i64 0
  %val = bitcast i8* %1 to float*
  %D = alloca float, i64 20, align 64
  %2 = call i8* @sc_aligned_malloc(i8* %__stream_arg, i64 8000)
  %E = bitcast i8* %2 to float*
  %len_v = load i32, i32* %len, align 4
  %3 = sext i32 %len_v to i64
  %4 = mul i64 %3, 4
  %5 = call i8* @sc_aligned_malloc(i8* %__stream_arg, i64 %4)
  %F = bitcast i8* %5 to float*
  %6 = getelementptr float, float* %F, i32 3
  %F_view = bitcast float* %6 to i32*
  %i = alloca i64, align 8
  store i64 0, i64* %i, align 8
  br label %for_check

for_check:                                        ; preds = %for_cont3, %entry
  %7 = load i64, i64* %i, align 8
  %8 = icmp ult i64 %7, 128
  br i1 %8, label %for_body, label %for_cont

for_body:                                         ; preds = %for_check
  %j = alloca i64, align 8
  store i64 0, i64* %j, align 8
  br label %for_check1

for_check1:                                       ; preds = %for_cont6, %for_body
  %9 = load i64, i64* %j, align 8
  %10 = icmp ult i64 %9, 128
  br i1 %10, label %for_body2, label %for_cont3

for_body2:                                        ; preds = %for_check1
  %k = alloca i64, align 8
  store i64 0, i64* %k, align 8
  br label %for_check4

for_check4:                                       ; preds = %for_body5, %for_body2
  %11 = load i64, i64* %k, align 8
  %12 = icmp ult i64 %11, 128
  br i1 %12, label %for_body5, label %for_cont6

for_body5:                                        ; preds = %for_check4
  %i_v = load i64, i64* %i, align 8
  %13 = mul i64 %i_v, 128
  %j_v = load i64, i64* %j, align 8
  %14 = add i64 %13, %j_v
  %15 = getelementptr float, float* %C_arg, i64 %14
  %16 = load float, float* %15, align 4, !alias.scope !0, !noalias !3
  %i_v7 = load i64, i64* %i, align 8
  %17 = mul i64 %i_v7, 128
  %k_v = load i64, i64* %k, align 8
  %18 = add i64 %17, %k_v
  %19 = getelementptr float, float* %A_arg, i64 %18
  %20 = load float, float* %19, align 4, !alias.scope !8, !noalias !9
  %j_v8 = load i64, i64* %j, align 8
  %k_v9 = load i64, i64* %k, align 8
  %21 = mul i64 %k_v9, 128
  %22 = add i64 %j_v8, %21
  %23 = getelementptr float, float* %B_arg, i64 %22
  %24 = load float, float* %23, align 4, !alias.scope !10, !noalias !11
  %25 = fmul reassoc nnan contract float %20, %24
  %26 = fadd reassoc nnan contract float %16, %25
  %i_v10 = load i64, i64* %i, align 8
  %27 = mul i64 %i_v10, 128
  %j_v11 = load i64, i64* %j, align 8
  %28 = add i64 %27, %j_v11
  %29 = getelementptr float, float* %C_arg, i64 %28
  store float %26, float* %29, align 4, !alias.scope !0, !noalias !3
  %30 = load i64, i64* %k, align 8
  %31 = add i64 %30, 1
  store i64 %31, i64* %k, align 8
  br label %for_check4

for_cont6:                                        ; preds = %for_check4
  %32 = load i64, i64* %j, align 8
  %33 = add i64 %32, 1
  store i64 %33, i64* %j, align 8
  br label %for_check1

for_cont3:                                        ; preds = %for_check1
  %34 = load i64, i64* %i, align 8
  %35 = add i64 %34, 1
  store i64 %35, i64* %i, align 8
  br label %for_check

for_cont:                                         ; preds = %for_check
  %36 = getelementptr i32, i32* %F_view, i32 0
  store i32 1, i32* %36, align 4, !alias.scope !12, !noalias !13
  %37 = getelementptr float, float* %gtsr, i32 0
  store float 1.000000e+00, float* %37, align 4
  store float 1.000000e+00, float* %val, align 4
  %38 = bitcast float* %F to i8*
  call void @sc_aligned_free(i8* %__stream_arg, i8* %38)
  %39 = bitcast float* %E to i8*
  call void @sc_aligned_free(i8* %__stream_arg, i8* %39)
  ret void
}

; Function Attrs: nounwind
declare noalias i8* @sc_aligned_malloc(i8*, i64) #1

; Function Attrs: nounwind
declare void @sc_aligned_free(i8*, i8*) #1

; Function Attrs: nounwind
define void @ddd(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, i32 %len_arg) #0 {
entry:
  %len = alloca i32, align 4
  store i32 %len_arg, i32* %len, align 4
  ret void
}

; Function Attrs: nounwind
define void @__sc_init__(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg) #0 {
entry:
  %0 = getelementptr i8, i8* %__module_data_arg, i64 0
  %val = bitcast i8* %0 to float*
  %1 = getelementptr i8, i8* %__module_data_arg, i64 4
  %val2 = bitcast i8* %1 to float*
  %2 = getelementptr i8, i8* %__module_data_arg, i64 8
  %val3 = bitcast i8* %2 to float*
  store float 0x4028AE1480000000, float* %val, align 4
  %3 = call reassoc nnan contract float @ginit()
  store float %3, float* %val2, align 4
  %4 = call reassoc nnan contract float @ginit()
  store float %4, float* %val3, align 4
  ret void
}

; Function Attrs: nounwind
declare float @ginit() #1

attributes #0 = { nounwind "frame-pointer"="all" "no-frame-pointer-elim"="true" }
attributes #1 = { nounwind }

!0 = !{!1}
!1 = distinct !{!1, !2, !"-4"}
!2 = distinct !{!2, !"aaa"}
!3 = !{!4, !5, !6, !7}
!4 = distinct !{!4, !2, !"-3"}
!5 = distinct !{!5, !2, !"-2"}
!6 = distinct !{!6, !2, !"-1"}
!7 = distinct !{!7, !2, !"1"}
!8 = !{!5}
!9 = !{!1, !4, !6, !7}
!10 = !{!4}
!11 = !{!1, !5, !6, !7}
!12 = !{!7}
!13 = !{!1, !4, !5, !6}
)";

static const char *expected_parallel_for = R"(; ModuleID = 'name'
source_filename = "name"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind
define void @aaa(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, float* noalias nocapture nonnull %A_arg, float* noalias nocapture nonnull %B_arg, float* noalias nocapture nonnull %C_arg, i32 %len_arg) #0 {
entry:
  %len = alloca i32, align 4
  store i32 %len_arg, i32* %len, align 4
  %__tempargs0 = alloca i64, i64 4, align 8
  %0 = ptrtoint float* %C_arg to i64
  %1 = getelementptr i64, i64* %__tempargs0, i64 0
  store i64 %0, i64* %1, align 8
  %2 = ptrtoint float* %A_arg to i64
  %3 = getelementptr i64, i64* %__tempargs0, i64 1
  store i64 %2, i64* %3, align 8
  %4 = ptrtoint float* %B_arg to i64
  %5 = getelementptr i64, i64* %__tempargs0, i64 2
  store i64 %4, i64* %5, align 8
  %len_v = load i32, i32* %len, align 4
  %6 = zext i32 %len_v to i64
  %7 = getelementptr i64, i64* %__tempargs0, i64 3
  store i64 %6, i64* %7, align 8
  call void @sc_parallel_call_cpu_with_env(i8* bitcast (void (i8*, i8*, i64, i64*)* @aaa0_closure_0_0wrapper to i8*), i64 0, i8* %__stream_arg, i8* %__module_data_arg, i64 0, i64 128, i64 1, i64* %__tempargs0)
  %t = alloca i32, align 4
  %len_v1 = load i32, i32* %len, align 4
  store i32 %len_v1, i32* %t, align 4
  %__tempargs1 = alloca i64, i64 2, align 8
  %8 = ptrtoint float* %A_arg to i64
  %9 = getelementptr i64, i64* %__tempargs1, i64 0
  store i64 %8, i64* %9, align 8
  %t_v = load i32, i32* %t, align 4
  %10 = zext i32 %t_v to i64
  %11 = getelementptr i64, i64* %__tempargs1, i64 1
  store i64 %10, i64* %11, align 8
  call void @sc_parallel_call_cpu_with_env(i8* bitcast (void (i8*, i8*, i64, i64*)* @aaa0_closure_1_0wrapper to i8*), i64 0, i8* %__stream_arg, i8* %__module_data_arg, i64 1, i64 100, i64 2, i64* %__tempargs1)
  ret void
}

; Function Attrs: nounwind
declare void @sc_parallel_call_cpu_with_env(i8*, i64, i8*, i8*, i64, i64, i64, i64*) #1

; Function Attrs: nounwind
define internal void @aaa0_closure_0_0wrapper(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, i64 %i_arg, i64* noalias nocapture nonnull %args_arg) #0 {
entry:
  %i = alloca i64, align 8
  store i64 %i_arg, i64* %i, align 8
  %i_v = load i64, i64* %i, align 8
  %0 = getelementptr i64, i64* %args_arg, i64 0
  %1 = load i64, i64* %0, align 8
  %2 = inttoptr i64 %1 to float*
  %3 = getelementptr i64, i64* %args_arg, i64 1
  %4 = load i64, i64* %3, align 8
  %5 = inttoptr i64 %4 to float*
  %6 = getelementptr i64, i64* %args_arg, i64 2
  %7 = load i64, i64* %6, align 8
  %8 = inttoptr i64 %7 to float*
  %9 = getelementptr i64, i64* %args_arg, i64 3
  %10 = load i64, i64* %9, align 8
  %11 = trunc i64 %10 to i32
  call void @aaa0_closure_0(i8* %__stream_arg, i8* %__module_data_arg, i64 %i_v, float* %2, float* %5, float* %8, i32 %11)
  ret void
}

; Function Attrs: nounwind
define internal void @aaa0_closure_1_0wrapper(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, i64 %i_arg, i64* noalias nocapture nonnull %args_arg) #0 {
entry:
  %i = alloca i64, align 8
  store i64 %i_arg, i64* %i, align 8
  %i_v = load i64, i64* %i, align 8
  %0 = getelementptr i64, i64* %args_arg, i64 0
  %1 = load i64, i64* %0, align 8
  %2 = inttoptr i64 %1 to float*
  %3 = getelementptr i64, i64* %args_arg, i64 1
  %4 = load i64, i64* %3, align 8
  %5 = trunc i64 %4 to i32
  call void @aaa0_closure_1(i8* %__stream_arg, i8* %__module_data_arg, i64 %i_v, float* %2, i32 %5)
  ret void
}

; Function Attrs: nounwind
define internal void @aaa0_closure_0(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, i64 %i_arg, float* noalias nocapture nonnull %C_arg, float* noalias nocapture nonnull %A_arg, float* noalias nocapture nonnull %B_arg, i32 %len_arg) #0 {
entry:
  %i = alloca i64, align 8
  store i64 %i_arg, i64* %i, align 8
  %len = alloca i32, align 4
  store i32 %len_arg, i32* %len, align 4
  %0 = getelementptr i8, i8* %__module_data_arg, i64 0
  %gv = bitcast i8* %0 to i32*
  store i32 1, i32* %gv, align 4
  %v1 = alloca float, align 4
  %D = alloca float, i64 20, align 64
  %1 = call i8* @sc_thread_aligned_malloc(i8* %__stream_arg, i64 8000)
  %E = bitcast i8* %1 to float*
  %j = alloca i64, align 8
  store i64 0, i64* %j, align 8
  br label %for_check

for_check:                                        ; preds = %for_cont3, %entry
  %2 = load i64, i64* %j, align 8
  %3 = icmp ult i64 %2, 128
  br i1 %3, label %for_body, label %for_cont

for_body:                                         ; preds = %for_check
  %k = alloca i64, align 8
  store i64 0, i64* %k, align 8
  br label %for_check1

for_check1:                                       ; preds = %for_body2, %for_body
  %4 = load i64, i64* %k, align 8
  %5 = icmp ult i64 %4, 128
  br i1 %5, label %for_body2, label %for_cont3

for_body2:                                        ; preds = %for_check1
  %i_v = load i64, i64* %i, align 8
  %6 = mul i64 %i_v, 128
  %j_v = load i64, i64* %j, align 8
  %7 = add i64 %6, %j_v
  %8 = getelementptr float, float* %C_arg, i64 %7
  %9 = load float, float* %8, align 4
  %i_v4 = load i64, i64* %i, align 8
  %10 = mul i64 %i_v4, 10
  %k_v = load i64, i64* %k, align 8
  %11 = add i64 %10, %k_v
  %12 = getelementptr float, float* %D, i64 %11
  %13 = load float, float* %12, align 4
  %14 = fadd reassoc nnan contract float %9, %13
  %i_v5 = load i64, i64* %i, align 8
  %15 = mul i64 %i_v5, 128
  %k_v6 = load i64, i64* %k, align 8
  %16 = add i64 %15, %k_v6
  %17 = getelementptr float, float* %A_arg, i64 %16
  %18 = load float, float* %17, align 4
  %j_v7 = load i64, i64* %j, align 8
  %k_v8 = load i64, i64* %k, align 8
  %19 = mul i64 %k_v8, 128
  %20 = add i64 %j_v7, %19
  %21 = getelementptr float, float* %B_arg, i64 %20
  %22 = load float, float* %21, align 4
  %23 = fmul reassoc nnan contract float %18, %22
  %24 = fadd reassoc nnan contract float %14, %23
  %len_v = load i32, i32* %len, align 4
  %25 = sitofp i32 %len_v to float
  %26 = fadd reassoc nnan contract float %24, %25
  %v1_v = load float, float* %v1, align 4
  %27 = fadd reassoc nnan contract float %26, %v1_v
  %i_v9 = load i64, i64* %i, align 8
  %28 = mul i64 %i_v9, 128
  %j_v10 = load i64, i64* %j, align 8
  %29 = add i64 %28, %j_v10
  %30 = getelementptr float, float* %C_arg, i64 %29
  store float %27, float* %30, align 4
  %31 = load i64, i64* %k, align 8
  %32 = add i64 %31, 1
  store i64 %32, i64* %k, align 8
  br label %for_check1

for_cont3:                                        ; preds = %for_check1
  %33 = load i64, i64* %j, align 8
  %34 = add i64 %33, 1
  store i64 %34, i64* %j, align 8
  br label %for_check

for_cont:                                         ; preds = %for_check
  %35 = bitcast float* %E to i8*
  call void @sc_thread_aligned_free(i8* %__stream_arg, i8* %35)
  ret void
}

; Function Attrs: nounwind
declare noalias i8* @sc_thread_aligned_malloc(i8*, i64) #1

; Function Attrs: nounwind
declare void @sc_thread_aligned_free(i8*, i8*) #1

; Function Attrs: nounwind
define internal void @aaa0_closure_1(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, i64 %i_1_arg, float* noalias nocapture nonnull %A_arg, i32 %t_arg) #0 {
entry:
  %i_1 = alloca i64, align 8
  store i64 %i_1_arg, i64* %i_1, align 8
  %t = alloca i32, align 4
  store i32 %t_arg, i32* %t, align 4
  %t_v = load i32, i32* %t, align 4
  %0 = sext i32 %t_v to i64
  %i_1_v = load i64, i64* %i_1, align 8
  %1 = add i64 %0, %i_1_v
  %2 = uitofp i64 %1 to float
  %i_1_v1 = load i64, i64* %i_1, align 8
  %i_1_v2 = load i64, i64* %i_1, align 8
  %3 = mul i64 %i_1_v2, 128
  %4 = add i64 %i_1_v1, %3
  %5 = getelementptr float, float* %A_arg, i64 %4
  store float %2, float* %5, align 4
  ret void
}

attributes #0 = { nounwind "frame-pointer"="all" "no-frame-pointer-elim"="true" }
attributes #1 = { nounwind }
)";

static const char *expected_vector = R"(; ModuleID = 'name'
source_filename = "name"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind
define void @aaa(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, float* noalias nocapture nonnull %A_arg, float* noalias nocapture nonnull %B_arg, float* noalias nocapture nonnull %C_arg, i32* noalias nocapture nonnull %D_arg) #0 {
entry:
  %A_val = alloca <8 x float>, align 32
  %i = alloca i64, align 8
  store i64 0, i64* %i, align 8
  br label %for_check

for_check:                                        ; preds = %for_body, %entry
  %0 = load i64, i64* %i, align 8
  %1 = icmp ult i64 %0, 512
  br i1 %1, label %for_body, label %for_cont

for_body:                                         ; preds = %for_check
  %i_v = load i64, i64* %i, align 8
  %2 = getelementptr i32, i32* %D_arg, i64 %i_v
  %3 = bitcast i32* %2 to <8 x i32>*
  store <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>, <8 x i32>* %3, align 1
  %i_v1 = load i64, i64* %i, align 8
  %4 = getelementptr float, float* %C_arg, i64 %i_v1
  %5 = bitcast float* %4 to <8 x float>*
  store <8 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00, float 8.000000e+00>, <8 x float>* %5, align 1
  %i_v2 = load i64, i64* %i, align 8
  %6 = getelementptr i32, i32* %D_arg, i64 %i_v2
  %7 = bitcast i32* %6 to <8 x i32>*
  store <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <8 x i32>* %7, align 1
  %i_v3 = load i64, i64* %i, align 8
  %8 = getelementptr float, float* %C_arg, i64 %i_v3
  %9 = bitcast float* %8 to <8 x float>*
  store <8 x float> <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>, <8 x float>* %9, align 1
  %i_v4 = load i64, i64* %i, align 8
  %10 = getelementptr float, float* %A_arg, i64 %i_v4
  %11 = bitcast float* %10 to <8 x float>*
  %12 = load <8 x float>, <8 x float>* %11, align 1
  store <8 x float> %12, <8 x float>* %A_val, align 32
  %A_val_v = load <8 x float>, <8 x float>* %A_val, align 32
  %i_v5 = load i64, i64* %i, align 8
  %13 = getelementptr float, float* %B_arg, i64 %i_v5
  %14 = bitcast float* %13 to <8 x float>*
  %15 = load <8 x float>, <8 x float>* %14, align 1
  %16 = fadd reassoc nnan contract <8 x float> %A_val_v, %15
  %i_v6 = load i64, i64* %i, align 8
  %17 = getelementptr float, float* %C_arg, i64 %i_v6
  %18 = bitcast float* %17 to <8 x float>*
  store <8 x float> %16, <8 x float>* %18, align 1
  %19 = load i64, i64* %i, align 8
  %20 = add i64 %19, 8
  store i64 %20, i64* %i, align 8
  br label %for_check

for_cont:                                         ; preds = %for_check
  ret void
}

attributes #0 = { nounwind "frame-pointer"="all" "no-frame-pointer-elim"="true" }
)";

constexpr const char *expected_alias = R"(; ModuleID = 'name'
source_filename = "name"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind
define void @aaa(i8* %__stream_arg, i8* noalias nocapture nonnull %__module_data_arg, float* nocapture nonnull %A_arg, float* nocapture nonnull %B_arg, float* noalias nocapture nonnull %C_arg, float* noalias nocapture nonnull %D_arg) #0 {
entry:
  %0 = getelementptr float, float* %A_arg, i32 0
  %1 = load float, float* %0, align 4, !alias.scope !0, !noalias !3
  %2 = fadd reassoc nnan contract float %1, 1.000000e+00
  %3 = getelementptr float, float* %A_arg, i32 0
  store float %2, float* %3, align 4, !alias.scope !0, !noalias !3
  %4 = getelementptr float, float* %B_arg, i32 0
  %5 = load float, float* %4, align 4, !alias.scope !0, !noalias !3
  %6 = fadd reassoc nnan contract float %5, 1.000000e+00
  %7 = getelementptr float, float* %B_arg, i32 0
  store float %6, float* %7, align 4, !alias.scope !0, !noalias !3
  %8 = getelementptr float, float* %C_arg, i32 0
  %9 = load float, float* %8, align 4, !alias.scope !7, !noalias !8
  %10 = fadd reassoc nnan contract float %9, 1.000000e+00
  %11 = getelementptr float, float* %C_arg, i32 0
  store float %10, float* %11, align 4, !alias.scope !7, !noalias !8
  %12 = getelementptr float, float* %D_arg, i32 0
  %13 = load float, float* %12, align 4, !alias.scope !9, !noalias !10
  %14 = fadd reassoc nnan contract float %13, 1.000000e+00
  %15 = getelementptr float, float* %D_arg, i32 0
  store float %14, float* %15, align 4, !alias.scope !9, !noalias !10
  ret void
}

attributes #0 = { nounwind "frame-pointer"="all" "no-frame-pointer-elim"="true" }

!0 = !{!1}
!1 = distinct !{!1, !2, !"0"}
!2 = distinct !{!2, !"aaa"}
!3 = !{!4, !5, !6}
!4 = distinct !{!4, !2, !"-3"}
!5 = distinct !{!5, !2, !"-2"}
!6 = distinct !{!6, !2, !"-1"}
!7 = !{!5}
!8 = !{!4, !6, !1}
!9 = !{!4}
!10 = !{!5, !6, !1}
)";
#endif
