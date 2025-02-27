#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <CL/cl.h>
void checkErr(cl_int err, int num) {
    if (CL_SUCCESS != err) {
        printf("OpenCL error(%d) at %d\n", err, num - 1);
        exit(1);
    }
}

// 执行Reduce内核并计时
void run_reduce(cl_kernel kernel, cl_command_queue queue, cl_mem input_buffer,
        cl_mem output_buffer, size_t global_size, size_t local_size,
        uint element_count, const char *name) {
    cl_int err;
    // 设置内核参数
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    if (strstr(name, "Local") || strstr(name, "Tiled")) {
        err |= clSetKernelArg(kernel, 2, local_size * sizeof(int), NULL);
        err |= clSetKernelArg(kernel, 3, sizeof(uint), &element_count);

    } else {
        err |= clSetKernelArg(kernel, 2, sizeof(uint), &element_count);
    }
    checkErr(err, __LINE__);

    // 清空输出缓冲区
    int zero = 0;
    err = clEnqueueFillBuffer(queue, output_buffer, &zero, sizeof(int), 0,
            sizeof(int), 0, NULL, NULL);
    checkErr(err, __LINE__);

    // 执行内核
    auto start = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(
            queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    checkErr(err, __LINE__);
    clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    // 读取结果
    int result = 1;
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(int),
            &result, 0, NULL, NULL);
    checkErr(err, __LINE__);

    // 打印结果
    auto duration
            = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                      .count();
    std::cout << name << " Result: " << result << " Time: " << duration << "us"
              << " (" << global_size / local_size << " workgroups)"
              << std::endl;
}

int main(int argc, char **argv) {
    cl_device_id *device;
    std::vector<cl_platform_id> platform(2);
    cl_int err;
    cl_uint NumDevice;
    //选择第一个平台
    err = clGetPlatformIDs(2, platform.data(), nullptr);
    checkErr(err, __LINE__);
    err = clGetDeviceIDs(
            platform[1], CL_DEVICE_TYPE_GPU, 0, nullptr, &NumDevice);
    checkErr(err, __LINE__);

    device = (cl_device_id *)malloc(sizeof(cl_device_id) * NumDevice);
    //选择GPU设备
    err = clGetDeviceIDs(
            platform[1], CL_DEVICE_TYPE_GPU, NumDevice, device, nullptr);
    checkErr(err, __LINE__);

    cl_context_properties properites[]
            = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform[1], 0};

    // 指定设备类型创建上下文
    // cl_context context = clCreateContextFromType(
    //         properites, CL_DEVICE_TYPE_ALL, nullptr, nullptr, &err);

    // 指定设备创建上下文
    cl_context context = clCreateContext(
            properites, NumDevice, device, nullptr, nullptr, &err);

    // OpenCL 2.0 API
#ifdef CL_VERSION_2_0
    // cl_queue_properties props[] = {CL_QUEUE_PROPERTIES,
    //         CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(
            context, *device, nullptr, &err);
    checkErr(err, __LINE__);
#else
    // OpenCL 1.x API
    //     cl_command_queue_properties prop = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cl_command_queue_properties prop = 0;
    cl_command_queue queue = clCreateCommandQueue(context, *device, prop, &err);
    checkErr(err, __LINE__);
#endif

    // kernel code
    const char *ocl_code = R"(
        // 基础版本 Reduce: 全局内存直接操作
        __kernel void reduce_naive(
                        __global int *input, __global int *output, uint size) {
            int gid = get_global_id(0);
            if (gid >= size) return;

            // 直接全局原子操作
            atom_add((__global int *)output, (int)input[gid]);
        }

        // 优化版本1: 使用局部内存
        __kernel void reduce_local(__global int *input, __global int *output,
                __local int *local_data, uint size) {
            int gid = get_global_id(0);
            int lid = get_local_id(0);
            int group_size = get_local_size(0);

            // 加载到局部内存
            local_data[lid] = (gid < size) ? input[gid] : 0;
            barrier(CLK_LOCAL_MEM_FENCE);

            // 局部归约
            for (int stride = group_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) { local_data[lid] += local_data[lid + stride]; }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // 写回全局内存
            if (lid == 0) { atom_add((__global int *)output, (int)local_data[0]); }
        }

        // 优化版本2: 分块处理 + 展开
        __kernel void reduce_tiled(__global int *input, __global int *output,
                __local int *local_data, uint size) {
            int gid = get_global_id(0);
            int lid = get_local_id(0);
            int group_size = get_local_size(0);

            // 每个工作项处理多个元素
            int sum = 0;
            for (int i = gid; i < size; i += get_global_size(0)) {
                sum += input[i];
            }
            local_data[lid] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);

            // 更高效的树形归约
            for (int stride = group_size / 2; stride >= 1; stride >>= 1) {
                if (lid < stride) { local_data[lid] += local_data[lid + stride]; }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) { atom_add((__global int *)output, (int)local_data[0]); }
        }
    )";
    const char *sources[] = {ocl_code};

    // create program
    cl_program ocl_program
            = clCreateProgramWithSource(context, 1, sources, nullptr, &err);
    checkErr(err, __LINE__);

    clBuildProgram(ocl_program, 0, nullptr, nullptr, nullptr, nullptr);
    checkErr(err, __LINE__);

    size_t num_kernels;
    err = clGetProgramInfo(ocl_program, CL_PROGRAM_NUM_KERNELS, sizeof(size_t),
            &num_kernels, nullptr);

    // 创建内核
    cl_kernel native_kernel = clCreateKernel(ocl_program, "reduce_naive", &err);
    checkErr(err, __LINE__);
    cl_kernel local_kernel = clCreateKernel(ocl_program, "reduce_local", &err);
    checkErr(err, __LINE__);
    cl_kernel tiled_kernel = clCreateKernel(ocl_program, "reduce_tiled", &err);
    checkErr(err, __LINE__);

    // 创建缓冲区（示例）
    cl_mem input_buffer, output_buffer;
    const size_t element_count = 1024 * 1024 * 10;
    const size_t buffer_size = element_count * sizeof(int);

    // 初始化 input 的内容
    int *host_input = (int *)malloc(buffer_size);
    for (size_t i = 0; i < element_count; i++) {
        host_input[i] = 1;
    }

    input_buffer = clCreateBuffer(
            context, CL_MEM_READ_ONLY, buffer_size, nullptr, &err);
    checkErr(err, __LINE__);
    output_buffer = clCreateBuffer(
            context, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);
    checkErr(err, __LINE__);

    // 将数据拷贝到设备
    err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, buffer_size,
            host_input, 0, NULL, NULL);
    checkErr(err, __LINE__);

    // 获取设备信息
    size_t max_workgroup_size;
    clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
            &max_workgroup_size, NULL);
    //     size_t workgroup_size = std::min(max_workgroup_size, 1024ul);
    printf("Max workgroup size: %lu\n", max_workgroup_size);
    size_t workgroup_size = 1024ul;
    size_t global_size = ((element_count + workgroup_size - 1) / workgroup_size)
            * workgroup_size;

    // 运行不同版本的Reduce
    run_reduce(native_kernel, queue, input_buffer, output_buffer, global_size,
            1, element_count, "Native Reduce");
    run_reduce(local_kernel, queue, input_buffer, output_buffer, global_size,
            workgroup_size, element_count, "Local Memory");
    run_reduce(tiled_kernel, queue, input_buffer, output_buffer,
            global_size / 4, workgroup_size, element_count, "Tiled Reduce");

    // 清理资源
    free(host_input);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(native_kernel);
    clReleaseKernel(local_kernel);
    clReleaseKernel(tiled_kernel);

    checkErr(err, __LINE__);

    clReleaseProgram(ocl_program);
    checkErr(err, __LINE__);

    err = clReleaseCommandQueue(queue);
    checkErr(err, __LINE__);

    err = clReleaseContext(context);
    checkErr(err, __LINE__);
    free(device);
    return 0;
}
