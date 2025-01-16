#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <CL/cl.h>

#define N 1024 // 向量大小

#define OCL_CHECK(call) \
    do { \
        cl_int err = call; \
        if (err != CL_SUCCESS) { \
            std::cerr << "OpenCL error: " << err << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

int main() {
    cl_int err;
    // 1. 初始化数据
    float a[N], b[N], result[N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)i * 2;
    }

    // 获取平台和设备
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    // OCL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (numPlatforms <= 0) {
        printf("Failed to find any OpenCL platforms.");
        return 0;
    }
    printf("numPlatforms:%d\n", numPlatforms);
    cl_platform_id platform = platforms[1];
    cl_uint num_devices = 0;
    OCL_CHECK(clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices));

    cl_device_id device;
    OCL_CHECK(
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    // 3. 创建上下文和命令队列
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // 4. 分配设备内存
    cl_mem buffer_a
            = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(float) * N, a, &err);
    OCL_CHECK(err);
    cl_mem buffer_b
            = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(float) * N, b, NULL);
    cl_mem buffer_result = clCreateBuffer(
            context, CL_MEM_WRITE_ONLY, sizeof(float) * N, NULL, NULL);

    // 5. 编译内核
    const char *kernel_source
            = "__kernel void vector_add(__global const float *a, __global "
              "const float *b, __global float *result) { \n"
              "    int id = get_global_id(0); \n"
              "    result[id] = a[id] + b[id]; \n"
              "}";
    cl_program program
            = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

    // 6. 设置内核参数
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_result);

    // 7. 执行内核
    size_t global_size = N; // 全局工作项数量
    clEnqueueNDRangeKernel(
            queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);

    // 8. 读取结果
    clEnqueueReadBuffer(queue, buffer_result, CL_TRUE, 0, sizeof(float) * N,
            result, 0, NULL, NULL);

    // 9. 输出结果
    for (int i = 0; i < 10; i++) {
        printf("result[%d] = %f\n", i, result[i]);
    }

    // 10. 释放资源
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
