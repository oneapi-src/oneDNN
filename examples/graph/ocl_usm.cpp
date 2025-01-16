#include <iostream>
#include <vector>
#include "graph_example_utils.hpp"
#include <CL/cl.h>

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
    cl_uint numPlatforms;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // 获取平台和设备
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    // OCL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (numPlatforms <= 0) {
        printf("Failed to find any OpenCL platforms.");
        return 0;
    }
    printf("numPlatforms:%d\n", numPlatforms);
    platform = platforms[1];
    cl_uint num_devices = 0;
    OCL_CHECK(clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices));
    printf("num_devices:%d\n", num_devices);
    OCL_CHECK(
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    // 创建上下文和命令队列
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    OCL_CHECK(err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    OCL_CHECK(err);

    // 检查设备是否支持 USM
    // cl_bool is_usm_supported = false;
    // OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY,
    //         sizeof(cl_bool), &is_usm_supported, nullptr));
    // if (!is_usm_supported) {
    //     std::cerr << "USM is not supported!" << std::endl;
    //     return -1;
    // }

    // 分配 USM 内存
    size_t size = 1024 * sizeof(int);
    void *p = ocl_malloc_shared(size, 0, device, context);

    int a[1024], result[1024];
    for (int i = 0; i < 1024; i++)
        a[i] = 0;

    cl_mem buffer_a = clCreateBuffer(
            context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, a, &err);
    // if (buffer_a == nullptr) {
    //     printf("Error creating memory objects.");
    //     return 0;
    // }
    // OCL_CHECK(err);

    // 动态加载扩展函数
    using F = cl_int (*)(cl_kernel, cl_uint, const void *);
    const char *f_name = "clSetKernelArgMemPointerINTEL";
    auto usm_set_arg = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    if (!usm_set_arg) {
        std::cerr << "Failed to load extension function: " << f_name
                  << std::endl;
        return -1;
    }

    // 创建内核程序
    const char *kernel_source = R"(
        __kernel void vector_add(__global int* a) {
            int id = get_global_id(0);
            a[id] = id;
        }
    )";
    program = clCreateProgramWithSource(
            context, 1, &kernel_source, nullptr, &err);
    OCL_CHECK(err);

    OCL_CHECK(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));
    kernel = clCreateKernel(program, "vector_add", &err);
    OCL_CHECK(err);

    // 设置内核参数
    OCL_CHECK(usm_set_arg(kernel, 0, p));
    // clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);

    // 执行内核
    size_t global_size = 1024;
    OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size,
            nullptr, 0, nullptr, nullptr));

    // 8. 读取结果
    clEnqueueReadBuffer(queue, buffer_a, CL_TRUE, 0,
            sizeof(float) * global_size, result, 0, NULL, NULL);

    // 9. 输出结果
    for (int i = 0; i < 10; i++) {
        // printf("result[%d] = %d\n", i, result[i]);
        printf("p[%d] = %d\n", i, ((int *)p)[i]);
    }

    // 释放资源
    ocl_free(p, device, context, {});
    clReleaseMemObject(buffer_a);
    OCL_CHECK(clReleaseKernel(kernel));
    OCL_CHECK(clReleaseProgram(program));
    OCL_CHECK(clReleaseCommandQueue(queue));
    OCL_CHECK(clReleaseContext(context));

    return 0;
}