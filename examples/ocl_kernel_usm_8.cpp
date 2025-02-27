#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <CL/cl.h>
#include <CL/cl_ext.h>

// 不成功的尝试
void checkErr(cl_int err, int num) {
    if (CL_SUCCESS != err) {
        printf("OpenCL error(%d) at %d\n", err, num - 1);
        exit(1);
    }
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
    cl_context context = clCreateContextFromType(
            properites, CL_DEVICE_TYPE_ALL, nullptr, nullptr, &err);

    // 指定设备创建上下文
    // cl_context context = clCreateContext(
    //         properites, NumDevice, device, nullptr, nullptr, &err);

    // OpenCL 2.0 API
#ifdef CL_VERSION_2_0
    // cl_queue_properties props[] = {CL_QUEUE_PROPERTIES,
    //         CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(
            context, *device, nullptr, &err);
    checkErr(err, __LINE__);
#else
    // OpenCL 1.x API
    cl_command_queue_properties prop = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cl_command_queue queue = clCreateCommandQueue(context, *device, prop, &err);
    checkErr(err, __LINE__);
#endif

    // kernel code
    const char *ocl_code
            = "__kernel void init(__global float *data) {"
              "    int id = get_global_id(0);"
              "    data[id] = (id % 2) ? -id : id;"
              "}"
              "__kernel void add(__global float *src0,__global float "
              "*src1,__global float *dst) {"
              "    int id = get_global_id(0);"
              "    dst[id] = src0[id] + src1[id];"
              "}";
    const char *sources[] = {ocl_code};

    // create program
    cl_program ocl_program
            = clCreateProgramWithSource(context, 1, sources, nullptr, &err);
    checkErr(err, __LINE__);

    clBuildProgram(ocl_program, 0, nullptr, nullptr, nullptr, nullptr);
    checkErr(err, __LINE__);

    // 创建内核
    cl_kernel kernel_init = clCreateKernel(ocl_program, "init", &err);
    checkErr(err, __LINE__);

    // 创建缓冲区（示例）
    const size_t element_count = 1024;
    const size_t usm_size = element_count * sizeof(float);
    const cl_uint alignment = 64;

    // ... 以下是新加的代码
    // 分配 USM 内存，直接调用扩展函数

    //     void *p = clSharedMemAllocINTEL(
    //             context, *device, nullptr, usm_size, alignment, &err);
    //     checkErr(err, __LINE__);
    //... 以上是新加的代码

    //     // 执行 init 内核
    //     using clSetKernelArgMemPointerINTEL_func_t
    //             = cl_int (*)(cl_kernel, cl_uint, const void *);
    //     auto usm_set_kernel_arg
    //             = reinterpret_cast<clSetKernelArgMemPointerINTEL_func_t>(
    //                     clGetExtensionFunctionAddressForPlatform(
    //                             platform[1], "clSetKernelArgMemPointerINTEL"));
    //     err = usm_set_kernel_arg(kernel_init, 0, p);
    //     checkErr(err, __LINE__);

    //     size_t global_size = 1024;
    //     err = clEnqueueNDRangeKernel(queue, kernel_init, 1, nullptr, &global_size,
    //             nullptr, 0, nullptr, nullptr);
    //     checkErr(err, __LINE__);

    //     // 等待所有命令执行完成
    //     clFinish(queue);

    //     // 打印初始化后的 data_buffer 结果
    //     printf("\n=== init kernel 结果 ===\n");
    //     for (int i = 0; i < 10; i++) {
    //         printf("data[%d] = %+.1f\n", i, ((float *)p)[i]);
    //     }

    //     // 清理资源
    //     using FREE = cl_int (*)(cl_context, void *);
    //     auto usm_free
    //             = reinterpret_cast<FREE>(clGetExtensionFunctionAddressForPlatform(
    //                     platform[1], "clMemBlockingFreeINTEL"));
    //     usm_free(context, p);

    clReleaseKernel(kernel_init);

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
